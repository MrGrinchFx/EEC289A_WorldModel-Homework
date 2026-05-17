"""Student one-step plus rollout loss utilizing Smooth L1 and Exponential Weighting."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .rollout import open_loop_rollout

def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    batch_size = states.shape[0]
    seq_len = actions.shape[1]
    hidden = model.initial_hidden(batch_size, states.device)

    preds = []
    for t in range(seq_len):
        obs_norm = normalizer.normalize_obs(states[:, t])
        act_norm = normalizer.normalize_act(actions[:, t])
        pred_norm, hidden = model(obs_norm, act_norm, hidden)
        
        if (t + 1) % 32 == 0:
            hidden = hidden.detach()
            
        preds.append(pred_norm)

    preds_norm_stack = torch.stack(preds, dim=1)
    target_delta = states[:, 1:] - states[:, :-1]
    target_norm = normalizer.normalize_delta(target_delta)
    
    burn_in = 10
    # Use Huber Loss (Smooth L1) to absorb massive outliers without exploding gradients
    if seq_len > burn_in:
        return F.smooth_l1_loss(preds_norm_stack[:, burn_in:], target_norm[:, burn_in:])
    
    return F.smooth_l1_loss(preds_norm_stack, target_norm)

def rollout_loss(
    model, states: torch.Tensor, actions: torch.Tensor, normalizer, warmup_steps: int, horizon: int,
) -> torch.Tensor:
    needed_states = int(warmup_steps) + int(horizon) + 1
    start = 0

    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]

    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer, warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]

    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)

    # Inside rollout_loss, replace the exponential weights with:
    h = pred_norm.shape[1]
    
    # Linear ramp penalizes late-stage drift without causing gradient explosions
    weights = torch.linspace(1.0, 5.0, h, device=pred_norm.device)
    weights = weights / weights.mean() 
    
    abs_err = F.smooth_l1_loss(pred_norm, target_norm, reduction='none')
    return (abs_err.mean(dim=-1) * weights.unsqueeze(0)).mean()

def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]

    one = one_step_delta_loss(model, states, actions, normalizer)
    horizon = int(loss_cfg.get("rollout_train_horizon", 45))
    warmup = int(cfg["eval"].get("warmup_steps", 10))
    roll = rollout_loss(model, states, actions, normalizer, warmup_steps=warmup, horizon=horizon)

    total = float(loss_cfg.get("one_step_weight", 1.0)) * one + float(loss_cfg.get("rollout_weight", 0.5)) * roll

    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
    }
