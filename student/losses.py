"""Student one-step plus rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    batch_size = states.shape[0]
    seq_len = actions.shape[1]
    
    # Initialize the hidden state at the start of the sequence window
    hidden = model.initial_hidden(batch_size, states.device)
    
    preds = []
    # Step through time to allow the GRU to build up memory correctly
    for t in range(seq_len):
        obs_norm = normalizer.normalize_obs(states[:, t])
        act_norm = normalizer.normalize_act(actions[:, t])
        
        # Pass the hidden state sequentially
        pred_norm, hidden = model(obs_norm, act_norm, hidden)
        preds.append(pred_norm)
        
    # Stack predictions back into (Batch, Time, Dim)
    preds_norm_stack = torch.stack(preds, dim=1)
    
    # Calculate targets
    target_delta = states[:, 1:] - states[:, :-1]
    target_norm = normalizer.normalize_delta(target_delta)
    
    return F.mse_loss(preds_norm_stack, target_norm)


def rollout_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer, warmup_steps: int, horizon: int) -> torch.Tensor:
    # Train local open-loop stability at random positions, not only at the
    # beginning of each stored window.
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer, warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    return F.mse_loss(pred_norm, target_norm)


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]
    one = one_step_delta_loss(model, states, actions, normalizer)
    horizon = int(loss_cfg.get("rollout_train_horizon", 5))
    warmup = int(cfg["eval"].get("warmup_steps", 5))
    roll = rollout_loss(model, states, actions, normalizer, warmup_steps=warmup, horizon=horizon)
    total = float(loss_cfg.get("one_step_weight", 1.0)) * one + float(loss_cfg.get("rollout_weight", 0.3)) * roll
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
    }
