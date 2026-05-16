"""Student one-step plus rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    batch_size = states.shape[0]
    # actions.shape[1] == T == number of valid transitions == states.shape[1] - 1
    # target_delta = states[:, 1:] - states[:, :-1]  has shape (B, T, obs_dim)
    # So we need exactly T predictions — range(actions.shape[1]) is correct.
    seq_len = actions.shape[1]

    hidden = model.initial_hidden(batch_size, states.device)

    preds = []
    for t in range(seq_len):                          # T iterations → T predictions
        obs_norm = normalizer.normalize_obs(states[:, t])
        act_norm = normalizer.normalize_act(actions[:, t])
        pred_norm, hidden = model(obs_norm, act_norm, hidden)
        # Truncated BPTT: detach hidden every step so gradients don't vanish
        # through the full sequence length, while the GRU still builds memory
        # in the forward pass.
        hidden = hidden.detach()
        preds.append(pred_norm)

    preds_norm_stack = torch.stack(preds, dim=1)      # (B, T, obs_dim)

    target_delta = states[:, 1:] - states[:, :-1]    # (B, T, obs_dim) ✓
    target_norm  = normalizer.normalize_delta(target_delta)

    return F.mse_loss(preds_norm_stack, target_norm)


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
) -> torch.Tensor:
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    start = int(torch.randint(0, max_start + 1, (), device=states.device).item()) if max_start > 0 else 0

    sub_states  = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]

    preds   = open_loop_rollout(model, sub_states, sub_actions, normalizer,
                                warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]

    pred_norm   = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)

    # Horizon-weighted loss: ramp penalty from 1x at step 0 to 2.5x at step H.
    # Later steps are exactly what VPT80 measures, so we pressure them more.
    h       = pred_norm.shape[1]
    weights = torch.linspace(1.0, 5.0, h, device=pred_norm.device)
    weights = weights / weights.mean()                # keep overall loss scale stable
    sq_err  = (pred_norm - target_norm) ** 2          # (B, H, obs_dim)
    return (sq_err.mean(dim=-1) * weights.unsqueeze(0)).mean()


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states   = batch["states"]
    actions  = batch["actions"]

    one     = one_step_delta_loss(model, states, actions, normalizer)
    horizon = int(loss_cfg.get("rollout_train_horizon", 5))
    warmup  = int(cfg["eval"].get("warmup_steps", 5))
    roll    = rollout_loss(model, states, actions, normalizer,
                           warmup_steps=warmup, horizon=horizon)

    total = float(loss_cfg.get("one_step_weight", 1.0)) * one \
          + float(loss_cfg.get("rollout_weight",  0.3)) * roll

    return total, {
        "loss/total":    float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout":  float(roll.detach().cpu()),
    }
