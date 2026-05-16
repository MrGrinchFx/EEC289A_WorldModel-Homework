"""Student one-step plus rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    batch_size = states.shape[0]
    seq_len = actions.shape[1]          # number of transitions = T-1

    hidden = model.initial_hidden(batch_size, states.device)

    preds = []
    # Only iterate over valid transitions: t=0..seq_len-2
    # At each t we observe states[:,t], act[:,t], and target is states[:,t+1]-states[:,t]
    for t in range(seq_len - 1):       # FIX: was range(seq_len), produced seq_len preds
        obs_norm = normalizer.normalize_obs(states[:, t])
        act_norm = normalizer.normalize_act(actions[:, t])
        pred_norm, hidden = model(obs_norm, act_norm, hidden)
        hidden = hidden.detach()        # truncated BPTT: prevents vanishing grads through long seqs
        preds.append(pred_norm)

    preds_norm_stack = torch.stack(preds, dim=1)   # (B, T-1, obs_dim)

    target_delta = states[:, 1:] - states[:, :-1]  # (B, T-1, obs_dim)
    target_norm = normalizer.normalize_delta(target_delta)

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
            f"train_sequence_length too short: need {needed_states}, got {states.shape[1]}"
        )
    max_start = states.shape[1] - needed_states
    start = int(torch.randint(0, max_start + 1, (), device=states.device).item()) if max_start > 0 else 0

    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]

    preds = open_loop_rollout(
        model, sub_states, sub_actions, normalizer,
        warmup_steps=warmup_steps, horizon=horizon,
    )
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]

    pred_norm   = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)

    # Horizon-weighted loss: penalize late-step errors MORE.
    # This directly pressures VPT80 — errors at step 60-80 are what kill the score.
    h = pred_norm.shape[1]
    weights = torch.linspace(1.0, 2.5, h, device=pred_norm.device)  # ramp from 1x to 2.5x
    weights = weights / weights.mean()                                # keep total loss scale stable
    sq_err = (pred_norm - target_norm) ** 2                           # (B, H, obs_dim)
    weighted_loss = (sq_err.mean(dim=-1) * weights.unsqueeze(0)).mean()

    return weighted_loss


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states   = batch["states"]
    actions  = batch["actions"]

    one      = one_step_delta_loss(model, states, actions, normalizer)
    horizon  = int(loss_cfg.get("rollout_train_horizon", 5))
    warmup   = int(cfg["eval"].get("warmup_steps", 5))
    roll     = rollout_loss(model, states, actions, normalizer, warmup_steps=warmup, horizon=horizon)

    total = float(loss_cfg.get("one_step_weight", 1.0)) * one \
          + float(loss_cfg.get("rollout_weight", 0.3)) * roll

    return total, {
        "loss/total":    float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout":  float(roll.detach().cpu()),
    }
