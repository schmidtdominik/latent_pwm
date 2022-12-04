from collections import defaultdict
from typing import Tuple

import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import torch
from torch.optim import Optimizer

from dark_mvp.ppo.buffer import RolloutBatch, RolloutBuffer
from dark_mvp.ppo.model import ActorCritic


def compute_policy_loss(adv, behav_logp, logp, clip_ratio):
    log_prob_ratio = logp - behav_logp

    prob_ratio = torch.exp(log_prob_ratio)
    prob_ratio_clipped = torch.clip(
        prob_ratio,
        1 - clip_ratio,
        1 + clip_ratio
    )

    ppo_clip_obj = -torch.minimum(
        prob_ratio * adv,
        prob_ratio_clipped * adv
    ).mean()

    with torch.no_grad():
        # http://joschu.net/blog/kl-approx.html
        kl_est = ((prob_ratio - 1) - log_prob_ratio).mean()
        clip_frac = (
            torch.logical_or(prob_ratio < 1 - clip_ratio, prob_ratio > 1 + clip_ratio)
            .float()
            .mean()
        )

    return ppo_clip_obj, kl_est, clip_frac


def action_log_probs(log_probs, actions):
    return -F.nll_loss(log_probs, actions.flatten(), reduction="none").view_as(actions)


def ppo_step(
    args: DictConfig,
    opt: Optimizer,
    scaler: GradScaler,
    ac: ActorCritic,
    v_targets,
    advs: Tensor,
    obs: Tensor,
    actions: Tensor,
    behav_logp: Tensor,
) -> dict:

    opt.zero_grad()
    with autocast(enabled=args.amp):
        values, ent, targ_logp = ac.step(obs)

    targ_logp_actions = action_log_probs(targ_logp, actions)
    behav_logp_actions = action_log_probs(behav_logp, actions)

    ppo_clip_obj, est_kl, clip_frac = compute_policy_loss(
        advs, behav_logp_actions, targ_logp_actions, args.opt.clip_ratio
    )

    td_delta = v_targets - values
    value_loss = td_delta.square().mean()
    residual_var = torch.var(
        td_delta.flatten().to(torch.float32)) / torch.var(v_targets.flatten().to(torch.float32)
    )

    loss = (
        ppo_clip_obj
        + args.opt.value_cost * value_loss
        - args.opt.entropy_cost * ent
    )

    scaler.scale(loss).backward()
    scaler.unscale_(opt)

    grad_norm = torch.nn.utils.clip_grad_norm_(ac.parameters(), args.opt.max_grad_norm)

    scaler.step(opt)
    scaler.update()

    return dict(
            kl_div=est_kl.item(),
            policy_loss=ppo_clip_obj.item(),
            value_loss=value_loss.item(),
            ent=ent.item(),
            grad_norm=grad_norm.item(),
            clip_frac=clip_frac.item(),
            residual_var=residual_var.item(),
        )


@torch.no_grad()
def compute_v_targets(rollout: RolloutBatch, bootstrap_values, gamma):
    # rewards + bootstrapped value estimates
    v_targets = torch.cat([rollout.rewards, bootstrap_values.unsqueeze(0)], dim=0)
    discounts = (~rollout.dones).float() * gamma
    for t in range(len(discounts) - 1, -1, -1):
        v_targets[t] = v_targets[t] + discounts[t] * v_targets[t + 1]
    return v_targets[:-1]


@torch.no_grad()
def compute_gae_advs(rollout: RolloutBatch, bootstrap_values, gamma, gae_lambda):
    values_w_bootstrap = torch.cat(
        [rollout.values, bootstrap_values.unsqueeze(0)], dim=0
    )
    discounts = (~rollout.dones).float() * gamma * gae_lambda
    # deltas (single-step td-errors)
    advs = rollout.rewards + gamma * values_w_bootstrap[1:] - rollout.values
    for t in range(len(advs) - 2, -1, -1):
        advs[t] = advs[t] + discounts[t] * advs[t + 1]
    return advs


def update(
    buf: RolloutBuffer,
    args: DictConfig,
    scaler: GradScaler,
    opt: Optimizer,
    ac: ActorCritic,
) -> Tuple[dict, Tensor, Tensor]:

    rollout = buf.get_batch()

    with torch.no_grad():
        with autocast(enabled=args.amp):
            bootstrap_values, *_ = ac.step(rollout.obs[-1])

        advs = compute_gae_advs(
            rollout, bootstrap_values, args.opt.gamma, args.opt.gae_lambda
        )
        v_targets = compute_v_targets(rollout, bootstrap_values, args.opt.gamma)
        #v_targets = advs + rollout.values

        v_targets, advs, obs, actions, log_probs = map(
            lambda x: x.flatten(0, 1),
            (
                v_targets,
                advs,
                rollout.obs[:-1],
                rollout.actions,
                rollout.log_probs
            ),
        )

        # ppo batchwise advantage normalization
        advs -= torch.mean(advs)
        advs /= torch.std(advs).clamp(min=1e-6)

    batch_ind = np.arange(args.alg.rollout_len * args.alg.n_workers)
    metrics = defaultdict(list)
    batch_size = (args.alg.n_workers * args.alg.rollout_len) // args.alg.minibatches

    assert len(obs) == args.alg.rollout_len * args.alg.n_workers

    for epoch in range(args.alg.epochs):
        np.random.shuffle(batch_ind)
        for start in range(0, len(obs), batch_size):
            minibatch_inds = batch_ind[start:start + batch_size]

            batch_metrics = ppo_step(
                args,
                opt,
                scaler,
                ac,
                *[
                    x[minibatch_inds].detach()
                    for x in (
                        v_targets,
                        advs,
                        obs,
                        actions,
                        log_probs,
                    )
                ],
            )

            for k, v in batch_metrics.items():
                metrics[k].append(v)

    log_dict = {k: np.mean(v) for k, v in metrics.items()}
    return log_dict, advs.cpu(), v_targets.cpu()
