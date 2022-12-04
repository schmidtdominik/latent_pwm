import random
import time

import wandb
from omegaconf import DictConfig
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import torch
from torch.optim import Optimizer
import torch.nn.functional as F

from impala.common import vtrace, special
from impala.common.networks import ActorCritic
from impala.common.vtrace import action_log_probs


def slice_wrapped(arr, start, end, second_dim=...):
    if 0 <= start <= end <= len(arr):
        r = arr[start:end, second_dim]
    elif start < 0 <= end <= len(arr):
        r = torch.cat((arr[start:, second_dim], arr[:end, second_dim]), dim=0)
    elif 0 <= start <= len(arr) < end:
        r = torch.cat((arr[start:, second_dim], arr[:end%len(arr), second_dim]), dim=0)
    else:
        raise ValueError(f'Invalid slice: {start}:{end}, len={len(arr)}')

    if second_dim != ...:
        r = r.unsqueeze(1)
    return r

class RolloutBuffer:

    def __init__(self, rollout_len, num_envs, gamma, frame_stack, amp, device, replay_factor, replay_buffer_rollouts):
        """Set up storage"""
        self.gamma = gamma
        self.frame_stack, self.num_envs, self.rollout_len = frame_stack, num_envs, rollout_len
        self.amp, self.device = amp, device
        self.ptr, self.init = 0, False
        self.total_steps = 0
        self.last_get_steps = 0
        self.replay_factor, self.replay_buffer_rollouts = replay_factor, replay_buffer_rollouts

        # replay_buffer_rollouts + 1 (current rollout) + 1 (space for a frame_stack-1 prefix and 1 frame bootstrapping suffix)
        self.l = (2+replay_buffer_rollouts)*rollout_len
        self.obs = torch.zeros(self.l, num_envs, 72, 72, dtype=torch.uint8)
        self.task_ids = torch.zeros((self.l, num_envs), dtype=torch.int64)
        self.acts = torch.zeros((self.l, num_envs), dtype=torch.int64)
        self.rews = torch.zeros((self.l, num_envs), dtype=torch.float32)
        self.dones = torch.zeros((self.l, num_envs), dtype=torch.bool)
        self.behavior_log_probs = torch.zeros((self.l, num_envs, 15), dtype=torch.float32)

    def put_obs(self, obs, task_ids):
        self.obs[self.ptr] = obs
        self.task_ids[self.ptr] = torch.Tensor(task_ids)
        if not self.init:
            self.obs[-self.frame_stack+1:] = obs
            self.init = True

    def put(self, a, rew, log_prob, done):
        self.acts[self.ptr] = a
        self.rews[self.ptr] = torch.from_numpy(rew)
        self.behavior_log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = torch.from_numpy(done)
        self.ptr = (self.ptr + 1) % self.l
        self.total_steps += 1

    def get_curr_obs(self):
        """ Get current frame-stacked obs with shape (num_envs, frame_stack, res, res) """

        start, end = self.ptr - self.frame_stack + 1, self.ptr + 1
        inf_obs = slice_wrapped(self.obs, start, end)

        inf_obs = (inf_obs.transpose(0, 1)
                          .to(self.device, non_blocking=True)
                          .to(dtype=torch.float16 if self.amp else torch.float32, non_blocking=True))
        inf_obs /= 127.5
        inf_obs -= 1.0
        inf_task_ids = self.task_ids[self.ptr].to(self.device, non_blocking=True)
        return inf_obs, inf_task_ids

    def get(self):
        self.last_get_steps = self.total_steps

        # the last step is the bootstrapping step, we don't want to include that here
        slices = [(self.ptr - self.rollout_len, self.ptr, ...)]

        traj_start_range = (  # +1, -3 is just to be safe
            self.ptr + self.frame_stack + 1,
            self.ptr + self.replay_buffer_rollouts * self.rollout_len-3
        )
        if self.total_steps == self.ptr:
            traj_start_range = (
                0,
                self.ptr - 2 * self.rollout_len - 2
            )

        for _ in range(self.num_envs*self.replay_factor):
            eid = random.randint(0, self.num_envs-1)
            start = random.randint(traj_start_range[0], traj_start_range[1]) % self.l
            end = start + self.rollout_len
            slices.append((start, end, eid))

        train_obs = torch.cat([slice_wrapped(self.obs, start-self.frame_stack+1, end+1, eid) for (start, end, eid) in slices], dim=1)
        train_obs = train_obs.to(self.device).unfold(0, self.frame_stack, 1).permute(0, 1, 4, 2, 3) \
                             .to(dtype=torch.float16 if self.amp else torch.float32, non_blocking=True).contiguous()
        train_obs /= 127.5
        train_obs -= 1.0
        task_ids = torch.cat([slice_wrapped(self.task_ids, start, end+1, eid) for (start, end, eid) in slices], dim=1).to(self.device)

        arrs = []
        for arr in (self.acts, self.rews, self.dones, self.behavior_log_probs):
            arrs.append(torch.cat([slice_wrapped(arr, start, end, eid) for (start, end, eid) in slices], dim=1).to(self.device))

        result = [train_obs] + arrs + [task_ids]

        return result

    def ready(self):
        k = (self.total_steps-self.last_get_steps)
        return k>0 and k % self.rollout_len == 0

    def replay_ready(self):
        return self.total_steps/self.rollout_len >= self.replay_buffer_rollouts//2


def compute_baseline_loss(advantages):
    return 0.5 * torch.mean(advantages**2) # fixme: doing *0.5 twice?


def compute_policy_gradient_loss(log_probs, advantages):
    policy_gradient_loss_per_timestep = -log_probs.view_as(advantages) * advantages.detach()
    return torch.mean(policy_gradient_loss_per_timestep)


def vtrace_update(args: DictConfig,
                  buf: RolloutBuffer,
                  ac: ActorCritic,
                  opt: Optimizer,
                  scaler: GradScaler) -> dict:
    """
    Perform one V-trace update step on the given actor-critic model using data from the
    rollout buffer.

    Args:
        args: ConfigDict containing the hyperparameters.
        buf: RolloutBuffer that is .ready()
        ac: ActorCritic model to update.
        opt: Optimizer for the ac.
        scaler: GradScaler for amp.

    Returns:
        A dictionary with log metrics.
    """
    obs, actions, rewards, dones, behav_log_probs, task_ids = buf.get()
    obs = obs.flatten(0, 1)

    batch_size = actions.shape[1]
    discounts = (~dones).float() * args.opt.gamma

    # actions at the value bootstrap step don't matter
    actions_padded = torch.cat(
        [actions, torch.zeros((1, batch_size), device=args.device, dtype=torch.long)],
        dim=0
    ).flatten(0, 1)

    opt.zero_grad()
    with autocast(enabled=args.amp):
        # compute target policy value estimates, policy, and entropy
        values, norm_values, entropy, target_log_probs = ac.step(
            obs=obs,
            actions=actions_padded,
            task_ids=task_ids.flatten(0, 1),
            augment=False
        )

        if args.spec.dr_ac:
            # do forward pass on small batch of augmented obs
            batch_ind = torch.randperm(len(obs), device=args.device)[:args.spec.dr_ac_samples]

            aug_values, aug_norm_values, _, aug_log_probs = ac.step(
                obs=obs[batch_ind],
                actions=actions_padded[batch_ind],
                task_ids=task_ids.flatten(0, 1)[batch_ind],
                augment=True
            )

        values = values.view(args.alg.rollout_len + 1, batch_size)
        traj_values = values[:-1]
        target_log_probs = target_log_probs.view(args.alg.rollout_len + 1, batch_size, 15)[:-1]

        behav_action_log_probs = action_log_probs(behav_log_probs, actions)
        target_action_log_probs = action_log_probs(target_log_probs, actions)

        # compute V-trace targets and advantages
        vtrace_returns = vtrace.from_logits(
            behavior_policy_log_probs=behav_action_log_probs,
            target_policy_log_probs=target_action_log_probs,
            actions=actions,
            discounts=discounts,
            rewards=rewards,
            values=traj_values,
            bootstrap_value=values[-1]
        )

        pg_adv = vtrace_returns.pg_advs
        pg_adv_unnorm = pg_adv.clone()
        td = vtrace_returns.vs - traj_values

        with torch.no_grad():
            # https://andyljones.com/posts/rl-debugging.html
            residual_var = torch.var(td.flatten().to(torch.float32))/torch.var(vtrace_returns.vs.flatten().to(torch.float32))

            # shape of log_probs is (rollout_len, n_workers*replay_factor) here, e.g. (20, 400)
            # [:, :n_workers] is the on-policy batch, rest is replay
            kl_div = F.kl_div(target_log_probs.flatten(0, 1), behav_log_probs.flatten(0,1),
                                                reduction='batchmean', log_target=True)

            #if kl_div > 3.0:
            #    print(f'Skipping update, KL-div is {kl_div.item()}')
            #    return {'kl_div': kl_div.item()}

        if args.spec.popart:
            sigma = ac.popart.sigma.gather(0, task_ids[:-1].view(-1))
            shape = pg_adv.shape
            pg_adv = pg_adv.view(-1).div(sigma).view(shape)
            td = td.view(-1).div(sigma).view(shape) # todo: directly use `normalized_values` for computing the vf loss?

        with torch.no_grad():
            td_errors = torch.abs(td).flatten()
            if args.spec.popart:
                td_errors_norm = td_errors.div(sigma)

        # ppo-style advantage normalization
        #mean, std = torch.mean(pg_adv), torch.std(pg_adv)
        #pg_adv -= mean
        #pg_adv /= std.clip(min=1e-2, max=1e2)

        pg_loss = compute_policy_gradient_loss(target_action_log_probs, pg_adv)
        baseline_loss = compute_baseline_loss(td)

        loss = (
            pg_loss
            + args.opt.baseline_cost * baseline_loss
            - args.opt.entropy_cost * entropy
        )
        if args.spec.dr_ac:
            dr_ac_loss = special.compute_dr_ac_loss(
                v=(norm_values if args.spec.popart else values.flatten())[batch_ind],
                v_aug=aug_norm_values if args.spec.popart else aug_values,
                log_probs=target_log_probs.flatten(0, 1)[batch_ind],
                log_probs_aug=aug_log_probs
            )
            loss = loss + args.spec.dr_ac_cost * dr_ac_loss

    scaler.scale(loss).backward()
    scaler.unscale_(opt)

    grad_norm = torch.nn.utils.clip_grad_norm_(ac.parameters(), args.opt.max_grad_norm)

    scaler.step(opt)
    scaler.update()

    log_dict = dict(
        policy_loss=pg_loss.item(),
        value_loss=baseline_loss.item(),
        grad_norm=grad_norm.item(),
        ent=entropy.item(),
        values=wandb.Histogram(values.flatten().detach().cpu()),
        td_errors=wandb.Histogram(td_errors.flatten().detach().cpu()),
        pg_adv=wandb.Histogram(pg_adv_unnorm.flatten().detach().cpu()),
        norm_pg_adv=wandb.Histogram(pg_adv.flatten().detach().cpu()),
        value_targets=wandb.Histogram(vtrace_returns.vs.flatten().detach().cpu()),
        rewards=wandb.Histogram(rewards.flatten().detach().cpu()),
        residual_var=residual_var.item(),
        kl_div=kl_div.item()
    )

    if args.spec.popart:
        log_dict['norm_values'] = wandb.Histogram(norm_values.detach().cpu())
        log_dict['td_errors_norm'] = wandb.Histogram(td_errors_norm.detach().cpu())
        # update popart running stats
        ac.popart.update(vtrace_returns.vs.flatten(), task_ids[:-1].flatten())
        #log_dict['popart_sigma'] = wandb.Histogram(ac.popart.sigma.detach().cpu())
        #log_dict['popart_mu'] = wandb.Histogram(ac.popart.mu.detach().cpu())

    if args.spec.dr_ac:
        log_dict['dr_ac_loss'] = dr_ac_loss.item()

    return log_dict