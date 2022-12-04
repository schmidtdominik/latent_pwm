from typing import NamedTuple

import torch


class RolloutBatch(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor


class RolloutBuffer:

    def __init__(self, rollout_len, n_workers, frame_stack, amp, device, depth, res, n_actions):
        """Set up storage"""
        self.frame_stack, self.n_workers = frame_stack, n_workers
        self.amp, self.device, self.depth = amp, device, depth
        self.rollout_len = rollout_len
        self.obs_dtype = torch.float16 if self.amp else torch.float32

        # obs buffer has length increased by frame_stack-1 to allow for the frame-stacking for the first observation
        # and by +1 for the final value bootstrapping
        self.obs = torch.zeros(rollout_len + frame_stack, n_workers, res, res, depth, dtype=torch.uint8, pin_memory=True)
        self.actions = torch.zeros(rollout_len, n_workers, dtype=torch.int64, pin_memory=True)
        self.rewards = torch.zeros(rollout_len, n_workers, dtype=torch.float32, pin_memory=True)
        self.values = torch.zeros(rollout_len, n_workers, dtype=torch.float32, pin_memory=True)
        self.log_probs = torch.zeros(rollout_len, n_workers, n_actions, dtype=torch.float32, pin_memory=True)
        self.dones = torch.zeros(rollout_len, n_workers, dtype=torch.bool, pin_memory=True)

        self.ptr = None

    def init(self, obs):
        self.obs[0:self.frame_stack] = torch.from_numpy(obs)
        self.ptr = 0

    def push(self, obs, rewards, dones, actions, values, log_probs):
        assert self.ptr < self.rollout_len

        # obs are one step further along than a, rew, v, and logp
        self.obs[self.ptr + self.frame_stack] = torch.from_numpy(obs)

        self.rewards[self.ptr] = torch.from_numpy(rewards)
        self.dones[self.ptr] = torch.from_numpy(dones)
        self.actions[self.ptr] = actions
        self.values[self.ptr] = values
        self.log_probs[self.ptr] = log_probs

        self.ptr += 1

    def get_obs(self) -> torch.Tensor:
        """ Get current frame-stacked obs with shape (num_envs, frame_stack, res, res) """

        inf_obs = self.obs[self.ptr:self.ptr + self.frame_stack].permute(1, 0, 4, 2, 3)
        inf_obs = inf_obs.reshape(self.n_workers, self.frame_stack * self.depth, *self.obs.shape[2:4])
        inf_obs = inf_obs\
            .to(self.device, non_blocking=True)\
            .to(dtype=self.obs_dtype, non_blocking=True)

        return inf_obs / 255.0

    def get_batch(self) -> RolloutBatch:
        assert self.ptr == self.rollout_len

        train_obs = self.obs\
            .to(self.device, non_blocking=True)\
            .to(dtype=self.obs_dtype, non_blocking=True)\
            .unfold(0, self.frame_stack, 1)\
            .permute(0, 1, 4, 5, 2, 3)

        train_obs = train_obs\
            .reshape(self.rollout_len+1, self.n_workers, self.depth * self.frame_stack, *train_obs.shape[4:])\
            .contiguous()  # important! normalization is in-place op

        train_obs /= 255.0

        return RolloutBatch(
            train_obs,
            *map(lambda x: x.to(self.device, non_blocking=True),
                [
                    self.actions,
                    self.rewards,
                    self.values,
                    self.log_probs,
                    self.dones
                ]
             )
        )

    def rollover(self):
        assert self.ptr == self.rollout_len
        # store first frame-stacked obs for the next rollout
        self.obs[:self.frame_stack] = self.obs[-self.frame_stack:]
        self.ptr = 0


