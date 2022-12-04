import random

import kornia
import numpy as np
from gym.wrappers.normalize import RunningMeanStd
from kornia.morphology import dilation, erosion, opening, closing
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.distributions import Categorical

from dojo.common.task_configs import global_task_ids
import torch.nn.functional as F


def compute_dr_ac_loss(v, v_aug, log_probs, log_probs_aug) -> torch.Tensor:
    """
    Implements the data-regularized actor-critic loss from Raileanu et al. 2021.
    When using PopArt normalization, `v` and `v_aug` can (and should) be the normalized
    value estimates.

    Args:
        v: value estimate on batch of un-augmented observations
        v_aug: value estimate on batch of augmented observations
        log_probs: policy logits on batch of un-augmented observations
        log_probs_aug: policy logits on batch of augmented observations

    Returns:
        Unscaled DrAC loss.
    """
    assert len(log_probs_aug.shape) == 2

    print(v.shape, v_aug.shape, log_probs.shape, log_probs_aug.shape)

    v, log_probs = v.detach(), log_probs.detach()  # see paper section E

    G_v = F.mse_loss(v, v_aug)
    G_pi = torch.nn.functional.kl_div(
        log_probs_aug,
        log_probs,
        log_target=True,
        reduction='batchmean'
    )

    return G_v+G_pi


class Augmentation:

    def __init__(self, resolution, device):
        crop = nn.Sequential(nn.ReplicationPad2d(4), kornia.augmentation.RandomCrop((resolution, resolution)))
        dropout = nn.Dropout(p=0.01, inplace=True)
        blur = kornia.augmentation.RandomGaussianBlur((3, 3), (0.45, 0.45), p=1.0)
        noise = kornia.augmentation.RandomGaussianNoise(std=0.025, p=1.0)
        rotate = kornia.augmentation.RandomRotation(2.5, p=1.0)

        self.augmentations = [crop, dropout, blur, noise, rotate]
        self.kernels = [torch.tensor([[1, 0], [0, 1]]).to(device),
                        torch.tensor([[0, 1], [1, 0]]).to(device)]
        #for f in [dilation, erosion, opening, closing]:
        #    self.augmentations.append(
        #        lambda x: f(x, random.choice(self.kernels), border_type='constant'))

    @torch.no_grad()
    def __call__(self, obs):
        obs = torch.tensor_split(obs, 1, dim=0)
        obs = [random.choice(self.augmentations)(ob) for ob in obs]

        return torch.clamp(torch.cat(obs, dim=0), -1.0, 1.0)


class PopArt(nn.Module):

    def __init__(self, beta: float):
        super().__init__()

        self.beta = beta

        self.w = torch.nn.parameter.Parameter(torch.ones(len(global_task_ids)))
        self.b = torch.nn.parameter.Parameter(torch.zeros(len(global_task_ids)))
        self.register_buffer('sigma', torch.ones(len(global_task_ids)))
        self.register_buffer('mu', torch.zeros(len(global_task_ids)))
        self.register_buffer('nu', torch.ones(len(global_task_ids)))

    def forward(self, values, task_ids):
        # compute pop-art normalized vf n_theta(s)
        normalized_values = (self.w.gather(0, task_ids) * values +
                             self.b.gather(0, task_ids))

        # compute the unnormalized vf
        unnorm_values = (self.sigma.gather(0, task_ids) * normalized_values +
                         self.mu.gather(0, task_ids))

        return normalized_values, unnorm_values

    @torch.no_grad()
    def update(self, unnorm_v_targets, task_ids):
        mu_old, sigma_old = self.mu.clone(), self.sigma.clone()

        count = torch.zeros_like(self.nu).scatter_add_(0, task_ids, torch.ones_like(unnorm_v_targets))
        alpha = (1-self.beta)**count

        #print(count[count > 0].mean(), alpha[count > 0].mean())

        sum_ = torch.zeros_like(self.mu).scatter_add_(0, task_ids, unnorm_v_targets)
        self.mu[:] = alpha * self.mu + (1-alpha) * (sum_ / count.clamp(min=1))
        #self.mu[count > 0] = ((1-self.beta)*self.mu + self.beta*(sum_ / count.clamp(min=1)))[count > 0]

        sum_ = torch.zeros_like(self.nu).scatter_add_(0, task_ids, torch.square(unnorm_v_targets))
        self.nu[:] = alpha * self.nu + (1-alpha) * (sum_ / count.clamp(min=1))
        #self.nu[count > 0] = ((1-self.beta)*self.nu + self.beta*(sum_ / count.clamp(min=1)))[count > 0]

        self.sigma[:] = torch.clamp(torch.sqrt(self.nu - torch.square(self.mu)), min=0.0001, max=1e6)

        # output preserving updates
        self.w *= (sigma_old / self.sigma)
        self.b[:] = (sigma_old*self.b+mu_old-self.mu)/self.sigma


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RND():
    # adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rnd_ppo.py

    def __init__(self, device, amp):
        self.amp = amp
        feature_output = 5 * 5 * 64

        # Prediction network
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        ).to(device)

        # Target network
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
        ).to(device)
        # Set that target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=0.0001)
        self.scaler = GradScaler()
        self.rew_running_avg = RunningMeanStd()

    def __call__(self, obs):
        with autocast(enabled=self.amp):
            rnd_targ = self.target(obs)
            rnd_pred = self.predictor(obs)
            rnd_loss = F.mse_loss(rnd_pred, rnd_targ, reduction='none').sum(dim=1)

        self.opt.zero_grad()
        self.scaler.scale(rnd_loss.mean()).backward()
        self.scaler.step(self.opt)
        self.scaler.update()

        rnd_reward = rnd_loss.detach().cpu().numpy()
        self.rew_running_avg.update(rnd_reward)
        return rnd_reward / np.sqrt(self.rew_running_avg.var)
