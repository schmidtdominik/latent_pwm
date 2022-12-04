from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, depth):
        super().__init__()

        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=depth,
                out_channels=depth,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=depth,
                out_channels=depth,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        return x + self.main(x)


class ImpalaCNN(nn.Module):
    def __init__(self, in_depth, depth_scale, out_size=512):
        super().__init__()

        self.main = []
        for depth, size in zip([16, 32, 32], [36, 18, 8]):
            depth *= depth_scale

            self.main.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_depth,
                        out_channels=depth,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.MaxPool2d(3, 2, padding=1),
                    ResidualBlock(depth),
                    ResidualBlock(depth),
                    nn.ReLU(),
                )
            )
            in_depth = depth

        self.main = nn.Sequential(
            *self.main, nn.Flatten(), nn.Linear(depth * size**2, out_size)
        )

    def forward(self, x):
        return self.main(x)


class ActorCritic(nn.Module):
    def __init__(self, in_depth, act_n, model_depth_scale):
        super().__init__()

        encoder_out_size = 512
        self.encoder = ImpalaCNN(in_depth, model_depth_scale, encoder_out_size)
        self.policy = layer_init(nn.Linear(encoder_out_size, act_n), std=0.01)
        self.vf = layer_init(nn.Linear(encoder_out_size, 1), std=1)

    def forward(self, obs):
        lat = self.encoder(obs)
        policy_logits = self.policy(lat)
        values = self.vf(lat).view(len(obs))

        probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)

        return values, probs, log_probs

    @torch.no_grad()
    def act(self, obs) -> Tuple[Tensor, Tensor, Tensor]:
        values, probs, log_probs = self(obs)
        actions = torch.multinomial(probs, num_samples=1).flatten()

        return actions, log_probs, values

    def step(self, obs):
        values, probs, log_probs = self(obs)
        entropy = torch.mean(-torch.sum(probs * log_probs, dim=-1))

        return values, entropy, log_probs


def save_checkpoint(save_dir: Path, steps: int, model: nn.Module, opt: torch.optim.Optimizer):
    print(f"Saving checkpoint to disk at step {steps}.")
    checkpoint_path = save_dir / f"checkpoints/{steps}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict()), checkpoint_path)


def load_checkpoint(path: Path):
    print(f"Loading model checkpoint from {path}.")
    checkpoint = torch.load(path)
    return checkpoint["model"], checkpoint["opt"]
