from pathlib import Path
from typing import Tuple

import kornia
import numpy as np
import timm
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torchsummary import summary

from dojo.common.task_configs import global_task_ids
from impala.common.mobilenet import MobileNetV3
from impala.common.scalable_arch import ScalableArch
from impala.common.scalable_mbn import MobileNetV3Scalable
from impala.common.special import PopArt, Augmentation


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class IMPALANet(nn.Module):
    def __init__(self, in_depth, model_scale=1):
        super(IMPALANet, self).__init__()

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = in_depth
        for num_ch in [int(16 * model_scale), int(32 * model_scale), int(32 * model_scale)]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        #self.down = nn.Conv2d(
        #                in_channels=32*model_scale,
        #                out_channels=32,
        #                kernel_size=1,
        #                stride=1,
        #            )

        self.fc = nn.Linear(int(32*model_scale*9*9), 256)


    def forward(self, ob):
        x = ob

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)

        #x = self.down(x)
        #x = F.relu(x)

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc(x))

        return x






def get_nature_model(act_n, in_depth):
    main = nn.Sequential(
        layer_init(nn.Conv2d(in_depth, 32, 8, stride=4)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        layer_init(nn.Linear(64 * 5 * 5, 512)),
        nn.ReLU(),
    )
    actor = layer_init(nn.Linear(512, act_n), std=0.01)
    critic = layer_init(nn.Linear(512, 1), std=1)
    return main, actor, critic


def get_impala_tb_model(act_n, in_depth, model_scale):
    impala_cnn = IMPALANet(in_depth, model_scale)

    policy = nn.Linear(impala_cnn.fc.out_features, act_n)
    baseline = nn.Linear(impala_cnn.fc.out_features, 1)
    return impala_cnn, policy, baseline

def get_timm_model(act_n, in_depth, name):
    # https://rwightman.github.io/pytorch-image-models/models/vision-transformer/
    timm_model = timm.create_model(name, in_chans=in_depth) # , global_pool='catavgmax'
    timm_model.global_pool = timm.models.layers.SelectAdaptivePool2d(pool_type='catavgmax', flatten=True)
    timm_model.fc = nn.ReLU()

    actor = nn.Linear(1024, act_n)
    critic = nn.Linear(1024, 1)
    return timm_model, actor, critic

def get_mobilenet_model(act_n, in_depth):
    mobilenet = MobileNetV3Scalable(n_class=256, input_size=72, width_mult=0.5)

    policy = nn.Linear(256, act_n)
    baseline = nn.Linear(256, 1)
    return mobilenet, policy, baseline

def get_scalable_arch_model(act_n, in_depth, model_scale):
    sa = ScalableArch(in_depth=in_depth, model_scale=model_scale)

    policy = nn.Linear(256, act_n)
    baseline = nn.Linear(256, 1)
    return sa, policy, baseline

def get_model(act_n, in_depth, model_scale, arch_str):
    if arch_str == 'nature_cnn':
        return get_nature_model(act_n, in_depth)
    elif arch_str == 'impala_tb':
        return get_impala_tb_model(act_n, in_depth, model_scale)
    elif arch_str == 'mobilenet':
        return get_mobilenet_model(act_n, in_depth)
    elif arch_str == 'sa':
        return get_scalable_arch_model(act_n, in_depth, model_scale)
    else:
        return get_timm_model(act_n, in_depth, arch_str)



class ActorCritic(nn.Module):

    def __init__(self, in_depth, act_n, model_scale, arch_str, popart, popart_beta, augmentation):
        super().__init__()
        self.main, self.actor, self.critic = get_model(act_n, in_depth, model_scale, arch_str)
        self.popart = PopArt(popart_beta) if popart else False
        self.aug: Augmentation = augmentation

    @torch.no_grad()
    def act(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_logits = self.actor(self.main(obs))
        probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)

        actions = torch.multinomial(probs, num_samples=1).flatten()
        return actions, log_probs

    def step(self, obs, actions, task_ids=None, augment=False):
        if augment:
            obs = self.aug(obs)

        lat = self.main(obs)
        policy_logits = self.actor(lat)
        values = self.critic(lat).view(len(obs))

        probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        entropy = torch.mean(-torch.sum(probs * log_probs, dim=-1))

        norm_values = None
        if self.popart:
            norm_values, values = self.popart(values, task_ids)

        return values, norm_values, entropy, log_probs

    def save(self, save_dir: Path, opt: torch.optim.Optimizer, steps: int):
        print(f'Saving checkpoint to disk at step {steps}.')
        checkpoint_path = save_dir / f'checkpoints/{steps}.pt'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(dict(model=self.state_dict(), opt=opt.state_dict()), checkpoint_path)

    def load(self, path: Path) -> torch.optim.Optimizer:
        print(f'Loading model checkpoint from {path}.')
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])
        return checkpoint['opt']
