import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchsummary import summary
from tqdm import trange

master_act = nn.LeakyReLU


class SAResidual(nn.Module):
    def __init__(self, depth):
        super().__init__()

        self.main = nn.Sequential(
            master_act(),
            nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1),
            master_act(),
            nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x+self.main(x)


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class ScalableArch(nn.Module):
    def __init__(self, in_depth, model_scale=1):
        super().__init__()

        self.main = []
        for l in [16, 32, 64]:
            l = l*model_scale
            self.main.append(nn.Sequential(
                nn.Conv2d(in_channels=in_depth, out_channels=l, kernel_size=3, padding=1),
                nn.MaxPool2d(3, 2, padding=1),
                SAResidual(l),
                SAResidual(l),
            ))
            in_depth = l

        self.main = nn.Sequential(
            *self.main,
            nn.AdaptiveAvgPool2d((6, 6)),
            master_act(),
            nn.Flatten(),
            nn.Linear(6*6*64, 256),
            master_act(),
        )

    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':
    from rich import print
    from torch.profiler import profile, record_function, ProfilerActivity

    bs = 80*21*3
    device = 'cuda'
    input = torch.rand(bs, 4, 72, 72).to(device).to(memory_format=torch.channels_last)
    net = ScalableArch(in_depth=4).to(device).to(memory_format=torch.channels_last)

    print(net)
    summary(net, (4, 72, 72), batch_size=bs, device='cuda')

    for i in range(3):
        net(input)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            for i in range(3):
                net(input)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

    t0 = time.time()

    with autocast():
        for i in trange(100):
            net(input)

    torch.cuda.synchronize()
    print(time.time() - t0)