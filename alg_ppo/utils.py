import random
import shutil
import socket
import string
import time
from pathlib import Path
from typing import Tuple, Optional

import gym
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf, DictConfig
from procgen import ProcgenEnv
from rich.syntax import Syntax
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torchsummary import summary

from dark_mvp.ppo.model import ActorCritic, load_checkpoint
from dark_mvp.ppo.buffer import RolloutBuffer

from rich import print as printr


def ema(prev_v, update_v, alpha=0.99):
    return prev_v * alpha + update_v * (1 - alpha)


class PiecewiseLinearSchedule:
    def __init__(self, points, values):
        self.points = points
        self.values = values
        for i in range(len(self.points) - 1):
            if self.points[i + 1] < self.points[i]:
                raise ValueError("points must be monotonically increasing")
        if len(self.points) != len(self.values):
            raise ValueError("points and values need to be of the same length")

    def __call__(self, t):
        if not self.points[0] <= t <= self.points[-1]:
            raise ValueError("t must be in the interval [points[0], points[-1]]")

        for i in range(len(self.points) - 1):
            if self.points[i] <= t <= self.points[i + 1]:
                interp = (t - self.points[i]) / (self.points[i + 1] - self.points[i])
                return self.values[i] * (1 - interp) + self.values[i + 1] * (interp)


class Timer:
    def __init__(self, timer_name, alpha=0.95):
        self.alpha = alpha
        self.timer_name = timer_name
        self.iter = 0
        self.data = {}

    def reset(self):
        self.iter += 1
        self.t = time.time()

    def finish(self, name):
        if self.iter <= 3:
            return
        m_t = time.time() - self.t
        self.data[name] = m_t * (1 - self.alpha) + self.data.get(name, m_t) * self.alpha
        self.t = time.time()

    def __str__(self):
        return f"[{self.timer_name}]\n" + "\n".join(
            f"| {k:<10} {v:9.3f}" for k, v in self.data.items()
        )

    def print(self):
        if self.data:
            print(self)

    @property
    def log_dict(self):
        return {self.timer_name + "/" + k: v for k, v in self.data.items()}


def get_run_name() -> str:
    """
    Generate a unique run name like "abcd-efgh"
    """
    s = random.choices(string.ascii_lowercase, k=9)
    s[4] = "-"
    return "".join(s)


def parse_checkpoint_str(checkpoint_str: str, data_dir: Path) -> Tuple[Path, str]:
    """
    Parse checkpoint_str from config
    Args:
        checkpoint_str: string of form '[run_name]@[step]' or '[run_name]@latest'

    Returns:
        The actual path to the checkpoint and the given checkpoint_str, replacing 'latest' with
        the latest checkpoint.
    """
    eval_run_name, step = checkpoint_str.split("@")
    checkpoints_dir = data_dir / eval_run_name / "checkpoints"

    if not checkpoints_dir.exists():
        raise ValueError(f"Could not find {checkpoints_dir}")

    checkpoints = [int(c.name.replace(".pt", "")) for c in checkpoints_dir.iterdir()]

    step = max(checkpoints + [-1]) if step == "latest" else int(step)
    if step not in checkpoints:
        raise ValueError(f"Could not find checkpoint {step}")

    return checkpoints_dir / f"{step}.pt", f"{eval_run_name}@{step}"


def get_args(data_dir) -> DictConfig:
    """
    Get and validate cli and config args
    """
    conf = OmegaConf.load("dark_mvp/ppo/config.yaml")
    OmegaConf.set_struct(conf, True)
    args = OmegaConf.merge(conf, OmegaConf.from_cli())
    OmegaConf.set_struct(args, False)

    args.run_name = get_run_name()
    args.checkpoint_path = None
    if args.checkpoint is not None:
        args.checkpoint_path, args.checkpoint = parse_checkpoint_str(
            args.checkpoint, data_dir
        )

    assert (args.alg.n_workers * args.alg.rollout_len) % args.alg.minibatches == 0
    return args


def create_env(n_workers, procgen_name, distribution_mode, gamma):
    env = ProcgenEnv(
        n_workers,
        env_name=procgen_name,
        distribution_mode=distribution_mode,
    )

    env = gym.wrappers.TransformObservation(env, lambda obs: obs["rgb"])
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space["rgb"]
    env.is_vector_env = True
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    return env


def setup_run(
    args, data_dir
) -> Tuple[
    Path,
    gym.Env,
    ActorCritic,
    RolloutBuffer,
    Optional[Optimizer],
    Optional[GradScaler],
]:

    assert not args.eval_mode  # not fully implemented yet

    if args.eval_mode and args.checkpoint_path is None:
        raise ValueError("Must specify checkpoint to evaluate!")

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = not args.deterministic
    torch.backends.cudnn.deterministic = args.deterministic

    # save run in checkpoint dir if we are evaluating
    if args.eval_mode:
        print(f"Evaluating checkpoint {args.checkpoint}")
        save_dir = args.checkpoint_path.parent.parent / "eval" / args.run_name
    else:
        save_dir = data_dir / args.run_name

    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=args, f=str(save_dir / "config.yaml"), resolve=True)

    printr(Syntax(OmegaConf.to_yaml(args), "yaml"))
    print(f"save_dir = {save_dir}")

    env = create_env(
        args.alg.n_workers, args.env.name, args.env.distribution_mode, args.opt.gamma
    )

    resolution = 64
    act_n = env.action_space.n

    buf = RolloutBuffer(
        args.alg.rollout_len,
        args.alg.n_workers,
        args.env.frame_stack,
        args.amp,
        args.device,
        3,
        resolution,
        act_n
    )

    ac = ActorCritic(
        in_depth=3*args.env.frame_stack,
        act_n=act_n,
        model_depth_scale=args.model_depth_scale,
    ).to(args.device)
    print(f"NN has {sum(p.numel() for p in ac.parameters())} parameters")

    #summary(
    #    ac.encoder,
    #    (3*args.env.frame_stack, resolution, resolution),
    #    batch_size=(args.alg.n_workers * args.alg.rollout_len) // args.alg.minibatches,
    #    device=args.device,
    #)

    opt = torch.optim.Adam(ac.parameters(), lr=args.opt.lr, eps=args.opt.adam_eps)
    scaler = GradScaler(enabled=args.amp)

    if args.checkpoint_path is not None:
        # load checkpoint whether we are evaluating or not

        model_sd, opt_sd = load_checkpoint(args.checkpoint_path)
        ac.load_state_dict(model_sd)
        opt.load_state_dict(opt_sd)

    ac.eval() if args.eval_mode else ac.train()

    wandb_config = dict(
        **OmegaConf.to_container(args, resolve=True),
        instance=socket.gethostname(),
        algorithm="ppo",
    )

    wandb.init(
        project="tp_dark_mvp_ppo",
        name=args.run_name,
        config=wandb_config,
        save_code=False,
        mode="offline" if args.eval_mode else "online",
    )

    return save_dir, env, ac, buf, opt, scaler


def teardown(save_path: Path):
    shutil.rmtree(save_path)
