import random
import shutil
import socket
import string
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf, DictConfig
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torchsummary import summary

from dojo import tasks
from dojo.common.task_configs import generate_task_configs
from dojo.dojo import Dojo
from impala.common import special
from impala.common.impala import RolloutBuffer
from impala.common.networks import ActorCritic

from rich import print as printr
from rich.syntax import Syntax

def get_run_name() -> str:
    """
    Generate a unique run name like "abcd-efgh"
    """
    s = random.choices(string.ascii_lowercase, k=9)
    s[4] = '-'
    return ''.join(s)


def parse_checkpoint_str(checkpoint_str: str) -> Tuple[Path, str]:
    """
    Parse checkpoint_str from config
    Args:
        checkpoint_str: string of form '[run_name]@[step]' or '[run_name]@latest'

    Returns:
        The actual path to the checkpoint and the given checkpoint_str, replacing 'latest' with
        the latest checkpoint.
    """
    eval_run_name, step = checkpoint_str.split('@')
    checkpoints_dir = Path(f'./data/{eval_run_name}') / 'checkpoints'

    if not checkpoints_dir.exists():
        raise ValueError(f'Could not find {checkpoints_dir}')

    checkpoints = [int(c.name.replace('.pt', '')) for c in checkpoints_dir.iterdir()]
    if step == 'latest':
        step = max(checkpoints + [-1])
    else:
        step = int(step)
    if step not in checkpoints:
        raise ValueError(f'Could not find checkpoint {step}')
    return checkpoints_dir / f'{step}.pt', f'{eval_run_name}@{step}'


def get_args() -> Tuple[DictConfig, Optional[Path]]:
    """
    Get and validate cli and config args
    """
    conf = OmegaConf.load('impala/config.yaml')
    OmegaConf.set_struct(conf, True)
    args = OmegaConf.merge(conf, OmegaConf.from_cli())
    OmegaConf.set_struct(args, False)

    if args.model.scale != 1 and not args.model.arch.startswith('impala'):
        raise ValueError('model_scale != 1 is only valid for impala-CNNs')

    args.run_name = get_run_name()
    checkpoint_path = None
    if args.checkpoint is not None:
        checkpoint_path, args.checkpoint = parse_checkpoint_str(args.checkpoint)

    return args, checkpoint_path


def setup_run(eval_run: bool) -> Tuple[DictConfig, Path, Dojo, ActorCritic, RolloutBuffer, Optional[Optimizer], Optional[GradScaler]]:
    args, checkpoint_path = get_args()
    if eval_run and checkpoint_path is None:
        raise ValueError('Must specify checkpoint to evaluate!')

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = not args.deterministic
    torch.backends.cudnn.deterministic = args.deterministic

    # save run in checkpoint dir if we are evaluating
    if eval_run:
        print(f'Evaluating checkpoint {args.checkpoint} on taskset {args.env.taskset}')
        save_dir = checkpoint_path.parent.parent / f'eval/{args.run_name}'
    else:
        save_dir = Path(f'./data/{args.run_name}')
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=args, f=str(save_dir/'config.yaml'), resolve=True)

    printr(Syntax(OmegaConf.to_yaml(args), 'yaml'))
    print(f'save_dir = {save_dir}')
    print(f'replay fraction = {round(args.alg.replay_factor/(args.alg.replay_factor + 1), 3)},'
          f' replay buffer size = {args.alg.replay_buffer_rollouts * args.alg.n_workers * args.alg.rollout_len}')

    dojo = Dojo(
        env_names=None,
        task_configs=tasks.tr_sets[args.env.taskset],
        sched_config=generate_task_configs(tasks.dojo_full)[1],
        log_dir=str(save_dir),
        n_workers=args.alg.n_workers, seed=args.seed,
        resolution=(args.env.resolution, args.env.resolution),
        grayscale=args.env.grayscale,
        record_individual=False,
        sched_policy=('least_episodes' if eval_run else 'least_frames'),
        return_norm=args.env.return_norm,
        reward_clipping=args.env.reward_clipping,
        aux_rewards=args.env.aux_rewards,
        eval_mode=eval_run
    )

    buf = RolloutBuffer(args.alg.rollout_len, args.alg.n_workers, args.opt.gamma, args.env.frame_stack, args.amp,
                        args.device, args.alg.replay_factor,
                        args.alg.replay_buffer_rollouts if not eval_run else 1)

    augmentation = special.Augmentation(args.env.resolution, args.device) if args.spec.dr_ac else None

    ac = ActorCritic(
        in_depth=args.env.frame_stack*(1 if args.env.grayscale else 3),
        act_n=15,
        model_scale=args.model.scale,
        arch_str=args.model.arch,
        popart=args.spec.popart,
        popart_beta=args.spec.popart_beta,
        augmentation=augmentation
    ).to(args.device)

    summary(ac.main, (args.env.frame_stack*(1 if args.env.grayscale else 3),
                      args.env.resolution, args.env.resolution),
            batch_size=args.alg.n_workers*(args.alg.rollout_len+1)*(args.alg.replay_factor+1), device=args.device)
    printr(ac.main)

    print(f'NN has {sum(p.numel() for p in ac.parameters())} parameters')

    opt, scaler = None, None
    if checkpoint_path is not None:
        # load checkpoint whether we are evaluating or not
        opt = ac.load(checkpoint_path)
        scaler = GradScaler(enabled=args.amp)
    elif not eval_run:
        # no checkpoint given and not evaluating
        opt = torch.optim.Adam(ac.parameters(), lr=args.opt.lr, eps=args.opt.adam_eps)
        scaler = GradScaler(enabled=args.amp)
    if eval_run: ac.eval()
    else: ac.train()

    wandb_config = dict(
        **OmegaConf.to_container(args, resolve=True),
        taskset=args.env.taskset,
        instance=socket.gethostname(),
        algorithm='impala'
    )
    wandb.init(project='dojo_v1', name=args.run_name, config=wandb_config,
               save_code=False, mode='offline' if eval_run else 'online')

    return args, save_dir, dojo, ac, buf, opt, scaler


def teardown(save_path: Path):
    shutil.rmtree(save_path)


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
        if self.iter <= 3: return
        m_t = (time.time() - self.t)
        self.data[name] = m_t*(1-self.alpha) + self.data.get(name, m_t)*self.alpha
        self.t = time.time()

    def __str__(self):
        return f'[{self.timer_name}]\n' + '\n'.join(f'| {k:<10} {v:9.3f}' for k, v in self.data.items())

    def print(self):
        if self.data: print(self)

    @property
    def log_dict(self):
        return {self.timer_name + '/' + k: v for k, v in self.data.items()}

