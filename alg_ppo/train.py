import signal
import time
from pathlib import Path

import wandb
from torch.cuda.amp import autocast
from tqdm import trange

from dark_mvp.ppo import utils, ppo_core
from dark_mvp.ppo.model import save_checkpoint
from dark_mvp.ppo.utils import ema


def run():
    data_dir = Path('dark_mvp_storage/ppo_data')
    args = utils.get_args(data_dir)
    save_path, env, ac, buf, opt, scaler = utils.setup_run(args, data_dir)

    progressbar = trange(
        0, args.training_frames + args.alg.n_workers, args.alg.n_workers
    )
    progressbar_iter = iter(progressbar)
    total_steps = 0
    iter_start = time.time()
    fps = 1
    buf.init(env.reset()[0])

    interrupted = False

    def signal_handler(signal, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, signal_handler)

    while total_steps < args.training_frames and not interrupted:
        # do env rollout
        for rollout_step in range(args.alg.rollout_len):
            with autocast(enabled=args.amp):
                actions, log_probs, values = [x.cpu() for x in ac.act(buf.get_obs())]

            *step_results, info = env.step(actions.numpy())
            for i in info:
                if 'episode' in i:
                    wandb.log(dict(total_steps=total_steps, ep_ret=i['episode']['r'], ep_len=i['episode']['l']))

            buf.push(*step_results, actions, values, log_probs)

            total_steps += args.alg.n_workers
            next(progressbar_iter)
            progressbar.set_description(
                f"[{total_steps} frames, "
                f"{round((args.training_frames - total_steps) / fps / 60 / 60, 1)} hours left]",
                refresh=True,
            )

            # save model checkpoint
            save_every = 1_000_000
            assert save_every % args.alg.n_workers == 0
            if (total_steps//args.alg.n_workers) % (save_every//args.alg.n_workers) == 0:
                save_checkpoint(save_path, total_steps, ac, opt)

        # update step
        log_dict, advs, v_targets = ppo_core.update(
            buf,
            args,
            scaler,
            opt,
            ac,
        )

        for pg in opt.param_groups:
            pg["lr"] = args.opt.lr * (1 - total_steps / args.training_frames)

        fps = 0.8 * fps + 0.2 * (args.alg.n_workers * args.alg.rollout_len) / (
            time.time() - iter_start
        )

        fps = ema(fps, (args.alg.n_workers * args.alg.rollout_len) / (time.time() - iter_start), 0.8)
        iter_start = time.time()
        wandb.log(dict(**log_dict, total_steps=total_steps, fps=fps))

        # prepare buffer for next rollout
        buf.rollover()


if __name__ == "__main__":
    run()
