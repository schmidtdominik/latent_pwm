import random
import signal
import time

import numpy as np
import torch
import wandb
from torch.cuda.amp import autocast
from torch.nn.functional import mse_loss
from tqdm import trange

from dojo.common.task_configs import global_task_ids
from impala.common import utils
from impala.common.impala import vtrace_update
from impala.common.special import RND


def run():
    args, save_path, dojo, ac, buf, opt, scaler = utils.setup_run(eval_run=False)

    if args.spec.rnd:
        rnd = RND(args.device, args.amp)

    progressbar = trange(0, args.training_frames + args.alg.n_workers, args.alg.n_workers)
    progressbar_iter = iter(progressbar)
    total_steps = 0
    iter_start_t = time.time()
    fps = 1000

    interrupted = False
    def signal_handler(signal, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, signal_handler)
    #signal.signal(signal.SIGTERM, signal_handler)

    min_ep_count = 0

    while total_steps < args.training_frames:
        if buf.ready() and buf.replay_ready():

            # decay the lr
            for g in opt.param_groups:
                g['lr'] = args.opt.lr * max(1 - total_steps / args.training_frames, 0.001)

            fps = 0.9*fps + 0.1*(args.alg.n_workers*args.alg.rollout_len) / (time.time()-iter_start_t)
            iter_start_t = time.time()

            min_ep_count = min(map(len, dojo.logger.task_ep_stats.values()))
            log_dict = dict(
                **vtrace_update(args, buf, ac, opt, scaler),
                actual_lr=g['lr'],
                steps_per_sec=fps,
                hpr=100_000_000/fps/60/60,
                total_steps=total_steps,
                min_ep_count=min_ep_count,
            )

            if args.spec.rnd:
                log_dict['rnd_reward'] = rnd_reward.mean()

            wandb.log(log_dict)

            #for k, m in log_dict.items():
            #    dojo.logger.alg_metrics[k].append((dojo.logger.total_steps, m))

        else:
            if interrupted: break
            ob_batch, rew_batch, done_batch, info_batch, task_ids = dojo.step_wait()
            ob_batch = ob_batch.squeeze()

            if total_steps > 0:
                if args.spec.rnd:
                    # todo (bug): the rnd reward is one step behind
                    rew_batch += args.spec.rnd_coeff*rnd_reward
                buf.put(a, rew_batch, log_probs, done_batch)
            buf.put_obs(ob_batch, task_ids)

            total_steps += args.alg.n_workers
            next(progressbar_iter)
            progressbar.set_description(
                f'[{dojo.logger.total_steps} frames, '
                f'{dojo.logger.total_episodes:} eps, '
                f'{min_ep_count:} min eps, '
                f'{round((args.training_frames-total_steps)/fps/60/60, 1)} hours left]',
                refresh=True
            )

            with autocast(enabled=args.amp), torch.no_grad():
                inf_obs, _ = buf.get_curr_obs()
                a, log_probs = map(torch.Tensor.cpu, ac.act(inf_obs))
            dojo.step_async(a)

            if args.spec.rnd:
                rnd_reward = rnd(inf_obs)

        if (total_steps // args.alg.n_workers) % (1_000_000 // args.alg.n_workers * 5) == 0:
            ac.save(save_path, opt, total_steps)

    dojo.close()
    ac.save(save_path, opt, total_steps)
    wandb.finish()

    if total_steps < 300_000:
        utils.teardown(save_path)


if __name__ == '__main__':
    run()
