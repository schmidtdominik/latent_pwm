import signal
from itertools import count

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from ppo.common import utils

def run():
    args, save_path, dojo, ac, buf, _, _ = utils.setup_run(eval_run=True)

    progressbar = tqdm(count(0, 1))
    progressbar_iter = iter(progressbar)
    total_steps = 0

    interrupted = False
    def signal_handler(signal, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, signal_handler)

    ob_batch, _, _, _, task_ids = dojo.step_wait()
    buf.obs[0:args.env.frame_stack] = torch.from_numpy(ob_batch)
    buf.task_ids[0] = torch.from_numpy(task_ids)

    while (min_ep_count := min(map(len, dojo.logger.task_ep_stats.values()))) < args.eval_eps and not interrupted:

        obs, task_ids = buf.get_curr_obs()
        with autocast(enabled=args.amp), torch.no_grad():
            actions, log_probs, values, state_repr = map(torch.Tensor.cpu, ac.act(obs))

        dojo.step_async(actions)
        step_result = dojo.step_wait()
        buf.push(step_result, actions, values, log_probs, state_repr)

        total_steps += args.alg.n_workers
        next(progressbar_iter)
        progressbar.set_description(f'[{dojo.logger.total_steps} frames, {dojo.logger.total_episodes:} episodes, {min_ep_count} min task episodes]', refresh=True)

        if buf.ptr == buf.rollout_len:
            buf.rollover()

    dojo.close()
    if total_steps < 100_000:
        utils.teardown(save_path)

if __name__ == '__main__':
    run()
