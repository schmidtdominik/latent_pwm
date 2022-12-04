import signal
from itertools import count

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from impala.common import utils


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
    signal.signal(signal.SIGTERM, signal_handler)

    while (min_ep_count := min(map(len, dojo.logger.task_ep_stats.values()))) < args.eval_eps and not interrupted:
        ob_batch, rew_batch, done_batch, info_batch, task_ids = dojo.step_wait()
        ob_batch = ob_batch.squeeze()
        if total_steps > 0:
            buf.put(a, rew_batch, log_probs, done_batch)
        buf.put_obs(ob_batch, task_ids)

        total_steps += args.alg.n_workers
        next(progressbar_iter)
        progressbar.set_description(f'[{dojo.logger.total_steps} frames, {dojo.logger.total_episodes:} episodes, {min_ep_count} min task episodes]', refresh=True)

        with autocast(enabled=args.amp), torch.no_grad():
            inf_obs, inf_task_ids = buf.get_curr_obs()
            a, log_probs = map(torch.Tensor.cpu, ac.act(inf_obs))
        dojo.step_async(a)

    dojo.close()
    if total_steps < 100_000:
        utils.teardown(save_path)

if __name__ == '__main__':
    run()
