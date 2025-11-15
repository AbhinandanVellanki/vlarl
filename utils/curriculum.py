import math
from typing import List, Dict

import numpy as np
import ray


@ray.remote
class CurriculumManager:
    """Simple Ray actor that tracks per-task EMA success rates and serves sampled task ids.

    Behavior:
    - Tracks EMA success rate per task (initialized to 0.5).
    - get_batch(batch_size): returns a list of task ids sampled with probability higher when
      success rate is closer to 0.5 (configurable via tau).
    - report_results(task_ids, successes): updates EMA success rates and recomputes sampling probs.
    """

    def __init__(self, num_tasks: int, ema_alpha: float = 0.99, tau: float = 0.02):
        self.num_tasks = int(num_tasks)
        self.ema_alpha = float(ema_alpha)
        self.tau = float(tau)

        # initialize EMA success rates to 0.5 (encourages exploration)
        self.ema_success = np.ones(self.num_tasks, dtype=float) * 0.5
        self.counts = np.zeros(self.num_tasks, dtype=int)
        self._recompute_weights()

    def _recompute_weights(self):
        # weight tasks that have success close to 0.5 higher
        s = self.ema_success
        # avoid divide-by-zero
        tau = max(self.tau, 1e-8)
        scores = np.exp(-((s - 0.5) ** 2) / tau)
        probs = scores / (scores.sum() + 1e-12)
        self.probs = probs

    def get_batch(self, batch_size: int) -> List[int]:
        if self.num_tasks == 0:
            return []
        idxs = list(np.random.choice(self.num_tasks, size=batch_size, replace=True, p=self.probs))
        return idxs

    def report_results(self, task_ids: List[int], successes: List[int]) -> None:
        # Expect task_ids and successes to have same length
        for tid, suc in zip(task_ids, successes):
            tid = int(tid)
            self.counts[tid] += 1
            self.ema_success[tid] = self.ema_alpha * self.ema_success[tid] + (1.0 - self.ema_alpha) * float(suc)
        self._recompute_weights()

    def get_stats(self) -> Dict[str, any]:
        return {
            "ema_success": self.ema_success.tolist(),
            "counts": self.counts.tolist(),
            "probs": self.probs.tolist(),
        }
