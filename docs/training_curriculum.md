# Training Curriculum Guide

Last updated: 2025-11-15

This document describes the curriculum strategies implemented in this repository, how they are wired into the training loop, the available tuning knobs, recommended defaults, metrics to watch, and guidance for adding new curricula.

Contents
- Overview
- Implemented curriculum strategies
  - Task-level adaptive curriculum (CurriculumManager)
  - State-level adaptive curriculum (CurriculumWrapper)
  - Static / manual task-selection (task_ids)
  - Uniform / random sampling (implicit)
- How curricula are integrated into training
- Tuning recommendations and pitfalls
- Metrics & instrumentation
- Examples / CLI
- How to add a new curriculum strategy
- Troubleshooting

## Overview

A curriculum controls which tasks and/or initial states the agent sees during training. This codebase implements a competence-based curriculum that focuses sampling on the "learning frontier" — tasks or states where the agent's success rate is neither too high nor too low (roughly near 50%). The implementation operates at two levels:

1. Task-level: Decide which tasks to schedule for rollouts across the actor pool (global scheduler implemented as a Ray actor).
2. State-level: For a given task, decide which initial state (or episode start) to select when resetting the environment.

Both components implement a similar principle: prefer examples with success ≈ 0.5. This directs training to items that are neither trivial nor impossible and tends to improve sample efficiency.

## Implemented curriculum strategies

### Task-level adaptive curriculum — `CurriculumManager`

Files
- `vlarl/utils/curriculum.py`
- Created and used in `vlarl/ppo_vllm_ray_fsdp_v3.py`

Description
- Implemented as a Ray actor (`CurriculumManager`) that tracks an exponential moving average (EMA) success rate per task.
- It exposes two main RPCs:
  - `get_batch(batch_size)` — returns a list of sampled task IDs according to the adaptive distribution.
  - `report_results(task_ids, successes)` — updates the EMA per task using reported episodic outcomes and recomputes sampling probabilities.

Sampling math (summary)
- Each task's EMA success s ∈ [0,1] is kept (initialized to 0.5).
- A score for sampling is computed as:
  score = exp(-((s - 0.5)^2) / tau)
- Sampling probability p_i = score_i / sum_j score_j
- Intuition: tasks with s closer to 0.5 have larger score and higher sample probability; tau controls sharpness.

Configurable parameters (constructor / defaults)
- `num_tasks` (provided on instantiation)
- `ema_alpha` (default 0.99): smoothing factor for EMA updates; higher = slower change.
- `tau` (default 0.02): controls width/sharpness of the preference around 0.5.

When it's used
- `main()` attempts to instantiate the actor:
  `curriculum = CurriculumManager.options().remote(num_tasks=args.num_tasks_per_suite, ema_alpha=0.99, tau=0.02)`
- In the actor training loop (`PolicyTrainerRayProcess.train()`), the actor requests a batch of tasks for its local rollout size via `curriculum.get_batch.remote(...)`. If this call fails, training falls back to the preconfigured `args.task_ids` slice.
- After rollouts in each training iteration, the trainer reports episodic outcomes using `curriculum.report_results.remote(episodic_task_ids, episodic_returns)`.

When to use
- Enable a dynamic task scheduler (recommended for suites with many tasks and heterogeneous difficulty).

Pros and cons
- Pros: focuses training effort on the learning frontier, adapts to agent progress, simple and robust.
- Cons: tuned hyperparameters (ema_alpha/tau) affect responsiveness and exploration; extremely high alpha may be too slow, very low tau may concentrate too heavily.

### State-level adaptive curriculum — `CurriculumWrapper`

Files
- `vlarl/envs/wrappers.py` (class `CurriculumWrapper`)

Description
- The wrapper sits on top of the environment and replaces the environment's state sampler with a curriculum-aware sampler: `env.state_sampler = self._sample_state_with_curriculum`.
- For a given task, it queries the environment's `get_task_state_results()` (expected to return a mapping from (task_id, state_id) → list of observed success flags) and computes per-state empirical success rates.
- It computes a weight per state inversely proportional to the absolute distance to the target success (target hard-coded to 0.5), optionally softened by a temperature and floored by a minimum probability.

Sampling math (summary)
- For each state i with success rate r_i:
  distance = |r_i - 0.5|
  raw_weight = 1 / (distance + eps)
  weight = raw_weight ** (1 / temp)
  After normalizing weights to sum=1, each probability is floored at `min_prob / n_states`, renormalized if necessary.

Configurable parameters
- `temp` (wrapper arg; default used from `args.curriculum_temp`) — higher `temp` produces a flatter distribution; lower `temp` sharpens the preference.
- `min_prob` (wrapper arg; default from `args.curriculum_min_prob`) — minimum exploration floor across states.
- `recompute_freq` — how often wrapper adds curriculum stats to env `info` (used for logging). Default is configurable when instantiating wrapper.

When it's used
- `get_environment()` in `ppo_vllm_ray_fsdp_v3.py` wraps train environments in `CurriculumWrapper` when `args.use_curriculum` is True.
- During step/reset, the wrapper occasionally writes `curriculum_stats` into `info` for logging and metrics collection.

Pros and cons
- Pros: fine-grained control per initial state; focuses sampling on states at the learning frontier.
- Cons: requires the environment to store and return state-level outcomes (via `get_task_state_results()`); sensitive to `temp` and `min_prob` choices.

### Static / manual task selection — `task_ids`

Description
- The simplest option: pre-specify `args.task_ids` when launching training. The trainer uses these values (sliced per actor) and no adaptive scheduling is performed.
- This mode is used when `curriculum` actor is not provided or the `get_batch` call fails (the code falls back to the `args.task_ids` slice).

When to use
- When you want deterministic, reproducible task selection or want to manually control distribution of tasks.

### Uniform / random sampling

Description
- If you do not enable curriculum or set curriculum temperature/tau values such that the sampling distribution becomes approximately uniform (e.g., very large tau or temp), sampling approaches uniform random across tasks or states.

When to use
- Baseline comparison, or when you want to ensure even coverage across tasks without adaptive focusing.

## How curricula are integrated into training

- Task-level manager is created in `main()` and passed to each actor's `train()` call as the `curriculum` argument.
- In `PolicyTrainerRayProcess.train()` each actor calls `curriculum.get_batch(...)` (non-blocking RPC via Ray) at start of training to receive its task_ids for rollouts.
- After each training iteration, episodic outcomes (task ids and binary success signals) are reported back to the manager using `curriculum.report_results.remote(...)` so the EMA and sampling probabilities are updated.
- `CurriculumWrapper` is applied at the environment construction time by `get_environment()` when `args.use_curriculum` is True. The wrapper sets the environment's internal state sampler to use the wrapper's policy.

## Tuning recommendations & pitfalls

Hyperparameters
- Task manager:
  - `ema_alpha` (default 0.99): smoothing for per-task success EMA. Lower values make the manager react faster to changes but increase variance; higher values make it more stable but slow to adapt.
  - `tau` (default 0.02): small values make the manager focus tightly on success near 0.5; larger values flatten probabilities.
- Wrapper (state-level):
  - `temp` (`args.curriculum_temp`, default 1.0): >1 flattens weights, <1 sharpens preference.
  - `min_prob` (`args.curriculum_min_prob`, default 0.0): ensures exploration; pick small non-zero values to avoid starvation of rare states.

Practical tips
- Start with defaults (ema_alpha=0.99, tau=0.02, temp=1.0, min_prob=0.0). If the curriculum reacts too slowly, reduce `ema_alpha` to 0.95 or 0.9.
- If sampling collapses to too few tasks/states, increase `tau` or `temp`, or increase `min_prob` slightly to maintain exploration.
- If you see oscillation (task selected then quickly abandoned), increase `ema_alpha` (smoother) or lower `tau` (less sharp but more stable).
- Always monitor episode counts (`counts` in the manager stats) to ensure tasks/states are seeing enough coverage.

Pitfalls
- Starvation: with `min_prob=0` and very sharp sampling, some tasks/states may get almost no visits; set a small floor to avoid it.
- Wrong success reporting: make sure the reported `success` signal is consistently 0/1 and corresponds exactly to the manager's expectation; inconsistent reporting will corrupt EMAs.
- Too large `ema_alpha`: manager doesn't adapt (stays near 0.5) and sampling doesn't reflect actual agent competence.

## Metrics & instrumentation

- The `CurriculumManager` provides `get_stats()` returning:
  - `ema_success`: list of EMA success rates per task
  - `counts`: per-task visit counts
  - `probs`: current sampling probabilities per task

- The `CurriculumWrapper` occasionally writes `curriculum_stats` into the environment `info` dict; training collects these and logs them under the `curriculum/` prefix (keys like `curriculum/task_<id>_avg_success_rate`, `curriculum/total_visited_tasks`, etc.). The training loop shuttles these into the metrics queue and they are logged to TensorBoard / wandb.

- Watch metrics:
  - `curriculum/total_visited_tasks`
  - `curriculum/<task>_avg_success_rate`
  - The CurriculumManager `probs` and `counts` (via a debugging RPC or adding an explicit logging call) to detect starvation or over-concentration.

## Examples / CLI

Enable state-level curriculum and use the default task manager (task manager is created automatically in `main()`):

```bash
python ppo_vllm_ray_fsdp_v3.py \
  --use_curriculum True \
  --curriculum_temp 1.0 \
  --curriculum_min_prob 0.01 \
  --num_tasks_per_suite 10 \
  --task_ids "[0,1,2,3,4,5,6,7,8,9]"
```

Notes:
- `num_tasks_per_suite` is used when creating the `CurriculumManager` actor.
- The code will call `CurriculumManager.get_batch(...)` for each actor with the actor's `local_rollout_batch_size`.

If you prefer manual control (static tasks), omit `--use_curriculum` and supply `--task_ids`:

```bash
python ppo_vllm_ray_fsdp_v3.py --task_ids "[0,0,1,1,2,2,3,3,4,4]" --use_curriculum False
```

## How to add a new curriculum strategy

Suggested steps for a new global/task-level strategy
1. Implement a Ray actor similar to `CurriculumManager` with the same public RPCs:
   - `get_batch(batch_size)` — returns a list of task ids
   - `report_results(task_ids, successes)` — updates internals
   - Optional: `get_stats()` for debugging/monitoring
2. Replace creation in `main()` or provide a CLI option to select which manager to instantiate. The training loop expects the actor to implement those methods.
3. Tune and add metrics reporting so you can observe the manager's internal state.

Suggested local/state-level strategies
- Extend `CurriculumWrapper` or provide a different wrapper implementing `env.state_sampler` as desired. Keep the same API so the training code does not have to change.

Ideas for curriculum strategies you might implement
- Prioritized Replay-style: prioritize transitions/states by TD error or novelty.
- Uncertainty-based: sample tasks where the model's prediction entropy is high.
- Competence progress: prefer tasks where competence gain (delta of success over window) is highest.
- Rarity-based: maintain coverage by boosting rarely visited states/tasks until minimum visit counts are reached.

## Troubleshooting

- If `curriculum.get_batch` fails or raises exceptions, training falls back to `args.task_ids` slice; check logs to diagnose the failure and ensure the Ray actor is reachable.
- If sampling becomes degenerate (single task), increase `tau` or `temp` or set a non-zero `min_prob`.
- If reported success rates look wrong, verify the trainer is sending binary 0/1 success flags and that `episodic_task_ids` aligns with episodic_returns.

## Summary

- This repo provides a competence-based curriculum at both task and state levels, both focused on preferring items where success ≈ 0.5.
- Key knobs: `ema_alpha`, `tau` (task manager), and `curriculum_temp`, `curriculum_min_prob` (state wrapper).
- The curriculum is safe and modular: the training loop falls back to static `task_ids` if the manager is unavailable and the wrapper only affects sampling inside the env when enabled.

If you'd like, I can:
- Add a small example script that prints `CurriculumManager.get_stats()` periodically (for debugging),
- Add a `--curriculum_strategy` CLI argument to select between manager implementations, or
- Implement an alternative curriculum (e.g., competence-progress based) as a new Ray actor.

