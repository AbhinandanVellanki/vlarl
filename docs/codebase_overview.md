# VLA-RL Codebase Overview (Updated)

Repository root: `vlarl/`

Last updated: 2025-11-15

Purpose
-------
This repository fine-tunes vision+language action-prediction models (OpenVLA-style) with reinforcement learning. The primary training loop uses PPO and integrates three complex subsystems:

- Ray for multi-process orchestration and GPU placement.
- FSDP (PyTorch FullyShardedDataParallel) to shard very large models across GPUs.
- vLLM for fast, large-scale autoregressive inference used during rollouts.

Design goals
------------
- Support PEFT/LoRA adapters for parameter-efficient tuning.
- Keep vLLM inference decoupled from FSDP training while allowing efficient parameter broadcasts.
- Provide robust reward computation (learned PRMs and heuristics) with graceful fallbacks on failure.

Top-level layout (important paths)
----------------------------------

- `vlarl/ppo_vllm_thread_ray_fsdp_vla_v3.py` — main training/orchestration script. Drives actors, FSDP wrapping, vLLM broadcasts, rollouts, and PPO updates.
- `vlarl/scripts/` — convenience shell scripts (cluster or local launch helpers):
  - `train_rl_vllm_ray_fsdp.sh`, `eval_vllm_ray.sh`, `finetune.sh`, etc.
- `vlarl/ppo/` — core PPO code and helpers:
  - `envs/` — env wrappers (e.g., `libero_env.py`) used for rollouts.
  - `models/` — critic implementations and policy-related model helpers.
  - `utils/` — batching, vLLM helpers, ray helpers, logging and FSDP utilities.
- `vlarl/prismatic/` — model/processor library for OpenVLA-compatible models and tokenizers (custom HF registrations, processors, and model classes).
- `vlarl/experiments/robot/` — task-specific utilities and reward helpers used in robotics-focused experiments.
- `vlarl/utils/` — misc helpers: FSDP utilities, logging, ray utilities, vLLM helpers.

Key concepts and components
---------------------------

1) Model loading & PEFT
   - Custom HF types are registered so that `transformers`/`accelerate` can instantiate the OpenVLA model classes and processors.
   - Code supports loading a base pretrained VLA model (`AutoModelForVision2Seq.from_pretrained`) and optionally wrapping it with PEFT adapters (`PeftModel.from_pretrained`) or building LoRA adapters in-place.
   - For memory-efficiency the code prefers mixed precision (bfloat16) and can prepare the model for k-bit training when quantization/low-precision is enabled.

2) FSDP wrapping and device mesh
   - A device mesh is initialized and an FSDP wrap policy is selected to shard the model consistently by module/type.
   - When broadcasting parameters to vLLM, `FSDP.summon_full_params` is used locally and parameters are serialized/batched so vLLM can be supplied with full-precision parameters on a device without triggering huge memory spikes.

3) vLLM integration
   - vLLM engines are responsible for fast generation during rollouts. The training process broadcasts model weights to vLLM engines and uses a queue-based RPC style to send prompts and receive sampled actions/logprobs.
   - The code contains utilities to translate model parameter names between PEFT/LoRA-named params and the full-model param names vLLM expects.

4) Curriculum & datasets
   - The dataset layer (`RLDSDataset`, batch transforms) builds rollouts from RLDS/Open-X-style episode files and task definitions.
   - Prompt builders and processors (e.g., `ActionTokenizer`, `PrismaticImageProcessor`) construct model inputs for vision+language tasks.
   - Curriculum is driven by `Args` (task suites, `task_ids`, `num_tasks_per_suite`, `num_trials_per_task`) and slicing is performed per actor to distribute tasks across rollouts.

5) Reward machinery
   - Rewards may come from learned reward models (PRMs), heuristics (stop-token checks, length penalties), or from a value model (critic).
   - `safe_get_reward` wraps reward computation to catch CUDA/worker failures and fall back to CPU-based repro or a zero reward to keep actors alive.

6) Training loop (PPO)
   - High-level flow in `PolicyTrainerRayProcess.train()`:
     1. Collect rollouts across `num_steps` per iteration using vLLM generations.
     2. Compute advantages (GAE) and optionally normalize/whiten them.
     3. Run PPO minibatch updates across configured epochs (policy and value updates, clipping, entropy, etc.).
     4. Synchronize parameters, broadcast to vLLM, and checkpoint.
   - Supports async modes where actors use slightly stale policies to improve throughput.

Operational notes and common issues
---------------------------------

- If Git or local operations show a branch/worktree issue (e.g., `used by worktree`), inspect `.git` metadata and `git worktree list` to locate/clean stale entries.
- CUDA OOM during parameter broadcast: lower broadcast batch size, enable CPU offload, or use fewer engines per node.
- vLLM deadlocks (NCCL or process-group issues): confirm `vllm_sync_backend` setting, ensure correct NCCL env vars, and that engine group sizes match worker counts.
- Ray workers crashing during reward computation: check `safe_get_reward` logs — it will attempt CPU repro and write tracebacks to help diagnose.

Quick start — single GPU developer flow
-------------------------------------

1) Create a small dev `Args` override (or edit script) for a smoke run:

```bash
# Example: run one actor locally (adjust paths and envs as needed)
python -m vlarl.pp o_vllm_thread_ray_fsdp_vla_v3 \
  --actor_num_gpus_per_node "[1]" \
  --vllm_num_engines 0 \
  --task_suite_name "libero_spatial" \
  --num_steps 64 \
  --local_rollout_batch_size 1
```

2) For a conservative smoke test, set `vllm_num_engines=0` so the training process uses a CPU-based or simplified inference path.

3) To run the full distributed setup, use the provided scripts under `vlarl/scripts/` (cluster-specific); ensure Ray and GPUs are accessible and `actor_num_gpus_per_node` matches available hardware.

Testing and CI recommendations
------------------------------

- Add unit tests for pure Python helpers: `calculate_runtime_args`, batching utilities, and prompt builders.
- Mock HF model loads in tests to verify `from_pretrained()` paths without GPUs.
- Add a GitHub action (or similar) that runs linting (`ruff`), type checks (`mypy`), and unit tests on PRs.

Maintenance and next steps
-------------------------

- Add `docs/quick_start_single_gpu.md` with a copy of the above quick-start plus a minimal `Args` YAML.
- Add a `configs/` directory containing example `Args` YAMLs for common runs (single-GPU dev, 4-GPU, cluster).
- Consider adding an automated validation step that runs a tiny end-to-end smoke test (no GPU) to catch import/runtime regressions.

Acknowledgements
----------------
This overview complements `docs/ppo_vllm_thread_ray_fsdp_vla_v3.md` which documents the main training script in detail. For task-specific details, consult `vlarl/experiments/robot/` and the dataset helpers under `vlarl/prismatic/preprocessing/`.

If you want, I can now:
- create `docs/quick_start_single_gpu.md` and/or example `configs/` entries,
- add unit-test scaffolding for helper functions, or
- update `README.md` to reference the new quick-start and testing instructions.

