# VLA-RL Codebase Overview

Location: `vlarl/` (primary files under `vlarl/`)

Last reviewed: 2025-11-04

This document summarizes the major parts of the codebase, how the VLA model is loaded, how the curriculum and datasets are configured, how pseudo (or learned) rewards are obtained, and how the model is trained end-to-end. It is written to complement `docs/ppo_vllm_thread_ray_fsdp_vla_v3.md` (which documents the single training file in detail) and to provide a codebase-level map you can use to navigate implementation files quickly.

Table of contents
-----------------
- Overview and purpose
- Key directories & scripts
- How the VLA model is loaded (detailed)
- Curriculum & dataset flow
- Pseudo-rewards and reward model integration
- Training algorithm & mechanics (PPO loop)
- Orchestration: Ray, vLLM and FSDP interactions
- Checkpoints, saving & pushing to HF hub
- Troubleshooting & common failure modes
- Quick dev steps & recommended tests

Overview and purpose
--------------------
VLA-RL fine-tunes vision+language action-prediction models (OpenVLA-style) with reinforcement learning. The training uses distributed actors (Ray) for GPU isolation, FSDP for efficient memory usage, and vLLM for fast model inference during rollouts. The code supports PEFT/LoRA for parameter-efficient tuning and optional reward models (PRMs) for dense reward signals.

Key directories & scripts
-------------------------
- `vlarl/ppo_vllm_thread_ray_fsdp_vla_v3.py` — primary training script. Contains the full orchestration: actor class (`PolicyTrainerRayProcess`), model initialization, FSDP wrapping, vLLM broadcasting, rollout logic, PPO updates, and saving.
- `vlarl/scripts/` — shell scripts to launch training and evaluation (e.g., `train_rl_vllm_ray_fsdp.sh`, `eval_vllm_ray.sh`). These are convenience entrypoints for common cluster setups.
- `vlarl/ppo/` — PPO-related helpers (models, utils, environments). Notable modules:
  - `ppo/envs/libero_env.py` — `VLAEnv` environment wrapper used for rollouts.
  - `ppo/models/` — `CriticVLA`, `CriticFilm` and other critic variants.
  - `ppo/utils/` — utilities for vLLM, ray scheduling, logging, fsdp helpers, and RL data transforms.
- `prismatic/` — OpenVLA/OpenVLA-related model and processing utilities (processors, tokenizer wrappers, prompt builders). Contains `PrismaticImageProcessor`, `PrismaticProcessor`, and `OpenVLAForActionPrediction` model classes.
- `experiments/robot/` — experiment-level utilities and reward helpers (e.g., `robot_utils.py`, `openvla_utils.py`). Exports `get_reward`, `process_with_padding_side`, and other helpers used by the training loop.
- `data/`, `checkpoints/`, `runs/` — common storage locations for dataset and artifacts (may be project-root relative; check `Args` defaults in the training file).

How the VLA model is loaded (detailed)
-------------------------------------
Where: `PolicyTrainerRayProcess.from_pretrained()` in `ppo_vllm_thread_ray_fsdp_vla_v3.py`

Steps (summary):
- Register HF classes for the custom OpenVLA types so `transformers` knows how to instantiate them:
  - `AutoConfig.register("openvla", OpenVLAConfig)`
  - `AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)`
  - `AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)`
  - `AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)`
- Load the base model with `AutoModelForVision2Seq.from_pretrained(args.vla_path, ...)` using `torch.bfloat16` to reduce memory footprint.
- If a LoRA/adapter checkpoint is provided, load as a PEFT model: `PeftModel.from_pretrained(...)` and attach dataset normalization stats if available.
- If training from scratch with LoRA: create a `LoraConfig` and call `get_peft_model(model, lora_config)`. Optionally call `prepare_model_for_kbit_training()` if quantization is enabled.
- Move model to `bfloat16` dtype and disable dropout. Optionally enable gradient checkpointing.
- Construct a device mesh (`init_device_mesh`) and select a FSDP wrap policy via `get_fsdp_wrap_policy_openvla()`. Wrap the model using `torch.distributed.fsdp.FullyShardedDataParallel` with `MixedPrecision` settings.
- Initialize optimizers and HF-style schedulers.

Why this matters:
- FSDP reduces per-GPU memory usage. PEFT/LoRA allows inexpensive adapter training. The code supports both full fine-tuning and PEFT flows, and integrates carefully with vLLM for inference consistency.

Curriculum & dataset flow
-------------------------
Where: dataset creation in `main()` and how `PolicyTrainerRayProcess.train()` slices `task_ids`.

Key components:
- `RLDSDataset` — dataset loader for RLDS/Open-X datasets. It reads episodes, tasks, and returns examples for training/eval.
- `RLDSBatchTransform` — transforms dataset examples into training batches (tokenization, image preprocessing, prompt construction). Uses `ActionTokenizer` and `PrismaticImageProcessor`.
- Prompt builders: `PurePromptBuilder` or `VicunaV15ChatPromptBuilder` — influence textual prompts and thus task framing.

Curriculum knobs (in `Args` dataclass):
- `task_suite_name` — pick a suite (libero_spatial/object/goal etc.).
- `task_ids` / `num_tasks_per_suite` — which tasks and how many to include.
- `num_trials_per_task` — number of rollouts per task.
- `local_rollout_batch_size` / `n_rollout_threads` — parallelism and allocation of tasks to actors (each actor slices `task_ids` by rank and `local_rollout_batch_size`).

Implementation notes:
- Curriculum is dataset-driven. You configure which tasks/examples the dataset exposes and the script shards tasks across Ray actors. To change curriculum, adjust the dataset source, `task_ids`, or the prompt-builder logic.

Pseudo-rewards & reward model integration
----------------------------------------
Where: `safe_get_reward()` in `PolicyTrainerRayProcess` and calls to `get_reward()` from `experiments.robot.robot_utils`.

Reward sources supported:
- Learned PRM (pretrained reward model): `prm_model_name_or_path` and optional `prm_checkpoint_path` can be used to load a reward model to generate dense rewards from observations + generated responses.
- Heuristics based on generation: `verify_reward_value`, `penalty_reward_value`, `non_stop_penalty` are heuristic fallbacks for rewarding responses (e.g., rewarding correct stop tokens or penalizing overlong outputs).
- Value model (critic): when `args.use_value_model` is set, `CriticVLA` or `CriticFilm` provides baseline/value estimates used to compute TD targets and advantages.

Robustness:
- `safe_get_reward(...)` wraps reward computation in try/except. If the GPU-side reward computation fails (CUDA assert), the function attempts to reproduce the call on CPU to get a Python traceback for debugging, logs the error, and returns a zero-tensor fallback so the Ray actor does not die.

Training algorithm & mechanics (PPO)
----------------------------------
Where: `PolicyTrainerRayProcess.train()`

High-level flow:
1. Rollout collection across `args.num_steps` per iteration: gather observations, broadcast to vLLM engines, call vLLM to sample actions/responses, step environment, collect rewards and dones.
2. Compute advantages (GAE) using `args.gamma` and `args.lam` and optionally whiten advantages.
3. Perform PPO minibatch updates (policy loss with clipping `args.cliprange`, optional value loss with `args.cliprange_value` and coefficient `args.vf_coef`).
4. Update schedulers and optimizers for policy and value separately.
5. Save checkpoints and metrics periodically; optionally push to HF hub.

Important training specifics:
- Uses `bfloat16` and FSDP mixed precision.
- Supports LoRA/PEFT training — only adapters can be trainable, lowering memory/compute usage.
- vLLM engines are kept synchronized with the current policy parameters by broadcasting parameters in batches (using `FSDP.summon_full_params` temporarily) to prevent memory spikes.
- Supports `async_mode` for asynchronous learning from slightly stale policies (Cleanba-style update mechanism).

Orchestration: Ray, vLLM and FSDP interactions
----------------------------------------------
Overview:
- Ray: runs `PolicyTrainerRayProcess` actors in a placement group to map each actor to a GPU. Ray's queues and remote actors are used to communicate between the actor that drives vLLM and the actor processes that run rollouts and training.
- vLLM: used for fast multi-modal inference (predict_action). One actor (main rank) runs a vLLM generation thread that waits for global prompts and returns sampled responses and logprobs via a Queue.
- FSDP: wraps the large model to shard parameters across ranks. For broadcasting to vLLM (which expects full parameters on a device), the code uses `FSDP.summon_full_params` to reconstruct parameters in manageable batches. The broadcast implementation handles PEFT naming conventions and only sends required parameters.

Key patterns:
- Device mesh: `init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])` for FSDP-aware topologies.
- Weight broadcast: `_broadcast_to_vllm()` serializes batches of parameters and calls `engine.init_process_group` or `engine.predict_action` with these parameters on vLLM side.
- Watchdog: `kill_ray_cluster_if_a_worker_dies()` polls remote object refs and triggers shutdown if a worker fails to avoid silent hangs.

Checkpoints, saving & pushing to HF hub
--------------------------------------
- Models (or LoRA adapters) are saved to `args.exp_dir` (derived from `run_dir`/`adapter_dir` in `main`).
- If `args.merge_model` is set, LoRA weights may be fused into the base model before saving.
- Optional HF hub push: `args.push_to_hub` controls whether the final checkpoint is pushed to Hugging Face. The code sets `hf_repo_id`, `hf_repo_revision`, etc., and will push when enabled.

Troubleshooting & common failure modes (summary)
-----------------------------------------------
- CUDA OOM during broadcasting: reduce broadcast batch_size, enable CPU offload (careful with grad accumulation), or reduce model size.
- Ray workers dying due to CUDA asserts on reward computation: inspect logs; `safe_get_reward` attempts CPU repro; use smaller batches to reproduce.
- vLLM deadlocks on NCCL sync: try setting `NCCL_CUMEM_ENABLE=0` (script already toggles this when `vllm_sync_backend == 'nccl'`). Ensure vLLM engines are started with correct group sizes.
- Mismatched GPUs vs `actor_num_gpus_per_node`: verify cluster GPU counts and placement group bundles.

Quick dev steps & recommended tests
----------------------------------
1. Import smoke test (no GPU): create a tiny Python script to import the training module, instantiate `Args()` with safe defaults, and run `calculate_runtime_args()` to confirm derived values.
2. Unit tests:
   - `test_helpers.py` for `calculate_runtime_args`, `add_padding`, `get_num_patches`.
   - Mock HF calls to test `from_pretrained()` init flow without GPUs (monkeypatch `AutoModelForVision2Seq.from_pretrained`).
3. Single-GPU smoke run:
   - Edit `scripts/train_rl_vllm_ray_fsdp.sh` to set `actor_num_gpus_per_node=[1]`, `vllm_num_engines=0` (disable vLLM to test pure training path) or `vllm_num_engines=1` for local inference.
4. Lint/type checks: run `ruff` or `mypy` per repo config files (`mypy.ini`, `pyproject.toml`).

Next steps (suggested)
----------------------
- Add a `docs/quick_start_single_gpu.md` with a minimal `Args` override and safe defaults for developers.
- Add unit tests for pure-Python helpers and a CI job that runs them.
- Extract a small YAML config example in `configs/` to drive the `Args` creation instead of direct edits to the Python file.

If you'd like, I can now:
- create the quick-start single-GPU README under `vlarl/docs/` (recommended),
- add the minimal unit test file for helpers, or
- edit the `README.md` to link to these new docs and add a short "Quick Start" entry.
