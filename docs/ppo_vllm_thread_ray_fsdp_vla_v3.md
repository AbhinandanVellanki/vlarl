## ppo_vllm_thread_ray_fsdp_vla_v3.py — Detailed documentation

Location: `vlarl/ppo_vllm_thread_ray_fsdp_vla_v3.py`

Last reviewed: 2025-11-04

Overview
--------
This module implements a distributed PPO-style training loop that fine-tunes an OpenVLA-style vision+language action-prediction model using:

- Ray actors for process isolation and multi-GPU orchestration
- PyTorch FSDP (Fully Sharded Data Parallel) for model sharding and memory efficiency
- vLLM engines for fast multi-modal inference (action / response generation)
- PEFT/LoRA support for low-cost adapter training
- A custom Libero environment wrapper (`VLAEnv`) and RL datasets utilities

The training loop is designed to run across multiple GPUs/nodes with explicit device mesh construction, broadcasting of model weights to vLLM inference engines, and a separate thread-based vLLM generation worker that returns actions to the training code.

Primary responsibilities
------------------------
- configure runtime arguments and derived runtime values (batch sizes, world size, etc.)
- create Ray placement groups and actors that run the policy trainer processes
- initialize models (policy and optional critic/value model) with FSDP wrapping
- broadcast model parameters to vLLM inference engines so vLLM can run deterministic inference consistent with the training policy
- run rollouts in parallel using the `VLAEnv` environment, collect rewards (via reward model / critic), and perform PPO updates
- provide robust behavior for runtime failures (e.g., safe reward evaluation and a watchdog that kills the Ray cluster if a worker dies)

Key components
--------------

1) Args dataclass
   - Definition: `Args` dataclass near the top of the file.
   - Purpose: centralizes nearly all configuration knobs used by the script — model paths, LoRA options, optimizer hyperparameters, PPO parameters, vLLM and Ray settings, FSDP options, and logging/wandb/hf push options.
   - Notes: Many fields have helpful docstrings in the source. `calculate_runtime_args()` computes derived values like `world_size`, `micro_batch_size`, `rollout_batch_size`, `mini_batch_size`, and `num_training_steps` (in-place mutation).

2) Utility helpers
   - `get_num_patches(image_size: int, patch_size: int) -> int` : returns number of image patches for a ViT-like grid.
   - `calculate_runtime_args(args: Args)` : populates dependent runtime fields and constructs `exp_id`.
   - `add_padding(sequences, pad_token_id, length)` : right-pads integer sequences to a target length.

3) RayProcess base class
   - Light wrapper used as base for remote Ray actors.
   - Sets up environment variables for distributed training (MASTER_ADDR/PORT, WORLD_SIZE, RANK, LOCAL_RANK) and seeds for reproducibility.
   - Helper utilities to get current node IP and a free port.

4) `PolicyTrainerRayProcess` (Ray actor)
   - Decorated with `@ray.remote(num_gpus=1)` and inherits `RayProcess`.
   - Key methods:
     - `from_pretrained(self, args)` — initializes model(s) and optimizers, builds the FSDP-wrapped policy model, optionally sets up a value/critic model, configures schedulers, and prepares model stats required by the VLA action normalization.
     - `safe_get_reward(self, model, query, pixel_value, pad_token_id)` — defensive wrapper that captures exceptions during reward computation. It attempts a CPU repro to fetch a readable traceback and returns a safe zero-tensor fallback to keep training alive.
     - `get_max_image_tokens(self)` — inspects the HF config for the vision backbone and computes the maximum number of image tokens (uses mapping for dinosiglip backbone types).
     - `train(self, processor, vllm_engines, metrics_queue)` — the central training loop (rollout collection, vLLM generation, batching across ranks, gathering, and PPO updates). This method contains many important sub-steps explained in the runtime flow section.
     - `save_model(self, model_to_save, processor, output_dir)` — helper to save model and processor artifacts (exists in file; used during training checkpoints).

5) ModelGroup
   - Utility to create a Ray placement group and spawn multiple `PolicyTrainerRayProcess` actors with proper GPU/CPU scheduling strategy. It computes `master_addr/port`, ranks, and arranges worker initialization calls.

6) `main(args: Args)` entrypoint
   - Sets up directories, logging/wandb/tensorboard, builds a `placement_group` for Ray, instantiates `ModelGroup`, runs actor initialization (`from_pretrained`) remotely, initializes vLLM engines, prepares datasets and RL-specific transforms, starts a watchdog thread that kills the Ray cluster if a worker dies, and orchestrates the high-level training loop (obtains rollouts and saves final models; optionally pushes to HF Hub).

Runtime flow (high-level)
-------------------------

1. Parse or construct `Args`.
2. Call `calculate_runtime_args(args)` to compute derived parameters used through training.
3. Create or ensure output directories and set random seeds.
4. Build a Ray `placement_group` according to `args.actor_num_gpus_per_node`.
5. Create a `ModelGroup`, which spawns Ray actors of `PolicyTrainerRayProcess` type. Each actor calls `from_pretrained(args)` to:
   - Register OpenVLA HF model classes and processors
   - Optionally adjust environment variables for vLLM sync
   - Initialize PyTorch process group if not already
   - Build device mesh for FSDP
   - Load the base `AutoModelForVision2Seq` and apply LoRA/PEFT adapters if requested
   - Wrap the policy model with FSDP using `get_fsdp_wrap_policy_openvla` and `MixedPrecision` policy
   - Optionally build and FSDP-wrap value/critic model
   - Initialize optimizers and schedulers for policy/value models

6. Once all actors are ready, `main` creates vLLM engines via `create_vllm_engines(...)`. vLLM engines are used for batched inference during rollouts.
7. The training process begins: actors coordinate to gather local observations, perform distributed all_gather to form global inputs, send these to a vLLM generation thread (running only on the main actor rank) through thread-safe Queue objects, and receive generated actions/responses back.
8. The code contains a `_broadcast_to_vllm()` function that serializes and broadcasts the policy model's parameters to all vLLM engines in batches to reduce memory pressure (handles PEFT/LoRA differences, uses FSDP.summon_full_params to temporarily reconstruct full params for broadcasting).
9. Rollouts are collected for `args.num_steps` per iteration. Rewards are computed via `get_reward` or the value model. Collected data are used to compute advantages and update the policy with PPO-style clipped objective; optionally value updates are run when a critic is present.
10. Periodically, models are saved, and metrics are pushed to TensorBoard/WandB or optionally the Hugging Face Hub.

Important notes & design choices
--------------------------------
- FSDP is used to reduce GPU memory usage; however, the code uses `FSDP.summon_full_params` during parameter broadcast, which temporarily increases memory usage — the code splits parameter broadcasting into batches to reduce peak memory.
- There's careful handling for PEFT/LoRA models: when a model is a PEFT model, only the required set of parameters are included in the broadcast and PEFT-specific prefixes are handled.
- vLLM prefix caching: when enabled (`args.enable_prefix_caching`), the code clears caches on engines before weight broadcasts.
- The code tries to be defensive: `safe_get_reward` attempts CPU repro on exceptions; a watchdog thread kills the Ray cluster when any worker dies so the job won't hang silently.

Configuration and major arguments
--------------------------------

The `Args` dataclass contains dozens of options; here are the most commonly adjusted ones:

- Model & LoRA
  - `vla_path` — HF repo/model id for the OpenVLA policy
  - `use_lora`, `lora_rank`, `lora_dropout` — LoRA/PEFT training settings
  - `load_adapter_checkpoint` — path to adapter checkpoint to load

- Training & PPO
  - `per_device_train_batch_size`, `local_mini_batch_size`, `local_rollout_batch_size` — micro- and mini-batch sizes
  - `num_steps` — environment steps per rollout
  - `total_episodes`, `num_training_steps`, `num_epochs`, `gradient_accumulation_steps`
  - `learning_rate`, `value_learning_rate`, `vf_coef`, `cliprange`, `gamma`, `lam`

- Ray & vLLM
  - `actor_num_gpus_per_node` — list describing how many GPUs per actor on each node
  - `vllm_num_engines`, `vllm_tensor_parallel_size`, `vllm_enforce_eager`
  - `vllm_sync_backend` — backend string used for synchronizing weights between FSDP and vLLM

- FSDP
  - `sharding_strategy` — 'full-shard' or similar
  - `offload` — whether to offload parameters to CPU (note: disabled for critic due to grad acc issues)

Example: minimal run (single-node, single-GPU)
-------------------------------------------

The recommended way is to run this script through Python in an environment with the project's dependencies installed (Torch, Ray, vLLM, transformers, accelerate, peft, etc.). The example below runs with tiny safe settings and assumes Ray is already started (or local ray will be used):

```bash
# from the repo root
python vlarl/ppo_vllm_thread_ray_fsdp_vla_v3.py
```

To pass custom args you can add an argparse wrapper or set environment variables as required; the production script expects a populated `Args` dataclass (the file uses `draccus.wrap()` decorator in `main`). A common pattern is to modify a small starter config to set `vla_path`, `data_root_dir`, `actor_num_gpus_per_node=[1]` and `vllm_num_engines=1`.

Troubleshooting & common failure modes
-------------------------------------

- Ray worker dies / cluster hangs
  - Symptom: some workers crash with CUDA errors or the job hangs. The file includes `kill_ray_cluster_if_a_worker_dies` which monitors remote object refs and triggers a shutdown. Check logs for the worker exception and run locally with fewer processes to reproduce.

- CUDA OOM or FSDP memory spikes
  - Cause: `FSDP.summon_full_params` temporarily reconstructs full parameters; if broadcast batches are too large this can OOM.
  - Mitigation: reduce broadcast `batch_size`, disable unnecessary model gathering, use mixed precision (bfloat16) and verify `device_mesh` is optimal. Consider enabling CPU offload for non-critical parts (but note comment about critic and grad accumulation).

- vLLM synchronization problems
  - Symptom: deadlocks during NCCL sync or vLLM engine init.
  - Mitigation: the code toggles `NCCL_CUMEM_ENABLE=0` when using NCCl backend. Also ensure `vllm_sync_backend` matches expected backend and engines are started with correct `world_size` and `rank_offset`.

- Reward model / critic crashes
  - Symptom: CUDA asserts from the reward model cause Ray worker to die.
  - Mitigation: `safe_get_reward` attempts a CPU repro to capture a Python traceback and returns zeros to avoid killing the worker; inspect logs to debug the failing case and re-run with smaller batch size to reproduce.

Developer contract (short)
-------------------------

- Inputs:
  - `Args` dataclass + datasets under `args.data_root_dir`
  - Ray + vLLM services available (or local single-node fallback for development)

- Outputs:
  - Training logs (TensorBoard, optional WandB)
  - Checkpoints saved in `args.exp_dir` (fused LoRA weights if `merge_model` used)
  - Optionally pushed HF repo when `push_to_hub` is True

- Success criteria:
  - Script runs without fatal worker crashes for at least a few iterations
  - Model checkpoints and training metrics appear in `args.exp_dir` and TensorBoard

Edge cases to consider
---------------------

- Running with mismatched `actor_num_gpus_per_node` vs actual available GPUs — verify GPU inventories per node before launching.
- Running with `enable_gradient_checkpointing` may change memory and performance characteristics and interacts with FSDP—test carefully.
- PEFT/LoRA + quantization paths (`prepare_model_for_kbit_training`) introduce different memory and parameter handling — debug broadcasts carefully.

Suggested tests and quality gates
--------------------------------

Build (quick import test)
 - Create a small Python script that imports the module and instantiates `Args()` and runs `calculate_runtime_args` with a tiny setting; assert no ImportError.

Lint/typecheck
 - Run flake8 / ruff and mypy (project already contains `mypy.ini` and pyproject config). Fix any new warnings.

Unit tests (recommended)
 - Test `calculate_runtime_args()` for expected derived values given small world sizes.
 - Test `add_padding()` and `get_num_patches()` for simple inputs.
 - Mock a small `PolicyTrainerRayProcess.from_pretrained` path with a tiny toy model (or monkeypatch HF calls) to validate FSDP wrap call path without launching full Ray/GPU.

Minimal script to validate imports (example)
```python
from vlarl import ppo_vllm_thread_ray_fsdp_vla_v3 as trainer
args = trainer.Args()
args.per_device_train_batch_size = 1
args.local_mini_batch_size = 1
args.actor_num_gpus_per_node = [1]
trainer.calculate_runtime_args(args)
print('derived:', args.world_size, args.rollout_batch_size)
```

Appendix: recommended environment & dependencies
-----------------------------------------------

At a minimum you'll need:

- Python 3.10+ (project pyproject suggests a modern Python)
- torch with CUDA and FSDP support (matching your GPU and driver)
- ray (compatible with your Ray cluster and installed extras for CUDA scheduling)
- transformers, accelerate, peft, vllm, bitsandbytes (if using quantization paths)
- torchvision / PIL for image handling

Check `vlarl/requirements-min.txt` and `vlarl/requirements.txt` (or top-level requirements) for precise pins.

Final notes
-----------

This file is a complex orchestration binding Ray, vLLM, and FSDP for RL fine-tuning of a big vision+language model. Approach changes conservatively: small config changes, single-GPU or CPU mock-mode for reproducing issues, and add telemetry (more logging) when debugging distributed problems.

If you want, I can also:
- generate a concise quick-start guide with exact commands and environment setup for a single-GPU dev run
- add a small unit test file to validate basic helpers
- extract a YAML example configuration with recommended safe defaults for experimenting

---
Documentation generated by an automated code-assistant. If you want edits (shorter, more technical, or with examples tuned to your cluster), tell me which sections to expand or trim.
