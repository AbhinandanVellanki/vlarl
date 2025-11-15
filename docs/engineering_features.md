# Engineering features

Last updated: 2025-11-15

This document summarizes the repository's engineering choices and the concrete implementations / behaviors present in the codebase as of the latest edits. It is an engineer-facing reference describing why systems were chosen, where to inspect implementations, and practical notes for debugging and extension.

## High-level summary

This repo fine-tunes OpenVLA-style vision+language action-prediction models with PPO. It integrates three major systems for scale and throughput:

- FSDP (PyTorch FullyShardedDataParallel) for sharded training of large models.
- Ray for multi-process orchestration, GPU placement, and lightweight RPC actors.
- vLLM for high-throughput autoregressive inference used during rollouts and evaluation.

Additional engineering features implemented:
- PEFT / LoRA adapter support and merge tooling (adapter -> HF-style model).
- Adaptive curriculum starter (CurriculumManager Ray actor).
- Checkpointing improvements: EMA-based "best_return" saving and metadata, timestamped eval run directories to avoid overwrites.
- Critic warmup: `--value_init_steps` enables value-only warmup steps early in training.
- Clipped value loss (PPO-style) and standard GAE for advantages.

## Core distributed & inference systems

- FSDP (Fully Sharded Data Parallel)
  - Purpose: shard model params, optimizer state and gradients to enable training very large models across GPUs.
  - Where to inspect:
    - `ppo_vllm_thread_ray_fsdp_vla_v3.py` — FSDP initialization, model wrapping, and `save_model()` (gathering FSDP params for saving).
    - `utils/fsdp_utils.py` for helper utilities.
  - Notes:
    - Saves gather parameters before exporting models; `save_model()` handles PEFT vs full-model cases.
    - Offload and gradient-checkpointing flags are used in production scripts to reduce GPU memory.

- Ray scheduling / actors
  - Purpose: place trainers, rollout actors and LLM servers on specific GPUs; provide fault isolation and RPC coordination (metrics, curriculum).
  - Where to inspect:
    - `ppo_vllm_thread_ray_fsdp_vla_v3.py` — Ray actor classes and main orchestration.
    - `utils/ray_utils.py` and `ppo/utils/*` for helpers.
  - Notes:
    - New CurriculumManager is implemented as a Ray actor to centralize curriculum state.
    - Avoid blocking `ray.get()` in hot loops; the starter integration uses non-blocking report RPCs.

- vLLM inference engines (broadcast / serving)
  - Purpose: fast token generation for policy rollouts and evaluation.
  - Where to inspect:
    - `run_libero_eval_vllm.py`, `utils/vllm_utils2.py`.
  - Notes / pitfalls:
    - vLLM API/version mismatches can break constructor kwargs (e.g., `local_files_only` on some EngineArgs versions).
    - Recommended to load merged HF-style model folders (config + tokenizer + processor + weights) to avoid HF validation errors.

## Model tuning & efficiency

- PEFT / LoRA
  - Purpose: parameter-efficient fine-tuning (adapters) to drastically reduce compute/memory during RL tuning.
  - Where to inspect:
    - Model load/save paths in `ppo_vllm_thread_ray_fsdp_vla_v3.py` and `vla-scripts/merge.py`.
  - Behavior:
    - Adapter checkpoints are handled via `PeftModel.save_pretrained()` for adapters.
    - The repo provides `vla-scripts/merge.py` to merge adapters into HF-style model folders for inference with vLLM/transformers.

## Curriculum & dataset selection

- Static RLDS mixtures (baseline)
  - Curriculum is normally supplied by named RLDS mixtures (materialized by `openvla-oft` files).
  - Where to inspect:
    - `prismatic/vla/datasets/*` and `openvla-oft/.../prismatic/vla/datasets/rlds/oxe/mixtures.py`.

- Adaptive curriculum (starter implementation)
  - Implemented: `ppo/utils/curriculum.py` — `CurriculumManager` Ray actor.
    - Tracks per-task EMA success rates (configurable EMA alpha).
    - Computes sampling probabilities that favor tasks near 50% success (configurable formula via `tau` / temperature).
    - Exposes `get_batch()` to sample task ids and `report_results()` to update EMAs.
  - Wiring:
    - `ppo_vllm_thread_ray_fsdp_vla_v3.py` creates the CurriculumManager and passes it to trainer/actor processes.
    - Actors request task batches before rollouts and report per-episode successes back to the manager (non-blocking RPC).
  - Caveats:
    - VLAEnv must map/track task IDs so reported success corresponds to the sampled task id — the starter integration reuses existing env construction and keeps backward compatibility.

## Checkpointing and "best" model saving

- Periodic & final saves
  - Periodic checkpoints: `step_<N>/` created when `training_step % args.save_freq == 0`.
  - Final model saved to `args.exp_dir` at training end.

- Best-return checkpoint (implemented)
  - EMA of episodic return with alpha=0.99 tracked inside trainer actor (`self._ema_return`).
  - When EMA improves strictly, model is saved to: `args.exp_dir/best_return/`.
  - A metadata file is written: `best_return/metadata.json` (timestamp, ema_return, current_return, training_step, global_step, exp_id, vla_path).
  - Uses existing `save_model()` logic (handles FSDP consolidation and PEFT `save_pretrained()`).

## Critic (value) warmup & losses

- Critic warmup
  - CLI flag: `--value_init_steps N`.
  - Behavior: for training_step <= `value_init_steps` the training loop performs value-only updates (value optimizer & scheduler step) and skips policy updates. This is implemented inside the PPO update section so rollouts are still collected and GAE/returns computed, but only value regression updates run for those early steps.

- Value target & loss
  - Advantage computation:
    - Standard GAE: delta_t = r_t + γ * V_{t+1} * (1 - done_t) - V_t
    - A_t = delta_t + γ * λ * (1 - done_t) * A_{t+1}
    - Returns R_t = A_t + V_t
  - Value regression target: `mb_return` (GAE returns).
  - PPO-style clipped value loss used:
    - vpred = current value prediction
    - vpred_clipped = clamp(vpred, mb_values - cliprange_value, mb_values + cliprange_value)
    - vf_loss = 0.5 * mean( max( (vpred - mb_return)^2, (vpred_clipped - mb_return)^2 ) ) * vf_coef
  - Notes:
    - `mb_values` are rollout-time value estimates used for clipping.
    - Value optimizer and scheduler step during warmup as well as during full PPO updates.

- Policy loss (clipped surrogate)
  - ratio = exp(new_logprob - old_logprob)
  - L_clip = -mean( min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv) )
  - Total loss = L_clip + value_coef * L_v + entropy_coef * L_ent

## Merge process (PEFT adapter -> HF-style model)

- Purpose: create HF-style model folders so vLLM / transformers accept local model paths.
- Recommended merge behavior & post-merge checks:
  - Ensure merged folder contains: `config.json`, tokenizer files (tokenizer.json / tokenizer.model), processor files (image_processor_config.json / preprocessor), and a model weights file (`pytorch_model.bin` or `model.safetensors` / `pytorch_model.pt`).
  - Save merged weights as float32 for maximum loader compatibility (convert from bfloat16 if needed).
  - Copy tokenizer/processor from the base model if missing in the adapter folder.
  - Write `merge_metadata.json` listing base model, adapter path, and timestamp.

## Data / input pipeline

- On-disk TFRecord datasets
  - Example dataset root included: `data/modified_libero_rlds/libero_spatial_no_noops/1.0.0/` with TFRecord shards and `features.json` / `dataset_info.json`.
  - RLDS materializer and dataset wrappers convert TFRecords into prompt + action-label pairs used by the rollout process.
  - Where to inspect:
    - `prismatic/vla/datasets/*` and dataset-materialization utilities in the `openvla-oft` components.

## Observability & logging

- Eval logs
  - Each eval run writes to a timestamped directory under `experiments/logs/` (prevents overwriting past runs).
  - Text logs, rollouts, and mp4s (if produced) are written inside the run folder.

- Ray & GPUs
  - Inspect Ray dashboard (default local at `127.0.0.1:8265`) and `nvidia-smi` to map Python processes / Ray actors to GPU memory usage.

## Troubleshooting hotspots & common fixes

- vLLM EngineArgs mismatch
  - Symptom: `TypeError: EngineArgs.__init__() got an unexpected keyword argument 'local_files_only'`.
  - Fix: either upgrade vLLM/transformers or avoid passing unsupported kwargs and use merged HF-style model directories.

- OOMs / memory spikes
  - Typical cause: vLLM and environment/model running on the same GPU.
  - Fix: run vLLM on separate GPU(s) and set `--env_gpu_id` to a different device; tune `--gpu_memory_utilization`; use tensor-parallel size >1 if multiple GPUs available for vLLM.

- Merge issues (adapter-only folders)
  - If vLLM/transformers treat your path as a HF repo id or fail to find tokenizer/processor, ensure merged HF-style folder includes all required files and pass an absolute path.

## Quick developer tasks & next steps

- Inspect these primary files (fast path):
  - `ppo_vllm_thread_ray_fsdp_vla_v3.py` — training loop, FSDP, checkpointing and curriculum wiring.
  - `run_libero_eval_vllm.py` — vLLM eval flow and engine creation.
  - `utils/vllm_utils2.py` — vLLM engine creation and Ray wrappers.
  - `ppo/utils/curriculum.py` — starter CurriculumManager actor.
  - `vla-scripts/merge.py` — merge adapter tooling (consider the recommended fixes above).
  - `prismatic/vla/datasets/*` — dataset materialization and RLDS integration.

- Suggested small improvements to add:
  - Persist curriculum stats periodically (to disk / wandb) for monitoring.
  - Add `min_delta` / patience for best_return saving to avoid noisy saves.
  - Expand `merge.py` to auto-copy tokenizer/processor and ensure float32 weights; write `merge_metadata.json`.
  - Expose curriculum sampling formula parameters as CLI args.

(End)