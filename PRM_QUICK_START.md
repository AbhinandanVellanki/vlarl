# PRM Integration - Quick Start Guide

## What Changed?

The Process Reward Model (PRM) has been integrated into the training pipeline to provide **dense rewards** at each step instead of only sparse terminal rewards.

## How to Use

### Option 1: Use PRM with Qwen2-VL (Recommended)

In `scripts/train_rl_vllm_ray_fsdp_mini.sh`, the following arguments are now active:

```bash
--process_reward_model True \
--prm_model_name_or_path "MODEL/Qwen2-VL-2B-Instruct" \
--prm_reward_weight 0.1 \
```

**Make sure you have the model downloaded:**
```bash
# The model should be at MODEL/Qwen2-VL-2B-Instruct/
# If not, download it from HuggingFace or adjust the path
```

### Option 2: Test Without PRM First

To disable PRM (original behavior):
```bash
--process_reward_model False \
```

### Option 3: Use DummyRM (For Testing)

To test integration without loading the actual PRM model:
```bash
--process_reward_model True \
# Remove or comment out the --prm_model_name_or_path line
```

## New CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--process_reward_model` | bool | False | Enable/disable PRM |
| `--prm_model_name_or_path` | str | "MODEL/Qwen2-VL-2B-Instruct" | Path to PRM model |
| `--prm_checkpoint_path` | str | None | Optional fine-tuned checkpoint |
| `--prm_reward_weight` | float | 0.1 | Weight for combining PRM with env rewards |

## What to Expect

### Training Logs

You should see these new log messages:

```
[PRM] Initializing Process Reward Model
[PRM] Before PRM init - GPU Memory: X.XX GB
[PRM] Loaded QwenProcessRM from MODEL/Qwen2-VL-2B-Instruct
[PRM] After PRM init - GPU Memory: X.XX GB
...
[PRM] Computed rewards: [0.0, 1.0, 0.0, ...]
[PRM] Added weighted PRM rewards (weight=0.1): [0.0, 0.1, 0.0, ...]
```

### Performance Impact

- **Memory**: +2-4GB GPU memory for Qwen2-VL-2B
- **Speed**: ~50-200ms additional per rollout step
- **Training**: Potentially faster convergence due to dense rewards

### Videos

When `--save_video True`, the rollout videos will now show:
- **prm_rewards**: The PRM prediction at each step (displayed on the frame)

## Troubleshooting

### "Model not found" error
```bash
# Make sure the model path is correct
ls MODEL/Qwen2-VL-2B-Instruct/
# Should contain config.json, model files, etc.
```

### Out of GPU memory
```bash
# Option 1: Use smaller batch size
--local_rollout_forward_batch_size 2  # instead of 4

# Option 2: Test with DummyRM first
--process_reward_model True
# (remove --prm_model_name_or_path line)

# Option 3: Disable PRM temporarily
--process_reward_model False
```

### PRM predictions always 0 or 1
This is expected! The PRM is a binary classifier:
- **0**: Task not yet completed
- **1**: Task appears completed

The `prm_reward_weight=0.1` scales this to avoid overwhelming the sparse terminal reward.

## Tuning PRM Weight

The `prm_reward_weight` controls how much the PRM influences training:

| Weight | Behavior | Use Case |
|--------|----------|----------|
| 0.0 | PRM disabled | Baseline (sparse rewards only) |
| 0.01 | Very weak signal | Conservative, stable training |
| 0.1 | Moderate signal | **Default, good starting point** |
| 0.5 | Strong signal | If PRM is highly accurate |
| 1.0 | Equal to terminal | PRM dominates training |

**Recommendation**: Start with 0.1, adjust based on training stability and convergence speed.

## Next Steps

1. **Run training** with default PRM settings
2. **Monitor logs** for PRM initialization and reward computation
3. **Check wandb/tensorboard** for reward statistics
4. **Adjust `prm_reward_weight`** if needed based on performance
5. **Compare** with/without PRM runs to evaluate benefit

## Advanced: Using a Fine-tuned PRM

If you have a task-specific fine-tuned PRM checkpoint:

```bash
--process_reward_model True \
--prm_model_name_or_path "MODEL/Qwen2-VL-2B-Instruct" \
--prm_checkpoint_path "path/to/your/finetuned/prm/adapter" \
--prm_reward_weight 0.1 \
```

The checkpoint should be a LoRA adapter saved in HuggingFace format.

## Code Locations

If you need to modify the PRM implementation:

- **PRM Models**: `ppo/models/prm.py`
- **PRM Integration**: `ppo_vllm_thread_ray_fsdp_vla_v3.py` (lines 744-772, 1301-1408)
- **Training Script**: `scripts/train_rl_vllm_ray_fsdp_mini.sh`

## Questions?

Refer to `PRM_INTEGRATION_SUMMARY.md` for detailed technical documentation.


