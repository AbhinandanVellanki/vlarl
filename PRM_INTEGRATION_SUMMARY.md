# PRM (Process Reward Model) Integration Summary

## Overview
Successfully integrated the Process Reward Model (PRM) into the PPO training pipeline for dense reward computation during robot task execution.

## Changes Made

### 1. **Main Training Script** (`ppo_vllm_thread_ray_fsdp_vla_v3.py`)

#### A. Added PRM Arguments (Lines 342-345)
```python
process_reward_model: bool = False
"""the process reward model (prm), for dense reward"""
prm_reward_weight: float = 0.1
"""weight for PRM rewards when combining with environment rewards"""
```

#### B. PRM Initialization (Lines 744-772)
Added PRM initialization in `PolicyTrainerRayProcess.from_pretrained()`:
- Initializes either `QwenProcessRM` or `DummyRM` based on configuration
- Handles `prm_model_name_or_path` and `prm_checkpoint_path` arguments
- Manages `use_vllm` flag to avoid conflicts with policy model
- Logs GPU memory usage before/after initialization

#### C. PRM Reward Computation in Rollout (Lines 1301-1353)
During the rollout phase, for each mini-batch:
- Converts pixel values from [B, C, H, W] to [B, H, W, C] in [0, 255] range
- Handles stacked channels (6-channel dinosiglip → 3-channel RGB)
- Decodes query tokens to text prompts
- Calls `self.reward_model.get_reward(text_list, image_list)`
- Stores PRM scores in `self.prm_scores_buffer` for later use
- Includes error handling with fallback to zeros

#### D. PRM Rewards Integration (Lines 1363-1408)
- Passes PRM rewards to environment for video logging
- Adds weighted PRM rewards to environment rewards
- Uses configurable `prm_reward_weight` (default: 0.1) to balance dense vs sparse rewards
- Logs combined rewards for debugging

### 2. **Training Script** (`scripts/train_rl_vllm_ray_fsdp_mini.sh`)

#### Added PRM Configuration (Lines 71-73)
```bash
--process_reward_model True \
--prm_model_name_or_path "MODEL/Qwen2-VL-2B-Instruct" \
--prm_reward_weight 0.1 \
```

## How PRM Works

### Purpose
The PRM provides **dense, step-by-step rewards** instead of only sparse terminal rewards:
- **Without PRM**: Agent gets reward (1.0 or 0.0) only when task succeeds/fails
- **With PRM**: Agent gets intermediate rewards at each timestep based on task progress

### Implementation
The `QwenProcessRM` uses a vision-language model (Qwen2-VL) to:
1. Take the current observation (image + task description)
2. Ask: "The task is {task_label}, is it completed?"
3. Return a binary score (0 or 1) indicating task progress

### Reward Combination
Final reward at each step:
```python
final_reward = env_reward + prm_weight × prm_reward
```
- `env_reward`: Sparse terminal reward from environment (0 or 1)
- `prm_reward`: Dense PRM prediction (0 or 1)
- `prm_weight`: Scaling factor (default: 0.1)

## Configuration Options

### Required CLI Arguments
```bash
--process_reward_model True           # Enable PRM
--prm_model_name_or_path "MODEL/..."  # Path to PRM base model
```

### Optional CLI Arguments
```bash
--prm_checkpoint_path "path/to/checkpoint"  # Fine-tuned PRM checkpoint (optional)
--prm_reward_weight 0.1                     # Weight for PRM rewards (default: 0.1)
```

### Using Different PRM Implementations

#### 1. **QwenProcessRM** (Default - Vision-Language Model)
```bash
--prm_model_name_or_path "MODEL/Qwen2-VL-2B-Instruct"
```
- Uses Qwen2-VL for visual task completion prediction
- Requires GPU memory (~2GB for 2B model)
- Can use vLLM for faster inference (set `use_vllm_prm=True` in code)

#### 2. **DummyRM** (For Testing)
```bash
--process_reward_model True
# Don't specify --prm_model_name_or_path
```
- Returns zeros (no actual reward)
- Useful for testing integration without GPU overhead

#### 3. **Custom PRM Checkpoint**
```bash
--prm_model_name_or_path "MODEL/Qwen2-VL-2B-Instruct"
--prm_checkpoint_path "path/to/your/finetuned/prm"
```
- Loads base model + LoRA adapter
- Useful if you have a task-specific fine-tuned PRM

## Memory Considerations

### GPU Memory Usage
- **Policy Model (FSDP)**: ~20-30GB for 7B model
- **Value Model (optional)**: ~20-30GB for 7B model
- **PRM (Qwen2-VL-2B)**: ~2-4GB
- **vLLM Engine**: ~10-15GB

### Recommendations
1. **For limited GPU memory**: Start with `DummyRM` to verify integration
2. **For moderate memory**: Use Qwen2-VL-2B on CPU or separate GPU
3. **For ample memory**: Use Qwen2-VL with vLLM on separate GPU

## Performance Impact

### Latency Added per Rollout Step
- **PRM Forward Pass**: ~50-200ms per batch (depending on batch size and model)
- **Total Rollout**: 128 steps × rollout_batch_size → significant overhead

### Optimization Strategies
1. Set `use_vllm_prm=True` in code for faster PRM inference
2. Reduce `local_rollout_forward_batch_size` if PRM is bottleneck
3. Use smaller PRM model (e.g., 2B instead of 7B)

## Testing the Integration

### 1. Test with DummyRM (No GPU overhead)
```bash
# In the shell script, remove --prm_model_name_or_path line
--process_reward_model True \
# --prm_model_name_or_path "..." <-- comment this out
```

### 2. Test with QwenProcessRM
```bash
--process_reward_model True \
--prm_model_name_or_path "MODEL/Qwen2-VL-2B-Instruct" \
--prm_reward_weight 0.1 \
```

### 3. Monitor Logs
Look for these log messages:
```
[PRM] Initializing Process Reward Model
[PRM] Loaded QwenProcessRM from MODEL/...
[PRM] Computed rewards: [0.0, 1.0, ...]
[PRM] Added weighted PRM rewards (weight=0.1): [...]
```

## Troubleshooting

### Issue: Out of GPU Memory
**Solution**: 
1. Use DummyRM initially
2. Reduce model sizes or enable offloading
3. Use separate GPU for PRM (modify code to specify device)

### Issue: PRM inference too slow
**Solution**:
1. Set `use_vllm_prm=True` in the PRM initialization section
2. Reduce batch size
3. Use smaller PRM model

### Issue: PRM rewards not showing up
**Solution**:
1. Check logs for "[PRM] Computed rewards"
2. Verify `--process_reward_model True` is set
3. Check `prm_scores_buffer` is being populated

### Issue: Training unstable with PRM
**Solution**:
1. Reduce `--prm_reward_weight` (try 0.01 or 0.05)
2. PRM might be giving too strong signals vs sparse rewards
3. Monitor reward distributions in wandb/tensorboard

## Future Improvements

1. **Batch PRM across all workers**: Currently each GPU computes PRM independently
2. **Cache PRM predictions**: Avoid recomputing for similar states
3. **Learned PRM weight**: Make `prm_reward_weight` adaptive during training
4. **Multi-step PRM**: Predict success probability N steps ahead
5. **PRM as auxiliary loss**: Train policy to match PRM predictions

## References

- **PRM Implementation**: `ppo/models/prm.py`
- **Main Training Loop**: `ppo_vllm_thread_ray_fsdp_vla_v3.py`
- **Environment**: `ppo/envs/libero_env.py`
- **Training Script**: `scripts/train_rl_vllm_ray_fsdp_mini.sh`


