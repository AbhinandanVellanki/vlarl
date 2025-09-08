"""
Usage:
    pytest tests/test_ppo.py -s
    pytest tests/test_ppo.py::test_ppo_single_gpu -s
    pytest tests/test_ppo.py::test_ppo_multi_gpu -s
"""

import subprocess
import os

def test_ppo_single_gpu():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env.update({
        "MESA_GL_VERSION_OVERRIDE": "4.1",
        "PYOPENGL_PLATFORM": "egl", 
        "MUJOCO_GL": "egl",
        "CUDA_VISIBLE_DEVICES": "0,1" # single-gpu training
    })
    cmd = [
        "/opt/conda/envs/vlarl/bin/python",
        "ppo_vllm_ray_fsdp_v3.py",
        "--pretrained_checkpoint", "MODEL/openvla-7b-finetuned-libero-goal",
        "--data_root_dir", "./data/modified_libero_rlds",
        "--dataset_name", "libero_goal_no_noops",
        "--task_suite_name", "libero_goal",
        "--num_trials_per_task", "1",  # Modified for testing
        "--task_ids", "[0]",  # Modified for testing
        "--run_root_dir", "checkpoints/debug/root",
        "--adapter_tmp_dir", "checkpoints/debug/adapter",
        "--per_device_train_batch_size", "1",
        "--local_mini_batch_size", "1",
        "--local_rollout_batch_size", "1",
        "--local_rollout_forward_batch_size", "1",
        "--actor_num_gpus_per_node", "[1]",
        "--temperature", "1.7",
        "--num_epochs", "1",
        "--value_init_steps", "0",  # Modified for testing
        "--learning_rate", "2e-5",
        "--value_learning_rate", "5e-5",
        "--policy_max_grad_norm", "1.0",
        "--value_max_grad_norm", "1.0",
        "--cliprange_high", "0.4",
        "--cliprange_low", "0.2",
        "--gamma", "1.0",
        "--num_steps", "2",  # Modified for testing
        "--max_env_length", "2",  # Modified for testing
        "--total_episodes", "100000",
        "--vllm_tensor_parallel_size", "1",
        "--vllm_enforce_eager", "True",
        "--enable_prefix_caching", "False",
        "--gpu_memory_utilization", "0.9",
        "--use_lora", "True",
        "--enable_gradient_checkpointing", "False",
        "--sharding_strategy", "shard-grad-op",
        "--offload", "False",
        "--use_value_model", "True",
        "--value_model_type", "vla",
        "--value_use_lora", "False",
        "--clip_vloss", "False",
        "--norm_adv", "False",
        "--use_curriculum", "True",
        "--curriculum_temp", "1.0",
        # "--success_history_window", "20",
        "--save_freq", "1",  # Save more frequently for testing
        "--init_eval", "True",
        "--eval_freq", "1",  # Evaluate more frequently for testing
        "--save_video", "True",
        # "--use_wandb", "True",
        "--use_wandb", "False",
        "--wandb_offline", "False",
        "--wandb_project", "openvla",
        "--wandb_entity", "openvla_cvpr",
        "--debug", "False"
    ]
    subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        check=True,
    )

def test_ppo_multi_gpu():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env.update({
        "MESA_GL_VERSION_OVERRIDE": "4.1",
        "PYOPENGL_PLATFORM": "egl", 
        "MUJOCO_GL": "egl",
        # "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", # multi-gpu training
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    })
    cmd = [
        "/opt/conda/envs/vlarl/bin/python",
        "ppo_vllm_ray_fsdp_v3.py",
        "--pretrained_checkpoint", "MODEL/openvla-7b-finetuned-libero-goal",
        "--data_root_dir", "./data/modified_libero_rlds",
        "--dataset_name", "libero_goal_no_noops",
        "--task_suite_name", "libero_goal",
        "--num_trials_per_task", "1",  # Modified for testing
        "--task_ids", "[0,1,2,3,4,5,6,7,8,9]",  # Modified for testing
        "--run_root_dir", "checkpoints/debug/root",
        "--adapter_tmp_dir", "checkpoints/debug/adapter",
        "--per_device_train_batch_size", "4",
        "--local_mini_batch_size", "4",
        "--local_rollout_batch_size", "10",
        "--local_rollout_forward_batch_size", "10",
        # "--actor_num_gpus_per_node", "[7]",
        "--actor_num_gpus_per_node", "[3]",
        "--temperature", "1.7",
        "--num_epochs", "1",
        "--value_init_steps", "0",  # Modified for testing
        "--learning_rate", "2e-5",
        "--value_learning_rate", "5e-5",
        "--policy_max_grad_norm", "1.0",
        "--value_max_grad_norm", "1.0",
        "--cliprange_high", "0.4",
        "--cliprange_low", "0.2",
        "--gamma", "1.0",
        "--num_steps", "4",  # Modified for testing
        "--max_env_length", "4",  # Modified for testing
        "--total_episodes", "100000",
        "--vllm_tensor_parallel_size", "1",
        "--vllm_enforce_eager", "True",
        "--enable_prefix_caching", "False",
        "--gpu_memory_utilization", "0.9",
        "--use_lora", "True",
        "--enable_gradient_checkpointing", "False",
        "--sharding_strategy", "shard-grad-op",
        "--offload", "False",
        "--use_value_model", "True",
        "--value_model_type", "vla",
        "--value_use_lora", "False",
        "--clip_vloss", "False",
        "--norm_adv", "False",
        "--use_curriculum", "True",
        "--curriculum_temp", "1.0",
        # "--success_history_window", "20",
        "--save_freq", "1",  # Save more frequently for testing
        "--init_eval", "True",
        "--eval_freq", "1",  # Evaluate more frequently for testing
        "--save_video", "True",
        # "--use_wandb", "True",
        "--use_wandb", "False",
        "--wandb_offline", "False",
        "--wandb_project", "openvla",
        "--wandb_entity", "openvla_cvpr",
        "--debug", "False"
    ]
    subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        check=True,
    )
