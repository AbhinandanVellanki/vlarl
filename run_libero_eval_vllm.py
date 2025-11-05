"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Iterator, List, Literal, Optional, Tuple, Union
import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from termcolor import cprint, colored
import wandb
import pprint

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder, QwenPromptBuilder
from envs.libero_env import LiberoVecEnv
from utils.vllm_utils2 import create_vllm_engines
from vllm import SamplingParams
import time
from datetime import datetime
import ray
import threading

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
current_path = os.getcwd()

from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    set_seed_everywhere,
)
from utils.util import TimingManager


@dataclass
class GenerateConfig:
    # fmt: off
    vla_path: str = "openvla-7b"       # OpenVLA model path
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    # Environment Parameters
    task_suite_name: str = "libero_spatial"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim"""
    num_tasks_per_suite: int = 10
    """Number of tasks per suite"""
    num_trials_per_task: int = 50
    """Number of rollouts per task"""
    n_rollout_threads: int = 10
    """Number of parallel vec environments"""
    task_ids: Optional[List[int]] = None
    """Task ids to run"""
    max_env_length: int = 0
    """0 for default libero length"""
    env_gpu_id: int = 0
    """GPU id for the vectorized environments"""
    context_length: int = 64
    """Length of the query"""
    save_video: bool = False
    """Whether to save videos"""
    penalty_reward_value: float = 0.0
    """Penalty reward value"""
    non_stop_penalty: bool = False
    """Whether to penalize responses that do not contain `stop_token_id`"""
    verify_reward_value: float = 1.0

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    return_thought: bool = False                     # whether return decoded thought chain
    verbose: bool = False                            # Verbose mode for debugging
    subgoal_steps: int = 5                           # Number of steps to take for each subgoal (keep thought chain consistent)
    # fmt: on

    load_adapter_checkpoint: Optional[str] = None    # Path to adapter checkpoint to load
    save_images: bool = False                        # Whether to save images besides videos
    
    # generation config
    response_length: int = 8
    """the length of the response"""
    stop_token_id: Optional[int] = None
    """the truncation token id"""
    min_response_length: int = 0
    """stop only after this many tokens"""
    temperature: float = 1.0
    """the sampling temperature"""
    verify_reward_value: float = 10.0
    """the reward value for responses that do not contain `stop_token_id`"""
    penalty_reward_value: float = -1.0
    """the reward value for responses that do not contain `stop_token_id`"""
    non_stop_penalty: bool = False
    """whether to penalize responses that do not contain `stop_token_id`"""
    number_envs_per_task: int = 1
    """the number of samples to generate per prompt"""

    # ray
    vllm_num_engines: int = 1
    """number of vLLM Engines, set to 0 to disable vLLM"""
    vllm_tensor_parallel_size: int = 1
    """tensor parallel size of vLLM Engine for multi-GPU inference"""
    vllm_enforce_eager: bool = False
    """whether to enforce eager mode for vLLM -- slow inference but needed for multi-node"""
    vllm_sync_backend: str = "nccl"
    """DeepSpeed -> vLLM weight sync backend"""
    enable_prefix_caching: bool = False
    """whether to enable prefix caching"""
    gpu_memory_utilization: float = 0.9
    """pre-allocated GPU memory utilization for vLLM"""


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # NOTE: this may affect the performance.
    # set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    max_len = 256 + cfg.context_length + cfg.response_length
    vllm_engines = create_vllm_engines(
        num_engines=cfg.vllm_num_engines,
        tensor_parallel_size=cfg.vllm_tensor_parallel_size,
        enforce_eager=cfg.vllm_enforce_eager,
        pretrain=cfg.pretrained_checkpoint,
        trust_remote_code=True,
        revision=None,
        seed=cfg.seed,
        enable_prefix_caching=cfg.enable_prefix_caching,
        max_model_len=max_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        # enable_lora=True if cfg.load_adapter_checkpoint is not None else False,
        # norm_stats=norm_stats,
    )
    generation_config = SamplingParams(
        temperature=cfg.temperature,    # for greedy sampling
        max_tokens=cfg.response_length,
        include_stop_str_in_output=False,
        detokenize=False,
        n=1,
        seed=cfg.seed,
        logprobs=1,
    )
    print(f"generation_config: {generation_config}")

    # (Optional) Load processor
    # processor = get_processor(cfg)
    processor = None

    def vllm_generate(
            generation_config: SamplingParams,
            response_ids_Q: Queue,
            param_prompt_Q: Queue,
        ):
            llm = vllm_engines[0]
            while True:
                g_queries_list = param_prompt_Q.get()
                if g_queries_list is None:
                    break

                pixel_values = g_queries_list["pixel_values"]
                if processor is None:   # use this to avoid loading additional processor
                    prompts = g_queries_list["prompts"]
                    prompts = ["<PAD>" + prompt + "▁" for prompt in prompts]
                    # print(f"🔥🔥🔥 prompts: {prompts}")
                    llm_inputs = [
                        {
                            "prompt": prompt,
                            "multi_modal_data": {"image": pixel_value},
                        } for prompt, pixel_value in zip(prompts, pixel_values)
                    ]
                else:
                    prompts = g_queries_list["prompts"]
                    prompts = ["<PAD>" + prompt + "▁" for prompt in prompts]
                    prompt_token_ids = processor.tokenizer(prompts, return_tensors="pt").input_ids
                    print(f"🔥🔥🔥 prompt_token_ids: {prompt_token_ids}")
                    llm_inputs = [
                        {
                            "prompt_token_ids": prompt_token_id,
                            "multi_modal_data": {"image": pixel_value},
                        } for prompt_token_id, pixel_value in zip(prompt_token_ids, pixel_values)
                    ]

                generation_start_time = time.time()
                actions, response_ids, response_logprobs = ray.get(
                    llm.predict_action.remote(
                        llm_inputs,
                        sampling_params=generation_config, 
                        use_tqdm=False,
                        unnorm_key=cfg.unnorm_key,
                        # lora_request=LoRARequest("vla_adapter", 1, cfg.load_adapter_checkpoint) if cfg.load_adapter_checkpoint is not None else None
                    )
                )
                # print(f"{response_logprobs=}")
                print(
                    f"🔥🔥🔥 Action generation time: {time.time() - generation_start_time:.2f} s, "
                    f"with bs: {len(llm_inputs)}"
                )
                response_ids_Q.put(actions)

    response_ids_Q = Queue(maxsize=1)
    prompt_ids_Q = Queue(maxsize=1)
    thread = threading.Thread(
                target=vllm_generate,
                args=(
                    generation_config,
                    response_ids_Q,
                    prompt_ids_Q,
                ),
            )
    thread.start()
    print("vllm generate thread starts")

    # Load prompt builder
    # if 'qwen' in cfg.pretrained_checkpoint:
    #     prompt_builder_fn = QwenPromptBuilder
    #     cprint(f"Using QwenPromptBuilder for QWEN model", "yellow")
    # elif 'v01' in cfg.pretrained_checkpoint:
    #     prompt_builder_fn = VicunaV15ChatPromptBuilder
    # else:
    #     prompt_builder_fn = PurePromptBuilder

    # Initialize local logging
    # max_pet_name_len = len("openvla-7b")
    # model_pet_name = cfg.pretrained_checkpoint.split('/')[-1][:max_pet_name_len]
    model_pet_name = cfg.load_adapter_checkpoint.split('/')[-1] if cfg.load_adapter_checkpoint else cfg.pretrained_checkpoint.split('/')[-1]
    run_id = f"EVAL-{cfg.task_suite_name}-{model_pet_name}-t-{cfg.temperature}-s-{cfg.seed}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    # Append a timestamp to the run ID to avoid overwriting previous runs with identical args.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_run_id = f"{run_id}-{timestamp}"
    local_log_dir = os.path.join(cfg.local_log_dir, unique_run_id)
    os.makedirs(local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(local_log_dir, unique_run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    cprint(f"Logging to local log file: {local_log_filepath}", "cyan")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # cfg.exp_dir = cfg.pretrained_checkpoint if cfg.load_adapter_checkpoint is None else cfg.load_adapter_checkpoint    # for saved video
    cfg.exp_dir = local_log_dir
    # Clear exp_dir to avoid video duplication
    video_dir = os.path.join(cfg.exp_dir, "rollouts")
    cprint(f"Clearing existing videos in {video_dir}", "red")
    if os.path.exists(video_dir):
        for f in os.listdir(video_dir):
            if f.endswith(".mp4"):
                os.remove(os.path.join(video_dir, f))

    # Initialize vectorized environment
    if cfg.task_ids is None:
        base_task_ids = list(range(cfg.num_tasks_per_suite))
    else:
        base_task_ids = cfg.task_ids
    num_envs = len(base_task_ids) * max(1, int(cfg.number_envs_per_task))
    task_ids = []
    for tid in base_task_ids:
        task_ids.extend([tid] * max(1, int(cfg.number_envs_per_task)))

    eval_envs = LiberoVecEnv(
        task_suite_name=cfg.task_suite_name,
        task_ids=task_ids,
        num_trials_per_task=cfg.num_trials_per_task,
        seed=cfg.seed,
        model_family=cfg.model_family,
        center_crop=cfg.center_crop,
        rand_init_state=False,
        num_envs=num_envs,
        num_steps_wait=cfg.num_steps_wait,
        max_episode_length=None,
        resize_size=(224, 224),
    )
    timer = TimingManager()


    total_eval_episodes = sum(len(s) for s in eval_envs.initial_states_list)
    pbar = tqdm.tqdm(total=total_eval_episodes, desc="Evaluating", unit="episode")
    completed_so_far = 0

    # Main evaluation loop
    pre_thought = None
    step = 0
    obs, infos = eval_envs.reset()

    while not eval_envs.is_eval_complete():
        # Batch inference
        with timer.timer("vllm_generate"):
            prompt_ids_Q.put(obs)
            actions = response_ids_Q.get()

        # Step environments
        cprint(f"🕹️🕹️🕹️ Env {step=}", "cyan")
        with timer.timer("env_step"):
            next_obs, rewards, dones, truncated, infos = eval_envs.step(actions)

        if np.any(dones):
            completed_status = eval_envs.get_completed_status()
            current_success_rate = completed_status["success_rate"]
            current_episodes = completed_status["completed_episodes"]
            pbar.n = current_episodes
            pbar.refresh()
            pbar.set_postfix({
                'Success Rate': f'{current_success_rate:.3f}',
            })
        step += 1
        obs = next_obs
    
    # Compute results
    eval_infos = eval_envs.get_completed_status()
    total_episodes = eval_infos["completed_episodes"]
    success_rate = eval_infos["success_rate"]

    print(f"Episodes completed: {total_episodes}")
    print(f"Success rate: {success_rate:.2%}")
    log_file.write(f"Episodes completed: {total_episodes}\n")
    log_file.write(f"Success rate: {success_rate:.2%}\n")
    
    time_infos = timer.get_log()
    log_infos = {
        "success_rate/total": float(success_rate),
        "num_episodes/total": int(total_episodes),
    }
    # Include per-task success rates if available
    for k, v in eval_infos.items():
        if k.startswith("task_"):
            log_infos[f"success_rate/{k}"] = float(v)
    log_infos.update(time_infos)

    for k, v in log_infos.items():
        log_file.write(f"{k}: {v}\n")
    log_file.flush()
    pprint.pprint(log_infos)

    if cfg.use_wandb:
        wandb.log(log_infos)
        wandb.save(local_log_filepath)

    # Cleanup
    log_file.close()
    timer.close()
    eval_envs.close()
    prompt_ids_Q.put(None)  # Signal thread to stop
    thread.join()  # Wait for thread to finish
    
    ray.shutdown()

    cprint("Evaluation complete!", "green")


if __name__ == "__main__":
    eval_libero()
