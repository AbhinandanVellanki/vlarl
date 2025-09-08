import os
import time
import numpy as np
# import gymnasium as gym
import gym
from typing import Dict, Any, Tuple, List, Optional
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from envs.base import EnvOutput
from envs.venv import SubprocVectorEnv
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
)
from experiments.robot.openvla_utils import preprocess_input_batch
from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action


class LiberoVecEnv(gym.Env):
    def __init__(
        self,
        task_suite_name: str,
        task_ids: List[int],
        num_trials_per_task: int = 50,
        seed: int = 42,
        model_family: str = "openvla",
        center_crop: bool = True,
        rand_init_state: bool = True,
        max_episode_length: Optional[int] = None,
        resolution: Optional[int] = 256,
        resize_size: Optional[Tuple[int, int]] = None,
        num_envs: Optional[int] = None,
    ):
        super().__init__()
        self.task_suite_name = task_suite_name
        self.task_ids = task_ids
        self.num_envs = num_envs or len(task_ids)
        self.is_vector_env = True
        self.model_family = model_family
        self.center_crop = center_crop
        self.resize_size = resize_size or (224, 224)
        self.num_trials_per_task = num_trials_per_task
        self.resolution = resolution
        self.seed_ = seed
        self.rand_init_state = rand_init_state
        
        if len(task_ids) < self.num_envs:
            raise ValueError(f"Not enough task_ids ({len(task_ids)}) for n_envs ({self.num_envs})")
        
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = self.benchmark_dict[self.task_suite_name]()
        self.tasks = []
        self.initial_states_list = []
        self.task_descriptions = []
        self.max_steps = max_episode_length if max_episode_length is not None else self._get_max_step(self.task_suite_name)
        
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({
            "pixel_values": gym.spaces.Box(
                low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
            ),
            "prompts": gym.spaces.Text(max_length=1000)
        })
        
        env_creators = []
        for i in range(self.num_envs):
            task_id = task_ids[i % len(task_ids)]
            task = self.task_suite.get_task(task_id)
            self.tasks.append(task)
            self.task_descriptions.append(task.language)
            
            task_initial_states = self.task_suite.get_task_init_states(task_id)
            if len(task_initial_states) > self.num_trials_per_task:
                task_initial_states = task_initial_states[:self.num_trials_per_task]
            self.initial_states_list.append(task_initial_states)
            
            bddl_file = os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file
            )
            env_args = {
                "bddl_file_name": bddl_file,
                "camera_heights": resolution,
                "camera_widths": resolution,
                # "render_gpu_device_id": -1,
                # "seed": self.seed_ + i,
            }
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            env_creators.append(lambda args=env_args: OffScreenRenderEnv(**args))
        
        self.envs = SubprocVectorEnv(env_creators)
        self.envs.seed(self.seed_)
        self.step_counts = np.zeros(self.num_envs)
        self.current_state_indices = np.zeros(self.num_envs, dtype=int)
        self.task_state_results = {}  # {(task_id, state_id): [success_results]}
        self.current_task_state_pairs = []  # Current (task_id, state_id) for each env
        
    def _get_max_step(self, task_suite_name: str) -> int:
        task_max_steps = {
            "libero_spatial": 220,
            "libero_object": 280,
            "libero_goal": 300,
            "libero_10": 520,
            "libero_90": 400,
        }
        return task_max_steps.get(task_suite_name, 300)
        
    def reset(self,):
        # reset_start_time = time.time()
        self.step_counts = np.zeros(self.num_envs)
        self.task_state_results = {}
        self.current_task_state_pairs = []
        
        init_states = []
        for i in range(self.num_envs):
            task_id = self.task_ids[i % len(self.task_ids)]
            if self.rand_init_state:
                state_id = np.random.randint(0, len(self.initial_states_list[i]))
            else:
                # state_id = self.current_state_indices[i] % len(self.initial_states_list[i])
                state_id = 0
                self.current_state_indices[i] += 1
            init_states.append(self.initial_states_list[i][state_id])
            self.current_task_state_pairs.append((task_id, state_id))

        obs_list = self.envs.set_init_state(init_states)
        dummy_action = get_libero_dummy_action()
        dummy_actions = [dummy_action] * self.num_envs
        
        for _ in range(10): # Stablize the env
            obs_list, _, _, _ = self.envs.step(dummy_actions)
        
        pixel_values = []
        prompts = []
        for i, obs in enumerate(obs_list):
            img = get_libero_image(obs, self.resize_size)
            if isinstance(img, Image.Image):
                img = np.array(img)
            pixel_values.append(img)
            prompts.append(self.task_descriptions[i])
        
        img_list, prompt_list = preprocess_input_batch(
            pixel_values, prompts, 
            pre_thought_list=None, center_crop=self.center_crop
        )
        env_output = EnvOutput(pixel_values=img_list, prompts=prompt_list)
        info = {
            "task_descriptions": prompts,
            "step_counts": self.step_counts.copy(),
        }
        # print(f"Env reset time: {time.time() - reset_start_time:.2f} seconds")
        return env_output, info
    
    def step(self, actions, **kwargs):
        actions = normalize_gripper_action(actions, binarize=True)
        if self.model_family == "openvla":
            actions = invert_gripper_action(actions)
        
        obs_list, rewards, dones, infos = self.envs.step(actions)
        self.step_counts += 1
        
        for i in range(self.num_envs):
            if self.step_counts[i] >= self.max_steps:
                dones[i] = True
        
        if np.any(dones):
            done_indices = np.where(dones)[0]
            # Record success/fail results for completed episodes
            for i in done_indices:
                task_id, state_id = self.current_task_state_pairs[i]
                success = rewards[i] > 0
                if (task_id, state_id) not in self.task_state_results:
                    self.task_state_results[(task_id, state_id)] = []
                self.task_state_results[(task_id, state_id)].append(success)
            
            # Auto-reset environments that are done
            new_initial_states = []
            dummy_actions = []
            for i in done_indices:
                task_id = self.task_ids[i % len(self.task_ids)]
                if self.rand_init_state:
                    state_id = np.random.randint(0, len(self.initial_states_list[i]))
                else:
                    state_id = self.current_state_indices[i] % len(self.initial_states_list[i])
                    self.current_state_indices[i] += 1
                new_initial_states.append(self.initial_states_list[i][state_id])
                self.current_task_state_pairs[i] = (task_id, state_id)
                dummy_action = get_libero_dummy_action()
                dummy_actions.append(dummy_action)
            self.envs.reset(id=done_indices.tolist())
            obs = self.envs.set_init_state(new_initial_states, id=done_indices.tolist())

            for _ in range(10): # Stablize the env
                obs, _, _, _ = self.envs.step(dummy_actions, id=done_indices.tolist())

            for i, done_idx in enumerate(done_indices):
                obs_list[done_idx] = obs[i]
        
        pixel_values = []
        prompts = []
        for i, obs in enumerate(obs_list):
            img = get_libero_image(obs, self.resize_size)
            pixel_values.append(img)
            prompts.append(self.task_descriptions[i])
        img_list, prompt_list = preprocess_input_batch(
            pixel_values, prompts, 
            pre_thought_list=None, center_crop=True
        )
        env_output = EnvOutput(pixel_values=img_list, prompts=prompt_list)
        info = {
            "task_descriptions": prompts,
            "step_counts": self.step_counts.copy(),
        }
        truncated = np.array([False] * self.num_envs)

        for i, done in enumerate(dones):
            if done:
                self.step_counts[i] = 0
        return env_output, np.array(rewards), np.array(dones), truncated, info

    def is_eval_complete(self) -> bool:
        """Check if all task+initial state pairs have been tested at least once."""
        all_pairs = set()
        for i in range(self.num_envs):
            task_id = self.task_ids[i % len(self.task_ids)]
            for state_id in range(len(self.initial_states_list[i])):
                all_pairs.add((task_id, state_id))
        tested_pairs = set(self.task_state_results.keys())
        return all_pairs.issubset(tested_pairs)
    
    def get_task_state_results(self) -> Dict[Tuple[int, int], List[bool]]:
        return self.task_state_results.copy()

    def get_completed_status(self) -> Tuple[float, int]:
        if not self.task_state_results:
            return 0.0, 0
        total_pairs = len(self.task_state_results)
        successful_pairs = sum(1 for results in self.task_state_results.values() if results[0])
        success_rate = successful_pairs / total_pairs if total_pairs > 0 else 0.0
        per_task_success_rates = {
            f"task_{task_id}": sum(1 for (t_id, _), results in self.task_state_results.items()
                                                if t_id == task_id and results[0]) /
            sum(1 for (t_id, _ ) in self.task_state_results.keys() if t_id == task_id)
            for task_id in set(self.task_ids)
        }
        infos = {
            "success_rate": success_rate,
            "completed_episodes": total_pairs,
            # per-task success rate
            **per_task_success_rates,
        }
        return infos

    def close(self):
        self.envs.close()
    
    def __len__(self):
        return self.num_envs
