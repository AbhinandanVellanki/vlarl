import os
import shutil
from termcolor import cprint
import numpy as np
# import gymnasium as gym
import gym
from collections import deque
from typing import Dict, List, Optional, Any, Tuple
from experiments.robot.libero.libero_utils import save_rollout_video, get_libero_image
from utils.util import add_info_board


class VideoWrapper(gym.Wrapper):
    def __init__(
        self, 
        env, 
        save_dir: str = "video", 
        save_freq: int = 1,
        save_stats: bool = True,
        max_videos_per_env: int = 10000,
        env_gpu_id: int = 0,
    ):
        super().__init__(env)
        self.env = env
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_stats = save_stats
        self.max_videos_per_env = max_videos_per_env
        self.env_gpu_id = env_gpu_id
        
        self.num_envs = env.num_envs
        self.frames = [[] for _ in range(self.num_envs)]
        self.episode_counts = [0] * self.num_envs
        self.video_counts = [0] * self.num_envs
        self.task_descriptions = env.task_descriptions
        self.replay_images = {i: [] for i in range(self.num_envs)}
        self.total_episodes = 0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        self.frames = [[] for _ in range(self.num_envs)]
        self.replay_images = {i: [] for i in range(self.num_envs)}

        pixel_values = obs["pixel_values"]
        for i in range(min(len(pixel_values), self.num_envs)):
        # for i in range(self.num_envs):
            if self.video_counts[i] < self.max_videos_per_env:
                img = pixel_values[i]
                if self.task_descriptions:
                    img_args = {
                        "goal": self.task_descriptions[i],
                        "step": self.env.step_counts[i],
                    }
                    img = add_info_board(img, **img_args)
                self.frames[i].append(img)
                self.replay_images[i].append(img)
        
        return obs, info
    
    def step(self, actions, **kwargs):
        values = kwargs.get('values', None)
        log_probs = kwargs.get('log_probs', None)
        prm_rewards = kwargs.get('prm_rewards', None)
        
        obs, rewards, dones, truncated, info = self.env.step(actions, **kwargs)

        pixel_values = obs["pixel_values"]
        for i in range(min(len(pixel_values), self.num_envs)):
        # for i in range(self.num_envs):
            if self.video_counts[i] < self.max_videos_per_env and len(self.frames[i]) < 1000:
                img = pixel_values[i]
                if self.save_stats:
                    img_args = {
                        "goal": self.task_descriptions[i],
                        "step": self.env.step_counts[i],
                        "action": actions[i],
                    }
                    if values is not None:
                        img_args["value"] = values[i]
                    if log_probs is not None:
                        img_args["prob"] = np.exp(log_probs[i])
                        img_args["entropy"] = (-log_probs[i]).mean()
                    if prm_rewards is not None:
                        img_args["prm_rewards"] = prm_rewards[i]
                    
                    img = add_info_board(img, **img_args)
                self.frames[i].append(img)
                self.replay_images[i].append(img)

        if np.any(dones):
            done_indices = np.where(dones)[0]
            for i in done_indices:
                if self.frames[i] and self.video_counts[i] < self.max_videos_per_env:
                    self._save_video(i, rewards[i])
                    self.video_counts[i] += 1
                    self.total_episodes += 1
                else:
                    cprint(f"[VideoWrapper] Skipping video save for env {i}", "yellow")
                self.episode_counts[i] += 1
                self.frames[i] = []
        
        return obs, rewards, dones, truncated, info
    
    def _save_video(self, env_idx: int, reward=None):
        """Save video for a specific environment."""
        success = False
        if reward is not None:
            success = reward > 0
        
        task_description = self.task_descriptions[env_idx]
        processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
        mp4_path = os.path.join(
            self.save_dir, 
            f"rk={self.env_gpu_id}+epi={self.total_episodes}+s={success}+"
            f"task={env_idx}+inst={processed_task_description}.mp4"
        )
        save_rollout_video(
            self.frames[env_idx], 
            self.episode_counts[env_idx], 
            success=success,
            task_description=str(task_description),
            mp4_path=mp4_path,
            backend="cv2",
        )
        self.frames[env_idx] = []
        self.replay_images[env_idx] = []
    
    def close(self):
        for i in range(self.num_envs):
            if self.frames[i]:
                self._save_video(i)
            else:
                cprint(f"[VideoWrapper] No frames to save for env {i} on close.", "yellow")
        self.env.close()

from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_image
from experiments.robot.openvla_utils import preprocess_input_batch
from envs.base import EnvOutput
class CurriculumWrapper(gym.Wrapper):
    """
    An adaptive curriculum that selects tasks based on the agent's current capabilities, which 
    prioritizes tasks with ~50% success rate as the frontier of the agent's capabilities, 
    while maintaining exposure to both mastered and challenging tasks, improving sample 
    efficiency and generalization.
    """
    def __init__(
        self, 
        env, 
        temp: float = 1.0, 
        min_prob: float = 0.0, 
        recompute_freq: int = 10,
        enable_logging: bool = True,
        exp_dir: str = "./"
    ):
        super().__init__(env)
        self.env = env
        self.temp = temp
        self.min_prob = min_prob
        self.target_success_rate = 0.5
        self.recompute_freq = recompute_freq
        self.enable_logging = enable_logging
        self.exp_dir = exp_dir
        self.step_count = 0
        self.num_envs = env.num_envs
        self.task_descriptions = env.task_descriptions

        self._patch_auto_reset_logic()
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, actions, **kwargs):
        obs, rewards, dones, truncated, info = self.env.step(actions, **kwargs)
        self.step_count += 1
        if self.step_count % self.recompute_freq == 0 and self.enable_logging:
            curriculum_stats = self.get_curriculum_stats()
            if 'curriculum_stats' not in info:
                info['curriculum_stats'] = {}
            info['curriculum_stats'].update(curriculum_stats)
        
        return obs, rewards, dones, truncated, info
    
    def _patch_auto_reset_logic(self):
        """Monkey patch the base environment's auto-reset logic to use curriculum."""
        def curriculum_step(actions, **kwargs):
            actions = normalize_gripper_action(actions, binarize=True)
            if self.env.model_family == "openvla":
                actions = invert_gripper_action(actions)
            
            obs_list, rewards, dones, infos = self.env.envs.step(actions)
            self.env.step_counts += 1
            
            for i in range(self.env.num_envs):
                if self.env.step_counts[i] >= self.env.max_steps:
                    dones[i] = True
            if np.any(dones):
                done_indices = np.where(dones)[0]
                # Record success/fail results for completed episodes
                for i in done_indices:
                    task_id, state_id = self.env.current_task_state_pairs[i]
                    success = rewards[i] > 0
                    if (task_id, state_id) not in self.env.task_state_results:
                        self.env.task_state_results[(task_id, state_id)] = []
                    self.env.task_state_results[(task_id, state_id)].append(success)
                
                # Auto-reset with curriculum-based state selection
                new_initial_states = []
                dummy_actions = []
                
                for i in done_indices:
                    task_id = self.env.task_ids[i % len(self.env.task_ids)]

                    state_id = self._sample_state_with_curriculum(task_id, len(self.env.initial_states_list[i]))
                    
                    new_initial_states.append(self.env.initial_states_list[i][state_id])
                    self.env.current_task_state_pairs[i] = (task_id, state_id)
                    dummy_action = get_libero_dummy_action()
                    dummy_actions.append(dummy_action)
                
                self.env.envs.reset(id=done_indices.tolist())
                obs = self.env.envs.set_init_state(new_initial_states, id=done_indices.tolist())

                for _ in range(10): # Stabilize the env
                    obs, _, _, _ = self.env.envs.step(dummy_actions, id=done_indices.tolist())

                for i, done_idx in enumerate(done_indices):
                    obs_list[done_idx] = obs[i]
            
            # Continue with original processing
            pixel_values = []
            prompts = []
            for i, obs in enumerate(obs_list):
                
                img = get_libero_image(obs, self.env.resize_size)
                pixel_values.append(img)
                prompts.append(self.env.task_descriptions[i])
            
            img_list, prompt_list = preprocess_input_batch(
                pixel_values, prompts, 
                pre_thought_list=None, center_crop=True
            )
            env_output = EnvOutput(pixel_values=img_list, prompts=prompt_list)
            info = {
                "task_descriptions": prompts,
                "step_counts": self.env.step_counts.copy(),
            }
            truncated = np.array([False] * self.env.num_envs)

            for i, done in enumerate(dones):
                if done:
                    self.env.step_counts[i] = 0
            return env_output, np.array(rewards), np.array(dones), truncated, info
        self.env.step = curriculum_step

    def _get_success_rate(self, task_id: int, state_id: int) -> float:
        # if not hasattr(self.env, 'get_task_state_results'):
        #     return 0.0
        task_state_results = self.env.get_task_state_results()
        if (task_id, state_id) not in task_state_results:
            return 0.0
        results = task_state_results[(task_id, state_id)]
        if not results:
            return 0.0
        return sum(results) / len(results)
    
    def _sample_state_with_curriculum(self, task_id: int, n_states: int) -> int:
        # if not hasattr(self.env, 'get_task_state_results'):
        #     return np.random.randint(0, n_states)
        weights = []
        state_ids = list(range(n_states))
        
        for state_id in state_ids:
            success_rate = self._get_success_rate(task_id, state_id)
            distance_from_target = abs(success_rate - self.target_success_rate)
            weight = 1.0 / (distance_from_target + 1e-9)
            weight = weight ** (1.0 / self.temp)
            weights.append(weight)
        total_weight = sum(weights)
        if total_weight == 0:
            return np.random.randint(0, n_states)
        
        probabilities = [w / total_weight for w in weights]
        # Ensure minimum sampling probability for exploration
        probabilities = [max(p, self.min_prob / n_states) for p in probabilities]
        probabilities = [p / sum(probabilities) for p in probabilities]
        
        return np.random.choice(state_ids, p=probabilities)
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        # if not hasattr(self.env, 'get_task_state_results'):
        #     return {}
        task_state_results = self.env.get_task_state_results()
        if not task_state_results:
            return {}
        task_groups = {}
        for (task_id, state_id), results in task_state_results.items():
            if task_id not in task_groups:
                task_groups[task_id] = {}
            task_groups[task_id][state_id] = results
        
        stats = {
            'total_visited_tasks': len(task_groups),
            'total_visited_states': len(task_state_results),
        }
        for task_id, states in task_groups.items():
            state_success_rates = []
            for state_id, results in states.items():
                if results:
                    success_rate = sum(results) / len(results)
                    state_success_rates.append(success_rate)
                    # distance = abs(success_rate - self.target_success_rate)
            if state_success_rates:
                avg_success_rate = sum(state_success_rates) / len(state_success_rates)
                stats[f'task_{task_id}_avg_success_rate'] = avg_success_rate
                stats[f'task_{task_id}_visited_states'] = len(state_success_rates)
        return stats
    
    def close(self):
        self.env.close()
