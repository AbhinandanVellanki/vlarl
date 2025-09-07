import os
import numpy as np
import gymnasium as gym
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
        max_videos_per_env: int = 10,
        add_info_overlay: bool = True,
        env_gpu_id: int = 0
    ):
        super().__init__(env)
        self.env = env
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_stats = save_stats
        self.max_videos_per_env = max_videos_per_env
        self.add_info_overlay = add_info_overlay
        self.env_gpu_id = env_gpu_id
        
        self.num_envs = env.num_envs
        self.frames = [[] for _ in range(self.num_envs)]
        self.episode_counts = [0] * self.num_envs
        self.video_counts = [0] * self.num_envs
        self.current_step = 0

        os.makedirs(save_dir, exist_ok=True)
        
        self.task_descriptions = env.task_descriptions
        self.replay_images = {i: [] for i in range(self.num_envs)}
        self.total_episodes = 0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Reset frame buffers for all environments
        self.frames = [[] for _ in range(self.num_envs)]
        self.replay_images = {i: [] for i in range(self.num_envs)}
        self.current_step = 0
        
        # Capture initial frames if enabled
        pixel_values = obs["pixel_values"]
        if isinstance(pixel_values, list):
            for i in range(min(len(pixel_values), self.num_envs)):
                if self.video_counts[i] < self.max_videos_per_env:
                    img = pixel_values[i]
                    if self.add_info_overlay and self.task_descriptions:
                        img_args = {
                            "goal": self.task_descriptions[i],
                            "step": self.current_step,
                        }
                        img = add_info_board(img, **img_args)
                    self.frames[i].append(img)
                    self.replay_images[i].append(img)
        
        return obs, info
    
    def step(self, actions, **kwargs):
        # Get additional info from kwargs for video overlay
        values = kwargs.get('values', None)
        log_probs = kwargs.get('log_probs', None)
        prm_rewards = kwargs.get('prm_rewards', None)
        
        obs, rewards, dones, truncated, info = self.env.step(actions, **kwargs)
        self.current_step += 1
        
        # Capture frames for each environment
        pixel_values = obs["pixel_values"]
        if isinstance(pixel_values, list):
            for i in range(min(len(pixel_values), self.num_envs)):
                if self.video_counts[i] < self.max_videos_per_env and len(self.frames[i]) < 1000:  # Limit video length
                    img = pixel_values[i]
                    
                    if self.add_info_overlay:
                        img_args = {
                            "goal": self.task_descriptions[i],
                            "step": self.current_step,
                            "action": actions[i],
                        }
                        
                        if self.save_stats:
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
        
        # Handle episode completion
        if np.any(dones):
            done_indices = np.where(dones)[0]
            for i in done_indices:
                if self.frames[i] and self.video_counts[i] < self.max_videos_per_env:
                    self._save_video(i, rewards[i])
                    self.video_counts[i] += 1
                
                self.episode_counts[i] += 1
                self.frames[i] = []  # Reset frame buffer for this environment
        
        return obs, rewards, dones, truncated, info
    
    def _save_video(self, env_idx: int, reward=None):
        """Save video for a specific environment."""
        if not self.frames[env_idx]:
            return
        
        success = False
        if reward is not None:
            success = reward > 0
        
        task_description = self.task_descriptions[env_idx]
        processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
        
        mp4_path = os.path.join(
            self.save_dir, 
            f"rk_{self.env_gpu_id}--epi={self.total_episodes}--s={success}--"
            f"task={env_idx}--inst={processed_task_description}.mp4"
        )
        save_rollout_video(
            self.frames[env_idx], 
            self.episode_counts[env_idx], 
            success=success,
            task_description=str(task_description),
            log_file=None,
            mp4_path=mp4_path,
        )
        print(f"Video saved to: {mp4_path}")
        
        self.frames[env_idx] = []
        self.replay_images[env_idx] = []
    
    def close(self):
        # Save any remaining videos
        for i in range(self.num_envs):
            if self.frames[i]:
                self._save_video(i)
        self.env.close()


class CurriculumWrapper(gym.Wrapper):
    def __init__(
        self, 
        env, 
        temp: float = 1.0, 
        min_prob: float = 0.1, 
        window_size: int = 5,
        recompute_freq: int = 10,
        enable_logging: bool = True,
        exp_dir: str = "./"
    ):
        super().__init__(env)
        self.env = env
        self.temp = temp
        self.min_prob = min_prob
        self.window_size = window_size
        self.recompute_freq = recompute_freq
        self.enable_logging = enable_logging
        self.exp_dir = exp_dir
        
        # Get number of environments from LiberoVecEnv
        self.num_envs = env.num_envs
        
        # Success tracking: {task_id: {state_id: deque([successes])}}
        self.success_tracker = {}
        self.step_count = 0
        
        # Get task suite info from wrapped env
        self.task_suite_name = env.task_suite_name
        self.task_descriptions = env.task_descriptions
        
        # Track current state IDs for each environment
        self.current_state_ids = [None] * self.num_envs
        self.task_id_mapping = getattr(env, 'task_id_mapping', None)
        
        # Success tracking for episodes
        self.success = np.zeros(self.num_envs, dtype=bool)
        self.total_episodes = 0
        self.initial_state_ids = []
        self.replay_images = {i: [] for i in range(self.num_envs)}
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Reset episode tracking
        self.success = np.zeros(self.num_envs, dtype=bool)
        
        # Extract current state IDs if available in info
        if info and 'current_state_ids' in info:
            self.current_state_ids = info['current_state_ids']
        
        return obs, info
    
    def step(self, actions, **kwargs):
        step_count_tmp = self.step_count
        
        obs, rewards, dones, truncated, info = self.env.step(actions, **kwargs)
        self.step_count += 1
        
        # Update success tracker for completed episodes
        if np.any(dones):
            done_indices = np.where(dones)[0]
            
            for i in done_indices:
                # Get success status
                success = rewards[i] > 0
                self.success[i] = success
                
                # Update success tracker if we have state info
                if (self.current_state_ids[i] is not None and 
                    self.task_id_mapping is not None and 
                    i < len(self.task_id_mapping)):
                    
                    # Get actual task ID from mapping
                    actual_task_id = self.task_id_mapping[i]
                    state_id = self.current_state_ids[i]
                    
                    self._update_success_tracker(actual_task_id, state_id, success)
        
        # Add curriculum stats to info periodically
        if self.step_count % self.recompute_freq == 0 and self.enable_logging:
            curriculum_stats = self.get_curriculum_stats()
            if 'curriculum_stats' not in info:
                info['curriculum_stats'] = {}
            info['curriculum_stats'].update(curriculum_stats)
        
        # Add step count info
        info['step_count_tmp'] = step_count_tmp
        
        return obs, rewards, dones, truncated, info
    
    def _update_success_tracker(self, task_id: int, state_id: int, success: bool):
        """Update success history for a task-state pair."""
        if task_id not in self.success_tracker:
            self.success_tracker[task_id] = {}
        
        if state_id not in self.success_tracker[task_id]:
            self.success_tracker[task_id][state_id] = deque(maxlen=self.window_size)
        
        self.success_tracker[task_id][state_id].append(1.0 if success else 0.0)
        
        if self.enable_logging:
            success_rate = self._get_success_rate(task_id, state_id)
            print(f"[Curriculum] Task {task_id}, State {state_id}: Success rate = {success_rate:.2f}")
    
    def _get_success_rate(self, task_id: int, state_id: int) -> float:
        """Calculate success rate for a task-state pair."""
        if (task_id not in self.success_tracker or 
            state_id not in self.success_tracker[task_id]):
            return 0.0
        
        history = self.success_tracker[task_id][state_id]
        if not history:
            return 0.0
        
        return sum(history) / len(history)
    
    def sample_state_with_curriculum(self, task_id: int, n_states: int) -> int:
        """Sample state using curriculum learning based on success rates."""
        if task_id not in self.success_tracker:
            return np.random.randint(0, n_states)
        
        weights = []
        state_ids = list(range(n_states))
        
        for state_id in state_ids:
            success_rate = self._get_success_rate(task_id, state_id)
            # Power law weighting: focus on harder states (lower success rates)
            weight = ((1.0 - success_rate + 1e-9) ** (1.0 / self.temp))
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return np.random.randint(0, n_states)
        
        probabilities = [w / total_weight for w in weights]
        
        # Ensure minimum sampling probability
        probabilities = [max(p, self.min_prob / n_states) for p in probabilities]
        probabilities = [p / sum(probabilities) for p in probabilities]
        
        return np.random.choice(state_ids, p=probabilities)
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics for logging."""
        stats = {
            'total_tasks': len(self.success_tracker),
            'total_states_tracked': sum(len(states) for states in self.success_tracker.values()),
        }
        
        # Per-task statistics
        for task_id, states in self.success_tracker.items():
            state_success_rates = []
            for _state_id, history in states.items():
                if history:
                    success_rate = sum(history) / len(history)
                    state_success_rates.append(success_rate)
            
            if state_success_rates:
                avg_success_rate = sum(state_success_rates) / len(state_success_rates)
                stats[f'task_{task_id}_avg_success_rate'] = avg_success_rate
                stats[f'task_{task_id}_states'] = len(state_success_rates)
                stats[f'task_{task_id}_min_success_rate'] = min(state_success_rates)
                stats[f'task_{task_id}_max_success_rate'] = max(state_success_rates)
        
        return stats
    
    def reset_task_statistics(self, task_id: int = None):
        """Reset statistics for a specific task or all tasks."""
        if task_id is None:
            self.success_tracker.clear()
        elif task_id in self.success_tracker:
            del self.success_tracker[task_id]
    
    def close(self):
        if self.enable_logging:
            final_stats = self.get_curriculum_stats()
            print(f"[Curriculum] Final statistics: {final_stats}")
        self.env.close()
