"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import cv2
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(rollout_images, idx, success, task_description, mp4_path=None, backend="cv2"):
    """
    Saves an MP4 replay of an episode.
    
    Args:
        rollout_images: List of images to save as video
        idx: Episode index
        success: Whether the episode was successful
        task_description: Description of the task
        mp4_path: Path to save the video (if None, auto-generated)
        backend: Video backend to use ("imageio" or "cv2")
    """
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]

    if mp4_path is None:    # original
        rollout_dir = f"./rollouts/{DATE}"
        os.makedirs(rollout_dir, exist_ok=True)
        mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"

    if not rollout_images:
        print("Warning: No images to save in rollout video")
        return mp4_path

    # Get original dimensions
    first_img = rollout_images[0]
    h, w = first_img.shape[:2]
    
    if backend == "cv2":
        new_h = h if h % 2 == 0 else h + 1
        new_w = w if w % 2 == 0 else w + 1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(mp4_path, fourcc, 30.0, (new_w, new_h))
        for img in rollout_images:
            if new_h != h or new_w != w:
                img_resized = resize_image(img, (new_h, new_w))
            else:
                img_resized = img
            img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            video_writer.write(img_bgr)
        video_writer.release()

    elif backend == "imageio":  # imageio backend
        # Resize images to ensure dimensions are divisible by 16 for video codec compatibility
        new_h = ((h + 15) // 16) * 16
        new_w = ((w + 15) // 16) * 16
        if new_h != h or new_w != w:
            resized_images = []
            for img in rollout_images:
                # print(f"{img.shape=}")    # (224, 504, 3)
                resized_img = resize_image(img, (new_h, new_w))
                resized_images.append(resized_img)
            rollout_images = resized_images
        video_writer = imageio.get_writer(mp4_path, fps=30)
        for img in rollout_images:
            video_writer.append_data(img)
        video_writer.close()
    else:
        raise NotImplementedError(f"Unsupported video saving backend: {backend}")
    
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
