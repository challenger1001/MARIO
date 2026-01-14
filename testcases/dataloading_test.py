import gym
import numpy as np
import cv2
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym_super_mario_bros


# ===============================
# 一、图像处理函数
# ===============================
def preprocess_observation(obs, resize_dim=(84, 84), grayscale=True):
    """
    输入: 原始 obs (H, W, 3)
    输出: 处理后的 obs
    """
    print(f"[Raw obs] shape={obs.shape}, dtype={obs.dtype}")

    # 1. Resize
    obs = cv2.resize(obs, resize_dim, interpolation=cv2.INTER_AREA)
    print(f"[After resize] shape={obs.shape}")

    # 2. Gray scale
    if grayscale:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        print(f"[After grayscale] shape={obs.shape}")

        # (H, W) -> (1, H, W)
        obs = np.expand_dims(obs, axis=0)
        print(f"[After expand dims] shape={obs.shape}")
    else:
        # (H, W, C) -> (C, H, W)
        obs = obs.transpose(2, 0, 1)
        print(f"[After transpose] shape={obs.shape}")

    # 3. Normalize
    obs = obs.astype(np.float32) / 255.0
    print(f"[After normalize] dtype={obs.dtype}, min={obs.min():.3f}, max={obs.max():.3f}")

    return obs


# ===============================
# 二、帧堆叠测试
# ===============================
def stack_frames(frames, new_frame, stack_size=4):
    """
    frames: list of frames
    new_frame: (1, H, W)
    """
    if len(frames) == 0:
        frames = [new_frame for _ in range(stack_size)]
    else:
        frames.append(new_frame)
        frames = frames[-stack_size:]

    stacked = np.concatenate(frames, axis=0)
    print(f"[After stack] shape={stacked.shape}")
    return frames, stacked


# ===============================
# 三、主测试函数
# ===============================
def test_observation_pipeline():
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0",
        apply_api_compatibility=True
    )
    env = gym.wrappers.JoypadSpace(env, SIMPLE_MOVEMENT)

    obs, _ = env.reset()
    frames = []

    print("\n===== Observation Transform Test Start =====\n")

    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\n--- Step {step} ---")
        processed_obs = preprocess_observation(obs)

        frames, stacked_obs = stack_frames(frames, processed_obs)

        print(f"[Final input to model] shape={stacked_obs.shape}")

        if terminated or truncated:
            obs, _ = env.reset()
            frames = []

    env.close()
    print("\n===== Test Finished =====")


if __name__ == "__main__":
    test_observation_pipeline()
