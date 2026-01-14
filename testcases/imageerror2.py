import psutil
import os


def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

import cv2
import numpy as np
from collections import deque


class HighQualityRenderer:
    def __init__(
        self,
        resize=(512, 480),
        max_buffer_size=120,   # 最多缓存 120 帧（≈2 秒）
        enable_postprocess=True
    ):
        self.resize = resize
        self.enable_postprocess = enable_postprocess
        self.buffer = deque(maxlen=max_buffer_size)

    def postprocess(self, frame):
        # 对比度增强 + 锐化
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.enable_postprocess:
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            frame = cv2.filter2D(frame, -1, kernel)

        return frame

    def render(self, frame):
        frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_CUBIC)
        frame = self.postprocess(frame)

        self.buffer.append(frame)
        cv2.imshow("High Quality Mario Render", frame)
        cv2.waitKey(1)

import time
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def test_high_quality_render():
    print("\n===== High Quality Render Test Start =====\n")

    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0",
        apply_api_compatibility=True,
        render_mode="rgb_array"   # 关键：切换渲染模式
    )
    env = gym.wrappers.JoypadSpace(env, SIMPLE_MOVEMENT)

    renderer = HighQualityRenderer(
        resize=(512, 480),
        max_buffer_size=120,
        enable_postprocess=True
    )

    obs, info = env.reset()
    start_time = time.time()

    for step in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()   # rgb_array
        renderer.render(frame)

        if step % 30 == 0:
            mem = get_memory_mb()
            print(f"[Step {step}] Memory usage: {mem:.2f} MB")

        time.sleep(0.03)

        if terminated or truncated:
            env.reset()

    env.close()
    cv2.destroyAllWindows()

    print("\n✅ High quality render test finished\n")


if __name__ == "__main__":
    test_high_quality_render()
