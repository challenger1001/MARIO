import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import numpy as np
from collections import deque
import cv2
from nes_py.wrappers import JoypadSpace
from config import *

class MarioEnv:
    def __init__(self, simple_movement=True, stack_frames=4, resize_dim=(84, 84)):
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")

        # 🔥 核心修改：显式移除 TimeLimit wrapper
        while hasattr(env, "env"):
            if env.__class__.__name__ == "TimeLimit":
                env = env.env
                print("remove timelimit")
                break
            env = env.env

        # 替换 SIMPLE_MOVEMENT
        SIMPLE_MOVEMENT = [
            ['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'],
            ['A'], ['left'], ['left', 'A'], ['left', 'B'], ['left', 'A', 'B'],
            ['down'], ['up']
        ]

        self.action_space = SIMPLE_MOVEMENT if simple_movement else COMPLEX_MOVEMENT
        print("SIMPLE_MOVEMENT:", SIMPLE_MOVEMENT)
        print("COMPLEX_MOVEMENT:", COMPLEX_MOVEMENT)
        print("动作集类型:", type(self.action_space))
        print("动作集内容:", self.action_space)
        self.env = JoypadSpace(env, self.action_space)
        self.stack_frames = stack_frames
        self.resize_dim = resize_dim
        self.frames = deque(maxlen=stack_frames)
        
    def preprocess(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.resize_dim)
        obs = obs / 255.0
        return obs
    
    def reset(self):
        obs= self.env.reset()
        # print("obs的类型：", type(obs))
        if isinstance(obs, tuple):  # 防止 gymnasium / wrapper 行为
            obs = obs[0]

        obs = self.preprocess(obs)
        self.frames.clear()
        for _ in range(self.stack_frames):
            self.frames.append(obs)
        return np.stack(self.frames, axis=0)
    
    def step(self, action_idx):
        result = self.env.step(action_idx)
        # print("返回result的数量", len(result))

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        obs = self.preprocess(obs)
        self.frames.append(obs)
        stacked_state = np.stack(self.frames, axis=0)
        
        return stacked_state, reward, done, info
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
