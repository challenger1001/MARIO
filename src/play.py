import torch
import numpy as np
from envs.mario_env import MarioEnv
from models.cnn_actor_critic import ActorCritic
from config import *

def play():
    # 初始化环境
    env = MarioEnv(simple_movement=SIMPLE_MOVEMENT, stack_frames=STACK_FRAMES, resize_dim=RESIZE_DIM)
    n_actions = len(env.action_space)
    state_shape = (STACK_FRAMES, RESIZE_DIM[0], RESIZE_DIM[1])

    # 创建模型
    model = ActorCritic(state_shape, n_actions).to(DEVICE)

    # 加载训练好的模型参数
    model.load_state_dict(torch.load("mario_ppo_model.pth", map_location=DEVICE))
    model.eval()  # 推理模式

    # 重置环境
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # 转换状态为 tensor
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(DEVICE)

        # 前向计算动作概率
        with torch.no_grad():
            logits, _ = model(state_tensor)
            action_prob = torch.softmax(logits, dim=-1)
            action = torch.argmax(action_prob, dim=-1).item()

        # 执行动作
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        # 渲染游戏
        env.render()

        state = next_state

    print(f"游戏结束，总得分：{total_reward}")
    env.close()

if __name__ == "__main__":
    play()
