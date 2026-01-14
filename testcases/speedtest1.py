import torch
import torch.optim as optim
import numpy as np

from envs.mario_env import MarioEnv
from models.cnn_actor_critic import ActorCritic
from test_hyperparams_config import TEST_HYPERPARAMS


def test_hyperparams():
    print("\n===== Hyperparameter Test Start =====\n")

    # 1. 初始化环境
    env = MarioEnv()
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # 2. 初始化模型
    model = ActorCritic(obs_shape, n_actions)
    model.train()

    # 3. 使用“测试用超参数”
    optimizer = optim.Adam(
        model.parameters(),
        lr=TEST_HYPERPARAMS["learning_rate"]
    )

    print("[Hyperparameters in use]")
    for k, v in TEST_HYPERPARAMS.items():
        print(f"  {k}: {v}")

    # 4. rollout buffer（刻意很小）
    batch_size = TEST_HYPERPARAMS["batch_size"]
    states, actions, rewards = [], [], []

    obs = env.reset()
    total_steps = 0

    # ===============================
    # 核心测试循环
    # ===============================
    while total_steps < TEST_HYPERPARAMS["max_steps"]:
        obs_tensor = torch.tensor(obs).unsqueeze(0)

        with torch.no_grad():
            action, _ = model.act(obs_tensor)

        next_obs, reward, done, info = env.step(action)

        states.append(obs)
        actions.append(action)
        rewards.append(reward)

        obs = next_obs
        total_steps += 1

        # -------------------------------
        # 触发一次“更新”
        # -------------------------------
        if len(states) >= batch_size:
            states_t = torch.tensor(np.array(states))
            actions_t = torch.tensor(actions)
            rewards_t = torch.tensor(rewards)

            loss = model.compute_test_loss(states_t, actions_t, rewards_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"[Update] step={total_steps}, "
                f"batch_size={batch_size}, "
                f"lr={optimizer.param_groups[0]['lr']:.1e}, "
                f"loss={loss.item():.4f}"
            )

            states, actions, rewards = [], [], []

        if done:
            obs = env.reset()

    env.close()
    print("\n===== Hyperparameter Test Finished =====")


if __name__ == "__main__":
    test_hyperparams()
