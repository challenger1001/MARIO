import numpy as np
from envs.mario_env import MarioEnv
from models.cnn_actor_critic import ActorCritic
from ppo.ppo_agent import PPOAgent
from config import *

def train():
    env = MarioEnv(simple_movement=SIMPLE_MOVEMENT, stack_frames=STACK_FRAMES, resize_dim=RESIZE_DIM)
    n_actions = len(env.action_space)
    state_shape = (STACK_FRAMES, RESIZE_DIM[0], RESIZE_DIM[1])
    model = ActorCritic(state_shape, n_actions)
    agent = PPOAgent(model, n_actions)

    print("=== 开始训练 Mario PPO ===")
    for episode in range(20):
        state = env.reset()
        done = False
        rewards, log_probs, states, actions, masks, values = [], [], [], [], [], []
        step_count = 0
        MAX_STEPS = 5000  # 1000 / 100

        while not done and step_count < MAX_STEPS:
            step_count += 1
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            masks.append(1 - done)
            # Critic value
            _, value = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE))
            values.append(value.item())

            state = next_state

            # 每 100 步打印一次进度
            # if step_count % 100 == 0:
            #     print(f"[Episode {episode}] Step {step_count}, Reward Sum: {sum(rewards):.2f}")

            # env.render()

        # GAE 计算
        _, next_value = model(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(DEVICE))
        returns = agent.compute_gae(rewards, masks, values, next_value.item())
        advantages = np.array(returns) - np.array(values)
        agent.ppo_update(states, actions, log_probs, returns, advantages)

        print(f"Episode {episode}, Total Reward: {sum(rewards)}, Step Count: {step_count}")

    torch.save(model.state_dict(), "mario_ppo_model.pth")
    print("模型已保存到 mario_ppo_model.pth")

if __name__ == "__main__":
    train()
