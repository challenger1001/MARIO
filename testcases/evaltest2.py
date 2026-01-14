import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class RewardShapingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        progress_weight=0.1,
        time_penalty=0.01,
        stall_penalty=0.05
    ):
        super().__init__(env)
        self.progress_weight = progress_weight
        self.time_penalty = time_penalty
        self.stall_penalty = stall_penalty

        self.prev_x = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = info.get("x_pos", 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        x_pos = info.get("x_pos", self.prev_x)
        delta_x = x_pos - self.prev_x

        # 距离奖励（重点）
        r_progress = self.progress_weight * delta_x

        # 时间惩罚（重点）
        r_time = -self.time_penalty

        # 原地惩罚（可选）
        r_stall = -self.stall_penalty if delta_x <= 0 else 0.0

        shaped_reward = reward + r_progress + r_time + r_stall

        # 日志用
        info["reward_env"] = reward
        info["reward_progress"] = r_progress
        info["reward_time"] = r_time
        info["reward_stall"] = r_stall
        info["reward_total"] = shaped_reward

        self.prev_x = x_pos

        return obs, shaped_reward, terminated, truncated, info

def test_reward_shaping():
    print("\n===== Reward Shaping Test Start =====\n")

    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0",
        apply_api_compatibility=True
    )
    env = gym.wrappers.JoypadSpace(env, SIMPLE_MOVEMENT)

    env = RewardShapingWrapper(
        env,
        progress_weight=0.1,   # 距离奖励权重 ↑
        time_penalty=0.02,     # 时间惩罚 ↑
        stall_penalty=0.05
    )

    obs, info = env.reset()
    total_reward = 0.0

    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"Step {step:02d} | "
            f"x={info['x_pos']:4d} | "
            f"env={info['reward_env']:6.2f} | "
            f"progress={info['reward_progress']:6.2f} | "
            f"time={info['reward_time']:6.2f} | "
            f"stall={info['reward_stall']:6.2f} | "
            f"total={info['reward_total']:6.2f}"
        )

        if terminated or truncated:
            break

    print(f"\nTotal shaped reward: {total_reward:.2f}")
    env.close()
    print("\n✅ Reward shaping test finished\n")


if __name__ == "__main__":
    test_reward_shaping()
