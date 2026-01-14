import time
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def test_render_window():
    print("\n===== Render Window Compatibility Test =====\n")

    # 1. 创建环境（不包任何 RL 逻辑）
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0",
        apply_api_compatibility=True,
        render_mode="human"
    )

    env = gym.wrappers.JoypadSpace(env, SIMPLE_MOVEMENT)

    obs, info = env.reset()
    print("Environment reset OK")

    # 2. 渲染测试
    print("Rendering window... (5 seconds)")
    start = time.time()

    while time.time() - start < 5:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        time.sleep(0.05)  # 控制刷新速率

        if terminated or truncated:
            env.reset()

    env.close()
    print("\n✅ Render window test passed (window should be visible)\n")


if __name__ == "__main__":
    test_render_window()
