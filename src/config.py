import torch

DEVICE = torch.device("cuda")
# "cuda" if torch.cuda.is_available() else "cpu"

# Mario 环境
ENV_NAME = "SuperMarioBros-1-1-v0"
SIMPLE_MOVEMENT = True  # True 用 SIMPLE_MOVEMENT 动作空间

# PPO 超参数
GAMMA = 0.99
LR = 2.5e-4
CLIP_EPSILON = 0.1
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 4             # 4 / 1
BATCH_SIZE = 64                # 64 / 8
ROLLOUT_STEPS = 128             # 128 / 16

# 状态处理
STACK_FRAMES = 4              # 4 / 1
RESIZE_DIM = (84, 84)        #(84, 84) / (42, 42)
