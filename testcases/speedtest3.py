import torch
import torch.nn as nn
import time


# ===============================
# Residual Block
# ===============================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


# ===============================
# 高复杂度 Actor-Critic CNN
# ===============================
class ComplexActorCritic(nn.Module):
    def __init__(self, input_channels=4, n_actions=7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=8, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            ResidualBlock(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            ResidualBlock(256),
        )

        self.flatten = nn.Flatten()

        # 假设输入为 (4, 84, 84)
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.actor(x), self.critic(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_complexity():
    print("\n===== Model Complexity Test Start =====\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟 Mario 输入 (batch=1, 4 帧, 84x84)
    dummy_input = torch.randn(1, 4, 84, 84).to(device)

    model = ComplexActorCritic(input_channels=4, n_actions=7).to(device)
    model.eval()

    # 1. 参数量
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")

    # 2. 前向 shape + 时间
    with torch.no_grad():
        start = time.time()
        actor_out, critic_out = model(dummy_input)
        elapsed = time.time() - start

    print(f"Actor output shape : {actor_out.shape}")
    print(f"Critic output shape: {critic_out.shape}")
    print(f"Forward time       : {elapsed * 1000:.2f} ms")

    # 3. device 验证
    print(f"Model device       : {next(model.parameters()).device}")

    print("\n✅ Model complexity test passed\n")


if __name__ == "__main__":
    test_model_complexity()
