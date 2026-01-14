import random
import torch
import torch.nn as nn
import numpy as np
import os


# ===============================
# 随机数据增强
# ===============================
class RandomAugmentation:
    def __init__(self, crop_size=80, noise_std=0.01):
        self.crop_size = crop_size
        self.noise_std = noise_std

    def random_crop(self, x):
        # x: (C, H, W)
        _, H, W = x.shape
        top = random.randint(0, H - self.crop_size)
        left = random.randint(0, W - self.crop_size)
        return x[:, top:top+self.crop_size, left:left+self.crop_size]

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def __call__(self, x):
        x = self.random_crop(x)
        x = self.add_noise(x)
        return x

# ===============================
# Rollout Buffer（简化）
# ===============================
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def __len__(self):
        return len(self.states)

    def shuffle(self):
        idx = np.random.permutation(len(self))
        self.states = [self.states[i] for i in idx]
        self.actions = [self.actions[i] for i in idx]
        self.rewards = [self.rewards[i] for i in idx]

# ===============================
# 数据划分工具
# ===============================
def train_val_split(buffer, val_ratio=0.2):
    size = len(buffer)
    split = int(size * (1 - val_ratio))

    train = slice(0, split)
    val = slice(split, size)

    return train, val


def k_fold_indices(n, k):
    indices = np.arange(n)
    np.random.shuffle(indices)
    return np.array_split(indices, k)

# ===============================
# Checkpoint 工具
# ===============================
def save_checkpoint(model, optimizer, step, path="checkpoints"):
    os.makedirs(path, exist_ok=True)
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    file = os.path.join(path, f"model_step_{step}.pt")
    torch.save(ckpt, file)
    print(f"[Checkpoint saved] {file}")

def test_randomization_and_checkpoint():
    print("\n===== Randomization & Overfitting Test Start =====\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 假设你的模型接口
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(4 * 84 * 84, 256),
        nn.ReLU(),
        nn.Linear(256, 7),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    buffer = RolloutBuffer()
    augmenter = RandomAugmentation()

    # ===============================
    # 构造“伪 rollout 数据”
    # ===============================
    for _ in range(200):
        state = torch.rand(4, 84, 84)
        action = random.randint(0, 6)
        reward = random.random()
        buffer.add(state, action, reward)

    buffer.shuffle()
    print(f"Total samples collected: {len(buffer)}")

    # ===============================
    # n-fold 测试
    # ===============================
    folds = k_fold_indices(len(buffer), k=5)

    for fold_id, val_idx in enumerate(folds):
        print(f"\n--- Fold {fold_id + 1} ---")

        for step in range(10):
            idx = random.choice(val_idx)
            x = buffer.states[idx].to(device)
            y = torch.tensor(buffer.actions[idx]).to(device)

            x = augmenter(x).unsqueeze(0)

            logits = model(x)
            loss = loss_fn(logits, y.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                print(f"[Fold {fold_id+1}] step={step}, loss={loss.item():.4f}")

        save_checkpoint(model, optimizer, step=fold_id)

    print("\n✅ Randomization & checkpoint test finished\n")


if __name__ == "__main__":
    test_randomization_and_checkpoint()
