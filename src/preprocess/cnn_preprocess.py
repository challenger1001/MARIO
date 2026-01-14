from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn


class CNNPreprocess(nn.Module):
    """
    Use CNN to process input frame

    Input:  (B, 3, H, W)  e.g. (B, 3, 224, 224)
    Output: logits (B, num_classes)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 224 -> 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 112 -> 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 56 -> 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),# 28 -> 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),               # -> (B, 256, 1, 1)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ----------------------------
# Helper: build + (optional) load weights
# ----------------------------
@dataclass
class CNNConfig:
    num_classes: int
    weight_path: Optional[str] = None
    device: Optional[str] = None  # "cuda" or "cpu"


def build_cnn(cfg: CNNConfig) -> nn.Module:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LevelCNN(num_classes=cfg.num_classes).to(device)
    model.eval()

    if cfg.weight_path:
        ckpt = torch.load(cfg.weight_path, map_location=device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=True)

    return model
