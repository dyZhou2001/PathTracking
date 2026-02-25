from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNNBackbone(nn.Module):
    """A small CNN backbone to avoid external deps (torchvision).

    Input: (B,3,H,W) where H=W=224 recommended.
    Output: feature map (B,C,h,w)
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        def block(cin: int, cout: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 4

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = block(c1, c2, stride=2)  # 56 -> 28
        self.stage2 = block(c2, c3, stride=2)  # 28 -> 14
        self.stage3 = block(c3, c4, stride=2)  # 14 -> 7

        self.out_channels = c4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class BaselinePlannerNet(nn.Module):
    """CNN encoder + MLP head to predict N points (x,y) and remaining length."""

    def __init__(
        self,
        *,
        num_points: int = 15,
        backbone_channels: int = 64,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_points = int(num_points)

        self.backbone = SimpleCNNBackbone(base_channels=int(backbone_channels))
        c = self.backbone.out_channels

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(c, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.points_out = nn.Linear(hidden_dim, self.num_points * 2)
        self.length_out = nn.Linear(hidden_dim, 1)

    def forward(self, image: torch.Tensor) -> dict:
        feat = self.backbone(image)  # (B,C,h,w)
        pooled = self.pool(feat).flatten(1)  # (B,C)
        h = self.head(pooled)  # (B,H)

        pts = self.points_out(h).view(-1, self.num_points, 2)
        rem = self.length_out(h).view(-1)

        return {"points": pts, "remaining_length_m": rem}
