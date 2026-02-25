"""Stable-Baselines3 PPO custom policy: actor=pretrained TransformerPlannerNet, critic=small CNN.

Notes:
- Actor output is unchanged: 15 points (x,y) in vehicle frame.
- PPO exploration is implemented as a diagonal Gaussian around actor mean (learnable log_std).
- Critic is independent (does not share weights with the actor), per project requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn

from nn_path_planner.models_transformer import TransformerPlannerNet


@dataclass
class TransformerArchArgs:
    d_model: int = 256
    nhead: int = 8
    enc_layers: int = 4
    dec_layers: int = 4
    use_state: bool = False
    state_xy_scale: float = 100.0


def build_transformer_from_sl_ckpt(ckpt: Dict[str, Any]) -> Tuple[TransformerPlannerNet, Dict[str, Any]]:
    args = ckpt.get("args", {}) or {}
    use_state = bool(args.get("use_state", False))

    model = TransformerPlannerNet(
        num_points=15,
        d_model=int(args.get("d_model", 256)),
        nhead=int(args.get("nhead", 8)),
        num_encoder_layers=int(args.get("enc_layers", 4)),
        num_decoder_layers=int(args.get("dec_layers", 4)),
        state_dim=(4 if use_state else 0),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    return model, args


class ImageValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, obs_img: torch.Tensor) -> torch.Tensor:
        x = self.net(obs_img)
        v = self.head(x)
        return v.view(-1)


# SB3 import is optional at module import time; policy class only used when SB3 is installed.
try:
    from stable_baselines3.common.policies import ActorCriticPolicy
except Exception:  # pragma: no cover
    ActorCriticPolicy = object  # type: ignore


class TransformerActorCriticPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy with custom actor/critic.

    Actor:
      - TransformerPlannerNet (pretrained via supervised learning)
      - outputs mean action (B,30)
    Critic:
      - small independent CNN -> scalar value

    Distribution:
      - Diagonal Gaussian with learnable log_std.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *,
        sl_checkpoint_path: str,
        **kwargs,
    ):
        self._sl_checkpoint_path = str(sl_checkpoint_path)
        self._sl_args: Dict[str, Any] = {}
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        # Load SL checkpoint
        ckpt_path = Path(self._sl_checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(str(ckpt_path))

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        actor, sl_args = build_transformer_from_sl_ckpt(ckpt)
        self._sl_args = dict(sl_args)

        self.actor: TransformerPlannerNet = actor
        self.critic: nn.Module = ImageValueNet()

        # Diagonal Gaussian std
        action_dim = int(np.prod(self.action_space.shape))
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

        # Optimizer (SB3 sets optimizer_class/kwargs)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    @staticmethod
    def _split_obs(obs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # SB3 may pass a Tensor (Box obs) or a dict of tensors (Dict obs)
        if isinstance(obs, dict):
            image = obs.get("image")
            state = obs.get("state")
            if image is None:
                raise ValueError("Dict observation missing key 'image'")
            return image, state
        return obs, None

    def _actor_mean(self, obs: Any) -> torch.Tensor:
        image, state = self._split_obs(obs)
        image = image.float()

        if getattr(self.actor, "state_dim", 0) and int(getattr(self.actor, "state_dim")) > 0:
            if state is None:
                raise ValueError("Actor requires state but observation provided no 'state'")
            out = self.actor(image, state=state.float())
        else:
            out = self.actor(image)

        points = out["points"]  # (B,15,2)
        return points.reshape(points.shape[0], -1)

    def _dist(self, mean_actions: torch.Tensor) -> torch.distributions.Normal:
        std = torch.exp(self.log_std).to(mean_actions.device)
        return torch.distributions.Normal(mean_actions, std)

    @staticmethod
    def _log_prob(dist: torch.distributions.Normal, actions: torch.Tensor) -> torch.Tensor:
        # Sum over action dims
        return dist.log_prob(actions).sum(dim=1)

    @staticmethod
    def _entropy(dist: torch.distributions.Normal) -> torch.Tensor:
        return dist.entropy().sum(dim=1)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        mean_actions = self._actor_mean(obs)
        dist = self._dist(mean_actions)
        actions = mean_actions if deterministic else dist.rsample()
        log_prob = self._log_prob(dist, actions)
        image, _ = self._split_obs(obs)
        values = self.critic(image.float())
        return actions, values, log_prob

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean_actions = self._actor_mean(observation)
        if deterministic:
            return mean_actions
        dist = self._dist(mean_actions)
        return dist.rsample()

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        mean_actions = self._actor_mean(obs)
        dist = self._dist(mean_actions)
        log_prob = self._log_prob(dist, actions)
        entropy = self._entropy(dist)
        image, _ = self._split_obs(obs)
        values = self.critic(image.float())
        return values, log_prob, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        image, _ = self._split_obs(obs)
        return self.critic(image.float())

    def get_sl_args_for_export(self) -> Dict[str, Any]:
        """Args needed to rebuild TransformerPlannerNet for closed-loop test script."""
        return dict(self._sl_args)

