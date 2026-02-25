"""Fine-tune the supervised Transformer path planner with PPO in CARLA.

Key requirements (from user):
- Actor network MUST be the existing TransformerPlannerNet (same input/output), initialized from SL checkpoint.
- Critic network is a new independent small network.
- Observation preprocessing MUST match test_nn_path_planner_control.py.
- Save best/last actor weights WITHOUT overwriting supervised checkpoints.
- Support resuming training from checkpoints.
- Train/test on the same road (fixed town/spawn/destination).

Outputs:
- checkpoints_transformer/best_rl.pt, last_rl.pt  (actor weights for closed-loop test)
- checkpoints_transformer/ppo_rl_last.zip         (SB3 checkpoint for resume)
- optionally periodic checkpoints in checkpoints_transformer/ppo_checkpoints/

Run (Windows / PowerShell):
- Train:
    python train_path_planner_rl_ppo.py --sl_checkpoint checkpoints_transformer/best.pt --total_timesteps 200000 --device cuda
- Resume:
    python train_path_planner_rl_ppo.py --resume_zip checkpoints_transformer/ppo_rl_last.zip --total_timesteps 200000 --device cuda
- Test:
    python test_nn_path_planner_control.py --checkpoint checkpoints_transformer/best_rl.pt --device cuda
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

import torch

from rl_carla_path_env import CarlaPathFollowingRLEnv, RewardConfig
from rl_transformer_policy import TransformerActorCriticPolicy


def _load_sl_args(sl_checkpoint_path: str) -> Dict[str, Any]:
    ckpt = torch.load(str(sl_checkpoint_path), map_location="cpu")
    args = ckpt.get("args", {}) or {}
    return dict(args)


def _try_import_sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback

        return PPO, BaseCallback
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "stable-baselines3 is required for PPO fine-tuning. "
            "Install a compatible version for your Python (e.g. on Python 3.7: stable-baselines3==1.8.0).\n\n"
            f"Original error: {e}"
        )


PPO, BaseCallback = _try_import_sb3()


class SaveAndEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        save_dir: Path,
        sl_checkpoint_path: str,
        eval_freq_steps: int = 5000,
        checkpoint_freq_steps: int = 20000,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sl_checkpoint_path = str(sl_checkpoint_path)

        self.eval_freq_steps = int(eval_freq_steps)
        self.checkpoint_freq_steps = int(checkpoint_freq_steps)

        self.best_mean_reward = -float("inf")

        self.best_actor_path = self.save_dir / "best_rl.pt"
        self.last_actor_path = self.save_dir / "last_rl.pt"

        self.last_sb3_path = self.save_dir / "ppo_rl_last.zip"
        self.best_sb3_path = self.save_dir / "ppo_rl_best.zip"

        self.ckpt_dir = self.save_dir / "ppo_checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _export_actor_pt(self, out_path: Path, *, mean_reward: Optional[float], tag: str) -> None:
        policy = self.model.policy

        actor = getattr(policy, "actor", None)
        if actor is None:
            raise RuntimeError("Policy has no 'actor' attribute; cannot export transformer weights")

        # Needed to rebuild TransformerPlannerNet in the closed-loop test script.
        sl_args = {}
        if hasattr(policy, "get_sl_args_for_export"):
            sl_args = policy.get_sl_args_for_export()

        ckpt = {
            "model": actor.state_dict(),
            "args": sl_args,
            "rl": {
                "tag": str(tag),
                "timesteps": int(self.num_timesteps),
                "mean_reward": None if mean_reward is None else float(mean_reward),
            },
        }
        torch.save(ckpt, str(out_path))

    def _recent_training_ep_stats(self, *, window: int = 10) -> Tuple[float, Dict[str, Any]]:
        """Use training Monitor stats as a lightweight proxy for evaluation.

        Important: CARLA cannot reliably run a separate eval env in parallel within the same
        server/world (spawn collisions at fixed spawn points). Using training ep stats avoids
        concurrent spawning.
        """

        buf = getattr(self.model, "ep_info_buffer", None)
        if not buf:
            return 0.0, {"mean_reward": 0.0, "mean_ep_len": 0.0, "n_episodes": 0}

        items = list(buf)[-int(window) :]
        rewards = [float(x.get("r", 0.0)) for x in items]
        lens = [float(x.get("l", 0.0)) for x in items]
        mean_r = float(np.mean(rewards)) if rewards else 0.0
        mean_l = float(np.mean(lens)) if lens else 0.0

        stats = {
            "mean_reward": mean_r,
            "mean_ep_len": mean_l,
            "n_episodes": int(len(items)),
        }
        return mean_r, stats

    def _maybe_save_periodic_sb3(self) -> None:
        if self.checkpoint_freq_steps <= 0:
            return
        if (self.num_timesteps % int(self.checkpoint_freq_steps)) != 0:
            return
        out = self.ckpt_dir / f"ppo_rl_step_{int(self.num_timesteps):09d}.zip"
        self.model.save(str(out))

    def _on_step(self) -> bool:
        # periodic checkpoint
        self._maybe_save_periodic_sb3()

        # periodic save (using training ep stats)
        if self.eval_freq_steps > 0 and (self.num_timesteps % int(self.eval_freq_steps) == 0):
            mean_r, stats = self._recent_training_ep_stats(window=10)

            if self.verbose:
                print(f"[eval] t={self.num_timesteps} stats={stats}")

            # Save last
            self.model.save(str(self.last_sb3_path))
            self._export_actor_pt(self.last_actor_path, mean_reward=mean_r, tag="last")

            # Save best
            if mean_r > float(self.best_mean_reward):
                self.best_mean_reward = float(mean_r)
                self.model.save(str(self.best_sb3_path))
                self._export_actor_pt(self.best_actor_path, mean_reward=mean_r, tag="best")

        return True


def main() -> None:
    p = argparse.ArgumentParser()

    # IO
    p.add_argument("--sl_checkpoint", default="checkpoints_transformer/best.pt")
    p.add_argument("--save_dir", default="checkpoints_transformer")
    p.add_argument("--resume_zip", default=None, help="Path to PPO .zip checkpoint to resume")

    # CARLA env
    p.add_argument("--town", default="Town03")
    p.add_argument("--spawn_point_index", type=int, default=0)
    p.add_argument("--destination_index", type=int, default=1)
    p.add_argument("--goal_radius", type=float, default=3.0)
    p.add_argument("--max_episode_steps", type=int, default=1000)

    p.add_argument("--target_speed", type=float, default=5.0)
    p.add_argument("--lookahead_index", type=int, default=5)

    p.add_argument("--synchronous_mode", action="store_true", default=True)
    p.add_argument("--no_synchronous_mode", action="store_false", dest="synchronous_mode")
    p.add_argument("--fixed_delta_seconds", type=float, default=0.05)

    p.add_argument("--camera_width", type=int, default=800)
    p.add_argument("--camera_height", type=int, default=600)
    p.add_argument("--camera_fov", type=float, default=90.0)
    p.add_argument("--image_timeout_s", type=float, default=1.0)

    p.add_argument(
        "--spectator_follow",
        action="store_true",
        default=False,
        help="Let Carla spectator camera follow the vehicle (visualization; may reduce fps)",
    )

    # PPO
    p.add_argument("--device", default="cuda", help="cpu/cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--n_steps", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--vf_coef", type=float, default=0.5)

    # Save/eval
    p.add_argument("--eval_freq_steps", type=int, default=5000)
    p.add_argument(
        "--n_eval_episodes",
        type=int,
        default=3,
        help="(deprecated) kept for backward compatibility; eval uses training ep stats",
    )
    p.add_argument("--checkpoint_freq_steps", type=int, default=20000)

    # Reward weights
    p.add_argument("--lane_center_w", type=float, default=2.0)
    p.add_argument("--lane_center_k", type=float, default=3.0)
    p.add_argument("--heading_w", type=float, default=1.0)
    p.add_argument("--speed_w", type=float, default=0.3)
    p.add_argument("--speed_k", type=float, default=0.8)
    p.add_argument("--steer_smooth_w", type=float, default=0.2)
    p.add_argument("--lane_change_penalty", type=float, default=-0.2)

    p.add_argument("--success_bonus", type=float, default=20.0)
    p.add_argument("--collision_penalty", type=float, default=-20.0)
    p.add_argument("--offroad_penalty", type=float, default=-20.0)
    p.add_argument("--wrong_way_penalty", type=float, default=-20.0)

    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sl_args = _load_sl_args(str(args.sl_checkpoint))
    include_state = bool(sl_args.get("use_state", False))
    state_xy_scale = float(sl_args.get("state_xy_scale", 100.0))

    reward_cfg = RewardConfig(
        lane_center_w=float(args.lane_center_w),
        lane_center_k=float(args.lane_center_k),
        heading_w=float(args.heading_w),
        speed_w=float(args.speed_w),
        speed_k=float(args.speed_k),
        steer_smooth_w=float(args.steer_smooth_w),
        lane_change_penalty=float(args.lane_change_penalty),
        success_bonus=float(args.success_bonus),
        collision_penalty=float(args.collision_penalty),
        offroad_penalty=float(args.offroad_penalty),
        wrong_way_penalty=float(args.wrong_way_penalty),
    )

    env = CarlaPathFollowingRLEnv(
        town=str(args.town),
        spawn_point_index=int(args.spawn_point_index),
        destination_index=int(args.destination_index),
        goal_radius=float(args.goal_radius),
        max_episode_steps=int(args.max_episode_steps),
        synchronous_mode=bool(args.synchronous_mode),
        fixed_delta_seconds=float(args.fixed_delta_seconds),
        target_speed=float(args.target_speed),
        lookahead_index=int(args.lookahead_index),
        image_timeout_s=float(args.image_timeout_s),
        camera_width=int(args.camera_width),
        camera_height=int(args.camera_height),
        camera_fov=float(args.camera_fov),
        spectator_follow=bool(args.spectator_follow),
        include_state=include_state,
        state_xy_scale=state_xy_scale,
        reward_cfg=reward_cfg,
    )

    callback = SaveAndEvalCallback(
        save_dir=save_dir,
        sl_checkpoint_path=str(args.sl_checkpoint),
        eval_freq_steps=int(args.eval_freq_steps),
        checkpoint_freq_steps=int(args.checkpoint_freq_steps),
    )

    # Build or resume PPO
    if args.resume_zip:
        resume_path = Path(args.resume_zip)
        if not resume_path.exists():
            raise FileNotFoundError(str(resume_path))
        model = PPO.load(str(resume_path), env=env, device=str(args.device))
        # Save immediately to unify output paths.
        model.save(str(save_dir / "ppo_rl_last.zip"))
    else:
        policy_kwargs = {
            "sl_checkpoint_path": str(args.sl_checkpoint),
            "ortho_init": False,
        }
        model = PPO(
            policy=TransformerActorCriticPolicy,
            env=env,
            device=str(args.device),
            seed=int(args.seed),
            verbose=1,
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            learning_rate=float(args.lr),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_range=float(args.clip_range),
            ent_coef=float(args.ent_coef),
            vf_coef=float(args.vf_coef),
            policy_kwargs=policy_kwargs,
        )

    try:
        model.learn(total_timesteps=int(args.total_timesteps), callback=callback)
    finally:
        # Always save last on exit.
        try:
            model.save(str(save_dir / "ppo_rl_last.zip"))
        except Exception:
            pass

        try:
            # Export transformer weights for closed-loop use.
            policy = model.policy
            actor = getattr(policy, "actor", None)
            if actor is not None:
                sl_args = policy.get_sl_args_for_export() if hasattr(policy, "get_sl_args_for_export") else {}
                torch.save({"model": actor.state_dict(), "args": sl_args, "rl": {"tag": "last", "timesteps": int(getattr(model, "num_timesteps", 0))}}, str(save_dir / "last_rl.pt"))
        except Exception:
            pass

        try:
            env.close()
        except Exception:
            pass
        # no separate eval_env (avoids concurrent CARLA spawn collisions)


if __name__ == "__main__":
    main()
