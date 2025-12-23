"""
Datasets from the U-Maze environment in Minari.
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import minari
import mujoco
import numpy as np
import torch
from matplotlib.patches import Rectangle
from torch.utils.data import Dataset


class UMazeFlatDataset(Dataset):
    """
    Dataset for U-Maze (v2) with the flattened (position-only) state.
    Used to test Diffuser under the differential flatness assumption.
    """

    def __init__(self, horizon=32):
        self.horizon = horizon
        self.dataset = minari.load_dataset("D4RL/pointmaze/umaze-v2", download=True)

        self.trajectories = []
        for episode in self.dataset:
            obs = episode.observations
            if isinstance(obs, dict):
                obs = obs["observation"]
            self.trajectories.append(obs[:, :2])

        self.state_dim = 2
        all_data = np.concatenate(self.trajectories, axis=0)
        self.mean = all_data.mean(axis=0)
        self.std = all_data.std(axis=0) + 1e-8

        self.indices = []
        for traj_idx, trajectory in enumerate(self.trajectories):
            for t in range(len(trajectory) - horizon + 1):
                self.indices.append((traj_idx, t))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalization for diffusion training."""
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Denormalization for diffusion sampling."""
        return x * self.std + self.mean

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx) -> torch.Tensor:
        traj_idx, start_t = self.indices[idx]
        trajectory = self.trajectories[traj_idx][start_t : start_t + self.horizon]
        return torch.FloatTensor(self.normalize(trajectory))

    def visualize(self, flat_traj: np.ndarray, save_file: str = ""):
        """Visualize the U-Maze environment while plotting a given flat trajectory."""

        env = self.dataset.recover_environment()

        def unwrap_env(env) -> gym.Env:
            while hasattr(env, "env"):
                env = env.env
            return env

        env = unwrap_env(env)
        model = env.model  # type: ignore

        # Create figure
        _, ax = plt.subplots(figsize=(6, 6))

        # Plot the trajectory
        if flat_traj is not None:
            ax.scatter(
                flat_traj[:, 0],
                flat_traj[:, 1],
                s=30,
                c="#0088ff",
                edgecolors="k",
                zorder=4,
            )

            # Mark start and end points
            ax.scatter(
                flat_traj[0, 0],
                flat_traj[0, 1],
                c="lime",
                s=25,
                marker="D",  # type: ignore
                edgecolors="green",
                linewidth=1,
                zorder=4,
                label="Start",
            )
            ax.scatter(
                flat_traj[-1, 0],
                flat_traj[-1, 1],
                c="red",
                s=50,
                marker="8",  # type: ignore
                edgecolors="darkred",
                linewidth=1,
                zorder=4,
                label="End",
            )

        # Render MuJoCo walls
        for geom_id in range(model.ngeom):
            # Get geom name
            name = mujoco.mj_id2name(  # pylint: disable=no-member # type: ignore
                model,
                mujoco.mjtObj.mjOBJ_GEOM,  # pylint: disable=no-member # type: ignore
                geom_id,
            )

            if name is None or "block" not in name:
                continue

            cx, cy = model.geom_pos[geom_id][:2]  # center
            hx, hy = model.geom_size[geom_id][:2]  # half-extents

            rect = Rectangle(
                (cx - hx, cy - hy),
                2 * hx,
                2 * hy,
                facecolor="black",
                alpha=0.35,
                zorder=0,
            )
            ax.add_patch(rect)

        # Final figure styling
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_title("PointMaze Trajectory", fontsize=14)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)

        plt.tight_layout()

        if save_file:
            plt.savefig(save_file)
        else:
            plt.show()
