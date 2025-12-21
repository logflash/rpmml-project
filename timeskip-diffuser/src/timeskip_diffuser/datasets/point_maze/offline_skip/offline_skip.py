"""Offline dataset loading for diffusion planning with timeskips."""

import gymnasium as gym
import matplotlib.pyplot as plt
import minari
import mujoco
import numpy as np
import torch
from matplotlib.patches import Rectangle

from timeskip_diffuser.datasets.point_maze.offline_skip.traj_verifiers.verifier import (
    extract_wall_rects,
)
from timeskip_diffuser.diffuser.planner import expand_spline_from_skip_list


class OfflineSkipDataset:
    """Offline Skip Dataset"""

    def __init__(self, file_path, dataset_id="D4RL/pointmaze/umaze-v2", horizon=32):
        archive = np.load(file_path)

        if "medium" in dataset_id:
            self.axis_min = -4
            self.axis_max = 4
        else:
            self.axis_min = -2.5
            self.axis_max = 2.5

        self.dataset_id = dataset_id
        self.env_dataset = minari.load_dataset(dataset_id, download=True)
        self.wall_rects = extract_wall_rects(dataset_id)

        self.data = torch.from_numpy(archive["data"]).float()  # (N,H,3)
        self.horizon = horizon

        self.state_dim = 2
        self.action_dim = 1
        self.traj_dim = 3

        self.flat_mean = archive["flat_mean"]
        self.flat_std = archive["flat_std"]

        self.skip_mean = float(archive["skip_mean"])
        self.skip_std = float(archive["skip_std"])

        self.mean = archive["full_mean"]
        self.std = archive["full_std"]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def denormalize(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Denormalization for diffusion sampling."""
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        assert isinstance(x, np.ndarray)
        pos = x[:, :2] * self.flat_std + self.flat_mean
        skip = x[:, 2] * self.skip_std + self.skip_mean
        skip = np.expand_dims(skip, axis=1)
        print(pos.shape)
        print(skip.shape)
        return np.concatenate([pos, skip], axis=1)

    def visualize(self, skip_traj: np.ndarray, save_file: str = ""):
        """Visualize the environment while plotting a given flat trajectory."""

        pos = skip_traj[:, :2]
        skip = skip_traj[:, 2]
        skip_list = [(pos[j], float(skip[j])) for j in range(len(pos))]

        env = self.env_dataset.recover_environment()

        def unwrap_env(env) -> gym.Env:
            while hasattr(env, "env"):
                env = env.env
            return env

        env = unwrap_env(env)
        model = env.model  # type: ignore

        pos_dense, _, _ = expand_spline_from_skip_list(skip_list)
        color = "#0088ff"

        # Create figure
        _, ax = plt.subplots(figsize=(6, 6))

        # Plot the dense trajectory
        ax.scatter(
            pos_dense[:, 0],
            pos_dense[:, 1],
            s=1,
            color=color,
            alpha=0.6,
            zorder=2,
        )

        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            s=30,
            color=color,
            edgecolors="k",
            zorder=4,
        )

        # Mark start and end points
        ax.scatter(
            skip_traj[0, 0],
            skip_traj[0, 1],
            c="lime",
            s=25,
            marker="D",  # type: ignore
            edgecolors="green",
            linewidth=1,
            zorder=4,
            label="Start",
        )
        ax.scatter(
            skip_traj[-1, 0],
            skip_traj[-1, 1],
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
        ax.set_xlim(self.axis_min, self.axis_max)
        ax.set_ylim(self.axis_min, self.axis_max)

        plt.tight_layout()

        if save_file:
            plt.savefig(save_file)
        else:
            plt.show()
