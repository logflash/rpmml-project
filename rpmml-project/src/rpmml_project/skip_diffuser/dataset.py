import minari
import numpy as np
import torch
from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv
from torch.utils.data import Dataset


class MinariTrajectoryFlatDataset(Dataset):
    """Dataset for loading flat-only sequences from Minari."""

    def __init__(
        self, dataset_name="D4RL/pointmaze/umaze-v2", horizon=32, normalize=True
    ):
        self.horizon = horizon
        self.dataset = minari.load_dataset(dataset_name, download=True)

        # Extract all state trajectories
        self.trajectories = []
        self.action_traj = []
        for episode in self.dataset:
            obs = episode.observations
            act = episode.actions

            # PointMaze obs is typically dict with 'observation' and 'achieved_goal'
            if isinstance(obs, dict):
                obs = obs["observation"]

            # Store only states
            self.trajectories.append(obs)
            self.action_traj.append(act)

        # Compute normalization statistics
        all_data = np.concatenate(self.trajectories, axis=0)
        self.state_dim = obs.shape[-1]
        self.flat_dim = 2
        self.action_dim = 2
        if normalize:
            self.state_mean = all_data.mean(axis=0)
            self.state_std = all_data.std(axis=0) + 1e-8
            self.flat_mean = self.state_mean[: self.flat_dim]
            self.flat_std = self.state_std[: self.flat_dim]
        else:
            self.state_mean = np.zeros(self.state_dim)
            self.state_std = np.ones(self.state_dim)
            self.flat_mean = np.zeros(self.flat_dim)
            self.flat_std = np.ones(self.flat_dim)

        # Create indices for sampling
        self.indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            for t in range(len(traj) - horizon + 1):
                self.indices.append((traj_idx, t))

    def get_env(self) -> PointMazeEnv:
        """Return an unwrapped environment associated with this dataset"""
        env = self.dataset.recover_environment()

        # Unwrap the environment recursively.
        def unwrap_env(env):
            while hasattr(env, "env"):
                env = env.env
            return env

        env = unwrap_env(env)
        assert isinstance(env, PointMazeEnv)
        return env

    def normalize_flat(self, x):
        """Normalize flat output (x, y)"""
        return (x - self.flat_mean) / self.flat_std

    def normalize_full(self, x):
        """Normalize full state (x, y, vx, vy)"""
        return (x - self.state_mean) / self.state_std

    def denormalize_flat(self, x):
        """Denormalize flat output (x, y)"""
        return x * self.flat_std + self.flat_mean

    def denormalize_full(self, x):
        """Normalize full state (x, y, vx, vy)"""
        return x * self.state_std + self.state_mean

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_t = self.indices[idx]
        full_states = self.trajectories[traj_idx][start_t : start_t + self.horizon]

        # Extract only flat outputs (x, y)
        flat_trajectory = full_states[:, : self.flat_dim]
        flat_trajectory = self.normalize_flat(flat_trajectory)

        return torch.FloatTensor(flat_trajectory)

    def get_full_unnormalized_state(self, idx):
        """Get the full state trajectory associated with a dataset index."""
        traj_idx, start_t = self.indices[idx]
        full_states = self.trajectories[traj_idx][start_t : start_t + self.horizon]
        return torch.FloatTensor(full_states)

    def get_full_unnormalized_action(self, idx):
        """Get the full action trajectory associated with a dataset index."""
        traj_idx, start_t = self.indices[idx]
        full_states = self.action_traj[traj_idx][start_t : start_t + self.horizon]
        return torch.FloatTensor(full_states)
