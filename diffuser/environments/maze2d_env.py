# environments/maze2d_env.py
"""
Skeleton of a Maze2D environment for DiffuserLite-style planning.
Your partner can later populate the logic for reset(), step(), render(), etc.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Maze2DEnv(gym.Env):
    """
    Placeholder Maze2D environment.
    To be populated with simulation, collision, and goal logic later.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        size=(10, 10),
        start=(0, 0),
        goal=(9, 9),
        max_episode_steps=500,
    ):
        super().__init__()
        self.size = size
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Define placeholder observation/action spaces
        self.observation_space = spaces.Box(
            low=0.0, high=float(max(size)), shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        print("[Maze2DEnv] Initialized empty Maze2D environment (to be populated).")

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        TODO: Implement reset logic.
        """
        super().reset(seed=seed)
        self.current_step = 0
        obs = np.zeros(2, dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        """
        Simulate one timestep in the environment.
        TODO: Implement dynamics, collision checking, and rewards.
        """
        obs = np.zeros(2, dtype=np.float32)
        reward = 0.0
        done = False
        truncated = False
        info = {}
        self.current_step += 1
        return obs, reward, done, truncated, info

    def render(self):
        """
        Render environment state.
        TODO: Implement visualization.
        """
        print("[Maze2DEnv] Render not yet implemented.")
        return None

    def get_dataset(self):
        """
        Return offline dataset if available.
        TODO: Connect to dataset loader (e.g., maze2d_loader.py).
        Must return dict with:
          'observations', 'actions', 'rewards', 'terminals', 'timeouts'
        """
        raise NotImplementedError("Dataset retrieval not implemented yet.")
