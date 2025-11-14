from gymnasium.envs.registration import register
from typing import Literal

def register_pointmaze_environments() -> Literal[True]:
    """
    Registers the simple PointMaze environments:
        - PointMaze_UMaze-v3 / PointMaze_UMazeDense-v3
        - PointMaze_Medium-v3 / PointMaze_MediumDense-v3
        - PointMaze_Large-v3 / PointMaze_LargeDense-v3

    These correspond to the single-goal mazes (no Diverse_G or Diverse_GR)
    provided by Gymnasium-Robotics.  Each environment simulates a 2D
    ball-in-maze task using MuJoCo, with either sparse or dense rewards.
    """

    # Small maze
    register(
        id="PointMaze_UMaze-v3",
        entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
        kwargs={"maze_map": "U_MAZE", "reward_type": "sparse"},
        max_episode_steps=1000,
    )
    register(
        id="PointMaze_UMazeDense-v3",
        entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
        kwargs={"maze_map": "U_MAZE", "reward_type": "Dense"},
        max_episode_steps=1000,
    )
    # Medium maze
    register(
        id="PointMaze_Medium-v3",
        entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
        kwargs={"maze_map": "MEDIUM_MAZE", "reward_type": "Sparse"},
        max_episode_steps=1000,
    )
    register(
        id="PointMaze_MediumDense-v3",
        entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
        kwargs={"maze_map": "MEDIUM_MAZE", "reward_type": "Dense"},
        max_episode_steps=1000,
    )

    # Large maze
    register(
        id="PointMaze_Large-v3",
        entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
        kwargs={"maze_map": "LARGE_MAZE", "reward_type": "Sparse"},
        max_episode_steps=1000,
    )
    register(
        id="PointMaze_LargeDense-v3",
        entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
        kwargs={"maze_map": "LARGE_MAZE", "reward_type": "Dense"},
        max_episode_steps=1000,
    )

    print("[environments] Registered PointMaze simple (Umaze, Medium, Large) environments")
    return True
