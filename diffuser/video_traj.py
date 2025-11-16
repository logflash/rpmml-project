"""
Export a Minari PointMaze trajectory to a video using SimpleMazeRenderer.
"""

import numpy as np
from datasets.maze2d_loader import load_environment, sequence_dataset
from utils.rendering import SimpleMazeRenderer
from utils.video import save_video

#Load a PointMaze environment & Minari dataset,
#select a single trajectory, and render a video of its sequence.
def export_minari_trajectory_video(env_name: str, episode_index: int = 100, 
                                   out_path: str = "trajectory.mp4",  fps: int = 30, render_dim: int = 256,):
    



    #obs: [ pos_x, pos_y, vel_x, vel_y, achieved_goal_x, achieved_goal_y, desired_goal_x, desired_goal_y ]
    #achieved goal is same as desired_goal in maze2d but in general achieved goal is current state in
    #terms of desired goal space
    # load environment + renderer
    env = load_environment(env_name)
    renderer = SimpleMazeRenderer(env)

    # load Minari dataset episodes (generator)
    print(f"[Loading Minari Dataset] {env_name}")
    episodes = list(sequence_dataset(env))

    if episode_index >= len(episodes):
        raise ValueError(f"Episode index {episode_index} out of range ({len(episodes)} total).")

    ep = episodes[episode_index]
    print(f"Start: {ep['observations'][0][0:2]}")
    print(f"Goal: {ep['observations'][0][6:8]}")
    
    #I'm pretty sure top left coordinates are [-max num, max_num] (pointmaze is centered middle (0,0))
    obs_seq = ep["observations"] # shape (T, obs_dim)
    print(f"[Episode] length T={len(obs_seq)}")

    #roll the environment forward to match dataset starting state
    
    
    #Minari observations already encode qpos (pos, vel, achieved_goal, desired_goal)
    #so we simply feed each obs into the renderer.

    frames = []
    for t, obs in enumerate(obs_seq):
        frame = renderer.render(obs, dim=render_dim)
        frames.append(frame)

    frames = np.array(frames)

    #save video
    save_video(out_path, frames, fps=fps)
    print(f"[Saved video] {out_path} ({len(frames)} frames)")


#run script
if __name__ == "__main__":
    export_minari_trajectory_video(
        env_name="PointMaze_MediumDense-v3",
        episode_index=0,
        out_path="demo_trajectory.mp4",
        fps=30,
        render_dim=256,
    )