import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gymnasium as gym
import warnings
import pdb


from .arrays import to_np
from .video import save_video, save_videos

from datasets.maze2d_loader import load_environment



def render_pointmaze(env, obs, size=256):
    maze = np.array(env.maze.maze_map, dtype=float)
    H, W = maze.shape

    fig, ax = plt.subplots(figsize=(3, 3), dpi=size // 3)
    ax.imshow(maze, cmap="gray_r", origin="upper")

   
    obs = np.asarray(obs)

    x, y = obs[0], obs[1]      

    cell_size = env.maze.maze_size_scaling

    row, col = env.maze.cell_xy_to_rowcol([x, y])
    wx, wy   = env.maze.cell_rowcol_to_xy([row, col])

    agent_x = col + (x - wx) / cell_size
    agent_y = row - (y - wy) / cell_size

    grow, gcol = env.maze.cell_xy_to_rowcol([gx, gy])
    gwx, gwy   = env.maze.cell_rowcol_to_xy([grow, gcol])

    goal_x = gcol + (gx - gwx) / cell_size
    goal_y = grow + (gy - gwy) / cell_size

    ax.scatter(agent_x, agent_y, c="red",   s=160, edgecolor="black", zorder=5)
    ax.scatter(goal_x,  goal_y,  c="green", s=160, edgecolor="black", zorder=5)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5) 
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(pad=0)
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return frame


#-----------------------------------------------------------------------------#
#------------------------------- helper structs ------------------------------#
#-----------------------------------------------------------------------------#

def env_map(env_name):
    '''
        map D4RL dataset names to custom fully-observed
        variants for rendering
    '''
    if 'maze2d' in env_name:
        return 'Maze2D-v0'
    else:
        return env_name

#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#

class SimpleMazeRenderer:
    """
    A pure-python, matplotlib-based drop-in replacement for MuJoCoRenderer.
    It keeps the same API but does NOT use mujoco, offscreen viewers, qpos, qvel, etc.
    Only works with Maze2D/PointMaze envs.
    """

    def __init__(self, env):
        if isinstance(env, str):
            env = env_map(env)
            self.env = gym.make(env)
        else:
            self.env = env

        # Preserve dims for downstream Diffuser code
        obs_space = self.env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            self.observation_dim = int(sum(np.prod(s.shape)
                                           for s in obs_space.spaces.values()))
        else:
            self.observation_dim = int(np.prod(obs_space.shape))

        self.action_dim = int(np.prod(self.env.action_space.shape))

        print("[SimpleMazeRenderer] Using headless matplotlib–based renderer.")

    # -------------------------------------------------------------
    # Core single-frame render
    # -------------------------------------------------------------
    def render(self, observation, dim=256, partial=False, qvel=True,
               render_kwargs=None, conditions=None):

        # Normalize shape
        if isinstance(dim, int):
            dim = (dim, dim)

        # We DO NOT reconstruct MuJoCo state → we just call your renderer
        frame = render_pointmaze(self.env, observation, size=dim[0])
        return frame

    # -------------------------------------------------------------
    # Frame list rendering
    # -------------------------------------------------------------
    def _renders(self, observations, **kwargs):
        imgs = []
        for obs in observations:
            imgs.append(self.render(obs, **kwargs))
        return np.stack(imgs, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        batch = self._renders(samples, **kwargs)

        # The old API returned a "composite" (one frame),
        # but Maze2D doesn't need compositing. Just return last frame.
        return batch[-1]

    # -------------------------------------------------------------
    # Composite
    # -------------------------------------------------------------
    def composite(self, savepath, paths, dim=(1024, 256), **kwargs):
        outs = []
        for path in paths:
            imgs = self._renders(path, dim=dim[0])
            outs.append(imgs[-1])

        out = np.concatenate(outs, axis=0)

        if savepath is not None:
            imageio.imsave(savepath, out)
        return out

    # -------------------------------------------------------------
    # Videos
    # -------------------------------------------------------------
    def render_rollout(self, savepath, states, fps=30, **kwargs):
        if isinstance(states, list):
            states = np.array(states)

        frames = self._renders(states)
        save_video(savepath, frames, fps=fps)

    # -------------------------------------------------------------
    # Diffuser plan renderer
    # -------------------------------------------------------------
    def render_plan(self, savepath, actions, observations_pred, state, fps=30):
        # No mujoco rollout — visualize predictions only
        frames = []
        for horizon_preds in observations_pred:
            imgs = self._renders(horizon_preds)
            frames.append(imgs)
        save_video(savepath, frames, fps=fps)

    # -------------------------------------------------------------
    # Diffusion trajectory rendering
    # -------------------------------------------------------------
    def render_diffusion(self, savepath, diffusion_path, fps=30):
        dp = to_np(diffusion_path)
        T, B, _, H, D = dp.shape

        frames = []
        for t in reversed(range(T)):
            composite = []
            for traj in dp[t]:
                imgs = self._renders(traj[:, :self.observation_dim])
                composite.append(imgs[-1])
            frames.append(np.concatenate(composite, axis=0))

        save_video(savepath, frames, fps=fps)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

