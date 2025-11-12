import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import matplotlib.pyplot as plt

#Agent lives in continuous coordinate space R^2
#Maze is defined as a discrete occupancy grid
#where walls have thickness 1 


class Maze2DEnv(gym.Env):
    def __init__(self, maze_map, step_size=0.1, max_steps=200):
        super().__init__()
        self.maze = np.array(maze_map, dtype=np.uint8)  # 1 = wall, 0 = open
        self.h, self.w = self.maze.shape
        self.step_size = step_size
        self.max_steps = max_steps
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) #spaces.Box is 2D continuous vector
        self.observation_space = spaces.Box(low=0, high=max(self.w, self.h), shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.goal = np.array(options.get("goal", [self.w-1, self.h-1]), dtype=np.float32)
        self.pos = np.array(options.get("start", [1., 1.]), dtype=np.float32)
        self.steps = 0
        return self.pos.copy(), {}

    def step(self, action):
        dx, dy = np.clip(action, -1, 1) * self.step_size
        new_pos = self.pos + np.array([dx, dy], dtype=np.float32)
        if not self._is_wall(new_pos):
            self.pos = new_pos
        self.steps += 1
        done = np.linalg.norm(self.pos - self.goal) < 0.2 or self.steps >= self.max_steps
        reward = -np.linalg.norm(self.pos - self.goal)
        return self.pos.copy(), reward, done, False, {}

    def _is_wall(self, p):
        row, col = p 
        if row < 0 or col < 0 or row >= self.h or col >= self.w:
            return True
        r, c = int(np.floor(row)), int(np.floor(col))
        return self.maze[r, c] == 1

    #text debugging
    def render(self):
        grid = np.copy(self.maze).astype(str)
        gx, gy = int(self.goal[0]), int(self.goal[1])
        px, py = int(self.pos[0]), int(self.pos[1])
        grid[gy, gx] = 'G'
        grid[py, px] = 'A'
        print("\n".join("".join(row) for row in grid[::-1]))
        
    
    #view path
    def render_maze(self, path=None, save_path=None):
        h, w = self.maze.shape
        fig, ax = plt.subplots(figsize=(6, 6))

       
        ax.imshow(self.maze, cmap="gray_r", origin="upper", extent=[0, w, 0, h])

      
        for x in range(w + 1):
            ax.axvline(x, color='black', linewidth=0.5, alpha=0.3)
        for y in range(h + 1):
            ax.axhline(y, color='black', linewidth=0.5, alpha=0.3)

  
        if path is not None:
            path = np.array(path)
           
            ax.plot(path[:, 1], h - path[:, 0],
                    color="blue", linewidth=2, marker='o', markersize=3, label="Path", zorder=2)

       
        ax.scatter(self.pos[1], h - self.pos[0],
                c="green", s=100, label="Start", edgecolors='black', zorder=3)
        ax.scatter(self.goal[1], h - self.goal[0],
                c="red", s=100, label="Goal", edgecolors='black', zorder=3)

       
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(w))
        ax.set_yticks(np.arange(h))
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)
        ax.legend()
        plt.tight_layout()

     
        if save_path:
            plt.savefig(save_path, dpi=200)
            plt.close()
        else:
            plt.show()
