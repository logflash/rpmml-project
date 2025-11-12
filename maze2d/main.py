from maze2d import Maze2DEnv
from planner import rrt_star
import numpy as np



def main():
    """
    maze_map = [
        [0,0,0,0,0,0,0,0,0,0],
        [0,1,1,0,0,0,1,1,1,0],
        [0,0,0,0,1,0,0,0,1,0],
        [0,1,1,1,1,1,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
    ]
    """
    """
    maze_map = np.zeros((10, 10), dtype=np.uint8) #no walls to make sure RRT* is working
    maze_map[3,3] = 1
    """
    
    maze_map = [
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,1,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0],
        [1,1,1,1,0,1,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
    ]
    
    for r in range(len(maze_map)):
        for c in range(len(maze_map[0])):
            if maze_map[r][c]:
                print((r,c))
    
    
    
    env = Maze2DEnv(maze_map)
    start = np.array([1.0, 1.0])
    goal = np.array([8.0, 8.0])
    
    path = rrt_star(env, start, goal, step_size=0.05, radius=0.25, max_iter=2000, goal_thresh=0.5)
    print("path:", path)
    
    if path is not None:
        print(f"Path found with {len(path)} waypoints.")
        env.pos, env.goal = path[0], path[-1]
        env.render_maze(path=path, save_path="rrt_star_result.png")
    else:
        print("No valid path found.")
        
        
    
    
    
    
    

if __name__ == "__main__":
    main()