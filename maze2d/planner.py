#RRT* implementation
#RRT* works better than A* in this continous setting in my opinion
#Even if I tried to make A* more fine grained, it would've still followed coarse
#horizontal/vertical plans ... I think...?
import numpy as np, math, random

class Node:
    def __init__(self, pos, parent=None):
        self.pos = np.array(pos, dtype=np.float32)
        self.parent = parent #for backtracking to get path
        self.cost = 0.0 if parent is None else parent.cost + np.linalg.norm(pos - parent.pos)

def rrt_star(env, start, goal, step_size=0.5, radius=1.0, max_iter=5000, goal_thresh=0.5):
    
    #check if straight line between two points is valid (free of walls)
    def collision_free(p1, p2):
        dist = np.linalg.norm(p2 - p1)
        steps = max(2, int(dist / 0.05)) 
        for t in np.linspace(0, 1, steps):
            p = p1 + t * (p2 - p1)
            if env._is_wall(p):
                return False
        return True

    nodes = [Node(np.array(start, dtype=np.float32))]
    best_goal_node = None
    best_cost = float('inf')

    for _ in range(max_iter):
        #sample random point or goal w/ 10% chance
        if random.random() < 0.1:
            rnd = np.array(goal, dtype=np.float32)
        else:
            rnd = np.array([random.uniform(0, env.w), random.uniform(0, env.h)], dtype=np.float32)
            if env._is_wall(rnd):  
                continue

        #find nearest node in the tree
        nearest = min(nodes, key=lambda n: np.linalg.norm(rnd - n.pos)) 
        direction = rnd - nearest.pos
        dist = np.linalg.norm(direction)
        if dist == 0:
            continue
        new_pos = nearest.pos + (direction / dist) * min(step_size, dist)

        if env._is_wall(new_pos) or not collision_free(nearest.pos, new_pos):
            continue

        new_node = Node(new_pos, nearest)
        new_node.cost = nearest.cost + np.linalg.norm(new_pos - nearest.pos)

        #rewire nearby nodes
        for n in nodes:
            if np.linalg.norm(n.pos - new_pos) < radius and collision_free(n.pos, new_pos):
                new_cost = new_node.cost + np.linalg.norm(n.pos - new_pos)
                if new_cost < n.cost:
                    n.parent = new_node
                    n.cost = new_cost

        nodes.append(new_node)

        # Check if goal is reached (keep searching for better path)
        if np.linalg.norm(new_node.pos - goal) < goal_thresh:
            if new_node.cost < best_cost:
                best_goal_node = new_node
                best_cost = new_node.cost

    # Return after max iterations -> theoretical optimality
    if best_goal_node is not None:
        path = []
        cur = best_goal_node
        while cur is not None:
            path.append(cur.pos)
            cur = cur.parent
        return np.array(path[::-1])

    return None
