import mujoco
import minari
import numpy as np

#LAST CELL OF SKIP_INDEPENDENT.IPYNB HAS DEBUG FOR THIS MODULE
#IT PLOTS RED CIRCLE AT FIRST COLLISION FOR GIVEN TRAJECTORY

#Pointmaze-agnostic verifier -> just put in correct wall_rects list for given env and call
#verify_trajectory_dense; this assumes a dense trajectory as it doesn't check interpolated
#positions for collisions, but if traj is very dense, then no need to
def extract_wall_rects(dataset_name):
    """
    Given a Minari dataset name (e.g. 'D4RL/pointmaze/umaze-v2'),
    load the environment and return wall rectangles:
    [(xmin, xmax, ymin, ymax), ...]
    """

    # Load dataset
    dataset = minari.load_dataset(dataset_name)

    # Recover environment
    env = dataset.recover_environment()

    # Unwrap gym / wrappers
    while hasattr(env, "env"):
        env = env.env

    # MuJoCo model
    mj_model = env.model

    # Extract wall rectangles
    rects = []
    for geom_id in range(mj_model.ngeom):
        name = mujoco.mj_id2name(
            mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )

        if name is None or "block" not in name:
            continue

        cx, cy = mj_model.geom_pos[geom_id][:2]
        hx, hy = mj_model.geom_size[geom_id][:2]

        rects.append((
            cx - hx, cx + hx,
            cy - hy, cy + hy
        ))

    return rects


def point_in_any_wall(p, wall_rects):
    x, y = p
    for xmin, xmax, ymin, ymax in wall_rects:
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True
    return False

def verify_trajectory_dense(pos_dense, wall_rects):
    """
    Returns:
        feasible (bool)
        collision_index (int or None)
    """
    for i, p in enumerate(pos_dense):
        if point_in_any_wall(p, wall_rects):
            return False, i
    return True, None



#for experiments
def sample_free_point(wall_rects,rng, bounds=(-1.4, 1.4, -1.4, 1.4), max_tries=10_000):
    
    xmin, xmax, ymin, ymax = bounds

    for _ in range(max_tries):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)

        if not point_in_any_wall((x, y), wall_rects):
            return np.array([x, y], dtype=np.float32)

    raise RuntimeError("Failed to sample free point")



def endpoint_within_eps(pos_dense, start_xy, goal_xy, start_eps=0.05, goal_eps=0.05):
    start_err = np.linalg.norm(pos_dense[0]  - start_xy)
    goal_err  = np.linalg.norm(pos_dense[-1] - goal_xy)

    return (
        start_err <= start_eps,
        goal_err  <= goal_eps,
        start_err,
        goal_err,
    )
