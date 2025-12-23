"""Trajectory verification functions."""

import minari
import mujoco
import numpy as np


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
        env = env.env  # type: ignore

    # MuJoCo model
    mj_model = env.model  # type: ignore

    # Extract wall rectangles
    rects = []
    for geom_id in range(mj_model.ngeom):
        name = mujoco.mj_id2name(  # type: ignore # pylint: disable=no-member
            mj_model,
            mujoco.mjtObj.mjOBJ_GEOM,  # type: ignore # pylint: disable=no-member
            geom_id,
        )

        if name is None or "block" not in name:
            continue

        cx, cy = mj_model.geom_pos[geom_id][:2]
        hx, hy = mj_model.geom_size[geom_id][:2]

        rects.append((cx - hx, cx + hx, cy - hy, cy + hy))

    return rects


def point_in_any_wall(p, wall_rects):
    """Check if a point collides with any wall (obstacle)."""
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


def sample_free_point(wall_rects, rng, bounds=(-1.4, 1.4, -1.4, 1.4), max_tries=10_000):
    """Sample a point in free space."""

    xmin, xmax, ymin, ymax = bounds

    for _ in range(max_tries):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)

        if not point_in_any_wall((x, y), wall_rects):
            return np.array([x, y], dtype=np.float32)

    raise RuntimeError("Failed to sample free point")

#start/end eps is always 0.2
#consecutive jumps: 0.2 for open, 0.2 for umaze, 0.2 for medium
#32 for open, 32 for umaze, 64 for medium
def check_consecutive_points(traj_model, max_allowed):
    """Check if consecutive points aren't too jumpy"""
    diffs = np.linalg.norm(traj_model[1:, :2] - traj_model[:-1, :2], axis=1)
    max_jump = diffs.max()
    #print(f"Max step size: {max_jump:.4f}")
    if max_jump > max_allowed:  # Adjust threshold based on your maze
        #print("  ⚠️  WARNING: Large discontinuity detected!")
        return False
    else:
        #print("  ✓ Trajectory appears continuous")
        return True

def endpoint_within_eps(pos_dense, start_xy, goal_xy, start_eps=0.05, goal_eps=0.05):
    """Check if the endpoints are in the correct locations."""
    start_err = np.linalg.norm(pos_dense[0] - start_xy)
    goal_err = np.linalg.norm(pos_dense[-1] - goal_xy)

    return (
        start_err <= start_eps,
        goal_err <= goal_eps,
        start_err,
        goal_err,
    )
