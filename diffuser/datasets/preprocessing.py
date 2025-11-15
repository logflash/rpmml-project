# pointmaze_preprocessing.py
import numpy as np

# -----------------------------------------------------------------------------#
#                                general helpers                               #
# -----------------------------------------------------------------------------#

def compose(*fns):
    """
    Compose multiple preprocessing functions together.
    dataset → f1 → f2 → f3 → dataset
    """
    def _fn(x):
        for fn in fns:
            x = fn(x)
        return x
    return _fn


def get_preprocess_fn(fn_names, env):
    """
    fn_names : list of strings
    each string names a function below (e.g. "arctanh_actions")
    """
    fns = [eval(name)(env) for name in fn_names]
    return compose(*fns)


# -----------------------------------------------------------------------------#
#                           PointMaze-specific functions                        #
# -----------------------------------------------------------------------------#

def arctanh_actions(*args, **kwargs):
    """
    Convert actions from [-1,1] → (-∞,∞) using arctanh.
    Diffuser models tanh outputs, so training data should be arctanh-transformed.
    """
    epsilon = 1e-4  # avoid numerical issues near ±1

    def _fn(dataset):
        actions = dataset["actions"]
        assert actions.min() >= -1 and actions.max() <= 1, \
            f"[preprocessing] arctanh_actions received out-of-range actions!"

        clipped = np.clip(actions, -1 + epsilon, 1 - epsilon)
        dataset["actions"] = np.arctanh(clipped)
        return dataset

    return _fn


def add_deltas(env):
    """
    Compute transition deltas for PointMaze observations.

    PointMaze flattened obs format is:
        [x, y, vx, vy, achieved_x, achieved_y, goal_x, goal_y]

    We compute:
        pos_delta  = next_pos - pos
        vel_delta  = next_vel - vel
        ach_delta  = next_ach - ach
        goal_delta = next_goal - goal (usually 0)
        whole_delta = next_obs - obs

    All deltas are concatenated for richer modeling.
    """

    def _fn(dataset):
        obs  = dataset["observations"]
        next_obs = dataset["next_observations"]

        # full delta
        deltas_full = next_obs - obs

        # break into components for inspection or ablations
        pos      = obs[:, :2]
        next_pos = next_obs[:, :2]
        vel      = obs[:, 2:4]
        next_vel = next_obs[:, 2:4]
        ach      = obs[:, 4:6]
        next_ach = next_obs[:, 4:6]
        goal      = obs[:, 6:8]
        next_goal = next_obs[:, 6:8]

        deltas_pos  = next_pos  - pos
        deltas_vel  = next_vel  - vel
        deltas_ach  = next_ach  - ach
        deltas_goal = next_goal - goal  # usually zero except resets

        # concatenate everything (Diffuser likes a single vector)
        deltas = np.concatenate([
            deltas_pos,
            deltas_vel,
            deltas_ach,
            deltas_goal,
            deltas_full,
        ], axis=-1)

        dataset["deltas"] = deltas
        return dataset

    return _fn


# -----------------------------------------------------------------------------#
#                                 Example usage                                #
# -----------------------------------------------------------------------------#

if __name__ == "__main__":
    # Dummy example for sanity check

    from maze2d_loader import load_environment, sequence_dataset

    env = load_environment("PointMaze_MediumDense-v3")

    # build preprocess function pipeline
    preprocess = compose(
        arctanh_actions(env),
        add_pointmaze_deltas(env),
    )

    # take the first episode
    episode = next(sequence_dataset(env, lambda x: x))
    processed = preprocess(episode)

    print("Original obs shape:  ", episode["observations"].shape)
    print("Original act shape:  ", episode["actions"].shape)
    print("Delta shape:         ", processed["deltas"].shape)
    print("Example delta row:   ", processed["deltas"][0])
    print("✓ PointMaze preprocessing OK")
