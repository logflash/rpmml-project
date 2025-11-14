# pointmaze2d_dataset.py
import os
import collections
import numpy as np
import gymnasium as gym
import minari
from contextlib import contextmanager, redirect_stderr, redirect_stdout

@contextmanager
def suppress_output():
    """Redirect stdout/stderr to /dev/null."""
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull), redirect_stdout(fnull):
            yield (fnull, fnull)

with suppress_output():
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    """
    Load a PointMaze2D environment from gymnasium-robotics.
    Example IDs:
        'PointMaze_UMaze-v3', 'PointMaze_MediumDense-v3', 'PointMaze_LargeDense-v3'
    """
    if type(name) != str:
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env.spec.max_episode_steps
    env.name = name
    return env

def get_dataset(env):
    """
    Return the Minari dataset object corresponding to the PointMaze env.
    Maps Gymnasium-Robotics env names to D4RL/pointmaze datasets.
    """
    name = env.name.lower()

    mapping = {
        "pointmaze_umaze-v3": "D4RL/pointmaze/umaze-v2",
        "pointmaze_umazedense-v3": "D4RL/pointmaze/umaze-dense-v2",
        "pointmaze_medium-v3": "D4RL/pointmaze/medium-v2",
        "pointmaze_mediumdense-v3": "D4RL/pointmaze/medium-dense-v2",
        "pointmaze_large-v3": "D4RL/pointmaze/large-v2",
        "pointmaze_largedense-v3": "D4RL/pointmaze/large-dense-v2",
    }

    dataset_name = mapping.get(name)
    if dataset_name is None:
        raise ValueError(f"No known Minari dataset for environment '{env.name}'")

    with suppress_output():
        ds = minari.load_dataset(dataset_name, download=True)

    return ds



def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.

    Args:
        env: A PointMaze env.
        preprocess_fn: function that expects a D4RL-style dict and returns a dict.

    Yields episode dicts with keys:
        observations, actions, rewards, terminals, next_observations
    """
    # 1) Get the **dataset object** (Minari)
    ds = get_dataset(env)

    # 2) Flatten all episodes into a single D4RL-style dict (like env.get_dataset())
    obs_buf, act_buf, rew_buf, term_buf = [], [], [], []
    for ep in ds.iterate_episodes():
        obs_buf.append(ep.observations)
        act_buf.append(ep.actions)
        rew_buf.append(ep.rewards)
        term_buf.append(ep.terminations)

    flat = {
        "observations": np.concatenate(obs_buf),
        "actions":      np.concatenate(act_buf),
        "rewards":      np.concatenate(rew_buf),
        "terminals":    np.concatenate(term_buf),
    }

    # 3) Apply your existing preprocessing function (same as before)
    dataset = preprocess_fn(flat)

    # 4) Segment back into episodes and yield
    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)
    episode_step = 0

    for i in range(N):
        done = bool(dataset["terminals"][i])
        final_timestep = (episode_step == env.max_episode_steps - 1)

        for k in dataset:
            if "metadata" in k:
                continue
            data_[k].append(dataset[k][i])

        if done or final_timestep:
            episode_step = 0
            episode_data = {k: np.array(v) for k, v in data_.items()}
            if "pointmaze" in env.name.lower():
                episode_data = process_pointmaze2d_episode(episode_data)
            yield episode_data
            data_.clear()

        episode_step += 1

#-----------------------------------------------------------------------------#
#------------------------------ pointmaze2d fixes -----------------------------#
#-----------------------------------------------------------------------------#

def process_pointmaze2d_episode(episode):
    """Adds 'next_observations' field to a PointMaze2D episode."""
    assert "next_observations" not in episode
    next_obs = episode["observations"][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode["next_observations"] = next_obs
    return episode

#-----------------------------------------------------------------------------#
#------------------------------------- main ----------------------------------#
#-----------------------------------------------------------------------------#

if __name__ == "__main__":
    def identity_preprocess(d): return d

    env = load_environment("PointMaze_MediumDense-v3")
    print(f"Loaded env: {env.name} | max_steps={env.max_episode_steps}")

    ds_obj = get_dataset(env)  # Minari dataset object
    print(f"Minari dataset loaded with {ds_obj.total_episodes} episodes")

    for ep in sequence_dataset(env, identity_preprocess):
        print("Episode keys:", list(ep.keys()))
        print("Episode length:", len(ep["actions"]))
        print("First obs:", ep["observations"][0])
        print("First act:", ep["actions"][0])
        break

