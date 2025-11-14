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

import os
import collections
import numpy as np
import gymnasium as gym
import minari
from contextlib import contextmanager, redirect_stderr, redirect_stdout

#-----------------------------------------------------------------------------#
#------------------------- helper: suppress noisy output ---------------------#
#-----------------------------------------------------------------------------#

@contextmanager
def suppress_output():
    """Redirect stdout/stderr to /dev/null."""
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull), redirect_stdout(fnull):
            yield (fnull, fnull)

#-----------------------------------------------------------------------------#
#--------------------------- 1. Load environment -----------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    """
    Load a Gymnasium-Robotics PointMaze environment by name.
    Returns an unwrapped environment with metadata set, just like Diffuser expects.
    """
    if not isinstance(name, str):
        return name

    with suppress_output():
        wrapped_env = gym.make(name)

    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env.spec.max_episode_steps
    env.name = name
    return env

#-----------------------------------------------------------------------------#
#---------------------------- 2. Get Minari dataset --------------------------#
#-----------------------------------------------------------------------------#

def get_dataset(env):
    """
    Map a Gymnasium-Robotics PointMaze env to its corresponding Minari dataset.
    Returns a MinariDataset object (episodic data container).
    """
    name = env.name.lower()

    mapping = {
        "pointmaze_umaze-v3":        "D4RL/pointmaze/umaze-v2",
        "pointmaze_umazedense-v3":   "D4RL/pointmaze/umaze-dense-v2",
        "pointmaze_medium-v3":       "D4RL/pointmaze/medium-v2",
        "pointmaze_mediumdense-v3":  "D4RL/pointmaze/medium-dense-v2",
        "pointmaze_large-v3":        "D4RL/pointmaze/large-v2",
        "pointmaze_largedense-v3":   "D4RL/pointmaze/large-dense-v2",
    }

    dataset_name = mapping.get(name)
    if dataset_name is None:
        raise ValueError(f"No known Minari dataset for environment '{env.name}'")

    with suppress_output():
        ds = minari.load_dataset(dataset_name, download=True)

    return ds  # keep episodic structure for sequence_dataset




def sequence_dataset(env, preprocess_fn=lambda x: x):
    """
    Returns an iterator over episode dictionaries compatible with Diffuser.
    Automatically flattens goal-conditioned observations into D4RL-like format.
    """

    ds = get_dataset(env)
    print(f"[Minari] Loaded dataset: {ds.spec.dataset_id}  |  {ds.total_episodes} episodes")

    for episode in ds.iterate_episodes():
        obs = episode.observations
        acts = episode.actions
        rews = episode.rewards
        T = len(rews)

        # Handle dict-style observations (goal-conditioned)
        if isinstance(obs, dict):
            # order keys consistently: observation, achieved_goal, desired_goal
            obs_flat = np.concatenate(
                [obs[k] for k in ["observation", "achieved_goal", "desired_goal"]],
                axis=-1,
            )
        else:
            obs_flat = obs

        next_obs_flat = np.concatenate([obs_flat[1:], obs_flat[-1:]], axis=0)

        terminals = np.zeros(T, dtype=bool)
        terminals[-1] = True
        timeouts = np.zeros(T, dtype=bool)

        ep_data = {
            "observations": obs_flat,
            "actions": acts,
            "rewards": rews,
            "terminals": terminals,
            "timeouts": timeouts,
            "next_observations": next_obs_flat,
        }

        ep_data = preprocess_fn(ep_data)
        yield ep_data


#-----------------------------------------------------------------------------#
#------------------------------ pointmaze2d fixes -----------------------------#
#-----------------------------------------------------------------------------#
if __name__ == "__main__":

    def identity_preprocess(d): 
        return d

    # Load environment
    env = load_environment("PointMaze_MediumDense-v3")
    print(f"Loaded env: {env.name} | max_steps={env.max_episode_steps}")

    # Load dataset
    ds_obj = get_dataset(env)
    print(f"Minari dataset loaded with {ds_obj.total_episodes} episodes")

    # Iterate through one episode to inspect
    for ep in sequence_dataset(env, identity_preprocess):
        print("\nEpisode keys:", list(ep.keys()))
        print("Episode length:", len(ep["actions"]))

        print("\nFirst 5 observations (flattened):")
        for i in range(5):
            o = ep["observations"][i]
            print(
                f"t={i}: pos=({o[0]:.3f}, {o[1]:.3f}), vel=({o[2]:.3f}, {o[3]:.3f}), "
                f"ach=({o[4]:.3f}, {o[5]:.3f}), goal=({o[6]:.3f}, {o[7]:.3f})"
            )

        print("\nFirst 5 actions (accelerations):")
        print(ep["actions"][:5])

        print("\nFirst 5 rewards:")
        print(ep["rewards"][:5])
        break