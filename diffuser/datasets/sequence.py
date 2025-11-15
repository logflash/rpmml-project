from collections import namedtuple
import numpy as np
import torch
import pdb

# Try relative imports (package mode)
# man idk this is a mish mash of imports but it was working when i was calling the script 
# but not when running main and this fixed it let this shitty code be pls
try:
    from . import preprocessing
    from .preprocessing import get_preprocess_fn
    from .preprocessing import *
    from .maze2d_loader import load_environment, sequence_dataset
    from .normalization import DatasetNormalizer
    from .buffer import ReplayBuffer

# Fallback to absolute imports (script mode)
except ImportError:
    import preprocessing
    from preprocessing import get_preprocess_fn
    from preprocessing import *
    from maze2d_loader import load_environment, sequence_dataset
    from normalization import DatasetNormalizer
    from buffer import ReplayBuffer

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='maze2d', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env) 
        self.env.reset(seed=seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn) 

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon) # this is fine

        self.observation_dim = fields.observations.shape[-1] # this is fine
        self.action_dim = fields.actions.shape[-1] # this is fine
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths # this is fine
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch


class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch

if __name__ == "__main__":
    import pprint

    print("\n=== Testing SequenceDataset ===")

    # 1. Create the dataset
    ds = SequenceDataset(
        env="PointMaze_MediumDense-v3",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=["add_deltas"],
        max_path_length=600,
        max_n_episodes=5000,
        use_padding=True,
    )

    print("\n=== Dataset Loaded ===")
    print(f" Episodes: {ds.n_episodes}")
    print(f" Observation dim: {ds.observation_dim}")
    print(f" Action dim: {ds.action_dim}")
    print(f" Total indices (samples): {len(ds)}")

    # Show ReplayBuffer summary (fields shapes)
    print("\n=== ReplayBuffer Fields ===")
    for k, v in ds.fields.items():
        try:
            print(f"{k:20s}: {tuple(v.shape)}")
        except:
            pass

    # 2. Fetch one dataset sample
    print("\n=== Testing __getitem__ ===")
    batch = ds[0]
    print("Trajectories shape:", batch.trajectories.shape)  # (H, A+O)
    print("Conditions:")
    pprint.pprint(batch.conditions)

    # 3. Test GoalDataset
    print("\n=== Testing GoalDataset ===")
    gds = GoalDataset(
        env="PointMaze_MediumDense-v3",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=["add_deltas"],
        max_path_length=600,
        max_n_episodes=5000,
        use_padding=True,
    )
    gsample = gds[10]
    print("Goal conditions:", gsample.conditions)

    # 4. Test ValueDataset
    print("\n=== Testing ValueDataset ===")
    vds = ValueDataset(
        env="PointMaze_MediumDense-v3",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=["add_deltas"],
        max_path_length=600,
        max_n_episodes=5000,
        use_padding=True,
        discount=0.99,
    )
    vsample = vds[20]
    print("Value sample trajectories shape:", vsample.trajectories.shape)
    print("Value:", vsample.values)

    print("\n=== All tests finished successfully ===")
