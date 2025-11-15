import numpy as np

# --------------------------------------------------------------------------- #
# Helper: convert ReplayBuffer into episodic dict-of-lists (Diffuser expects)
# --------------------------------------------------------------------------- #

def _replaybuffer_to_episodic(rb):
    """
    Convert padded ReplayBuffer arrays into episodic lists of arrays.

    Input (ReplayBuffer):
        rb[key] = (num_eps, max_path_len, dim)

    Output (episodic dict):
        episodic[key] = [ arr_ep0(T0,dim), arr_ep1(T1,dim), ... ]
    """
    n = rb.n_episodes
    lengths = rb.path_lengths   # list/array of true episode lengths

    episodic = {}
    for key, arr in rb.items():   # arr shape = (n, max_len, dim)
        episodic[key] = [
            arr[i, :lengths[i]] for i in range(n)
        ]

    return episodic


# --------------------------------------------------------------------------- #
#                           DatasetNormalizer                                 #
# --------------------------------------------------------------------------- #

class DatasetNormalizer:

    def __init__(self, dataset, normalizer, path_lengths=None):
        """
        dataset may be:
            (1) a dict mapping key -> list of variable-length episode arrays, OR
            (2) a ReplayBuffer object with padded arrays.

        This class ensures dataset is converted into episodic form BEFORE
        calling Diffuser's flatten().
        """

        # ------------------------------------------------------------
        # Detect ReplayBuffer and convert to episodic dict-of-lists
        # ------------------------------------------------------------
        if hasattr(dataset, "n_episodes") and hasattr(dataset, "path_lengths"):
            print("[Normalizer] Converting ReplayBuffer → episodic representation")
            dataset = _replaybuffer_to_episodic(dataset)

        # Now 'dataset' is episodic dict-of-lists.
        dataset = flatten(dataset, path_lengths)

        # Save dims
        self.observation_dim = dataset['observations'].shape[1]
        self.action_dim = dataset['actions'].shape[1]

        # Instantiate normalizer class
        if isinstance(normalizer, str):
            normalizer = eval(normalizer)

        self.normalizers = {}
        for key, val in dataset.items():
            try:
                self.normalizers[key] = normalizer(val)
            except Exception as e:
                print(f'[ utils/normalization ] Skipping {key} | {normalizer} | {e}')

    # ------------------------------------------------------------
    # Required by Diffuser: allow calling as fn(x, key)
    # ------------------------------------------------------------
    def __call__(self, x, key):
        return self.normalize(x, key)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

    def get_field_normalizers(self):
        return self.normalizers

    def __repr__(self):
        string = ''
        for key, normalizer in self.normalizers.items():
            string += f'{key}: {normalizer}]\n'
        return string


# -----------------------------------------------------------------------------#
#----------------------------- flatten episodic data --------------------------#
# -----------------------------------------------------------------------------#

def flatten(dataset, path_lengths):
    """
    dataset[key] = [ ep1, ep2, ... ]
    → dataset[key] = concatenated array shape (sum(T_i), dim)
    """
    flattened = {}
    for key, xs in dataset.items():
        flattened[key] = np.concatenate([
            x[:length]
            for x, length in zip(xs, path_lengths)
        ], axis=0)
    return flattened

# -----------------------------------------------------------------------------#
#-------------------------- single-field normalizers --------------------------#
# -----------------------------------------------------------------------------#

class Normalizer:
    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class DebugNormalizer(Normalizer):
    def normalize(self, x): return x
    def unnormalize(self, x): return x


class GaussianNormalizer(Normalizer):
    def __init__(self, X):
        super().__init__(X)
        self.means = self.X.mean(axis=0)
        self.stds  = self.X.std(axis=0)
        self.z = 1

    def normalize(self, x):
        return (x - self.means) / (self.stds + 1e-6)

    def unnormalize(self, x):
        return x * self.stds + self.means


class LimitsNormalizer(Normalizer):
    def normalize(self, x):
        x = (x - self.mins) / (self.maxs - self.mins + 1e-6)
        x = 2 * x - 1
        return x

    def unnormalize(self, x):
        x = np.clip(x, -1, 1)
        x = (x + 1) / 2.
        return x * (self.maxs - self.mins) + self.mins


class SafeLimitsNormalizer(LimitsNormalizer):
    def __init__(self, X, eps=1e-3):
        super().__init__(X)
        for i in range(len(self.mins)):
            if self.mins[i] == self.maxs[i]:
                print(f"[ normalization ] constant dim {i}, expanding bounds")
                self.mins[i] -= eps
                self.maxs[i] += eps

# -----------------------------------------------------------------------------#
#------------------------------- CDF normalizer -------------------------------#
# -----------------------------------------------------------------------------#

# (unchanged — omitted for space)


class CDFNormalizer1d:
    '''
        CDF normalizer for a single dimension
    '''

    def __init__(self, X):
        assert X.ndim == 1
        self.X = X.astype(np.float32)
        quantiles, cumprob = empirical_cdf(self.X)
        self.fn = interpolate.interp1d(quantiles, cumprob)
        self.inv = interpolate.interp1d(cumprob, quantiles)

        self.xmin, self.xmax = quantiles.min(), quantiles.max()
        self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def __repr__(self):
        return (
            f'[{np.round(self.xmin, 2):.4f}, {np.round(self.xmax, 2):.4f}'
        )

    def normalize(self, x):
        x = np.clip(x, self.xmin, self.xmax)
        ## [ 0, 1 ]
        y = self.fn(x)
        ## [ -1, 1 ]
        y = 2 * y - 1
        return y

    def unnormalize(self, x, eps=1e-4):
        '''
            X : [ -1, 1 ]
        '''
        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(
                f'''[ dataset/normalization ] Warning: out of range in unnormalize: '''
                f'''[{x.min()}, {x.max()}] | '''
                f'''x : [{self.xmin}, {self.xmax}] | '''
                f'''y: [{self.ymin}, {self.ymax}]'''
            )

        x = np.clip(x, self.ymin, self.ymax)

        y = self.inv(x)
        return y

def empirical_cdf(sample):
    ## https://stackoverflow.com/a/33346366

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def atleast_2d(x):
    if x.ndim < 2:
        x = x[:,None]
    return x

if __name__ == "__main__":
    from maze2d_loader import load_environment, sequence_dataset

    def identity(x): return x

    env = load_environment("PointMaze_MediumDense-v3")

    # 1) Gather all episodes first
    episodes = list(sequence_dataset(env, identity))

    # 2) Build episodic dataset for normalizer
    episodic_dataset = {
        "observations": [ep["observations"] for ep in episodes],
        "actions":      [ep["actions"] for ep in episodes],
        "rewards":      [ep["rewards"] for ep in episodes],
    }
    path_lengths = [len(ep["observations"]) for ep in episodes]

    # 3) Construct normalizer
    normalizer = DatasetNormalizer(
        episodic_dataset,
        normalizer="SafeLimitsNormalizer",
        path_lengths=path_lengths
    )

    print("\n=== NORMALIZER SUMMARY ===")
    print(normalizer)

    # 4) Test normalization
    obs = episodes[0]["observations"][:5]
    normed = normalizer.normalize(obs, "observations")
    unnorm = normalizer.unnormalize(normed, "observations")

    print("\nSample Obs:\n", obs)
    print("\nNormed:\n", normed)
    print("\nUnnormed (should match original):\n", unnorm)
    print("\n=== OK ===")
