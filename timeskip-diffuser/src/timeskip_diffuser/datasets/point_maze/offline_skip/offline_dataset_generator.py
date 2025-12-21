from pathlib import Path

import minari
import numpy as np
from tqdm import tqdm

from timeskip_diffuser.datasets.point_maze.offline_skip.traj_verifiers.verifier import (
    extract_wall_rects,
    verify_trajectory_dense,
)
from timeskip_diffuser.diffuser.planner import expand_spline_from_skip_list

# ============================================================
#                 OFFLINE DATASET GENERATOR
# ============================================================


class OfflineIndependentSkipDatasetBuilder:
    """
    Builds an offline dataset of VALID coarse (pos, skip) trajectories.

    Pipeline:
        coarse (pos, skip)
            ↓
        spline reconstruction
            ↓
        collision check
            ↓
        SAVE coarse only
            ↓
        normalize after collection
    """

    def __init__(
        self,
        dataset_name,
        horizon,
        samples_per_trajectory,
        max_rejection_attempts,
        spline_func,
        collision_checker,
        lognormal_mu=1.0,
        lognormal_sigma=1.0,
        starts_per_skip=4,
    ):
        self.dataset_name = dataset_name
        self.horizon = horizon
        self.samples_per_trajectory = samples_per_trajectory
        self.max_rejection_attempts = max_rejection_attempts

        self.spline_func = spline_func
        self.collision_checker = collision_checker

        self.log_mu = lognormal_mu
        self.log_sigma = lognormal_sigma
        self.starts_per_skip = starts_per_skip

        # --------------------------------------------------
        # Load Minari trajectories (positions only)
        # --------------------------------------------------
        dataset = minari.load_dataset(dataset_name, download=True)

        self.trajectories = []
        for ep in dataset:
            obs = ep.observations
            if isinstance(obs, dict):
                obs = obs["observation"]

            pos = obs[:, :2].astype(np.float32)
            if len(pos) >= 2:
                self.trajectories.append(pos)

        assert len(self.trajectories) > 0, "No valid trajectories found"

    # --------------------------------------------------
    # Sampling utilities
    # --------------------------------------------------

    def _sample_skips(self):
        return np.array(
            np.random.lognormal(self.log_mu, self.log_sigma, self.horizon)
        ).astype(np.float32)

    def _interp(self, pos, tau):
        T = len(pos)
        tau = np.clip(tau, 0, T - 1)
        i = int(np.floor(tau))
        if i >= T - 1:
            return pos[-1].copy()
        a = tau - i
        return pos[i] + a * (pos[i + 1] - pos[i])

    # --------------------------------------------------
    # Generate ONE valid coarse window
    # --------------------------------------------------

    def _generate_valid_windows(self, positions):
        T = len(positions)
        windows = []

        for _ in range(self.max_rejection_attempts):
            # 1) sample ONE skip pattern
            skips = self._sample_skips()
            total_skip = skips.sum()

            # cannot fit anywhere → resample skips
            if total_skip > T - 1:
                continue

            # 2) latest feasible start
            max_start = (T - 1) - total_skip

            # 3) sample multiple starts
            start_taus = np.random.uniform(0, max_start, size=self.starts_per_skip)

            cum = np.cumsum(skips)

            # 4) try ALL placements
            for start_tau in start_taus:
                taus = start_tau + np.concatenate([[0], cum[:-1]])

                coarse_pos = np.array(
                    [self._interp(positions, tau) for tau in taus],
                    dtype=np.float32,
                )

                skip_list = [(coarse_pos[i], skips[i]) for i in range(self.horizon)]
                pos_dense, _, _ = self.spline_func(skip_list)

                feasible, _ = self.collision_checker(pos_dense)
                if feasible:
                    windows.append(np.concatenate([coarse_pos, skips[:, None]], axis=1))

            # If we got at least one valid placement, stop resampling skips
            if len(windows) > 0:
                break

        return windows  # may be empty

    # --------------------------------------------------
    # Main build function
    # --------------------------------------------------

    def build(self):
        """
        Returns:
            raw_dataset: (N, H, 3) array of UNNORMALIZED coarse windows
        """
        collected = []

        total = (
            len(self.trajectories) * self.samples_per_trajectory * self.starts_per_skip
        )
        print("Collecting valid coarse trajectories...")

        with tqdm(total=total) as pbar:
            for traj in self.trajectories:
                for _ in range(self.samples_per_trajectory):

                    windows = self._generate_valid_windows(traj)
                    # windows is a LIST (possibly empty)

                    for w in windows:
                        collected.append(w)

                    pbar.update(self.starts_per_skip)

        assert len(collected) > 0, "No valid samples collected"
        return np.stack(collected, axis=0)


def normalize_and_save(dataset, out_path):
    """
    dataset: (N, H, 3) unnormalized
    """

    pos = dataset[..., :2].reshape(-1, 2)
    skip = dataset[..., 2].reshape(-1)

    pos_mean = pos.mean(axis=0).astype(np.float32)
    pos_std = pos.std(axis=0).astype(np.float32) + 1e-8

    skip_mean = skip.mean().astype(np.float32)
    skip_std = skip.std().astype(np.float32) + 1e-8

    print("\nNormalization statistics (from SURVIVING samples):")
    print(f"  pos_mean : {pos_mean}")
    print(f"  pos_std  : {pos_std}")
    print(f"  skip_mean: {skip_mean}")
    print(f"  skip_std : {skip_std}\n")

    dataset_norm = dataset.copy()
    dataset_norm[..., :2] = (dataset[..., :2] - pos_mean) / pos_std
    dataset_norm[..., 2] = (dataset[..., 2] - skip_mean) / skip_std

    np.savez(
        out_path,
        data=dataset_norm,
        flat_mean=pos_mean,
        flat_std=pos_std,
        skip_mean=skip_mean,
        skip_std=skip_std,
        full_mean=np.array([*pos_mean, skip_mean], dtype=np.float32),
        full_std=np.array([*pos_std, skip_std], dtype=np.float32),
    )

    print("Saved dataset:", out_path)
    print("Shape:", dataset_norm.shape)


if __name__ == "__main__":

    DATASET_NAME = "D4RL/pointmaze/umaze-v2"

    wall_rects = extract_wall_rects(DATASET_NAME)

    builder = OfflineIndependentSkipDatasetBuilder(
        dataset_name=DATASET_NAME,
        horizon=32,
        samples_per_trajectory=100,
        max_rejection_attempts=1000,
        spline_func=expand_spline_from_skip_list,
        collision_checker=lambda p: verify_trajectory_dense(p, wall_rects),
        starts_per_skip=1,
    )

    OUT_PATH = (
        Path(__file__).parent.parent.parent
        / "offline_datasets"
        / "umaze_h32_mu1_sig1.npz"
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.touch()

    raw_dataset = builder.build()

    normalize_and_save(raw_dataset, out_path=OUT_PATH)
