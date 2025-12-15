import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm

BASE_DT = 0.002 #2 ms


class TxtTrajectoryDatasetIndependentSkips(Dataset):
    """
    TXT analogue of MinariTrajectoryDatasetIndependentSkips.

    Files: indoor_45*.txt
    Format per row (after header):
        timestamp tx ty tz qx qy qz qw

    We extract:
        state = [tx, ty, tz]
        action = skip  (sampled lognormal, dimensionless "steps")

    Key features preserved:
    - On-the-fly independent skip sampling (LogNormal(mu=1, sigma=1))
    - Rejection sampling: ensure window fits without clipping
    - Parallel skip sampling
    - Interpolate positions using real timestamps (query times = start_time + tau*BASE_DT)
    - Empirical skip_mean / skip_std estimated by sampling windows
    """

    def __init__(
        self,
        txt_dir,
        horizon=32,
        normalize=True,
        samples_per_trajectory=100,
        max_rejection_attempts=1000,
        file_glob="indoor_45*.txt",
        skip_header_rows=1,
        estimate_skip_windows=50_000,
    ):
        self.horizon = horizon
        self.samples_per_trajectory = samples_per_trajectory
        self.max_rejection_attempts = max_rejection_attempts
        self.normalize_flag = normalize

        # Log-normal params (same as your Minari code)
        self.lognormal_mu = 1.0
        self.lognormal_sigma = 1.0

        # dims
        self.state_dim = 3   # tx, ty, tz
        self.action_dim = 1  # skip
        self.traj_dim = self.state_dim + self.action_dim  # 4

        # ------------------------------------------------------------
        # Load trajectories from matching TXT files
        # ------------------------------------------------------------
        files = sorted(glob(os.path.join(txt_dir, file_glob)))
        if len(files) == 0:
            raise ValueError(f"No files matched pattern {file_glob} in {txt_dir}")
        print("\n" + "="*80)
        print("TXT DATASET LOADED")
        print("="*80)
        print(f"Number of TXT files matched: {len(files)}")
        
        self.trajectories = []  # list of dicts: {"t": (T,), "p": (T,3)}
        all_positions = []

        for path in files:
            raw = np.loadtxt(path, dtype=np.float32, skiprows=skip_header_rows)
            if raw.ndim != 2 or raw.shape[1] < 4:
                continue

            t = raw[:, 0].astype(np.float32)
            p = raw[:, 1:4].astype(np.float32)  # tx ty tz

            # must be increasing-ish and long enough
            if len(t) < 2 or len(p) < 2:
                continue

            # If timestamps are not strictly increasing, you can sort:
            # idx = np.argsort(t); t = t[idx]; p = p[idx]
            # but usually your logs are already ordered.

            self.trajectories.append({"t": t, "p": p})
            all_positions.append(p)

        if len(self.trajectories) == 0:
            raise ValueError("No valid trajectories loaded from TXT files.")
        print(f"Number of valid trajectories kept: {len(self.trajectories)}")

        lengths = [len(traj["p"]) for traj in self.trajectories]
        print(f"Trajectory length stats: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

        print("First trajectory sample:")
        print("  timestamps (first 5):", self.trajectories[0]["t"][:5])
        print("  positions  (first 5):", self.trajectories[0]["p"][:5])
        all_positions = np.concatenate(all_positions, axis=0)

        # ------------------------------------------------------------
        # Normalization stats
        # ------------------------------------------------------------
        if normalize:
            self.pos_mean = all_positions.mean(axis=0).astype(np.float32)
            self.pos_std = all_positions.std(axis=0).astype(np.float32) + 1e-8

            # skip stats estimated empirically under YOUR rejection sampler
            self.skip_mean, self.skip_std = self._estimate_skip_stats(
                num_windows=estimate_skip_windows
            )
            print("\n" + "="*80)
            print("SKIP STATISTICS (EMPIRICAL)")
            print("="*80)
            print(f"lognormal params: mu={self.lognormal_mu}, sigma={self.lognormal_sigma}")
            print(f"Estimated skip_mean: {self.skip_mean:.4f}")
            print(f"Estimated skip_std : {self.skip_std:.4f}")
        else:
            self.pos_mean = np.zeros(self.state_dim, dtype=np.float32)
            self.pos_std = np.ones(self.state_dim, dtype=np.float32)
            self.skip_mean = 0.0
            self.skip_std = 1.0

        # full [tx,ty,tz,skip] stats for planner denorm
        self.mean = np.zeros(self.traj_dim, dtype=np.float32)
        self.std = np.ones(self.traj_dim, dtype=np.float32)
        self.mean[: self.state_dim] = self.pos_mean
        self.std[: self.state_dim] = self.pos_std
        self.mean[self.state_dim] = self.skip_mean
        self.std[self.state_dim] = self.skip_std

        print("skip_mean:", self.skip_mean)
        print("skip_std :", self.skip_std)

        # ------------------------------------------------------------
        # Build indices: each trajectory gets multiple virtual samples
        # ------------------------------------------------------------
        self.indices = []
        for traj_idx in range(len(self.trajectories)):
            for _ in range(self.samples_per_trajectory):
                self.indices.append(traj_idx)

    # ------------------------- skip sampling -------------------------

    def _sample_skips_parallel(self, n_skips):
        return np.random.lognormal(
            mean=self.lognormal_mu,
            sigma=self.lognormal_sigma,
            size=n_skips
        ).astype(np.float32)

    def _estimate_skip_stats(self, num_windows=50_000):
        all_skips = []
        for _ in range(num_windows):
            traj_idx = np.random.randint(len(self.trajectories))
            traj = self.trajectories[traj_idx]
            _, window_skips = self._generate_window_with_rejection(traj["t"], traj["p"])
            all_skips.append(window_skips)

        all_skips = np.concatenate(all_skips, axis=0)
        return all_skips.mean().astype(np.float32), (all_skips.std().astype(np.float32) + 1e-8)

    # ------------------------- interpolation -------------------------

    def _interpolate_position_time(self, t, p, tq):
        """
        Linear interpolate position p(t) at query time tq.
        t: (T,) increasing timestamps
        p: (T,3)
        tq: float in [t[0], t[-1]]
        """
        tq = float(np.clip(tq, t[0], t[-1]))
        j = np.searchsorted(t, tq, side="right") - 1
        j = int(np.clip(j, 0, len(t) - 2))

        t0, t1 = t[j], t[j + 1]
        p0, p1 = p[j], p[j + 1]

        if t1 <= t0:
            return p0.copy()

        alpha = (tq - t0) / (t1 - t0)
        return p0 + alpha * (p1 - p0)

    # ------------------------- window generation -------------------------

    def _generate_window_with_rejection(self, t, p):
        """
        Like your Minari version, but:
        - tau is in "base steps"
        - query time = start_time + tau * BASE_DT
        - interpolate using real timestamps
        """
        T = len(t)
        t0, tN = t[0], t[-1]

        for _ in range(self.max_rejection_attempts):
            # sample all skips (dimensionless)
            skips = self._sample_skips_parallel(self.horizon)  # (H,)

            # tau_i = sum(skips[0:i]) in steps (tau_0 = 0)
            cum = np.cumsum(skips)
            taus = np.concatenate([[0.0], cum[:-1]])  # (H,)

            # total duration in timestamp-units
            total_steps = float(cum[-1])
            total_time = total_steps * BASE_DT

            # sample start time so end fits
            if (tN - t0) <= total_time:
                # too short to fit typical sample; try again
                continue

            start_time = np.random.uniform(t0, tN - total_time)

            # compute query times
            query_times = start_time + taus * BASE_DT  # (H,)

            # guaranteed within [t0, tN] by construction, but keep safe
            if query_times[-1] + skips[-1] * BASE_DT > tN:
                continue

            window_positions = np.array(
                [self._interpolate_position_time(t, p, tq) for tq in query_times],
                dtype=np.float32
            )
            return window_positions, skips

        # fallback (rare): shrink skips to fit
        skips = self._sample_skips_parallel(self.horizon)
        cum = np.cumsum(skips)
        total_time = float(cum[-1]) * BASE_DT
        available = max(1e-6, (tN - t0) * 0.95)  # 95% safety

        if total_time > available:
            scale = available / total_time
            skips = skips * scale

        cum = np.cumsum(skips)
        taus = np.concatenate([[0.0], cum[:-1]])
        start_time = t0
        query_times = start_time + taus * BASE_DT

        window_positions = np.array(
            [self._interpolate_position_time(t, p, tq) for tq in query_times],
            dtype=np.float32
        )
        return window_positions, skips

    # ------------------------- normalization -------------------------

    def normalize_position(self, x):
        return (x - self.pos_mean) / self.pos_std

    def normalize_skip(self, s):
        return (s - self.skip_mean) / self.skip_std

    # ------------------------- dataset protocol -------------------------

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx = self.indices[idx]
        traj = self.trajectories[traj_idx]

        window_positions, window_skips = self._generate_window_with_rejection(traj["t"], traj["p"])

        pos_norm = self.normalize_position(window_positions)
        skip_norm = self.normalize_skip(window_skips)

        sample = np.zeros((self.horizon, self.traj_dim), dtype=np.float32)
        sample[:, : self.state_dim] = pos_norm
        sample[:, self.state_dim] = skip_norm

        return torch.from_numpy(sample).float()



if __name__ == "__main__":
    # ============================================================
    # CONFIG
    # ============================================================
    TXT_DIR = "/scratch/network/dd6849/rpmml-project/timeskip-diffuser/src/timeskip_diffuser/datasets/groundtruth"  
    FILE_GLOB = "indoor_45*.txt"

    OUT_PATH = "fixed_stats_offline_indoor45_independent_skips_h32_mu1_sig1_dt200.npz"

    HORIZON = 32
    SAMPLES_PER_TRAJ = 100
    MAX_REJECTION_ATTEMPTS = 50
    ESTIMATE_SKIP_WINDOWS = 20_000   

    torch.manual_seed(0)
    np.random.seed(0)

    print("\n" + "="*80)
    print("TXT INDEPENDENT-SKIP OFFLINE DATASET GENERATION")
    print("="*80)

    # ============================================================
    # DATASET
    # ============================================================
    print("\nLoading TXT dataset...")
    txt_ds = TxtTrajectoryDatasetIndependentSkips(
        txt_dir=TXT_DIR,
        horizon=HORIZON,
        normalize=True,
        samples_per_trajectory=SAMPLES_PER_TRAJ,
        max_rejection_attempts=MAX_REJECTION_ATTEMPTS,
        file_glob=FILE_GLOB,
        skip_header_rows=1,
        estimate_skip_windows=ESTIMATE_SKIP_WINDOWS,
    )

    print("\nDataset instantiated.")
    print(f"Total virtual samples: {len(txt_ds)}")
    print(f"Trajectory dim: {txt_ds.traj_dim}")
    print(f"State dim: {txt_ds.state_dim}, Skip dim: {txt_ds.action_dim}")

    # ============================================================
    # OFFLINE SAMPLING
    # ============================================================
    print("\nGenerating offline samples...")
    all_samples = []

    for i in tqdm(range(len(txt_ds))):
        sample = txt_ds[i].numpy()   # (H, 4)
        all_samples.append(sample)

        # Optional early break for quick testing
        # if i == 1000:
        #     break

    dataset_array = np.stack(all_samples, axis=0)

    # ============================================================
    # FINAL SANITY CHECKS
    # ============================================================
    print("\n" + "="*80)
    print("FINAL DATASET SANITY CHECK")
    print("="*80)

    print("Dataset array shape:", dataset_array.shape)
    print("Position (norm) mean:", dataset_array[..., :3].mean(axis=(0, 1)))
    print("Position (norm) std :", dataset_array[..., :3].std(axis=(0, 1)))
    print("Skip (norm) mean    :", dataset_array[..., 3].mean())
    print("Skip (norm) std     :", dataset_array[..., 3].std())

    assert np.all(np.isfinite(dataset_array)), "NaNs detected in dataset!"
    assert dataset_array.shape[1] == HORIZON, "Horizon mismatch!"

    # ============================================================
    # SAVE
    # ============================================================
    np.savez(
        OUT_PATH,
        data=dataset_array,
        flat_mean=txt_ds.pos_mean,
        flat_std=txt_ds.pos_std,
        skip_mean=txt_ds.skip_mean,
        skip_std=txt_ds.skip_std,
        full_mean=txt_ds.mean,
        full_std=txt_ds.std,
    )

    print("\n" + "="*80)
    print("DATASET SAVED SUCCESSFULLY")
    print("="*80)
    print("Output path:", OUT_PATH)
    print("Final dataset size:", dataset_array.shape)
