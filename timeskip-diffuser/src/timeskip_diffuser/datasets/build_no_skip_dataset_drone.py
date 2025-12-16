import os
import numpy as np
from glob import glob
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

TXT_DIR = "/scratch/network/dd6849/rpmml-project/timeskip-diffuser/src/timeskip_diffuser/datasets/groundtruth"
FILE_GLOB = "indoor_45*.txt"

OUT_PATH = "fixed_stats_offline_indoor45_plain.npz"

HORIZON = 32
SAMPLES_PER_TRAJ = 100
SKIP_HEADER_ROWS = 1

np.random.seed(0)

# ============================================================
# LOAD RAW TRAJECTORIES
# ============================================================

print("\n" + "=" * 80)
print("LOADING TXT TRAJECTORIES (POSITION ONLY)")
print("=" * 80)

files = sorted(glob(os.path.join(TXT_DIR, FILE_GLOB)))
assert len(files) > 0, "No TXT files found"

trajectories = []   # each: (T, 3)

for path in files:
    raw = np.loadtxt(path, dtype=np.float32, skiprows=SKIP_HEADER_ROWS)
    if raw.ndim != 2 or raw.shape[1] < 4:
        continue

    pos = raw[:, 1:4]  # tx ty tz

    if len(pos) >= HORIZON:
        trajectories.append(pos)

print(f"Loaded {len(trajectories)} valid trajectories")

# ============================================================
# COLLECT WINDOWS
# ============================================================

windows = []

for traj in trajectories:
    T = len(traj)
    max_start = T - HORIZON

    for _ in range(SAMPLES_PER_TRAJ):
        start = np.random.randint(0, max_start + 1)
        window = traj[start : start + HORIZON]
        windows.append(window)

windows = np.stack(windows, axis=0)  # (N, H, 3)

print(f"Collected {windows.shape[0]} windows")
print("Raw window shape:", windows.shape)

# ============================================================
# NORMALIZATION
# ============================================================

flat_positions = windows.reshape(-1, 3)
pos_mean = flat_positions.mean(axis=0)
pos_std  = flat_positions.std(axis=0) + 1e-8

windows_norm = (windows - pos_mean) / pos_std

print("\nNormalization stats:")
print("mean:", pos_mean)
print("std :", pos_std)

# ============================================================
# FINAL CHECKS
# ============================================================

assert np.all(np.isfinite(windows_norm)), "NaNs detected"
assert windows_norm.shape[1] == HORIZON
assert windows_norm.shape[2] == 3

print("Post-norm mean:", windows_norm.mean(axis=(0, 1)))
print("Post-norm std :", windows_norm.std(axis=(0, 1)))

# ============================================================
# SAVE
# ============================================================

np.savez(
    OUT_PATH,
    data=windows_norm.astype(np.float32),  # (N, H, 3)
    flat_mean=pos_mean.astype(np.float32),
    flat_std=pos_std.astype(np.float32),
)

print("\n" + "=" * 80)
print("DATASET SAVED")
print("=" * 80)
print("Path:", OUT_PATH)
print("Final shape:", windows_norm.shape)
