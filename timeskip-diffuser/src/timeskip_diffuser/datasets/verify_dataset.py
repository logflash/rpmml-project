import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm



class OfflineSkipDataset(Dataset):
    """
    Loads precomputed, normalized windows (N, horizon, 3)
    from an .npz archive that also contains normalization statistics.

    Mirrors exactly the attributes of MinariTrajectoryDatasetIndependentSkips:
        - state_dim = 2
        - action_dim = 1
        - traj_dim = 3
        - pos_mean, pos_std
        - flat_mean, flat_std
        - skip_mean, skip_std
        - mean, std  (full 3-dim stats)
    """

    def __init__(self, file_path, horizon=32):
        archive = np.load(file_path)

        # -------------------------------------------------------
        # Load normalized trajectory data
        # -------------------------------------------------------
        data = archive["data"]  # (N, H, 3)
        assert data.ndim == 3 and data.shape[1] == horizon

        self.data = torch.from_numpy(data).float()
        self.horizon = horizon

        # Exactly match MinariIndependent dims
        self.state_dim = 2          # (x, y)
        self.action_dim = 1         # skip
        self.traj_dim = 3           # 2 + 1

        # -------------------------------------------------------
        # Load same normalization fields as the on-the-fly dataset
        # -------------------------------------------------------

        # Position normalization
        self.pos_mean = archive["flat_mean"].astype(np.float32)  # (2,)
        self.pos_std  = archive["flat_std"].astype(np.float32)   # (2,)

        # IMPORTANT: aliases so planner code works
        self.flat_mean = self.pos_mean
        self.flat_std  = self.pos_std

        # Skip normalization
        self.skip_mean = float(archive["skip_mean"])
        self.skip_std  = float(archive["skip_std"])

        # Full 3-d normalization (for denorm in planner)
        self.mean = archive["full_mean"].astype(np.float32)      # (3,)
        self.std  = archive["full_std"].astype(np.float32)       # (3,)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Returns: (H, 3) tensor:
            [:,0:2] normalized positions
            [:,2]   normalized skip
        """
        return self.data[idx]
    
# Each wall is (xmin, xmax, ymin, ymax)
WALLS = [
    # Outer boundary
    (-2.5,  2.5,  1.5,  2.5),   # top
    (-2.5,  2.5, -2.5, -1.5),  # bottom
    (-2.5, -1.5, -2.5,  2.5),  # left
    ( 1.5,  2.5, -2.5,  2.5),  # right

    # Center block
    (-0.5,  0.8, -0.2,  0.5),
]


def point_in_any_wall(x, y, walls):
    for xmin, xmax, ymin, ymax in walls:
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True
    return False

def segment_hits_wall(p0, p1, walls, steps=100):
    for s in range(steps + 1):
        t = s / steps
        x = (1 - t) * p0[0] + t * p1[0]
        y = (1 - t) * p0[1] + t * p1[1]
        if point_in_any_wall(x, y, walls):
            return True
    return False


def scan_dataset_for_wall_violations(dataset, walls):
    bad_trajs = []

    data = dataset.data.cpu().numpy()   # (N, T, D)

    for i in tqdm(range(len(data)), desc="Scanning trajectories"):
        traj = data[i, :, :2] * dataset.flat_std + dataset.flat_mean

        violated = False

        # 1. Point-wise check
        for t in range(len(traj)):
            if point_in_any_wall(traj[t][0], traj[t][1], walls):
                bad_trajs.append((i, "POINT", t))
                violated = True
                print("found violation")
                break

        if violated:
            continue

        # 2. Segment-wise check
        for t in range(len(traj) - 1):
            if segment_hits_wall(traj[t], traj[t+1], walls):
                bad_trajs.append((i, "SEGMENT", t))
                print("found violation")
                break

    print(f"\nScan complete")
    print(f"Total trajectories checked: {len(data)}")
    print(f"Trajectories with wall violations: {len(bad_trajs)}")

    return bad_trajs

dataset = MinariTrajectoryDatasetWithPseudoActions(
    "D4RL/pointmaze/umaze-v2", horizon=32, n_chunks_frac= 0.5
)

minari_dataset = OfflineSkipDataset(
    "offline_umaze_independent_skips.npz",
    horizon=32
)

violations = scan_dataset_for_wall_violations(
    dataset=minari_dataset,
    walls=WALLS
)

# Print first few
print("First 10 violations:", violations[:10])