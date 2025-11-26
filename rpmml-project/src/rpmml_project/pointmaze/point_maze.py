import gymnasium as gym
import minari
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================


class MinariTrajectoryDataset(Dataset):
    """Dataset for loading state-only trajectory sequences from Minari."""

    def __init__(
        self, dataset_name="D4RL/pointmaze/umaze-v2", horizon=32, normalize=True
    ):
        self.horizon = horizon
        self.dataset = minari.load_dataset(dataset_name, download=True)

        # Extract all state trajectories (no actions)
        self.trajectories = []
        for episode in self.dataset:
            obs = episode.observations

            # PointMaze obs is typically dict with 'observation' and 'achieved_goal'
            if isinstance(obs, dict):
                obs = obs["observation"]

            # Store only states, not actions
            self.trajectories.append(obs)

        # Compute normalization statistics
        all_data = np.concatenate(self.trajectories, axis=0)
        self.state_dim = obs.shape[-1]
        self.action_dim = 2  # Still track this for action recovery

        if normalize:
            self.mean = all_data.mean(axis=0)
            self.std = all_data.std(axis=0) + 1e-8
        else:
            self.mean = np.zeros_like(all_data[0])
            self.std = np.ones_like(all_data[0])

        # Create indices for sampling
        self.indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            for t in range(len(traj) - horizon + 1):
                self.indices.append((traj_idx, t))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_t = self.indices[idx]
        traj = self.trajectories[traj_idx][start_t : start_t + self.horizon]
        traj = self.normalize(traj)
        return torch.FloatTensor(traj)

class MinariTrajectoryDatasetWithPseudoActions(Dataset):
    """Dataset for loading state (position) and skip-value trajectory sequences from Minari."""

    def __init__(
        self, dataset_name="D4RL/pointmaze/umaze-v2", horizon=3, normalize=True,
        n_chunks_frac=0.1, alpha=3.0,
    ):
        # TODO: tune n_chunks_frac and alpha for better performance
        self.horizon = horizon
        self.n_chunks_frac = n_chunks_frac
        self.alpha = alpha
        self.normalize_flag = normalize

        # Load dataset
        self.dataset = minari.load_dataset(dataset_name, download=True)

        # Store full state trajectories separately (for normalization)
        self.state_trajectories = []
        # Store skip lists separately
        self.skip_trajectories = []

        # ---------------------------------------------------------
        # Process each episode
        # ---------------------------------------------------------
        for episode in self.dataset:
            obs = episode.observations

            # Handle dict observations used in PointMaze
            if isinstance(obs, dict):
                obs = obs["observation"]

            states = obs                   # full raw states (T, state_dim)
            T = len(states)
            positions = states[:, :2]      # (T, 2)
            velocities = states[:, 2:4]    # (T, 2) (not used yet but kept)

            # --- Dirichlet chunks ---
            n_chunks = max(1, int(self.n_chunks_frac * T))
            chunks = self._dirichlet_chunks(T, n_chunks=n_chunks, alpha=self.alpha)

            # --- Build skip_list ---
            skip_list = []
            tau = 0.0
            print(chunks)
            for chunk in chunks:
                
                tau_clamped = min(tau, T - 1)
                idx = int(np.floor(tau_clamped))
                delta_t = tau_clamped - idx  # in [0,1)

                if idx >= T - 1:
                    pos = positions[-1]
                else:
                    # simple linear interpolation
                    pos = positions[idx] + (positions[idx + 1] - positions[idx]) * delta_t

                skip_list.append((pos, chunk, tau_clamped))
                tau += chunk


            # Save both sequences
            self.state_trajectories.append(states)
            self.skip_trajectories.append(skip_list)

        # ---------------------------------------------------------
        # NORMALIZATION over raw states
        # ---------------------------------------------------------
        all_states = np.concatenate(self.state_trajectories, axis=0)
        self.state_dim = all_states.shape[-1]

        if normalize:
            self.mean = all_states.mean(axis=0)
            self.std  = all_states.std(axis=0) + 1e-8
        else:
            self.mean = np.zeros(self.state_dim)
            self.std  = np.ones(self.state_dim)

        # ---------------------------------------------------------
        # Build horizon indices over skip trajectories
        # ---------------------------------------------------------
        self.indices = []
        for traj_idx, skip_list in enumerate(self.skip_trajectories):
            S = len(skip_list)
            if S >= self.horizon:
                for t in range(S - self.horizon + 1):
                    self.indices.append((traj_idx, t))

    # ---------------------------------------------------------
    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    # ---------------------------------------------------------
    @staticmethod
    def _dirichlet_chunks(total_T, n_chunks, alpha):
        weights = np.random.dirichlet([alpha] * n_chunks)
        return weights * total_T

    # ---------------------------------------------------------
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start = self.indices[idx]

        skip_list = self.skip_trajectories[traj_idx]

        # Extract window of length horizon
        window = skip_list[start:start + self.horizon]

        # Unpack window into arrays
        positions = np.array([p for (p, c, tau) in window], dtype=np.float32)  # (H,2)
        skip_vals = np.array([c for (p, c, tau) in window], dtype=np.float32)  # (H,)
        tau_vals  = np.array([tau for (p, c, tau) in window], dtype=np.float32)  # (H,)

        return (
            torch.FloatTensor(positions),   # (H,2)
            torch.FloatTensor(skip_vals),   # (H,)
            torch.FloatTensor(tau_vals)     # (H,)
        )
 


class MinariTrajectoryDatasetWithActions(Dataset):
    """Dataset for loading state + action trajectory windows from Minari."""

    def __init__(self, dataset_name="D4RL/pointmaze/umaze-v2", horizon=32, normalize=True):
        self.horizon = horizon
        self.dataset = minari.load_dataset(dataset_name, download=True)

        # List of dict trajectories: {"obs": (T,4), "act": (T,2)}
        self.trajectories = []
        for episode in self.dataset:
            obs = episode.observations
            act = episode.actions

            # Handle dict observations used in PointMaze
            if isinstance(obs, dict):
                obs = obs["observation"]

            self.trajectories.append({
                "obs": obs,      # (T, state_dim)
                "act": act       # (T, action_dim)
            })

        # Infer dims
        example_obs = self.trajectories[0]["obs"]
        example_act = self.trajectories[0]["act"]

        self.state_dim = example_obs.shape[-1]   # normally 4: [x,y,vx,vy]
        self.action_dim = example_act.shape[-1]  # normally 2: [ax,ay]

        # -----------------------------------------------------------
        # Compute normalization statistics (ONLY for states)
        # -----------------------------------------------------------
        all_obs = np.concatenate([traj["obs"] for traj in self.trajectories], axis=0)

        if normalize:
            self.mean = all_obs.mean(axis=0)
            self.std = all_obs.std(axis=0) + 1e-8
        else:
            self.mean = np.zeros(self.state_dim)
            self.std  = np.ones(self.state_dim)

        # -----------------------------------------------------------
        # Build sampling index list (traj_idx, start_time)
        # -----------------------------------------------------------
        self.indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            T = len(traj["obs"])
            for t in range(T - horizon + 1):
                self.indices.append((traj_idx, t))

    # -----------------------------------------------------------
    # Normalization utilities
    # -----------------------------------------------------------
    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    # -----------------------------------------------------------
    # PyTorch Dataset API
    # -----------------------------------------------------------
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_t = self.indices[idx]
        traj = self.trajectories[traj_idx]

        obs_window = traj["obs"][start_t:start_t + self.horizon]
        act_window = traj["act"][start_t:start_t + self.horizon]
        
        

        return {
            "obs": torch.FloatTensor(obs_window),   # (T, state_dim)
            "act": torch.FloatTensor(act_window)         # raw (T, action_dim)
        }

# ============================================================================
# ACTION RECOVERY FROM STATE TRAJECTORIES
# ============================================================================
# computing dt 
def estimate_dt_from_obs(pos, vel):
    dx = pos[1:] - pos[:-1]                 # Δx
    dist = np.linalg.norm(dx, axis=1)       # ‖Δx‖
    speed = np.linalg.norm(vel[:-1], axis=1)

    mask = speed > 1e-6                     # avoid division by zero
    dt_est = dist[mask] / speed[mask]

    return np.mean(dt_est)


def recover_actions_from_states(states, dt=0.01, m= 0.0424):
    """
    Compute velocity (T-1,2) and acceleration (T-2,2) 
    aligned with transitions.

    Args:
        states: (T, >=2) array containing positions in first 2 dims
        dt: timestep

    Returns:
        velocities:    (T-1, 2)
        accelerations: (T-2, 2)
    """

    # PointMaze typically has state = [x, y, vx, vy]
    # Actions are [ax, ay] (acceleration commands) or [vx_desired, vy_desired]

    # # Option 1: If actions are velocity commands, extract from state
    # if states.shape[-1] >= 4:
    #     velocities = states[:, 2:4]  # Extract vx, vy
    #     velocities = velocities[:-1]
    #     accelerations = (velocities[1:] - velocities[:-1]) / dt
    #     return velocities, accelerations

    # Option 2: If only positions, compute velocities via finite differences
    # sticking with this since we are now pursuing the flattened state definition
    positions = states[:, :2]        # (T,2)

    
    # v_t associated with transition s_t -> s_{t+1}
    velocities = (positions[1:] - positions[:-1]) / dt    # (T-1,2)

    # a_t associated with transition v_t -> v_{t+1}
    accelerations = (velocities[1:] - velocities[:-1]) / (dt)   # (T-2,2)

    return velocities, accelerations

# ============================================================
# 1. Estimate sparse velocities (average per segment)
# ============================================================

def estimate_sparse_velocities(skip_list, dt):
    """
    Given skip_list = [(pos_i, skip_i), ...],
    extract positions, skip amounts, and estimate average velocities.

    Returns:
        positions: (N,2)
        velocities: (N,2)   (average for each segment)
        skips: list of ints
    """
    positions = np.array([p for (p, k) in skip_list])
    skips = [k for (p, k) in skip_list]

    N = len(positions)
    velocities = np.zeros((N, 2))

    for i in range(N - 1):
        k = skips[i]
        T = k * dt
        velocities[i] = (positions[i+1] - positions[i]) / T

    velocities[-1] = velocities[-2]  # last velocity: just copy previous

    return positions, velocities, skips


# ============================================================
# 2. Hermite spline segment (WITH CORRECT VELOCITY SCALING)
# ============================================================

def hermite_segment(p0, v0_scaled, p1, v1_scaled, num_points):
    """
    Hermite spline between p0 and p1 with endpoint derivatives v0_scaled, v1_scaled.

    v0_scaled, v1_scaled MUST BE SCALED BY SEGMENT DURATION.

    Returns:
        p: (num_points, 2)
        v: (num_points, 2)   derivative wrt spline time (not physical!)
        a: (num_points, 2)
    """
    t = np.linspace(0, 1, num_points)

    # Hermite basis
    h00 =  2*t**3 - 3*t**2 + 1
    h10 =      t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 =      t**3 -   t**2

    p = (h00[:,None]*p0 +
         h10[:,None]*v0_scaled +
         h01[:,None]*p1 +
         h11[:,None]*v1_scaled)

    # Velocity basis
    dh00 =  6*t**2 - 6*t
    dh10 =  3*t**2 - 4*t + 1
    dh01 = -6*t**2 + 6*t
    dh11 =  3*t**2 - 2*t

    v = (dh00[:,None]*p0 +
         dh10[:,None]*v0_scaled +
         dh01[:,None]*p1 +
         dh11[:,None]*v1_scaled)

    # Acceleration basis
    d2h00 = 12*t - 6
    d2h10 =  6*t - 4
    d2h01 = -12*t + 6
    d2h11 =  6*t - 2

    a = (d2h00[:,None]*p0 +
         d2h10[:,None]*v0_scaled +
         d2h01[:,None]*p1 +
         d2h11[:,None]*v1_scaled)

    return p, v, a


# ============================================================
# 3. Expand skip_list into full spline trajectory
# ============================================================

def expand_spline_from_skip_list(skip_list, dt=0.05):
    """
    Convert skip_list → dense spline-based trajectory.
    Ensures the last sample of each segment equals the next sparse waypoint.

    Returns:
        full_p: (T,2)
        full_v: (T,2)
        full_a: (T,2)
    """

    positions, velocities, skips = estimate_sparse_velocities(skip_list, dt)

    full_p, full_v, full_a = [], [], []

    for i in range(len(positions) - 1):

        p0 = positions[i]
        p1 = positions[i+1]

        k  = skips[i]
        T  = k * dt          # physical duration of this segment

        # SCALE velocities to spline coordinates
        v0_scaled = velocities[i]     * T
        v1_scaled = velocities[i+1]   * T

        num_samples = k + 1           # must produce exactly k+1 samples

        P, V, A = hermite_segment(p0, v0_scaled, p1, v1_scaled, num_samples)

        # avoid duplication at segment seam
        if len(full_p) > 0:
            P = P[1:]
            V = V[1:]
            A = A[1:]

        full_p.extend(P)
        full_v.extend(V)
        full_a.extend(A)

    return np.array(full_p), np.array(full_v), np.array(full_a)

# ============================================================================
# 2. TEMPORAL U-NET COMPONENTS
# ============================================================================


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual block with group normalization and FiLM conditioning."""

    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))

        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, dim), nn.SiLU(), nn.Conv1d(dim, dim_out, 3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
            nn.Conv1d(dim_out, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        """Forward for residual block."""

        h = self.block1(x)

        # FiLM conditioning
        scale, shift = self.mlp(time_emb).chunk(2, dim=-1)
        scale = scale[:, :, None]
        shift = shift[:, :, None]
        h = h * (scale + 1) + shift

        h = self.block2(h)
        return h + self.res_conv(x)


class TemporalAttention(nn.Module):
    """Self-attention over temporal dimension."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        """Forward for temporal attention."""

        b, c, t = x.shape
        x = rearrange(x, "b c t -> b t c")

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b t (h d) -> b h t d", h=self.heads), qkv)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.to_out(out)

        return rearrange(out, "b t c -> b c t")


class TemporalUNet(nn.Module):
    """1D U-Net for STATE trajectory denoising."""

    def __init__(self, state_dim, hidden_dims=[128, 256, 512], time_dim=64):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4),
        )

        # Initial projection (input is now just state_dim, not state+action)
        self.init_conv = nn.Conv1d(state_dim, hidden_dims[0], 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList([])
        self.encoder_attns = nn.ModuleList([])
        self.downsamples = nn.ModuleList([])

        dims = [hidden_dims[0]] + list(hidden_dims)
        for i in range(len(hidden_dims)):
            self.encoder_blocks.append(
                ResidualBlock(dims[i], dims[i + 1], time_dim * 4)
            )
            self.encoder_attns.append(TemporalAttention(dims[i + 1]))
            self.downsamples.append(nn.Conv1d(dims[i + 1], dims[i + 1], 4, 2, 1))

        # Bottleneck
        mid_dim = hidden_dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_dim * 4)
        self.mid_attn = TemporalAttention(mid_dim)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_dim * 4)

        # Decoder
        self.decoder_blocks = nn.ModuleList([])
        self.decoder_attns = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])

        for i in reversed(range(len(hidden_dims))):
            self.upsamples.append(nn.ConvTranspose1d(dims[i + 1], dims[i + 1], 4, 2, 1))
            self.decoder_blocks.append(
                ResidualBlock(dims[i + 1] * 2, dims[i], time_dim * 4)
            )
            self.decoder_attns.append(TemporalAttention(dims[i]))

        # Final projection (output is state_dim)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, hidden_dims[0]),
            nn.SiLU(),
            nn.Conv1d(hidden_dims[0], state_dim, 3, padding=1),
        )

    def forward(self, x, time):
        """Forward for the temporal u-net."""

        # x: (batch, horizon, input_dim)
        x = rearrange(x, "b t c -> b c t")

        t_emb = self.time_mlp(time)

        x = self.init_conv(x)

        # Encoder
        skips = []
        for block, attn, downsample in zip(
            self.encoder_blocks, self.encoder_attns, self.downsamples
        ):
            x = block(x, t_emb)
            x = attn(x)
            skips.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # Decoder
        for block, attn, upsample in zip(
            self.decoder_blocks, self.decoder_attns, self.upsamples
        ):
            x = upsample(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, t_emb)
            x = attn(x)

        x = self.final_conv(x)

        return rearrange(x, "b c t -> b t c")


# ============================================================================
# 3. DIFFUSION PROCESS
# ============================================================================


class GaussianDiffusion:
    """Gaussian diffusion process with linear noise schedule."""

    def __init__(self, timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: add noise to x_start."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][
            :, None, None
        ]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, model, x_start, t):
        """Compute denoising loss."""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """Single denoising step."""
        betas_t = self.betas[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][
            :, None, None
        ]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None, None]

        predicted_noise = model(x, t)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(betas_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device):
        """Full reverse diffusion sampling."""
        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, i)

        return x

    @torch.no_grad()
    def inpaint_sample_loop(
        self, model, shape, device, condition_mask, condition_value
    ):
        """Sampling with inpainting (fix certain timesteps/dimensions)."""
        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, i)

            # Apply conditioning
            x = torch.where(condition_mask, condition_value, x)

        return x


# ============================================================================
# 4. TRAINING
# ============================================================================


class DiffuserTrainer:
    """Training infrastructure for Diffuser."""

    def __init__(self, model, diffusion, dataset, lr=2e-4, device="cuda"):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.device = device

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.dataloader = DataLoader(
            dataset, batch_size=256, shuffle=True, num_workers=4
        )

        # Move diffusion schedule to device
        for attr in [
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
        ]:
            setattr(self.diffusion, attr, getattr(self.diffusion, attr).to(device))

        # EMA
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.995)
        )

    def train_step(self, batch):
        """A single training step for diffusion training."""

        self.optimizer.zero_grad()

        batch = batch.to(self.device)
        t = torch.randint(
            0, self.diffusion.timesteps, (batch.shape[0],), device=self.device
        ).long()

        loss = self.diffusion.p_losses(self.model, batch, t)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.ema_model.update_parameters(self.model)

        return loss.item()

    def train(self, epochs=100):
        self.model.train()

        for epoch in range(epochs):
            epoch_losses = []
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                pbar.set_postfix({"loss": f"{loss:.4f}"})

            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

            # Save checkpoint
            self.save_checkpoint(f"diffuser_checkpoint_epoch_{epoch+1}.pt")

    def save_checkpoint(self, path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema_model": self.ema_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.ema_model.load_state_dict(checkpoint["ema_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


# ============================================================================
# 5. PLANNING & INFERENCE
# ============================================================================


class DiffuserPlanner:
    """Planning with trained diffusion model (STATE-ONLY)."""

    def __init__(self, model, diffusion, dataset, device="cuda"):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.dataset = dataset
        self.device = device

        # Move diffusion schedule to device
        for attr in [
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
        ]:
            setattr(self.diffusion, attr, getattr(self.diffusion, attr).to(device))

    #Changed to also accept goal state to condition on
    @torch.no_grad()
    def plan(self, current_obs, goal_obs, horizon=32):
        """
        Generate a trajectory from current_obs → goal_obs using diffusion inpainting.
        """
        self.model.eval()

        # -------------------------
        # Convert → 1D np arrays
        # -------------------------
        current_obs = np.asarray(current_obs, dtype=np.float32).reshape(-1)
        goal_obs    = np.asarray(goal_obs,    dtype=np.float32).reshape(-1)

        # -------------------------
        # Normalize
        # -------------------------
        start_norm = (current_obs - self.dataset.mean) / self.dataset.std
        goal_norm  = (goal_obs    - self.dataset.mean) / self.dataset.std

        start_norm = torch.tensor(start_norm, dtype=torch.float32, device=self.device)
        goal_norm  = torch.tensor(goal_norm,  dtype=torch.float32, device=self.device)

        # -------------------------
        # Build conditioning mask/value
        # -------------------------
        shape = (1, horizon, self.dataset.state_dim)

        condition_mask  = torch.zeros(shape, dtype=torch.bool, device=self.device)
        condition_value = torch.zeros(shape, dtype=torch.float32, device=self.device)

        condition_mask[0, 0, :]  = True
        condition_mask[0, -1, :] = True

        condition_value[0, 0, :]  = start_norm
        condition_value[0, -1, :] = goal_norm

        # -------------------------
        # Run diffusion
        # -------------------------
        trajectory = self.diffusion.inpaint_sample_loop(
            self.model, shape, self.device, condition_mask, condition_value
        )

        # -------------------------
        # Denormalize
        # -------------------------
        trajectory = trajectory.cpu().numpy()[0]
        trajectory = self.dataset.denormalize(trajectory)

        actions = recover_actions_from_states(trajectory)
        return trajectory, actions




# ============================================================================
# 6. EVALUATION
# ============================================================================


def evaluate_policy(
    planner, env_name="PointMaze_UMaze-v3", num_episodes=10, max_steps=300
):
    """Evaluate diffuser policy in environment."""
    env = gym.make(env_name)

    successes = []
    returns = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        if isinstance(obs, dict):
            obs = obs["observation"]

        episode_return = 0

        for step in range(max_steps):
            # Plan STATE trajectory, recover actions via differential flatness
            states, actions = planner.plan(obs)

            # Execute first action (receding horizon)
            action = actions[0]
            obs, reward, terminated, truncated, info = env.step(action)

            if isinstance(obs, dict):
                obs = obs["observation"]

            episode_return += reward

            if terminated or truncated:
                break

        successes.append(info.get("success", False))
        returns.append(episode_return)

        print(f"Episode {ep+1}: Return={episode_return:.2f}, Success={successes[-1]}")

    env.close()

    print(f"\nAverage Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Success Rate: {np.mean(successes)*100:.1f}%")

    return returns, successes


# ============================================================================
# MAIN USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load dataset
    print("Loading dataset...")
    dataset = MinariTrajectoryDataset("D4RL/pointmaze/umaze-v2", horizon=32)
    print(f"Dataset loaded: {len(dataset)} trajectory segments")
    print(f"State dim: {dataset.state_dim}, Action dim: {dataset.action_dim}")

    # 2. Create model
    print("\nCreating model...")
    model = TemporalUNet(dataset.state_dim, hidden_dims=[128, 256, 512])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model input/output: state trajectories only (dim={dataset.state_dim})")

    # 3. Create diffusion
    diffusion = GaussianDiffusion(timesteps=100)

    # 4. Train
    print("\nTraining...")
    trainer = DiffuserTrainer(model, diffusion, dataset, device=device)
    trainer.train(epochs=100)

    # 5. Evaluate
    print("\nEvaluating...")
    planner = DiffuserPlanner(
        trainer.ema_model.module, diffusion, dataset, device=device
    )
    evaluate_policy(planner, num_episodes=10)
