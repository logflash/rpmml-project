"""
Improved Diffuser Implementation with Eq-Net Architecture

Based on the paper "What Do You Need for Diverse Trajectory Composition in Diffusion Planning?"
by Quentin Clark and Florian Shkurti (arXiv:2505.18083)
"""

import minari
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from matplotlib import pyplot as plt
# ============================================================================
# DATA
# ============================================================================


class MinariTrajectoryDataset(Dataset):
    """
    Original Minari trajectory dataset with built-in positional augmentation.
    (Baseline version – not used in the pseudo-action setup below.)
    """

    def __init__(self, dataset_name="D4RL/pointmaze/umaze-v2", horizon=32):
        self.horizon = horizon
        self.dataset = minari.load_dataset(dataset_name, download=True)

        self.trajectories = []
        for episode in self.dataset:
            obs = episode.observations
            if isinstance(obs, dict):
                obs = obs["observation"]
            self.trajectories.append(obs)

        self.state_dim = 4
        all_data = np.concatenate(self.trajectories, axis=0)
        self.mean = all_data.mean(axis=0)
        self.std = all_data.std(axis=0) + 1e-8

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
        traj = self.trajectories[traj_idx][start_t: start_t + self.horizon]
        return torch.FloatTensor(self.normalize(traj))


class MinariTrajectoryDatasetWithPseudoActions(Dataset):
    """
    Dataset for loading state (position) and skip-value trajectory sequences from Minari.

    Returned per-window sample shape: (H, 3)
      [:, 0:2] -> normalized positions (x, y)
      [:, 2]   -> normalized skip value (action)

    IMPORTANT: This dataset ALREADY implements positional augmentation!
    """

    def __init__(
        self, dataset_name="D4RL/pointmaze/umaze-v2", horizon=3, normalize=True,
        n_chunks_frac=0.1, alpha=3.0,
    ):
        self.horizon = horizon
        self.n_chunks_frac = n_chunks_frac
        self.alpha = alpha
        self.normalize_flag = normalize

        # Load dataset
        self.dataset = minari.load_dataset(dataset_name, download=True)

        # Containers for normalization
        self.state_trajectories = []
        self.skip_trajectories = []

        # TEMP arrays to gather skip/tau stats
        all_skips = []
        all_taus = []

        # ---------------------------------------------------------
        # Process episodes
        # ---------------------------------------------------------
        for episode in self.dataset:
            obs = episode.observations
            if isinstance(obs, dict):
                obs = obs["observation"]

            states = obs  # full env states, e.g. (x, y, vx, vy)
            T = len(states)
            positions = states[:, :2]

            # --- Dirichlet sampling ---
            n_chunks = max(1, int(self.n_chunks_frac * T))
            chunks = self._dirichlet_chunks(T, n_chunks=n_chunks, alpha=self.alpha)

            # --- Build skip_list ---
            skip_list = []
            tau = 0.0

            for chunk in chunks:
                tau_clamped = min(tau, T - 1)
                idx = int(np.floor(tau_clamped))
                delta_t = tau_clamped - idx

                if idx >= T - 1:
                    pos = positions[-1]
                else:
                    pos = positions[idx] + (positions[idx + 1] - positions[idx]) * delta_t

                skip_list.append((pos, chunk, tau_clamped))

                all_skips.append(chunk)
                all_taus.append(tau_clamped)

                tau += chunk

            self.state_trajectories.append(states)
            self.skip_trajectories.append(skip_list)

        # ---------------------------------------------------------
        # NORMALIZATION for positions (state) and skip (action)
        # ---------------------------------------------------------
        all_states = np.concatenate(self.state_trajectories, axis=0)  # (N, 4)
        all_pos = all_states[:, :2]

        # State = position (x, y)
        self.state_dim = 2
        self.action_dim = 1
        self.traj_dim = self.state_dim + self.action_dim  # 3

        if normalize:
            self.flat_mean = all_pos.mean(axis=0)
            self.flat_std = all_pos.std(axis=0) + 1e-8
        else:
            self.flat_mean = np.zeros(self.state_dim)
            self.flat_std = np.ones(self.state_dim)

        # Skip stats
        self.skip_mean = np.mean(all_skips)
        self.skip_std = np.std(all_skips) + 1e-8

        # Tau stats (not used directly in trajectory, but kept if needed)
        self.tau_mean = np.mean(all_taus)
        self.tau_std = np.std(all_taus) + 1e-8

        # Combined mean/std over full trajectory vector [x, y, skip]
        self.mean = np.zeros(self.traj_dim, dtype=np.float32)
        self.std = np.ones(self.traj_dim, dtype=np.float32)

        self.mean[:self.state_dim] = self.flat_mean
        self.std[:self.state_dim] = self.flat_std
        self.mean[self.state_dim] = self.skip_mean
        self.std[self.state_dim] = self.skip_std

        # ---------------------------------------------------------
        # Build horizon windows
        # ---------------------------------------------------------
        self.indices = []
        for traj_idx, skip_list in enumerate(self.skip_trajectories):
            S = len(skip_list)
            if S >= self.horizon:
                for t in range(S - self.horizon + 1):
                    self.indices.append((traj_idx, t))

    # ---------------------------------------------------------
    def normalize(self, x):
        """Normalize positions (x,y)."""
        return (x - self.flat_mean) / self.flat_std

    def denormalize(self, x):
        """Denormalize positions (x,y)."""
        return x * self.flat_std + self.flat_mean

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
        window = skip_list[start:start + self.horizon]

        positions = np.array([p for (p, c, tau) in window], dtype=np.float32)
        skip_vals = np.array([c for (p, c, tau) in window], dtype=np.float32)
        tau_vals = np.array([tau for (p, c, tau) in window], dtype=np.float32)  # not used downstream

        # ---- Apply normalization ----
        positions = self.normalize(positions)
        skip_vals = (skip_vals - self.skip_mean) / self.skip_std
        # tau_vals_norm = (tau_vals - self.tau_mean) / self.tau_std  # computed but unused here

        # ---- Build final trajectory [H, 3] = [x_norm, y_norm, skip_norm]
        state = np.zeros((self.horizon, self.traj_dim), dtype=np.float32)
        state[:, :self.state_dim] = positions
        state[:, self.state_dim] = skip_vals  # single action channel

        return torch.FloatTensor(state)


# ============================================================================
# REWARD FUNCTIONS (operate on *denormalized* trajectories)
# ============================================================================


class StartReachingReward:
    def __init__(self, start_pos, reward_scale=1.0):
        self.start_pos = torch.tensor(start_pos, dtype=torch.float32)
        self.reward_scale = reward_scale

    def __call__(self, trajectories):
        # trajectories: (batch, horizon, traj_dim) — use only x,y
        initial_pos = trajectories[:, 0, :2]
        start = self.start_pos.to(trajectories.device)
        dist_squared = ((initial_pos - start) ** 2).sum(dim=-1)
        return -dist_squared * self.reward_scale


class GoalReachingReward:
    def __init__(self, goal_pos, reward_scale=1.0):
        self.goal_pos = torch.tensor(goal_pos, dtype=torch.float32)
        self.reward_scale = reward_scale

    def __call__(self, trajectories):
        final_pos = trajectories[:, -1, :2]
        goal = self.goal_pos.to(trajectories.device)
        dist_squared = ((final_pos - goal) ** 2).sum(dim=-1)
        return -dist_squared * self.reward_scale


class PathLengthPenalty:
    def __init__(self, reward_scale=1.0):
        self.reward_scale = reward_scale

    def __call__(self, trajectories):
        diffs = trajectories[:, 1:, :2] - trajectories[:, :-1, :2]
        path_length = (diffs ** 2).sum(dim=-1).sum(dim=-1)
        return -path_length * self.reward_scale


class CompositeReward:
    def __init__(self, reward_fns, weights=None):
        self.reward_fns = reward_fns
        if weights is None:
            weights = [1.0] * len(reward_fns)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __call__(self, trajectories):
        rewards = torch.stack([fn(trajectories) for fn in self.reward_fns], dim=1)
        weights = self.weights.to(trajectories.device)
        return (rewards * weights[None, :]).sum(dim=1)


# ============================================================================
# EQ-NET MODEL
# ============================================================================


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LayerNorm1d(nn.Module):
    """Layer normalization for 1D convolutions"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class LocalResidualBlock(nn.Module):
    """
    Residual block with LOCAL receptive field.
    Uses small kernels and no downsampling to maintain shift equivariance.
    """

    def __init__(self, dim, dim_out, time_emb_dim, kernel_size=3):
        super().__init__()

        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))

        padding = kernel_size // 2
        self.block1 = nn.Sequential(
            LayerNorm1d(dim),
            nn.Mish(),
            nn.Conv1d(dim, dim_out, kernel_size, padding=padding, padding_mode="replicate"),
        )

        self.block2 = nn.Sequential(
            LayerNorm1d(dim_out),
            nn.Mish(),
            nn.Conv1d(dim_out, dim_out, kernel_size, padding=padding, padding_mode="replicate"),
        )

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        time_cond = self.mlp(time_emb)
        h = h + time_cond[:, :, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class EqNet(nn.Module):
    """
    Shift-Equivariant Network with Local Receptiveness for trajectory composition.

    Args:
        state_dim: Dimension of trajectory vector (here: traj_dim = 3)
    """

    def __init__(
        self,
        state_dim,
        hidden_dim=128,
        time_dim=32,
        n_layers=12,
        kernel_size=3,
        use_positional_encoding=False,
    ):
        super().__init__()

        self.use_positional_encoding = use_positional_encoding

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim * 4),
        )

        self.init_conv = nn.Conv1d(
            state_dim,
            hidden_dim,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode="replicate",
        )

        self.blocks = nn.ModuleList(
            [
                LocalResidualBlock(hidden_dim, hidden_dim, time_dim * 4, kernel_size)
                for _ in range(n_layers)
            ]
        )

        if use_positional_encoding:
            self.pos_emb = SinusoidalPosEmb(hidden_dim)

        self.final_conv = nn.Sequential(
            LayerNorm1d(hidden_dim),
            nn.Mish(),
            nn.Conv1d(hidden_dim, state_dim, 1),
        )

    def forward(self, x, time):
        """
        Args:
            x: (batch, horizon, state_dim = traj_dim)
            time: (batch,)
        Returns:
            (batch, horizon, state_dim)
        """
        x = x.transpose(1, 2)
        t_emb = self.time_mlp(time)
        h = self.init_conv(x)

        if self.use_positional_encoding:
            positions = torch.arange(h.shape[2], device=h.device, dtype=torch.float32)
            pos_emb = self.pos_emb(positions)  # (horizon, hidden_dim)
            h = h + pos_emb.T.unsqueeze(0)

        for block in self.blocks:
            h = block(h, t_emb)

        out = self.final_conv(h)
        return out.transpose(1, 2)


# ============================================================================
# DIFFUSION
# ============================================================================


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=200, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)
        )

        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_ac * x0 + sqrt_om * noise

    def predict_x0_from_noise(self, xt, t, noise):
        sqrt_recip = torch.sqrt(1.0 / self.alphas_cumprod[t]).view(-1, 1, 1)
        sqrt_recipm1 = torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(-1, 1, 1)
        return sqrt_recip * xt - sqrt_recipm1 * noise

    def p_mean_variance(self, x0, xt, t):
        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1)
        mean = coef1 * x0 + coef2 * xt
        var = self.posterior_variance[t].view(-1, 1, 1)
        return mean, var

    def p_losses(self, model, x0, t):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        predicted_noise = model(xt, t)
        return F.mse_loss(predicted_noise, noise)

    def p_sample_guided(
        self,
        model,
        xt,
        t,
        reward_fn,
        dataset,
        guidance_scale,
        mask=None,
        x0_known=None,
        soft_conditioning_weight=None,
    ):
        """Guided sampling with soft conditioning"""

        # Step 1: Compute guidance (if enabled)
        if reward_fn is not None and guidance_scale > 0:
            xt = xt.requires_grad_(True)

            with torch.enable_grad():
                predicted_noise = model(xt, t)
                x0_pred = self.predict_x0_from_noise(xt, t, predicted_noise)
                x0_pred = x0_pred.clamp(-3, 3)

                # Denormalize using dataset.mean/std over traj_dim
                mean = torch.tensor(dataset.mean, device=xt.device, dtype=xt.dtype)[
                    None, None, :
                ]
                std = torch.tensor(dataset.std, device=xt.device, dtype=xt.dtype)[
                    None, None, :
                ]
                x0_denorm = x0_pred * std + mean

                rewards = reward_fn(x0_denorm)
                grad = torch.autograd.grad(rewards.sum(), xt)[0]

            grad_norm = torch.norm(grad)
            if grad_norm > 1.0:
                grad = grad / grad_norm * 1.0

            xt = xt.detach() + guidance_scale * grad

        # Step 2: Regular denoising
        xt = xt.detach()
        predicted_noise = model(xt, t)
        x0_pred = self.predict_x0_from_noise(xt, t, predicted_noise)
        x0_pred = x0_pred.clamp(-3, 3)

        # Step 3: Soft conditioning
        if (
            mask is not None
            and x0_known is not None
            and soft_conditioning_weight is not None
        ):
            correction = x0_known - x0_pred
            x0_pred = x0_pred + soft_conditioning_weight * mask.float() * correction

        mean, var = self.p_mean_variance(x0_pred, xt, t)

        if (t == 0).all():
            return mean

        noise = torch.randn_like(xt)
        xt_next = mean + torch.sqrt(var) * noise
        return xt_next

    def sample_guided(
        self,
        model,
        shape,
        device,
        reward_fn,
        dataset,
        guidance_scale,
        condition_mask=None,
        condition_value=None,
        conditioning_schedule="cosine",
        conditioning_strength=1.0,
    ):
        """Sample with soft conditioning and annealing schedule"""
        xt = torch.randn(shape, device=device)
        model.eval()

        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)

            progress = i / self.timesteps  # 1.0 at t=T, 0.0 at t=0

            if conditioning_schedule == "linear":
                weight = conditioning_strength * progress
            elif conditioning_schedule == "cosine":
                weight = conditioning_strength * (
                    0.5 * (1 + np.cos(np.pi * (1 - progress)))
                )
            else:
                weight = conditioning_strength

            xt = self.p_sample_guided(
                model,
                xt,
                t,
                reward_fn,
                dataset,
                guidance_scale,
                mask=condition_mask,
                x0_known=condition_value,
                soft_conditioning_weight=weight,
            )

        return xt


# ============================================================================
# EMA
# ============================================================================


class EMA:
    def __init__(self, model, beta=0.999):
        self.model = model
        self.beta = beta
        self.shadow = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.beta).add_(param.data, alpha=1 - self.beta)

    def copy_to(self, model):
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])


# ============================================================================
# TRAINING
# ============================================================================


class DiffuserTrainer:
    def __init__(
        self,
        model,
        diffusion,
        dataset,
        lr=1e-4,
        device="cuda",
        use_ema=True,
        ema_beta=0.999,
    ):
        self.model = model.to(device)
        self.diffusion = diffusion.to(device)
        self.device = device
        self.use_ema = use_ema

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        if use_ema:
            self.ema = EMA(model, beta=ema_beta)
        else:
            self.ema = None

        self.dataloader = DataLoader(
            dataset, batch_size=128, shuffle=True, num_workers=4
        )

    def train_step(self, batch):
        batch = batch.to(self.device)
        t = torch.randint(
            0, self.diffusion.timesteps, (batch.shape[0],), device=self.device
        ).long()

        loss = self.diffusion.p_losses(self.model, batch, t)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.ema is not None:
            self.ema.update(self.model)

        return loss.item()

    def train(self, epochs=100, save_every=10):
        self.model.train()

        for epoch in range(epochs):
            losses = []
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in pbar:
                loss = self.train_step(batch)
                losses.append(loss)
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "avg_loss": f"{np.mean(losses):.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            avg_loss = np.mean(losses)
            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Loss = {avg_loss:.4f}, "
                f"LR = {self.scheduler.get_last_lr()[0]:.2e}"
            )

            self.scheduler.step()

            if save_every > 0 and (epoch + 1) % save_every == 0:
                import os

                checkpoint_path = f"checkpoints/diffuser_eqnet_epoch_{epoch + 1}.pt"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                self.save_checkpoint(checkpoint_path)
                print(f"  → Saved checkpoint to {checkpoint_path}")

    def save_checkpoint(self, path):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "ema_shadow" in checkpoint and self.ema is not None:
            self.ema.shadow = checkpoint["ema_shadow"]

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
            if "ema_shadow" in checkpoint and self.ema is not None:
                self.ema.shadow = checkpoint["ema_shadow"]
        else:
            self.model.load_state_dict(checkpoint)

    def use_ema_for_inference(self):
        if self.ema is not None:
            self.ema.copy_to(self.model)
            print("Switched to EMA parameters for inference")
        else:
            print("Warning: EMA not enabled, using regular parameters")


# ============================================================================
# PLANNING
# ============================================================================


class DiffuserPlanner:
    def __init__(self, model, diffusion, dataset, device="cuda"):
        self.model = model.to(device)
        self.diffusion = diffusion.to(device)
        self.dataset = dataset
        self.device = device

    @torch.no_grad()
    def plan_and_reconstruct(
        self,
        current_obs,
        goal_obs=None,
        horizon=32,
        reward_fn=None,
        guidance_scale=1.0,
        condition_on_start=True,
        condition_on_goal=False,
        conditioning_schedule="cosine",
        conditioning_strength=0.5,
        spline_func=None,
        
    ):
        """
        Combined planner + reconstruction:
        1. Perform diffusion planning (coarse trajectory of H points)
        2. Convert (pos, skip) into skip_list format
        3. Run skip-based spline expansion to produce dense trajectory
        4. (Optional) visualize
        """

       

        # ------------------------------------------------------------------
        # 1. DIFFUSER PLANNING (same computations as plan(), fully inlined)
        # ------------------------------------------------------------------
        self.model.eval()

        # parse observation
        current_obs = np.asarray(current_obs, dtype=np.float32).reshape(-1)
        current_pos = current_obs[: self.dataset.state_dim]

        # normalize start
        start_norm_pos = (current_pos - self.dataset.flat_mean) / self.dataset.flat_std
        start_norm_full = np.zeros(self.dataset.traj_dim, dtype=np.float32)
        start_norm_full[: self.dataset.state_dim] = start_norm_pos

        start_norm_full = torch.tensor(start_norm_full, device=self.device)

        # build conditioning tensors
        shape = (1, horizon, self.dataset.traj_dim)
        condition_mask = torch.zeros(shape, dtype=torch.bool, device=self.device)
        condition_value = torch.zeros(shape, dtype=torch.float32, device=self.device)

        # condition on start
        if condition_on_start:
            condition_mask[0, 0, : self.dataset.state_dim] = True
            condition_value[0, 0, : self.dataset.state_dim] = start_norm_full[: self.dataset.state_dim]

        # condition on goal
        if goal_obs is not None and condition_on_goal:
            goal_obs = np.asarray(goal_obs, dtype=np.float32).flatten()
            goal_pos = goal_obs[: self.dataset.state_dim]

            goal_norm_pos = (goal_pos - self.dataset.flat_mean) / self.dataset.flat_std
            goal_norm_full = np.zeros(self.dataset.traj_dim, dtype=np.float32)
            goal_norm_full[: self.dataset.state_dim] = goal_norm_pos
            goal_norm_full = torch.tensor(goal_norm_full, device=self.device)

            condition_mask[0, -1, : self.dataset.state_dim] = True
            condition_value[0, -1, : self.dataset.state_dim] = goal_norm_full[: self.dataset.state_dim]

        # run guided diffusion
        coarse_norm = self.diffusion.sample_guided(
            self.model,
            shape,
            self.device,
            reward_fn,
            self.dataset,
            guidance_scale,
            condition_mask=condition_mask,
            condition_value=condition_value,
            conditioning_schedule=conditioning_schedule,
            conditioning_strength=conditioning_strength,
        )

        coarse_norm = coarse_norm.cpu().numpy()[0]  # (H,3)

        # ------------------------------------------------------------------
        # 2. DENORMALIZE (pos, skip)
        # ------------------------------------------------------------------
        pos_norm = coarse_norm[:, : self.dataset.state_dim]
        skip_norm = coarse_norm[:, self.dataset.state_dim]

        pos = pos_norm * self.dataset.flat_std + self.dataset.flat_mean
        skip = skip_norm * self.dataset.skip_std + self.dataset.skip_mean

        coarse = np.zeros_like(coarse_norm)
        coarse[:, : self.dataset.state_dim] = pos
        coarse[:, self.dataset.state_dim] = skip

        # ------------------------------------------------------------------
        # 3. Convert to skip_list format 
        # ------------------------------------------------------------------
        skip_list = [(pos[i], float(skip[i])) for i in range(len(pos))]

        # ------------------------------------------------------------------
        # 4. Run skip-based spline expansion
        # ------------------------------------------------------------------
        pos_dense, vel_dense, acc_dense = spline_func(skip_list)


        # ------------------------------------------------------------------
        # 5. Return everything
        # ------------------------------------------------------------------
        return dict(
            coarse=coarse,                 # (H,3)
            coarse_pos=pos,                # (H,2)
            coarse_skip=skip,              # (H,)
            skip_list=skip_list,           # [(pos_i, skip_i)]
            pos_dense=pos_dense,           # (N,2)
            vel_dense=vel_dense,           # (N,2)
            acc_dense=acc_dense,           # (N,2)
        )
    
    @torch.no_grad()
    def plan_with_diversity(self, current_obs, goal_obs, num_samples=10, **kwargs):
        """
        Generate multiple diverse trajectories and select best.
        """
        trajectories = []
        rewards = []

        reward_fn = kwargs.get("reward_fn", None)

        for _ in range(num_samples):
            traj = self.plan(current_obs, goal_obs, **kwargs)

            if reward_fn is not None:
                traj_tensor = torch.tensor(
                    traj, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                reward = reward_fn(traj_tensor).item()
            else:
                goal_dist = np.linalg.norm(traj[-1, :2] - goal_obs[:2])
                reward = -goal_dist

            trajectories.append(traj)
            rewards.append(reward)

        best_idx = np.argmax(rewards)
        return trajectories[best_idx], trajectories, rewards

def expand_spline_from_skip_list(skip_list, dt=0.01):
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

        num_samples = np.ceil(k).astype(int) + 1  # +1 to include endpoint

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

def run_sanity_check(dataset, diffusion, model, device="cpu", batch_idx=0):
    print("\n" + "="*80)
    print("SANITY CHECK: Dataset Normalization Consistency")
    print("="*80)

    # ----------------------------------------------------------------------------------
    # 1. Grab a raw window from dataset the exact way __getitem__ uses it
    # ----------------------------------------------------------------------------------
    sample_norm = dataset[batch_idx]              # normalized window: (H, 3)
    sample_norm_np = sample_norm.numpy()

    print("\nNormalized window [x_norm, y_norm, skip_norm]:")
    print(sample_norm_np)

    # Denormalize manually (matching DiffuserPlanner logic)
    pos_norm = sample_norm_np[:, :dataset.state_dim]
    skip_norm = sample_norm_np[:, dataset.state_dim]

    pos_denorm = pos_norm * dataset.flat_std + dataset.flat_mean
    skip_denorm = skip_norm * dataset.skip_std + dataset.skip_mean

    sample_denorm = np.zeros_like(sample_norm_np)
    sample_denorm[:, :dataset.state_dim] = pos_denorm
    sample_denorm[:, dataset.state_dim] = skip_denorm

    print("\nReconstructed (denormalized) window:")
    print(sample_denorm)

    # Check consistency: should be nearly equal
    print("\nError statistics (denorm(norm(x)) - x_raw):")
    raw_positions = dataset.skip_trajectories[ dataset.indices[batch_idx][0] ]
    start = dataset.indices[batch_idx][1]
    raw_window = raw_positions[start:start+dataset.horizon]

    raw_pos = np.array([p for (p,c,tau) in raw_window])
    raw_skip = np.array([c for (p,c,tau) in raw_window])

    # Compare
    pos_err = np.abs(raw_pos - pos_denorm).mean()
    skip_err = np.abs(raw_skip - skip_denorm).mean()

    print(f"  mean position error: {pos_err:.8f}")
    print(f"  mean skip error    : {skip_err:.8f}")

    # ----------------------------------------------------------------------------------
    # 2. Run a single q-sample diffusion step (just to check no shape errors)
    # ----------------------------------------------------------------------------------
    print("\n" + "="*80)
    print("SANITY CHECK: Single Diffusion Step")
    print("="*80)

    model = model.to(device)
    diffusion = diffusion.to(device)

    batch = sample_norm.unsqueeze(0).to(device)   # shape (1, H, 3)
    t = torch.tensor([ diffusion.timesteps // 2 ], dtype=torch.long, device=device)

    with torch.no_grad():
        # This simulates a single forward diffusion step
        noise = torch.randn_like(batch)
        xt = diffusion.q_sample(batch, t, noise)

    print("\nxt (sample after one q-sample step):")
    print(xt.cpu().numpy()[0])

    print(f"\nShapes:")
    print(f"  batch: {batch.shape}")
    print(f"  xt   : {xt.shape}  (should be same)")

    # ----------------------------------------------------------------------------------
    # 3. Verify denorm(norm(x)) ≈ x for all dims
    # ----------------------------------------------------------------------------------
    print("\n" + "="*80)
    print("FINAL CHECK: norm→denorm identity")
    print("="*80)

    recon_norm = (sample_denorm - dataset.mean) / dataset.std
    recon_denorm = recon_norm * dataset.std + dataset.mean

    err_full = np.abs(recon_denorm - sample_denorm).mean()
    print(f"Mean reconstruction error over full (x,y,skip): {err_full:.10f}")

    if err_full < 1e-5:
        print("✓ PASSED: normalization pipeline is internally consistent.")
    else:
        print("⚠️ WARNING: inconsistency detected.")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    USE_POSITIONAL_ENCODING = False

    print("\n" + "=" * 80)
    print("EQ-NET CONFIGURATION")
    print("=" * 80)
    print(f"Positional Encoding: {USE_POSITIONAL_ENCODING}")
    print("Architecture: Local convolutions (no downsampling) for shift equivariance")
    print("Using MinariTrajectoryDatasetWithPseudoActions (pos + skip as action)")

    # ========================================================================
    # DATA
    # ========================================================================

    # Use pseudo-action dataset
    dataset = MinariTrajectoryDatasetWithPseudoActions(
        "D4RL/pointmaze/umaze-v2", horizon=3
    )
    print(f"\nDataset size: {len(dataset)} trajectory windows")
    print(f"Trajectory dim (pos+skip): {dataset.traj_dim}")

    # ========================================================================
    # MODEL
    # ========================================================================

    model = EqNet(
        state_dim=dataset.traj_dim,  # 3 (x, y, skip)
        hidden_dim=128,
        time_dim=32,
        n_layers=12,
        kernel_size=3,
        use_positional_encoding=USE_POSITIONAL_ENCODING,
    )

    diffusion = GaussianDiffusion(timesteps=200)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========================================================================
    # TRAINING
    # ========================================================================

    trainer = DiffuserTrainer(
        model,
        diffusion,
        dataset,
        device=device,
        use_ema=True,
        ema_beta=0.999,
    )

    TRAIN_NEW = False
    if TRAIN_NEW:
        print("\n" + "=" * 80)
        print("TRAINING")
        print("=" * 80)
        trainer.train(epochs=100, save_every=1)
    else:
        print("\nLoading pre-trained model...")
        try:
            trainer.load("checkpoints/diffuser_eqnet_epoch_100.pt")
            print("Loaded checkpoint successfully")
        except:
            print("No checkpoint found, using random initialization")

    trainer.use_ema_for_inference()

    # ========================================================================
    # PLANNING
    # ========================================================================

    planner = DiffuserPlanner(model, diffusion, dataset, device=device)

    current = np.array([1.06591915, 0.39449871, 0.88507522, 4.78210334])
    goal = np.array([0.55692841, 1.02245092])

    reward_fn = CompositeReward(
        [
            StartReachingReward(current[:2], reward_scale=5.0),
            GoalReachingReward(goal, reward_scale=5.0),
            PathLengthPenalty(reward_scale=0.1),
        ]
    )

    print("\n" + "=" * 80)
    print("PLANNING WITH EQ-NET (pos + skip)")
    print("=" * 80)

    # Experiment 1: standard planning
    print("\n--- Experiment 1: Standard Planning ---")
    traj = planner.plan_and_reconstruct(
        current,
        goal,
        reward_fn=reward_fn,
        guidance_scale=2.0,
        condition_on_start=True,
        condition_on_goal=False,
        conditioning_schedule="cosine",
        conditioning_strength=0.5,
        spline_func=expand_spline_from_skip_list,
    )
    coarse, = traj["coarse"],
    coarse_pos= traj["coarse_pos"] 
    coarse_skip=traj["coarse_skip"]              # (H,)
    skip = traj["skip_list"]     # [(pos_i, skip_i)]
    pos_dense=traj["pos_dense"]           # (N,2)
    vel_dense=traj["vel_dense"]           # (N,2)
    acc_dense=traj["acc_dense"]           # (N,2)  
    print(f"Start (actual): {pos_dense[0]}")
    print(f"Start (target): {current[:2]}")
    print(f"Start error: {np.linalg.norm(pos_dense[0] - current[:2]):.4f}")
    print(f"End: {pos_dense[-1]}")
    print(f"Goal: {goal}")
    print(f"Goal error: {np.linalg.norm(pos_dense[-1] - goal):.4f}")
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(pos_dense)
    # ============================================================
    # 3. PLOT TRAJECTORY + MUJOCO MAZE WALLS
    # ============================================================
   
    print("[")
    for row in pos_dense:
        print(f"    [{row[0]}, {row[1]}],")
    print("]")

    print("[")
    for row in coarse_pos:
        print(f"    [{row[0]}, {row[1]}],")
    print("]")

    print("[")
    for row in coarse_skip:
        print(f"{row},")
    print("]")