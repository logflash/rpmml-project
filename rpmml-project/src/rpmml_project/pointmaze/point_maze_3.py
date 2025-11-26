"""
Improved Diffuser Implementation with Eq-Net Architecture

Based on the paper "What Do You Need for Diverse Trajectory Composition in Diffusion Planning?"
by Quentin Clark and Florian Shkurti (arXiv:2505.18083)

Key improvements over standard U-Net diffusion models:
1. Eq-Net architecture: No downsampling/upsampling (maintains shift equivariance)
2. Local receptive fields: Small conv kernels for local processing
3. Soft conditioning: Respects learned physics during sampling
4. Built-in positional augmentation: Sliding window in dataset naturally provides this
5. EMA (Exponential Moving Average): Provides more stable and higher-quality inference

IMPORTANT NOTE ON POSITIONAL AUGMENTATION:
------------------------------------------
The original MinariTrajectoryDataset already implements positional augmentation
through its sliding window approach. When it creates indices for all possible
(trajectory, start_time) pairs, this naturally provides temporal shifts.

With DataLoader shuffling, each epoch sees the same trajectory segments at
different temporal offsets - exactly what the paper describes as "positional
augmentation". No explicit augmentation needed!

IMPORTANT NOTE ON EMA:
----------------------
EMA maintains a moving average of model parameters during training. At inference
time, using these averaged parameters typically produces better quality and more
stable results than using the raw trained parameters. Always call
trainer.use_ema_for_inference() before planning!
"""

import minari
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ============================================================================
# DATA
# ============================================================================


class MinariTrajectoryDataset(Dataset):
    """
    Minari trajectory dataset with built-in positional augmentation.

    IMPORTANT: This dataset ALREADY implements positional augmentation!
    -------------------------------------------------------------------
    The sliding window approach (for t in range(len(traj) - horizon + 1))
    creates indices for all possible temporal offsets. When combined with
    DataLoader shuffling, this naturally provides positional augmentation
    as described in the paper.

    Example trajectory of length 10 with horizon=4:
        Creates indices: [(traj_id, 0), (traj_id, 1), ..., (traj_id, 6)]
        Sample 1: [s0, s1, s2, s3]  (offset 0)
        Sample 2: [s1, s2, s3, s4]  (offset 1)
        Sample 3: [s2, s3, s4, s5]  (offset 2)
        etc.

    This is exactly the "positional augmentation" strategy from the paper,
    and it's more efficient than explicit augmentation schemes.
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
        traj = self.trajectories[traj_idx][start_t : start_t + self.horizon]
        return torch.FloatTensor(self.normalize(traj))


# ============================================================================
# REWARD FUNCTIONS
# ============================================================================


class StartReachingReward:
    def __init__(self, start_pos, reward_scale=1.0):
        self.start_pos = torch.tensor(start_pos, dtype=torch.float32)
        self.reward_scale = reward_scale

    def __call__(self, trajectories):
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
        path_length = (diffs**2).sum(dim=-1).sum(dim=-1)
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
# EQ-NET MODEL (Shift-Equivariant with Local Receptiveness)
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

        # Time embedding MLP
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))

        # Two conv layers with small kernels (LOCAL receptiveness)
        padding = kernel_size // 2
        self.block1 = nn.Sequential(
            LayerNorm1d(dim),
            nn.Mish(),
            nn.Conv1d(
                dim, dim_out, kernel_size, padding=padding, padding_mode="replicate"
            ),
        )

        self.block2 = nn.Sequential(
            LayerNorm1d(dim_out),
            nn.Mish(),
            nn.Conv1d(
                dim_out, dim_out, kernel_size, padding=padding, padding_mode="replicate"
            ),
        )

        # Residual connection
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)

        # Add time conditioning
        time_cond = self.mlp(time_emb)
        h = h + time_cond[:, :, None]

        h = self.block2(h)
        return h + self.res_conv(x)


class EqNet(nn.Module):
    """
    Shift-Equivariant Network with Local Receptiveness for trajectory composition.

    Key properties from the paper:
    1. NO downsampling/upsampling - maintains perfect shift equivariance
    2. Local receptive field - uses small conv kernels
    3. Can optionally add positional encoding to break equivariance (for locality without full equivariance)

    Args:
        state_dim: Dimension of state space
        hidden_dim: Hidden layer dimension
        time_dim: Time embedding dimension
        n_layers: Number of residual blocks
        kernel_size: Convolutional kernel size (keep small for locality)
        use_positional_encoding: If True, adds position encoding (makes local but not equivariant)
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

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim * 4),
        )

        # Initial projection
        self.init_conv = nn.Conv1d(
            state_dim,
            hidden_dim,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode="replicate",
        )

        # Stack of local residual blocks (NO downsampling!)
        self.blocks = nn.ModuleList(
            [
                LocalResidualBlock(hidden_dim, hidden_dim, time_dim * 4, kernel_size)
                for _ in range(n_layers)
            ]
        )

        # Optional positional encoding
        if use_positional_encoding:
            self.pos_emb = SinusoidalPosEmb(hidden_dim)

        # Final projection
        self.final_conv = nn.Sequential(
            LayerNorm1d(hidden_dim), nn.Mish(), nn.Conv1d(hidden_dim, state_dim, 1)
        )

    def forward(self, x, time):
        """
        Args:
            x: (batch, horizon, state_dim)
            time: (batch,)
        Returns:
            (batch, horizon, state_dim)
        """
        # Transpose to (batch, state_dim, horizon) for 1D conv
        x = x.transpose(1, 2)

        # Time embedding
        t_emb = self.time_mlp(time)

        # Initial projection
        h = self.init_conv(x)

        # Add positional encoding if using (breaks shift equivariance for locality)
        if self.use_positional_encoding:
            positions = torch.arange(h.shape[2], device=h.device, dtype=torch.float32)
            pos_emb = self.pos_emb(positions)  # (horizon, hidden_dim)
            h = h + pos_emb.T.unsqueeze(0)  # Broadcast to (batch, hidden_dim, horizon)

        # Apply residual blocks
        for block in self.blocks:
            h = block(h, t_emb)

        # Final projection
        out = self.final_conv(h)

        # Transpose back to (batch, horizon, state_dim)
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

                # Denormalize
                mean = torch.tensor(dataset.mean, device=xt.device, dtype=xt.dtype)[
                    None, None, :
                ]
                std = torch.tensor(dataset.std, device=xt.device, dtype=xt.dtype)[
                    None, None, :
                ]
                x0_denorm = x0_pred * std + mean

                # Compute reward and gradient
                rewards = reward_fn(x0_denorm)
                grad = torch.autograd.grad(rewards.sum(), xt)[0]

            # Clip gradient for stability
            grad_norm = torch.norm(grad)
            if grad_norm > 1.0:
                grad = grad / grad_norm * 1.0

            # Apply guidance
            xt = xt.detach() + guidance_scale * grad

        # Step 2: Regular denoising step
        xt = xt.detach()
        predicted_noise = model(xt, t)
        x0_pred = self.predict_x0_from_noise(xt, t, predicted_noise)
        x0_pred = x0_pred.clamp(-3, 3)

        # Step 3: SOFT CONDITIONING - apply as weighted correction
        if (
            mask is not None
            and x0_known is not None
            and soft_conditioning_weight is not None
        ):
            # Compute how far we are from the conditioning target
            correction = x0_known - x0_pred

            # Apply weighted correction only at masked positions
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

            # Compute conditioning weight for this timestep
            progress = i / self.timesteps  # 1.0 at t=T, 0.0 at t=0

            if conditioning_schedule == "linear":
                weight = conditioning_strength * progress
            elif conditioning_schedule == "cosine":
                weight = conditioning_strength * (
                    0.5 * (1 + np.cos(np.pi * (1 - progress)))
                )
            else:  # constant
                weight = conditioning_strength

            # Apply soft conditioning
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
# EMA (Exponential Moving Average)
# ============================================================================


class EMA:
    """
    Exponential Moving Average for model parameters.

    EMA maintains a shadow copy of model parameters that is updated with:
        shadow = beta * shadow + (1 - beta) * current_param

    This provides more stable predictions and better generalization.
    Higher beta (e.g., 0.999) means slower updates and more stability.
    """

    def __init__(self, model, beta=0.999):
        self.model = model
        self.beta = beta
        self.shadow = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

    def update(self, model):
        """Update shadow parameters with current model parameters"""
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.beta).add_(param.data, alpha=1 - self.beta)

    def copy_to(self, model):
        """Copy shadow parameters to model (for inference)"""
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])


# ============================================================================
# TRAINING
# ============================================================================


class DiffuserTrainer:
    """
    Improved trainer with EMA, learning rate scheduling, and gradient clipping.

    Features:
    - EMA for stable inference
    - Cosine annealing LR schedule
    - Gradient clipping for stability
    - Real-time loss and LR display
    - Checkpoint saving with all training state
    """

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

        # Cosine annealing schedule
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        # EMA (optional but recommended)
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

        # Update EMA if enabled
        if self.ema is not None:
            self.ema.update(self.model)

        return loss.item()

    def train(self, epochs=100, save_every=10):
        """
        Train the model with real-time progress display.

        Args:
            epochs: Number of training epochs
            save_every: Save checkpoint every N epochs (0 to disable)
        """
        self.model.train()

        for epoch in range(epochs):
            losses = []
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                loss = self.train_step(batch)
                losses.append(loss)

                # Update progress bar with loss and learning rate
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "avg_loss": f"{np.mean(losses):.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            avg_loss = np.mean(losses)
            print(
                f"Epoch {epoch+1}/{epochs}: "
                f"Loss = {avg_loss:.4f}, "
                f"LR = {self.scheduler.get_last_lr()[0]:.2e}"
            )

            # Step the scheduler
            self.scheduler.step()

            # Save checkpoint periodically
            if save_every > 0 and (epoch + 1) % save_every == 0:
                checkpoint_path = f"checkpoints/diffuser_eqnet_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path)
                print(f"  → Saved checkpoint to {checkpoint_path}")

    def save_checkpoint(self, path):
        """Save complete training state including EMA"""
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load complete training state including EMA"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if "ema_shadow" in checkpoint and self.ema is not None:
            self.ema.shadow = checkpoint["ema_shadow"]

    def load(self, path):
        """
        Simple load (backward compatible with old code).
        Handles both old format (just state_dict) and new format (full checkpoint).
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Handle both formats
        if "model" in checkpoint:
            # New format: full checkpoint
            self.model.load_state_dict(checkpoint["model"])

            if "ema_shadow" in checkpoint and self.ema is not None:
                self.ema.shadow = checkpoint["ema_shadow"]
        else:
            # Old format: direct state_dict
            self.model.load_state_dict(checkpoint)

    def use_ema_for_inference(self):
        """
        Switch to EMA parameters for inference (better quality).
        Call this before planning/evaluation!
        """
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
    def plan(
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
    ):
        """Plan trajectory from current state to goal"""
        self.model.eval()

        # Normalize current state
        current_obs = np.asarray(current_obs, dtype=np.float32).reshape(-1)
        start_norm = self.dataset.normalize(current_obs)
        start_norm = torch.tensor(start_norm, dtype=torch.float32, device=self.device)

        # Build conditioning
        shape = (1, horizon, self.dataset.state_dim)
        condition_mask = torch.zeros(shape, dtype=torch.bool, device=self.device)
        condition_value = torch.zeros(shape, dtype=torch.float32, device=self.device)

        # Always condition on start
        if condition_on_start:
            condition_mask[0, 0, :] = True
            condition_value[0, 0, :] = start_norm

        # Optionally condition on goal
        if goal_obs is not None and condition_on_goal:
            goal_obs = np.asarray(goal_obs, dtype=np.float32).flatten()
            if len(goal_obs) == 2:
                goal_state = np.zeros(self.dataset.state_dim, dtype=np.float32)
                goal_state[:2] = goal_obs
                goal_norm = self.dataset.normalize(goal_state)
            else:
                goal_norm = self.dataset.normalize(goal_obs)

            goal_norm = torch.tensor(goal_norm, dtype=torch.float32, device=self.device)
            condition_mask[0, -1, :] = True
            condition_value[0, -1, :] = goal_norm

        # Sample trajectory with soft conditioning
        trajectory = self.diffusion.sample_guided(
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

        # Denormalize and return
        trajectory = self.dataset.denormalize(trajectory.cpu().numpy()[0])
        return trajectory

    @torch.no_grad()
    def plan_with_diversity(self, current_obs, goal_obs, num_samples=10, **kwargs):
        """
        Generate multiple diverse trajectories and select best.
        As shown in the paper, this can be more effective than replanning.
        """
        trajectories = []
        rewards = []

        reward_fn = kwargs.get("reward_fn", None)

        for _ in range(num_samples):
            traj = self.plan(current_obs, goal_obs, **kwargs)

            if reward_fn is not None:
                # Convert to tensor for reward computation
                traj_tensor = torch.tensor(
                    traj, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                reward = reward_fn(traj_tensor).item()
            else:
                # Default: use goal distance as reward
                goal_dist = np.linalg.norm(traj[-1, :2] - goal_obs[:2])
                reward = -goal_dist

            trajectories.append(traj)
            rewards.append(reward)

        best_idx = np.argmax(rewards)
        return trajectories[best_idx], trajectories, rewards


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    # Model configuration based on paper findings
    USE_POSITIONAL_ENCODING = False  # Set True for locality without full equivariance

    print("\n" + "=" * 80)
    print("EQ-NET CONFIGURATION")
    print("=" * 80)
    print(f"Positional Encoding: {USE_POSITIONAL_ENCODING}")
    print("Architecture: Local convolutions (no downsampling) for shift equivariance")
    print("Note: Positional augmentation is ALREADY built into MinariTrajectoryDataset")
    print("      (sliding window creates naturally shifted trajectory samples)")

    # ========================================================================
    # DATA
    # ========================================================================

    dataset = MinariTrajectoryDataset("D4RL/pointmaze/umaze-v2", horizon=32)
    print(f"\nDataset size: {len(dataset)} trajectory windows")

    # ========================================================================
    # MODEL
    # ========================================================================

    # Eq-Net: shift-equivariant architecture with local receptiveness
    model = EqNet(
        state_dim=dataset.state_dim,
        hidden_dim=128,
        time_dim=32,
        n_layers=12,  # Deep but local
        kernel_size=3,  # Small kernel for locality
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
        use_ema=True,  # Enable EMA for better inference
        ema_beta=0.999,  # Higher = more stable
    )

    # Train or load
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

    # IMPORTANT: Switch to EMA parameters for inference (better quality!)
    if trainer.use_ema:
        trainer.use_ema_for_inference()
        print("Using EMA parameters for planning")

    # ========================================================================
    # PLANNING
    # ========================================================================

    planner = DiffuserPlanner(model, diffusion, dataset, device=device)

    current = np.array([1.06591915, 0.39449871, 0.88507522, 4.78210334])
    goal = np.array([0.55692841, 1.02245092])

    # Create reward function
    reward_fn = CompositeReward(
        [
            StartReachingReward(current[:2], reward_scale=5.0),
            GoalReachingReward(goal, reward_scale=5.0),
            PathLengthPenalty(reward_scale=0.1),
        ]
    )

    print("\n" + "=" * 80)
    print("PLANNING WITH EQ-NET")
    print("=" * 80)

    # ========================================================================
    # EXPERIMENT 1: Effect of positional equivariance
    # ========================================================================

    print("\n--- Experiment 1: Standard Planning ---")
    traj = planner.plan(
        current,
        goal,
        reward_fn=reward_fn,
        guidance_scale=2.0,
        condition_on_start=True,
        condition_on_goal=False,
        conditioning_schedule="cosine",
        conditioning_strength=0.5,
    )

    print(f"Start (actual): {traj[0, :2]}")
    print(f"Start (target): {current[:2]}")
    print(f"Start error: {np.linalg.norm(traj[0, :2] - current[:2]):.4f}")
    print(f"End: {traj[-1, :2]}")
    print(f"Goal: {goal}")
    print(f"Goal error: {np.linalg.norm(traj[-1, :2] - goal):.4f}")

    # Check for discontinuities
    diffs = np.linalg.norm(traj[1:, :2] - traj[:-1, :2], axis=1)
    max_jump = diffs.max()
    mean_step = diffs.mean()
    print(f"Max step size: {max_jump:.4f}")
    print(f"Mean step size: {mean_step:.4f}")
    if max_jump > 0.3:
        print("  ⚠️  WARNING: Large discontinuity detected!")
    else:
        print("  ✓ Trajectory appears continuous")

    # ========================================================================
    # EXPERIMENT 2: Diverse sampling (paper shows this can beat replanning)
    # ========================================================================

    print("\n--- Experiment 2: Diverse Sampling (10 samples) ---")
    best_traj, all_trajs, all_rewards = planner.plan_with_diversity(
        current,
        goal,
        num_samples=10,
        reward_fn=reward_fn,
        guidance_scale=2.0,
        condition_on_start=True,
        conditioning_schedule="cosine",
        conditioning_strength=0.5,
    )

    print(f"Best trajectory reward: {max(all_rewards):.4f}")
    print(f"Worst trajectory reward: {min(all_rewards):.4f}")
    print(f"Mean reward: {np.mean(all_rewards):.4f}")
    print(f"\nBest trajectory:")
    print(f"  Start error: {np.linalg.norm(best_traj[0, :2] - current[:2]):.4f}")
    print(f"  Goal error: {np.linalg.norm(best_traj[-1, :2] - goal):.4f}")

    # Check continuity
    diffs = np.linalg.norm(best_traj[1:, :2] - best_traj[:-1, :2], axis=1)
    print(f"  Max step size: {diffs.max():.4f}")
    print(f"  Mean step size: {diffs.mean():.4f}")

    # ========================================================================
    # EXPERIMENT 3: Effect of conditioning strength
    # ========================================================================

    print("\n--- Experiment 3: Conditioning Strength Comparison ---")
    for strength in [0.0, 0.3, 0.5, 0.7, 1.0]:
        traj = planner.plan(
            current,
            goal,
            reward_fn=reward_fn,
            guidance_scale=2.0,
            condition_on_start=True,
            conditioning_schedule="cosine",
            conditioning_strength=strength,
        )

        start_err = np.linalg.norm(traj[0, :2] - current[:2])
        goal_err = np.linalg.norm(traj[-1, :2] - goal)
        diffs = np.linalg.norm(traj[1:, :2] - traj[:-1, :2], axis=1)
        max_jump = diffs.max()

        print(
            f"\nStrength {strength:.1f}: start_err={start_err:.4f}, "
            f"goal_err={goal_err:.4f}, max_jump={max_jump:.4f}"
        )
        if max_jump > 0.3:
            print("  ⚠️  Physics violation")

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80)
    print("\nKey takeaways from the paper:")
    print("1. Local receptiveness + shift equivariance enable trajectory stitching")
    print("2. Diverse sampling can be as effective as replanning (but faster)")
    print("3. Soft conditioning respects learned physics better than hard constraints")
    print("4. Positional augmentation helps without architectural changes")
