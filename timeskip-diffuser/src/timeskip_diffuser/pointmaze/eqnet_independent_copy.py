"""
Improved Diffuser Implementation with Eq-Net Architecture
"""

import os

import minari
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
import minari
from matplotlib.patches import Rectangle
import mujoco
import random
from matplotlib import pyplot as plt
import wandb

class LogSkipReward:
    def __init__(self, reward_scale=1.0, eps=1e-4):
        self.weight = reward_scale
        self.eps = eps

    def __call__(self, traj):
        skips = traj[..., 2]
        log_skip = torch.log(torch.clamp(skips, min=self.eps))
        return log_skip.sum(dim=-1) * self.weight


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

    
    
class StartReachingReward:
    """A reward proportional to the squared error of the sampled start state."""

    def __init__(self, start_pos, reward_scale=1.0):
        self.start_pos = torch.tensor(start_pos, dtype=torch.float32)
        self.reward_scale = reward_scale

    def __call__(self, trajectories):
        # trajectories: (B, T, D) where D = 3 (x, y, skip)
        pos = trajectories[..., :2]      # use only x,y
        initial_pos = pos[:, 0]          # (B,2)
        start_pos = self.start_pos.to(trajectories.device)
        dist_squared = ((initial_pos - start_pos) ** 2).sum(dim=-1)
        return -dist_squared * self.reward_scale

class GoalReachingReward:
    def __init__(self, goal_pos, reward_scale=1.0):
        self.goal_pos = torch.tensor(goal_pos, dtype=torch.float32)
        self.reward_scale = reward_scale

    def __call__(self, trajectories):
        pos = trajectories[..., :2]      # (B,T,2)
        final_pos = pos[:, -1]
        goal_pos = self.goal_pos.to(trajectories.device)
        dist_squared = ((final_pos - goal_pos)**2).sum(dim=-1)
        return -dist_squared * self.reward_scale




class CurvaturePenalty:
    """A reward proportional to the sum of trajectory curvatures."""
    def __init__(self, reward_scale=1.0):
        self.reward_scale = reward_scale

    def __call__(self, traj):
        # traj: (B, T, D) where D >= 2
        pos = traj[..., :2]  # (B, T, 2)
        curvature = (pos[:, 2:] - 2*pos[:, 1:-1] + pos[:, :-2]).norm(dim=-1)
        return -curvature.sum(dim=-1) * self.reward_scale

class SkipTotalTimeSkipPenalty:
    """A reward proportional to the sum of step lengths (L2 norms)."""

    def __init__(self, reward_scale=1.0):
        self.reward_scale = reward_scale

    def __call__(self, trajectories):
        position_diffs = trajectories[:, 1:] - trajectories[:, :-1]
        step_lengths = position_diffs.norm(dim=-1)  # L2 length of each step
        path_length = step_lengths.sum(dim=-1)
        return -path_length * self.reward_scale


class CompositeReward:
    """A composite reward, which combines multiple reward functions with weights."""

    def __init__(self, reward_fns, weights=None):
        self.reward_fns = reward_fns
        if weights is None:
            weights = [1.0] * len(reward_fns)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __call__(self, trajectories):
        rewards = torch.stack([fn(trajectories) for fn in self.reward_fns], dim=1)
        weights = self.weights.to(trajectories.device)
        return (rewards * weights[None, :]).sum(dim=1)

    def print(self, trajectories):
        """Debug function for printing rewards."""
        rewards = torch.stack([fn(trajectories) for fn in self.reward_fns], dim=1)
        print(rewards)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Forward pass for positional embedding."""

        half_dim = self.dim // 2
        emb = np.log(1000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LayerNorm1d(nn.Module):
    """Layer normalization for 1D convolutions."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        """Forward pass for 1D layer norm."""
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
        """Forward pass for local residual block."""

        h = self.block1(x)

        # Add time conditioning
        time_cond = self.mlp(time_emb)
        h = h + time_cond[:, :, None]

        h = self.block2(h)
        return h + self.res_conv(x)


class EqNet(nn.Module):
    """
    Shift-Equivariant Network with Local Receptiveness for trajectory composition.
    """

    def __init__(self, state_dim, hidden_dim=128, time_dim=32, n_layers=10):
        super().__init__()

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
            3,
            padding=1,
            padding_mode="replicate",
        )

        # Stack of local residual blocks (NO downsampling!)
        self.blocks = nn.ModuleList(
            [
                LocalResidualBlock(hidden_dim, hidden_dim, time_dim * 4, kernel_size=3)
                for _ in range(n_layers // 2)
            ]
            + [
                LocalResidualBlock(hidden_dim, hidden_dim, time_dim * 4, kernel_size=5)
                for _ in range(n_layers // 2)
            ]
        )

        # Final projection
        self.final_conv = nn.Sequential(
            LayerNorm1d(hidden_dim), nn.Mish(), nn.Conv1d(hidden_dim, state_dim, 1)
        )

    def forward(self, x, time):
        """
        Eq-Net forward pass.

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

        # Apply residual blocks
        for block in self.blocks:
            h = block(h, t_emb)

        # Final projection
        out = self.final_conv(h)

        # Transpose back to (batch, horizon, state_dim)
        return out.transpose(1, 2)


class GaussianDiffusion(nn.Module):  # pylint: disable=abstract-method
    """Gaussian diffusion class."""

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
        """Sample from q."""

        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_ac * x0 + sqrt_om * noise

    def predict_x0_from_noise(self, xt, t, noise):
        """Predict the original x0 from noise."""

        sqrt_recip = torch.sqrt(1.0 / self.alphas_cumprod[t]).view(-1, 1, 1)
        sqrt_recipm1 = torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(-1, 1, 1)
        return sqrt_recip * xt - sqrt_recipm1 * noise

    def p_mean_variance(self, x0, xt, t):
        """Return the mean and variance of the new distribution."""

        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1)
        mean = coef1 * x0 + coef2 * xt
        var = self.posterior_variance[t].view(-1, 1, 1)
        return mean, var

    def p_losses(self, model, x0, t):
        """MSE Loss for diffusion."""

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
        """Guided sampling with soft conditioning."""

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
        model: EqNet,
        diffusion,
        dataset,
        lr=1e-4,
        device="cuda",
        ema_beta=0.999,
    ):
        self.model = model.to(device)
        self.diffusion = diffusion.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # Cosine annealing schedule
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        # Exponential moving averages
        self.ema = EMA(model, beta=ema_beta)

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

        # EMA update
        if self.ema is not None:
            self.ema.update(self.model)

        # --- W&B LOGGING ---
        wandb.log(
            {"train/loss": loss.item(), "train/lr": self.scheduler.get_last_lr()[0]},
            commit=True
        )

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
                checkpoint_path = f"checkpoints/diffuser_flat_eqnet_independent_epoch_{epoch+1}_fixed_over.pt"
                self.save_checkpoint(checkpoint_path)
                print(f"  → Saved checkpoint to {checkpoint_path}")

    def save_checkpoint(self, path):
        """Save complete training state including EMA"""

        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow

        torch.save(checkpoint, path)

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
        self.ema.copy_to(self.model)
        print("Switched to EMA parameters for inference")


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
    velocities[0] = np.zeros(2)  # first velocity: zero
    for i in range(1, N - 1):
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

if __name__ == "__main__":
    
    
    
    #for online benchmarking
    wandb.init(
        project="eqnet-diffuser",
        name="independent-skips",
        config={
            "horizon": 32,
            "timesteps": 200,
            "lr": 1e-4,
            "dataset": "independent_skips_umaze",
            "model": "EqNet",
            "hidden_dim": 128,
            "time_dim": 32,
            "n_layers": 10,
            "guidance_scale": 2.0,
        }
    )
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {torch_device}")

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    print("\n" + "=" * 80)
    print("EQ-NET CONFIGURATION")
    print("=" * 80)
    print("Architecture: Local convolutions (no downsampling) for shift equivariance")

    # ========================================================================
    # DATA
    # ========================================================================

    OFFLINE_FILE = "/scratch/network/dd6849/rpmml-project/fixed_stats_offline_umaze_independent_skips_h32_mean1_sig1_oversampled.npz"

    minari_dataset = OfflineSkipDataset(
        OFFLINE_FILE,
        horizon=32
    )
    print(f"Loaded offline dataset: {len(minari_dataset)} samples.")

    # ========================================================================
    # MODEL
    # ========================================================================

    # Eq-Net: shift-equivariant architecture with local receptiveness
    eqnet = EqNet(
        state_dim=minari_dataset.traj_dim,
        hidden_dim=128,
        time_dim=32,
        n_layers=10,  # Deep but local
    )

    gaussian_diffusion = GaussianDiffusion(timesteps=200)

    print(f"\nModel parameters: {sum(p.numel() for p in eqnet.parameters()):,}")

    # ========================================================================
    # TRAINING
    # ========================================================================

    trainer = DiffuserTrainer(
        eqnet,
        gaussian_diffusion,
        minari_dataset,
        device=torch_device,
    )

    # Train or load
    TRAIN_NEW = True
    if TRAIN_NEW:
        print("\n" + "=" * 80)
        print("TRAINING")
        print("=" * 80)
        trainer.train(epochs=100, save_every=1)
    else:
        print("\nLoading pre-trained model...")
        trainer.load("checkpoints/diffuser_flat_eqnet_epoch_100_fixed.pt")
        print("Loaded checkpoint successfully")

    # IMPORTANT: Switch to EMA parameters for inference (better quality!)
    trainer.use_ema_for_inference()
    print("Using EMA parameters for planning")

    # ========================================================================
    # PLANNING
    # ========================================================================

    print("\n" + "=" * 80)
    print("PLANNING WITH EQ-NET")
    print("=" * 80)

    planner = DiffuserPlanner(eqnet, gaussian_diffusion, minari_dataset, device=torch_device)

    current = np.array([1.06591915, 0.39449871, 0.88507522, 4.78210334])
    goal = np.array([0.55692841, 1.02245092])

    reward_fn = CompositeReward(
        [
            StartReachingReward(current[:2], reward_scale=5.0),
            GoalReachingReward(goal, reward_scale=5.0),
            SkipTotalTimeSkipPenalty(reward_scale=0.1),
            CurvaturePenalty(reward_scale=0.1),
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

    # these are problems for later
    # # ========================================================================
    # # EXPERIMENT 2: Diverse sampling (paper shows this can beat replanning)
    # # ========================================================================

    # print("\n--- Experiment 2: Diverse Sampling (10 samples) ---")
    # best_traj, all_trajs, all_rewards = planner.plan_with_diversity(
    #     current,
    #     goal,
    #     num_samples=10,
    #     reward_fn=custom_reward_fn,
    #     guidance_scale=2.0,
    #     condition_on_start=True,
    #     conditioning_schedule="cosine",
    #     conditioning_strength=0.5,
    # )

    # print(f"Best trajectory reward: {max(all_rewards):.4f}")
    # print(f"Worst trajectory reward: {min(all_rewards):.4f}")
    # print(f"Mean reward: {np.mean(all_rewards):.4f}")
    # print("\nBest trajectory:")
    # print(f"  Start error: {np.linalg.norm(best_traj[0] - current):.4f}")
    # print(f"  Goal error: {np.linalg.norm(best_traj[-1] - goal):.4f}")

    # # Check continuity
    # diffs = np.linalg.norm(best_traj[1:] - best_traj[:-1], axis=1)
    # print(f"  Max step size: {diffs.max():.4f}")
    # print(f"  Mean step size: {diffs.mean():.4f}")

    # # ========================================================================
    # # EXPERIMENT 3: Effect of conditioning strength
    # # ========================================================================

    # print("\n--- Experiment 3: Conditioning Strength Comparison ---")
    # for strength in [0.0, 0.3, 0.5, 0.7, 1.0]:
    #     traj_plan = planner.plan(
    #         current,
    #         goal,
    #         reward_fn=custom_reward_fn,
    #         guidance_scale=2.0,
    #         condition_on_start=True,
    #         conditioning_schedule="cosine",
    #         conditioning_strength=strength,
    #     )

    #     start_err = np.linalg.norm(traj_plan[0] - current)
    #     goal_err = np.linalg.norm(traj_plan[-1] - goal)
    #     diffs = np.linalg.norm(traj_plan[1:] - traj_plan[:-1], axis=1)
    #     max_jump = diffs.max()

    #     print(
    #         f"\nStrength {strength:.1f}: start_err={start_err:.4f}, "
    #         f"goal_err={goal_err:.4f}, max_jump={max_jump:.4f}"
    #     )
    #     if max_jump > 0.3:
    #         print("  ⚠️  Physics violation")

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80)
    print("\nKey takeaways from the paper:")
    print("1. Local receptiveness + shift equivariance enable trajectory stitching")
    print("2. Diverse sampling can be as effective as replanning (but faster)")
    print("3. Soft conditioning respects learned physics better than hard constraints")
    print("4. Positional augmentation helps without architectural changes")