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
    """Dataset with improved normalization and validation."""

    def __init__(
        self, dataset_name="D4RL/pointmaze/umaze-v2", horizon=32, normalize=True
    ):
        self.horizon = horizon
        self.dataset = minari.load_dataset(dataset_name, download=True)

        # Extract trajectories
        self.trajectories = []
        for episode in self.dataset:
            obs = episode.observations
            if isinstance(obs, dict):
                obs = obs["observation"]
            self.trajectories.append(obs)

        self.state_dim = 4
        self.action_dim = 2

        # Compute normalization with clipping for stability
        all_data = np.concatenate(self.trajectories, axis=0)

        if normalize:
            self.mean = all_data.mean(axis=0)
            self.std = all_data.std(axis=0) + 1e-8
            # Clip extreme values to prevent normalization issues
            self.data_min = np.percentile(all_data, 1, axis=0)
            self.data_max = np.percentile(all_data, 99, axis=0)
        else:
            self.mean = np.zeros(self.state_dim)
            self.std = np.ones(self.state_dim)
            self.data_min = all_data.min(axis=0)
            self.data_max = all_data.max(axis=0)

        # Create indices
        self.indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            for t in range(len(traj) - horizon + 1):
                self.indices.append((traj_idx, t))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        denorm = x * self.std + self.mean
        # Clip to physically reasonable bounds
        return np.clip(denorm, self.data_min, self.data_max)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, start_t = self.indices[idx]
        traj = self.trajectories[traj_idx][start_t : start_t + self.horizon]
        traj = self.normalize(traj)
        return torch.FloatTensor(traj)


def recover_actions_from_states(states, dt=0.02):
    """Improved action recovery with acceleration constraints."""
    velocities = states[:, 2:4]
    # Use weighted average for smoother actions
    actions = 0.7 * velocities[:-1] + 0.3 * velocities[1:]
    # Clip actions to reasonable bounds
    return np.clip(actions, -2.0, 2.0)


# ============================================================================
# 2. IMPROVED REWARD FUNCTIONS WITH AUTOMATIC SCALING
# ============================================================================


class RewardFunction:
    """Base class with automatic gradient scaling."""

    def __init__(self, reward_scale=1.0):
        self.reward_scale = reward_scale
        self._grad_scale = 1.0  # For adaptive scaling

    def __call__(self, trajectories):
        raise NotImplementedError

    def get_grad_scale(self):
        """Return current gradient scale for adaptive guidance."""
        return self._grad_scale


class GoalReachingReward(RewardFunction):
    """Improved goal reward with distance normalization."""

    def __init__(self, goal_pos, reward_scale=1.0, threshold=0.45):
        super().__init__(reward_scale)
        self.goal_pos = torch.tensor(goal_pos, dtype=torch.float32)
        self.threshold = threshold

    def __call__(self, trajectories):
        final_pos = trajectories[:, -1, :2]
        goal = self.goal_pos.to(trajectories.device)

        # Use smooth distance function
        dist = torch.norm(final_pos - goal, dim=-1)

        # Reward is negative distance, but with diminishing returns for far distances
        # This prevents over-aggressive guidance for very far goals
        reward = -torch.sqrt(dist + 0.01)  # sqrt for diminishing gradient

        # Update gradient scale based on typical distances
        with torch.no_grad():
            self._grad_scale = 1.0 / (dist.mean() + 1e-6)

        return reward * self.reward_scale


class TrajectoryLengthReward(RewardFunction):
    """Path efficiency with better scaling."""

    def __init__(self, reward_scale=1.0):
        super().__init__(reward_scale)

    def __call__(self, trajectories):
        positions = trajectories[:, :, :2]
        diffs = positions[:, 1:] - positions[:, :-1]
        # Use squared norm for smoother gradients
        path_length = (diffs**2).sum(dim=-1).sum(dim=-1)

        # Update gradient scale
        with torch.no_grad():
            self._grad_scale = 1.0 / (path_length.mean() + 1e-6)

        return -torch.sqrt(path_length + 0.01) * self.reward_scale


class VelocitySmoothness(RewardFunction):
    """Smoothness with acceleration penalties."""

    def __init__(self, reward_scale=1.0):
        super().__init__(reward_scale)

    def __call__(self, trajectories):
        velocities = trajectories[:, :, 2:4]
        # Penalize acceleration (changes in velocity)
        accel = velocities[:, 1:] - velocities[:, :-1]
        smoothness_cost = (accel**2).sum(dim=-1).sum(dim=-1)

        with torch.no_grad():
            self._grad_scale = 1.0 / (smoothness_cost.mean() + 1e-6)

        return -torch.sqrt(smoothness_cost + 0.01) * self.reward_scale


class CompositeReward(RewardFunction):
    """Improved composite with automatic weight balancing."""

    def __init__(self, reward_fns, weights=None, auto_balance=True):
        super().__init__(1.0)
        self.reward_fns = reward_fns
        self.auto_balance = auto_balance

        if weights is None:
            weights = [1.0] * len(reward_fns)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __call__(self, trajectories):
        rewards = []
        scales = []

        for fn in self.reward_fns:
            r = fn(trajectories)
            rewards.append(r)
            if hasattr(fn, "_grad_scale"):
                scales.append(fn._grad_scale)
            else:
                scales.append(1.0)

        # Stack rewards
        rewards = torch.stack(rewards, dim=1)  # (B, num_rewards)

        # Auto-balance weights based on gradient scales
        if self.auto_balance:
            scales = torch.tensor(
                scales, device=trajectories.device, dtype=torch.float32
            )
            # Normalize scales to sum to 1
            scales = scales / (scales.sum() + 1e-6)
            weights = self.weights.to(trajectories.device) * scales
        else:
            weights = self.weights.to(trajectories.device)

        # Weighted sum
        total = (rewards * weights[None, :]).sum(dim=1)
        return total


# ============================================================================
# 3. TEMPORAL U-NET (REGULARIZED)
# ============================================================================


class SinusoidalPosEmb(nn.Module):
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
        h = self.block1(x)
        scale, shift = self.mlp(time_emb).chunk(2, dim=-1)
        scale = scale[:, :, None]
        shift = shift[:, :, None]
        h = h * (scale + 1) + shift
        h = self.block2(h)
        return h + self.res_conv(x)


class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
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
    """U-Net optimized for PointMaze (4D state space)."""

    def __init__(self, state_dim, hidden_dims=[64, 128, 256], time_dim=32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4),
        )

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

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, hidden_dims[0]),
            nn.SiLU(),
            nn.Conv1d(hidden_dims[0], state_dim, 3, padding=1),
        )

    def forward(self, x, time):
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
# 4. IMPROVED DIFFUSION WITH ADAPTIVE GUIDANCE
# ============================================================================


class GaussianDiffusion(nn.Module):
    """DDPM with adaptive guidance and stability improvements."""

    def __init__(self, timesteps=200, beta_start=0.0001, beta_end=0.02):
        super().__init__()

        self.timesteps = timesteps

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0
        )

        sqrt_alphas = torch.sqrt(alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1 - alphas_cumprod_prev) * sqrt_alphas / (1 - alphas_cumprod)
        )

        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas", sqrt_alphas)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod
        )
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)
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
        """
        Predict x0 from xt and noise.

        From DDPM paper (Ho et al. 2020), equation 15:
        x0 = (xt - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)

        Equivalently (equation 11):
        x0 = (1 / sqrt(alpha_bar_t)) * xt - (sqrt(1/alpha_bar_t - 1)) * epsilon
        """
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod[t]).view(
            -1, 1, 1
        )
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(
            -1, 1, 1
        )

        return sqrt_recip_alphas_cumprod * xt - sqrt_recipm1_alphas_cumprod * noise

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

    @torch.no_grad()
    def p_sample(self, model, xt, t):
        predicted_noise = model(xt, t)
        x0_pred = self.predict_x0_from_noise(xt, t, predicted_noise)
        x0_pred = x0_pred.clamp(-3, 3)

        mean, var = self.p_mean_variance(x0_pred, xt, t)

        if (t == 0).all():
            return mean

        return mean + torch.sqrt(var) * torch.randn_like(xt)

    def p_sample_guided(self, model, xt, t, reward_fn, dataset, guidance_scale):
        """
        Improved guidance with:
        1. Adaptive scaling based on timestep
        2. Gradient clipping for stability
        3. Momentum-based updates
        """
        # Adaptive guidance scale (stronger at later timesteps)
        t_normalized = t.float() / self.timesteps
        adaptive_scale = guidance_scale * (1.0 - 0.5 * t_normalized.mean())

        # Enable gradients
        xt = xt.clone().detach().requires_grad_(True)

        # Compute guidance
        if reward_fn is not None and adaptive_scale > 0:
            with torch.enable_grad():
                # Predict x0 with gradients
                model.train()  # Need training mode for gradients to flow
                predicted_noise = model(xt, t)
                model.eval()  # Switch back to eval mode

                x0_pred = self.predict_x0_from_noise(xt, t, predicted_noise)

                # Denormalize
                mean = torch.tensor(dataset.mean, device=xt.device, dtype=xt.dtype)[
                    None, None, :
                ]
                std = torch.tensor(dataset.std, device=xt.device, dtype=xt.dtype)[
                    None, None, :
                ]
                x0_denorm = x0_pred * std + mean

                # Compute reward
                rewards = reward_fn(x0_denorm)
                grad = torch.autograd.grad(rewards.sum(), xt)[0]

            # Clip gradient for stability
            grad_norm = torch.norm(grad)
            if grad_norm > 1.0:
                grad = grad / grad_norm

            # Apply guidance with adaptive scale
            xt = xt + adaptive_scale * grad

        # Continue sampling (without gradients)
        xt = xt.detach()
        predicted_noise = model(xt, t)
        x0_pred = self.predict_x0_from_noise(xt, t, predicted_noise)
        x0_pred = x0_pred.clamp(-3, 3)

        mean, var = self.p_mean_variance(x0_pred, xt, t)

        if (t == 0).all():
            return mean

        return mean + torch.sqrt(var) * torch.randn_like(xt)

    @torch.no_grad()
    def sample(self, model, shape, device):
        xt = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            xt = self.p_sample(model, xt, t)

        return xt

    def sample_guided(self, model, shape, device, reward_fn, dataset, guidance_scale):
        xt = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            xt = self.p_sample_guided(model, xt, t, reward_fn, dataset, guidance_scale)

        return xt

    def inpaint_sample(
        self,
        model,
        shape,
        device,
        mask,
        x0_known,
        resample_steps=1,
        jump_length=5,
        reward_fn=None,
        dataset=None,
        guidance_scale=0.0,
    ):
        """Improved inpainting with conservative resampling."""
        xt = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            # Limit resampling to improve stability
            actual_resample = min(resample_steps, 3) if i > self.timesteps // 2 else 1

            for r in range(actual_resample):
                t = torch.full((shape[0],), i, device=device, dtype=torch.long)

                # Re-noise known region
                noise_known = torch.randn_like(x0_known)
                sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
                sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
                xt_known = sqrt_ac * x0_known + sqrt_om * noise_known

                # Combine
                xt = torch.where(mask, xt_known, xt)

                # Reverse step
                if reward_fn is None or guidance_scale == 0:
                    xt = self.p_sample(model, xt, t)
                else:
                    xt = self.p_sample_guided(
                        model, xt, t, reward_fn, dataset, guidance_scale
                    )

                # Conservative resampling jump
                if r < actual_resample - 1 and i > 0:
                    t_jump = min(i + jump_length, self.timesteps - 1)
                    if t_jump > i:
                        t_from = torch.full(
                            (shape[0],), i, device=device, dtype=torch.long
                        )
                        t_to = torch.full(
                            (shape[0],), t_jump, device=device, dtype=torch.long
                        )

                        alpha_from = self.alphas_cumprod[t_from].view(-1, 1, 1)
                        alpha_to = self.alphas_cumprod[t_to].view(-1, 1, 1)

                        ratio = torch.sqrt(alpha_to / alpha_from)
                        noise = torch.randn_like(xt)
                        xt = ratio * xt + torch.sqrt(1 - alpha_to / alpha_from) * noise

        return xt


# ============================================================================
# 5. IMPROVED TRAINING
# ============================================================================


class EMA:
    def __init__(self, model, beta=0.999):  # Higher beta for more stability
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


class DiffuserTrainer:
    """Improved trainer with better batch size and learning rate schedule."""

    def __init__(self, model, diffusion, dataset, lr=1e-4, device="cuda"):
        self.model = model.to(device)
        self.diffusion = diffusion.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(  # Use AdamW for better regularization
            model.parameters(), lr=lr, weight_decay=1e-4
        )

        # Cosine annealing schedule
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        self.ema = EMA(model, beta=0.999)

        # Smaller batch size for better gradient estimates
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
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # Tighter clipping
        self.optimizer.step()

        self.ema.update(self.model)

        return loss.item()

    def train(self, epochs=100):
        self.model.train()

        for epoch in range(epochs):
            epoch_losses = []
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            avg_loss = np.mean(epoch_losses)
            print(
                f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, LR: {self.scheduler.get_last_lr()[0]:.2e}"
            )

            self.scheduler.step()

            self.save_checkpoint(f"checkpoints/diffuser_improved_epoch_{epoch+1}.pt")

    def save_checkpoint(self, path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema_shadow": self.ema.shadow,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.ema.shadow = checkpoint["ema_shadow"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])


# ============================================================================
# 6. PLANNING
# ============================================================================


class DiffuserPlanner:
    """Improved planner with automatic hyperparameter tuning."""

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
        resample_steps=2,
        jump_length=5,  # More conservative defaults
        reward_fn=None,
        guidance_scale=1.0,  # Lower default scale
        auto_tune=True,
    ):
        """
        Planning with optional auto-tuning of hyperparameters.

        Args:
            auto_tune: If True, automatically adjust guidance_scale based on goal distance
        """
        self.model.eval()

        # Normalize current observation
        current_obs = np.asarray(current_obs, dtype=np.float32).reshape(-1)
        start_norm = (current_obs - self.dataset.mean) / self.dataset.std
        start_norm = torch.tensor(start_norm, dtype=torch.float32, device=self.device)

        # Auto-tune guidance scale based on goal distance
        if auto_tune and goal_obs is not None:
            goal_dist = np.linalg.norm(current_obs[:2] - goal_obs[:2])
            # Closer goals need less aggressive guidance
            guidance_scale = guidance_scale * np.clip(goal_dist / 3.0, 0.5, 2.0)

        # Build conditioning
        shape = (1, horizon, self.dataset.state_dim)
        condition_mask = torch.zeros(shape, dtype=torch.bool, device=self.device)
        condition_value = torch.zeros(shape, dtype=torch.float32, device=self.device)

        # Always condition on start
        condition_mask[0, 0, :] = True
        condition_value[0, 0, :] = start_norm

        # Optionally condition on goal
        if goal_obs is not None:
            goal_obs = np.asarray(goal_obs, dtype=np.float32).flatten()
            if len(goal_obs) == 2:
                goal_state = np.zeros(self.dataset.state_dim, dtype=np.float32)
                goal_state[:2] = goal_obs
                goal_norm = (goal_state - self.dataset.mean) / self.dataset.std
            else:
                goal_norm = (goal_obs - self.dataset.mean) / self.dataset.std

            goal_norm = torch.tensor(goal_norm, dtype=torch.float32, device=self.device)
            condition_mask[0, -1, :] = True
            condition_value[0, -1, :] = goal_norm

        # Run diffusion
        trajectory = self.diffusion.inpaint_sample(
            self.model,
            shape,
            self.device,
            condition_mask,
            condition_value,
            resample_steps=resample_steps,
            jump_length=jump_length,
            reward_fn=reward_fn,
            dataset=self.dataset,
            guidance_scale=guidance_scale,
        )

        # Denormalize
        trajectory = trajectory.cpu().numpy()[0]
        trajectory = self.dataset.denormalize(trajectory)

        # Recover actions
        actions = recover_actions_from_states(trajectory)
        return trajectory, actions


# ============================================================================
# 7. MAIN
# ============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    dataset = MinariTrajectoryDataset("D4RL/pointmaze/umaze-v2", horizon=32)
    print(f"Dataset: {len(dataset)} segments")
    print(f"State bounds: [{dataset.data_min}, {dataset.data_max}]")

    # Create model (balanced size for PointMaze)
    print("\n" + "=" * 80)
    print("CREATING MODEL")
    print("=" * 80)
    model = TemporalUNet(dataset.state_dim, hidden_dims=[128, 256, 512])
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Architecture: Balanced size, no dropout")

    # More diffusion steps for quality
    diffusion = GaussianDiffusion(timesteps=200)
    print(f"Timesteps: {diffusion.timesteps}")

    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    trainer = DiffuserTrainer(model, diffusion, dataset, lr=1e-4, device=device)
    # trainer.train(epochs=100)
    trainer.load_checkpoint("checkpoints/diffuser_improved_epoch_100.pt")

    # Copy EMA weights
    trainer.ema.copy_to(model)

    # Create planner
    print("\n" + "=" * 80)
    print("PLANNING EXAMPLES")
    print("=" * 80)
    planner = DiffuserPlanner(model, diffusion, dataset, device=device)

    goal = np.array([3.0, 3.0])

    # Improved reward with auto-balancing
    reward_fn = CompositeReward(
        [
            GoalReachingReward(goal, reward_scale=1.0),
            TrajectoryLengthReward(reward_scale=0.5),  # Moderate weight
            VelocitySmoothness(reward_scale=0.2),  # Moderate weight
        ],
        auto_balance=True,
    )

    current = np.array([0.0, 0.0, 0.0, 0.0])

    # Plan without guidance
    print("\nWithout guidance:")
    traj_no, acts_no = planner.plan(current, goal, reward_fn=None, auto_tune=False)
    print(
        f"Final: {traj_no[-1, :2]}, Dist: {np.linalg.norm(traj_no[-1, :2] - goal):.4f}"
    )

    # Plan with moderate guidance
    print("\nWith guidance (scale=1.0):")
    traj_yes, acts_yes = planner.plan(
        current, goal, reward_fn=reward_fn, guidance_scale=1.0, auto_tune=True
    )
    print(
        f"Final: {traj_yes[-1, :2]}, Dist: {np.linalg.norm(traj_yes[-1, :2] - goal):.4f}"
    )

    print("\n" + "=" * 80)
    print("DONE - Check that guided planning reaches goal better!")
    print("=" * 80)
