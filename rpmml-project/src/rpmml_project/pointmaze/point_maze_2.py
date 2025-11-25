import minari
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ============================================================================
# DATA
# ============================================================================


class MinariTrajectoryDataset(Dataset):
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
# MODEL
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
        h = h * (scale[:, :, None] + 1) + shift[:, :, None]
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
    def __init__(self, state_dim, hidden_dims=[128, 256, 512], time_dim=32):
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
        original_length = x.shape[2]

        # Pad to make divisible by 2^num_downsamples (8 in this case: 2^3)
        num_downsamples = len(self.downsamples)
        pad_factor = 2**num_downsamples
        padded_length = ((original_length + pad_factor - 1) // pad_factor) * pad_factor

        if padded_length != original_length:
            pad_amount = padded_length - original_length
            x = F.pad(x, (0, pad_amount), mode="replicate")

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
            skip = skips.pop()
            # Match dimensions if there's a mismatch due to upsampling
            if x.shape[2] != skip.shape[2]:
                x = x[:, :, : skip.shape[2]]
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)
            x = attn(x)

        x = self.final_conv(x)

        # Remove padding if we added any
        if padded_length != original_length:
            x = x[:, :, :original_length]

        return rearrange(x, "b c t -> b t c")


# ============================================================================
# DIFFUSION - FIXED VERSION
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
        """
        FIXED VERSION: Uses soft conditioning that respects model physics

        Key changes:
        1. Apply soft conditioning as an additive correction rather than hard replacement
        2. Condition only AFTER the model prediction (not before)
        3. Use annealing schedule for conditioning strength
        """

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
        conditioning_schedule="cosine",  # "linear", "cosine", or "constant"
        conditioning_strength=1.0,  # Max strength at t=T
    ):
        """
        FIXED VERSION: Soft conditioning with annealing schedule

        Args:
            conditioning_schedule: How to anneal conditioning over time
                - "linear": Linear decay from conditioning_strength to 0
                - "cosine": Cosine decay (gentler at beginning and end)
                - "constant": Keep conditioning_strength constant
            conditioning_strength: Maximum conditioning weight (at t=T)
        """
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
# TRAINING
# ============================================================================


class DiffuserTrainer:
    def __init__(self, model, diffusion, dataset, lr=1e-4, device="cuda"):
        self.model = model.to(device)
        self.diffusion = diffusion.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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

        return loss.item()

    def train(self, epochs=100):
        self.model.train()
        for epoch in range(epochs):
            losses = []
            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}"):
                losses.append(self.train_step(batch))
            print(f"Epoch {epoch+1}: Loss = {np.mean(losses):.4f}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        # Handle both formats: direct state_dict or full checkpoint
        if "model" in checkpoint:
            # Full checkpoint with model, ema_shadow, optimizer, scheduler
            self.model.load_state_dict(checkpoint["model"])
        else:
            # Direct state_dict
            self.model.load_state_dict(checkpoint)


# ============================================================================
# PLANNING - FIXED VERSION
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
        conditioning_schedule="cosine",  # NEW: annealing schedule
        conditioning_strength=0.5,  # NEW: reduced from 1.0 for softer conditioning
    ):
        """
        FIXED VERSION: Uses soft conditioning to respect learned physics

        Key improvements:
        1. Soft conditioning instead of hard replacement
        2. Annealing schedule that reduces conditioning over time
        3. Lower default conditioning strength

        Args:
            conditioning_schedule: "linear", "cosine", or "constant"
            conditioning_strength: Max weight for conditioning (0.0-1.0)
                - 0.5-0.7: Good balance between constraint and physics
                - 1.0: Stronger constraint (may violate physics)
                - 0.0: No conditioning (pure generation)
        """
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

        # Optionally condition on goal (NOT RECOMMENDED with guidance!)
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


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    dataset = MinariTrajectoryDataset("D4RL/pointmaze/umaze-v2", horizon=32)

    # Create model
    model = TemporalUNet(dataset.state_dim, hidden_dims=[128, 256, 512])
    diffusion = GaussianDiffusion(timesteps=200)

    # Train (or load)
    trainer = DiffuserTrainer(model, diffusion, dataset, device=device)
    # trainer.train(epochs=100)
    # trainer.save("diffuser_model.pt")
    trainer.load("checkpoints/diffuser_improved_epoch_100.pt")

    # Plan
    planner = DiffuserPlanner(model, diffusion, dataset, device=device)

    current = np.array([1.06591915, 0.39449871, 0.88507522, 4.78210334])
    goal = np.array([0.55692841, 1.02245092])

    # Create reward function with BOTH start and goal rewards
    reward_fn = CompositeReward(
        [
            StartReachingReward(current[:2], reward_scale=5.0),  # Match start position
            GoalReachingReward(goal, reward_scale=5.0),  # Reach goal position
            PathLengthPenalty(reward_scale=0.1),  # Prefer shorter paths
        ]
    )

    print("\n" + "=" * 80)
    print("BEST APPROACH: Start + Goal rewards with soft conditioning")
    print("=" * 80)

    # Try different conditioning strengths
    for strength in [0.0, 0.3, 0.5]:
        print(f"\n--- Conditioning strength: {strength} ---")
        traj = planner.plan(
            current,
            goal,
            reward_fn=reward_fn,
            guidance_scale=2.0,
            condition_on_start=True,
            condition_on_goal=False,
            conditioning_schedule="cosine",
            conditioning_strength=strength,
        )
        print(f"Start (actual): {traj[0, :2]}")
        print(f"Start (target): {current[:2]}")
        print(f"Start error: {np.linalg.norm(traj[0, :2] - current[:2]):.4f}")
        print(f"End: {traj[-1, :2]}")
        print(f"Goal: {goal}")
        print(f"Goal error: {np.linalg.norm(traj[-1, :2] - goal):.4f}")

        # Check for discontinuities (physics violations)
        diffs = np.linalg.norm(traj[1:, :2] - traj[:-1, :2], axis=1)
        max_jump = diffs.max()
        print(f"Max step size: {max_jump:.4f}")
        if max_jump > 0.3:  # Adjust threshold based on your maze
            print("  ⚠️  WARNING: Large discontinuity detected!")
        else:
            print("  ✓ Trajectory appears continuous")
