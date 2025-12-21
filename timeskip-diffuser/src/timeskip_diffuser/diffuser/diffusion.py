"""
Gaussian Diffusion class for diffusion planning.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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
