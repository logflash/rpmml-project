"""
Diffuser Trainer class.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from timeskip_diffuser.diffuser.ema import EMA
from timeskip_diffuser.diffuser.nets import TrajNet


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
        model: TrajNet,
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
        """A single diffusion training step."""

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
                checkpoint_path = f"checkpoints/diffuser_flat_eqnet_epoch_{epoch+1}.pt"
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

    def load_checkpoint(self, path):
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
