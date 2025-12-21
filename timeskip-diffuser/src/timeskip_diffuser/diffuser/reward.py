"""
Differentiable reward functions for diffusion-based trajectory optimization.
"""

from abc import ABC, abstractmethod

import torch


class DiffReward(ABC):
    """Abstract base class for differentiable reward functions.

    All reward functions should inherit from this class and implement
    the __call__ method which takes trajectories and returns scalar rewards.
    """

    def __init__(self, reward_scale=1.0):
        self.reward_scale = reward_scale

    @abstractmethod
    def __call__(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Compute reward for a batch of trajectories.

        Args:
            trajectories: Tensor of shape (B, T, D) where
                B = batch size
                T = trajectory length
                D = state dimension (typically 3: x, y, skip)

        Returns:
            Tensor of shape (B,) containing scalar rewards for each trajectory.
        """


class StartReachingReward(DiffReward):
    """Reward based on proximity of initial state to a target start position."""

    def __init__(self, start_pos, reward_scale=1.0):
        super().__init__(reward_scale)
        self.start_pos = torch.tensor(start_pos, dtype=torch.float32)

    def __call__(self, trajectories):
        pos = trajectories[..., :2]  # (B, T, 2) - extract x, y coordinates
        initial_pos = pos[:, 0]  # (B, 2)
        start_pos = self.start_pos.to(trajectories.device)
        dist_squared = ((initial_pos - start_pos) ** 2).sum(dim=-1)
        return -dist_squared * self.reward_scale


class GoalReachingReward(DiffReward):
    """Reward based on proximity of final state to a target goal position."""

    def __init__(self, goal_pos, reward_scale=1.0):
        super().__init__(reward_scale)
        self.goal_pos = torch.tensor(goal_pos, dtype=torch.float32)

    def __call__(self, trajectories):
        pos = trajectories[..., :2]  # (B, T, 2)
        final_pos = pos[:, -1]  # (B, 2)
        goal_pos = self.goal_pos.to(trajectories.device)
        dist_squared = ((final_pos - goal_pos) ** 2).sum(dim=-1)
        return -dist_squared * self.reward_scale


class PathLengthPenalty(DiffReward):
    """Penalty based on total path length (sum of step L2 norms)."""

    def __call__(self, trajectories):
        pos = trajectories[..., :2]  # (B, T, 2)
        position_diffs = pos[:, 1:] - pos[:, :-1]  # (B, T-1, 2)
        step_lengths = position_diffs.norm(dim=-1)  # (B, T-1)
        path_length = step_lengths.sum(dim=-1)  # (B,)
        return -path_length * self.reward_scale


class CurvaturePenalty(DiffReward):
    """Penalty based on trajectory curvature (encourages smooth paths)."""

    def __call__(self, trajectories):
        pos = trajectories[..., :2]  # (B, T, 2)
        # Second-order finite difference approximation of curvature
        curvature = (pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]).norm(dim=-1)
        return -curvature.sum(dim=-1) * self.reward_scale


class LogSkipReward(DiffReward):
    """Reward based on the logarithm of timeskips (encourages larger skips)."""

    def __init__(self, reward_scale=1.0, eps=1e-4):
        super().__init__(reward_scale)
        self.eps = eps

    def __call__(self, trajectories):
        skips = trajectories[..., 2]  # (B, T)
        log_skip = torch.log(torch.clamp(skips, min=self.eps))
        return log_skip.sum(dim=-1) * self.reward_scale


class SkipTotalTimeSkipPenalty(DiffReward):
    """Penalty based on total physical time elapsed (sum of all timeskips)."""

    def __call__(self, trajectories):
        skips = trajectories[..., 2]  # (B, T)
        total_time = skips.sum(dim=-1)  # (B,)
        return -total_time * self.reward_scale


class CompositeReward(DiffReward):
    """Composite reward combining multiple reward functions with optional weights."""

    def __init__(self, reward_fns, weights=None):
        super().__init__(reward_scale=1.0)
        self.reward_fns = reward_fns
        if weights is None:
            weights = [1.0] * len(reward_fns)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __call__(self, trajectories):
        rewards = torch.stack(
            [fn(trajectories) for fn in self.reward_fns], dim=1
        )  # (B, N)
        weights = self.weights.to(trajectories.device)
        return (rewards * weights[None, :]).sum(dim=1)  # (B,)

    def print_components(self, trajectories):
        """Print individual reward components for debugging."""
        rewards = torch.stack([fn(trajectories) for fn in self.reward_fns], dim=1)
        for fn, reward_vals in zip(self.reward_fns, rewards.T):
            print(f"{fn.__class__.__name__}: {reward_vals}")
