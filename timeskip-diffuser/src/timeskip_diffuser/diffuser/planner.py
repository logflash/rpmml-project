"""
Diffuser Planner class.
"""

import numpy as np
import torch


class DiffuserPlanner:
    """Diffuser Planner, with the ability to plan and reconstruct trajectories"""

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
            condition_value[0, 0, : self.dataset.state_dim] = start_norm_full[
                : self.dataset.state_dim
            ]

        # condition on goal
        if goal_obs is not None and condition_on_goal:
            goal_obs = np.asarray(goal_obs, dtype=np.float32).flatten()
            goal_pos = goal_obs[: self.dataset.state_dim]

            goal_norm_pos = (goal_pos - self.dataset.flat_mean) / self.dataset.flat_std
            goal_norm_full = np.zeros(self.dataset.traj_dim, dtype=np.float32)
            goal_norm_full[: self.dataset.state_dim] = goal_norm_pos
            goal_norm_full = torch.tensor(goal_norm_full, device=self.device)

            condition_mask[0, -1, : self.dataset.state_dim] = True
            condition_value[0, -1, : self.dataset.state_dim] = goal_norm_full[
                : self.dataset.state_dim
            ]

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
        try:
            assert spline_func is not None
            pos_dense, vel_dense, acc_dense = spline_func(skip_list)
        except TypeError:
            print("Wrong type for spline_func")
            return None

        # ------------------------------------------------------------------
        # 5. Return everything
        # ------------------------------------------------------------------
        return dict(
            coarse=coarse,  # (H,3)
            coarse_pos=pos,  # (H,2)
            coarse_skip=skip,  # (H,)
            skip_list=skip_list,  # [(pos_i, skip_i)]
            pos_dense=pos_dense,  # (N,2)
            vel_dense=vel_dense,  # (N,2)
            acc_dense=acc_dense,  # (N,2)
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
        p1 = positions[i + 1]

        k = skips[i]
        T = k * dt  # physical duration of this segment

        # SCALE velocities to spline coordinates
        v0_scaled = velocities[i] * T
        v1_scaled = velocities[i + 1] * T
        #Fixed negative predicted timeskips by clamping to positive - Tomasz
        if not np.isfinite(k) or k <= 0:
            print("k invalid")
            k = 0
            #Don't raise error anymore
            #raise ValueError(f"Invalid skip k={k} at segment {i}")
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
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2

    p = (
        h00[:, None] * p0
        + h10[:, None] * v0_scaled
        + h01[:, None] * p1
        + h11[:, None] * v1_scaled
    )

    # Velocity basis
    dh00 = 6 * t**2 - 6 * t
    dh10 = 3 * t**2 - 4 * t + 1
    dh01 = -6 * t**2 + 6 * t
    dh11 = 3 * t**2 - 2 * t

    v = (
        dh00[:, None] * p0
        + dh10[:, None] * v0_scaled
        + dh01[:, None] * p1
        + dh11[:, None] * v1_scaled
    )

    # Acceleration basis
    d2h00 = 12 * t - 6
    d2h10 = 6 * t - 4
    d2h01 = -12 * t + 6
    d2h11 = 6 * t - 2

    a = (
        d2h00[:, None] * p0
        + d2h10[:, None] * v0_scaled
        + d2h01[:, None] * p1
        + d2h11[:, None] * v1_scaled
    )

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
        velocities[i] = (positions[i + 1] - positions[i]) / T

    velocities[-1] = velocities[-2]  # last velocity: just copy previous

    return positions, velocities, skips


def run_sanity_check(dataset, diffusion, model, device="cpu", batch_idx=0):
    print("\n" + "=" * 80)
    print("SANITY CHECK: Dataset Normalization Consistency")
    print("=" * 80)

    # ----------------------------------------------------------------------------------
    # 1. Grab a raw window from dataset the exact way __getitem__ uses it
    # ----------------------------------------------------------------------------------
    sample_norm = dataset[batch_idx]  # normalized window: (H, 3)
    sample_norm_np = sample_norm.numpy()

    print("\nNormalized window [x_norm, y_norm, skip_norm]:")
    print(sample_norm_np)

    # Denormalize manually (matching DiffuserPlanner logic)
    pos_norm = sample_norm_np[:, : dataset.state_dim]
    skip_norm = sample_norm_np[:, dataset.state_dim]

    pos_denorm = pos_norm * dataset.flat_std + dataset.flat_mean
    skip_denorm = skip_norm * dataset.skip_std + dataset.skip_mean

    sample_denorm = np.zeros_like(sample_norm_np)
    sample_denorm[:, : dataset.state_dim] = pos_denorm
    sample_denorm[:, dataset.state_dim] = skip_denorm

    print("\nReconstructed (denormalized) window:")
    print(sample_denorm)

    # Check consistency: should be nearly equal
    print("\nError statistics (denorm(norm(x)) - x_raw):")
    raw_positions = dataset.skip_trajectories[dataset.indices[batch_idx][0]]
    start = dataset.indices[batch_idx][1]
    raw_window = raw_positions[start : start + dataset.horizon]

    raw_pos = np.array([p for (p, c, tau) in raw_window])
    raw_skip = np.array([c for (p, c, tau) in raw_window])

    # Compare
    pos_err = np.abs(raw_pos - pos_denorm).mean()
    skip_err = np.abs(raw_skip - skip_denorm).mean()

    print(f"  mean position error: {pos_err:.8f}")
    print(f"  mean skip error    : {skip_err:.8f}")

    # ----------------------------------------------------------------------------------
    # 2. Run a single q-sample diffusion step (just to check no shape errors)
    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SANITY CHECK: Single Diffusion Step")
    print("=" * 80)

    model = model.to(device)
    diffusion = diffusion.to(device)

    batch = sample_norm.unsqueeze(0).to(device)  # shape (1, H, 3)
    t = torch.tensor([diffusion.timesteps // 2], dtype=torch.long, device=device)

    with torch.no_grad():
        # This simulates a single forward diffusion step
        noise = torch.randn_like(batch)
        xt = diffusion.q_sample(batch, t, noise)

    print("\nxt (sample after one q-sample step):")
    print(xt.cpu().numpy()[0])

    print("\nShapes:")
    print(f"  batch: {batch.shape}")
    print(f"  xt   : {xt.shape}  (should be same)")

    # ----------------------------------------------------------------------------------
    # 3. Verify denorm(norm(x)) ≈ x for all dims
    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL CHECK: norm→denorm identity")
    print("=" * 80)

    recon_norm = (sample_denorm - dataset.mean) / dataset.std
    recon_denorm = recon_norm * dataset.std + dataset.mean

    err_full = np.abs(recon_denorm - sample_denorm).mean()
    print(f"Mean reconstruction error over full (x,y,skip): {err_full:.10f}")

    if err_full < 1e-5:
        print("✓ PASSED: normalization pipeline is internally consistent.")
    else:
        print("⚠️ WARNING: inconsistency detected.")
