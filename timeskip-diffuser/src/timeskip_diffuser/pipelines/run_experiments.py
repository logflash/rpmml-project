import matplotlib.pyplot as plt
import numpy as np
import random
from eqnet_independent_copy import EqNet, GaussianDiffusion, DiffuserTrainer, DiffuserPlanner, CompositeReward, expand_spline_from_skip_list, StartReachingReward, GoalReachingReward, SkipTotalTimeSkipPenalty, CurvaturePenalty, LogSkipReward
from point_maze_skip import MinariTrajectoryDatasetWithPseudoActions
import torch
from matplotlib.patches import Rectangle
import mujoco
from torch.utils.data import DataLoader, Dataset
from traj_verifiers.verifier import extract_wall_rects, verify_trajectory_dense, sample_free_point, endpoint_within_eps
from offlineskipdataset import OfflineSkipDataset
    

def run_planning_experiment(
    planner,
    dataset_name,
    num_tasks=100,
    max_tries=10,
    horizon=32,
    seed=0,
    guidance_scale=1.0,
    start_eps=0.05,
    goal_eps=0.05,
):
    """
    Runs planning experiments WITHOUT visualization.

    A task is successful if ANY attempt:
      - has no wall collision
      - starts within start_eps
      - ends within goal_eps
    """

    # -----------------------------
    # Reproducibility
    # -----------------------------
    rng = np.random.default_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # -----------------------------
    # Geometry
    # -----------------------------
    wall_rects = extract_wall_rects(dataset_name)

    # Hard-coded U-Maze bounds
    bounds = (-1.4, 1.4, -1.4, 1.4)

    successes = 0

    for task_id in range(num_tasks):
        print(f"\n=== Task {task_id + 1}/{num_tasks} ===")

        # -------------------------
        # Sample start & goal
        # -------------------------
        start_xy = sample_free_point(wall_rects, rng, bounds)
        goal_xy  = sample_free_point(wall_rects, rng, bounds)

        current = np.array(
            [start_xy[0], start_xy[1], 0.0, 0.0],
            dtype=np.float32,
        )
        goal = goal_xy
        
        reward_fn = CompositeReward(
            [
                StartReachingReward(start_xy, reward_scale=5.0),
                GoalReachingReward(goal, reward_scale=5.0),
                SkipTotalTimeSkipPenalty(reward_scale=0.00),
                CurvaturePenalty(reward_scale=0.05),
                LogSkipReward(reward_scale=0.0),
            ]
        )


        solved = False

        # -------------------------
        # Attempt loop
        # -------------------------
        for attempt in range(max_tries):
            traj = planner.plan_and_reconstruct(
                current,
                goal,
                reward_fn=reward_fn,
                horizon=horizon,
                guidance_scale=guidance_scale,
                condition_on_start=True,
                condition_on_goal=True,
                conditioning_schedule="constant",
                conditioning_strength=0.9,
                spline_func=expand_spline_from_skip_list,
            )

            pos_dense = traj["pos_dense"]

            feasible, _ = verify_trajectory_dense(
                pos_dense, wall_rects
            )

            start_ok, goal_ok, start_err, goal_err = endpoint_within_eps(
                pos_dense,
                start_xy,
                goal_xy,
                start_eps,
                goal_eps,
            )

            success = feasible and start_ok and goal_ok

            if success:
                print(
                    f"  ✓ Attempt {attempt + 1} succeeded "
                    f"(start_err={start_err:.3f}, goal_err={goal_err:.3f})"
                    f"(start={start_xy}, goal={goal_xy}"
                )
                successes += 1
                solved = True
                break
            else:
                print(
                    f"  ✗ Attempt {attempt + 1} failed "
                    f"(collision={not feasible}, "
                    f"start_err={start_err:.3f}, goal_err={goal_err:.3f})"
                    f"(start={start_xy}, goal={goal_xy}"
                )

        if solved:
            print("✓ Task solved")
        else:
            print("✗ Task failed (all attempts)")

    # -----------------------------
    # Final summary
    # -----------------------------
    print("\n==============================")
    print(f"Total tasks  : {num_tasks}")
    print(f"Max tries   : {max_tries}")
    print(f"Successes   : {successes}")
    print(f"Success rate: {successes / num_tasks:.3f}")
    print("==============================")

    return successes


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Load dataset that the model was trained on for denormalization statistics
    OFFLINE_FILE = "/scratch/network/ts4953/dataset_gen/rpmml-project/timeskip-diffuser/src/timeskip_diffuser/datasets/fixed_offline_umaze_independent_skips_h32_mu1_sig1.npz"

    dataset = OfflineSkipDataset(
        OFFLINE_FILE,
        horizon=32
    )

    #Load model + create planner
    eqnet = EqNet(
            state_dim=dataset.traj_dim,
            hidden_dim=128,
            time_dim=32,
            n_layers=10, 
        )

    diffusion = GaussianDiffusion(timesteps=200)
    trainer = DiffuserTrainer(
            model = eqnet,
            diffusion = diffusion,
            dataset=dataset,
            device = device)
    trainer.use_ema_for_inference()

    trainer.load("/scratch/network/ts4953/dataset_gen/rpmml-project/timeskip-diffuser/src/timeskip_diffuser/diffuser/checkpoints/diffuser_fixed_h32_m1_s1_epoch_10.pt")
    planner = DiffuserPlanner(eqnet, diffusion, dataset, device=device)
    
    run_planning_experiment(
        planner,
        dataset_name="D4RL/pointmaze/umaze-v2",
        num_tasks=100,
        max_tries=10,
        horizon=32,
        seed=42,
    )