Independent skips dataset gen: `timeskip-diffuser/src/timeskip_diffuser/datasets/build_offline_dataset.py`
Check independent skip trajectories that are near wall (near error): `timeskip-diffuser/src/timeskip_diffuser/datasets/verify.ipynb`
New EqNet with training on independent skips `timeskip-diffuser/src/timeskip_diffuser/diffuser/eqnet_independent.py`
Notebook to test independent skips `timeskip-diffuser/src/timeskip_diffuser/pointmaze/skip_independent.ipynb`



tomasz notes. 7/10/2025


This is trained on horizon=32 but does REALLY good on 24
OFFLINE_FILE = "/scratch/network/ts4953/dataset_gen/rpmml-project/timeskip-diffuser/src/timeskip_diffuser/datasets/fixed_offline_umaze_independent_skips_h32_mu1_sig1.npz"


trainer.load("/scratch/network/ts4953/dataset_gen/rpmml-project/timeskip-diffuser/src/timeskip_diffuser/diffuser/checkpoints/diffuser_fixed_h32_m1_s1_epoch_10.pt")


traj = planner.plan_and_reconstruct(
        current,
        goal,
        reward_fn=reward_fn,
        horizon=24,
        guidance_scale=1.0,
        condition_on_start=True,
        condition_on_goal=True,
        conditioning_schedule="constant",
        conditioning_strength=0.9,
        spline_func=expand_spline_from_skip_list,
    )



MISNAMED -> not actually trained on massive, just trained on normal size
/scratch/network/ts4953/dataset_gen/rpmml-project/timeskip-diffuser/src/timeskip_diffuser/diffuser/checkpoints/diffuser_massive_fixed_h48_m1_s1_epoch_3.pt