# diffuser/config/pointmaze.py

from diffuser.utils import watch

# ------------------------ experiment naming ------------------------ #

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

logbase = 'logs'


# ===================================================================
#                          DIFFUSION MODEL CONFIG
# ===================================================================

diffusion = {
    # -------------------- model definition -------------------- #
    "model": "models.TemporalUnet",
    "diffusion": "models.GaussianDiffusion",
    "loss_type": "l2",
 
    # -------------------- PointMaze settings ------------------- #
    "horizon": 64,                 # short-horizon planning
    "n_diffusion_steps": 100,
    "dim_mults": (1, 2, 4),        # smaller model for 2D maze
    "attention": False,
    "action_weight": 10,
    "loss_discount": 1.0,
    "loss_weights": None,
    "predict_epsilon": False,
    "clip_denoised": False,

    # -------------------- dataset loader ------------------------ #
    "loader": "datasets.sequence.SequenceDataset",
    "normalizer": "LimitsNormalizer",   # perfect for 2D state bounds
    "preprocess_fns": ["add_deltas"],   # give deltas as features
    "use_padding": True,
    "max_path_length": 600,

    # -------------------- renderer ------------------------------ #
    # No MuJoCo! PointMaze uses a lightweight renderer
    "renderer": "utils.noop.NoopRenderer",

    # -------------------- serialization ------------------------- #
    "logbase": logbase,
    "prefix": "pointmaze/diffusion",
    "exp_name": watch(args_to_watch),

    # -------------------- training settings --------------------- #
    "n_steps_per_epoch": 10000,
    "n_train_steps": 1e6,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "gradient_accumulate_every": 2,
    "ema_decay": 0.995,

    "save_freq": 20000,
    "sample_freq": 20000,
    "n_saves": 5,
    "save_parallel": False,
    "n_reference": 8,
    "bucket": None,
    "device": "cuda",
    "seed": None,
}


# ===================================================================
#                         VALUE FUNCTION CONFIG
# ===================================================================

values = {
    'model': 'models.ValueFunction',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 8),
        'renderer': 'utils.noop.NoopRenderer',

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': -100,
        'normed': False,

        ## dataset
        'loader': 'datasets.sequence.ValueDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'pointmaze/values',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
}


# ===================================================================
#                           PLANNING CONFIG
# ===================================================================

plan = {
    "guide": "sampling.ValueGuide",
    "policy": "sampling.GuidedPolicy",

    "max_episode_length": 600,
    "batch_size": 64,
    "preprocess_fns": [],
    "device": "cuda",
    "seed": None,

    # sampling guidance
    "n_guide_steps": 2,
    "scale": 0.1,
    "t_stopgrad": 2,
    "scale_grad_by_std": True,

    # loader
    "loadbase": None,
    "logbase": logbase,
    "prefix": "pointmaze/plans",
    "exp_name": watch(args_to_watch),

    # rendering
    "vis_freq": 100,
    "max_render": 8,

    # diffusion model
    "horizon": 64,
    "n_diffusion_steps": 100,

    # value function
    "discount": 0.997,

    # loading checkpoints
    "diffusion_loadpath":
        "f:pointmaze/diffusion_H{horizon}_T{n_diffusion_steps}",

    "value_loadpath":
        "f:pointmaze/values_H{horizon}_T{n_diffusion_steps}_d{discount}",

    "diffusion_epoch": "latest",
    "value_epoch": "latest",

    "verbose": True,
    "suffix": "0",
}


# ===================================================================
#                  export the final configuration dict
# ===================================================================

base = {
    "diffusion": diffusion,
    "values": values,
    "plan": plan,
}
