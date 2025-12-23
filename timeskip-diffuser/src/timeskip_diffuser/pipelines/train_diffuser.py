#!/usr/bin/env python3
"""
Production-ready Diffuser Training Pipeline.

This pipeline orchestrates the full ML workflow:
1) Load and validate YAML configuration
2) Build/load offline dataset via dataset creation script
3) Train model with comprehensive logging and checkpointing
4) Track experiments with metadata and artifacts

Usage:

> python train_diffuser.py --config configs/config_flat_umaze_eqnet.yaml

> python train_diffuser.py --config configs/config_flat_umaze_eqnet.yaml \
    --resume runs/experiment_123

> python train_diffuser.py --config configs/config_flat_umaze_eqnet.yaml \
    --no-train  # dataset only

"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
import socket
import subprocess
import sys
import traceback
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from tqdm import tqdm

from timeskip_diffuser.datasets.point_maze.medium import MediumFlatDataset
from timeskip_diffuser.datasets.point_maze.offline_skip.offline_skip import (
    OfflineSkipDataset,
)
from timeskip_diffuser.datasets.point_maze.open import OpenFlatDataset
from timeskip_diffuser.datasets.point_maze.umaze import UMazeFlatDataset
from timeskip_diffuser.diffuser.diffusion import GaussianDiffusion
from timeskip_diffuser.diffuser.nets import EqNet, TemporalUNet
from timeskip_diffuser.diffuser.trainer import DiffuserTrainer

# ----------------------------
# Logging Setup
# ----------------------------


def setup_logging(run_dir: Path, verbose: bool = False) -> logging.Logger:
    """
    Configure comprehensive logging to both file and console.

    Args:
        run_dir: Directory where logs will be saved
        verbose: If True, set console logging to DEBUG level

    Returns:
        Configured logger instance
    """
    log_file = run_dir / "pipeline.log"

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # File handler - always DEBUG level
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler - INFO or DEBUG based on verbose flag
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Configure logger - clear any existing handlers first to avoid duplicates
    logger = logging.getLogger("train_pipeline")
    logger.handlers.clear()  # Remove any existing handlers
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent duplicate logs from propagating
    logger.propagate = False

    return logger


# ----------------------------
# Utilities
# ----------------------------


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load and parse YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError(f"Config file is empty: {path}")
            return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file {path}: {e}")


def ensure_dir(p: str | Path) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        p: Directory path to create

    Returns:
        Path object for the directory
    """
    p = Path(p)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise PermissionError(f"Cannot create directory (permission denied): {p}")
    except OSError as e:
        raise OSError(f"Cannot create directory {p}: {e}")
    return p


def sha256_file(path: str | Path) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        path: Path to file

    Returns:
        Hex digest of file hash
    """
    h = hashlib.sha256()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {path}")

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(
    cmd: list[str],
    cwd: Optional[str | Path] = None,
    env: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Run a command, streaming stdout/stderr; raise on nonzero exit.

    Args:
        cmd: Command and arguments as list
        cwd: Working directory for command
        env: Environment variables
        logger: Logger for output (uses print if None)

    Raises:
        subprocess.CalledProcessError: If command exits with non-zero status
    """
    cmd_str = " ".join(cmd)
    if logger:
        logger.info(f"Running command: {cmd_str}")
    else:
        print(f"\n[cmd] {cmd_str}")

    try:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)
    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed with exit code {e.returncode}: {cmd_str}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    except FileNotFoundError:
        error_msg = f"Command not found: {cmd[0]}"
        if logger:
            logger.error(error_msg)
        raise FileNotFoundError(error_msg)


def validate_config_schema(config: Dict[str, Any]) -> None:
    """
    Validate that required configuration keys exist.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If required keys are missing or invalid
    """
    required_keys = {
        "env": ["name"],
        "model": ["architecture"],
        "paths": ["work_dir"],  # dataset_script and dataset_out are optional
    }

    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing required config key: '{section}.{key}'")

    # Validate environment name
    valid_envs = ["umaze", "medium", "open"]
    env_name = config["env"]["name"].lower()
    if env_name not in valid_envs:
        raise ValueError(f"Invalid env.name: '{env_name}'. Must be one of {valid_envs}")

    # Validate architecture
    valid_archs = ["unet", "eqnet"]
    arch = config["model"]["architecture"].lower()
    if arch not in valid_archs:
        raise ValueError(
            f"Invalid model.architecture: '{arch}'. Must be one of {valid_archs}"
        )

    # Validate seed
    seed = config.get("seed", 0)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Invalid seed: {seed}. Must be non-negative integer")


def env_to_dataset_id(env_name: str) -> str:
    """
    Map environment name to D4RL dataset ID.

    Args:
        env_name: Environment name (umaze, medium, open)

    Returns:
        D4RL dataset identifier

    Raises:
        ValueError: If environment name is unknown
    """
    env_name = env_name.lower()
    env_map = {
        "umaze": "D4RL/pointmaze/umaze-v2",
        "medium": "D4RL/pointmaze/medium-v2",
        "open": "D4RL/pointmaze/open-v2",
    }

    if env_name not in env_map:
        raise ValueError(
            f"Unknown env_name: '{env_name}'. Valid options: {list(env_map.keys())}"
        )

    return env_map[env_name]


# ----------------------------
# Experiment Metadata
# ----------------------------


def capture_experiment_metadata(cfg: "RunConfig", config_path: Path) -> Dict[str, Any]:
    """
    Capture comprehensive metadata about the experiment.

    Args:
        cfg: Run configuration
        config_path: Path to original config file

    Returns:
        Dictionary with experiment metadata
    """

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config_file": str(config_path),
        "experiment": {
            "env_name": cfg.env_name,
            "architecture": cfg.architecture,
            "seed": cfg.seed,
            "skips": cfg.skips,
            "dataset_type": cfg.dataset_type,
        },
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "working_directory": str(Path.cwd()),
        },
        "paths": {
            "dataset_path": str(cfg.dataset_path) if cfg.dataset_path else None,
        },
        "dataset_args": cfg.dataset_args,
        "train_args": cfg.train_args,
    }

    # Try to capture git info if available
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        metadata["git"] = {
            "commit": git_commit,
            "branch": git_branch,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        metadata["git"] = None

    return metadata


# ----------------------------
# Config schema
# ----------------------------


@dataclass
class RunConfig:
    """
    Configuration for a training run.

    This dataclass contains all parameters needed to execute a complete
    training pipeline including dataset generation and model training.
    """

    # High-level experiment configuration
    env_name: str  # e.g. "umaze", "medium", "open"
    architecture: str  # "unet" or "eqnet"
    seed: int  # Random seed for reproducibility
    skips: bool  # Whether model uses skip connections

    # Dataset configuration
    dataset_type: str  # "skip" (npz file with skip data) or "flat" (UMazeFlatDataset)
    dataset_path: Optional[Path]  # Path to npz file (for skip datasets only)

    # Directory and file paths
    work_dir: Path  # Root directory for experiment outputs

    # Arguments passed to dataset creation
    dataset_args: Dict[str, Any]

    # Arguments passed to training code
    train_args: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary with paths as strings."""
        return {
            "env_name": self.env_name,
            "architecture": self.architecture,
            "seed": self.seed,
            "skips": self.skips,
            "dataset_type": self.dataset_type,
            "work_dir": str(self.work_dir),
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "dataset_args": self.dataset_args,
            "train_args": self.train_args,
        }


def parse_config(d: Dict[str, Any]) -> RunConfig:
    """
    Parse and validate configuration dictionary into RunConfig.

    Args:
        d: Raw configuration dictionary from YAML

    Returns:
        Validated RunConfig object

    Raises:
        ValueError: If configuration is invalid
        KeyError: If required keys are missing
    """
    # Validate schema first
    validate_config_schema(d)

    # Extract and normalize values
    env_name = d["env"]["name"]
    arch = d["model"]["architecture"]
    seed = int(d.get("seed", 0))
    skips = bool(d["model"].get("skips", False))

    # Dataset type: 'flat' (position-only from Minari) or 'skip' (offline .npz with skip)
    dataset_type = d.get("dataset", {}).get("type", "flat")
    if dataset_type not in ["skip", "flat"]:
        raise ValueError(
            f"Invalid dataset.type: '{dataset_type}'. Must be 'flat' or 'skip'"
        )

    # Expand paths (support env variables and ~)
    work_dir = Path(d["paths"]["work_dir"]).expanduser()

    # For skip dataset, get npz_file path from dataset config or construct default
    if dataset_type == "skip":
        # Check if npz_file is specified in dataset args
        npz_file = d.get("dataset", {}).get("npz_file")
        if npz_file:
            dataset_path = Path(npz_file).expanduser()
        else:
            # Auto-construct path based on environment name
            # Default: timeskip-diffuser/datasets/offline_datasets/{env_name}_h32_mu1_sig1.npz
            base_dir = Path(__file__).parent.parent / "datasets" / "offline_datasets"
            horizon = d.get("dataset", {}).get("args", {}).get("horizon", 32)
            dataset_path = base_dir / f"{env_name}_h{horizon}_mu1_sig1.npz"
    else:
        # For flat dataset, no dataset path needed
        dataset_path = None

    # Build arguments dictionaries
    dataset_args = d.get("dataset", {}).get("args", {})
    train_args = d.get("train", {}).get("args", {})

    # Set defaults for dataset args
    dataset_args.setdefault("horizon", 32)
    if dataset_type == "skip":
        dataset_args.setdefault("dataset_id", env_to_dataset_id(env_name))

    # Set defaults for training args
    train_args.setdefault("env", env_name)
    train_args.setdefault("architecture", arch)
    train_args.setdefault("seed", seed)
    train_args.setdefault("dataset_type", dataset_type)

    # For skip: pass dataset path; for flat: pass horizon
    if dataset_type == "skip":
        train_args.setdefault("dataset_path", str(dataset_path))
        train_args.setdefault("horizon", dataset_args.get("horizon", 32))
    elif dataset_type == "flat":
        train_args.setdefault("horizon", dataset_args.get("horizon", 32))

    return RunConfig(
        env_name=env_name,
        architecture=arch,
        seed=seed,
        skips=skips,
        dataset_type=dataset_type,
        work_dir=work_dir,
        dataset_path=dataset_path,
        dataset_args=dataset_args,
        train_args=train_args,
    )


# ----------------------------
# Dataset build
# ----------------------------


def dataset_already_built(dataset_path: Path) -> bool:
    """Check if the dataset is already built."""
    return (
        dataset_path.exists()
        and dataset_path.is_file()
        and dataset_path.suffix == ".npz"
    )


def build_dataset(cfg: RunConfig, run_dir: Path, logger: logging.Logger) -> None:
    """
    Validate or build dataset.

    For 'flat' dataset type, this step is skipped as UMazeFlatDataset
    loads data directly from Minari.

    For 'skip' dataset type, validates that the npz file exists.

    Args:
        cfg: Run configuration
        run_dir: Directory for this run
        logger: Logger instance

    Raises:
        FileNotFoundError: If skip dataset file is missing
        RuntimeError: If dataset build fails or output not found
    """
    # Skip dataset building for flat dataset (loads directly from Minari)
    if cfg.dataset_type == "flat":
        logger.info(
            f"Dataset type is 'flat' - will use {cfg.env_name.capitalize()}FlatDataset"
        )
        logger.info("Skipping offline dataset build (loads from Minari directly)")
        return

    # For skip dataset, validate that npz file exists
    if cfg.dataset_type == "skip":
        if not cfg.dataset_path:
            raise ValueError("dataset_path is required for skip dataset type")

        if not cfg.dataset_path.exists():
            raise FileNotFoundError(
                f"Skip dataset file not found: {cfg.dataset_path}\n"
                f"Please ensure the npz file exists or specify the correct path in config."
            )

        logger.info(f"Using skip dataset: {cfg.dataset_path}")
        file_size = cfg.dataset_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Dataset size: {file_size:.2f} MB")

        # Copy into run_dir for archival
        archived = run_dir / "artifacts" / cfg.dataset_path.name
        ensure_dir(archived.parent)
        shutil.copy2(cfg.dataset_path, archived)
        logger.info(f"Dataset archived to: {archived.relative_to(run_dir)}")
        return

    # This shouldn't be reached with current valid dataset types
    raise ValueError(f"Unknown dataset type: {cfg.dataset_type}")


# ----------------------------
# Training hook
# ----------------------------


def train_model(cfg: RunConfig, run_dir: Path, logger: logging.Logger) -> None:
    """
    Train the diffusion model with embedded training logic.

    This function directly implements the training loop instead of using
    external training entry modules. It:
    1. Loads the appropriate dataset (flat or offline)
    2. Instantiates the model architecture (EqNet or TemporalUNet)
    3. Creates the diffusion model and trainer
    4. Runs the training loop with checkpointing

    Args:
        cfg: Run configuration
        run_dir: Directory for this run (for checkpoint saving)
        logger: Logger instance

    Raises:
        ValueError: If dataset type or architecture is invalid
        ImportError: If required dependencies cannot be imported
    """

    logger.info(
        f"Starting training with architecture={cfg.architecture}, "
        f"dataset_type={cfg.dataset_type}"
    )

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # ======================================================================
    # 1. Load Dataset
    # ======================================================================
    logger.info(f"Loading dataset (type={cfg.dataset_type})...")

    if cfg.dataset_type == "flat":
        # Flat dataset: load directly from Minari (position-only)
        # Select dataset class based on environment name
        if cfg.env_name.lower() == "umaze":
            DatasetClass = UMazeFlatDataset
            dataset_name = "UMazeFlatDataset"
        elif cfg.env_name.lower() == "medium":
            DatasetClass = MediumFlatDataset
            dataset_name = "MediumFlatDataset"
        elif cfg.env_name.lower() == "open":
            DatasetClass = OpenFlatDataset
            dataset_name = "OpenFlatDataset"
        else:
            raise ValueError(
                f"Unknown env_name for flat dataset: '{cfg.env_name}'. "
                f"Valid options: umaze, medium, open"
            )

        horizon = int(cfg.train_args.get("horizon", 32))
        dataset = DatasetClass(horizon=horizon)
        traj_dim = 2  # (x, y) only
        logger.info(
            f"Loaded {dataset_name}: {len(dataset)} samples, state_dim={traj_dim}, "
            f"horizon={horizon}"
        )

    elif cfg.dataset_type == "skip":
        # Skip dataset: load from pre-built .npz file (position + skip)

        dataset_path = cfg.train_args.get("dataset_path")
        if not dataset_path:
            raise ValueError("dataset_path required in train_args for skip dataset")

        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        horizon = int(cfg.train_args.get("horizon", 32))
        dataset_id = env_to_dataset_id(cfg.env_name)
        dataset = OfflineSkipDataset(
            file_path=dataset_path, dataset_id=dataset_id, horizon=horizon
        )
        traj_dim = 3  # (x, y, skip)
        logger.info(
            f"Loaded OfflineSkipDataset from {dataset_path}: {len(dataset)} samples, "
            f"traj_dim={traj_dim}, horizon={horizon}"
        )

    else:
        raise ValueError(
            f"Invalid dataset_type: {cfg.dataset_type}. Must be 'flat' or 'skip'"
        )

    # ======================================================================
    # 2. Instantiate Model Architecture
    # ======================================================================
    logger.info(f"Instantiating model architecture: {cfg.architecture}")

    if cfg.architecture.lower() == "eqnet":
        hidden_dim = int(cfg.train_args.get("hidden_dim", 128))
        time_dim = int(cfg.train_args.get("time_dim", 32))
        n_layers = int(cfg.train_args.get("n_layers", 10))

        model = EqNet(
            state_dim=traj_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            n_layers=n_layers,
        )
        logger.info(
            f"Created EqNet: state_dim={traj_dim}, hidden_dim={hidden_dim},"
            f"time_dim={time_dim}, n_layers={n_layers}"
        )

    elif cfg.architecture.lower() == "unet":
        hidden_dims = cfg.train_args.get("hidden_dims", [128, 256, 512])
        if isinstance(hidden_dims, list):
            hidden_dims = [int(x) for x in hidden_dims]
        time_dim = int(cfg.train_args.get("time_dim", 64))

        model = TemporalUNet(
            state_dim=traj_dim,
            hidden_dims=hidden_dims,
            time_dim=time_dim,
        )
        logger.info(
            f"Created TemporalUNet: state_dim={traj_dim}, hidden_dims={hidden_dims},"
            f"time_dim={time_dim}"
        )

    else:
        raise ValueError(
            f"Invalid architecture: {cfg.architecture}. Must be 'eqnet' or 'unet'"
        )

    # ======================================================================
    # 3. Create Diffusion Model
    # ======================================================================
    logger.info("Creating GaussianDiffusion model...")

    timesteps = int(cfg.train_args.get("timesteps", 200))
    diffusion = GaussianDiffusion(timesteps=timesteps)
    logger.info(f"Created GaussianDiffusion with timesteps={timesteps}")

    # ======================================================================
    # 4. Create Trainer
    # ======================================================================
    logger.info("Creating DiffuserTrainer...")

    lr = float(cfg.train_args.get("lr", 1e-4))
    ema_beta = float(cfg.train_args.get("ema_beta", 0.999))

    trainer = DiffuserTrainer(
        model=model,
        diffusion=diffusion,
        dataset=dataset,
        lr=lr,
        device=device,
        ema_beta=ema_beta,
    )
    logger.info(
        f"Created DiffuserTrainer: lr={lr}, device={device}, ema_beta={ema_beta}"
    )

    # ======================================================================
    # 5. Train
    # ======================================================================
    logger.info("Starting training loop...")
    epochs = int(cfg.train_args.get("epochs", 100))
    save_every = int(cfg.train_args.get("save_every", 10))

    # Update checkpoint path to use run_dir
    checkpoint_dir = run_dir / "checkpoints"
    ensure_dir(checkpoint_dir)

    # Monkey-patch the trainer's save_checkpoint to use our run_dir
    original_save = trainer.save_checkpoint

    def custom_save(epoch):
        path = checkpoint_dir / f"diffuser_{cfg.architecture}_epoch_{epoch}.pt"
        original_save(str(path))
        logger.info(f"Saved checkpoint: {path.name}")

    # Override the save method used in trainer.train()
    # We'll manually call training steps with custom checkpoint saving
    logger.info(f"Training for {epochs} epochs with save_every={save_every}")

    try:
        # Call the trainer's train method
        # Note: We need to temporarily override save behavior

        def custom_train(self, epochs, save_every):
            """Modified train method with custom checkpoint paths."""
            self.model.train()

            for epoch in range(epochs):
                losses = []

                pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")

                for batch in pbar:
                    loss = self.train_step(batch)
                    losses.append(loss)

                    # Update progress bar
                    pbar.set_postfix(
                        {
                            "loss": f"{loss:.4f}",
                            "avg_loss": f"{sum(losses)/len(losses):.4f}",
                            "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                        }
                    )

                avg_loss = sum(losses) / len(losses)
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Loss = {avg_loss:.4f}, "
                    f"LR = {self.scheduler.get_last_lr()[0]:.2e}"
                )

                # Step the scheduler
                self.scheduler.step()

                # Save checkpoint
                if save_every > 0 and (epoch + 1) % save_every == 0:
                    custom_save(epoch + 1)

        # Bind the custom train method
        trainer.train = types.MethodType(custom_train, trainer)

        # Run training
        trainer.train(epochs=epochs, save_every=save_every)

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.debug(traceback.format_exc())
        raise


# ----------------------------
# Main
# ----------------------------


def create_run_directory(cfg: RunConfig, resume_dir: Optional[Path] = None) -> Path:
    """
    Create or resume run directory with proper structure.

    Args:
        cfg: Run configuration
        resume_dir: If provided, resume from this directory

    Returns:
        Path to run directory
    """
    if resume_dir:
        if not resume_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
        return resume_dir

    # Create new run directory with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{cfg.env_name}_{cfg.architecture}_{cfg.dataset_type}_seed{cfg.seed}_{ts}"
    )
    run_dir = ensure_dir(cfg.work_dir / run_name)

    # Create subdirectories
    ensure_dir(run_dir / "artifacts")
    ensure_dir(run_dir / "checkpoints")

    return run_dir


def save_experiment_artifacts(
    cfg: RunConfig, run_dir: Path, config_path: Path, logger: logging.Logger
) -> None:
    """
    Save all experiment configuration and metadata.

    Args:
        cfg: Run configuration
        run_dir: Directory for this run
        config_path: Path to original config file
        logger: Logger instance
    """
    # Save resolved config in multiple formats
    resolved = cfg.to_dict()
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, default_flow_style=False)
    )
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2))
    logger.debug("Saved resolved configuration")

    # Save comprehensive metadata
    metadata = capture_experiment_metadata(cfg, config_path)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.debug("Saved experiment metadata")

    # Copy original config for reference
    shutil.copy2(config_path, run_dir / "original_config.yaml")
    logger.debug("Saved original configuration")


def main():
    """Main entry point for training pipeline."""
    # Parse command line arguments
    ap = argparse.ArgumentParser(
        description="Production-ready Diffuser Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config configs/run.yaml
  %(prog)s --config configs/run.yaml --resume runs/experiment_123
  %(prog)s --config configs/run.yaml --no-train --verbose
        """,
    )
    ap.add_argument("--config", required=True, help="Path to YAML configuration file")
    ap.add_argument("--resume", type=Path, help="Resume from existing run directory")
    ap.add_argument(
        "--no-train",
        action="store_true",
        help="Only build dataset and log metadata (skip training)",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Enable verbose (DEBUG level) logging"
    )
    args = ap.parse_args()

    config_path = Path(args.config)

    # Initial setup - create temporary run dir for logging
    temp_work_dir = Path("runs")
    temp_work_dir.mkdir(exist_ok=True)
    temp_run_dir = temp_work_dir / f"tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    temp_run_dir.mkdir(exist_ok=True)

    # Setup initial logger
    logger = setup_logging(temp_run_dir, verbose=args.verbose)
    logger.info("=" * 70)
    logger.info("Diffuser Training Pipeline")
    logger.info("=" * 70)

    try:
        # Load and parse configuration
        logger.info(f"Loading configuration from: {config_path}")
        raw_config = load_yaml(config_path)
        cfg = parse_config(raw_config)
        logger.info("Configuration loaded and validated successfully")

        # Create/resume run directory
        run_dir = create_run_directory(cfg, resume_dir=args.resume)
        logger.info(f"Run directory: {run_dir}")

        # Move logger to actual run directory if we created a temp one
        if temp_run_dir != run_dir:
            # Setup proper logger in run directory
            logger = setup_logging(run_dir, verbose=args.verbose)
            logger.info("=" * 70)
            logger.info("Diffuser Training Pipeline")
            logger.info("=" * 70)
            logger.info(f"Run directory: {run_dir}")

        # Save all configuration and metadata
        save_experiment_artifacts(cfg, run_dir, config_path, logger)

        # Build dataset
        logger.info("=" * 70)
        logger.info("Dataset Preparation")
        logger.info("=" * 70)
        build_dataset(cfg, run_dir, logger)

        # Log dataset hash (only for skip datasets)
        if cfg.dataset_type == "skip" and cfg.dataset_path:
            ds_hash = sha256_file(cfg.dataset_path)
            (run_dir / "dataset.sha256").write_text(ds_hash + "\n")
            logger.info(f"Dataset hash saved: {ds_hash[:16]}...")

        # Train model
        if not args.no_train:
            logger.info("=" * 70)
            logger.info("Model Training")
            logger.info("=" * 70)
            train_model(cfg, run_dir, logger)
        else:
            logger.info("Skipping training (--no-train flag set)")

        # Success
        logger.info("=" * 70)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {run_dir}")
        logger.info("=" * 70)

        # Cleanup temp directory if it was created
        if temp_run_dir.exists() and temp_run_dir != run_dir:
            shutil.rmtree(temp_run_dir, ignore_errors=True)

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("=" * 70)
        logger.error("Pipeline failed with error:")
        logger.error(str(e))
        logger.error("=" * 70)
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
