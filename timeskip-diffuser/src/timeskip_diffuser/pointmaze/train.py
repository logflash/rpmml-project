#!/usr/bin/env python3
"""
Verifier script:
1) Load YAML config
2) Build offline dataset via your dataset creation script
3) Train model (hooks for your training code)

Usage:
  python verify_run.py --config configs/run.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import importlib
import yaml


# ----------------------------
# Utilities
# ----------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str], cwd: Optional[str | Path] = None, env: Optional[dict] = None) -> None:
    """Run a command, streaming stdout/stderr; raise on nonzero exit."""
    print("\n[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)

def env_to_dataset_id(env_name: str) -> str:
    env_name = env_name.lower()
    if env_name == "umaze":
        return "D4RL/pointmaze/umaze-v2"
    elif env_name == "medium":
        return "D4RL/pointmaze/medium-v2"
    elif env_name == "open":
        return "D4RL/pointmaze/open-v2"
    else:
        raise ValueError(f"Unknown env_name: {env_name}")

def select_training_entry(
    arch: str,
    skips: bool,
    paths_cfg: Dict[str, Any],
) -> str:
    arch = arch.lower()

    key = f"{arch}_{'skips' if skips else 'noskips'}_training_entry"

    if key not in paths_cfg:
        raise KeyError(
            f"Missing training entry for (arch={arch}, skips={skips}). "
            f"Expected paths.{key} in config."
        )

    return paths_cfg[key]

# ----------------------------
# Config schema (lightweight)
# ----------------------------

@dataclass
class RunConfig:
    # High-level
    env_name: str                 # e.g. "umaze", "medium", "open"
    architecture: str             # "unet" or "eqnet"
    seed: int
    skips: bool
    # Paths
    work_dir: Path
    dataset_script: Path          # dataset creation script path
    dataset_out: Path             # where the .npz should be written
    training_entry: str           # python module path or script path to train 

    # Dataset build args (passed to dataset script)
    dataset_args: Dict[str, Any]

    # Train args (passed to training entry)
    train_args: Dict[str, Any]


def parse_config(d: Dict[str, Any]) -> RunConfig:
    # Required
    env_name = d["env"]["name"]
    arch = d["model"]["architecture"]
    seed = int(d.get("seed", 0))
    skips = bool(d["model"].get("skips", False))
    work_dir = Path(d["paths"]["work_dir"])
    dataset_script = Path(d["paths"]["dataset_script"])
    dataset_out = Path(d["paths"]["dataset_out"])
    training_entry = select_training_entry(
    arch=arch,
    skips=skips,
    paths_cfg=d["paths"],
    )

    dataset_args = d.get("dataset", {}).get("args", {})
    train_args = d.get("train", {}).get("args", {})

    dataset_args.setdefault("dataset_id", env_to_dataset_id(env_name))
    dataset_args.setdefault("seed", seed)

    train_args.setdefault("env", env_name)
    train_args.setdefault("architecture", arch)
    train_args.setdefault("seed", seed)
    train_args.setdefault("dataset_path", str(dataset_out))

    return RunConfig(
        env_name=env_name,
        architecture=arch,
        seed=seed,
        skips=skips,
        work_dir=work_dir,
        dataset_script=dataset_script,
        dataset_out=dataset_out,
        training_entry=training_entry,
        dataset_args=dataset_args,
        train_args=train_args,
    )


# ----------------------------
# Dataset build
# ----------------------------

def dataset_already_built(dataset_path: Path) -> bool:
    return dataset_path.exists() and dataset_path.is_file() and dataset_path.suffix == ".npz"


def build_dataset(cfg: RunConfig, run_dir: Path) -> None:
    """
    Calls dataset script as a subprocess.
    """
    ensure_dir(cfg.dataset_out.parent)

    # If already exists, keep it but record hash
    if dataset_already_built(cfg.dataset_out):
        print(f"[dataset] Exists: {cfg.dataset_out}")
        return

    # Build command line args from cfg.dataset_args
    cmd = [sys.executable, str(cfg.dataset_script)]

    # Standard convention: use --out for output path if present
    # If your script uses a different flag (e.g., --offline_file), change here.
    if "out" not in cfg.dataset_args and "offline_file" not in cfg.dataset_args:
        cmd += ["--out", str(cfg.dataset_out)]

    for k, v in cfg.dataset_args.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]

    # Run
    run_cmd(cmd)

    # Quick verify
    if not dataset_already_built(cfg.dataset_out):
        raise RuntimeError(f"[dataset] Script ran but output not found: {cfg.dataset_out}")

    print(f"[dataset] Built: {cfg.dataset_out} (sha256={sha256_file(cfg.dataset_out)[:12]}...)")

    # Copy into run_dir for archival if desired
    archived = run_dir / "artifacts" / cfg.dataset_out.name
    ensure_dir(archived.parent)
    shutil.copy2(cfg.dataset_out, archived)
    print(f"[dataset] Archived to: {archived}")


# ----------------------------
# Training hook
# ----------------------------

def train_model(cfg: RunConfig) -> None:
    """
    Import trainer module selected by (arch, skips) and run it.
    """

    module_name = cfg.training_entry
    print(f"[train] importing training module: {module_name}")

    mod = importlib.import_module(module_name)

    if not hasattr(mod, "run"):
        raise AttributeError(
            f"Training module '{module_name}' must define run(cfg)"
        )

    mod.run(cfg)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--no-train", action="store_true", help="Only build dataset + log metadata.")
    args = ap.parse_args()

    raw = load_yaml(args.config)
    cfg = parse_config(raw)

    # Create run directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(cfg.work_dir / f"{cfg.env_name}_{cfg.architecture}_seed{cfg.seed}_{ts}")
    ensure_dir(run_dir / "artifacts")

    # Save resolved config
    resolved = {
        "env_name": cfg.env_name,
        "architecture": cfg.architecture,
        "seed": cfg.seed,
        "skips": cfg.skips,
        "dataset_script": str(cfg.dataset_script),
        "dataset_out": str(cfg.dataset_out),
        "training_entry": cfg.training_entry,
        "dataset_args": cfg.dataset_args,
        "train_args": cfg.train_args,
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2))

    print(f"[run] run_dir: {run_dir}")

    # Build dataset
    build_dataset(cfg, run_dir)

    # Log dataset hash
    ds_hash = sha256_file(cfg.dataset_out)
    (run_dir / "dataset.sha256").write_text(ds_hash + "\n")
    print(f"[run] dataset sha256: {ds_hash}")

    # Train
    if not args.no_train:
        train_model(cfg)

    print("[done]")


if __name__ == "__main__":
    main()
