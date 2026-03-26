from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

CHECKPOINT_DIR = Path("checkpoints")
REFERENCE_SYMLINK = CHECKPOINT_DIR / "reference_model.pt"


def checkpoint_path(iteration: int, episode: int) -> Path:
    return CHECKPOINT_DIR / f"iter_{iteration:04d}_ep_{episode:06d}.pt"


def save_checkpoint(payload: dict[str, Any], keep_last: int = 10) -> str:
    if torch is None:
        raise RuntimeError("torch is required for checkpointing")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(payload["iteration"], payload["episode"])
    payload = dict(payload)
    payload["saved_at"] = time.time()
    torch.save(payload, path)
    _prune_checkpoints(keep_last)
    return str(path)


def load_checkpoint(path: str) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required for checkpointing")
    return torch.load(path, map_location="cpu", weights_only=False)


def mark_reference(path: str):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if REFERENCE_SYMLINK.exists() or REFERENCE_SYMLINK.is_symlink():
            REFERENCE_SYMLINK.unlink()
        os.symlink(Path(path).resolve(), REFERENCE_SYMLINK)
    except OSError:
        REFERENCE_SYMLINK.write_text(path, encoding="utf-8")


def list_checkpoints() -> list[str]:
    if not CHECKPOINT_DIR.exists():
        return []
    return [str(path) for path in sorted(CHECKPOINT_DIR.glob("iter_*.pt"))]


def clear_checkpoints():
    if not CHECKPOINT_DIR.exists():
        return
    for path in CHECKPOINT_DIR.glob("iter_*.pt"):
        path.unlink(missing_ok=True)
    if REFERENCE_SYMLINK.exists() or REFERENCE_SYMLINK.is_symlink():
        REFERENCE_SYMLINK.unlink(missing_ok=True)


def _prune_checkpoints(keep_last: int):
    checkpoints = sorted(CHECKPOINT_DIR.glob("iter_*.pt"))
    stale = checkpoints[:-keep_last]
    for path in stale:
        path.unlink(missing_ok=True)
