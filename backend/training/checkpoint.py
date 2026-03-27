from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

CHECKPOINT_DIR = Path("checkpoints")
REFERENCE_SYMLINK = CHECKPOINT_DIR / "reference_model.pt"
DEFAULT_S3_BUCKET = os.getenv("HEXTACTOE_S3_BUCKET", "hex-tactoe-checkpoints-hjbauer")
DEFAULT_S3_PREFIX = os.getenv("HEXTACTOE_S3_PREFIX", "checkpoints").strip("/")


def checkpoint_path(iteration: int, episode: int) -> Path:
    return CHECKPOINT_DIR / f"iter_{iteration:04d}_ep_{episode:06d}.pt"


def _checkpoint_sort_key(path_or_name: str) -> tuple[int, int, str]:
    name = Path(path_or_name).name
    try:
        iter_part = name.split("_")[1]
        ep_part = name.split("_")[3].split(".")[0]
        return (int(iter_part), int(ep_part), name)
    except Exception:
        return (-1, -1, name)


def _s3_uri(bucket: str, prefix: str, name: str = "") -> str:
    normalized_prefix = prefix.strip("/")
    if not name:
        return f"s3://{bucket}/{normalized_prefix}/"
    return f"s3://{bucket}/{normalized_prefix}/{name}"


def _run_aws_command(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def list_remote_checkpoints(
    bucket: str = DEFAULT_S3_BUCKET,
    prefix: str = DEFAULT_S3_PREFIX,
) -> list[str]:
    if not bucket:
        return []
    result = _run_aws_command(["aws", "s3", "ls", _s3_uri(bucket, prefix)])
    if result is None:
        return []
    entries: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[3].startswith("iter_") and parts[3].endswith(".pt"):
            entries.append(parts[3])
    return sorted(entries, key=_checkpoint_sort_key)


def download_remote_checkpoint(
    name: str,
    bucket: str = DEFAULT_S3_BUCKET,
    prefix: str = DEFAULT_S3_PREFIX,
) -> str | None:
    if not bucket or not name:
        return None
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = CHECKPOINT_DIR / name
    if local_path.exists():
        return str(local_path)
    result = _run_aws_command(["aws", "s3", "cp", _s3_uri(bucket, prefix, name), str(local_path)])
    if result is None:
        return None
    return str(local_path)


def latest_resumable_checkpoint(
    bucket: str = DEFAULT_S3_BUCKET,
    prefix: str = DEFAULT_S3_PREFIX,
) -> str | None:
    local_checkpoints = list_checkpoints()
    local_latest = local_checkpoints[-1] if local_checkpoints else None
    remote_checkpoints = list_remote_checkpoints(bucket=bucket, prefix=prefix)
    remote_latest = remote_checkpoints[-1] if remote_checkpoints else None
    if remote_latest is None:
        return local_latest
    if local_latest is None or _checkpoint_sort_key(remote_latest) > _checkpoint_sort_key(local_latest):
        downloaded = download_remote_checkpoint(remote_latest, bucket=bucket, prefix=prefix)
        return downloaded or local_latest
    return local_latest


def save_checkpoint(
    payload: dict[str, Any],
    keep_last: int = 10,
    s3_bucket: str = DEFAULT_S3_BUCKET,
    s3_prefix: str = DEFAULT_S3_PREFIX,
    s3_keep_last: int = 2,
) -> str:
    if torch is None:
        raise RuntimeError("torch is required for checkpointing")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(payload["iteration"], payload["episode"])
    payload = dict(payload)
    payload["saved_at"] = time.time()
    fd, tmp_path = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=CHECKPOINT_DIR)
    os.close(fd)
    backup_path: str | None = None
    try:
        torch.save(payload, tmp_path)
        if path.exists():
            backup_path = str(path.with_suffix(path.suffix + ".bak"))
            if os.path.exists(backup_path):
                os.unlink(backup_path)
            os.replace(path, backup_path)
        os.replace(tmp_path, path)
        torch.load(path, map_location="cpu", weights_only=False)
        if backup_path is not None and os.path.exists(backup_path):
            os.unlink(backup_path)
    except Exception:
        if backup_path is not None and os.path.exists(backup_path):
            os.replace(backup_path, path)
        raise
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    _prune_checkpoints(keep_last)
    _sync_checkpoint_to_s3(path, bucket=s3_bucket, prefix=s3_prefix, keep_last=s3_keep_last)
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
    return [str(path) for path in sorted(CHECKPOINT_DIR.glob("iter_*.pt"), key=lambda item: _checkpoint_sort_key(str(item)))]


def clear_checkpoints():
    if not CHECKPOINT_DIR.exists():
        return
    for path in CHECKPOINT_DIR.glob("iter_*.pt"):
        path.unlink(missing_ok=True)
    if REFERENCE_SYMLINK.exists() or REFERENCE_SYMLINK.is_symlink():
        REFERENCE_SYMLINK.unlink(missing_ok=True)


def _prune_checkpoints(keep_last: int):
    checkpoints = sorted(CHECKPOINT_DIR.glob("iter_*.pt"), key=lambda item: _checkpoint_sort_key(str(item)))
    stale = checkpoints[:-keep_last]
    for path in stale:
        path.unlink(missing_ok=True)


def _sync_checkpoint_to_s3(
    path: Path,
    bucket: str = DEFAULT_S3_BUCKET,
    prefix: str = DEFAULT_S3_PREFIX,
    keep_last: int = 2,
):
    if not bucket:
        return
    result = _run_aws_command(["aws", "s3", "cp", str(path), _s3_uri(bucket, prefix, path.name)])
    if result is None:
        return
    remote_checkpoints = list_remote_checkpoints(bucket=bucket, prefix=prefix)
    stale = remote_checkpoints[:-max(keep_last, 0)] if keep_last >= 0 else []
    for name in stale:
        _run_aws_command(["aws", "s3", "rm", _s3_uri(bucket, prefix, name)])
