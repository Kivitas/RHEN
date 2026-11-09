"""Run manifests: everything needed to say exactly what produced a result."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict, Optional

import torch

from rhen.research.device import describe_environment

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def canonical_hash(obj: Any) -> str:
    """SHA-256 of the canonical JSON of ``obj`` (sorted keys, no whitespace)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_state(repo: str = REPO_ROOT) -> Dict[str, Any]:
    """Commit, branch and whether the working tree has uncommitted changes."""

    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    status = run("status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "dirty_files": status.splitlines()[:50] if status else [],
    }


def build_manifest(
    train_config: Dict[str, Any],
    model_config: Dict[str, Any],
    device: Optional[torch.device] = None,
    world_size: int = 1,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "train_config": train_config,
        "train_config_sha256": canonical_hash(train_config),
        "model_config_sha256": canonical_hash(model_config),
        "git": git_state(),
        "environment": describe_environment(device),
        "world_size": world_size,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write JSON atomically (temp file + rename), so a crash never leaves a torn file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)
