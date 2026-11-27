"""Checkpoint save/load for RHEN-2.

A checkpoint is one ``torch.save`` dict: ``{"format": 1, "cfg": cfg.to_dict(), "state_dict": ..., "extra": ...}``.
It is loaded with ``weights_only=True`` so no pickled code can run. ``extra`` must be JSON-serialisable
(it is stored as a JSON string, so only plain data survives). Unknown format versions and unknown
config keys are rejected.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch

from rhen.v2.config import Rhen2Config
from rhen.v2.model import Rhen2Model

FORMAT_VERSION = 1


def save_checkpoint(path: str, model: Rhen2Model, cfg: Rhen2Config, extra: Optional[Dict[str, Any]] = None) -> None:
    extra_json = json.dumps(extra if extra is not None else {}, sort_keys=True)   # raises on non-JSON data
    payload = {
        "format": FORMAT_VERSION,
        "cfg": json.dumps(cfg.to_dict(), sort_keys=True),
        "state_dict": {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        "extra": extra_json,
    }
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)      # a crash never leaves a half-written checkpoint at ``path``


def load_checkpoint(path: str, map_location: Any = "cpu") -> Tuple[Rhen2Model, Dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {payload.get('format') if isinstance(payload, dict) else None!r}")
    cfg = Rhen2Config.from_dict(json.loads(payload["cfg"]))        # rejects unknown keys
    model = Rhen2Model(cfg)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, json.loads(payload["extra"])
