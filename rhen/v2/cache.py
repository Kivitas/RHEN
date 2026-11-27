"""Thread-safe LRU cache of encoded state memories.

Key = sha256(model tag, row length, token ids, numeric features) of one state row. The model tag is
derived from the model object, its config, the autocast flag and the in-place version counters of
all parameters, so a weight update or ``load_state_dict`` changes the tag and old entries are never
served. Rows are stored trimmed to their real length. ``memory_for`` returns [B, Ls, d] for the
batch's own Ls; positions beyond a row's length are zero (they are always masked, so they never
influence any output; fresh ``encode_state`` fills them with unused values). Real positions equal a
fresh encode up to float noise from a different batch composition (bit-equal for the same sub-batch).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Dict, List, Optional

import torch


class StateCache:
    def __init__(self, max_bytes: int):
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._items: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.hits = self.misses = self.evictions = 0
        self.bytes = 0

    @staticmethod
    def model_tag(model: torch.nn.Module) -> str:
        h = hashlib.sha256()
        h.update(f"{id(model)}|{json.dumps(model.cfg.to_dict(), sort_keys=True)}|{torch.is_autocast_enabled('cpu')}".encode())
        for p in model.parameters():
            h.update(f"{p._version},{p.dtype}".encode())
        return h.hexdigest()

    @staticmethod
    def _key(tag: str, ids: torch.Tensor, num: torch.Tensor) -> str:
        h = hashlib.sha256()
        h.update(tag.encode())
        h.update(str(ids.shape[0]).encode())
        h.update(ids.cpu().contiguous().numpy().tobytes())
        h.update(num.float().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                    "bytes": self.bytes, "entries": len(self._items)}

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.bytes = 0

    def _put(self, key: str, value: torch.Tensor) -> None:
        size = value.numel() * value.element_size()
        if size > self.max_bytes:
            return
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return
            self._items[key] = value
            self.bytes += size
            while self.bytes > self.max_bytes and self._items:
                _, old = self._items.popitem(last=False)
                self.bytes -= old.numel() * old.element_size()
                self.evictions += 1

    def _get(self, key: str) -> Optional[torch.Tensor]:
        with self._lock:
            v = self._items.get(key)
            if v is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return v

    @torch.no_grad()
    def memory_for(self, model, batch) -> torch.Tensor:
        ids, mask, num = batch.state_ids, batch.state_mask, batch.state_num
        b, ls = ids.shape
        tag = self.model_tag(model)
        lengths = [int(mask[i].nonzero().max().item()) + 1 if bool(mask[i].any()) else 1 for i in range(b)]
        keys = [self._key(tag, ids[i, : lengths[i]], num[i, : lengths[i]]) for i in range(b)]
        rows: List[Optional[torch.Tensor]] = [self._get(k) for k in keys]
        missing = [i for i, r in enumerate(rows) if r is None]
        if missing:
            # identical rows inside one batch are encoded once
            first: Dict[str, int] = {}
            todo = [i for i in missing if first.setdefault(keys[i], i) == i]
            lm = max(lengths[i] for i in todo)
            sub_ids = ids[todo][:, :lm]
            sub_mask = mask[todo][:, :lm]
            sub_num = num[todo][:, :lm]
            enc = model.encode_state(sub_ids, sub_mask, sub_num)
            for j, i in enumerate(todo):
                v = enc[j, : lengths[i]].detach().clone()
                self._put(keys[i], v)
                rows[i] = v
            for i in missing:
                if rows[i] is None:
                    rows[i] = rows[first[keys[i]]]
        d = rows[0].shape[-1]
        out = torch.zeros(b, ls, d, dtype=rows[0].dtype, device=rows[0].device)
        for i, r in enumerate(rows):
            n = min(r.shape[0], ls)
            out[i, :n] = r[:n]
        return out
