"""Dynamic class registry.

Maps string class names to integer ids on first sight. Class ids index into
SPELA's fixed-size per-layer embedding tables, so the registry's capacity
must match the trainer's `num_classes`. We over-allocate (default 256) and
hand out ids as labels appear — true dynamic resizing of the embedding tables
is a v2 problem (see docs/roadmap.md).
"""
from __future__ import annotations

from threading import Lock
from typing import Dict, List, Optional


class RegistryFull(Exception):
    pass


class LabelRegistry:
    def __init__(self, capacity: int = 256):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._name_to_id: Dict[str, int] = {}
        self._id_to_name: List[str] = []
        self._lock = Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._id_to_name)

    def get_or_create(self, name: str) -> int:
        key = name.strip().lower()
        if not key:
            raise ValueError("empty class name")
        with self._lock:
            cid = self._name_to_id.get(key)
            if cid is not None:
                return cid
            if len(self._id_to_name) >= self.capacity:
                raise RegistryFull(
                    f"registry capacity {self.capacity} reached; bump RuntimeConfig.num_classes"
                )
            cid = len(self._id_to_name)
            self._id_to_name.append(key)
            self._name_to_id[key] = cid
            return cid

    def get(self, name: str) -> Optional[int]:
        with self._lock:
            return self._name_to_id.get(name.strip().lower())

    def name(self, cid: int) -> Optional[str]:
        with self._lock:
            if 0 <= cid < len(self._id_to_name):
                return self._id_to_name[cid]
            return None

    def known_names(self) -> List[str]:
        with self._lock:
            return list(self._id_to_name)

    def state_dict(self) -> Dict[str, object]:
        with self._lock:
            return {"capacity": self.capacity, "id_to_name": list(self._id_to_name)}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        with self._lock:
            self.capacity = int(state["capacity"])  # type: ignore[arg-type]
            self._id_to_name = list(state["id_to_name"])  # type: ignore[arg-type]
            self._name_to_id = {n: i for i, n in enumerate(self._id_to_name)}
