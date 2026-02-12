from __future__ import annotations

import json
import mmap
from dataclasses import dataclass
from pathlib import Path

from offload_runtime.types import HostBuffer


@dataclass(frozen=True, slots=True)
class LayerEntry:
    path: str
    offset: int
    nbytes: int


class ShardedMMapStorage:
    """Reads layer bytes from shard files using mmap slices.

    Index JSON format:
    {
      "layers": [
        {"layer_id": 0, "path": "weights-000.bin", "offset": 0, "nbytes": 4096}
      ]
    }
    """

    def __init__(self, root: str | Path, index_path: str | Path) -> None:
        self.root = Path(root)
        self.index_path = Path(index_path)
        self._entries = self._load_index(self.index_path)
        self._files: dict[str, object] = {}
        self._maps: dict[str, mmap.mmap] = {}

    def _load_index(self, index_path: Path) -> dict[int, LayerEntry]:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        out: dict[int, LayerEntry] = {}
        for item in raw["layers"]:
            out[int(item["layer_id"])] = LayerEntry(
                path=str(item["path"]),
                offset=int(item["offset"]),
                nbytes=int(item["nbytes"]),
            )
        return out

    def _mmap_for(self, rel_path: str) -> mmap.mmap:
        if rel_path in self._maps:
            return self._maps[rel_path]

        abs_path = self.root / rel_path
        fp = abs_path.open("rb")
        mm = mmap.mmap(fp.fileno(), length=0, access=mmap.ACCESS_READ)
        self._files[rel_path] = fp
        self._maps[rel_path] = mm
        return mm

    def request(self, layer_id: int) -> None:
        _ = layer_id

    def wait(self, layer_id: int) -> None:
        _ = layer_id

    def get(self, layer_id: int) -> HostBuffer:
        entry = self._entries[layer_id]
        mm = self._mmap_for(entry.path)
        view = memoryview(mm)[entry.offset : entry.offset + entry.nbytes]
        return HostBuffer(view=view, pinned=False)

    def release(self, layer_id: int) -> None:
        _ = layer_id

    def close(self) -> None:
        for mm in self._maps.values():
            mm.close()
        for fp in self._files.values():
            fp.close()
        self._maps.clear()
        self._files.clear()

