from __future__ import annotations

import gc
import json
import mmap
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from offload_runtime.types import HostBuffer


@dataclass(frozen=True, slots=True)
class LayerEntry:
    path: str
    offset: int
    nbytes: int


class ShardedMMapStorage:
    """Reads layer bytes from shard files using mmap slices.

    Supports async prefetching via background threads:
      request(layer_id) -> triggers background load
      wait(layer_id)    -> blocks until loaded
      get(layer_id)     -> returns loaded buffer (or loads synchronously as fallback)
      release(layer_id) -> discards cached buffer

    Index JSON format:
    {
      "layers": [
        {"layer_id": 0, "path": "weights-000.bin", "offset": 0, "nbytes": 4096}
      ]
    }
    """

    def __init__(self, root: str | Path, index_path: str | Path, max_workers: int = 1) -> None:
        self.root = Path(root)
        self.index_path = Path(index_path)
        self._entries = self._load_index(self.index_path)
        self._files: dict[str, object] = {}
        self._maps: dict[str, mmap.mmap] = {}
        self._lock = threading.Lock()
        self._futures: dict[int, Future[HostBuffer]] = {}
        self._ready: dict[int, HostBuffer] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    def __enter__(self) -> "ShardedMMapStorage":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

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

    def _load_layer(self, layer_id: int) -> HostBuffer:
        entry = self._entries[layer_id]
        with self._lock:
            mm = self._mmap_for(entry.path)
        data = bytes(mm[entry.offset : entry.offset + entry.nbytes])
        return HostBuffer(view=memoryview(data), pinned=False)

    def request(self, layer_id: int) -> None:
        if layer_id in self._futures or layer_id in self._ready:
            return
        future = self._thread_pool.submit(self._load_layer, layer_id)
        self._futures[layer_id] = future

    def wait(self, layer_id: int) -> None:
        future = self._futures.pop(layer_id, None)
        if future is not None:
            self._ready[layer_id] = future.result()

    def get(self, layer_id: int) -> HostBuffer:
        buf = self._ready.pop(layer_id, None)
        if buf is not None:
            return buf
        return self._load_layer(layer_id)

    def release(self, layer_id: int) -> None:
        self._ready.pop(layer_id, None)
        future = self._futures.pop(layer_id, None)
        if future is not None:
            future.cancel()

    def close(self) -> None:
        self._thread_pool.shutdown(wait=False, cancel_futures=True)
        self._futures.clear()
        self._ready.clear()
        gc.collect()
        for mm in self._maps.values():
            mm.close()
        for fp in self._files.values():
            fp.close()
        self._maps.clear()
        self._files.clear()
