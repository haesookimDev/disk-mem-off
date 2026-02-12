from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from offload_runtime.types import HostBuffer, LayerSpec

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover
    safe_open = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


class SafetensorsStorage:
    """LayerStorage backed by safetensors files.

    Reads tensors from one or more ``.safetensors`` files, converts them to
    *compute_dtype* (default ``float32``), and packs all tensors for a single
    layer into a contiguous byte buffer matching the layout described by
    ``LayerSpec.metadata["tensors"]``.

    Follows the same async-prefetch contract as :class:`ShardedMMapStorage`:
    ``request()`` / ``wait()`` / ``get()`` / ``release()``.
    """

    def __init__(
        self,
        safetensors_paths: list[Path],
        layer_specs: list[LayerSpec],
        tensor_to_file: dict[str, Path],
        *,
        compute_dtype: str = "float32",
        max_workers: int = 1,
    ) -> None:
        self._safetensors_paths = safetensors_paths
        self._layer_specs: dict[int, LayerSpec] = {s.layer_id: s for s in layer_specs}
        self._tensor_to_file = tensor_to_file
        self._compute_dtype = compute_dtype

        self._handles: dict[Path, Any] = {}
        self._lock = threading.Lock()
        self._futures: dict[int, Future[HostBuffer]] = {}
        self._ready: dict[int, HostBuffer] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    # -- Context manager --

    def __enter__(self) -> SafetensorsStorage:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # -- Private helpers --

    def _get_handle(self, path: Path) -> Any:
        """Return a (cached) safe_open handle for the given file."""
        if path in self._handles:
            return self._handles[path]
        handle = safe_open(str(path), framework="numpy")
        self._handles[path] = handle
        return handle

    def _load_layer(self, layer_id: int) -> HostBuffer:
        spec = self._layer_specs[layer_id]
        tensor_meta: list[dict[str, Any]] = spec.metadata["tensors"]
        full_prefix: str = spec.metadata.get("full_prefix", "")

        buf = bytearray(spec.nbytes)

        for meta in tensor_meta:
            full_name = f"{full_prefix}{meta['name']}" if full_prefix else meta["name"]
            file_path = self._tensor_to_file[full_name]

            with self._lock:
                handle = self._get_handle(file_path)

            arr = handle.get_tensor(full_name)
            target_dtype = np.dtype(self._compute_dtype)
            if arr.dtype != target_dtype:
                arr = arr.astype(target_dtype)

            raw = arr.tobytes()
            offset = meta["offset"]
            buf[offset : offset + len(raw)] = raw

        return HostBuffer(view=memoryview(bytes(buf)), pinned=False)

    # -- LayerStorage protocol --

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
        self._handles.clear()
