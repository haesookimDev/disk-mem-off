from __future__ import annotations

import json
import struct
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from offload_runtime.types import HostBuffer, LayerSpec

try:
    from safetensors import safe_open
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    safe_open = None

try:
    import numpy as np
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    np = None


def _read_bf16_as_float32(path: Path, tensor_name: str) -> Any:
    """Read a bfloat16 tensor from a safetensors file and return as float32."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    meta = header[tensor_name]
    data_start = 8 + header_size
    start, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + start)
        raw = f.read(end - start)
    bf16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = bf16.astype(np.uint32) << 16
    return f32.view(np.float32).reshape(meta["shape"])


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
        max_workers: int = 2,
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
        self._disk_read_ms: dict[int, float] = {}

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

        # Batch handle resolution: acquire lock once instead of per-tensor
        handles_needed: dict[Path, Any] = {}
        for meta in tensor_meta:
            full_name = f"{full_prefix}{meta['name']}" if full_prefix else meta["name"]
            file_path = self._tensor_to_file[full_name]
            if file_path not in handles_needed:
                handles_needed[file_path] = None
        with self._lock:
            for path in handles_needed:
                handles_needed[path] = self._get_handle(path)

        t0 = time.perf_counter()
        target_dtype = np.dtype(self._compute_dtype)
        for meta in tensor_meta:
            full_name = f"{full_prefix}{meta['name']}" if full_prefix else meta["name"]
            file_path = self._tensor_to_file[full_name]
            handle = handles_needed[file_path]

            try:
                arr = handle.get_tensor(full_name)
            except TypeError:
                # bfloat16 not supported by numpy — read raw and convert
                arr = _read_bf16_as_float32(file_path, full_name)
            if arr.dtype != target_dtype:
                arr = arr.astype(target_dtype)

            nbytes = arr.nbytes
            offset = meta["offset"]
            dest = np.frombuffer(buf, dtype=np.uint8, count=nbytes, offset=offset)
            np.copyto(dest, arr.view(np.uint8).ravel())

        self._disk_read_ms[layer_id] = (time.perf_counter() - t0) * 1000.0
        return HostBuffer(view=memoryview(buf), pinned=False)

    def get_disk_read_ms(self, layer_id: int) -> float:
        return self._disk_read_ms.get(layer_id, 0.0)

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
