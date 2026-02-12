from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

try:
    import numpy as np
    from safetensors.numpy import save_file

    _has_deps = True
except Exception:
    _has_deps = False

pytestmark = pytest.mark.skipif(not _has_deps, reason="inference deps not installed")

if _has_deps:
    from offload_runtime.loader.safetensors_storage import SafetensorsStorage
    from offload_runtime.types import LayerSpec


def _save_tensors(path: Path, tensors: dict[str, Any]) -> Path:
    """Save numpy tensors to a safetensors file, return the path."""
    save_file(tensors, str(path))
    return path


def _build_layer_spec(
    layer_id: int,
    name: str,
    tensors: dict[str, Any],
    order: list[str],
    compute_dtype: str = "float32",
    full_prefix: str = "",
) -> LayerSpec:
    """Build a LayerSpec with tensor packing metadata."""
    meta: list[dict[str, Any]] = []
    offset = 0
    for tname in order:
        arr = tensors[tname]
        target_dtype = np.dtype(compute_dtype)
        nbytes = int(np.prod(arr.shape)) * target_dtype.itemsize
        meta.append({
            "name": tname,
            "shape": list(arr.shape),
            "dtype": compute_dtype,
            "offset": offset,
            "nbytes": nbytes,
        })
        offset += nbytes
    return LayerSpec(
        layer_id=layer_id,
        name=name,
        nbytes=offset,
        metadata={"tensors": meta, "full_prefix": full_prefix},
    )


class TestSafetensorsStorage:
    def test_basic_load_single_layer(self, tmp_path: Path) -> None:
        w = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([0.5], dtype=np.float32)
        st_path = _save_tensors(tmp_path / "model.safetensors", {"w": w, "b": b})

        spec = _build_layer_spec(0, "L0", {"w": w, "b": b}, ["w", "b"])
        tensor_to_file = {"w": st_path, "b": st_path}

        storage = SafetensorsStorage(
            safetensors_paths=[st_path],
            layer_specs=[spec],
            tensor_to_file=tensor_to_file,
        )
        buf = storage.get(0)
        raw = bytes(buf.view)

        # Unpack and verify
        w_out = np.frombuffer(raw, dtype=np.float32, count=3, offset=0)
        b_out = np.frombuffer(raw, dtype=np.float32, count=1, offset=12)
        np.testing.assert_array_equal(w_out, w)
        np.testing.assert_array_equal(b_out, b)
        storage.close()

    def test_multiple_layers(self, tmp_path: Path) -> None:
        t0 = {"a": np.array([1.0, 2.0], dtype=np.float32)}
        t1 = {"a": np.array([3.0, 4.0], dtype=np.float32)}
        # Store both in one file with distinct names
        st_path = _save_tensors(
            tmp_path / "model.safetensors",
            {"L0.a": np.array([1.0, 2.0], dtype=np.float32),
             "L1.a": np.array([3.0, 4.0], dtype=np.float32)},
        )

        spec0 = _build_layer_spec(0, "L0", {"a": t0["a"]}, ["a"], full_prefix="L0.")
        spec1 = _build_layer_spec(1, "L1", {"a": t1["a"]}, ["a"], full_prefix="L1.")
        tensor_to_file = {"L0.a": st_path, "L1.a": st_path}

        storage = SafetensorsStorage(
            safetensors_paths=[st_path],
            layer_specs=[spec0, spec1],
            tensor_to_file=tensor_to_file,
        )
        buf0 = storage.get(0)
        buf1 = storage.get(1)
        arr0 = np.frombuffer(bytes(buf0.view), dtype=np.float32)
        arr1 = np.frombuffer(bytes(buf1.view), dtype=np.float32)
        np.testing.assert_array_equal(arr0, [1.0, 2.0])
        np.testing.assert_array_equal(arr1, [3.0, 4.0])
        storage.close()

    def test_dtype_conversion_float16_to_float32(self, tmp_path: Path) -> None:
        w_fp16 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        st_path = _save_tensors(tmp_path / "model.safetensors", {"w": w_fp16})

        # LayerSpec expects float32 sizes
        spec = _build_layer_spec(0, "L0", {"w": w_fp16}, ["w"], compute_dtype="float32")
        tensor_to_file = {"w": st_path}

        storage = SafetensorsStorage(
            safetensors_paths=[st_path],
            layer_specs=[spec],
            tensor_to_file=tensor_to_file,
            compute_dtype="float32",
        )
        buf = storage.get(0)
        # float32: 3 * 4 = 12 bytes
        assert buf.nbytes == 12
        arr = np.frombuffer(bytes(buf.view), dtype=np.float32)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0], atol=1e-3)
        storage.close()

    def test_async_prefetch_lifecycle(self, tmp_path: Path) -> None:
        w = np.array([10.0, 20.0], dtype=np.float32)
        st_path = _save_tensors(tmp_path / "model.safetensors", {"w": w})
        spec = _build_layer_spec(0, "L0", {"w": w}, ["w"])
        tensor_to_file = {"w": st_path}

        storage = SafetensorsStorage(
            safetensors_paths=[st_path],
            layer_specs=[spec],
            tensor_to_file=tensor_to_file,
        )
        storage.request(0)
        storage.wait(0)
        buf = storage.get(0)
        arr = np.frombuffer(bytes(buf.view), dtype=np.float32)
        np.testing.assert_array_equal(arr, [10.0, 20.0])

        # release should not error
        storage.release(0)
        storage.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        w = np.array([1.0], dtype=np.float32)
        st_path = _save_tensors(tmp_path / "model.safetensors", {"w": w})
        spec = _build_layer_spec(0, "L0", {"w": w}, ["w"])
        tensor_to_file = {"w": st_path}

        with SafetensorsStorage(
            safetensors_paths=[st_path],
            layer_specs=[spec],
            tensor_to_file=tensor_to_file,
        ) as storage:
            buf = storage.get(0)
            assert buf.nbytes == 4

    def test_multiple_safetensors_files(self, tmp_path: Path) -> None:
        st1 = _save_tensors(tmp_path / "shard1.safetensors", {"a": np.array([1.0], dtype=np.float32)})
        st2 = _save_tensors(tmp_path / "shard2.safetensors", {"b": np.array([2.0], dtype=np.float32)})

        a_arr = np.array([1.0], dtype=np.float32)
        b_arr = np.array([2.0], dtype=np.float32)
        spec = _build_layer_spec(0, "L0", {"a": a_arr, "b": b_arr}, ["a", "b"])
        tensor_to_file = {"a": st1, "b": st2}

        storage = SafetensorsStorage(
            safetensors_paths=[st1, st2],
            layer_specs=[spec],
            tensor_to_file=tensor_to_file,
        )
        buf = storage.get(0)
        raw = bytes(buf.view)
        a_out = np.frombuffer(raw, dtype=np.float32, count=1, offset=0)
        b_out = np.frombuffer(raw, dtype=np.float32, count=1, offset=4)
        np.testing.assert_array_equal(a_out, [1.0])
        np.testing.assert_array_equal(b_out, [2.0])
        storage.close()

    def test_missing_layer_raises(self, tmp_path: Path) -> None:
        st_path = _save_tensors(tmp_path / "model.safetensors", {"w": np.array([1.0], dtype=np.float32)})
        spec = _build_layer_spec(0, "L0", {"w": np.array([1.0], dtype=np.float32)}, ["w"])
        tensor_to_file = {"w": st_path}

        storage = SafetensorsStorage(
            safetensors_paths=[st_path],
            layer_specs=[spec],
            tensor_to_file=tensor_to_file,
        )
        with pytest.raises(KeyError):
            storage.get(99)
        storage.close()
