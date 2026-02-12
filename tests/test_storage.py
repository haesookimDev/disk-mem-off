from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from offload_runtime.storage.in_memory import InMemoryStorage
from offload_runtime.storage.sharded_mmap import ShardedMMapStorage
from offload_runtime.types import HostBuffer


class TestInMemoryStorage:
    def test_get_returns_host_buffer(self) -> None:
        storage = InMemoryStorage({0: b"abcd", 1: b"efgh"})
        buf = storage.get(0)
        assert isinstance(buf, HostBuffer)
        assert bytes(buf.view) == b"abcd"
        assert buf.pinned is False

    def test_get_different_layers(self) -> None:
        storage = InMemoryStorage({0: b"aaaa", 1: b"bbbb"})
        assert bytes(storage.get(0).view) == b"aaaa"
        assert bytes(storage.get(1).view) == b"bbbb"

    def test_get_missing_layer_raises(self) -> None:
        storage = InMemoryStorage({0: b"x"})
        with pytest.raises(KeyError):
            storage.get(99)

    def test_request_wait_release_no_op(self) -> None:
        storage = InMemoryStorage({0: b"x"})
        storage.request(0)
        storage.wait(0)
        storage.release(0)


class TestShardedMMapStorage:
    def _create_shard(self, tmp: Path, shard_name: str, data: bytes) -> None:
        (tmp / shard_name).write_bytes(data)

    def _create_index(self, tmp: Path, layers: list[dict]) -> Path:
        index_path = tmp / "index.json"
        index_path.write_text(json.dumps({"layers": layers}), encoding="utf-8")
        return index_path

    def test_basic_get(self, tmp_path: Path) -> None:
        shard_data = b"A" * 64 + b"B" * 64
        self._create_shard(tmp_path, "w.bin", shard_data)
        index = self._create_index(tmp_path, [
            {"layer_id": 0, "path": "w.bin", "offset": 0, "nbytes": 64},
            {"layer_id": 1, "path": "w.bin", "offset": 64, "nbytes": 64},
        ])

        storage = ShardedMMapStorage(root=tmp_path, index_path=index)
        buf0 = storage.get(0)
        assert bytes(buf0.view) == b"A" * 64
        buf1 = storage.get(1)
        assert bytes(buf1.view) == b"B" * 64
        storage.close()

    def test_multiple_shards(self, tmp_path: Path) -> None:
        self._create_shard(tmp_path, "shard0.bin", b"X" * 32)
        self._create_shard(tmp_path, "shard1.bin", b"Y" * 48)
        index = self._create_index(tmp_path, [
            {"layer_id": 0, "path": "shard0.bin", "offset": 0, "nbytes": 32},
            {"layer_id": 1, "path": "shard1.bin", "offset": 0, "nbytes": 48},
        ])

        storage = ShardedMMapStorage(root=tmp_path, index_path=index)
        assert bytes(storage.get(0).view) == b"X" * 32
        assert bytes(storage.get(1).view) == b"Y" * 48
        storage.close()

    def test_missing_layer_raises(self, tmp_path: Path) -> None:
        self._create_shard(tmp_path, "w.bin", b"Z" * 16)
        index = self._create_index(tmp_path, [
            {"layer_id": 0, "path": "w.bin", "offset": 0, "nbytes": 16},
        ])
        storage = ShardedMMapStorage(root=tmp_path, index_path=index)
        with pytest.raises(KeyError):
            storage.get(99)
        storage.close()

    def test_close_clears_resources(self, tmp_path: Path) -> None:
        self._create_shard(tmp_path, "w.bin", b"D" * 16)
        index = self._create_index(tmp_path, [
            {"layer_id": 0, "path": "w.bin", "offset": 0, "nbytes": 16},
        ])
        storage = ShardedMMapStorage(root=tmp_path, index_path=index)
        _ = storage.get(0)
        storage.close()
        assert len(storage._maps) == 0
        assert len(storage._files) == 0

    def test_context_manager(self, tmp_path: Path) -> None:
        self._create_shard(tmp_path, "w.bin", b"E" * 16)
        index = self._create_index(tmp_path, [
            {"layer_id": 0, "path": "w.bin", "offset": 0, "nbytes": 16},
        ])
        with ShardedMMapStorage(root=tmp_path, index_path=index) as storage:
            assert bytes(storage.get(0).view) == b"E" * 16
        assert len(storage._maps) == 0


class TestShardedMMapStorageAsync:
    def _create_shard(self, tmp: Path, shard_name: str, data: bytes) -> None:
        (tmp / shard_name).write_bytes(data)

    def _create_index(self, tmp: Path, layers: list[dict]) -> Path:
        index_path = tmp / "index.json"
        index_path.write_text(json.dumps({"layers": layers}), encoding="utf-8")
        return index_path

    def test_request_wait_get_lifecycle(self, tmp_path: Path) -> None:
        self._create_shard(tmp_path, "w.bin", b"A" * 32 + b"B" * 32)
        index = self._create_index(tmp_path, [
            {"layer_id": 0, "path": "w.bin", "offset": 0, "nbytes": 32},
            {"layer_id": 1, "path": "w.bin", "offset": 32, "nbytes": 32},
        ])
        with ShardedMMapStorage(root=tmp_path, index_path=index) as storage:
            storage.request(0)
            storage.request(1)
            storage.wait(0)
            storage.wait(1)
            assert bytes(storage.get(0).view) == b"A" * 32
            assert bytes(storage.get(1).view) == b"B" * 32

    def test_get_without_request_still_works(self, tmp_path: Path) -> None:
        self._create_shard(tmp_path, "w.bin", b"X" * 16)
        index = self._create_index(tmp_path, [
            {"layer_id": 0, "path": "w.bin", "offset": 0, "nbytes": 16},
        ])
        with ShardedMMapStorage(root=tmp_path, index_path=index) as storage:
            assert bytes(storage.get(0).view) == b"X" * 16

    def test_double_request_is_idempotent(self, tmp_path: Path) -> None:
        self._create_shard(tmp_path, "w.bin", b"Y" * 16)
        index = self._create_index(tmp_path, [
            {"layer_id": 0, "path": "w.bin", "offset": 0, "nbytes": 16},
        ])
        with ShardedMMapStorage(root=tmp_path, index_path=index) as storage:
            storage.request(0)
            storage.request(0)  # should not raise or create duplicate
            storage.wait(0)
            assert bytes(storage.get(0).view) == b"Y" * 16

    def test_release_discards_and_allows_reload(self, tmp_path: Path) -> None:
        self._create_shard(tmp_path, "w.bin", b"Z" * 16)
        index = self._create_index(tmp_path, [
            {"layer_id": 0, "path": "w.bin", "offset": 0, "nbytes": 16},
        ])
        with ShardedMMapStorage(root=tmp_path, index_path=index) as storage:
            storage.request(0)
            storage.wait(0)
            storage.release(0)
            # Should still work via synchronous fallback
            assert bytes(storage.get(0).view) == b"Z" * 16

    def test_close_cleans_up_futures(self, tmp_path: Path) -> None:
        self._create_shard(tmp_path, "w.bin", b"Q" * 16)
        index = self._create_index(tmp_path, [
            {"layer_id": 0, "path": "w.bin", "offset": 0, "nbytes": 16},
        ])
        storage = ShardedMMapStorage(root=tmp_path, index_path=index)
        storage.request(0)
        storage.close()
        assert len(storage._futures) == 0
        assert len(storage._ready) == 0
        assert len(storage._maps) == 0

    def test_prefetch_multiple_layers(self, tmp_path: Path) -> None:
        data = b"".join(bytes([i]) * 64 for i in range(5))
        self._create_shard(tmp_path, "w.bin", data)
        index = self._create_index(tmp_path, [
            {"layer_id": i, "path": "w.bin", "offset": i * 64, "nbytes": 64}
            for i in range(5)
        ])
        with ShardedMMapStorage(root=tmp_path, index_path=index, max_workers=2) as storage:
            for i in range(5):
                storage.request(i)
            for i in range(5):
                storage.wait(i)
            for i in range(5):
                buf = storage.get(i)
                assert bytes(buf.view) == bytes([i]) * 64
