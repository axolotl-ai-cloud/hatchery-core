# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Local filesystem object store."""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import shutil
from pathlib import Path
from typing import BinaryIO, Optional


class LocalObjectStore:
    """Filesystem-backed object store.

    Keys are mapped to paths relative to ``root``. Parent directories are
    created on demand. Writes use a ``.tmp`` file and atomic ``os.replace``
    so concurrent readers never see a partial object.
    """

    def __init__(self, root: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Block path traversal: forbid absolute paths and '..' segments.
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"Invalid key: {key!r}")
        p = self.root / key
        return p

    async def put(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict[str, str]] = None,
    ) -> None:
        path = self._path(key)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)

        await asyncio.to_thread(_write)

    async def get(self, key: str) -> bytes:
        path = self._path(key)

        def _read() -> bytes:
            if not path.exists():
                raise KeyError(key)
            with open(path, "rb") as f:
                return f.read()

        return await asyncio.to_thread(_read)

    async def get_stream(self, key: str) -> BinaryIO:
        return io.BytesIO(await self.get(key))

    async def delete(self, key: str) -> None:
        path = self._path(key)

        def _delete() -> None:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()

        await asyncio.to_thread(_delete)

    async def exists(self, key: str) -> bool:
        return await asyncio.to_thread(self._path(key).exists)

    async def list_keys(self, prefix: str) -> list[str]:
        def _list() -> list[str]:
            base = self._path(prefix) if prefix else self.root
            if base.is_file():
                return [prefix]
            results: list[str] = []
            # Walk either the prefix dir or the root if prefix matches nothing.
            walk_root = base if base.exists() else self.root
            for dirpath, _, files in os.walk(walk_root):
                for fname in files:
                    full = Path(dirpath) / fname
                    rel = full.relative_to(self.root).as_posix()
                    if rel.startswith(prefix):
                        results.append(rel)
            return sorted(results)

        return await asyncio.to_thread(_list)

    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        return f"file://{self._path(key)}"

    async def copy(self, src_key: str, dst_key: str) -> None:
        src = self._path(src_key)
        dst = self._path(dst_key)

        def _copy() -> None:
            if not src.exists():
                raise KeyError(src_key)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

        await asyncio.to_thread(_copy)
