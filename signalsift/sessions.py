"""File-backed session store: one pickle (analysis) + one JSON (metadata) per session.

Session ids are validated against a strict pattern so a crafted id can never escape the
session directory. Pickles are only ever read from this directory, which the app itself
writes — never load session files from untrusted sources.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import uuid
from pathlib import Path
from typing import Any

_ID = re.compile(r"^[a-f0-9]{12}$")


class SessionNotFound(KeyError):
    pass


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    def _path(self, session_id: str, suffix: str) -> Path:
        if not _ID.fullmatch(session_id):
            raise SessionNotFound(session_id)
        return self.root / f"{session_id}{suffix}"

    def _atomic_write(self, path: Path, data: bytes) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def save(self, session_id: str, obj: Any, meta: dict[str, Any]) -> None:
        self._atomic_write(self._path(session_id, ".pkl"), pickle.dumps(obj))
        self._atomic_write(
            self._path(session_id, ".json"), json.dumps(meta, default=str, indent=2).encode()
        )

    def load(self, session_id: str) -> Any:
        path = self._path(session_id, ".pkl")
        if not path.exists():
            raise SessionNotFound(session_id)
        with path.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301 - trusted, app-written file

    def meta(self, session_id: str) -> dict[str, Any]:
        path = self._path(session_id, ".json")
        if not path.exists():
            raise SessionNotFound(session_id)
        data: dict[str, Any] = json.loads(path.read_text())
        return data

    def list(self) -> list[dict[str, Any]]:
        metas = []
        for path in self.root.glob("*.json"):
            try:
                metas.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(metas, key=lambda m: str(m.get("created_at", "")), reverse=True)

    def delete(self, session_id: str) -> None:
        found = False
        for suffix in (".pkl", ".json"):
            path = self._path(session_id, suffix)
            if path.exists():
                path.unlink()
                found = True
        if not found:
            raise SessionNotFound(session_id)
