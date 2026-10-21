"""Local DVC dataset store: add, conflict rejection, and rollback.

The DVC cache is the byte store. Each version keeps a ``.pointer`` copy of the
``workspace.dvc`` file that ``dvc add`` wrote, and a browsable snapshot under
``versions/``. Without encryption, sample images are hard-linked into the
workspace and the snapshot instead of copied. Rollback copies the pointer back
and runs ``dvc checkout``. There is no remote.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import zipfile
from contextlib import contextmanager
from pathlib import Path

from datetime import datetime, timezone

from firephantom.db import connect, ensure_schema
from firephantom.security import decrypt, enabled as encryption_enabled, encrypt
from firephantom.versioning.desensitize import desensitize

Source = bytes | Path


def _read(source: Source) -> bytes:
    return source.read_bytes() if isinstance(source, Path) else source


def _link_or_copy(source: str, destination: str) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


class VersionConflict(Exception):
    def __init__(self, version_id: str):
        self.version_id = version_id
        super().__init__(f"version identity {version_id} already exists with different content")


_STORE_LOCK = threading.RLock()


def _digest(files: dict[str, Source], metadata: dict) -> str:
    hasher = hashlib.sha256()
    for path in sorted(files):
        if path == "metadata.json":
            continue
        hasher.update(path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_read(files[path]))
        hasher.update(b"\0")
    body = {key: value for key, value in metadata.items() if key != "版本号"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    hasher.update(encoded.encode("utf-8"))
    return hasher.hexdigest()


def _safe_name(version_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", version_id).strip("._") or "id"
    suffix = hashlib.sha256(version_id.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:80]}_{suffix}"


class DatasetStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace = self.root / "workspace"
        self.refs = self.root / "refs"
        self.refs.mkdir(parents=True, exist_ok=True)
        self._ensure_dvc()
        self._ensure_catalog()

    def _dvc(self, *args: str) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["DVC_NO_ANALYTICS"] = "1"
        env["DVC_IGNORE_ISATTY"] = "1"
        proc = subprocess.run(
            [sys.executable, "-m", "dvc", *args],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"dvc {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc

    def _ensure_dvc(self) -> None:
        if not (self.root / ".dvc" / "config").exists():
            self._dvc("init", "--no-scm")
            self._dvc("config", "core.analytics", "false")

    @contextmanager
    def _catalog(self):
        with connect() as connection:
            yield connection

    def _ensure_catalog(self) -> None:
        ensure_schema()

    def _read_index(self) -> dict:
        return {"head": self.head(), "versions": self.list_versions()}

    def head(self) -> str | None:
        with self._catalog() as db:
            row = db.execute("SELECT state_value FROM store_state WHERE state_key='head'").fetchone()
        return row[0] if row else None

    def next_revision(self) -> int:
        with self._catalog() as db:
            return int(db.execute("SELECT COALESCE(MAX(revision), 0) + 1 FROM versions").fetchone()[0])

    def version_count(self, dataset_id: str | None = None) -> int:
        with self._catalog() as db:
            if dataset_id is None:
                return int(db.execute("SELECT COUNT(*) FROM versions").fetchone()[0])
            return int(db.execute("SELECT COUNT(*) FROM versions WHERE dataset_id=?", (dataset_id,)).fetchone()[0])

    def list_versions(
        self,
        offset: int = 0,
        limit: int | None = None,
        newest_first: bool = False,
        dataset_id: str | None = None,
    ) -> list[dict]:
        offset = max(0, int(offset))
        order = "DESC" if newest_first else "ASC"
        where = "" if dataset_id is None else "WHERE dataset_id=? "
        query = f"SELECT record FROM versions {where}ORDER BY revision {order}"
        parameters: tuple = () if dataset_id is None else (dataset_id,)
        if limit is not None:
            limit = max(1, int(limit))
            query += " LIMIT ? OFFSET ?"
            parameters = (*parameters, limit, offset)
        elif offset:
            query += " LIMIT 1000000000 OFFSET ?"
            parameters = (*parameters, offset)
        with self._catalog() as db:
            rows = db.execute(query, parameters).fetchall()
        return [json.loads(row["record"]) for row in rows]

    def get(self, version_id: str) -> dict | None:
        with self._catalog() as db:
            row = db.execute("SELECT record FROM versions WHERE version_id=?", (version_id,)).fetchone()
        return json.loads(row["record"]) if row else None

    def _pointer_path(self, version_id: str) -> Path:
        return self.refs / f"{_safe_name(version_id)}.pointer"

    def _write_workspace(self, files: dict[str, Source]) -> None:
        validated: list[tuple[Path, Source]] = []
        for relative, payload in files.items():
            rel = Path(relative)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"invalid dataset path {relative!r}")
            validated.append((rel, payload))
        workspace = self.workspace
        if workspace.is_symlink():
            workspace.unlink()
        elif workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)
        root = workspace.resolve()
        for rel, payload in validated:
            target = workspace / rel
            parent = target.parent.resolve()
            if parent != root and root not in parent.parents:
                raise ValueError(f"invalid dataset path {str(rel)!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, Path) and not encryption_enabled():
                _link_or_copy(str(payload), str(target))
            else:
                target.write_bytes(encrypt(_read(payload)))

    def _snapshot_path(self, version_id: str) -> Path:
        return self.root / "versions" / _safe_name(version_id)

    def _snapshot_workspace(self, version_id: str) -> None:
        destination = self._snapshot_path(version_id)
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.workspace, destination, copy_function=_link_or_copy)

    def list_files(self, version_id: str) -> list[dict]:
        """Paths and stored sizes of a version, without reading the files."""
        if self.get(version_id) is None:
            raise KeyError(f"unknown version {version_id}")
        source = self._snapshot_path(version_id)
        if not source.exists():
            raise KeyError(f"missing dataset snapshot for {version_id}")
        return [
            {"path": str(path.relative_to(source)), "size": path.stat().st_size}
            for path in sorted(source.rglob("*"))
            if path.is_file()
        ]

    def read_file(self, version_id: str, relative: str) -> bytes | None:
        if self.get(version_id) is None:
            raise KeyError(f"unknown version {version_id}")
        source = self._snapshot_path(version_id).resolve()
        target = (source / relative).resolve()
        if source not in target.parents or not target.is_file():
            return None
        return decrypt(target.read_bytes())

    def files(self, version_id: str) -> dict[str, bytes]:
        """Read a version snapshot, decrypting payloads when configured."""
        if self.get(version_id) is None:
            raise KeyError(f"unknown version {version_id}")
        source = self._snapshot_path(version_id)
        if not source.exists():
            raise KeyError(f"missing dataset snapshot for {version_id}")
        result: dict[str, bytes] = {}
        for path in sorted(source.rglob("*")):
            if path.is_file():
                result[str(path.relative_to(source))] = decrypt(path.read_bytes())
        return result

    def archive(self, version_id: str) -> bytes:
        """Return a decrypted ZIP export of one immutable dataset version."""
        files = self.files(version_id)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in files.items():
                archive.writestr(name, payload)
        return output.getvalue()

    def add(
        self,
        files: dict[str, Source],
        metadata: dict,
        version_id: str | None = None,
        *,
        parent_id: str | None = None,
        dataset_id: str | None = None,
    ) -> dict:
        """Add a version. An existing identity with different bytes is rejected.

        File values are bytes, or paths to hard-link when encryption is off.
        """
        with _STORE_LOCK:
            if not files:
                raise ValueError("refusing to add an empty file set")
            cleaned = desensitize(metadata)
            cleaned.pop("版本号", None)
            digest = _digest(files, cleaned)
            identity = digest if version_id is None else version_id
            existing = self.get(identity)
            if existing is not None:
                if existing["digest"] != digest:
                    raise VersionConflict(identity)
                return existing
            revision = self.next_revision()
            stored_meta = dict(cleaned)
            stored_meta["版本号"] = str(revision)
            payload = dict(files)
            payload.pop("metadata.json", None)
            payload["metadata.json"] = json.dumps(
                stored_meta, ensure_ascii=False, indent=2, sort_keys=True
            ).encode("utf-8")
            created_at = datetime.now(timezone.utc).isoformat()
            self._write_workspace(payload)
            self._dvc("add", "workspace")
            pointer = (self.root / "workspace.dvc").read_text(encoding="utf-8")
            self._pointer_path(identity).write_text(pointer, encoding="utf-8")
            self._snapshot_workspace(identity)
            record = {
                "version_id": identity,
                "版本号": str(revision),
                "revision": revision,
                "digest": digest,
                "metadata": stored_meta,
                "head_path": "workspace",
                "file_count": len(payload),
                "parent_id": parent_id,
                "dataset_id": dataset_id,
                "created_at": created_at,
            }
            with self._catalog() as db:
                db.execute(
                    """INSERT INTO versions(version_id,revision,digest,record,parent_id,dataset_id,created_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        identity,
                        revision,
                        digest,
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                        parent_id,
                        dataset_id,
                        created_at,
                    ),
                )
                db.execute("UPDATE store_state SET state_value=? WHERE state_key='head'", (identity,))
            return record

    def rollback(self, version_id: str) -> dict:
        """Restore a previous version's files and metadata via ``dvc checkout``."""
        with _STORE_LOCK:
            record = self.get(version_id)
            if record is None:
                raise KeyError(f"unknown version {version_id}")
            pointer_path = self._pointer_path(version_id)
            if not pointer_path.exists():
                raise KeyError(f"missing dvc pointer for {version_id}")
            (self.root / "workspace.dvc").write_text(pointer_path.read_text(encoding="utf-8"), encoding="utf-8")
            self._dvc("checkout", "--force")
            with self._catalog() as db:
                db.execute("UPDATE store_state SET state_value=? WHERE state_key='head'", (version_id,))
            return record
