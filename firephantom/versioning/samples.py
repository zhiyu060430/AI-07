"""Sample catalog written before a dataset version is published.

Generation records scene parameters, the random seed, the generator version,
and a stable sample id. Automatic filtering and human review update the
sample status. DVC is not involved until a dataset is published.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import os
import json
import threading
from pathlib import Path

from firephantom.annotation.classes import canonical
from firephantom.db import connect, ensure_schema
from firephantom.evaluation import annotation_accuracy


_STATUSES = ("generated", "dropped", "pending_review", "reviewed", "rejected")
_STORE_LOCK = threading.RLock()


def derive_seed(scene_params: dict) -> int:
    """Derive a stable 32-bit seed from the canonical scene parameters."""
    raw = json.dumps(scene_params, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:4], "little")


def make_sample_id(
    scene_params: dict,
    seed: int,
    generator_version: str,
    generator_profile: dict,
    augmentations: list[str],
) -> str:
    """Identity of one generated sample. The same inputs always hash together."""
    payload = {
        "augmentations": list(augmentations),
        "generator_profile": generator_profile,
        "generator_version": generator_version,
        "scene_params": scene_params,
        "seed": int(seed),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


SPLITS = ("train", "val", "test")


def split_for(group: str) -> str:
    """80/10/10 split from a stable hash of the group id."""
    bucket = int.from_bytes(hashlib.sha256(str(group).encode("utf-8")).digest()[:4], "little") % 10
    return "test" if bucket == 0 else "val" if bucket == 1 else "train"


def annotation_error_rate(predictions: list[dict], ground_truth: list[dict]) -> float:
    """Return ``1 - micro F1`` of automatic boxes against reviewed boxes.

    An image with no boxes on either side does not count as an error.
    ``predictions`` and ``ground_truth`` are per-image object lists.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must contain the same number of images")
    if all(not predicted and not truth for predicted, truth in zip(predictions, ground_truth)):
        return 0.0
    metrics = annotation_accuracy(
        [{"objects": objects} for objects in predictions],
        [{"objects": objects} for objects in ground_truth],
    )
    return 1.0 - float(metrics["f1"])


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _load(value: str | None, default):
    if value is None:
        return default
    return json.loads(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SampleCatalog:
    """PostgreSQL index plus one directory of files per sample."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure()

    def sample_dir(self, sample_id: str) -> Path:
        return self.root / "samples" / sample_id

    @contextmanager
    def _db(self):
        with connect() as connection:
            yield connection

    def _ensure(self) -> None:
        ensure_schema()

    def _row_to_record(self, row) -> dict:
        return {
            "sample_id": row["sample_id"],
            "status": row["status"],
            "seed": int(row["seed"]),
            "generator_version": row["generator_version"],
            "generator_profile": _load(row["generator_profile"], {}),
            "scene_params": _load(row["scene_params"], {}),
            "quality": _load(row["quality"], None),
            "filter_decision": _load(row["filter_decision"], None),
            "auto_objects": _load(row["auto_objects"], []),
            "reviewed_objects": _load(row["reviewed_objects"], None),
            "label_studio_task_id": row["label_studio_task_id"],
            "annotation_error": None if row["annotation_error"] is None else float(row["annotation_error"]),
            "sequence_id": row["sequence_id"],
            "split": row["split"],
            "derived_from": row["derived_from"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def statuses(self, sample_ids: list[str]) -> dict[str, str]:
        if not sample_ids:
            return {}
        with self._db() as db:
            rows = db.execute(
                "SELECT sample_id, status FROM samples WHERE sample_id = ANY(?)",
                (list(sample_ids),),
            ).fetchall()
        return {row["sample_id"]: row["status"] for row in rows}

    def get(self, sample_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM samples WHERE sample_id=?", (sample_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def list_samples(
        self,
        status: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[dict], int]:
        if status is not None and status not in _STATUSES:
            raise ValueError(f"未知样本状态 {status!r}")
        offset = max(0, int(offset))
        limit = max(1, int(limit))
        where = ""
        parameters: tuple = ()
        if status is not None:
            where = "WHERE status=?"
            parameters = (status,)
        with self._db() as db:
            total = int(db.execute(f"SELECT COUNT(*) FROM samples {where}", parameters).fetchone()[0])
            rows = db.execute(
                f"SELECT * FROM samples {where} ORDER BY created_at DESC, sample_id LIMIT ? OFFSET ?",
                (*parameters, limit, offset),
            ).fetchall()
        return [self._row_to_record(row) for row in rows], total

    def summary(self) -> dict:
        counts = {status: 0 for status in _STATUSES}
        with self._db() as db:
            for row in db.execute("SELECT status, COUNT(*) AS n FROM samples GROUP BY status"):
                counts[row["status"]] = int(row["n"])
            ready = int(
                db.execute(
                    """SELECT COUNT(*) FROM samples
                       WHERE status='reviewed'
                       AND sample_id NOT IN (SELECT sample_id FROM version_samples)"""
                ).fetchone()[0]
            )
        return {"counts": counts, "ready_to_publish": ready}

    def unpublished_reviewed_ids(self) -> list[str]:
        with self._db() as db:
            rows = db.execute(
                """SELECT sample_id FROM samples
                   WHERE status='reviewed'
                   AND sample_id NOT IN (SELECT sample_id FROM version_samples)
                   ORDER BY sample_id"""
            ).fetchall()
        return [row["sample_id"] for row in rows]

    def write_files(self, sample_id: str, files: dict[str, bytes]) -> None:
        root = self.sample_dir(sample_id)
        root.mkdir(parents=True, exist_ok=True)
        resolved_root = root.resolve()
        for relative, payload in files.items():
            rel = Path(relative)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"invalid sample path {relative!r}")
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            parent = target.parent.resolve()
            if parent != resolved_root and resolved_root not in parent.parents:
                raise ValueError(f"invalid sample path {relative!r}")
            # Published versions hard-link these files; replace, never rewrite in place.
            temporary = target.with_name(f".{target.name}.tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, target)

    def read_file(self, sample_id: str, relative: str) -> bytes:
        if self.get(sample_id) is None:
            raise KeyError(f"unknown sample {sample_id}")
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"invalid sample path {relative!r}")
        root = self.sample_dir(sample_id).resolve()
        target = (root / rel).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"invalid sample path {relative!r}")
        if not target.is_file():
            raise KeyError(f"unknown sample file {relative}")
        return target.read_bytes()

    def save_generation(
        self,
        *,
        sample_id: str,
        seed: int,
        generator_version: str,
        generator_profile: dict,
        scene_params: dict,
        status: str,
        quality: dict,
        filter_decision: dict,
        auto_objects: list[dict],
        files: dict[str, bytes],
        label_studio_task_id: str | None = None,
        sequence_id: str | None = None,
        derived_from: str | None = None,
    ) -> dict:
        """Insert a generated sample or refresh one nobody has picked up yet.

        ``dropped`` means the quality checks failed; such a sample cannot be
        admitted to a dataset.
        """
        if status not in {"dropped", "generated"}:
            raise ValueError(f"generation cannot set sample status {status!r}")
        with _STORE_LOCK:
            existing = self.get(sample_id)
            timestamp = _now()
            if existing and existing["status"] in {"reviewed", "rejected", "pending_review"}:
                status = existing["status"]
                reviewed = existing["reviewed_objects"]
                error = existing["annotation_error"]
                task_id = existing["label_studio_task_id"]
                created_at = existing["created_at"]
            else:
                reviewed = None
                error = None
                task_id = label_studio_task_id
                created_at = existing["created_at"] if existing else timestamp
            self.write_files(sample_id, files)
            with self._db() as db:
                db.execute(
                    """INSERT INTO samples(
                        sample_id, status, seed, generator_version, generator_profile,
                        scene_params, quality, filter_decision, auto_objects, reviewed_objects,
                        label_studio_task_id, annotation_error, created_at, updated_at,
                        sequence_id, derived_from
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT (sample_id) DO UPDATE SET
                        status=EXCLUDED.status,
                        seed=EXCLUDED.seed,
                        generator_version=EXCLUDED.generator_version,
                        generator_profile=EXCLUDED.generator_profile,
                        scene_params=EXCLUDED.scene_params,
                        quality=EXCLUDED.quality,
                        filter_decision=EXCLUDED.filter_decision,
                        auto_objects=EXCLUDED.auto_objects,
                        reviewed_objects=EXCLUDED.reviewed_objects,
                        label_studio_task_id=EXCLUDED.label_studio_task_id,
                        annotation_error=EXCLUDED.annotation_error,
                        updated_at=EXCLUDED.updated_at,
                        sequence_id=EXCLUDED.sequence_id,
                        derived_from=EXCLUDED.derived_from
                    """,
                    (
                        sample_id,
                        status,
                        int(seed),
                        generator_version,
                        _dump(generator_profile),
                        _dump(scene_params),
                        _dump(quality),
                        _dump(filter_decision),
                        _dump(auto_objects),
                        _dump(reviewed),
                        task_id,
                        error,
                        created_at,
                        timestamp,
                        sequence_id,
                        derived_from,
                    ),
                )
        saved = self.get(sample_id)
        if saved is None:
            raise RuntimeError(f"sample {sample_id} was not stored")
        return saved

    def save_sd_derived(
        self,
        source: dict,
        *,
        sample_id: str,
        seed: int,
        generator_version: str,
        generator_profile: dict,
        scene_params: dict,
        quality: dict,
        files: dict[str, bytes],
    ) -> dict:
        """Store an SD restyle of a reviewed sample, already reviewed.

        SD only repaints pixels outside the flame, so the reviewed boxes and
        the annotation error carry over. The child keeps the source's
        sequence and split. A child that already exists is returned as is.
        """
        if source["status"] != "reviewed":
            raise ValueError(f"样本 {source['sample_id']} 还没有审核，不能做 SD 增强")
        with _STORE_LOCK:
            existing = self.get(sample_id)
            if existing is not None:
                return existing
            timestamp = _now()
            self.write_files(sample_id, files)
            with self._db() as db:
                db.execute(
                    """INSERT INTO samples(
                        sample_id, status, seed, generator_version, generator_profile,
                        scene_params, quality, filter_decision, auto_objects, reviewed_objects,
                        label_studio_task_id, annotation_error, created_at, updated_at,
                        sequence_id, split, derived_from
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT (sample_id) DO NOTHING""",
                    (
                        sample_id,
                        "reviewed",
                        int(seed),
                        generator_version,
                        _dump(generator_profile),
                        _dump(scene_params),
                        _dump(quality),
                        _dump(source["filter_decision"]),
                        _dump(source["auto_objects"]),
                        _dump(rebase_objects(source["reviewed_objects"] or [], sample_id)),
                        None,
                        source["annotation_error"],
                        timestamp,
                        timestamp,
                        source["sequence_id"],
                        source["split"],
                        source["sample_id"],
                    ),
                )
        saved = self.get(sample_id)
        if saved is None:
            raise RuntimeError(f"sample {sample_id} was not stored")
        return saved

    def mark_pending_review(self, sample_id: str, task_id: str | None) -> dict:
        """Record that this generated sample was sent to Label Studio."""
        with _STORE_LOCK:
            existing = self.get(sample_id)
            if existing is None:
                raise KeyError(f"unknown sample {sample_id}")
            if existing["status"] != "generated":
                raise ValueError(f"样本 {sample_id} 的状态是 {existing['status']}，不能送去标注")
            with self._db() as db:
                db.execute(
                    """UPDATE samples
                       SET status='pending_review', label_studio_task_id=?, updated_at=?
                       WHERE sample_id=?""",
                    (task_id, _now(), sample_id),
                )
        saved = self.get(sample_id)
        if saved is None:
            raise KeyError(f"unknown sample {sample_id}")
        return saved

    def is_published(self, sample_id: str) -> bool:
        with self._db() as db:
            row = db.execute("SELECT 1 FROM version_samples WHERE sample_id=? LIMIT 1", (sample_id,)).fetchone()
        return row is not None

    def record_review(
        self,
        sample_id: str,
        objects: list[dict],
        decision: str,
        task_id: str | None = None,
        *,
        rereview: bool = False,
    ) -> dict:
        """Store human-reviewed boxes and the resulting annotation error rate.

        Only a sample waiting for review can be decided. A decided sample can
        be decided again with ``rereview=True``, as long as no published
        version contains it; a published sample is frozen.
        """
        if decision not in {"reviewed", "rejected"}:
            raise ValueError("decision must be reviewed or rejected")
        with _STORE_LOCK:
            existing = self.get(sample_id)
            if existing is None:
                raise KeyError(f"unknown sample {sample_id}")
            allowed = {"pending_review"} | ({"reviewed", "rejected"} if rereview else set())
            if existing["status"] not in allowed:
                raise ValueError(f"样本 {sample_id} 的状态是 {existing['status']}，不能记录审核")
            if self.is_published(sample_id):
                raise ValueError(f"样本 {sample_id} 已在发布的版本里，不能再改审核")
            normalized = _normalize_objects(sample_id, objects, existing["auto_objects"])
            error = annotation_error_rate([existing["auto_objects"]], [normalized])
            with self._db() as db:
                db.execute(
                    """UPDATE samples
                       SET status=?, reviewed_objects=?, annotation_error=?,
                           label_studio_task_id=COALESCE(?, label_studio_task_id),
                           updated_at=?
                       WHERE sample_id=?""",
                    (
                        decision,
                        _dump(normalized),
                        error,
                        task_id,
                        _now(),
                        sample_id,
                    ),
                )
        saved = self.get(sample_id)
        if saved is None:
            raise KeyError(f"unknown sample {sample_id}")
        return saved

    def assign_splits(self, sample_ids: list[str]) -> dict[str, str]:
        """Split of each sample, fixed the first time it is asked for.

        Frames of one sequence are near duplicates, so the split follows the
        sequence; a sample without one is split by its own id.
        """
        splits: dict[str, str] = {}
        with _STORE_LOCK, self._db() as db:
            for sample_id in sample_ids:
                row = db.execute("SELECT split, sequence_id FROM samples WHERE sample_id=?", (sample_id,)).fetchone()
                if row is None:
                    raise KeyError(f"unknown sample {sample_id}")
                split = row["split"] or split_for(row["sequence_id"] or sample_id)
                if row["split"] is None:
                    db.execute("UPDATE samples SET split=? WHERE sample_id=?", (split, sample_id))
                splits[sample_id] = split
        return splits

    def link_version(self, version_id: str, sample_ids: list[str]) -> None:
        with _STORE_LOCK, self._db() as db:
            db.executemany(
                """INSERT INTO version_samples(version_id, sample_id) VALUES(?, ?)
                   ON CONFLICT DO NOTHING""",
                [(version_id, sample_id) for sample_id in sample_ids],
            )

    def record_evaluation(self, version_id: str, metrics: dict) -> dict:
        recorded_at = _now()
        stored = {
            "map50": float(metrics["map50"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "map50_95": None if metrics.get("map50_95") is None else float(metrics["map50_95"]),
            "recorded_at": recorded_at,
        }
        with _STORE_LOCK, self._db() as db:
            db.execute(
                "INSERT INTO version_evaluations(version_id, recorded_at, metrics) VALUES(?,?,?)",
                (version_id, recorded_at, _dump(stored)),
            )
        return stored

    def latest_evaluation(self, version_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute(
                """SELECT metrics FROM version_evaluations
                   WHERE version_id=? ORDER BY recorded_at DESC LIMIT 1""",
                (version_id,),
            ).fetchone()
        return json.loads(row["metrics"]) if row else None


def rebase_objects(objects: list[dict], sample_id: str) -> list[dict]:
    """Reviewed boxes of one sample, pointed at another sample's image."""
    image_file = f"samples/{sample_id}/image.png"
    return [{**obj, "image_file": image_file} for obj in objects]


def _normalize_objects(sample_id: str, objects: list[dict], auto_objects: list[dict]) -> list[dict]:
    fallback_width = int(auto_objects[0]["image_width"]) if auto_objects else 0
    fallback_height = int(auto_objects[0]["image_height"]) if auto_objects else 0
    image_file = f"samples/{sample_id}/image.png"
    normalized = []
    for obj in objects:
        if "bbox" not in obj:
            raise ValueError("reviewed object requires bbox")
        bbox = [float(value) for value in obj["bbox"]]
        if len(bbox) != 4:
            raise ValueError("bbox must contain x, y, width, height")
        width = int(obj.get("image_width") or fallback_width or 0)
        height = int(obj.get("image_height") or fallback_height or 0)
        if width <= 0 or height <= 0:
            raise ValueError("reviewed object requires image_width and image_height")
        normalized.append(
            {
                "category": canonical(obj.get("category") or "flame"),
                "bbox": bbox,
                "image_width": width,
                "image_height": height,
                "image_file": image_file,
                "keypoints": list(obj.get("keypoints") or []),
                "attributes": list(obj.get("attributes") or []),
            }
        )
    return normalized
