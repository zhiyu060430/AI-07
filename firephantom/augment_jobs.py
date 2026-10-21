"""Background SD augmentation of a dataset's reviewed samples.

One job is one dataset, a list of reviewed source samples and the restyle
options. Each source is one step; its variants are stored as reviewed
children as they finish, so a cancelled or failed job keeps what it made.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid

from firephantom.db import connect, ensure_schema


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AugmentJobs:
    def __init__(self, root: Path | str, sd_endpoint: str | None = None):
        self.root = Path(root)
        self.sd_endpoint = sd_endpoint
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="firephantom-sd")
        ensure_schema()
        with connect() as db:
            db.execute("UPDATE augment_jobs SET state='queued', updated_at=? WHERE state='running'", (_now(),))
            ids = [row[0] for row in db.execute("SELECT id FROM augment_jobs WHERE state='queued'")]
        for job_id in ids:
            self._executor.submit(self._run, job_id)

    def submit(self, dataset_id: str, sample_ids: list[str] | None, options: object) -> dict:
        from firephantom.synthesis.sd_augment import eligible_sources, normalize_options
        from firephantom.versioning.datasets import DatasetCatalog

        if DatasetCatalog(self.root).get(dataset_id) is None:
            raise KeyError(f"unknown dataset {dataset_id}")
        settings = normalize_options(options)
        sources = [record["sample_id"] for record in eligible_sources(self.root, dataset_id, sample_ids)]
        if not sources:
            raise ValueError("数据集里还没有可以增强的已审核样本")
        job_id = uuid.uuid4().hex
        timestamp = _now()
        payload = json.dumps({"sources": sources, "options": settings, "results": []}, ensure_ascii=False)
        with self._lock, connect() as db:
            db.execute(
                """INSERT INTO augment_jobs(id, dataset_id, state, total, payload, created_at, updated_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (job_id, dataset_id, "queued", len(sources), payload, timestamp, timestamp),
            )
        self._executor.submit(self._run, job_id)
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        with connect() as db:
            row = db.execute("SELECT * FROM augment_jobs WHERE id=?", (job_id,)).fetchone()
        return None if row is None else _public(row)

    def list(self, dataset_id: str, limit: int = 20) -> list[dict]:
        with connect() as db:
            rows = db.execute(
                "SELECT * FROM augment_jobs WHERE dataset_id=? ORDER BY created_at DESC LIMIT ?",
                (dataset_id, int(limit)),
            ).fetchall()
        return [_public(row) for row in rows]

    def cancel(self, job_id: str) -> dict | None:
        with self._lock, connect() as db:
            row = db.execute("SELECT state FROM augment_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return None
            if row["state"] == "queued":
                db.execute(
                    "UPDATE augment_jobs SET state='cancelled', cancel_requested=1, updated_at=? WHERE id=?",
                    (_now(), job_id),
                )
            elif row["state"] == "running":
                db.execute("UPDATE augment_jobs SET cancel_requested=1, updated_at=? WHERE id=?", (_now(), job_id))
        return self.get(job_id)

    def _cancelled(self, job_id: str) -> bool:
        with connect() as db:
            row = db.execute("SELECT cancel_requested, state FROM augment_jobs WHERE id=?", (job_id,)).fetchone()
        return bool(row is None or row["cancel_requested"] or row["state"] == "cancelled")

    def _update(self, job_id: str, **values) -> None:
        assignments = ",".join(f"{key}=?" for key in values)
        with self._lock, connect() as db:
            db.execute(
                f"UPDATE augment_jobs SET {assignments}, updated_at=? WHERE id=?",
                [*values.values(), _now(), job_id],
            )

    def _run(self, job_id: str) -> None:
        from firephantom.synthesis.sd_augment import augment_source
        from firephantom.versioning.samples import SampleCatalog

        with connect() as db:
            row = db.execute("SELECT * FROM augment_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None or row["state"] == "cancelled":
            return
        payload = json.loads(row["payload"])
        done = {item["source"] for item in payload["results"]}
        self._update(job_id, state="running")
        catalog = SampleCatalog(self.root)
        try:
            for source_id in payload["sources"]:
                if source_id in done:
                    continue
                if self._cancelled(job_id):
                    self._update(job_id, state="cancelled", error="cancelled by user")
                    return
                source = catalog.get(source_id)
                if source is None or source["status"] != "reviewed":
                    outcomes = [{"sample_id": None, "state": "failed", "reason": "source is no longer reviewed"}]
                else:
                    outcomes = augment_source(self.root, row["dataset_id"], source, payload["options"], endpoint=self.sd_endpoint)
                payload["results"].append({"source": source_id, "outcomes": outcomes})
                created = sum(item["state"] in {"created", "exists"} for result in payload["results"] for item in result["outcomes"])
                failed = sum(item["state"] == "failed" for result in payload["results"] for item in result["outcomes"])
                self._update(
                    job_id,
                    completed=len(payload["results"]),
                    created=created,
                    failed=failed,
                    payload=json.dumps(payload, ensure_ascii=False),
                )
        except Exception as exc:
            self._update(job_id, state="failed", error=f"{type(exc).__name__}: {exc}")
            return
        reasons = sorted({
            str(item.get("reason"))
            for result in payload["results"]
            for item in result["outcomes"]
            if item["state"] == "failed"
        })
        made = any(item["state"] != "failed" for result in payload["results"] for item in result["outcomes"])
        state = "succeeded" if not reasons else "partial" if made else "failed"
        self._update(job_id, state=state, error="; ".join(reasons)[:2000] or None)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)


def _public(row) -> dict:
    payload = json.loads(row["payload"])
    return {
        "job_id": row["id"],
        "dataset_id": row["dataset_id"],
        "state": row["state"],
        "total": int(row["total"]),
        "completed": int(row["completed"]),
        "created": int(row["created"]),
        "failed": int(row["failed"]),
        "options": payload.get("options"),
        "results": payload.get("results", []),
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
