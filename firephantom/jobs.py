"""Persistent background jobs: each one bakes and renders a list of sequences."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid

from firephantom.db import connect, ensure_schema
from firephantom.images import encode_png
from firephantom.runs import STAGES, total_combos
from firephantom.runtime import memory_status
from firephantom.serialize import plain
from firephantom.validation import expand_job_items


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_results(results: list[dict], job_id: str, include_details: bool) -> list[dict]:
    formatted = []
    for source in results:
        result = dict(source)
        preview_path = result.pop("preview_path", None)
        if preview_path:
            result["preview_url"] = f"/api/v1/jobs/{job_id}/results/{result['index']}/preview"
        if result.get("sequence_id") and result.get("video"):
            result["video_url"] = f"/api/v1/sequences/{result['sequence_id']}/files/{result['video']}"
        if not include_details:
            result.pop("contributors", None)
            result.pop("tool_runs", None)
            result.pop("samples", None)
        formatted.append(result)
    return formatted


def _encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _request_of(payload: str) -> list[dict] | None:
    """The workbench rows a job was submitted with, to reopen its scene."""
    try:
        body = json.loads(payload)
    except (TypeError, ValueError):
        return None
    return body.get("items") if isinstance(body, dict) else None


class JobManager:
    """Run sequence jobs in a worker and persist progress in PostgreSQL.

    ``total`` and ``completed`` count combinations (one result each);
    ``kept`` and ``dropped`` count the samples those combinations produced.
    """

    def __init__(self, root: Path | str, max_workers: int = 1):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="firephantom-job")
        self._init_db()
        self._recover()

    @contextmanager
    def _db(self):
        with connect() as connection:
            yield connection

    def _init_db(self) -> None:
        ensure_schema()
        with self._db() as db:
            db.execute(
                """UPDATE jobs SET state='queued',error=NULL,updated_at=?
                   WHERE state='running'""",
                (_now(),),
            )

    def _recover(self) -> None:
        with self._db() as db:
            ids = [row[0] for row in db.execute("SELECT id FROM jobs WHERE state='queued'")]
        for job_id in ids:
            self._executor.submit(self._run, job_id)

    def _prepare(self, items: list[dict]) -> list[dict]:
        from firephantom.scenes import resolve_scene_items

        return expand_job_items(resolve_scene_items(self.root, items))

    def submit(self, items: list[dict], staged: bool = True) -> dict:
        """Queue a job. ``staged`` is accepted for old clients; every job runs the same way."""
        del staged
        sequences = self._prepare(items)
        job_id = uuid.uuid4().hex
        timestamp = _now()
        payload = _encode({"items": items, "sequences": sequences, "combo_index": None})
        with self._lock, self._db() as db:
            db.execute(
                "INSERT INTO jobs(id,state,total,created_at,updated_at,payload) VALUES(?,?,?,?,?,?)",
                (job_id, "queued", total_combos(sequences), timestamp, timestamp, payload),
            )
        self._executor.submit(self._run, job_id)
        return self.get(job_id)

    def continue_job(self, job_id: str, from_stage: str, items: list[dict], combo_index: int | None = None) -> dict | None:
        """Run the job again with (possibly edited) items, or only one combination.

        Bakes are cached by their inputs, so nothing upstream of a change is
        recomputed; ``from_stage`` is accepted for old clients.
        """
        # Old clients name stages that no longer exist. SD left generation.
        if from_stage in {"physics", "diffusion"}:
            from_stage = "engine"
        if from_stage not in STAGES and from_stage != "reference":
            raise ValueError(f"未知阶段 {from_stage}")
        sequences = self._prepare(items)
        total = total_combos(sequences)
        if combo_index is not None and not 0 <= int(combo_index) < total:
            raise ValueError("combo_index is out of range")
        payload = _encode({
            "items": items,
            "sequences": sequences,
            "combo_index": None if combo_index is None else int(combo_index),
        })
        with self._lock, self._db() as db:
            row = db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return None
            if row["state"] in {"queued", "running"}:
                raise ValueError("作业还在跑")
            if combo_index is None:
                db.execute("DELETE FROM job_results WHERE job_id=?", (job_id,))
                completed, kept, dropped = 0, 0, 0
            else:
                db.execute(
                    "DELETE FROM job_results WHERE job_id=? AND result_index=?",
                    (job_id, int(combo_index)),
                )
                completed, kept, dropped = self._tally_unlocked(db, job_id)
            db.execute(
                """UPDATE jobs
                   SET state='queued', total=?, completed=?, kept=?, dropped=?, error=NULL,
                       cancel_requested=0, payload=?, updated_at=?
                   WHERE id=?""",
                (total, completed, kept, dropped, payload, _now(), job_id),
            )
        self._executor.submit(self._run, job_id)
        return self.get(job_id)

    def _update(self, job_id: str, **values) -> None:
        assignments = ",".join(f"{key}=?" for key in values)
        params = [values[key] for key in values] + [_now(), job_id]
        with self._lock, self._db() as db:
            db.execute(f"UPDATE jobs SET {assignments},updated_at=? WHERE id=?", params)

    def _save_result(self, job_id: str, result: dict) -> None:
        with self._lock, self._db() as db:
            db.execute(
                """INSERT INTO job_results(job_id,result_index,result) VALUES(?,?,?)
                   ON CONFLICT (job_id, result_index) DO UPDATE SET result=EXCLUDED.result""",
                (job_id, int(result["index"]), _encode(result)),
            )
            completed, kept, dropped = self._tally_unlocked(db, job_id)
            db.execute(
                "UPDATE jobs SET completed=?,kept=?,dropped=?,updated_at=? WHERE id=?",
                (completed, kept, dropped, _now(), job_id),
            )

    def _item_result(self, job_id: str, index: int, result: dict) -> dict:
        if result.get("state") == "failed":
            return {"index": index, "state": "failed", "error": result.get("error")}
        preview_dir = self.root / "job_previews" / job_id
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"{index:06d}.png"
        preview_path.write_bytes(encode_png(result["preview"]))
        return {
            "index": index,
            "state": "succeeded",
            "sequence_index": result["sequence_index"],
            "sequence_id": result["sequence_id"],
            "scene": result["scene"],
            "weather": result["weather"],
            "viewpoint": result["viewpoint"],
            "frames": result["frames"],
            "kept": result["kept"],
            "dropped": result["dropped"],
            "samples": plain(result["samples"]),
            "video": result["video"],
            "preview_path": str(preview_path.relative_to(self.root)),
            "contributors": plain(result["contributors"]),
        }

    def _run(self, job_id: str) -> None:
        from firephantom.runs import execute

        with self._db() as db:
            row = db.execute("SELECT payload,state FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None or row["state"] == "cancelled":
            return
        payload = json.loads(row["payload"])
        if not isinstance(payload, dict) or "sequences" not in payload:
            self._update(job_id, state="failed", error="作业来自旧版本，请重新提交")
            return
        self._update(job_id, state="running")
        if self._is_cancel_requested(job_id):
            self._update(job_id, state="cancelled", error="cancelled by user")
            return
        halt: dict = {"reason": None}

        def stop() -> bool:
            if self._is_cancel_requested(job_id):
                halt["reason"] = "cancelled"
                return True
            status = memory_status()
            if status.get("state") == "critical":
                halt["reason"] = "memory"
                halt["status"] = status
                return True
            return False

        def on_combo(index: int, result: dict) -> None:
            self._save_result(job_id, self._item_result(job_id, index, result))

        try:
            outcome = execute(self.root, job_id, payload["sequences"], payload.get("combo_index"), stop, on_combo)
        except Exception as exc:
            self._update(job_id, state="failed", error=f"{type(exc).__name__}: {exc}")
            return
        results = self._stored_map(job_id)
        failed = [item for item in results.values() if item.get("state") == "failed"]
        succeeded = [item for item in results.values() if item.get("state") == "succeeded"]
        if halt["reason"] == "memory":
            status = halt.get("status") or {}
            available = status.get("available_gib")
            minimum = status.get("minimum_available_gib")
            state, error = "failed", (
                "memory guard: refusing to start the next combination; "
                f"available {available if available is not None else 'unknown'} GiB, "
                f"minimum {minimum:g} GiB"
            )
        elif halt["reason"] == "cancelled" or outcome == "cancelled":
            state, error = "cancelled", "cancelled by user"
        else:
            state = "succeeded" if not failed else "partial" if succeeded else "failed"
            error = "; ".join(sorted({str(item.get("error")) for item in failed})) or None
        self._update(job_id, state=state, error=error)

    def _stored_map(self, job_id: str) -> dict:
        with self._db() as db:
            rows = db.execute(
                "SELECT result FROM job_results WHERE job_id=?",
                (job_id,),
            ).fetchall()
        return {int(json.loads(row["result"])["index"]): json.loads(row["result"]) for row in rows}

    def _tally_unlocked(self, db, job_id: str) -> tuple[int, int, int]:
        rows = db.execute("SELECT result FROM job_results WHERE job_id=?", (job_id,)).fetchall()
        items = [json.loads(row["result"]) for row in rows]
        kept = sum(int(item.get("kept") or 0) for item in items if item.get("state") == "succeeded")
        dropped = sum(int(item.get("dropped") or 0) for item in items if item.get("state") == "succeeded")
        return len(items), kept, dropped

    def _is_cancel_requested(self, job_id: str) -> bool:
        with self._db() as db:
            row = db.execute(
                "SELECT cancel_requested,state FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return bool(row and (row["cancel_requested"] or row["state"] == "cancelled"))

    def cancel(self, job_id: str) -> dict | None:
        """Request cancellation of a queued/running job.

        A running pipeline item is allowed to finish because Blender
        is an external process. The worker stops before the next item and
        persists the completed prefix.
        """
        with self._lock, self._db() as db:
            row = db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return None
            if row["state"] in {"queued", "running"}:
                if row["state"] == "queued":
                    db.execute(
                        "UPDATE jobs SET cancel_requested=1,state='cancelled',error=?,updated_at=? WHERE id=?",
                        ("cancelled by user", _now(), job_id),
                    )
                else:
                    db.execute(
                        "UPDATE jobs SET cancel_requested=1,error=?,updated_at=? WHERE id=?",
                        ("cancellation requested", _now(), job_id),
                    )
        return self.get(job_id)

    def get(
        self,
        job_id: str,
        include_details: bool = True,
        result_offset: int = 0,
        result_limit: int = 50,
        include_results: bool = True,
        before_index: int | None = None,
    ) -> dict | None:
        with self._db() as db:
            row = db.execute(
                """SELECT id,state,total,completed,kept,dropped,created_at,updated_at,error,payload
                   FROM jobs WHERE id=?""",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            total_results = db.execute(
                "SELECT COUNT(*) FROM job_results WHERE job_id=?", (job_id,)
            ).fetchone()[0]
            offset = max(0, int(result_offset))
            limit = max(1, min(int(result_limit), 100))
            if include_results:
                if before_index is None:
                    rows = db.execute(
                        """SELECT result FROM job_results WHERE job_id=?
                           ORDER BY result_index DESC LIMIT ? OFFSET ?""",
                        (job_id, limit, offset),
                    ).fetchall()
                else:
                    rows = db.execute(
                        """SELECT result FROM job_results WHERE job_id=? AND result_index<?
                           ORDER BY result_index DESC LIMIT ?""",
                        (job_id, max(0, int(before_index)), limit),
                    ).fetchall()
                page = [json.loads(item["result"]) for item in reversed(rows)]
            else:
                page = []
            if before_index is not None:
                has_more = bool(
                    page
                    and db.execute(
                        "SELECT 1 FROM job_results WHERE job_id=? AND result_index<? LIMIT 1",
                        (job_id, page[0]["index"]),
                    ).fetchone()
                )
            else:
                has_more = offset + len(page) < total_results
            next_before = page[0]["index"] if has_more and page else None
        return {
            "job_id": row["id"],
            "state": row["state"],
            "total": row["total"],
            "completed": row["completed"],
            "kept": row["kept"],
            "dropped": row["dropped"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "total_results": total_results,
            "result_offset": offset,
            "result_limit": limit,
            "before_index": before_index,
            "next_before": next_before,
            "has_more_results": has_more,
            "results": _format_results(page, job_id, include_details),
            "error": row["error"],
            "request": _request_of(row["payload"]),
        }

    def get_result(self, job_id: str, result_index: int, include_details: bool = True) -> dict | None:
        with self._db() as db:
            row = db.execute(
                "SELECT result FROM job_results WHERE job_id=? AND result_index=?",
                (job_id, result_index),
            ).fetchone()
        if row is None:
            return None
        source = json.loads(row["result"])
        result = _format_results([source], job_id, include_details)[0]
        return {"job_id": job_id, **result}

    def get_preview_path(self, job_id: str, result_index: int) -> Path | None:
        if result_index < 0:
            return None
        with self._db() as db:
            row = db.execute(
                "SELECT result FROM job_results WHERE job_id=? AND result_index=?",
                (job_id, result_index),
            ).fetchone()
        if row is None:
            return None
        result = json.loads(row["result"])
        if result is None or result.get("state") != "succeeded":
            return None
        relative = result.get("preview_path")
        if not relative:
            return None
        root = self.root.resolve()
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            return None
        return path if path.is_file() else None

    def list(self, limit: int = 100) -> list[dict]:
        with self._db() as db:
            rows = db.execute(
                """SELECT id,state,total,completed,kept,dropped,created_at,updated_at,error,payload
                   FROM jobs ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {
                "job_id": row["id"],
                "state": row["state"],
                "total": row["total"],
                "completed": row["completed"],
                "kept": row["kept"],
                "dropped": row["dropped"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "total_results": row["completed"],
                "result_offset": 0,
                "result_limit": 50,
                "before_index": None,
                "next_before": None,
                "has_more_results": False,
                "results": [],
                "error": row["error"],
                "scene_ids": sorted({item.get("scene_id") for item in _request_of(row["payload"]) or [] if item.get("scene_id")}),
                "scenario": next((item.get("scenario") for item in _request_of(row["payload"]) or []), None),
            }
            for row in rows
        ]

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)
