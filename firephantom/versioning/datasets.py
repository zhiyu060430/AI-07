"""Named datasets that receive samples only after a person confirms them.

A dataset is a member list plus a head version. Publishing builds the next
version on top of the head; rolling back moves the head to an older version.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from firephantom.annotation.labelstudio import push_task
from firephantom.db import connect
from firephantom.versioning.samples import SampleCatalog


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatasetCatalog:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def create(self, name: str) -> dict:
        text = str(name or "").strip()
        if not text:
            raise ValueError("数据集需要一个名称")
        if len(text) > 80:
            raise ValueError("数据集名称不能超过 80 个字")
        dataset_id = uuid.uuid4().hex
        with connect() as db:
            db.execute(
                "INSERT INTO datasets(id, name, created_at) VALUES(?,?,?)",
                (dataset_id, text, _now()),
            )
        return self.get(dataset_id)

    def get(self, dataset_id: str) -> dict | None:
        with connect() as db:
            row = db.execute(
                "SELECT id, name, created_at, head_version_id FROM datasets WHERE id=?",
                (dataset_id,),
            ).fetchone()
        if row is None:
            return None
        return self._with_counts(dict(row))

    def list_all(self) -> list[dict]:
        with connect() as db:
            rows = db.execute(
                "SELECT id, name, created_at, head_version_id FROM datasets ORDER BY created_at DESC, id"
            ).fetchall()
        return [self._with_counts(dict(row)) for row in rows]

    def dataset_of(self, sample_id: str) -> str | None:
        return self.datasets_of([sample_id]).get(sample_id)

    def datasets_of(self, sample_ids: list[str]) -> dict[str, str]:
        if not sample_ids:
            return {}
        with connect() as db:
            rows = db.execute(
                "SELECT sample_id, dataset_id FROM dataset_samples WHERE sample_id = ANY(?)",
                (list(sample_ids),),
            ).fetchall()
        return {row["sample_id"]: row["dataset_id"] for row in rows}

    def admit(self, dataset_id: str, sample_ids: list[str]) -> dict:
        """Add confirmed samples and send each one to Label Studio.

        Every sample is checked before anything is sent. Samples that failed
        the quality checks, are already sent, or belong to another dataset are
        refused. If Label Studio refuses one part-way, the ones already sent
        stay admitted and the error names the rest.
        """
        if self.get(dataset_id) is None:
            raise KeyError(f"unknown dataset {dataset_id}")
        catalog = SampleCatalog(self.root)
        owners = self.datasets_of(sample_ids)
        tasks = []
        for sample_id in dict.fromkeys(sample_ids):
            record = catalog.get(sample_id)
            if record is None:
                raise KeyError(f"unknown sample {sample_id}")
            owner = owners.get(sample_id)
            if owner not in {None, dataset_id}:
                raise ValueError(f"样本 {sample_id} 已在别的数据集里")
            if record["status"] == "dropped":
                raise ValueError(f"样本 {sample_id} 没有通过质量检查，不能加入数据集")
            if record["status"] == "pending_review" and owner == dataset_id:
                raise ValueError(f"样本 {sample_id} 已经送去标注")
            if record["status"] != "generated":
                raise ValueError(f"样本 {sample_id} 的状态是 {record['status']}，不能加入数据集")
            task_path = catalog.sample_dir(sample_id) / "label_studio.json"
            if not task_path.is_file():
                raise ValueError(f"样本 {sample_id} 没有可送出的标注任务")
            tasks.append((sample_id, json.loads(task_path.read_text(encoding="utf-8"))))
        admitted = []
        for sample_id, task in tasks:
            pushed = push_task(task)
            if not pushed.get("success"):
                left = len(tasks) - len(admitted)
                detail = pushed.get("error") or "Label Studio 没有收下任务"
                raise ValueError(f"{detail}（已送出 {len(admitted)} 个，剩下 {left} 个没有送出）")
            task_id = None if pushed.get("task_id") is None else str(pushed["task_id"])
            self._link(dataset_id, sample_id)
            catalog.mark_pending_review(sample_id, task_id)
            admitted.append(sample_id)
        return {"dataset_id": dataset_id, "sample_ids": admitted}

    def attach_derived(self, dataset_id: str, sample_id: str) -> None:
        """Add an SD child of a reviewed member. It is already reviewed, so nothing goes to Label Studio."""
        owner = self.dataset_of(sample_id)
        if owner not in {None, dataset_id}:
            raise ValueError(f"样本 {sample_id} 已在别的数据集里")
        self._link(dataset_id, sample_id)

    def remove(self, dataset_id: str, sample_ids: list[str]) -> dict:
        """Drop samples that have not been sent to Label Studio."""
        if self.get(dataset_id) is None:
            raise KeyError(f"unknown dataset {dataset_id}")
        catalog = SampleCatalog(self.root)
        owners = self.datasets_of(sample_ids)
        removed = []
        for sample_id in sample_ids:
            if owners.get(sample_id) != dataset_id:
                raise ValueError(f"样本 {sample_id} 不在这个数据集里")
            record = catalog.get(sample_id)
            if record is None or record["status"] != "generated" or record["label_studio_task_id"]:
                raise ValueError(f"样本 {sample_id} 已经送去标注，不能移出")
            with connect() as db:
                db.execute(
                    "DELETE FROM dataset_samples WHERE dataset_id=? AND sample_id=?",
                    (dataset_id, sample_id),
                )
            removed.append(sample_id)
        return {"dataset_id": dataset_id, "sample_ids": removed}

    def set_head(self, dataset_id: str, version_id: str) -> None:
        with connect() as db:
            db.execute("UPDATE datasets SET head_version_id=? WHERE id=?", (version_id, dataset_id))

    def rollback(self, dataset_id: str, version_id: str) -> dict:
        """Make an older version of this dataset the head the next publish builds on."""
        with connect() as db:
            row = db.execute("SELECT dataset_id FROM versions WHERE version_id=?", (version_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown version {version_id}")
        if row["dataset_id"] != dataset_id:
            raise ValueError("这个版本不属于该数据集")
        self.set_head(dataset_id, version_id)
        return self.get(dataset_id)

    def _link(self, dataset_id: str, sample_id: str) -> None:
        with connect() as db:
            db.execute(
                """INSERT INTO dataset_samples(dataset_id, sample_id, added_at) VALUES(?,?,?)
                   ON CONFLICT (dataset_id, sample_id) DO NOTHING""",
                (dataset_id, sample_id, _now()),
            )

    def _with_counts(self, row: dict) -> dict:
        with connect() as db:
            counts = {
                item["status"]: int(item["n"])
                for item in db.execute(
                    """SELECT samples.status AS status, COUNT(*) AS n FROM dataset_samples
                       JOIN samples ON samples.sample_id = dataset_samples.sample_id
                       WHERE dataset_samples.dataset_id=? GROUP BY samples.status""",
                    (row["id"],),
                ).fetchall()
            }
            head = None
            if row["head_version_id"]:
                record = db.execute(
                    "SELECT revision FROM versions WHERE version_id=?", (row["head_version_id"],)
                ).fetchone()
                head = None if record is None else int(record["revision"])
        body = {
            "dataset_id": row["id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "head_version_id": row["head_version_id"],
            "head_revision": head,
            "samples": sum(counts.values()),
            "status_counts": counts,
        }
        return body | _pending(self.root, body)


def _pending(root: Path, dataset: dict) -> dict:
    """What the next publish would add and remove, for the publish button."""
    from firephantom.versioning.publish import plan_members

    if not dataset["samples"] and not dataset["head_version_id"]:
        return {"ready_to_publish": 0, "pending_added": 0, "pending_removed": 0}
    plan = plan_members(root, dataset["dataset_id"])
    added, removed = len(plan["added"]), len(plan["removed"])
    ready = len(plan["members"]) if (added or removed) else 0
    return {"ready_to_publish": ready, "pending_added": added, "pending_removed": removed}
