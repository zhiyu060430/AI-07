"""DVC incremental add, conflict rejection, rollback, and desensitized metadata."""

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from firephantom.api import create_app
from firephantom.versioning.desensitize import desensitize
from firephantom.versioning.store import DatasetStore, VersionConflict


def _snapshot(root):
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_desensitize_keeps_dataset_fields_and_drops_personal_identifiers():
    cleaned = desensitize(
        {
            "版本号": "7",
            "生成参数": {"scenario": "房间火灾", "email": "pilot@example.com", "wind_speed": 1.5},
            "质量报告": {"fid": 0.2, "note": "联系 13800138000 身份证110101199001011234"},
            "姓名": "张三",
            "电话": "13900139000",
        }
    )
    text = json.dumps(cleaned, ensure_ascii=False)
    assert cleaned["版本号"] == "7"
    assert cleaned["生成参数"]["scenario"] == "房间火灾"
    assert cleaned["生成参数"]["wind_speed"] == 1.5
    assert "email" not in cleaned["生成参数"]
    assert "fid" in cleaned["质量报告"]
    for raw in ("张三", "pilot@example.com", "13800138000", "13900139000", "110101199001011234"):
        assert raw not in text


def test_dvc_add_conflict_and_rollback(tmp_path):
    store = DatasetStore(tmp_path)
    assert store.list_versions() == []
    meta_v1 = {
        "生成参数": {"scenario": "房间火灾", "email": "secret@example.com"},
        "质量报告": {"fid": 0.1, "is": 1.2, "ssim": 0.8},
        "姓名": "李四",
    }
    files_v1 = {"a.txt": b"version-one", "nested/b.txt": b"keep"}
    v1 = store.add(files_v1, meta_v1, version_id="v1")
    assert v1["版本号"] == "1"
    assert (store.workspace / "a.txt").read_bytes() == b"version-one"
    stored = json.loads((store.workspace / "metadata.json").read_text(encoding="utf-8"))
    assert stored["版本号"] == "1"
    assert "生成参数" in stored and "质量报告" in stored
    raw_meta = json.dumps(stored, ensure_ascii=False)
    assert "李四" not in raw_meta
    assert "secret@example.com" not in raw_meta
    assert "md5" in (tmp_path / "workspace.dvc").read_text(encoding="utf-8")
    pointer_v1 = store._pointer_path("v1").read_text(encoding="utf-8")

    files_v2 = {"a.txt": b"version-two", "extra.txt": b"extra"}
    meta_v2 = {
        "生成参数": {"scenario": "林区火灾"},
        "质量报告": {"fid": 0.4, "is": 1.4, "ssim": 0.7},
    }
    v2 = store.add(files_v2, meta_v2, version_id="v2")
    assert v2["revision"] > v1["revision"]
    assert (store.workspace / "a.txt").read_bytes() == b"version-two"
    assert (store.workspace / "extra.txt").read_bytes() == b"extra"

    cache_before = _snapshot(tmp_path / ".dvc" / "cache")
    workspace_before = _snapshot(store.workspace)
    pointer_before = store._pointer_path("v1").read_bytes()
    catalog_before = store.list_versions()
    with pytest.raises(VersionConflict):
        store.add({"a.txt": b"conflict"}, meta_v2, version_id="v1")
    assert _snapshot(tmp_path / ".dvc" / "cache") == cache_before
    assert _snapshot(store.workspace) == workspace_before
    assert store._pointer_path("v1").read_bytes() == pointer_before
    assert store.list_versions() == catalog_before
    assert store.version_count() == 2

    # Same identity and same bytes is not a rewrite.
    snapshot_after_conflict = _snapshot(tmp_path)
    again = store.add(files_v2, meta_v2, version_id="v2")
    assert again["version_id"] == "v2"
    assert again["digest"] == v2["digest"]
    assert _snapshot(tmp_path) == snapshot_after_conflict

    restored = store.rollback("v1")
    assert restored["版本号"] == "1"
    assert (store.workspace / "a.txt").read_bytes() == b"version-one"
    assert (store.workspace / "nested" / "b.txt").read_bytes() == b"keep"
    assert not (store.workspace / "extra.txt").exists()
    rolled_meta = json.loads((store.workspace / "metadata.json").read_text(encoding="utf-8"))
    assert rolled_meta["版本号"] == "1"
    assert rolled_meta["生成参数"]["scenario"] == "房间火灾"
    assert rolled_meta["质量报告"]["fid"] == 0.1
    assert (tmp_path / "workspace.dvc").read_text(encoding="utf-8") == pointer_v1
    assert store.get("v2")["版本号"] == "2"


def test_rollback_unknown_version_raises(tmp_path):
    store = DatasetStore(tmp_path)
    store.add({"a.txt": b"only"}, {"生成参数": {}, "质量报告": {}}, version_id="only")
    with pytest.raises(KeyError):
        store.rollback("missing")


def test_version_api_paginates_newest_first_and_validates_bounds(tmp_path):
    store = DatasetStore(tmp_path)
    for revision in range(1, 6):
        store.add(
            {"sample.txt": f"sample-{revision}".encode()},
            {"生成参数": {"revision": revision}, "质量报告": {}},
            version_id=f"v{revision}",
        )

    with TestClient(create_app(tmp_path)) as client:
        first = client.get("/api/v1/versions?limit=2")
        assert first.status_code == 200
        assert first.json() == {
            "head": "v5",
            "total": 5,
            "offset": 0,
            "limit": 2,
            "has_more": True,
            "versions": [store.get("v5"), store.get("v4")],
        }
        last = client.get("/api/v1/versions?offset=4&limit=2").json()
        assert [record["version_id"] for record in last["versions"]] == ["v1"]
        assert last["has_more"] is False
        assert client.get("/api/v1/versions?offset=-1").status_code == 422
        assert client.get("/api/v1/versions?limit=201").status_code == 422
