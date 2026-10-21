"""Benchmark the PostgreSQL version directory without creating image artifacts."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from firephantom.versioning.store import DatasetStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--parent", type=Path, default=Path(tempfile.gettempdir()))
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    args.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="firephantom-version-catalog-", dir=args.parent) as directory:
        store = DatasetStore(directory)
        started = time.perf_counter()

        def rows():
            for revision in range(1, args.count + 1):
                version_id = f"benchmark-{revision:06d}"
                record = {
                    "version_id": version_id,
                    "版本号": str(revision),
                    "revision": revision,
                    "digest": f"{revision:064x}",
                    "metadata": {},
                }
                yield (
                    version_id,
                    revision,
                    record["digest"],
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                )

        with store._catalog() as db:
            db.executemany(
                "INSERT INTO versions(version_id,revision,digest,record) VALUES(?,?,?,?)",
                rows(),
            )
            db.execute(
                "UPDATE store_state SET state_value=? WHERE state_key='head'",
                (f"benchmark-{args.count:06d}",),
            )
        insert_seconds = time.perf_counter() - started

        started = time.perf_counter()
        count = store.version_count()
        count_seconds = time.perf_counter() - started

        started = time.perf_counter()
        newest_page = store.list_versions(offset=0, limit=50, newest_first=True)
        newest_page_seconds = time.perf_counter() - started

        started = time.perf_counter()
        oldest_page = store.list_versions(offset=max(0, count - 50), limit=50, newest_first=True)
        oldest_page_seconds = time.perf_counter() - started

        started = time.perf_counter()
        middle = store.get(f"benchmark-{(count + 1) // 2:06d}")
        lookup_seconds = time.perf_counter() - started

        assert count == args.count
        assert len(newest_page) == min(50, count)
        assert newest_page[0]["revision"] == count
        assert oldest_page[0]["revision"] == min(50, count)
        assert middle is not None
        print(
            json.dumps(
                {
                    "scope": "PostgreSQL version catalog only; no DVC image payloads or pipeline work",
                    "versions": count,
                    "insert_seconds": round(insert_seconds, 3),
                    "insert_records_per_second": round(count / insert_seconds, 1),
                    "count_query_ms": round(count_seconds * 1000, 3),
                    "50_row_newest_page_ms": round(newest_page_seconds * 1000, 3),
                    "50_row_deep_page_ms": round(oldest_page_seconds * 1000, 3),
                    "primary_key_lookup_ms": round(lookup_seconds * 1000, 3),
                    "head": store.head(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
