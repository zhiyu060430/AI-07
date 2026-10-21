"""PostgreSQL access for accounts, jobs, versions, and samples."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import re
import threading
from urllib.parse import unquote, urlparse

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError as exc:  # pragma: no cover - dependency is installed with the app
    raise RuntimeError("psycopg is required. Install requirements.txt.") from exc


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")
_SCHEMES = {"postgresql", "postgres"}
_schema_lock = threading.Lock()
_schema_ready: set[str] = set()


class Row(dict):
    """Dict row that also accepts a column position."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class Cursor:
    def __init__(self, raw):
        self.raw = raw

    def execute(self, sql: str, params=None):
        statement = _translate(sql)
        if params is None:
            self.raw.execute(statement)
        else:
            self.raw.execute(statement, params)
        return self

    def executemany(self, sql: str, seq):
        rows = list(seq)
        if rows:
            self.raw.executemany(_translate(sql), rows)
        return self

    @property
    def rowcount(self) -> int:
        return int(self.raw.rowcount)

    def fetchone(self):
        row = self.raw.fetchone()
        return Row(row) if row is not None else None

    def fetchall(self):
        return [Row(row) for row in self.raw.fetchall()]

    def __iter__(self):
        return iter(self.fetchall())


class Connection:
    def __init__(self, raw):
        self.raw = raw

    def execute(self, sql: str, params=None) -> Cursor:
        cursor = Cursor(self.raw.cursor())
        return cursor.execute(sql, params)

    def executemany(self, sql: str, seq) -> Cursor:
        cursor = Cursor(self.raw.cursor())
        return cursor.executemany(sql, seq)

    def commit(self) -> None:
        self.raw.commit()

    def rollback(self) -> None:
        self.raw.rollback()

    def close(self) -> None:
        self.raw.close()


def database_url() -> str:
    configured = os.environ.get("FIREPHANTOM_DATABASE_URL", "").strip()
    if configured:
        return configured
    password_file = Path(os.environ.get(
        "FIREPHANTOM_POSTGRES_PASSWORD_FILE",
        "/Volumes/T7/FirePhantom-runtime/postgres.password",
    ))
    if password_file.is_file():
        password = password_file.read_text(encoding="utf-8").strip()
        return f"postgresql://firephantom:{password}@127.0.0.1:5432/firephantom"
    raise RuntimeError("FIREPHANTOM_DATABASE_URL is not set and no PostgreSQL password file was found")


def parse_database_url(url: str | None = None) -> dict:
    selected = url or database_url()
    parsed = urlparse(selected)
    if parsed.scheme not in _SCHEMES:
        raise RuntimeError("FIREPHANTOM_DATABASE_URL must start with postgresql://")
    name = unquote(parsed.path.lstrip("/"))
    if not _IDENTIFIER.fullmatch(name or ""):
        raise RuntimeError("PostgreSQL database name must contain only letters, digits, and underscores")
    return {
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 5432,
        "database": name,
    }


def _translate(sql: str) -> str:
    return sql.replace("?", "%s")


def _connect_raw(selected: dict, database: str | None, *, autocommit: bool = False):
    return psycopg.connect(
        host=selected["host"],
        port=int(selected["port"]),
        user=selected["user"],
        password=selected["password"],
        dbname=database,
        autocommit=autocommit,
        connect_timeout=5,
        row_factory=dict_row,
    )


@contextmanager
def connect(url: str | None = None):
    selected = parse_database_url(url)
    raw = _connect_raw(selected, selected["database"])
    connection = Connection(raw)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _missing_database(exc: BaseException) -> bool:
    if getattr(exc, "sqlstate", None) == "3D000":
        return True
    message = str(exc)
    return "database" in message and "does not exist" in message


def ensure_database(url: str | None = None) -> None:
    selected = parse_database_url(url)
    name = selected["database"]
    try:
        raw = _connect_raw(selected, name)
    except psycopg.OperationalError as exc:
        if not _missing_database(exc):
            raise
    else:
        raw.close()
        return
    admin = _connect_raw(selected, "postgres", autocommit=True)
    try:
        admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        admin.close()


def ensure_schema(url: str | None = None) -> None:
    """Create tables once per database in this process.

    Each image request used to run the DDL. Concurrent ``ALTER TABLE`` takes
    exclusive locks and the gallery requests deadlock each other.
    """
    key = url if url is not None else database_url()
    if key in _schema_ready:
        return
    with _schema_lock:
        if key in _schema_ready:
            return
        _apply_schema(url)
        _schema_ready.add(key)


def _apply_schema(url: str | None = None) -> None:
    statements = [
        """CREATE TABLE IF NOT EXISTS users (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            email VARCHAR(255) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            is_admin SMALLINT NOT NULL DEFAULT 0,
            created_at VARCHAR(40) NOT NULL,
            label_studio_secret VARCHAR(255) NULL
        )""",
        """CREATE TABLE IF NOT EXISTS jobs (
            id VARCHAR(64) NOT NULL PRIMARY KEY,
            state VARCHAR(32) NOT NULL,
            total INT NOT NULL,
            completed INT NOT NULL DEFAULT 0,
            kept INT NOT NULL DEFAULT 0,
            dropped INT NOT NULL DEFAULT 0,
            created_at VARCHAR(40) NOT NULL,
            updated_at VARCHAR(40) NOT NULL,
            payload TEXT NOT NULL,
            error TEXT NULL,
            cancel_requested SMALLINT NOT NULL DEFAULT 0
        )""",
        """CREATE TABLE IF NOT EXISTS job_results (
            job_id VARCHAR(64) NOT NULL,
            result_index INT NOT NULL,
            result TEXT NOT NULL,
            PRIMARY KEY (job_id, result_index)
        )""",
        """CREATE TABLE IF NOT EXISTS versions (
            version_id VARCHAR(128) NOT NULL PRIMARY KEY,
            revision INT NOT NULL UNIQUE,
            digest VARCHAR(128) NOT NULL,
            record TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS versions_revision_idx ON versions (revision)",
        """CREATE TABLE IF NOT EXISTS store_state (
            state_key VARCHAR(64) NOT NULL PRIMARY KEY,
            state_value TEXT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS samples (
            sample_id VARCHAR(64) NOT NULL PRIMARY KEY,
            status VARCHAR(32) NOT NULL,
            seed BIGINT NOT NULL,
            generator_version VARCHAR(128) NOT NULL,
            generator_profile TEXT NOT NULL,
            scene_params TEXT NOT NULL,
            quality TEXT NULL,
            filter_decision TEXT NULL,
            auto_objects TEXT NULL,
            reviewed_objects TEXT NULL,
            label_studio_task_id VARCHAR(64) NULL,
            annotation_error DOUBLE PRECISION NULL,
            created_at VARCHAR(40) NOT NULL,
            updated_at VARCHAR(40) NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS version_samples (
            version_id VARCHAR(128) NOT NULL,
            sample_id VARCHAR(64) NOT NULL,
            PRIMARY KEY (version_id, sample_id)
        )""",
        """CREATE TABLE IF NOT EXISTS version_evaluations (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            version_id VARCHAR(128) NOT NULL,
            recorded_at VARCHAR(40) NOT NULL,
            metrics TEXT NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS version_evaluations_idx
           ON version_evaluations (version_id, recorded_at)""",
        """CREATE TABLE IF NOT EXISTS datasets (
            id VARCHAR(64) NOT NULL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            created_at VARCHAR(40) NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS dataset_samples (
            dataset_id VARCHAR(64) NOT NULL,
            sample_id VARCHAR(64) NOT NULL,
            added_at VARCHAR(40) NOT NULL,
            PRIMARY KEY (dataset_id, sample_id)
        )""",
        """CREATE UNIQUE INDEX IF NOT EXISTS dataset_samples_sample_idx
           ON dataset_samples (sample_id)""",
        # One bake renders many samples; frames of a sequence share a split.
        "ALTER TABLE samples ADD COLUMN IF NOT EXISTS sequence_id VARCHAR(64) NULL",
        "ALTER TABLE samples ADD COLUMN IF NOT EXISTS split VARCHAR(8) NULL",
        "ALTER TABLE samples ADD COLUMN IF NOT EXISTS derived_from VARCHAR(64) NULL",
        "CREATE INDEX IF NOT EXISTS samples_sequence_idx ON samples (sequence_id)",
        # Versions are cumulative: each one lists every member and its parent.
        "ALTER TABLE versions ADD COLUMN IF NOT EXISTS parent_id VARCHAR(128) NULL",
        "ALTER TABLE versions ADD COLUMN IF NOT EXISTS dataset_id VARCHAR(64) NULL",
        "ALTER TABLE versions ADD COLUMN IF NOT EXISTS created_at VARCHAR(40) NULL",
        "ALTER TABLE version_samples ADD COLUMN IF NOT EXISTS split VARCHAR(8) NULL",
        "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS head_version_id VARCHAR(128) NULL",
        """CREATE TABLE IF NOT EXISTS scenes (
            id VARCHAR(64) NOT NULL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            scenario VARCHAR(64) NOT NULL,
            origin_fuel VARCHAR(128) NOT NULL,
            placements TEXT NOT NULL,
            revision INT NOT NULL DEFAULT 1,
            has_thumbnail SMALLINT NOT NULL DEFAULT 0,
            created_by VARCHAR(255) NULL,
            created_at VARCHAR(40) NOT NULL,
            updated_at VARCHAR(40) NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS scene_revisions (
            scene_id VARCHAR(64) NOT NULL,
            revision INT NOT NULL,
            scenario VARCHAR(64) NOT NULL,
            origin_fuel VARCHAR(128) NOT NULL,
            placements TEXT NOT NULL,
            created_at VARCHAR(40) NOT NULL,
            PRIMARY KEY (scene_id, revision)
        )""",
        # Moved fire, per-viewpoint cameras, and a custom gas box. Null or {} means automatic.
        "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS origin TEXT NULL",
        "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS cameras TEXT NOT NULL DEFAULT '{}'",
        "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS gas_domain TEXT NULL",
        "ALTER TABLE scene_revisions ADD COLUMN IF NOT EXISTS origin TEXT NULL",
        "ALTER TABLE scene_revisions ADD COLUMN IF NOT EXISTS cameras TEXT NOT NULL DEFAULT '{}'",
        "ALTER TABLE scene_revisions ADD COLUMN IF NOT EXISTS gas_domain TEXT NULL",
        # Null lamps mean the automatic fill and sun. [] means the user removed every lamp.
        "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS lights TEXT NULL",
        "ALTER TABLE scene_revisions ADD COLUMN IF NOT EXISTS lights TEXT NULL",
        # Null means every Mantaflow and Cycles default in physics/simulation.py.
        "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS simulation TEXT NULL",
        "ALTER TABLE scene_revisions ADD COLUMN IF NOT EXISTS simulation TEXT NULL",
        # Null structures and domain mean the preset's boxes and site.
        "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS structures TEXT NULL",
        "ALTER TABLE scenes ADD COLUMN IF NOT EXISTS domain TEXT NULL",
        "ALTER TABLE scene_revisions ADD COLUMN IF NOT EXISTS structures TEXT NULL",
        "ALTER TABLE scene_revisions ADD COLUMN IF NOT EXISTS domain TEXT NULL",
        # SD restyles of reviewed samples, run per dataset after review.
        """CREATE TABLE IF NOT EXISTS augment_jobs (
            id VARCHAR(64) NOT NULL PRIMARY KEY,
            dataset_id VARCHAR(64) NOT NULL,
            state VARCHAR(32) NOT NULL,
            total INT NOT NULL,
            completed INT NOT NULL DEFAULT 0,
            created INT NOT NULL DEFAULT 0,
            failed INT NOT NULL DEFAULT 0,
            payload TEXT NOT NULL,
            error TEXT NULL,
            cancel_requested SMALLINT NOT NULL DEFAULT 0,
            created_at VARCHAR(40) NOT NULL,
            updated_at VARCHAR(40) NOT NULL
        )""",
    ]
    with connect(url) as db:
        for statement in statements:
            db.execute(statement)
        db.execute(
            """INSERT INTO store_state(state_key, state_value) VALUES('head', NULL)
               ON CONFLICT (state_key) DO NOTHING"""
        )


def reset_tables(url: str | None = None) -> None:
    """Remove rows from every application table. Tests use this for isolation."""
    ensure_schema(url)
    with connect(url) as db:
        names = [
            row[0]
            for row in db.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
        ]
        for name in names:
            if _IDENTIFIER.fullmatch(str(name)):
                db.execute(f'TRUNCATE TABLE "{name}" RESTART IDENTITY CASCADE')
        db.execute(
            """INSERT INTO store_state(state_key, state_value) VALUES('head', NULL)
               ON CONFLICT (state_key) DO NOTHING"""
        )
