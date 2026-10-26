from pathlib import Path
import os

import pytest


def _postgres_password() -> str:
    configured = os.environ.get("FIREPHANTOM_TEST_POSTGRES_PASSWORD", "").strip()
    if configured:
        return configured
    from firephantom.paths import postgres_password_file

    password_file = postgres_password_file()
    if password_file.is_file():
        return password_file.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "PostgreSQL is required for tests. Create "
        f"{password_file} or set FIREPHANTOM_TEST_POSTGRES_PASSWORD, and start the PostgreSQL service."
    )


@pytest.fixture(scope="session", autouse=True)
def postgres_test_database():
    if os.environ.get("FIREPHANTOM_SKIP_POSTGRES") == "1":
        return None
    password = _postgres_password()
    url = f"postgresql://firephantom:{password}@127.0.0.1:5432/firephantom_test"
    os.environ["FIREPHANTOM_DATABASE_URL"] = url
    from firephantom.db import ensure_database, ensure_schema

    ensure_database(url)
    ensure_schema(url)
    return url


@pytest.fixture(autouse=True)
def isolated_postgres(postgres_test_database, monkeypatch, tmp_path_factory):
    if postgres_test_database is None:
        return
    monkeypatch.setenv("FIREPHANTOM_DATABASE_URL", postgres_test_database)
    secret = tmp_path_factory.mktemp("auth") / "session.secret"
    monkeypatch.setenv("FIREPHANTOM_SESSION_SECRET_FILE", str(secret))
    from firephantom.db import reset_tables

    reset_tables(postgres_test_database)


@pytest.fixture(autouse=True)
def keep_tests_independent_of_local_services(monkeypatch):
    from firephantom.physics import adapters

    monkeypatch.setenv("SD_API_URL", "http://127.0.0.1:9/sdapi/v1/txt2img")
    monkeypatch.setenv("LABEL_STUDIO_URL", "http://127.0.0.1:9/api/tasks")
    for name in (
        "ANNOTATION_MODEL",
        "BLENDER_BIN",
        "FIRE_REFERENCE_DATASET",
        "INCEPTION_WEIGHTS",
        "LABEL_STUDIO_PROJECT_ID",
        "LABEL_STUDIO_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(adapters.shutil, "which", lambda _name: None)
