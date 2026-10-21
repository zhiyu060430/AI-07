"""Account records stored in PostgreSQL."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import os
import secrets

import psycopg

from firephantom.db import connect, ensure_schema


SERVICE_EMAIL = "firephantom@localhost"
_ITERATIONS = 200_000


class AccountError(ValueError):
    """A rejected account change that should be shown to the caller."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(candidate.hex(), digest_hex)
    except (TypeError, ValueError):
        return False


def _normalize_email(email: str) -> str:
    cleaned = email.strip().lower()
    if len(cleaned) > 255 or "@" not in cleaned or cleaned.startswith("@") or cleaned.endswith("@"):
        raise AccountError("请输入有效邮箱")
    if cleaned == SERVICE_EMAIL:
        raise AccountError(f"{SERVICE_EMAIL} 是 Label Studio 服务账号，不能用作工作台登录")
    return cleaned


def _check_password(password: str) -> None:
    if not isinstance(password, str) or not 8 <= len(password) <= 128:
        raise AccountError("密码需要 8 到 128 个字符")


def _public(row) -> dict:
    return {
        "id": int(row["id"]),
        "email": row["email"],
        "is_admin": bool(row["is_admin"]),
        "created_at": row["created_at"],
    }


def count_users() -> int:
    ensure_schema()
    with connect() as db:
        return int(db.execute("SELECT COUNT(*) FROM users").fetchone()[0])


def create_user(email: str, password: str, *, is_admin: bool) -> dict:
    cleaned = _normalize_email(email)
    _check_password(password)
    ensure_schema()
    secret = secrets.token_urlsafe(24)
    try:
        with connect() as db:
            db.execute(
                """INSERT INTO users(email, password_hash, is_admin, created_at, label_studio_secret)
                   VALUES(?,?,?,?,?)""",
                (cleaned, hash_password(password), int(is_admin), _now(), secret),
            )
            row = db.execute("SELECT * FROM users WHERE email=?", (cleaned,)).fetchone()
    except psycopg.errors.UniqueViolation as exc:
        raise AccountError("这个邮箱已经注册") from exc
    return _public(row)


def authenticate(email: str, password: str) -> dict | None:
    cleaned = email.strip().lower()
    ensure_schema()
    with connect() as db:
        row = db.execute("SELECT * FROM users WHERE email=?", (cleaned,)).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return _public(row)


def get_user(user_id: int) -> dict | None:
    ensure_schema()
    with connect() as db:
        row = db.execute("SELECT * FROM users WHERE id=?", (int(user_id),)).fetchone()
    return _public(row) if row else None


def label_studio_secret(user_id: int) -> tuple[str, str] | None:
    ensure_schema()
    with connect() as db:
        row = db.execute(
            "SELECT email, label_studio_secret FROM users WHERE id=?",
            (int(user_id),),
        ).fetchone()
    if row is None or not row["label_studio_secret"]:
        return None
    return row["email"], row["label_studio_secret"]
