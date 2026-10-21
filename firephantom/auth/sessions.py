"""HMAC session cookies. The cookie carries a user id, not a password."""

from __future__ import annotations

from pathlib import Path
import base64
import hashlib
import hmac
import json
import os
import time


COOKIE = "fp_session"
MAX_AGE_SECONDS = 14 * 24 * 60 * 60


def secret() -> bytes:
    configured = os.environ.get("FIREPHANTOM_SESSION_SECRET_FILE", "").strip()
    path = Path(configured or "/Volumes/T7/FirePhantom-runtime/firephantom-session.secret")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(os.urandom(32))
        path.chmod(0o600)
    return path.read_bytes().strip()


def issue(user_id: int) -> str:
    payload = {
        "uid": int(user_id),
        "exp": int(time.time()) + MAX_AGE_SECONDS,
    }
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii").rstrip("=")
    signature = hmac.new(secret(), body.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{body}.{signature}"


def read(token: str | None) -> int | None:
    if not token or "." not in token:
        return None
    body, signature = token.rsplit(".", 1)
    expected = hmac.new(secret(), body.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    padded = body + "=" * (-len(body) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        if int(payload["exp"]) < time.time():
            return None
        return int(payload["uid"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
