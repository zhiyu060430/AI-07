"""Optional authenticated encryption for datasets at rest."""

from __future__ import annotations

import base64
import hashlib
import os


def _fernet_key() -> bytes | None:
    value = os.environ.get("DATASET_ENCRYPTION_KEY", "").strip()
    if not value:
        return None
    try:
        decoded = base64.urlsafe_b64decode(value.encode("ascii"))
        if len(decoded) == 32:
            return value.encode("ascii")
    except (UnicodeEncodeError, ValueError):
        pass
    return base64.urlsafe_b64encode(hashlib.sha256(value.encode("utf-8")).digest())


def enabled() -> bool:
    return _fernet_key() is not None


def encrypt(payload: bytes) -> bytes:
    key = _fernet_key()
    if key is None:
        return payload
    from cryptography.fernet import Fernet

    return Fernet(key).encrypt(payload)


def decrypt(payload: bytes) -> bytes:
    key = _fernet_key()
    if key is None:
        return payload
    from cryptography.fernet import Fernet, InvalidToken

    try:
        return Fernet(key).decrypt(payload)
    except InvalidToken as exc:
        raise ValueError("dataset payload cannot be decrypted with DATASET_ENCRYPTION_KEY") from exc
