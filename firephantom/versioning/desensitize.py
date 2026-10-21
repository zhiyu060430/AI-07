"""Remove direct personal identifiers from dataset metadata."""

from __future__ import annotations

import copy
import re

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")

# Exact tokens only. Substring matching would erase innocent keys such as filename.
_KEY_TOKENS = {
    "name",
    "full_name",
    "first_name",
    "last_name",
    "email",
    "e-mail",
    "phone",
    "mobile",
    "telephone",
    "address",
    "username",
    "user",
    "contact",
    "ssn",
    "id_card",
    "idcard",
    "operator",
    "姓名",
    "名字",
    "电话",
    "手机",
    "身份证",
    "邮箱",
    "地址",
    "联系人",
}


def _personal_key(key: str) -> bool:
    parts = re.split(r"[_\-\s]+", str(key).strip().lower())
    if any(part in _KEY_TOKENS for part in parts):
        return True
    return str(key).strip() in _KEY_TOKENS


def _scrub_text(value: str) -> str:
    value = _EMAIL.sub("[已脱敏]", value)
    value = _PHONE.sub("[已脱敏]", value)
    value = _ID_CARD.sub("[已脱敏]", value)
    return value


def desensitize(metadata: dict) -> dict:
    """Return a deep copy with personal keys removed and identifier strings scrubbed."""

    def walk(node):
        if isinstance(node, dict):
            cleaned = {}
            for key, value in node.items():
                if _personal_key(str(key)):
                    continue
                cleaned[key] = walk(value)
            return cleaned
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, str):
            return _scrub_text(node)
        return node

    return walk(copy.deepcopy(metadata))
