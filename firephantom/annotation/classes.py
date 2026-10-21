"""One class table for annotation, Label Studio, exports, training and evaluation.

Ids are positions in ``CLASSES`` and never depend on which labels a version
happens to contain.
"""

from __future__ import annotations

CLASSES = ("flame", "smoke", "ember")
_ALIASES = {"fire": "flame"}


def canonical(name: object) -> str:
    text = str(name or "flame").strip().lower()
    text = _ALIASES.get(text, text)
    if text not in CLASSES:
        raise ValueError(f"未知类别 {name!r}，可选：{'、'.join(CLASSES)}")
    return text


def class_id(name: object) -> int:
    return CLASSES.index(canonical(name))
