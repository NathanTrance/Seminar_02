import hashlib
import json


def content_hash(content: str | bytes | None) -> str:
    if content is None:
        content = ""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def dict_hash(d: dict) -> str:
    raw = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
