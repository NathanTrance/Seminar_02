import json
import sqlite3
from pathlib import Path

from .hashing import content_hash, dict_hash


class Cache:
    def __init__(self, path: str | Path = "data/cache/cache.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  key TEXT PRIMARY KEY,"
            "  value TEXT,"
            "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def llm_cache_key(
        self,
        model: str,
        prompt_hash: str,
        sample_hash: str,
        evidence_ids: list[str] | None = None,
        temperature: float = 0.0,
    ) -> str:
        parts = ["llm", model, prompt_hash, sample_hash, str(temperature)]
        if evidence_ids:
            parts.append("|".join(sorted(evidence_ids)))
        return ":".join(parts)

    def get_llm(self, key: str) -> dict | None:
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set_llm(self, key: str, value: dict) -> None:
        self.set(key, json.dumps(value, default=str))

    def embedding_cache_key(self, model: str, content: str) -> str:
        return f"emb:{model}:{content_hash(content)}"

    def close(self) -> None:
        self._conn.close()
