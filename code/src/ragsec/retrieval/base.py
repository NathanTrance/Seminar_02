from dataclasses import dataclass


@dataclass
class RetrievedDoc:
    doc_id: str
    content: str
    score: float
    rank: int
    metadata: dict | None = None
    pool: str = "malicious"
