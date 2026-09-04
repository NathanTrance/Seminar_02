from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Sample:
    sample_id: str
    dataset: str
    label: int
    raw_code: str | None = None
    behavior_text: str | None = None
    package_name: str | None = None
    file_path: str | None = None
    metadata: dict = field(default_factory=dict)


class DatasetAdapter(Protocol):
    def load(self, split: str) -> list[Sample]:
        ...
