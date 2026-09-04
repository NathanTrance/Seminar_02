import logging
import json
from pathlib import Path

import pandas as pd

from .base import Sample

logger = logging.getLogger(__name__)


class EaseRagDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self._data: list[Sample] | None = None

    def load(self, split: str = "all") -> list[Sample]:
        if self._data is not None and split == "all":
            return self._data

        search_root = self.path.parent.parent.parent / "mal-LLM" / "RQ_experiments" / "data"

        if split == "all":
            samples = self._load_json_set(search_root, "test_malicious", 1)
            samples += self._load_json_set(search_root, "test_benign", 0)
            samples += self._load_json_set(search_root, "train_malicious", 1)
            samples += self._load_json_set(search_root, "train_benign", 0)
        elif split == "train":
            samples = self._load_json_set(search_root, "train_malicious", 1)
            samples += self._load_json_set(search_root, "train_benign", 0)
        elif split == "test":
            samples = self._load_json_set(search_root, "test_malicious", 1)
            samples += self._load_json_set(search_root, "test_benign", 0)
        elif split == "dev":
            mal = self._load_json_set(search_root, "test_malicious", 1)
            ben = self._load_json_set(search_root, "test_benign", 0)
            combined = mal + ben
            n = len(combined)
            samples = combined[: n // 2]
        else:
            raise ValueError(f"Unknown split: {split}")

        self._data = samples
        return samples

    def _load_json_set(self, data_dir: Path, name: str, label: int) -> list[Sample]:
        path_map = {
            "test_malicious": "test_malicious_packages_final.json",
            "test_benign": "test_benign_packages_final.json",
            "train_malicious": "train_malicious_packages_final.json",
            "train_benign": "train_benign_packages_final.json",
        }
        fname = path_map.get(name)
        if not fname:
            return []

        json_path = data_dir / fname
        if not json_path.exists():
            logger.warning("JSON not found: %s", json_path)
            return []

        with open(json_path) as f:
            entries = json.load(f)

        samples = []
        for i, entry in enumerate(entries):
            pkg_name = entry.get("package_name", f"{name}_{i}")
            setup_code = entry.get("setup.py")
            textual_desc = entry.get("textual_description")
            file_list = entry.get("file_list", "")

            sample = Sample(
                sample_id=f"ease_{name}_{i:05d}",
                dataset="ease_rag",
                label=label,
                raw_code=setup_code,
                behavior_text=textual_desc,
                package_name=pkg_name,
                metadata={"file_list": str(file_list), "source": name},
            )
            samples.append(sample)

        logger.info("Loaded %d samples from %s (label=%d)", len(samples), fname, label)
        return samples

    @staticmethod
    def _sample_split(samples: list[Sample], split: str) -> list[Sample]:
        if split == "all":
            return samples
        return samples
