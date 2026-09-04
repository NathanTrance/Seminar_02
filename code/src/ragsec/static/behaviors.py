import ast

from .ast_features import ASTFeatureExtractor


BEHAVIOR_LABELS = [
    "shell_execution",
    "process_creation",
    "network_access",
    "file_write",
    "file_delete",
    "dynamic_execution",
    "environment_access",
    "base64_decode",
    "remote_download",
    "persistence",
    "obfuscation",
    "credential_access",
]


def extract_behavior_flags(code: str) -> dict[str, bool]:
    extractor = ASTFeatureExtractor(code)
    behaviors = extractor.detect_behaviors()
    return {k: len(v) > 0 for k, v in behaviors.items()}


def behavior_text_to_flags(text: str) -> dict[str, bool]:
    flags = {b: False for b in BEHAVIOR_LABELS}
    for part in text.split(";"):
        part = part.strip()
        if part.startswith("behaviors="):
            for b in part[len("behaviors="):].split(","):
                b = b.strip()
                if b in flags:
                    flags[b] = True
    return flags
