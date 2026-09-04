#!/usr/bin/env python3
"""
Offline recomputation of experiment metrics from raw JSONL results.
Stdlib-only (json + ast), no external dependencies.

Produces:
  metrics.json      - full per-method metrics (unfiltered + NA-filtered)
  tables.tex        - LaTeX table bodies for the report

Usage:
  python recompute_metrics.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPORT_ROOT = Path(__file__).resolve().parents[1]
SEMINAR_ROOT = REPORT_ROOT.parent
CODE_ROOT = SEMINAR_ROOT / "code"
RESULTS_DIR = CODE_ROOT / "results" / "raw"
DATA_DIR = CODE_ROOT / "mal-LLM" / "RQ_experiments" / "data"

sys.path.insert(0, str(CODE_ROOT))

from src.ragsec.static.ast_features import ASTFeatureExtractor  # noqa: E402

METHODS = ["no_rag", "dense", "bm25", "hybrid", "behavior",
           "contrastive", "behavior_contrastive"]
MODEL = "Llama-3.1-8B-Instruct"

BEHAVIOR_VOCAB = [
    "shell_execution", "process_creation", "network_access", "file_write",
    "file_delete", "dynamic_execution", "environment_access",
    "base64_decode", "remote_download", "persistence", "obfuscation",
    "credential_access",
]

KEYWORDS = {
    "shell_execution": ["shell", "powershell", "command_execution", "os.system",
                        "system_call", "command execution"],
    "process_creation": ["process", "subprocess", "spawn", "fork", "popen"],
    "network_access": ["network", "socket", "url", "http", "webhook", "dns",
                       "ip_address", "connection", "connect"],
    "file_write": ["file_write", "write_file", "file_modification", "file_create",
                   "write to file", "file writing"],
    "file_delete": ["file_delete", "remove_file", "file_removal",
                    "delete file", "unlink"],
    "dynamic_execution": ["dynamic_execution", "code_execution", "exec(", "eval",
                          "execute_code", "arbitrary_code", "code injection",
                          "code_injection", "remote_code", "rce"],
    "environment_access": ["environment", "environ", "env_var", "getenv", "os.environ"],
    "base64_decode": ["base64", "decode", "encoded", "obfuscated string",
                      "hex_decode", "rot13"],
    "remote_download": ["download", "urlretrieve", "fetch_and_execute",
                        "download_and_execute", "remote_file", "dropper"],
    "persistence": ["persistence", "startup", "registry", "autostart", "login_item",
                    "cron", "scheduled_task", "boot"],
    "obfuscation": ["marshaled", "marshal", "pickle", "compression", "zlib",
                    "b64decode", "hidden"],
    "credential_access": ["credential", "token", "password", "steal", "exfiltrat",
                          "keylog", "cookie", "browser_cookie", "discord token",
                          "wallet", "clipboard", "sensitive"],
    "file_access": ["file_access", "file_reading", "read_file", "reads_from_file",
                    "open(", "read file"],
}


def normalize_claim(claim_type: str):
    t = claim_type.strip().lower().replace(" ", "_")
    hits = []
    for behavior, kws in KEYWORDS.items():
        if any(kw in t for kw in kws):
            hits.append(behavior)
    return hits[0] if hits else None


def load_results(method: str) -> list[dict]:
    path = RESULTS_DIR / f"{method}_{MODEL}_ease_rag.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def load_test_data() -> tuple[dict[str, dict], set[str]]:
    """Return {sample_id: sample} and set of sample_ids with NA code."""
    mal = json.loads((DATA_DIR / "test_malicious_packages_final.json").read_text(encoding="utf-8"))
    ben = json.loads((DATA_DIR / "test_benign_packages_final.json").read_text(encoding="utf-8"))
    id_to_sample, na_ids = {}, set()
    for i, e in enumerate(mal):
        sid = f"ease_test_malicious_{i:05d}"
        code = e.get("setup.py")
        id_to_sample[sid] = {
            "sample_id": sid, "label": 1, "code": code, "pkg": e["package_name"],
        }
        if code == "Not Available":
            na_ids.add(sid)
    for i, e in enumerate(ben):
        sid = f"ease_test_benign_{i:05d}"
        code = e.get("setup.py")
        id_to_sample[sid] = {
            "sample_id": sid, "label": 0, "code": code, "pkg": e["package_name"],
        }
        if code == "Not Available":
            na_ids.add(sid)
    return id_to_sample, na_ids


def static_flags(code: str) -> dict:
    if code in (None, "", "Not Available"):
        return {b: False for b in BEHAVIOR_VOCAB}
    ext = ASTFeatureExtractor(code)
    detected = ext.detect_behaviors()
    return {b: len(detected.get(b, [])) > 0 for b in BEHAVIOR_VOCAB}


def classification_metrics(results: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    n_fail = 0
    for r in results:
        if not r["parse_ok"]:
            n_fail += 1
            continue
        g, p = r["gold_label"], r["predicted_label"]
        if g == 1 and p == 1:
            tp += 1
        elif g == 1 and p == 0:
            fn += 1
        elif g == 0 and p == 1:
            fp += 1
        else:
            tn += 1
    mp = tp / (tp + fp) if (tp + fp) else 0.0
    mr = tp / (tp + fn) if (tp + fn) else 0.0
    bp = tn / (tn + fn) if (tn + fn) else 0.0
    br = tn / (tn + fp) if (tn + fp) else 0.0
    mf = 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0
    bf = 2 * bp * br / (bp + br) if (bp + br) else 0.0
    total = len(results)
    return {
        "n": total, "n_fail": n_fail,
        "coverage": (total - n_fail) / total if total else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "mal_precision": mp, "mal_recall": mr, "mal_f1": mf,
        "ben_precision": bp, "ben_recall": br, "ben_f1": bf,
        "macro_f1": (mf + bf) / 2,
        "balanced_accuracy": (mr + br) / 2,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
    }


def faithfulness_metrics(results, id_to_sample, na_ids):
    rows = []
    for r in results:
        sid = r["sample_id"]
        if sid in na_ids or not r["parse_ok"]:
            continue
        flags = static_flags(id_to_sample.get(sid, {}).get("code"))
        claimed = set()
        for b in r.get("behaviors", []):
            norm = normalize_claim(b.get("type", ""))
            if norm:
                claimed.add(norm)
        observed = {b for b in BEHAVIOR_VOCAB if flags.get(b)}
        supported = claimed & observed
        sup_rate = len(supported) / len(claimed) if claimed else None
        rec = len(supported & observed) / len(observed) if observed else None
        rows.append((claimed, observed, sup_rate, rec))

    p_vals = [x[2] for x in rows if x[2] is not None]
    r_vals = [x[3] for x in rows if x[3] is not None]
    return {
        "n_relevant_claims": sum(len(x[0]) for x in rows),
        "n_supported": sum(len(x[0] & x[1]) for x in rows),
        "n_samples_with_claims": len(p_vals),
        "unsupported_claim_rate":
            1 - sum(len(x[0] & x[1]) for x in rows) / sum(len(x[0]) for x in rows)
            if sum(len(x[0]) for x in rows) else 0.0,
        "behavior_precision":
            sum(x[2] for x in rows if x[2] is not None) / len(p_vals) if p_vals else 0.0,
        "behavior_recall":
            sum(x[3] for x in rows if x[3] is not None) / len(r_vals) if r_vals else 0.0,
    }


def err_overlap(results, id_to_sample, na_ids):
    always_fn, always_fp = {}, {}
    for r in results:
        sid = r["sample_id"]
        if sid in na_ids or not r["parse_ok"]:
            continue
        g, p = r["gold_label"], r["predicted_label"]
        if g == 1 and p == 0:
            always_fn.setdefault(sid, 0)
            always_fn[sid] += 1
        if g == 0 and p == 1:
            always_fp.setdefault(sid, 0)
            always_fp[sid] += 1
    return always_fn, always_fp


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def main():
    id_to_sample, na_ids = load_test_data()
    print(f"Test set: {len(id_to_sample)} samples, NA: {len(na_ids)}")
    print(f"  NA packages: {sorted(na_ids)}")

    out = {}
    fn_counts, fp_counts = Counter(), Counter()
    for method in METHODS:
        results = load_results(method)
        filtered = [r for r in results if r["sample_id"] not in na_ids]
        cls_all = classification_metrics(results)
        cls_fil = classification_metrics(filtered)
        faith = faithfulness_metrics(results, id_to_sample, na_ids)
        out[method] = {"unfiltered": cls_all, "filtered": cls_fil, "faithfulness": faith}
        # error overlap on filtered
        for r in filtered:
            if not r["parse_ok"]:
                continue
            if r["gold_label"] == 1 and r["predicted_label"] == 0:
                fn_counts[r["sample_id"]] += 1
            if r["gold_label"] == 0 and r["predicted_label"] == 1:
                fp_counts[r["sample_id"]] += 1
        print(f"{method:22s} | filtered F1_mal={fmt(cls_fil['mal_f1'])} "
              f"R_mal={fmt(cls_fil['mal_recall'])} P_mal={fmt(cls_fil['mal_precision'])} "
              f"F1_ben={fmt(cls_fil['ben_f1'])} BA={fmt(cls_fil['balanced_accuracy'])} "
              f"FPR={fmt(cls_fil['fpr'])} cov={fmt(cls_fil['coverage'])} "
              f"unsup={fmt(faith['unsupported_claim_rate'])}")

    print("\n=== FN overlap across methods (filtered) ===")
    for sid, n in fn_counts.most_common(10):
        print(f"  {n}/7  {sid}  ({id_to_sample.get(sid, {}).get('pkg', '?')})")
    print("=== FP overlap across methods (filtered) ===")
    for sid, n in fp_counts.most_common(10):
        print(f"  {n}/7  {sid}  ({id_to_sample.get(sid, {}).get('pkg', '?')})")

    out["fn_overlap"] = dict(fn_counts)
    out["fp_overlap"] = dict(fp_counts)
    out["na_ids"] = sorted(na_ids)

    out_path = REPORT_ROOT / "metrics.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    # LaTeX table
    tex_lines = [r"% generated by recompute_metrics.py"]
    tex_lines.append(r"\begin{tabular}{lrrrrrrrr}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r"Method & Mal. P & Mal. R & Mal. F1 & Ben. F1 & Macro F1 & Bal. Acc. & FPR & Cov. \\")
    tex_lines.append(r"\midrule")
    names = {
        "no_rag": r"No RAG (E0)", "dense": r"Dense (E1)", "bm25": r"BM25 (E2)",
        "hybrid": r"Hybrid (E3)", "behavior": r"Behavior (E4)",
        "contrastive": r"Contrastive (E5)", "behavior_contrastive": r"Beh.+Contr. (E6)",
    }
    for method in METHODS:
        c = out[method]["filtered"]
        tex_lines.append(
            f"{names[method]} & {fmt(c['mal_precision'])} & {fmt(c['mal_recall'])} & "
            f"{fmt(c['mal_f1'])} & {fmt(c['ben_f1'])} & {fmt(c['macro_f1'])} & "
            f"{fmt(c['balanced_accuracy'])} & {fmt(100*c['fpr'])} & {fmt(c['coverage'])} \\\\"
        )
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    (REPORT_ROOT / "tables.tex").write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"Wrote {REPORT_ROOT / 'tables.tex'}")


if __name__ == "__main__":
    main()
