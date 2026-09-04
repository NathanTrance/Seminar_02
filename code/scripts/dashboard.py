#!/usr/bin/env python3
"""Build a self-contained HTML dashboard from experiment results."""
import json
import argparse
from pathlib import Path
import base64


def load_results(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_dataset_samples():
    import sys
    sys.path.insert(0, ".")
    from src.ragsec.datasets.ease_rag import EaseRagDataset
    ds = EaseRagDataset("data/raw/ease_rag")
    samples = ds.load("test")
    return {s.sample_id: s for s in samples}


def main():
    results_dir = Path("results/raw")
    out_path = "dashboard.html"

    methods = sorted(results_dir.glob("*Llama-3.1-8B-Instruct_ease_rag.jsonl"))
    all_data = {}
    for m in methods:
        name = m.stem.split("_Llama")[0]
        all_data[name] = load_results(m)

    sample_map = load_dataset_samples()

    cards_html = ""
    for method, results in sorted(all_data.items()):
        fps = [r for r in results if r["parse_ok"] and r["gold_label"] == 0 and r["predicted_label"] == 1 and not _is_na(r)]
        fns = [r for r in results if r["parse_ok"] and r["gold_label"] == 1 and r["predicted_label"] == 0 and not _is_na(r)]
        total_ok = sum(1 for r in results if r["parse_ok"])

        cards_html += f"""
        <div class="method-card">
            <div class="method-header" onclick="toggle('{method}')">
                <h2>{method}</h2>
                <span>FP:{len(fps)} FN:{len(fns)} OK:{total_ok}/{len(results)}</span>
            </div>
            <div id="{method}" class="method-body hidden">
        """

        if fps:
            cards_html += '<h3 class="fp">False Positives (benign predicted as malicious)</h3>'
            for r in fps:
                s = sample_map.get(r["sample_id"])
                code = (s.raw_code or "")[:2000] if s else ""
                cards_html += f"""
                <div class="sample fp">
                    <div class="sample-header">{r["sample_id"]} | conf={r["confidence"]:.2f}</div>
                    <div class="sample-behaviors">{json.dumps(r["behaviors"], indent=2)}</div>
                    <pre class="sample-code">{escape(code)}</pre>
                </div>
                """

        if fns:
            cards_html += '<h3 class="fn">False Negatives (malicious predicted as benign)</h3>'
            for r in fns:
                s = sample_map.get(r["sample_id"])
                code = (s.raw_code or "")[:2000] if s else ""
                cards_html += f"""
                <div class="sample fn">
                    <div class="sample-header">{r["sample_id"]} | conf={r["confidence"]:.2f}</div>
                    <div class="sample-behaviors">{json.dumps(r["behaviors"], indent=2)}</div>
                    <pre class="sample-code">{escape(code)}</pre>
                </div>
                """

        cards_html += "</div></div>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RAGsec Error Dashboard</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f5f5f5; padding:20px; }}
h1 {{ margin-bottom:20px; }}
.method-card {{ background:#fff; border-radius:8px; margin-bottom:12px; box-shadow:0 1px 3px rgba(0,0,0,0.1); overflow:hidden; }}
.method-header {{ padding:16px 20px; cursor:pointer; display:flex; justify-content:space-between; align-items:center; }}
.method-header:hover {{ background:#f0f0f0; }}
.method-header h2 {{ font-size:16px; }}
.method-body {{ padding:0 20px 20px; }}
.hidden {{ display:none; }}
h3 {{ margin:12px 0 8px; font-size:14px; }}
h3.fp {{ color:#d32f2f; }}
h3.fn {{ color:#f57c00; }}
.sample {{ margin:8px 0; padding:12px; border-radius:4px; font-size:13px; }}
.sample.fp {{ background:#fff5f5; border-left:3px solid #d32f2f; }}
.sample.fn {{ background:#fff8e1; border-left:3px solid #f57c00; }}
.sample-header {{ font-weight:600; margin-bottom:4px; }}
.sample-behaviors {{ font-family:monospace; font-size:11px; white-space:pre-wrap; margin-bottom:6px; color:#666; }}
.sample-code {{ background:#fafafa; border:1px solid #eee; border-radius:4px; padding:8px; font-size:11px; max-height:300px; overflow:auto; white-space:pre-wrap; word-break:break-all; }}
</style>
</head>
<body>
<h1>RAGsec — Error Analysis Dashboard</h1>
{cards_html}
<script>
function toggle(id) {{ document.getElementById(id).classList.toggle('hidden'); }}
</script>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"Dashboard written to {out_path}")


def escape(s):
    if not s: return ""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _is_na(r):
    return r.get("raw_response", "").strip() in ("", "Not Available") or r.get("error") is not None


if __name__ == "__main__":
    main()
