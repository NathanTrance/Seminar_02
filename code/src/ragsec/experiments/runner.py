import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from ..clients.llm import LLMClient
from ..datasets.base import Sample
from ..evaluation.classification import compute_classification_metrics
from ..evaluation.faithfulness import compute_faithfulness_metrics
from ..prompts.classifier import build_rag_messages, build_zero_shot_messages
from ..prompts.schemas import CLASSIFICATION_SCHEMA
from ..static.ast_features import ASTFeatureExtractor
from ..static.line_numbering import add_line_numbers
from ..utils.cache import Cache

logger = logging.getLogger(__name__)


def _repair_json(content: str) -> str | None:
    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        pass
    brace_match = re.search(r"\{.*", content, re.DOTALL)
    if not brace_match:
        return None
    candidate = brace_match.group(0)
    depth = 0
    for i, ch in enumerate(candidate):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                repaired = candidate[: i + 1]
                try:
                    json.loads(repaired)
                    return repaired
                except json.JSONDecodeError:
                    pass
    return None


class ExperimentRunner:
    def __init__(
        self,
        llm_client: LLMClient,
        cache: Cache,
        results_dir: str | Path = "results/raw",
        seed: int = 42,
    ):
        self.llm_client = llm_client
        self.cache = cache
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    def run_experiment(
        self,
        method: str,
        model: str,
        dataset_name: str,
        samples: list[Sample],
        experiment_id: str,
        retrieval_func=None,
        top_k_malicious: int = 3,
        top_k_benign: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 512,
        limit: int | None = None,
    ) -> list[dict]:
        if limit is not None:
            samples = samples[:limit]

        results = []
        for i, sample in enumerate(samples):
            logger.info(
                "[%s] %s/%s: %s",
                experiment_id,
                i + 1,
                len(samples),
                sample.sample_id,
            )

            result = self._process_sample(
                sample=sample,
                method=method,
                model=model,
                dataset_name=dataset_name,
                experiment_id=experiment_id,
                retrieval_func=retrieval_func,
                top_k_malicious=top_k_malicious,
                top_k_benign=top_k_benign,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            results.append(result)

        return results

    def _process_sample(
        self,
        sample: Sample,
        method: str,
        model: str,
        dataset_name: str,
        experiment_id: str,
        retrieval_func,
        top_k_malicious: int,
        top_k_benign: int,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        code = sample.raw_code or sample.behavior_text or ""
        if not code:
            return {
                "experiment_id": experiment_id,
                "dataset": dataset_name,
                "model": model,
                "sample_id": sample.sample_id,
                "gold_label": sample.label,
                "predicted_label": 0,
                "confidence": 0.0,
                "behaviors": [],
                "retrieved_ids": [],
                "retrieval_scores": [],
                "raw_response": "",
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "parse_ok": False,
                "error": "empty code",
            }

        numbered_code = add_line_numbers(code)

        retrieval_context = ""
        retrieved_ids = []
        retrieval_scores = []
        evidence_ids = None

        if retrieval_func and method != "no_rag":
            try:
                if method == "contrastive" or method == "behavior_contrastive":
                    query = self._get_query(sample, method)
                    mal_docs, ben_docs = retrieval_func(query, k_malicious=top_k_malicious, k_benign=top_k_benign) if hasattr(retrieval_func, 'retrieve') else retrieval_func(query, top_k_malicious, top_k_benign)
                    all_docs = mal_docs + ben_docs
                    retrieval_context = self._format_contrastive_context(mal_docs, ben_docs)
                else:
                    query = self._get_query(sample, method)
                    docs = retrieval_func(query, k=top_k_malicious) if hasattr(retrieval_func, '__call__') else retrieval_func.retrieve(query, k=top_k_malicious)
                    all_docs = docs
                    retrieval_context = self._format_context(docs)

                retrieved_ids = [d.doc_id for d in all_docs]
                retrieval_scores = [d.score for d in all_docs]
                evidence_ids = retrieved_ids
            except Exception as e:
                logger.warning("Retrieval failed for %s: %s", sample.sample_id, e)

        messages = self._build_messages(sample, method, numbered_code, retrieval_context)
        prompt_hash = self._hash_messages(messages)

        cache_key = self.cache.llm_cache_key(
            model=model,
            prompt_hash=prompt_hash,
            sample_hash=sample.sample_id,
            evidence_ids=evidence_ids,
            temperature=temperature,
        )

        cached = self.cache.get_llm(cache_key)
        if cached is not None:
            return self._build_result(
                sample, experiment_id, dataset_name, model,
                method, retrieved_ids, retrieval_scores,
                cached, True, cache_key
            )

        response = self.llm_client.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=CLASSIFICATION_SCHEMA,
        )

        if response["error"] is None:
            self.cache.set_llm(cache_key, response)

        return self._build_result(
            sample, experiment_id, dataset_name, model,
            method, retrieved_ids, retrieval_scores,
            response, False, cache_key
        )

    def _get_query(self, sample: Sample, method: str) -> str:
        if method in ("behavior", "behavior_contrastive"):
            if sample.behavior_text:
                return sample.behavior_text
            extractor = ASTFeatureExtractor(sample.raw_code or "")
            return extractor.to_behavior_text()
        return sample.raw_code or sample.behavior_text or ""

    def _build_messages(
        self, sample: Sample, method: str, numbered_code: str, retrieval_context: str
    ) -> list[dict]:
        if method == "no_rag":
            return build_zero_shot_messages(
                code=numbered_code, package_name=sample.package_name
            )
        contrastive = method in ("contrastive", "behavior_contrastive")
        return build_rag_messages(
            code=numbered_code,
            retrieved_context=retrieval_context,
            package_name=sample.package_name,
            contrastive=contrastive,
        )

    @staticmethod
    def _hash_messages(messages: list[dict]) -> str:
        import hashlib
        raw = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _format_context(docs) -> str:
        parts = []
        for doc in docs:
            tag = "MAL" if doc.pool == "malicious" else "BEN"
            parts.append(
                f"[{tag}_{doc.doc_id}] (score={doc.score:.3f})\n"
                f"{doc.content}\n"
            )
        return "\n---\n".join(parts) if parts else ""

    @staticmethod
    def _format_contrastive_context(mal_docs, ben_docs) -> str:
        parts = []
        if mal_docs:
            parts.append("=== MALICIOUS REFERENCES ===")
            for doc in mal_docs:
                parts.append(
                    f"[MAL_{doc.doc_id}] (score={doc.score:.3f})\n"
                    f"{doc.content}"
                )
        if ben_docs:
            parts.append("=== BENIGN REFERENCES ===")
            for doc in ben_docs:
                parts.append(
                    f"[BEN_{doc.doc_id}] (score={doc.score:.3f})\n"
                    f"{doc.content}"
                )
        return "\n---\n".join(parts)

    @staticmethod
    def _build_result(
        sample, experiment_id, dataset_name, model,
        method, retrieved_ids, retrieval_scores,
        response, from_cache, cache_key
    ) -> dict:
        content = response.get("content", "")
        error = response.get("error")

        predicted_label = 0
        confidence = 0.0
        behaviors = []

        if error is None and content:
            try:
                parsed_content = _repair_json(content) or content
                parsed = json.loads(parsed_content)
                verdict = parsed.get("verdict", "benign")
                predicted_label = 1 if verdict == "malicious" else 0
                confidence = float(parsed.get("confidence", 0.0))
                behaviors = parsed.get("behaviors", [])
            except (json.JSONDecodeError, ValueError) as e:
                error = str(e)

        result = {
            "experiment_id": experiment_id,
            "method": method,
            "dataset": dataset_name,
            "model": model,
            "sample_id": sample.sample_id,
            "gold_label": sample.label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "behaviors": behaviors,
            "retrieved_ids": retrieved_ids,
            "retrieval_scores": retrieval_scores,
            "raw_response": content,
            "latency_ms": response.get("latency_ms", 0),
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "parse_ok": error is None,
            "error": error,
            "from_cache": from_cache,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return result

    def save_results(self, results: list[dict], experiment_id: str) -> Path:
        out_path = self.results_dir / f"{experiment_id}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, default=str) + "\n")
        logger.info("Saved %d results to %s", len(results), out_path)
        return out_path

    def evaluate_results(self, results: list[dict]) -> dict:
        y_true = [r["gold_label"] for r in results]
        y_pred = [r["predicted_label"] for r in results]
        failure_mask = [not r["parse_ok"] for r in results]

        cls_metrics = compute_classification_metrics(y_true, y_pred, failure_mask)

        faithfulness_scores = []
        for r in results:
            fs = compute_faithfulness_metrics(
                code=r.get("raw_response", ""),
                response={"behaviors": r.get("behaviors", [])},
            )
            faithfulness_scores.append(fs)

        avg_faithfulness = {}
        if faithfulness_scores:
            avg_faithfulness = {
                "avg_unsupported_claim_rate": float(
                    sum(f["unsupported_claim_rate"] for f in faithfulness_scores)
                    / len(faithfulness_scores)
                ),
                "avg_behavior_precision": float(
                    sum(f["behavior_precision"] for f in faithfulness_scores)
                    / len(faithfulness_scores)
                ),
                "avg_behavior_recall": float(
                    sum(f["behavior_recall"] for f in faithfulness_scores)
                    / len(faithfulness_scores)
                ),
            }

        return {**cls_metrics, **avg_faithfulness}
