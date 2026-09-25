import io
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError

from src.ragsec.clients.jev import OpenRouterJevClient, JevClient
from src.ragsec.experiments.jev_demo import (
    build_decision_request,
    load_run,
    run_samples,
    save_run,
)
from src.ragsec.experiments.jev_full_test import (
    load_ease_test_samples,
    run_full_test,
)
from src.ragsec.experiments.jev_behavior_guided import (
    BEHAVIOR_THRESHOLD,
    build_behavior_request,
    build_guided_classification_request,
    parse_behavior_answers,
)


def success_response(choice="malicious") -> bytes:
    probabilities = {
        "malicious": 0.82,
        "benign": 0.05,
        "insufficient_evidence": 0.13,
    }
    if choice == "benign":
        probabilities = {
            "malicious": 0.04,
            "benign": 0.89,
            "insufficient_evidence": 0.07,
        }
    return json.dumps(
        {
            "code": 0,
            "message": "ok",
            "data": {
                "answers": {
                    "classification": {
                        "type": "choice",
                        "choice": choice,
                        "probabilities": probabilities,
                        "confidence": 0.84,
                    }
                }
            },
        }
    ).encode("utf-8")


def openrouter_success_response(choice="malicious") -> bytes:
    return json.dumps(
        {
            "id": "gen-dec-test",
            "model": "typesafe/jev-1.13-20260917",
            "provider": "TypeSafe",
            "answers": {
                "classification": {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": {
                        "malicious": 0.82,
                        "benign": 0.05,
                        "insufficient_evidence": 0.13,
                    },
                    "confidence": 0.84,
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.0001},
        }
    ).encode("utf-8")


class JevClientTests(unittest.TestCase):
    def test_parses_choice_response_and_sends_auth(self):
        captured = {}

        def transport(request, timeout):
            captured["authorization"] = request.get_header("Authorization")
            captured["user_agent"] = request.get_header("User-agent")
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return success_response()

        client = JevClient(api_key="test-key", timeout=4.5, transport=transport)
        result = client.decide(build_decision_request("print('hello')"))

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["decision"], "malicious")
        self.assertEqual(result["probabilities"]["malicious"], 0.82)
        self.assertEqual(result["confidence"], 0.84)
        self.assertTrue(result["network_attempted"])
        self.assertEqual(captured["authorization"], "Bearer test-key")
        self.assertEqual(captured["user_agent"], "ragsec-jev-demo/0.1")
        self.assertEqual(captured["url"], "https://www.jevai.org/api/v1/decisions")
        self.assertEqual(captured["timeout"], 4.5)

    def test_missing_key_is_an_error_without_network(self):
        calls = []
        client = JevClient(
            api_key="",
            transport=lambda request, timeout: calls.append(request),
        )

        result = client.decide(build_decision_request("pass"))

        self.assertEqual(result["error_type"], "missing_api_key")
        self.assertIsNone(result["decision"])
        self.assertFalse(result["network_attempted"])
        self.assertEqual(calls, [])

    def test_http_error_keeps_failure_separate_from_decision(self):
        def transport(request, timeout):
            raise HTTPError(
                request.full_url,
                401,
                "Unauthorized",
                {},
                io.BytesIO(b'{"message":"invalid key"}'),
            )

        result = JevClient(api_key="test", transport=transport).decide(
            build_decision_request("pass")
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "http_error")
        self.assertEqual(result["http_status"], 401)
        self.assertTrue(result["network_attempted"])
        self.assertEqual(result["error"], "invalid key")
        self.assertIsNone(result["decision"])

    def test_nonzero_application_code_is_an_error(self):
        body = json.dumps(
            {"code": 1201, "message": "invalid request", "data": None}
        ).encode("utf-8")
        client = JevClient(api_key="test", transport=lambda request, timeout: body)

        result = client.decide(build_decision_request("pass"))

        self.assertEqual(result["error_type"], "application_error")
        self.assertEqual(result["error"], "invalid request")
        self.assertIsNone(result["decision"])

    def test_invalid_json_is_an_error(self):
        client = JevClient(
            api_key="test",
            transport=lambda request, timeout: b"not-json",
        )
        result = client.decide(build_decision_request("pass"))
        self.assertEqual(result["error_type"], "invalid_json")

    def test_missing_answer_is_an_error(self):
        body = b'{"code":0,"message":"ok","data":{"answers":{}}}'
        client = JevClient(api_key="test", transport=lambda request, timeout: body)

        result = client.decide(build_decision_request("pass"))

        self.assertEqual(result["error_type"], "invalid_response")
        self.assertIn("Missing choice answer", result["error"])

    def test_unsupported_choice_is_an_error(self):
        body = success_response(choice="unknown")
        client = JevClient(api_key="test", transport=lambda request, timeout: body)

        result = client.decide(build_decision_request("pass"))

        self.assertEqual(result["error_type"], "invalid_response")
        self.assertIn("Unsupported choice", result["error"])
        self.assertIsNone(result["decision"])

    def test_rejects_oversized_utf8_request_before_network(self):
        calls = []
        client = JevClient(
            api_key="test",
            transport=lambda request, timeout: calls.append(request),
        )

        result = client.decide(build_decision_request("é" * 20000))

        self.assertEqual(result["error_type"], "request_too_large")
        self.assertEqual(calls, [])


class OpenRouterJevClientTests(unittest.TestCase):
    def test_parses_openrouter_choice_response(self):
        captured = {}

        def transport(request, timeout):
            captured["authorization"] = request.get_header("Authorization")
            captured["url"] = request.full_url
            return openrouter_success_response()

        result = OpenRouterJevClient(api_key="test-key", transport=transport).decide(
            build_decision_request("pass", model="typesafe/jev-1.13")
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["decision"], "malicious")
        self.assertEqual(result["model"], "typesafe/jev-1.13-20260917")
        self.assertEqual(result["usage"]["cost"], 0.0001)
        self.assertEqual(captured["authorization"], "Bearer test-key")
        self.assertEqual(captured["url"], "https://openrouter.ai/api/alpha/decisions")

    def test_missing_openrouter_key_is_an_error(self):
        result = OpenRouterJevClient(api_key="").decide(build_decision_request("pass"))

        self.assertEqual(result["error_type"], "missing_api_key")
        self.assertEqual(result["error"], "OPENROUTER_API_KEY is not set")

    def test_submit_preserves_typed_answers(self):
        body = json.dumps(
            {
                "model": "typesafe/jev-test",
                "answers": {"network_access": {"type": "noul", "noul": 0.75}},
                "usage": {"cost": 0.01},
            }
        ).encode("utf-8")
        client = OpenRouterJevClient(
            api_key="test", transport=lambda request, timeout: body
        )

        result = client.submit({"model": "typesafe/jev-test", "state": {}, "questions": {}})

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            result["raw_response"]["answers"]["network_access"]["noul"], 0.75
        )


class JevExperimentTests(unittest.TestCase):
    def test_request_contains_source_but_not_local_label_metadata(self):
        sample = {
            "sample_id": "local-only-id",
            "source_code": "pass",
            "provenance": "test",
            "expected_label": "benign",
        }

        run = run_samples([sample], dry_run=True)
        encoded = json.dumps(run["results"][0]["request"])

        self.assertIn("pass", encoded)
        self.assertNotIn("expected_label", encoded)
        self.assertNotIn("local-only-id", encoded)
        self.assertNotIn("benign\"", json.dumps(run["results"][0]["request"]["state"]))
        self.assertEqual(run["live_calls"], 0)
        self.assertEqual(run["results"][0]["response"]["status"], "dry_run")

    def test_live_request_limit_is_checked_before_calls(self):
        class CountingClient:
            calls = 0

            def decide(self, payload, question_id):
                self.calls += 1
                return {}

        client = CountingClient()
        samples = [
            {"sample_id": str(index), "source_code": "pass", "provenance": "test"}
            for index in range(3)
        ]

        with self.assertRaisesRegex(ValueError, "Refusing 3 live requests"):
            run_samples(samples, client=client, max_requests=2)

        self.assertEqual(client.calls, 0)

    def test_live_run_counts_network_attempts(self):
        client = JevClient(
            api_key="test",
            transport=lambda request, timeout: success_response(choice="benign"),
        )
        samples = [
            {"sample_id": "one", "source_code": "pass", "provenance": "test"},
            {"sample_id": "two", "source_code": "x = 1", "provenance": "test"},
        ]

        run = run_samples(samples, client=client)

        self.assertEqual(run["live_calls"], 2)
        self.assertEqual(
            [item["response"]["decision"] for item in run["results"]],
            ["benign", "benign"],
        )

    def test_saved_run_can_be_loaded_for_offline_replay(self):
        run = run_samples(
            [{"sample_id": "one", "source_code": "pass", "provenance": "test"}],
            dry_run=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_run(run, directory)
            loaded = load_run(path)

        self.assertEqual(loaded["mode"], "dry_run")
        self.assertEqual(loaded["results"][0]["sample_id"], "one")


class JevFullTestRunnerTests(unittest.TestCase):
    def test_loads_paper_ids_and_excludes_unavailable_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_malicious_packages_final.json").write_text(
                json.dumps([
                    {"package_name": "mal", "setup.py": "x = 1"},
                    {"package_name": "missing", "setup.py": "Not Available"},
                ]),
                encoding="utf-8",
            )
            (root / "test_benign_packages_final.json").write_text(
                json.dumps([{"package_name": "ben", "setup.py": "pass"}]),
                encoding="utf-8",
            )

            samples, excluded = load_ease_test_samples(root)

        self.assertEqual(
            [sample["sample_id"] for sample in samples],
            ["ease_test_malicious_00000", "ease_test_benign_00000"],
        )
        self.assertEqual(excluded[0]["sample_id"], "ease_test_malicious_00001")

    def test_writes_compatible_rows_and_resumes_completed_ids(self):
        class FakeClient:
            def decide(self, payload, question_id):
                return {
                    "status": "ok",
                    "decision": "malicious",
                    "probabilities": {"malicious": 1.0, "benign": 0.0},
                    "confidence": 1.0,
                    "model": "typesafe/jev-test",
                    "usage": {"input_tokens": 2, "output_tokens": 1, "cost": 0.01},
                    "latency_ms": 1,
                    "error": None,
                    "error_type": None,
                    "raw_response": {"answers": {}},
                }

        samples = [
            {"sample_id": "one", "gold_label": 1, "source_code": "pass"},
            {"sample_id": "two", "gold_label": 0, "source_code": "x = 1"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "jev.jsonl"
            first = run_full_test(samples, FakeClient(), "typesafe/jev-test", output)
            second = run_full_test(samples, FakeClient(), "typesafe/jev-test", output)
            rows = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(first["attempted"], 2)
        self.assertEqual(second["attempted"], 0)
        self.assertEqual([row["predicted_label"] for row in rows], [1, 1])
        self.assertTrue(all(row["parse_ok"] for row in rows))


class JevBehaviorGuidedTests(unittest.TestCase):
    def test_behavior_request_and_guided_request_are_two_stages(self):
        behavior_request = build_behavior_request("pass", "typesafe/jev-test")
        answers = {
            behavior: {"type": "noul", "noul": 0.75}
            for behavior in behavior_request["questions"]
        }
        probabilities = parse_behavior_answers(
            {"raw_response": {"answers": answers}}
        )
        guided = build_guided_classification_request(
            "pass", probabilities, "typesafe/jev-test"
        )

        self.assertNotIn("classification", behavior_request["questions"])
        self.assertEqual(set(guided["questions"]), {"classification"})
        self.assertTrue(
            guided["state"]["behavior_indicators"]["network_access"]["present"]
        )
        self.assertEqual(guided["state"]["behavior_threshold"], BEHAVIOR_THRESHOLD)

    def test_behavior_answer_requires_every_behavior(self):
        with self.assertRaisesRegex(ValueError, "Invalid noul answer"):
            parse_behavior_answers({"raw_response": {"answers": {}}})


if __name__ == "__main__":
    unittest.main()
