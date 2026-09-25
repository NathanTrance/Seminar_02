import json
import math
import os
import time
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


JEV_BASE_URL = "https://www.jevai.org"
JEV_DECISIONS_PATH = "/api/v1/decisions"
JEV_MAX_BODY_BYTES = 32 * 1024
JEV_USER_AGENT = "ragsec-jev-demo/0.1"
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_JEV_MODEL = "typesafe/jev-1.13"


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def serialize_payload(payload: dict) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


class JevClient:
    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        transport: Callable[[Request, float], bytes] | None = None,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("JEV_API_KEY")
        self.timeout = timeout
        self.endpoint = f"{JEV_BASE_URL}{JEV_DECISIONS_PATH}"
        self._transport = transport or self._default_transport

    @staticmethod
    def _default_transport(request: Request, timeout: float) -> bytes:
        opener = build_opener(_NoRedirectHandler())
        with opener.open(request, timeout=timeout) as response:
            return response.read()

    def decide(self, payload: dict, question_id: str = "classification") -> dict:
        start = time.monotonic()

        if not self.api_key:
            return self._error_result(start, "missing_api_key", "JEV_API_KEY is not set")

        try:
            body = serialize_payload(payload)
        except (TypeError, ValueError) as exc:
            return self._error_result(start, "invalid_request", str(exc))

        if len(body) > JEV_MAX_BODY_BYTES:
            return self._error_result(
                start,
                "request_too_large",
                f"Serialized request is {len(body)} bytes; limit is {JEV_MAX_BODY_BYTES}",
            )

        request = Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": JEV_USER_AGENT,
            },
        )

        try:
            response_body = self._transport(request, self.timeout)
        except HTTPError as exc:
            message = self._http_error_message(exc)
            return self._error_result(
                start,
                "http_error",
                message,
                http_status=exc.code,
                network_attempted=True,
            )
        except (TimeoutError, URLError, OSError) as exc:
            return self._error_result(
                start,
                "transport_error",
                str(exc),
                network_attempted=True,
            )

        try:
            envelope = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self._error_result(
                start,
                "invalid_json",
                str(exc),
                network_attempted=True,
            )

        if not isinstance(envelope, dict):
            return self._error_result(
                start,
                "invalid_response",
                "Response is not an object",
                network_attempted=True,
            )

        if envelope.get("code") != 0:
            message = envelope.get("message")
            if not isinstance(message, str) or not message:
                message = "JEV returned a nonzero application code"
            return self._error_result(
                start,
                "application_error",
                message,
                raw_response=envelope,
                network_attempted=True,
            )

        parsed = self._parse_choice_answer(envelope, payload, question_id)
        if parsed["error"] is not None:
            return self._error_result(
                start,
                "invalid_response",
                parsed["error"],
                raw_response=envelope,
                network_attempted=True,
            )

        return {
            "status": "ok",
            "decision": parsed["decision"],
            "probabilities": parsed["probabilities"],
            "confidence": parsed["confidence"],
            "model": envelope.get("data", {}).get("model"),
            "usage": envelope.get("data", {}).get("usage"),
            "latency_ms": int((time.monotonic() - start) * 1000),
            "network_attempted": True,
            "http_status": 200,
            "error_type": None,
            "error": None,
            "raw_response": envelope,
        }

    @staticmethod
    def _parse_choice_answer(envelope: dict, payload: dict, question_id: str) -> dict:
        data = envelope.get("data")
        answers = data.get("answers") if isinstance(data, dict) else None
        answer = answers.get(question_id) if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            return {"error": f"Missing choice answer for {question_id!r}"}
        if answer.get("type") != "choice":
            return {"error": f"Answer for {question_id!r} is not a choice"}

        decision = answer.get("choice")
        probabilities = answer.get("probabilities")
        confidence = answer.get("confidence")
        questions = payload.get("questions")
        question = questions.get(question_id) if isinstance(questions, dict) else None
        criteria = question.get("criteria") if isinstance(question, dict) else None
        allowed = set(criteria) if isinstance(criteria, dict) else set()

        if not isinstance(decision, str) or decision not in allowed:
            return {"error": f"Unsupported choice for {question_id!r}: {decision!r}"}
        if not isinstance(probabilities, dict) or set(probabilities) != allowed:
            return {"error": f"Invalid probability keys for {question_id!r}"}
        if not all(JevClient._is_probability(value) for value in probabilities.values()):
            return {"error": f"Invalid probability values for {question_id!r}"}
        if not JevClient._is_probability(confidence):
            return {"error": f"Invalid confidence for {question_id!r}"}

        return {
            "error": None,
            "decision": decision,
            "probabilities": probabilities,
            "confidence": confidence,
        }

    @staticmethod
    def _is_probability(value) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0.0 <= value <= 1.0
        )

    @staticmethod
    def _http_error_message(exc: HTTPError) -> str:
        fallback = f"HTTP {exc.code}: {exc.reason}"
        try:
            body = exc.read()
            parsed = json.loads(body.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return fallback
        message = parsed.get("message") if isinstance(parsed, dict) else None
        return message if isinstance(message, str) and message else fallback

    @staticmethod
    def _error_result(
        start: float,
        error_type: str,
        error: str,
        http_status: int | None = None,
        raw_response: dict | None = None,
        network_attempted: bool = False,
    ) -> dict:
        return {
            "status": "error",
            "decision": None,
            "probabilities": None,
            "confidence": None,
            "model": None,
            "usage": None,
            "latency_ms": int((time.monotonic() - start) * 1000),
            "network_attempted": network_attempted,
            "http_status": http_status,
            "error_type": error_type,
            "error": error,
            "raw_response": raw_response,
        }


class OpenRouterJevClient(JevClient):
    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        transport: Callable[[Request, float], bytes] | None = None,
    ):
        self.api_key = (
            api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        )
        self.timeout = timeout
        self.endpoint = OPENROUTER_DECISIONS_URL
        self._transport = transport or self._default_transport

    def submit(self, payload: dict) -> dict:
        start = time.monotonic()

        if not self.api_key:
            return self._error_result(
                start,
                "missing_api_key",
                "OPENROUTER_API_KEY is not set",
            )

        try:
            body = serialize_payload(payload)
        except (TypeError, ValueError) as exc:
            return self._error_result(start, "invalid_request", str(exc))

        request = Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": JEV_USER_AGENT,
            },
        )

        try:
            response_body = self._transport(request, self.timeout)
        except HTTPError as exc:
            return self._error_result(
                start,
                "http_error",
                self._http_error_message(exc),
                http_status=exc.code,
                network_attempted=True,
            )
        except (TimeoutError, URLError, OSError) as exc:
            return self._error_result(
                start,
                "transport_error",
                str(exc),
                network_attempted=True,
            )

        try:
            response = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self._error_result(
                start,
                "invalid_json",
                str(exc),
                network_attempted=True,
            )

        if not isinstance(response, dict):
            return self._error_result(
                start,
                "invalid_response",
                "Response is not an object",
                network_attempted=True,
            )

        return {
            "status": "ok",
            "decision": None,
            "probabilities": None,
            "confidence": None,
            "model": response.get("model"),
            "usage": response.get("usage"),
            "latency_ms": int((time.monotonic() - start) * 1000),
            "network_attempted": True,
            "http_status": 200,
            "error_type": None,
            "error": None,
            "raw_response": response,
        }

    def decide(self, payload: dict, question_id: str = "classification") -> dict:
        result = self.submit(payload)
        if result["status"] != "ok":
            return result

        response = result["raw_response"]
        parsed = self._parse_choice_answer(
            {"data": {"answers": response.get("answers")}}, payload, question_id
        )
        if parsed["error"] is not None:
            result["status"] = "error"
            result["error_type"] = "invalid_response"
            result["error"] = parsed["error"]
            return result

        result["decision"] = parsed["decision"]
        result["probabilities"] = parsed["probabilities"]
        result["confidence"] = parsed["confidence"]
        return result
