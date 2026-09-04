import os
import time
import logging
from openai import OpenAI, DefaultHttpxClient

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.base_url = os.environ.get(
            "OPENAI_BASE_URL", "http://localhost:8000/v1"
        )
        self.api_key = os.environ.get("OPENAI_API_KEY", "dummy")
        ssl_verify = os.environ.get("SSL_VERIFY", "true").lower() in ("true", "1", "yes")
        if ssl_verify:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                http_client=DefaultHttpxClient(verify=False),
            )

    def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict | None = None,
    ) -> dict:
        start = time.time()
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        try:
            response = self._client.chat.completions.create(**kwargs)
            elapsed = time.time() - start
            choice = response.choices[0]
            content = choice.message.content or ""
            usage = response.usage
            return {
                "content": content,
                "finish_reason": choice.finish_reason,
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
                "latency_ms": int(elapsed * 1000),
                "error": None,
            }
        except Exception as e:
            elapsed = time.time() - start
            logger.error("LLM call failed: %s", e)
            return {
                "content": "",
                "finish_reason": "error",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": int(elapsed * 1000),
                "error": str(e),
            }
