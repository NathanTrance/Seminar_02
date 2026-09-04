import os
import time
import logging
from openai import OpenAI, DefaultHttpxClient

logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(self):
        self.base_url = os.environ.get(
            "EMBEDDING_BASE_URL",
            os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        )
        self.api_key = os.environ.get(
            "EMBEDDING_API_KEY",
            os.environ.get("OPENAI_API_KEY", "dummy"),
        )
        self.model = os.environ.get(
            "EMBEDDING_MODEL", "text-embedding-ada-002"
        )
        ssl_verify = os.environ.get("SSL_VERIFY", "true").lower() in ("true", "1", "yes")
        if ssl_verify:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                http_client=DefaultHttpxClient(verify=False),
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        start = time.time()
        max_chars = 6000
        truncated = [t[:max_chars] if t else "" for t in texts]
        try:
            response = self._client.embeddings.create(
                model=self.model, input=truncated
            )
            elapsed = time.time() - start
            embeddings = [item.embedding for item in response.data]
            logger.debug(
                "Embedded %d texts in %.2fs", len(texts), elapsed
            )
            return embeddings
        except Exception as e:
            elapsed = time.time() - start
            logger.error("Embedding call failed: %s", e)
            raise

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
