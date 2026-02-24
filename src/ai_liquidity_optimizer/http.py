from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib import error, parse, request


LOGGER = logging.getLogger(__name__)


class JsonHttpClient:
    def __init__(self, timeout_seconds: float = 15.0, max_retries: int = 2, backoff_seconds: float = 1.0):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def get_json(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        final_url = url
        if params:
            query = parse.urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)
            final_url = f"{url}?{query}"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = request.Request(final_url, headers=headers or {}, method="GET")
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    body = resp.read().decode("utf-8")
                    return json.loads(body)
            except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_for = self.backoff_seconds * (attempt + 1)
                LOGGER.warning("HTTP GET failed (%s). Retrying in %.1fs: %s", final_url, sleep_for, exc)
                time.sleep(sleep_for)

        assert last_error is not None
        raise RuntimeError(f"GET {final_url} failed: {last_error}") from last_error

