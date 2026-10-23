"""Client for the remote inference API (src/serving/api.py).

``RemotePredictor.predict`` has the same input and output as
``AbsaPredictor.predict``, so ``service.analyze`` works with either. Every
failure becomes an ``AppError`` with a message the user can act on; there is
no fallback to local inference.

Configuration (environment variables):
    CFI_API_URL    base URL of the API, e.g. https://<name>.trycloudflare.com;
                   when set, the app uses the API instead of a local model
    CFI_API_TOKEN  the API's bearer token
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

from src.app.service import MAX_NEW_TOKENS, AppError
from src.data.prepare_dataset import review_id

URL_ENV = "CFI_API_URL"
TOKEN_ENV = "CFI_API_TOKEN"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
PREDICTION_KEYS = ("review_id", "prompt_version", "response", "generated_tokens", "truncated",
                   "status", "errors", "aspects", "server_seconds", "device", "max_new_tokens")
ASPECT_KEYS = ("aspect", "sentiment", "start", "end", "label_valid", "invented")
HEALTH_TIMEOUT = 10
# A cold GPU request takes seconds; leave room for a slow tunnel.
PREDICT_TIMEOUT = 120


@dataclass(frozen=True)
class RemoteSettings:
    url: str
    token: str

    @property
    def host(self) -> str:
        return urlparse(self.url).netloc


def remote_settings(environ: Mapping[str, str] = os.environ) -> RemoteSettings | None:
    """The API settings, or None when CFI_API_URL is not set (local inference)."""
    url = environ.get(URL_ENV, "").strip().rstrip("/")
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise AppError(f"{URL_ENV} must be a full URL such as https://example.trycloudflare.com, not {url!r}.")
    # The token must never travel in clear text over the internet.
    if parsed.scheme == "http" and parsed.hostname not in LOCAL_HOSTS:
        raise AppError(f"{URL_ENV} must use https:// for a remote host ({parsed.netloc}).")
    token = environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise AppError(f"{URL_ENV} is set but {TOKEN_ENV} is not. Set it to the token the API was started with.")
    return RemoteSettings(url, token)


class RemotePredictor:
    def __init__(self, settings: RemoteSettings, session=None):
        self.settings = settings
        self.session = session or requests.Session()
        self.max_new_tokens = MAX_NEW_TOKENS

    def health(self) -> dict:
        return self._request("GET", "/health", timeout=HEALTH_TIMEOUT)

    def predict(self, records: list[dict], allow_code_fence: bool = False) -> list[dict]:
        if allow_code_fence:
            raise ValueError("the API parses answers strictly; allow_code_fence is not supported")
        return [self._predict_one(record) for record in records]

    def _predict_one(self, record: dict) -> dict:
        prediction = self._request("POST", "/predict", json={"text": record["text"]}, timeout=PREDICT_TIMEOUT)
        missing = [key for key in PREDICTION_KEYS if key not in prediction]
        if missing or any(key not in aspect for aspect in prediction.get("aspects", []) for key in ASPECT_KEYS):
            raise AppError(f"The inference API returned an unexpected response (missing {missing or 'aspect fields'}). "
                           "Make sure the Colab service runs the same version of this repository.")
        # The server strips the text before analyzing it, like the local path.
        if prediction["review_id"] != review_id(record["text"].strip()):
            raise AppError("The inference API answered for a different review text.")
        self.max_new_tokens = prediction["max_new_tokens"]
        return {**prediction, "review_id": record["review_id"]}

    def _request(self, method: str, path: str, timeout: float, **kwargs) -> dict:
        url = self.settings.url + path
        headers = {"Authorization": f"Bearer {self.settings.token}"}
        try:
            response = self.session.request(method, url, headers=headers, timeout=timeout, **kwargs)
        except requests.Timeout as error:
            raise AppError(f"The inference API at {self.settings.host} did not answer within {timeout} s.") from error
        except requests.RequestException as error:
            raise AppError(
                f"Cannot reach the inference API at {self.settings.host}. The Colab session may have stopped or the "
                f"tunnel URL may have changed: restart the service and update {URL_ENV}."
            ) from error

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as error:
                raise AppError(f"The inference API at {self.settings.host} returned a non-JSON response.") from error
        raise AppError(self._error_message(response))

    def _error_message(self, response) -> str:
        detail = _detail(response)
        code = response.status_code
        if code == 401:
            return f"The inference API rejected the token. Check that {TOKEN_ENV} matches the token in Colab."
        if code == 422:
            return detail or "The inference API rejected the review."
        if code == 503 and detail:
            return detail  # model loading or failed to load
        if code in (502, 503, 504, 530) or detail is None:
            return (f"The inference API at {self.settings.host} is unavailable (HTTP {code}). The Colab session or "
                    f"tunnel has probably stopped: restart the service and update {URL_ENV}.")
        return f"The inference API failed (HTTP {code}): {detail}"


def _detail(response) -> str | None:
    """FastAPI's error detail, or None for non-JSON responses (e.g. a tunnel error page)."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return None
    if isinstance(detail, list):  # request validation errors
        return "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail)
    return None if detail is None else str(detail)
