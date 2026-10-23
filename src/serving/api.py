"""GPU inference API for the Streamlit app (FastAPI).

The model is loaded once, in a background thread when the service starts,
through ``src.app.service.load_predictor``: the same ``AbsaPredictor``, prompt,
parser and 4-bit GPU configuration the app and the evaluation CLI use. After
loading, a warm-up request is run and timed (prompt prefill vs. token
generation) so the first user request does not pay one-time CUDA costs; the
timings are reported by /health.

Every endpoint requires ``Authorization: Bearer <CFI_API_TOKEN>``; the service
refuses to start without a token of at least 32 characters. It binds to
127.0.0.1 by default, so it is reachable from outside only through a tunnel.

Usage (Colab, GPU runtime):
    CFI_API_TOKEN=... CFI_ADAPTER_PATH=outputs/qwen2.5-3b-absa-qlora/checkpoint-195 \
        python -m src.serving.api --port 8000

Endpoints:
    GET  /health   {"status": "loading" | "ready" | "error", device, timings, ...}
    POST /predict  {"text": "<review>"} -> AbsaPredictor.predict output for the
                   review, plus server_seconds, device and max_new_tokens
"""

from __future__ import annotations

import argparse
import hmac
import os
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from src.app import service
from src.data.prepare_dataset import review_id
from src.data.prompts import PROMPT_VERSION
from src.modeling.inference import DEFAULT_MODEL, AbsaPredictor

TOKEN_ENV = "CFI_API_TOKEN"
MIN_TOKEN_LENGTH = 32
# Generous cap on the request body; validate_review enforces the real limit.
MAX_TEXT_CHARS = 20_000

Loader = Callable[[], tuple[AbsaPredictor, dict]]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=MAX_TEXT_CHARS)


class PredictedAspect(BaseModel):
    aspect: str
    sentiment: str
    start: int | None
    end: int | None
    label_valid: bool
    invented: bool


class PredictResponse(BaseModel):
    review_id: str
    prompt_version: str
    response: str
    generated_tokens: int
    truncated: bool
    status: str
    errors: list[str]
    aspects: list[PredictedAspect]
    server_seconds: float
    device: str
    max_new_tokens: int


def load_model(environ: Mapping[str, str] = os.environ) -> tuple[AbsaPredictor, dict]:
    """Load the predictor on the GPU (CPU only when CFI_DEVICE=cpu is set explicitly)."""
    runtime = service.resolve_runtime(environ)
    if runtime.device != "cuda" and environ.get(service.DEVICE_ENV, "").strip().lower() != "cpu":
        raise RuntimeError(
            f"No suitable GPU: {runtime.reason}. In Colab choose Runtime > Change runtime type > T4 GPU. "
            f"Set {service.DEVICE_ENV}=cpu only for local testing."
        )
    predictor = service.load_predictor(runtime.adapter_path, runtime.device)

    import torch

    return predictor, {
        "model": DEFAULT_MODEL,
        "adapter": runtime.adapter_path.name,
        "device": runtime.device,
        "model_device": str(predictor.model.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": runtime.description,
    }


def warm_up(predictor: AbsaPredictor, review: str = service.EXAMPLE_REVIEWS[0]) -> dict:
    """Run and time one review: a first full prediction, prefill alone, then a full generation."""
    clock = time.perf_counter
    started = clock()
    predictor.predict([{"review_id": review_id(review), "text": review}])
    first = clock() - started

    limit = predictor.max_new_tokens
    predictor.max_new_tokens = 1
    try:
        started = clock()
        predictor.generate([review])
        prefill = clock() - started
    finally:
        predictor.max_new_tokens = limit

    started = clock()
    tokens = predictor.generate([review])[0]["generated_tokens"]
    full = clock() - started
    return {
        "first_request_seconds": round(first, 3),
        "prefill_seconds": round(prefill, 3),
        "full_generation_seconds": round(full, 3),
        "generated_tokens": tokens,
        "decode_seconds_per_token": round((full - prefill) / max(tokens - 1, 1), 4),
    }


class ModelState:
    def __init__(self, loader: Loader, run_warm_up: bool = True):
        self.loader, self.run_warm_up = loader, run_warm_up
        self.status, self.error = "loading", None
        self.predictor: AbsaPredictor | None = None
        self.info: dict = {}
        self.timings: dict = {}
        # One model, one GPU: requests are generated one at a time.
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self.load, name="model-loader", daemon=True)
        self.thread.start()

    def load(self) -> None:
        try:
            started = time.perf_counter()
            predictor, info = self.loader()
            self.timings["load_seconds"] = round(time.perf_counter() - started, 1)
            if self.run_warm_up:
                with self.lock:
                    self.timings["warm_up"] = warm_up(predictor)
            self.predictor, self.info, self.status = predictor, info, "ready"
        except Exception as error:  # reported through /health and /predict
            self.status, self.error = "error", f"{type(error).__name__}: {error}"


def create_app(
    loader: Loader | None = None,
    environ: Mapping[str, str] = os.environ,
    run_warm_up: bool = True,
) -> FastAPI:
    token = environ.get(TOKEN_ENV, "")
    if len(token) < MIN_TOKEN_LENGTH:
        raise RuntimeError(
            f"Set {TOKEN_ENV} to a random secret of at least {MIN_TOKEN_LENGTH} characters, e.g. "
            "python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )
    expected = token.encode()
    state = ModelState(loader or (lambda: load_model(environ)), run_warm_up)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.start()
        yield

    app = FastAPI(title="Customer Feedback Intelligence API", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.model = state
    bearer = HTTPBearer(auto_error=False)

    def require_token(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if credentials is None or not hmac.compare_digest(credentials.credentials.encode(), expected):
            raise HTTPException(401, "Missing or invalid API token.", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health", dependencies=[Depends(require_token)])
    def health() -> dict:
        return {
            "status": state.status,
            "error": state.error,
            "prompt_version": PROMPT_VERSION,
            "max_new_tokens": state.predictor.max_new_tokens if state.predictor else None,
            **state.info,
            **state.timings,
        }

    @app.post("/predict", dependencies=[Depends(require_token)], response_model=PredictResponse)
    def predict(request: PredictRequest) -> dict:
        if state.status == "error":
            raise HTTPException(503, f"The model failed to load: {state.error}")
        if state.status != "ready":
            raise HTTPException(503, "The model is still loading; retry shortly.", headers={"Retry-After": "30"})
        try:
            review = service.validate_review(request.text)
        except service.AppError as error:
            raise HTTPException(422, str(error)) from error

        try:
            with state.lock:
                started = time.perf_counter()
                prediction = state.predictor.predict([{"review_id": review_id(review), "text": review}])[0]
                seconds = time.perf_counter() - started
        except Exception as error:
            raise HTTPException(500, f"Inference failed: {type(error).__name__}: {error}") from error
        return {**prediction, "server_seconds": round(seconds, 3), "device": state.info["device"],
                "max_new_tokens": state.predictor.max_new_tokens}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 and expose the port through a tunnel")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
