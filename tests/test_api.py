"""Inference API tests with a canned predictor: no model, GPU or network."""

import threading
from contextlib import contextmanager

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.app import service  # noqa: E402
from src.data.prepare_dataset import review_id  # noqa: E402
from src.serving import api  # noqa: E402
from tests.test_app_service import CannedPredictor, answer  # noqa: E402

TOKEN = "t" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
REVIEW = "The screen is bright, but the battery life is disappointing."
INFO = {"model": "Qwen/Qwen2.5-3B-Instruct", "adapter": "checkpoint-195", "device": "cuda",
        "model_device": "cuda:0", "gpu": "Tesla T4", "precision": "CUDA GPU, 4-bit NF4"}


def responses():
    canned = {REVIEW: answer(("screen", "positive"), ("battery life", "negative"))}
    canned.update({text: "[]" for text in service.EXAMPLE_REVIEWS})
    return canned


class CountingLoader:
    def __init__(self, predictor=None, error=None, gate=None):
        self.predictor = predictor or CannedPredictor(responses())
        self.error, self.gate, self.calls = error, gate, 0

    def __call__(self):
        self.calls += 1
        if self.gate:
            self.gate.wait(5)
        if self.error:
            raise self.error
        return self.predictor, dict(INFO)


@contextmanager
def started(loader, run_warm_up=False):
    """A test client whose model has finished loading (or failing)."""
    app = api.create_app(loader, environ={"CFI_API_TOKEN": TOKEN}, run_warm_up=run_warm_up)
    with TestClient(app) as client:
        app.state.model.thread.join(5)
        yield client


@pytest.fixture
def client():
    with started(CountingLoader()) as client:
        yield client


# --- startup and security ---------------------------------------------------------


@pytest.mark.parametrize("environ", [{}, {"CFI_API_TOKEN": ""}, {"CFI_API_TOKEN": "short-token"}])
def test_service_refuses_to_start_without_a_strong_token(environ):
    with pytest.raises(RuntimeError, match="at least 32 characters"):
        api.create_app(CountingLoader(), environ=environ)


@pytest.mark.parametrize("path, method", [("/health", "get"), ("/predict", "post")])
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}])
def test_every_endpoint_requires_the_token(client, path, method, headers):
    response = getattr(client, method)(path, headers=headers, **({"json": {"text": REVIEW}} if method == "post" else {}))
    assert response.status_code == 401
    assert response.json() == {"detail": "Missing or invalid API token."}
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_api_documentation_is_not_published(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_default_bind_address_is_localhost():
    assert '"--host", default="127.0.0.1"' in open(api.__file__, encoding="utf-8").read()


# --- model lifecycle --------------------------------------------------------------


def test_health_reports_loading_then_ready_and_predict_waits():
    gate = threading.Event()
    loader = CountingLoader(gate=gate)
    app = api.create_app(loader, environ={"CFI_API_TOKEN": TOKEN}, run_warm_up=False)
    with TestClient(app) as client:
        assert client.get("/health", headers=AUTH).json()["status"] == "loading"
        busy = client.post("/predict", headers=AUTH, json={"text": REVIEW})
        assert busy.status_code == 503
        assert busy.headers["Retry-After"] == "30"
        assert "still loading" in busy.json()["detail"]

        gate.set()
        app.state.model.thread.join(5)
        health = client.get("/health", headers=AUTH).json()
        assert health["status"] == "ready"
        assert health["gpu"] == "Tesla T4" and health["model_device"] == "cuda:0"
        assert health["max_new_tokens"] == 256
        assert health["load_seconds"] >= 0


def test_model_is_loaded_once_for_all_requests():
    loader = CountingLoader()
    with started(loader) as client:
        for _ in range(3):
            assert client.post("/predict", headers=AUTH, json={"text": REVIEW}).status_code == 200
    assert loader.calls == 1
    assert len(loader.predictor.calls) == 3


def test_load_failure_is_reported_by_health_and_predict():
    with started(CountingLoader(error=RuntimeError("No suitable GPU"))) as client:
        health = client.get("/health", headers=AUTH).json()
        assert health["status"] == "error"
        assert health["error"] == "RuntimeError: No suitable GPU"
        response = client.post("/predict", headers=AUTH, json={"text": REVIEW})
        assert response.status_code == 503
        assert "failed to load: RuntimeError: No suitable GPU" in response.json()["detail"]


def test_warm_up_times_prefill_and_generation_before_ready():
    loader = CountingLoader()
    with started(loader, run_warm_up=True) as client:
        warm = client.get("/health", headers=AUTH).json()["warm_up"]
    assert set(warm) == {"first_request_seconds", "prefill_seconds", "full_generation_seconds",
                         "generated_tokens", "decode_seconds_per_token"}
    # A full predict, a 1-token prefill run, then a full generation; the limit is restored.
    assert len(loader.predictor.calls) == 3
    assert loader.predictor.max_new_tokens == 256


def test_cpu_is_refused_unless_requested_explicitly(monkeypatch):
    monkeypatch.setattr(service, "detect_cuda_memory_gb", lambda: None)
    with pytest.raises(RuntimeError, match="No suitable GPU: no CUDA GPU is available"):
        api.load_model({"CFI_DEVICE": "auto"})
    with pytest.raises(RuntimeError, match="No suitable GPU: the GPU has 2.0 GB"):
        monkeypatch.setattr(service, "detect_cuda_memory_gb", lambda: 2.0)
        api.load_model({})


# --- predictions --------------------------------------------------------------------


def test_predict_returns_the_predictor_output_with_server_details(client):
    response = client.post("/predict", headers=AUTH, json={"text": f"  {REVIEW} "})
    assert response.status_code == 200
    body = response.json()
    assert body["review_id"] == review_id(REVIEW)
    assert body["status"] == "valid"
    assert body["aspects"] == [
        {"aspect": "screen", "sentiment": "positive", "start": 4, "end": 10, "label_valid": True, "invented": False},
        {"aspect": "battery life", "sentiment": "negative", "start": 30, "end": 42, "label_valid": True,
         "invented": False},
    ]
    assert body["device"] == "cuda"
    assert body["max_new_tokens"] == 256
    assert body["server_seconds"] >= 0
    assert body["response"] == responses()[REVIEW]


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"text": "   "}, "Please enter a laptop review"),
        ({"text": "a" * 1501}, "please keep it under 1,500"),
        ({}, "Field required"),
        ({"text": 5}, "Input should be a valid string"),
        ({"text": REVIEW, "extra": 1}, "Extra inputs are not permitted"),
        ({"text": "a" * 20_001}, "at most 20000 characters"),
    ],
)
def test_invalid_requests_get_clear_422_errors(client, payload, message):
    response = client.post("/predict", headers=AUTH, json=payload)
    assert response.status_code == 422
    assert message in str(response.json()["detail"])


def test_inference_errors_return_500_with_the_cause():
    class Broken(CannedPredictor):
        def generate(self, texts):
            raise RuntimeError("CUDA error: device-side assert")

    with started(CountingLoader(predictor=Broken({}))) as client:
        response = client.post("/predict", headers=AUTH, json={"text": REVIEW})
    assert response.status_code == 500
    assert response.json()["detail"] == "Inference failed: RuntimeError: CUDA error: device-side assert"


def test_invalid_model_answers_are_returned_not_hidden():
    with started(CountingLoader(predictor=CannedPredictor({REVIEW: "not json"}))) as client:
        body = client.post("/predict", headers=AUTH, json={"text": REVIEW}).json()
    assert (body["status"], body["response"], body["aspects"]) == ("invalid_json", "not json", [])
    assert body["errors"]
