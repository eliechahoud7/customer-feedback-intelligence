"""Remote inference client tests: a fake HTTP session, plus the real API in-process."""

import pytest
import requests

from src.app import remote, service
from src.app.remote import RemotePredictor, RemoteSettings
from src.app.service import AppError
from src.data.prepare_dataset import review_id
from src.serving.benchmark import latency_summary, timed_requests, validation_report
from tests.test_app_service import CannedPredictor, answer

URL = "https://abc.trycloudflare.com"
TOKEN = "t" * 40
SETTINGS = RemoteSettings(URL, TOKEN)
REVIEW = "The screen is bright, but the battery life is disappointing."


class FakeResponse:
    def __init__(self, status_code, body=None, text=""):
        self.status_code, self.body, self.text = status_code, body, text

    def json(self):
        if self.body is None:
            raise ValueError("not JSON")
        return self.body


class FakeSession:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def prediction(text=REVIEW, **overrides):
    body = {
        "review_id": review_id(text), "prompt_version": "v1",
        "response": answer(("screen", "positive")), "generated_tokens": 12, "truncated": False,
        "status": "valid", "errors": [],
        "aspects": [{"aspect": "screen", "sentiment": "positive", "start": 4, "end": 10,
                     "label_valid": True, "invented": False}],
        "server_seconds": 0.8, "device": "cuda", "max_new_tokens": 256,
    }
    return {**body, **overrides}


def record(text=REVIEW):
    return {"review_id": review_id(text), "text": text}


# --- settings ---------------------------------------------------------------------


def test_local_mode_when_no_url_is_set():
    assert remote.remote_settings({}) is None
    assert remote.remote_settings({"CFI_API_URL": "  "}) is None


def test_settings_from_environment():
    settings = remote.remote_settings({"CFI_API_URL": f" {URL}/ ", "CFI_API_TOKEN": f" {TOKEN} "})
    assert settings == RemoteSettings(URL, TOKEN)
    assert settings.host == "abc.trycloudflare.com"
    assert TOKEN not in repr(settings.host)


@pytest.mark.parametrize(
    "environ, message",
    [
        ({"CFI_API_URL": URL}, "CFI_API_TOKEN is not"),
        ({"CFI_API_URL": "http://abc.trycloudflare.com", "CFI_API_TOKEN": TOKEN}, "must use https://"),
        ({"CFI_API_URL": "abc.trycloudflare.com", "CFI_API_TOKEN": TOKEN}, "must be a full URL"),
    ],
)
def test_unsafe_or_incomplete_settings_are_rejected(environ, message):
    with pytest.raises(AppError, match=message):
        remote.remote_settings(environ)


def test_missing_remote_configuration_explains_how_to_fix_it():
    with pytest.raises(AppError, match="Remote GPU inference is not configured.*CFI_INFERENCE_MODE=local"):
        remote.require_remote_settings({})
    assert remote.require_remote_settings({"CFI_API_URL": URL, "CFI_API_TOKEN": TOKEN}) == SETTINGS


def test_plain_http_is_allowed_only_for_localhost():
    for url in ("http://127.0.0.1:8000", "http://localhost:8000"):
        assert remote.remote_settings({"CFI_API_URL": url, "CFI_API_TOKEN": TOKEN}).url == url


# --- requests -----------------------------------------------------------------------


def test_predict_sends_the_review_with_the_bearer_token():
    session = FakeSession(FakeResponse(200, prediction()))
    predictor = RemotePredictor(SETTINGS, session)
    [result] = predictor.predict([record()])

    call = session.calls[0]
    assert (call["method"], call["url"]) == ("POST", f"{URL}/predict")
    assert call["json"] == {"text": REVIEW}
    assert call["headers"] == {"Authorization": f"Bearer {TOKEN}"}
    assert call["timeout"] == remote.PREDICT_TIMEOUT
    assert result == prediction()


def test_analyze_works_unchanged_with_the_remote_predictor():
    session = FakeSession(FakeResponse(200, prediction(max_new_tokens=128, truncated=True)))
    analysis = service.analyze(RemotePredictor(SETTINGS, session), REVIEW)
    assert analysis.rows == [{"aspect": "screen", "sentiment": "positive", "note": ""}]
    assert analysis.counts == {"positive": 1, "negative": 0, "neutral": 0, "conflict": 0}
    assert analysis.server_seconds == 0.8
    # The token limit reported by the server is used in the warning.
    assert any("128-token limit" in w for w in analysis.warnings)


def test_health_uses_get_and_a_short_timeout():
    session = FakeSession(FakeResponse(200, {"status": "ready"}))
    assert RemotePredictor(SETTINGS, session).health() == {"status": "ready"}
    assert session.calls[0]["method"] == "GET" and session.calls[0]["url"] == f"{URL}/health"
    assert session.calls[0]["timeout"] == remote.HEALTH_TIMEOUT


@pytest.mark.parametrize(
    "outcome, message",
    [
        (requests.ConnectionError("refused"), "Cannot reach the inference API at abc.trycloudflare.com"),
        (requests.Timeout("slow"), "did not answer within 120 s"),
        (FakeResponse(401, {"detail": "Missing or invalid API token."}), "rejected the token"),
        (FakeResponse(503, {"detail": "The model is still loading; retry shortly."}), "still loading"),
        (FakeResponse(503, {"detail": "The model failed to load: RuntimeError: No suitable GPU"}), "No suitable GPU"),
        (FakeResponse(422, {"detail": "Please enter a laptop review to analyze."}), "Please enter a laptop review"),
        (FakeResponse(422, {"detail": [{"msg": "Field required"}]}), "Field required"),
        (FakeResponse(500, {"detail": "Inference failed: RuntimeError: CUDA error"}), "HTTP 500): Inference failed"),
        (FakeResponse(530, None, "<html>Cloudflare error 1033</html>"), "unavailable (HTTP 530)"),
        (FakeResponse(502, None, "Bad gateway"), "Colab session or tunnel has probably stopped"),
        (FakeResponse(200, None, "<html>"), "non-JSON response"),
        (FakeResponse(200, {"status": "valid"}), "unexpected response"),
        (FakeResponse(200, prediction(aspects=[{"aspect": "screen"}])), "unexpected response"),
        (FakeResponse(200, prediction(review_id="lap-0000000000")), "different review text"),
    ],
)
def test_failures_become_clear_errors(outcome, message):
    predictor = RemotePredictor(SETTINGS, FakeSession(outcome))
    with pytest.raises(AppError, match=message.replace("(", r"\(").replace(")", r"\)")):
        predictor.predict([record()])


def test_errors_never_contain_the_token():
    for outcome in (requests.ConnectionError(f"failed {TOKEN}"), FakeResponse(401, {"detail": "no"})):
        with pytest.raises(AppError) as error:
            RemotePredictor(SETTINGS, FakeSession(outcome)).predict([record()])
        assert TOKEN not in str(error.value)


# --- the real API, in-process ------------------------------------------------------------


def test_client_and_api_work_together():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.serving import api

    canned = CannedPredictor({REVIEW: answer(("screen", "positive"), ("battery life", "negative"))})
    app = api.create_app(lambda: (canned, {"device": "cuda", "gpu": "Tesla T4"}),
                         environ={"CFI_API_TOKEN": TOKEN}, run_warm_up=False)
    with TestClient(app) as session:
        app.state.model.thread.join(5)
        predictor = RemotePredictor(RemoteSettings("http://testserver", TOKEN), session)
        assert predictor.health()["status"] == "ready"
        analysis = service.analyze(predictor, f" {REVIEW} ")

        wrong = RemotePredictor(RemoteSettings("http://testserver", "x" * 40), session)
        with pytest.raises(AppError, match="rejected the token"):
            wrong.predict([record()])

    assert analysis.status == "valid"
    assert analysis.counts == {"positive": 1, "negative": 1, "neutral": 0, "conflict": 0}
    assert analysis.server_seconds is not None


# --- benchmark helpers ---------------------------------------------------------------------


class ScriptedPredictor:
    def __init__(self, responses=None):
        self.responses = responses or {}

    def predict(self, records):
        return [prediction(r["text"], response=self.responses.get(r["text"], "[]"), server_seconds=0.5)
                for r in records]


def test_latency_is_reported_for_first_and_subsequent_requests():
    ticks = iter([0.0, 3.0, 10.0, 11.0, 20.0, 21.5])
    results = timed_requests(ScriptedPredictor(), ["a", "b", "c"], clock=lambda: next(ticks))
    summary = latency_summary(results)
    assert summary["first_request"]["client_seconds"] == 3.0
    assert summary["subsequent_requests"] == {
        "count": 2, "client_median_seconds": 1.25, "client_max_seconds": 1.5,
        "server_median_seconds": 0.5, "network_overhead_median_seconds": 0.75,
    }


def test_validation_report_scores_and_compares_with_the_reference():
    texts = ["The battery is great.", "The screen is dim."]
    records = [{"review_id": review_id(t), "text": t,
                "aspects": [{"term": term, "start": t.index(term), "end": t.index(term) + len(term), "sentiment": s}]}
               for t, (term, s) in zip(texts, [("battery", "positive"), ("screen", "negative")])]
    api_answers = {texts[0]: answer(("battery", "positive")), texts[1]: answer(("screen", "positive"))}
    reference = {records[0]["review_id"]: answer(("battery", "positive")),
                 records[1]["review_id"]: answer(("screen", "negative"))}

    report = validation_report(ScriptedPredictor(api_answers), records, reference)
    assert report["split"] == "validation"
    assert report["identical_to_reference"] == {"identical": 1, "reviews": 2}
    assert report["scores"]["aspect"]["f1"] == 1.0
    assert report["scores"]["aspect_sentiment"]["f1"] == 0.5
    assert report["reference_scores"]["aspect_sentiment"]["f1"] == 1.0


def test_benchmark_never_reads_the_test_split():
    from src.serving import benchmark

    source = open(benchmark.__file__, encoding="utf-8").read()
    assert '"validation.jsonl"' in source
    assert "test.jsonl" not in source and "final_test" not in source
