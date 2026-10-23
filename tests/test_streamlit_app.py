"""Drive streamlit_app.py with Streamlit's AppTest and a canned predictor.

The model loader is replaced, so no weights are downloaded or loaded.
"""

import json

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.app import service  # noqa: E402
from src.data.prepare_dataset import PROJECT_ROOT  # noqa: E402
from src.modeling.inference import DEFAULT_MODEL  # noqa: E402
from tests.test_app_service import CannedPredictor, answer  # noqa: E402

APP = str(PROJECT_ROOT / "streamlit_app.py")
REVIEW = "The screen is bright, but the battery life is disappointing."


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    directory = tmp_path / "checkpoint-195"
    directory.mkdir()
    (directory / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": DEFAULT_MODEL, "peft_type": "LORA"}), encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("CFI_ADAPTER_PATH", str(directory))
    monkeypatch.setenv("CFI_DEVICE", "cpu")
    monkeypatch.setenv("CFI_INFERENCE_MODE", "local")
    monkeypatch.delenv("CFI_API_URL", raising=False)
    monkeypatch.delenv("CFI_API_TOKEN", raising=False)
    st.cache_resource.clear()
    st.cache_data.clear()
    yield directory
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def loads(monkeypatch):
    """Replace model loading with a canned predictor and record each load."""
    calls = []
    responses = {REVIEW: answer(("screen", "positive"), ("battery life", "negative"))}
    responses.update({text: "[]" for text in service.EXAMPLE_REVIEWS})

    def load(path, device):
        calls.append((path, device))
        return CannedPredictor(responses)

    monkeypatch.setattr(service, "load_predictor", load)
    return calls


def run_app():
    return AppTest.from_file(APP, default_timeout=30).run()


def analyze_button(app):
    return next(b for b in app.button if b.label == "Analyze Review")


def test_page_renders_title_and_runtime(adapter, loads):
    app = run_app()
    assert not app.exception
    assert app.title[0].value == "Customer Feedback Intelligence"
    sidebar = " ".join(m.value for m in app.sidebar.markdown)
    assert "CPU, bfloat16" in sidebar
    assert "85.55%" in sidebar
    assert loads == []  # the model is loaded only when a review is analyzed


def test_empty_review_shows_a_warning_without_loading_the_model(adapter, loads):
    app = run_app()
    analyze_button(app).click().run()
    assert [w.value for w in app.warning] == ["Please enter a laptop review to analyze."]
    assert loads == []


def test_example_button_fills_the_text_area(adapter, loads):
    app = run_app()
    next(b for b in app.button if b.label == "Try an example").click().run()
    assert app.text_area(key="review").value == service.EXAMPLE_REVIEWS[0]


def test_analysis_shows_counts_table_and_reuses_the_loaded_model(adapter, loads):
    app = run_app()
    app.text_area(key="review").input(REVIEW)
    analyze_button(app).click().run()
    assert not app.exception
    assert not app.error

    metrics = {m.label: m.value for m in app.metric}
    assert metrics == {"🟢 Positive": "1", "🔴 Negative": "1", "⚪ Neutral": "0", "🟠 Conflict": "0"}
    table = app.dataframe[0].value
    assert list(table["aspect"]) == ["screen", "battery life"]
    assert list(table["sentiment"]) == ["🟢 positive", "🔴 negative"]

    analyze_button(app).click().run()
    assert loads == [(adapter, "cpu")]


def test_invalid_model_output_shows_warning_and_raw_output(adapter, monkeypatch):
    monkeypatch.setattr(service, "load_predictor", lambda path, device: CannedPredictor({REVIEW: "not json"}))
    app = run_app()
    app.text_area(key="review").input(REVIEW)
    analyze_button(app).click().run()
    assert any("not valid JSON" in w.value for w in app.warning)
    assert any(e.label == "Raw model output" for e in app.expander)
    assert any(c.value == "not json" for c in app.code)


def test_load_failure_shows_a_friendly_error(adapter, monkeypatch):
    def fail(path, device):
        raise OSError("couldn't connect to huggingface.co")

    monkeypatch.setattr(service, "load_predictor", fail)
    app = run_app()
    app.text_area(key="review").input(REVIEW)
    analyze_button(app).click().run()
    assert not app.exception
    assert "could not be downloaded" in app.error[0].value


TOKEN = "t" * 40
READY = {"status": "ready", "gpu": "Tesla T4", "model_device": "cuda:0", "precision": "CUDA GPU, 4-bit NF4",
         "adapter": "checkpoint-195", "load_seconds": 95.0}


class FakeRemote:
    """Stands in for RemotePredictor; ``outcome`` is a prediction dict or an exception."""

    instances = []

    def __init__(self, settings, outcome=None, health=READY):
        self.settings, self.outcome, self._health, self.max_new_tokens = settings, outcome, health, 256
        FakeRemote.instances.append(self)

    def health(self):
        if isinstance(self._health, Exception):
            raise self._health
        return self._health

    def predict(self, records):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        canned = CannedPredictor({REVIEW: answer(("screen", "positive"), ("battery life", "negative"))})
        return [{**p, "server_seconds": 0.9} for p in canned.predict(records)]


@pytest.fixture
def remote_mode(adapter, loads, monkeypatch):
    from src.app import remote

    monkeypatch.delenv("CFI_INFERENCE_MODE")
    monkeypatch.setenv("CFI_API_URL", "https://abc.trycloudflare.com")
    monkeypatch.setenv("CFI_API_TOKEN", TOKEN)
    FakeRemote.instances = []

    def use(outcome=None, health=READY):
        monkeypatch.setattr(remote, "RemotePredictor", lambda settings: FakeRemote(settings, outcome, health))

    use()
    return use


def test_remote_mode_analyzes_on_the_api_and_never_loads_locally(remote_mode, loads):
    app = run_app()
    assert not app.exception
    sidebar = " ".join(m.value for m in app.sidebar.markdown)
    assert "Runs on:** Remote inference API" in sidebar
    assert "Qwen/Qwen2.5-3B-Instruct" in sidebar and "QLoRA" in sidebar and "85.55%" in sidebar
    assert app.sidebar.success[0].value == "Model ready on Tesla T4 (CUDA GPU, 4-bit NF4)"
    assert "Fine-tuned adapter loaded successfully." in [c.value for c in app.sidebar.caption]
    # Implementation details stay off the page.
    shown = " ".join(e.value for e in (*app.sidebar.markdown, *app.sidebar.caption, *app.sidebar.success))
    for hidden in ("trycloudflare", TOKEN, "cuda:0", "95.0", "loaded in"):
        assert hidden not in shown

    app.text_area(key="review").input(REVIEW)
    analyze_button(app).click().run()
    assert not app.error
    assert {m.label: m.value for m in app.metric}["🟢 Positive"] == "1"
    assert any("0.9 s on the server" in c.value for c in app.caption)
    assert loads == []


def test_remote_failure_is_shown_without_falling_back_to_cpu(remote_mode, loads):
    remote_mode(outcome=service.AppError("Cannot reach the inference API."))
    app = run_app()
    app.text_area(key="review").input(REVIEW)
    analyze_button(app).click().run()
    assert not app.exception
    assert app.error[0].value == "Cannot reach the inference API."
    assert not app.metric
    assert loads == []


def test_remote_status_problems_are_shown_in_the_sidebar(remote_mode):
    remote_mode(health=service.AppError("The inference API is unavailable (HTTP 530)."))
    app = run_app()
    assert "unavailable (HTTP 530)" in app.sidebar.warning[0].value
    assert not analyze_button(app).disabled  # the user can retry once the service is back

    st.cache_data.clear()
    st.cache_resource.clear()
    remote_mode(health={"status": "loading"})
    assert "still loading" in run_app().sidebar.info[0].value


def test_remote_mode_without_a_token_is_a_setup_error(remote_mode, monkeypatch):
    monkeypatch.delenv("CFI_API_TOKEN")
    app = run_app()
    assert any("CFI_API_TOKEN is not" in e.value for e in app.error)
    assert analyze_button(app).disabled


def test_missing_adapter_disables_analysis(tmp_path, monkeypatch, loads):
    monkeypatch.setenv("CFI_ADAPTER_PATH", str(tmp_path / "nowhere"))
    monkeypatch.setenv("CFI_DEVICE", "cpu")
    monkeypatch.setenv("CFI_INFERENCE_MODE", "local")
    app = run_app()
    assert any("Adapter folder not found" in e.value for e in app.error)
    assert analyze_button(app).disabled


# --- deployment defaults ------------------------------------------------------------------


@pytest.fixture
def no_local_model(adapter, loads, monkeypatch):
    """Remote mode by default; fail loudly if anything tries to set up local inference."""
    monkeypatch.delenv("CFI_INFERENCE_MODE")

    def forbidden(*args, **kwargs):
        raise AssertionError("local inference was attempted")

    monkeypatch.setattr(service, "resolve_runtime", forbidden)
    monkeypatch.setattr(service, "detect_cuda_memory_gb", forbidden)
    return loads


def test_without_remote_configuration_the_app_starts_but_never_runs_locally(no_local_model):
    app = run_app()
    assert not app.exception
    assert any("Remote GPU inference is not configured" in e.value for e in app.error)
    assert analyze_button(app).disabled
    assert no_local_model == []


def test_remote_settings_can_come_from_streamlit_secrets(no_local_model, monkeypatch):
    from src.app import remote

    monkeypatch.setattr(remote, "RemotePredictor", lambda settings: FakeRemote(settings))
    app = AppTest.from_file(APP, default_timeout=30)
    app.secrets["CFI_API_URL"] = "https://secret-host.trycloudflare.com"
    app.secrets["CFI_API_TOKEN"] = TOKEN
    app.run()
    assert not app.error
    assert "secret-host" not in " ".join(m.value for m in app.sidebar.markdown)

    app.text_area(key="review").input(REVIEW)
    analyze_button(app).click().run()
    assert {m.label: m.value for m in app.metric}["🔴 Negative"] == "1"
    assert FakeRemote.instances[-1].settings.token == TOKEN
    assert no_local_model == []


def test_unknown_inference_mode_is_a_setup_error(no_local_model, monkeypatch):
    monkeypatch.setenv("CFI_INFERENCE_MODE", "cpu")
    app = run_app()
    assert any("CFI_INFERENCE_MODE must be one of remote, local" in e.value for e in app.error)
    assert analyze_button(app).disabled
