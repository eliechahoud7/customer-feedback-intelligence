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
    st.cache_resource.clear()
    yield directory
    st.cache_resource.clear()


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


def test_missing_adapter_disables_analysis(tmp_path, monkeypatch, loads):
    monkeypatch.setenv("CFI_ADAPTER_PATH", str(tmp_path / "nowhere"))
    monkeypatch.setenv("CFI_DEVICE", "cpu")
    app = run_app()
    assert any("Adapter folder not found" in e.value for e in app.error)
    assert analyze_button(app).disabled
