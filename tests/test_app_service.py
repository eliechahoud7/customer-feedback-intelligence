"""Service tests with a canned predictor: no model, torch or Streamlit needed.

``CannedPredictor`` replaces only generation, so the real ``AbsaPredictor.predict``
and the project's parser run on every canned answer.
"""

import json

import pytest

from src.app import service
from src.app.service import AppError, Runtime
from src.data.prepare_dataset import PROJECT_ROOT, review_id
from src.modeling.inference import DEFAULT_MODEL, AbsaPredictor

REVIEW = "The screen is bright, but the battery life is disappointing."


class CannedPredictor(AbsaPredictor):
    def __init__(self, responses, truncated=False, max_new_tokens=256):
        self.responses, self.truncated, self.max_new_tokens, self.batch_size = responses, truncated, max_new_tokens, 1
        self.calls = []

    def generate(self, texts):
        self.calls.append(texts)
        return [{"response": self.responses[t], "generated_tokens": 17, "truncated": self.truncated} for t in texts]


def answer(*pairs):
    return json.dumps([{"aspect": a, "sentiment": s} for a, s in pairs])


def make_adapter(directory, base_model=DEFAULT_MODEL, peft_type="LORA"):
    directory.mkdir(parents=True, exist_ok=True)
    config = {"base_model_name_or_path": base_model, "peft_type": peft_type}
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"weights")
    return directory


def fake_clock(*times):
    values = iter(times)
    return lambda: next(values)


# --- configuration -----------------------------------------------------------


def test_adapter_path_defaults_to_project_models_folder():
    assert service.adapter_path({}) == PROJECT_ROOT / "models" / "checkpoint-195"
    assert service.adapter_path({"CFI_ADAPTER_PATH": "   "}) == PROJECT_ROOT / "models" / "checkpoint-195"


def test_adapter_path_from_environment(tmp_path):
    assert service.adapter_path({"CFI_ADAPTER_PATH": str(tmp_path)}) == tmp_path
    assert service.adapter_path({"CFI_ADAPTER_PATH": "adapters/run2"}) == PROJECT_ROOT / "adapters" / "run2"


def test_check_adapter_accepts_lora_adapter_for_the_base_model(tmp_path):
    service.check_adapter(make_adapter(tmp_path / "checkpoint-195"))


@pytest.mark.parametrize(
    "setup, message",
    [
        (lambda d: None, "Adapter folder not found"),
        (lambda d: (make_adapter(d), (d / "adapter_model.safetensors").unlink()), "missing adapter_model.safetensors"),
        (lambda d: make_adapter(d, base_model="Qwen/Qwen2.5-7B-Instruct"), "was trained on 'Qwen/Qwen2.5-7B-Instruct'"),
        (lambda d: make_adapter(d, peft_type="PREFIX_TUNING"), "is not a LoRA adapter"),
        (lambda d: (make_adapter(d), (d / "adapter_config.json").write_text("{", encoding="utf-8")), "Cannot read"),
    ],
)
def test_check_adapter_rejects_unusable_folders(tmp_path, setup, message):
    directory = tmp_path / "adapter"
    setup(directory)
    with pytest.raises(AppError, match=message):
        service.check_adapter(directory)


def test_auto_uses_cpu_without_cuda_or_with_a_small_gpu():
    assert service.choose_device("auto", lambda: None) == ("cpu", "no CUDA GPU is available to PyTorch")
    device, reason = service.choose_device("auto", lambda: 2.0)
    assert device == "cpu"
    assert "2.0 GB" in reason and "6 GB" in reason


def test_auto_uses_a_large_enough_gpu():
    assert service.choose_device("auto", lambda: 15.0)[0] == "cuda"
    assert service.choose_device("", lambda: 15.0)[0] == "cuda"


def test_explicit_cpu_does_not_probe_the_gpu():
    def probe():
        raise AssertionError("GPU probed")

    assert service.choose_device(" CPU ", probe)[0] == "cpu"


def test_explicit_cuda_is_honored_but_never_faked():
    assert service.choose_device("cuda", lambda: 2.0)[0] == "cuda"
    with pytest.raises(AppError, match="cannot see a CUDA GPU"):
        service.choose_device("cuda", lambda: None)


def test_unknown_device_is_rejected():
    with pytest.raises(AppError, match="CFI_DEVICE must be one of"):
        service.choose_device("tpu", lambda: None)


def test_resolve_runtime_reads_both_variables(tmp_path):
    runtime = service.resolve_runtime({"CFI_ADAPTER_PATH": str(tmp_path), "CFI_DEVICE": "cpu"}, lambda: 16.0)
    assert runtime == Runtime(tmp_path, "cpu", "CFI_DEVICE=cpu")
    assert "CPU" in runtime.description


def test_cpu_loads_without_quantization_in_bfloat16():
    assert service.load_options("cpu") == {"load_in_4bit": False, "dtype": "bfloat16", "device_map": "cpu"}
    # The GPU path keeps the existing 4-bit defaults of AbsaPredictor.
    assert service.load_options("cuda") == {"load_in_4bit": True, "dtype": None, "device_map": "auto"}
    with pytest.raises(ValueError):
        service.load_options("mps")


def test_load_predictor_checks_the_adapter_then_reuses_from_pretrained(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(AbsaPredictor, "from_pretrained", classmethod(lambda cls, *a, **k: calls.append((a, k))))

    with pytest.raises(AppError):
        service.load_predictor(tmp_path / "missing", "cpu")
    assert calls == []

    adapter = make_adapter(tmp_path / "checkpoint-195")
    service.load_predictor(adapter, "cpu")
    assert calls == [((DEFAULT_MODEL,), {"adapter_path": str(adapter), "max_new_tokens": 256, "batch_size": 1,
                                         "load_in_4bit": False, "dtype": "bfloat16", "device_map": "cpu"})]


# --- input and analysis --------------------------------------------------------


@pytest.mark.parametrize("text", [None, "", "   \n\t "])
def test_empty_review_is_rejected(text):
    with pytest.raises(AppError, match="Please enter a laptop review"):
        service.validate_review(text)


def test_long_review_is_rejected_and_text_is_stripped():
    with pytest.raises(AppError, match="under 1,500"):
        service.validate_review("a" * (service.MAX_REVIEW_CHARS + 1))
    assert service.validate_review("  Nice screen. \n") == "Nice screen."


def test_analyze_valid_answer():
    predictor = CannedPredictor({REVIEW: answer(("screen", "positive"), ("battery life", "negative"))})
    analysis = service.analyze(predictor, f"  {REVIEW}  ", clock=fake_clock(10.0, 12.5))

    assert predictor.calls == [[REVIEW]]
    assert analysis.review == REVIEW
    assert analysis.status == "valid"
    assert analysis.rows == [
        {"aspect": "screen", "sentiment": "positive", "note": ""},
        {"aspect": "battery life", "sentiment": "negative", "note": ""},
    ]
    assert analysis.counts == {"positive": 1, "negative": 1, "neutral": 0, "conflict": 0}
    assert analysis.warnings == []
    assert analysis.seconds == 2.5
    assert analysis.generated_tokens == 17
    assert analysis.segments == [
        ("The ", None), ("screen", "positive"), (" is bright, but the ", None),
        ("battery life", "negative"), (" is disappointing.", None),
    ]
    assert "".join(text for text, _ in analysis.segments) == REVIEW


def test_analyze_passes_the_review_id_to_predict(monkeypatch):
    predictor = CannedPredictor({REVIEW: "[]"})
    records = []
    original = AbsaPredictor.predict
    monkeypatch.setattr(AbsaPredictor, "predict", lambda self, r, **k: records.extend(r) or original(self, r, **k))
    service.analyze(predictor, REVIEW)
    assert records == [{"review_id": review_id(REVIEW), "text": REVIEW}]


def test_counts_cover_all_four_labels():
    review = "Screen good, keyboard bad, ports exist, speakers mixed."
    predictor = CannedPredictor({review: answer(("Screen", "positive"), ("keyboard", "negative"),
                                                ("ports", "neutral"), ("speakers", "conflict"))})
    assert service.analyze(predictor, review).counts == {"positive": 1, "negative": 1, "neutral": 1, "conflict": 1}


def test_review_without_aspects_is_valid_and_empty():
    analysis = service.analyze(CannedPredictor({"Bought it yesterday.": "[]"}), "Bought it yesterday.")
    assert (analysis.status, analysis.rows, analysis.warnings) == ("valid", [], [])
    assert analysis.counts == {"positive": 0, "negative": 0, "neutral": 0, "conflict": 0}
    assert analysis.segments == [("Bought it yesterday.", None)]


def test_invalid_json_keeps_raw_output_and_warns():
    raw = '[{"aspect": "screen", "sentiment": "positive"'
    analysis = service.analyze(CannedPredictor({REVIEW: raw}), REVIEW)
    assert analysis.status == "invalid_json"
    assert analysis.raw_response == raw
    assert analysis.rows == []
    assert analysis.errors
    assert any("not valid JSON" in w for w in analysis.warnings)


def test_invented_aspects_and_unknown_labels_are_shown_and_flagged_not_counted():
    predictor = CannedPredictor({REVIEW: answer(("screen", "positive"), ("webcam", "negative"), ("battery life", "bad"))})
    analysis = service.analyze(predictor, REVIEW)

    assert analysis.status == "malformed"
    assert analysis.rows == [
        {"aspect": "screen", "sentiment": "positive", "note": ""},
        {"aspect": "webcam", "sentiment": "negative", "note": "not found in review"},
        {"aspect": "battery life", "sentiment": "bad", "note": "unknown label"},
    ]
    # The invented aspect still has a valid label and is counted; the unknown label is not.
    assert analysis.counts == {"positive": 1, "negative": 1, "neutral": 0, "conflict": 0}
    assert any("1 predicted aspect(s) do not appear" in w for w in analysis.warnings)
    assert any("1 aspect(s) have a sentiment label other than" in w for w in analysis.warnings)
    assert any("did not fully follow" in w for w in analysis.warnings)
    # Only grounded, valid aspects are highlighted.
    assert [label for _, label in analysis.segments if label] == ["positive"]


def test_truncated_answer_warns():
    analysis = service.analyze(CannedPredictor({REVIEW: "[]"}, truncated=True, max_new_tokens=64), REVIEW)
    assert any("64-token limit" in w for w in analysis.warnings)


def test_analyze_rejects_empty_input_before_calling_the_model():
    predictor = CannedPredictor({})
    with pytest.raises(AppError):
        service.analyze(predictor, "  ")
    assert predictor.calls == []


# --- errors ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "error, expected",
    [
        (AppError("Adapter folder not found: x"), "Adapter folder not found: x"),
        (ModuleNotFoundError("No module named 'peft'", name="peft"), "A required package is missing (peft)"),
        (MemoryError(), "Not enough memory to run the model on CPU"),
        (RuntimeError("CUDA out of memory. Tried to allocate"), "Not enough memory"),
        (OSError("couldn't connect to huggingface.co"), "could not be downloaded"),
        (ValueError("boom"), "The model failed with ValueError: boom"),
    ],
)
def test_errors_are_described_for_users(error, expected):
    assert expected in service.describe_error(error, Runtime(PROJECT_ROOT, "cpu", ""))


def test_heavy_dependencies_are_not_imported_at_module_level():
    source = open(service.__file__, encoding="utf-8").read().split("def detect_cuda_memory_gb")[0]
    assert "import torch" not in source
    assert "streamlit" not in source.split('"""', 2)[2]
