"""Application logic for the Streamlit app, kept free of Streamlit so it can be tested.

The app reuses ``AbsaPredictor`` for prompting, generation and parsing; this
module only decides where the model runs, checks the adapter and the input,
and turns one prediction into rows, counts and warnings for display.

Configuration (environment variables or Streamlit secrets):
    CFI_INFERENCE_MODE  remote (default): use the GPU inference API configured
                        by CFI_API_URL and CFI_API_TOKEN (src/app/remote.py);
                        local: load the model on this machine (opt-in, slow
                        on CPU, needs requirements-ml.txt)
    CFI_ADAPTER_PATH    local mode: LoRA adapter directory (default:
                        models/checkpoint-195, relative to the project root)
    CFI_DEVICE          local mode: auto (default), cpu or cuda

On ``auto`` the model runs 4-bit on a CUDA GPU with enough memory, and
otherwise on CPU in bfloat16 without quantization. The chosen device and the
reason are returned so the app can show them; ``cuda`` is never chosen for a
GPU below the memory threshold unless it is requested explicitly.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from src.data.prepare_dataset import LABELS, PROJECT_ROOT, review_id
from src.data.prompts import PROMPT_VERSION
from src.modeling.inference import DEFAULT_MODEL, AbsaPredictor

MODE_ENV = "CFI_INFERENCE_MODE"
MODES = ("remote", "local")
ADAPTER_ENV = "CFI_ADAPTER_PATH"
DEVICE_ENV = "CFI_DEVICE"
DEFAULT_ADAPTER = Path("models") / "checkpoint-195"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
DEVICES = ("auto", "cpu", "cuda")

# Qwen2.5-3B in 4-bit takes about 2 GB for the weights alone, so smaller GPUs
# (e.g. 2 GB laptop GPUs) fall back to CPU on "auto".
MIN_GPU_MEMORY_GB = 6.0
MAX_REVIEW_CHARS = 1500
MAX_NEW_TOKENS = 256

# Final evaluation of checkpoint-195 on the 222 held-out test reviews.
TEST_RESULTS = (
    ("Aspect F1", "85.55%"),
    ("Aspect + sentiment F1", "73.77%"),
    ("Sentiment accuracy", "86.23%"),
    ("Sentiment macro F1", "79.95%"),
    ("Valid structured responses", "222/222"),
)

EXAMPLE_REVIEWS = (
    "The screen is bright and the keyboard feels great, but the battery life is disappointing.",
    "Customer support was friendly, although the replacement charger took three weeks to arrive.",
    "It boots up fast and runs Windows smoothly, but the speakers are tinny and the fan gets loud.",
    "The price was fair for the hardware, and the trackpad is fine once you get used to it.",
)


class AppError(Exception):
    """A problem the user can fix; the message is shown as is."""


@dataclass(frozen=True)
class Runtime:
    adapter_path: Path
    device: str  # "cpu" or "cuda"
    reason: str

    @property
    def description(self) -> str:
        return "CUDA GPU, 4-bit NF4" if self.device == "cuda" else "CPU, bfloat16 (no quantization)"


@dataclass
class Analysis:
    review: str
    status: str
    rows: list[dict]
    counts: dict[str, int]
    warnings: list[str]
    raw_response: str
    errors: list[str]
    generated_tokens: int
    seconds: float
    prompt_version: str = PROMPT_VERSION
    segments: list[tuple[str, str | None]] = field(default_factory=list)
    # Model time reported by a remote inference API (``seconds`` includes the network).
    server_seconds: float | None = None


def inference_mode(environ: Mapping[str, str] = os.environ) -> str:
    """remote unless local inference is requested explicitly; never chosen as a fallback."""
    mode = environ.get(MODE_ENV, "").strip().lower() or "remote"
    if mode not in MODES:
        raise AppError(f"{MODE_ENV} must be one of {', '.join(MODES)}, not {mode!r}.")
    return mode


def adapter_path(environ: Mapping[str, str] = os.environ) -> Path:
    value = environ.get(ADAPTER_ENV, "").strip()
    path = Path(value).expanduser() if value else DEFAULT_ADAPTER
    return path if path.is_absolute() else PROJECT_ROOT / path


def check_adapter(path: Path, base_model: str = DEFAULT_MODEL) -> None:
    """Fail with a setup hint unless ``path`` holds a LoRA adapter for ``base_model``."""
    hint = f"Copy the trained adapter there or set {ADAPTER_ENV} to its folder (see README)."
    if not path.is_dir():
        raise AppError(f"Adapter folder not found: {path}. {hint}")
    missing = [name for name in ADAPTER_FILES if not (path / name).is_file()]
    if missing:
        raise AppError(f"Adapter folder {path} is missing {', '.join(missing)}. {hint}")
    try:
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AppError(f"Cannot read {path / 'adapter_config.json'}: {error}") from error
    if config.get("peft_type") != "LORA":
        raise AppError(f"{path} is not a LoRA adapter (peft_type={config.get('peft_type')!r}).")
    if config.get("base_model_name_or_path") != base_model:
        raise AppError(
            f"The adapter in {path} was trained on {config.get('base_model_name_or_path')!r}, not {base_model!r}."
        )


def detect_cuda_memory_gb() -> float | None:
    """Total memory of the first CUDA GPU, or None if PyTorch has no CUDA device."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def choose_device(requested: str, cuda_memory_gb: Callable[[], float | None]) -> tuple[str, str]:
    requested = requested.strip().lower() or "auto"
    if requested not in DEVICES:
        raise AppError(f"{DEVICE_ENV} must be one of {', '.join(DEVICES)}, not {requested!r}.")
    if requested == "cpu":
        return "cpu", f"{DEVICE_ENV}=cpu"
    memory = cuda_memory_gb()
    if requested == "cuda":
        if memory is None:
            raise AppError(f"{DEVICE_ENV}=cuda, but PyTorch cannot see a CUDA GPU (is the CPU-only build installed?).")
        return "cuda", f"{DEVICE_ENV}=cuda ({memory:.1f} GB GPU)"
    if memory is None:
        return "cpu", "no CUDA GPU is available to PyTorch"
    if memory < MIN_GPU_MEMORY_GB:
        return "cpu", f"the GPU has {memory:.1f} GB; 4-bit inference needs at least {MIN_GPU_MEMORY_GB:.0f} GB"
    return "cuda", f"{memory:.1f} GB CUDA GPU"


def resolve_runtime(
    environ: Mapping[str, str] = os.environ,
    cuda_memory_gb: Callable[[], float | None] | None = None,
) -> Runtime:
    device, reason = choose_device(environ.get(DEVICE_ENV, "auto"), cuda_memory_gb or detect_cuda_memory_gb)
    return Runtime(adapter_path(environ), device, reason)


def load_options(device: str) -> dict:
    """``AbsaPredictor.from_pretrained`` arguments for a device."""
    if device == "cuda":
        return {"load_in_4bit": True, "dtype": None, "device_map": "auto"}
    if device == "cpu":
        # bitsandbytes 4-bit is a GPU path; fp32 would need ~12 GB of RAM.
        return {"load_in_4bit": False, "dtype": "bfloat16", "device_map": "cpu"}
    raise ValueError(f"unknown device {device!r}")


def load_predictor(path: Path, device: str) -> AbsaPredictor:
    check_adapter(path)
    return AbsaPredictor.from_pretrained(
        DEFAULT_MODEL,
        adapter_path=str(path),
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=1,
        **load_options(device),
    )


def describe_error(error: BaseException, runtime: Runtime | None) -> str:
    """A short, actionable message for a failure while loading or running the model."""
    if isinstance(error, AppError):
        return str(error)
    if isinstance(error, ImportError):
        return (f"A required package is missing ({error.name or error}). "
                "Install the dependencies listed in the README into the project's .venv.")
    text = str(error)
    if isinstance(error, MemoryError) or "out of memory" in text.lower():
        device = runtime.device.upper() if runtime else "this machine"
        return (f"Not enough memory to run the model on {device}. "
                "Close other applications and retry, or set CFI_DEVICE=cpu if a small GPU was used.")
    if isinstance(error, OSError):
        return ("The base model could not be downloaded or read from the Hugging Face cache. "
                f"Check your internet connection for the first run. Details: {text}")
    return f"The model failed with {type(error).__name__}: {text}"


def validate_review(text: str | None) -> str:
    review = (text or "").strip()
    if not review:
        raise AppError("Please enter a laptop review to analyze.")
    if len(review) > MAX_REVIEW_CHARS:
        raise AppError(f"The review is {len(review):,} characters; please keep it under {MAX_REVIEW_CHARS:,}.")
    return review


def sentiment_counts(aspects: list[dict]) -> dict[str, int]:
    counts = Counter(aspect["sentiment"] for aspect in aspects if aspect["label_valid"])
    return {label: counts[label] for label in LABELS}


def highlight_segments(text: str, aspects: list[dict]) -> list[tuple[str, str | None]]:
    """Split the review into plain text and aspect spans labeled with their sentiment."""
    spans = sorted((a["start"], a["end"], a["sentiment"]) for a in aspects if not a["invented"] and a["label_valid"])
    segments, position = [], 0
    for start, end, sentiment in spans:
        if start > position:
            segments.append((text[position:start], None))
        segments.append((text[start:end], sentiment))
        position = end
    if position < len(text):
        segments.append((text[position:], None))
    return segments


def prediction_warnings(prediction: dict, max_new_tokens: int = MAX_NEW_TOKENS) -> list[str]:
    aspects = prediction["aspects"]
    warnings = []
    if prediction["status"] == "invalid_json":
        warnings.append("The model's answer was not valid JSON, so no aspects could be extracted.")
    elif prediction["status"] == "malformed":
        warnings.append("The model's answer did not fully follow the expected format; readable items are shown.")
    if prediction["truncated"]:
        warnings.append(f"The answer reached the {max_new_tokens}-token limit and may be incomplete.")
    invented = sum(a["invented"] for a in aspects)
    if invented:
        warnings.append(f"{invented} predicted aspect(s) do not appear in the review text.")
    invalid = sum(not a["label_valid"] for a in aspects)
    if invalid:
        warnings.append(f"{invalid} aspect(s) have a sentiment label other than {', '.join(LABELS)}.")
    return warnings


def aspect_rows(aspects: list[dict]) -> list[dict]:
    rows = []
    for aspect in aspects:
        notes = []
        if aspect["invented"]:
            notes.append("not found in review")
        if not aspect["label_valid"]:
            notes.append("unknown label")
        rows.append({"aspect": aspect["aspect"], "sentiment": aspect["sentiment"], "note": "; ".join(notes)})
    return rows


def analyze(predictor: AbsaPredictor, text: str, clock: Callable[[], float] = time.perf_counter) -> Analysis:
    review = validate_review(text)
    started = clock()
    prediction = predictor.predict([{"review_id": review_id(review), "text": review}])[0]
    seconds = clock() - started
    return Analysis(
        review=review,
        status=prediction["status"],
        rows=aspect_rows(prediction["aspects"]),
        counts=sentiment_counts(prediction["aspects"]),
        warnings=prediction_warnings(prediction, predictor.max_new_tokens),
        raw_response=prediction["response"],
        errors=prediction["errors"],
        generated_tokens=prediction["generated_tokens"],
        seconds=seconds,
        prompt_version=prediction["prompt_version"],
        segments=highlight_segments(review, prediction["aspects"]),
        server_seconds=prediction.get("server_seconds"),
    )
