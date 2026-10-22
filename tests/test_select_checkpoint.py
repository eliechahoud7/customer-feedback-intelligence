import json
import math
import random
from pathlib import Path

import pytest

import src.evaluation.select_checkpoint as select_module
from src.data.prepare_dataset import review_id, write_jsonl
from src.data.prompts import format_answer
from src.evaluation.select_checkpoint import (
    SELECTION_SPLIT,
    evaluate_checkpoints,
    find_checkpoints,
    inference_command,
    run_selection,
    select,
    validate_metrics,
)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def record(text, term, sentiment):
    start = text.index(term)
    return {"review_id": review_id(text), "text": text,
            "aspects": [{"term": term, "start": start, "end": start + len(term), "sentiment": sentiment}]}


VALIDATION = [
    record("The battery is really long.", "battery", "positive"),
    record("The screen is dim.", "screen", "negative"),
    record("Skip the cord.", "cord", "neutral"),
    record("The keyboard is great and awful.", "keyboard", "conflict"),
]
TEST = [record("The hinge broke.", "hinge", "negative")]


def answers(quality):
    """Responses per validation review: 'perfect', 'labels' (right terms, wrong labels) or 'empty'."""
    responses = {}
    for r in VALIDATION:
        aspects = r["aspects"]
        if quality == "labels":
            aspects = [{**a, "sentiment": "positive" if a["sentiment"] != "positive" else "negative"} for a in aspects]
        responses[r["review_id"]] = "[]" if quality == "empty" else format_answer(aspects)
    return responses


def write_predictions(path: Path, responses: dict):
    write_jsonl([{"review_id": rid, "response": text} for rid, text in responses.items()], path)


@pytest.fixture
def project(tmp_path):
    splits_dir, run_dir, eval_dir = tmp_path / "splits", tmp_path / "run", tmp_path / "eval"
    write_jsonl(VALIDATION, splits_dir / "validation.jsonl")
    write_jsonl(TEST, splits_dir / "test.jsonl")
    for step in (65, 130, 195):
        make_checkpoint(run_dir, step)
    (run_dir / "final").mkdir()
    (run_dir / "final" / "adapter_config.json").write_text("{}", encoding="utf-8")
    return splits_dir, run_dir, eval_dir


def make_checkpoint(run_dir: Path, step: int):
    path = run_dir / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": BASE_MODEL}), encoding="utf-8")
    return path


class FakeRunner:
    """Writes canned predictions where inference would, and records commands."""

    def __init__(self, quality_by_checkpoint):
        self.quality = quality_by_checkpoint
        self.commands = []

    def __call__(self, command):
        self.commands.append(command)
        checkpoint = Path(command[command.index("--adapter") + 1]).name
        write_predictions(Path(command[command.index("--output") + 1]), answers(self.quality[checkpoint]))


def summary(pair_f1, aspect_f1=0.5, macro=0.5, macro_wo=0.5):
    return {
        "aspect_sentiment": {"f1": pair_f1},
        "aspect": {"f1": aspect_f1},
        "sentiment": {"macro_f1": macro},
        "sentiment_without_conflict": {"macro_f1": macro_wo},
    }


def candidate(name, step, **metrics):
    return {"checkpoint": name, "step": step, "metrics": summary(**metrics)}


# --- Discovery -------------------------------------------------------------------------


def test_checkpoints_sorted_by_step_and_final_ignored(project):
    _, run_dir, _ = project
    make_checkpoint(run_dir, 1000)
    assert [p.name for _, p in find_checkpoints(run_dir)] == [
        "checkpoint-65", "checkpoint-130", "checkpoint-195", "checkpoint-1000",
    ]


def test_missing_run_dir_or_checkpoints_or_adapter_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        find_checkpoints(tmp_path / "nope")
    (tmp_path / "run").mkdir()
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        find_checkpoints(tmp_path / "run")
    (tmp_path / "run" / "checkpoint-5").mkdir()
    with pytest.raises(FileNotFoundError, match="adapter_config"):
        find_checkpoints(tmp_path / "run")


# --- Selection and ties -----------------------------------------------------------------------


def test_highest_pair_f1_wins():
    best = select([candidate("a", 65, pair_f1=0.6), candidate("b", 130, pair_f1=0.7), candidate("c", 195, pair_f1=0.65)])
    assert best["checkpoint"] == "b"


def test_tie_on_pair_f1_goes_to_higher_aspect_f1():
    best = select([candidate("a", 65, pair_f1=0.7, aspect_f1=0.8), candidate("b", 130, pair_f1=0.7, aspect_f1=0.9)])
    assert best["checkpoint"] == "b"


def test_full_tie_goes_to_earlier_checkpoint_in_any_input_order():
    candidates = [candidate(f"c{s}", s, pair_f1=0.7, aspect_f1=0.8) for s in (195, 65, 130)]
    for seed in range(5):
        random.Random(seed).shuffle(candidates)
        assert select(candidates)["step"] == 65


def test_all_zero_scores_still_select_deterministically():
    assert select([candidate("b", 130, pair_f1=0.0, aspect_f1=0.0),
                   candidate("a", 65, pair_f1=0.0, aspect_f1=0.0)])["step"] == 65


def test_no_candidates_is_an_error():
    with pytest.raises(ValueError, match="no candidates"):
        select([])


# --- Invalid metrics --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metrics, message",
    [
        ({"aspect": {"f1": 0.5}}, "aspect_sentiment.f1 is missing"),
        (summary(pair_f1=math.nan), "outside"),
        (summary(pair_f1=1.2), "outside"),
        (summary(pair_f1=-0.1), "outside"),
        (summary(pair_f1=math.inf), "outside"),
        (summary(pair_f1="0.7"), "not a number"),
        (summary(pair_f1=True), "not a number"),
        (summary(pair_f1=None), "not a number"),
        (summary(pair_f1=0.5, macro=float("nan")), "sentiment.macro_f1"),
    ],
)
def test_invalid_metrics_are_rejected_with_checkpoint_name(metrics, message):
    with pytest.raises(ValueError, match=message):
        validate_metrics(metrics, "checkpoint-65")
    with pytest.raises(ValueError, match="checkpoint-65"):
        select([{"checkpoint": "checkpoint-65", "step": 65, "metrics": metrics}])


def test_one_invalid_candidate_stops_selection():
    good = candidate("checkpoint-65", 65, pair_f1=0.9)
    bad = {"checkpoint": "checkpoint-130", "step": 130, "metrics": summary(pair_f1=math.nan)}
    with pytest.raises(ValueError, match="checkpoint-130"):
        select([good, bad])


# --- Generation, reuse and missing results ---------------------------------------------------------


def test_generates_validation_predictions_and_selects_best(project):
    splits_dir, run_dir, eval_dir = project
    runner = FakeRunner({"checkpoint-65": "empty", "checkpoint-130": "perfect", "checkpoint-195": "labels"})
    report = run_selection(run_dir, eval_dir, splits_dir, eval_dir / "selection.json", runner=runner)

    assert report["selected"]["checkpoint"] == "checkpoint-130"
    assert report["selected"]["metrics"]["aspect_sentiment.f1"] == 1.0
    scores = {c["checkpoint"]: c["metrics"]["aspect_sentiment"]["f1"] for c in report["candidates"]}
    assert scores == {"checkpoint-65": 0.0, "checkpoint-130": 1.0, "checkpoint-195": 0.0}
    assert json.loads((eval_dir / "selection.json").read_text(encoding="utf-8")) == report


def test_inference_is_run_on_validation_only(project):
    splits_dir, run_dir, eval_dir = project
    runner = FakeRunner({f"checkpoint-{s}": "perfect" for s in (65, 130, 195)})
    run_selection(run_dir, eval_dir, splits_dir, eval_dir / "selection.json", runner=runner)

    assert len(runner.commands) == 3
    for command in runner.commands:
        assert command[command.index("--split") + 1] == "validation"
        assert "--final-test" not in command
        assert "test" not in Path(command[command.index("--output") + 1]).name
        assert command[command.index("--model") + 1] == BASE_MODEL


def test_inference_command_targets_validation(tmp_path):
    checkpoint = make_checkpoint(tmp_path, 65)
    command = inference_command(checkpoint, tmp_path / "out.jsonl", batch_size=4, max_new_tokens=128)

    assert SELECTION_SPLIT == "validation"
    assert command[1:3] == ["-m", "src.modeling.inference"]
    assert command[command.index("--split") + 1] == "validation"
    assert "--final-test" not in command
    assert command[command.index("--adapter") + 1] == str(checkpoint)
    assert command[command.index("--model") + 1] == BASE_MODEL
    assert command[command.index("--batch-size") + 1] == "4"
    assert command[command.index("--max-new-tokens") + 1] == "128"


def test_test_split_is_never_read(project, monkeypatch):
    splits_dir, run_dir, eval_dir = project
    opened = []
    real_load = select_module.load_jsonl
    monkeypatch.setattr(select_module, "load_jsonl", lambda p: opened.append(Path(p).name) or real_load(p))
    runner = FakeRunner({f"checkpoint-{s}": "perfect" for s in (65, 130, 195)})
    run_selection(run_dir, eval_dir, splits_dir, eval_dir / "selection.json", runner=runner)

    assert opened == ["validation.jsonl"]


def test_predictions_for_another_split_are_rejected(project):
    splits_dir, run_dir, eval_dir = project
    for step in (65, 130, 195):
        write_predictions(eval_dir / f"checkpoint-{step}_validation.jsonl",
                          {TEST[0]["review_id"]: format_answer(TEST[0]["aspects"])})
    with pytest.raises(ValueError, match="do not match the split"):
        run_selection(run_dir, eval_dir, splits_dir, eval_dir / "selection.json", generate=False)


def test_score_only_with_missing_predictions_names_them_and_runs_nothing(project):
    splits_dir, run_dir, eval_dir = project
    write_predictions(eval_dir / "checkpoint-65_validation.jsonl", answers("perfect"))
    runner = FakeRunner({})
    with pytest.raises(FileNotFoundError, match=r"\['checkpoint-130', 'checkpoint-195'\]"):
        evaluate_checkpoints(run_dir, eval_dir, VALIDATION, generate=False, runner=runner)
    assert runner.commands == []


def test_existing_predictions_are_reused_unless_regenerate(project):
    splits_dir, run_dir, eval_dir = project
    for step in (65, 130, 195):
        write_predictions(eval_dir / f"checkpoint-{step}_validation.jsonl", answers("empty"))
    runner = FakeRunner({f"checkpoint-{s}": "perfect" for s in (65, 130, 195)})

    reused = evaluate_checkpoints(run_dir, eval_dir, VALIDATION, runner=runner)
    assert runner.commands == []
    assert all(c["metrics"]["aspect_sentiment"]["f1"] == 0.0 for c in reused)

    regenerated = evaluate_checkpoints(run_dir, eval_dir, VALIDATION, regenerate=True, runner=runner)
    assert len(runner.commands) == 3
    assert all(c["metrics"]["aspect_sentiment"]["f1"] == 1.0 for c in regenerated)


def test_inference_that_writes_nothing_is_an_error(project):
    splits_dir, run_dir, eval_dir = project
    with pytest.raises(FileNotFoundError, match="did not write"):
        evaluate_checkpoints(run_dir, eval_dir, VALIDATION, runner=lambda command: None)


# --- Reproducibility --------------------------------------------------------------------------------------


def test_report_is_byte_identical_for_the_same_predictions(project):
    splits_dir, run_dir, eval_dir = project
    runner = FakeRunner({"checkpoint-65": "labels", "checkpoint-130": "perfect", "checkpoint-195": "perfect"})
    first = run_selection(run_dir, eval_dir, splits_dir, eval_dir / "first.json", runner=runner)
    second = run_selection(run_dir, eval_dir, splits_dir, eval_dir / "second.json", generate=False)

    assert first == second
    assert (eval_dir / "first.json").read_bytes() == (eval_dir / "second.json").read_bytes()
    # checkpoint-130 and checkpoint-195 tie on every metric; the earlier one wins.
    assert first["selected"]["checkpoint"] == "checkpoint-130"
    assert all(len(c["predictions_sha256"]) == 64 for c in first["candidates"])
    assert len(first["validation_split_sha256"]) == 64
