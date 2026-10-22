import json
from pathlib import Path

import pytest

import src.data.build_sft_dataset as build_module
from src.data.build_sft_dataset import (
    DataIntegrityError,
    build,
    check_split_integrity,
    load_jsonl,
    to_sft_example,
    validate_source_records,
    verify_examples,
    verify_manifest,
)
from src.data.prepare_dataset import LABELS, PROJECT_ROOT, SPLIT_NAMES, review_id, write_jsonl
from src.data.prompts import PROMPT_VERSION, SYSTEM_PROMPT

REAL_SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def make_record(text, *aspects, source_split="train"):
    """aspects: (term, sentiment) pairs, located left to right in the text."""
    located, cursor = [], 0
    for term, sentiment in aspects:
        start = text.index(term, cursor)
        located.append({"term": term, "start": start, "end": start + len(term), "sentiment": sentiment})
        cursor = start + len(term)
    return {"review_id": review_id(text), "text": text, "aspects": located, "source_split": source_split}


def make_splits():
    return {
        "train": [
            make_record("The screen is bright but the fan is loud.", ("screen", "positive"), ("fan", "negative")),
            make_record('The "sales" team never called back.', ('"sales" team', "negative")),
        ],
        "validation": [
            make_record("The touch-pad is ok, the touch-pad buttons are not.", ("touch-pad", "neutral"), ("touch-pad", "negative")),
        ],
        "test": [
            make_record("The price is high, yet the price is fair for what you get.", ("price", "conflict")),
        ],
    }


def write_splits(splits_dir: Path, splits):
    for name, records in splits.items():
        write_jsonl(records, splits_dir / f"{name}.jsonl")


@pytest.fixture
def built(tmp_path):
    splits_dir, output_dir = tmp_path / "splits", tmp_path / "sft"
    write_splits(splits_dir, make_splits())
    manifest = build(splits_dir, output_dir, labeled_split="train")
    return splits_dir, output_dir, manifest


def test_example_uses_prompt_completion_chat_format():
    record = make_splits()["train"][0]
    example = to_sft_example(record, "train")

    assert example["review_id"] == record["review_id"]
    assert example["split"] == "train"
    assert example["prompt_version"] == PROMPT_VERSION
    assert example["prompt"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Review: {record['text']}"},
    ]
    assert example["completion"] == [
        {
            "role": "assistant",
            "content": '[{"aspect": "screen", "sentiment": "positive"}, {"aspect": "fan", "sentiment": "negative"}]',
        }
    ]


def test_system_prompt_defines_all_four_labels():
    for label in LABELS:
        assert f"- {label}:" in SYSTEM_PROMPT


def test_build_writes_one_example_per_source_review(built):
    splits_dir, output_dir, manifest = built
    for name in SPLIT_NAMES:
        source = load_jsonl(splits_dir / f"{name}.jsonl")
        examples = load_jsonl(output_dir / f"{name}.jsonl")
        assert [e["review_id"] for e in examples] == [r["review_id"] for r in source]
        assert manifest["splits"][name]["examples"] == len(source)


def test_build_preserves_annotations_and_labels(built):
    splits_dir, output_dir, _ = built
    for name in SPLIT_NAMES:
        source = load_jsonl(splits_dir / f"{name}.jsonl")
        examples = load_jsonl(output_dir / f"{name}.jsonl")
        for record, example in zip(source, examples):
            answer = json.loads(example["completion"][0]["content"])
            assert answer == [{"aspect": a["term"], "sentiment": a["sentiment"]} for a in record["aspects"]]


def test_build_records_checksums_and_label_counts(built):
    _, _, manifest = built
    assert manifest["prompt_version"] == PROMPT_VERSION
    assert manifest["splits"]["train"]["aspect_labels"] == {"positive": 1, "negative": 2, "neutral": 0, "conflict": 0}
    assert all(len(entry["source_sha256"]) == 64 for entry in manifest["splits"].values())


def test_build_is_reproducible(built, tmp_path):
    splits_dir, output_dir, manifest = built
    second_dir = tmp_path / "sft_again"
    second = build(splits_dir, second_dir, labeled_split="train")

    assert second == manifest
    for name in [*SPLIT_NAMES, "manifest"]:
        suffix = ".json" if name == "manifest" else ".jsonl"
        assert (output_dir / f"{name}{suffix}").read_bytes() == (second_dir / f"{name}{suffix}").read_bytes()


def test_build_does_not_modify_source_splits(tmp_path):
    splits_dir = tmp_path / "splits"
    write_splits(splits_dir, make_splits())
    before = {p.name: p.read_bytes() for p in splits_dir.iterdir()}
    build(splits_dir, tmp_path / "sft", labeled_split="train")
    assert {p.name: p.read_bytes() for p in splits_dir.iterdir()} == before


def test_verify_manifest_detects_changed_source_split(built):
    splits_dir, output_dir, _ = built
    verify_manifest(output_dir / "manifest.json", splits_dir)

    path = splits_dir / "test.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("conflict", "positive"), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="test split changed"):
        verify_manifest(output_dir / "manifest.json", splits_dir)


def test_verify_manifest_detects_edited_sft_file(built):
    splits_dir, output_dir, _ = built
    path = output_dir / "train.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("positive", "negative"), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="modified after it was built"):
        verify_manifest(output_dir / "manifest.json", splits_dir)


def test_verify_manifest_detects_missing_split(built):
    splits_dir, output_dir, manifest = built
    del manifest["splits"]["test"]
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="manifest covers"):
        verify_manifest(output_dir / "manifest.json", splits_dir)


def test_failed_build_leaves_previous_output_untouched(built, monkeypatch):
    splits_dir, output_dir, _ = built
    before = {p.name: p.read_bytes() for p in output_dir.iterdir()}

    # Change train so a rebuild would rewrite it, then make only the last
    # split fail verification: a write-as-you-go build would already have
    # overwritten train.jsonl.
    write_jsonl([*make_splits()["train"], make_record("The hinge is weak.", ("hinge", "negative"))],
                splits_dir / "train.jsonl")
    real_verify = build_module.verify_examples

    def fail_on_test(records, examples, split):
        if split == "test":
            raise DataIntegrityError("simulated round-trip failure")
        return real_verify(records, examples, split)

    monkeypatch.setattr(build_module, "verify_examples", fail_on_test)
    with pytest.raises(DataIntegrityError, match="simulated"):
        build(splits_dir, output_dir, labeled_split="train")

    assert {p.name: p.read_bytes() for p in output_dir.iterdir()} == before


def test_same_review_in_two_splits_is_rejected():
    splits = make_splits()
    splits["test"].append(splits["train"][0])
    with pytest.raises(DataIntegrityError):
        check_split_integrity(splits)


def test_case_and_space_variant_across_splits_is_rejected():
    splits = make_splits()
    splits["test"].append(make_record("the screen is  bright but the fan is loud.", ("screen", "positive")))
    with pytest.raises(DataIntegrityError, match="both"):
        check_split_integrity(splits)


def test_duplicate_review_id_within_split_is_rejected():
    splits = make_splits()
    splits["train"].append(splits["train"][0])
    with pytest.raises(DataIntegrityError, match="twice"):
        check_split_integrity(splits)


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda r: r.update(source_split="test"), "source_split"),
        (lambda r: r["aspects"][0].update(sentiment="mixed"), "invalid sentiment"),
        (lambda r: r["aspects"][0].update(sentiment=None), "invalid sentiment"),
        (lambda r: r.update(aspects=[]), "no aspects"),
        (lambda r: r["aspects"][0].update(start=0), "offsets"),
        (lambda r: r["aspects"][0].pop("start"), "integer start/end"),
        (lambda r: r["aspects"][0].update(end="10"), "integer start/end"),
        (lambda r: r["aspects"][0].pop("term"), "term"),
        (lambda r: r.update(review_id="lap-0000000000"), "review_id"),
    ],
)
def test_invalid_source_records_are_rejected(corrupt, message):
    record = make_splits()["train"][0]
    corrupt(record)
    with pytest.raises(DataIntegrityError, match=message):
        validate_source_records([record], "train", labeled_split="train")


def test_verify_examples_rejects_missing_or_tampered_examples():
    records = make_splits()["train"]
    examples = [to_sft_example(r, "train") for r in records]

    with pytest.raises(DataIntegrityError, match="counts"):
        verify_examples(records, examples[:-1], "train")

    tampered = json.loads(json.dumps(examples))
    tampered[0]["completion"][0]["content"] = tampered[0]["completion"][0]["content"].replace("positive", "neutral")
    with pytest.raises(DataIntegrityError, match="round-trip"):
        verify_examples(records, tampered, "train")


@pytest.mark.skipif(not REAL_SPLITS_DIR.exists(), reason="prepared splits not available")
def test_real_splits_are_disjoint_and_every_reference_answer_round_trips():
    splits = {name: load_jsonl(REAL_SPLITS_DIR / f"{name}.jsonl") for name in SPLIT_NAMES}
    check_split_integrity(splits)

    grounded_elsewhere = 0
    for name, records in splits.items():
        validate_source_records(records, name, labeled_split="train")
        examples = [to_sft_example(r, name) for r in records]
        grounded_elsewhere += verify_examples(records, examples, name)

    # Only "keyboard" used as a verb before the annotated noun is ambiguous.
    assert grounded_elsewhere == 1
