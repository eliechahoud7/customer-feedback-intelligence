"""Build instruction-tuning (SFT) examples from the prepared splits.

Each review in data/splits/{train,validation,test}.jsonl becomes one example in
TRL's conversational prompt/completion format, so the training loss covers only
the assistant's JSON answer. The split files are the source of truth: they are
read, never re-split or modified, and their checksums are recorded in a
manifest that training and evaluation can verify before running.

Usage:
    python -m src.data.build_sft_dataset [--config configs/data.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.data.prepare_dataset import (
    DEFAULT_CONFIG,
    LABELS,
    PROJECT_ROOT,
    SPLIT_NAMES,
    assert_no_leakage,
    load_config,
    review_id,
    write_jsonl,
)
from src.data.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_prompt, format_answer
from src.evaluation.parsing import parse_and_match


class DataIntegrityError(ValueError):
    pass


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source_records(records: list[dict], split: str, labeled_split: str) -> None:
    """Check that every source record is a well-formed, labeled review."""
    for record in records:
        rid = record.get("review_id")
        text = record.get("text")
        if not isinstance(text, str) or not text:
            raise DataIntegrityError(f"{split}/{rid}: missing review text")
        if rid != review_id(text):
            raise DataIntegrityError(f"{split}/{rid}: review_id does not match the review text")
        # Only the labeled source split may feed supervised training or evaluation.
        if record.get("source_split") != labeled_split:
            raise DataIntegrityError(
                f"{split}/{rid}: source_split is {record.get('source_split')!r}, expected {labeled_split!r}"
            )
        aspects = record.get("aspects")
        if not aspects:
            raise DataIntegrityError(f"{split}/{rid}: no aspects")
        for aspect in aspects:
            term, start, end = aspect.get("term"), aspect.get("start"), aspect.get("end")
            if not isinstance(term, str) or not term or not isinstance(start, int) or not isinstance(end, int):
                raise DataIntegrityError(f"{split}/{rid}: aspect needs a term and integer start/end: {aspect}")
            if aspect.get("sentiment") not in LABELS:
                raise DataIntegrityError(f"{split}/{rid}: invalid sentiment {aspect.get('sentiment')!r}")
            if text[start:end] != term:
                raise DataIntegrityError(f"{split}/{rid}: offsets do not match term {term!r}")


def check_split_integrity(splits: dict[str, list[dict]]) -> None:
    """Review IDs are unique within a split; IDs and texts never cross splits."""
    seen_ids: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            rid = record["review_id"]
            if rid in seen_ids:
                where = "twice in" if seen_ids[rid] == name else f"in both {seen_ids[rid]} and"
                raise DataIntegrityError(f"review_id {rid} appears {where} {name}")
            seen_ids[rid] = name
    try:
        assert_no_leakage(splits)
    except ValueError as error:
        raise DataIntegrityError(str(error)) from error


def to_sft_example(record: dict, split: str) -> dict:
    return {
        "review_id": record["review_id"],
        "split": split,
        "prompt_version": PROMPT_VERSION,
        "prompt": build_prompt(record["text"]),
        "completion": [{"role": "assistant", "content": format_answer(record["aspects"])}],
    }


def verify_examples(records: list[dict], examples: list[dict], split: str) -> int:
    """Examples mirror the source one-to-one and every answer parses back exactly.

    The answer carries terms, not offsets, so a term that occurs several times
    in a review (e.g. "keyboard" used once as a verb) may be grounded to a
    different occurrence than the annotated one. Such cases are counted and
    returned, not treated as errors; terms and labels must match exactly.
    """
    if [r["review_id"] for r in records] != [e["review_id"] for e in examples]:
        raise DataIntegrityError(f"{split}: example review IDs or counts differ from the source split")

    occurrence_mismatches = 0
    for record, example in zip(records, examples):
        result = parse_and_match(example["completion"][0]["content"], record["text"])
        if not result.is_valid:
            raise DataIntegrityError(f"{split}/{record['review_id']}: reference answer invalid: {result.errors}")
        if result.invented_count or not all(a.exact_case for a in result.aspects):
            raise DataIntegrityError(f"{split}/{record['review_id']}: reference answer not grounded in the review")

        parsed = [(a.aspect, a.sentiment) for a in result.aspects]
        expected = [(a["term"], a["sentiment"]) for a in record["aspects"]]
        if parsed != expected:
            raise DataIntegrityError(
                f"{split}/{record['review_id']}: reference answer does not round-trip: {parsed} != {expected}"
            )
        occurrence_mismatches += sum(
            (p.start, p.end) != (a["start"], a["end"]) for p, a in zip(result.aspects, record["aspects"])
        )
    return occurrence_mismatches


def label_counts(records: list[dict]) -> dict[str, int]:
    counts = Counter(a["sentiment"] for r in records for a in r["aspects"])
    return {label: counts.get(label, 0) for label in LABELS}


def build(splits_dir: Path, output_dir: Path, labeled_split: str) -> dict:
    source_paths = {name: splits_dir / f"{name}.jsonl" for name in SPLIT_NAMES}
    splits = {name: load_jsonl(path) for name, path in source_paths.items()}

    for name, records in splits.items():
        validate_source_records(records, name, labeled_split)
    check_split_integrity(splits)

    # Verify every split before writing anything, so a failure never leaves a
    # partly rebuilt output directory behind.
    examples = {name: [to_sft_example(record, name) for record in records] for name, records in splits.items()}
    occurrence_mismatches = {name: verify_examples(splits[name], examples[name], name) for name in SPLIT_NAMES}

    manifest: dict = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "splits": {},
    }
    for name, records in splits.items():
        output_path = output_dir / f"{name}.jsonl"
        write_jsonl(examples[name], output_path)
        manifest["splits"][name] = {
            "source_file": source_paths[name].name,
            "source_sha256": file_sha256(source_paths[name]),
            "output_sha256": file_sha256(output_path),
            "examples": len(examples[name]),
            "aspects": sum(len(r["aspects"]) for r in records),
            "aspect_labels": label_counts(records),
            "aspects_grounded_to_other_occurrence": occurrence_mismatches[name],
        }

    with (output_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return manifest


def verify_manifest(manifest_path: Path, splits_dir: Path) -> None:
    """Fail if the split files or the prompt changed since the SFT data was built."""
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest["prompt_version"] != PROMPT_VERSION:
        raise DataIntegrityError(
            f"SFT data uses prompt {manifest['prompt_version']}, code uses {PROMPT_VERSION}; rebuild it"
        )
    if manifest["system_prompt_sha256"] != hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest():
        raise DataIntegrityError("system prompt changed without a PROMPT_VERSION bump; rebuild the SFT data")
    if set(manifest["splits"]) != set(SPLIT_NAMES):
        raise DataIntegrityError(f"manifest covers {sorted(manifest['splits'])}, expected {sorted(SPLIT_NAMES)}")
    for name, entry in manifest["splits"].items():
        if file_sha256(splits_dir / f"{name}.jsonl") != entry["source_sha256"]:
            raise DataIntegrityError(f"{name} split changed since the SFT data was built")
        if file_sha256(manifest_path.parent / f"{name}.jsonl") != entry["output_sha256"]:
            raise DataIntegrityError(f"SFT file {name}.jsonl was modified after it was built")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    splits_dir = PROJECT_ROOT / config["output"]["splits_dir"]
    output_dir = PROJECT_ROOT / config["sft"]["output_dir"]

    manifest = build(splits_dir, output_dir, config["dataset"]["labeled_split"])
    verify_manifest(output_dir / "manifest.json", splits_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
