"""Prepare the SemEval laptop ABSA dataset for fine-tuning.

Groups aspect-level annotation rows into one record per review, validates the
annotations, and writes leakage-free train/validation/test splits as JSONL.

Usage:
    python -m src.data.prepare_dataset [--config configs/data.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"

LABELS = ("positive", "negative", "neutral", "conflict")
# A review is stratified by the rarest label among its aspects, so the rare
# classes (conflict, then neutral) are spread across every split.
STRATIFY_PRIORITY = ("conflict", "neutral", "negative", "positive")

SPLIT_NAMES = ("train", "validation", "test")


def normalize_text(text: str) -> str:
    """Collapse all whitespace (including non-breaking spaces) to single spaces."""
    return " ".join(text.split())


def text_key(text: str) -> str:
    """Key used to detect the same review across sources, ignoring case and spacing."""
    return normalize_text(text).casefold()


def review_id(text: str) -> str:
    return "lap-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def locate_span(text: str, span: str, ordinal: int) -> tuple[int, int] | None:
    """Return (start, end) of the ordinal-th occurrence of span in text, or None."""
    if ordinal < 0:
        return None
    start = -len(span)
    for _ in range(ordinal + 1):
        start = text.find(span, start + len(span))
        if start == -1:
            return None
    return start, start + len(span)


def clean_rows(raw: pd.DataFrame, labeled: bool) -> tuple[pd.DataFrame, dict]:
    """Validate and normalize annotation rows.

    Returns the clean rows (with character offsets) and a report of every issue
    found. Rows that cannot be trusted are dropped, never silently repaired.
    """
    df = raw[["text", "span", "label", "ordinal"]].copy()
    issues: dict = {"input_rows": len(df)}

    missing = df["text"].map(is_missing) | df["span"].map(is_missing) | df["ordinal"].isna()
    issues["missing_text_span_or_ordinal"] = int(missing.sum())
    df = df[~missing]

    if labeled:
        df = df.assign(label=df["label"].map(lambda v: None if is_missing(v) else str(v).strip().lower()))
        missing_label = df["label"].isna()
        issues["missing_label"] = int(missing_label.sum())
        df = df[~missing_label]

        invalid = ~df["label"].isin(LABELS)
        issues["invalid_labels"] = dict(Counter(df.loc[invalid, "label"]))
        df = df[~invalid]
    else:
        issues["non_empty_labels_ignored"] = int((~df["label"].map(is_missing)).sum())
        df = df.assign(label=None)

    key_columns = ["text", "span", "ordinal", "label"]
    issues["exact_duplicate_rows"] = int(df.duplicated(key_columns).sum())

    distinct_raw_texts = df["text"].nunique()
    df = df.assign(
        text=df["text"].map(normalize_text),
        span=df["span"].map(normalize_text),
        ordinal=df["ordinal"].astype(int),
    )
    issues["texts_merged_by_whitespace_normalization"] = distinct_raw_texts - df["text"].nunique()

    duplicates = df.duplicated(key_columns)
    issues["duplicate_rows_removed"] = int(duplicates.sum())
    df = df[~duplicates]

    offsets = [locate_span(t, s, o) for t, s, o in zip(df["text"], df["span"], df["ordinal"])]
    not_found = pd.Series([o is None for o in offsets], index=df.index)
    issues["span_not_found_in_text"] = int(not_found.sum())
    df = df[~not_found]
    offsets = [o for o in offsets if o is not None]
    df = df.assign(start=[o[0] for o in offsets], end=[o[1] for o in offsets])

    # The same span occurrence annotated with different sentiments is ambiguous.
    conflicting = df.groupby(["text", "start", "end"])["label"].transform("nunique") > 1
    issues["conflicting_annotation_rows_removed"] = int(conflicting.sum())
    df = df[~conflicting]

    issues["output_rows"] = len(df)
    return df.reset_index(drop=True), issues


def group_reviews(rows: pd.DataFrame, source_split: str) -> list[dict]:
    """One record per review text, keeping every aspect in text order."""
    records = []
    for text, group in rows.groupby("text", sort=False):
        group = group.sort_values(["start", "end"])
        aspects = [
            {"term": span, "start": int(start), "end": int(end), "sentiment": label}
            for span, start, end, label in zip(group["span"], group["start"], group["end"], group["label"])
        ]
        records.append(
            {
                "review_id": review_id(text),
                "text": text,
                "aspects": aspects,
                "source_split": source_split,
            }
        )
    return records


def stratify_key(record: dict) -> str:
    sentiments = {aspect["sentiment"] for aspect in record["aspects"]}
    return next(label for label in STRATIFY_PRIORITY if label in sentiments)


def split_reviews(records: list[dict], validation_size: float, test_size: float, seed: int) -> dict[str, list[dict]]:
    """Stratified review-level split; every aspect of a review stays in one split."""
    train_val, test = train_test_split(
        records,
        test_size=test_size,
        stratify=[stratify_key(r) for r in records],
        random_state=seed,
    )
    train, validation = train_test_split(
        train_val,
        test_size=validation_size / (1 - test_size),
        stratify=[stratify_key(r) for r in train_val],
        random_state=seed,
    )
    return {"train": train, "validation": validation, "test": test}


def assert_no_leakage(splits: dict[str, list[dict]]) -> None:
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = text_key(record["text"])
            if key in seen and seen[key] != name:
                raise ValueError(f"Review appears in both {seen[key]} and {name}: {record['text']!r}")
            seen[key] = name


def summarize(records: list[dict]) -> dict:
    sentiments = Counter(a["sentiment"] for r in records for a in r["aspects"])
    return {
        "reviews": len(records),
        "aspects": sum(len(r["aspects"]) for r in records),
        "aspect_labels": {label: sentiments.get(label, 0) for label in LABELS},
        "reviews_by_stratify_key": dict(Counter(stratify_key(r) for r in records)),
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare(config: dict) -> dict:
    from datasets import load_dataset

    dataset_cfg, split_cfg, output_cfg = config["dataset"], config["split"], config["output"]
    dataset = load_dataset(dataset_cfg["name"])

    labeled_rows, labeled_issues = clean_rows(dataset[dataset_cfg["labeled_split"]].to_pandas(), labeled=True)
    unlabeled_rows, unlabeled_issues = clean_rows(dataset[dataset_cfg["unlabeled_split"]].to_pandas(), labeled=False)

    labeled = group_reviews(labeled_rows, dataset_cfg["labeled_split"])
    unlabeled = group_reviews(unlabeled_rows, dataset_cfg["unlabeled_split"])

    # The labeled copy keeps its genuine labels; the unlabeled copy is flagged so
    # it is never mistaken for an unseen review at inference time.
    labeled_keys = {text_key(r["text"]) for r in labeled}
    overlap = []
    for record in unlabeled:
        record["in_labeled_data"] = text_key(record["text"]) in labeled_keys
        if record["in_labeled_data"]:
            overlap.append(record["text"])

    splits = split_reviews(labeled, split_cfg["validation_size"], split_cfg["test_size"], split_cfg["seed"])
    assert_no_leakage(splits)

    processed_dir = PROJECT_ROOT / output_cfg["processed_dir"]
    splits_dir = PROJECT_ROOT / output_cfg["splits_dir"]
    write_jsonl(labeled, processed_dir / "labeled_reviews.jsonl")
    write_jsonl(unlabeled, processed_dir / "unlabeled_reviews.jsonl")
    for name in SPLIT_NAMES:
        write_jsonl(splits[name], splits_dir / f"{name}.jsonl")

    report = {
        "dataset": dataset_cfg["name"],
        "labels": list(LABELS),
        "split_config": split_cfg,
        "validation": {
            dataset_cfg["labeled_split"]: labeled_issues,
            dataset_cfg["unlabeled_split"]: unlabeled_issues,
        },
        "labeled_reviews": summarize(labeled),
        "unlabeled_reviews": {
            "reviews": len(unlabeled),
            "aspects": sum(len(r["aspects"]) for r in unlabeled),
        },
        "labeled_unlabeled_overlap": {"count": len(overlap), "texts": overlap},
        "splits": {name: summarize(splits[name]) for name in SPLIT_NAMES},
    }
    with (processed_dir / "data_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    report = prepare(load_config(args.config))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
