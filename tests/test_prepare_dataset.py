import pandas as pd
import pytest

from src.data.prepare_dataset import (
    LABELS,
    assert_no_leakage,
    clean_rows,
    group_reviews,
    locate_span,
    normalize_text,
    split_reviews,
    stratify_key,
)


def make_rows(rows):
    return pd.DataFrame(rows, columns=["text", "span", "label", "ordinal"])


def test_normalize_text_handles_non_breaking_spaces():
    assert normalize_text("\xa0 Boot  time\xa0is fast. ") == "Boot time is fast."


def test_locate_span_uses_ordinal():
    text = "charge it, then charge again"
    assert locate_span(text, "charge", 0) == (0, 6)
    assert locate_span(text, "charge", 1) == (16, 22)
    assert locate_span(text, "charge", 2) is None
    assert locate_span(text, "battery", 0) is None


def test_clean_rows_drops_bad_rows_and_reports_them():
    raw = make_rows(
        [
            ["Good screen.", "screen", "positive", 0],
            ["Good screen.", "screen", "positive", 0],  # exact duplicate
            ["\xa0 Good screen.", "screen", "positive", 0],  # duplicate after normalization
            ["Bad fan.", "fan", "", 0],  # missing label
            ["Bad fan.", "fan", "awful", 0],  # invalid label
            ["Bad fan.", "keyboard", "negative", 0],  # span not in text
            ["", "fan", "negative", 0],  # missing text
            ["Ok keys.", "keys", "neutral", 0],
            ["Ok keys.", "keys", "negative", 0],  # conflicting with the row above
            ["Mixed price.", "price", "Conflict", 0],
        ]
    )
    rows, issues = clean_rows(raw, labeled=True)

    assert issues["missing_text_span_or_ordinal"] == 1
    assert issues["missing_label"] == 1
    assert issues["invalid_labels"] == {"awful": 1}
    assert issues["exact_duplicate_rows"] == 1
    assert issues["texts_merged_by_whitespace_normalization"] == 1
    assert issues["duplicate_rows_removed"] == 2
    assert issues["span_not_found_in_text"] == 1
    assert issues["conflicting_annotation_rows_removed"] == 2
    assert sorted(rows["label"]) == ["conflict", "positive"]


def test_clean_rows_ignores_labels_of_unlabeled_split():
    raw = make_rows([["Set up was easy.", "Set up", "", 0]])
    rows, issues = clean_rows(raw, labeled=False)
    assert rows["label"].isna().all()
    assert issues["non_empty_labels_ignored"] == 0


def test_group_reviews_keeps_every_aspect_in_text_order():
    raw = make_rows(
        [
            ["I skip the cord because of the good battery life.", "battery life", "positive", 0],
            ["I skip the cord because of the good battery life.", "cord", "neutral", 0],
            ["Fast boot.", "boot", "positive", 0],
        ]
    )
    rows, _ = clean_rows(raw, labeled=True)
    records = group_reviews(rows, "train")

    assert len(records) == 2
    first = records[0]
    assert [(a["term"], a["sentiment"]) for a in first["aspects"]] == [
        ("cord", "neutral"),
        ("battery life", "positive"),
    ]
    for aspect in first["aspects"]:
        assert first["text"][aspect["start"] : aspect["end"]] == aspect["term"]


def make_records(n_per_label=20):
    records = []
    for label in LABELS:
        for i in range(n_per_label):
            records.append(
                {
                    "review_id": f"{label}-{i}",
                    "text": f"Review {label} {i}",
                    "aspects": [
                        {"term": "screen", "start": 0, "end": 6, "sentiment": "positive"},
                        {"term": "fan", "start": 0, "end": 3, "sentiment": label},
                    ],
                }
            )
    return records


def test_stratify_key_prefers_rarest_label():
    record = {"aspects": [{"sentiment": "positive"}, {"sentiment": "conflict"}, {"sentiment": "neutral"}]}
    assert stratify_key(record) == "conflict"


def test_split_is_disjoint_reproducible_and_covers_every_label():
    records = make_records()
    splits = split_reviews(records, validation_size=0.15, test_size=0.15, seed=42)
    again = split_reviews(records, validation_size=0.15, test_size=0.15, seed=42)

    assert_no_leakage(splits)
    assert sum(len(v) for v in splits.values()) == len(records)
    for name, split in splits.items():
        assert [r["review_id"] for r in split] == [r["review_id"] for r in again[name]]
        assert {stratify_key(r) for r in split} == set(LABELS)


def test_assert_no_leakage_detects_case_and_space_variants():
    splits = {
        "train": [{"text": "Great keyboard."}],
        "test": [{"text": "great  keyboard."}],
    }
    with pytest.raises(ValueError):
        assert_no_leakage(splits)
