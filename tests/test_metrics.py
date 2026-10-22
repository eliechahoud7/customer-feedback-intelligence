import json

import pytest

from src.data.build_sft_dataset import load_jsonl
from src.data.prepare_dataset import PROJECT_ROOT, review_id
from src.data.prompts import format_answer
from src.evaluation.metrics import (
    evaluate,
    load_responses,
    precision_recall_f1,
    resolve_split,
    score_review,
    score_reviews,
    sentiment_scores,
)

VALIDATION_SPLIT = PROJECT_ROOT / "data" / "splits" / "validation.jsonl"


def record(text, *aspects):
    located, cursor = [], 0
    for term, sentiment in aspects:
        start = text.index(term, cursor)
        located.append({"term": term, "start": start, "end": start + len(term), "sentiment": sentiment})
        cursor = start + len(term)
    return {"review_id": review_id(text), "text": text, "aspects": located}


def answer(*pairs):
    return json.dumps([{"aspect": a, "sentiment": s} for a, s in pairs])


# --- Sanity check: perfect answers must score perfectly -----------------------


@pytest.mark.skipif(not VALIDATION_SPLIT.exists(), reason="prepared splits not available")
def test_reference_answers_score_perfectly_on_validation():
    records = load_jsonl(VALIDATION_SPLIT)
    summary = evaluate(records, {r["review_id"]: format_answer(r["aspects"]) for r in records})

    for block in ("aspect", "aspect_sentiment"):
        assert summary[block]["precision"] == summary[block]["recall"] == summary[block]["f1"] == 1.0
    assert summary["sentiment"]["macro_f1"] == 1.0
    assert summary["sentiment_without_conflict"]["macro_f1"] == 1.0
    assert summary["responses"] == {
        "valid": len(records), "malformed": 0, "invalid_json": 0, "empty_answer": 0, "code_fence": 0,
    }
    assert summary["invented_aspects"] == summary["malformed_items"] == summary["invalid_labels"] == 0


def test_empty_answers_score_zero_without_errors():
    records = [record("The battery is really long.", ("battery", "positive"))]
    summary = evaluate(records, {records[0]["review_id"]: "[]"})

    assert summary["aspect"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "predicted": 0, "gold": 1}
    assert summary["sentiment"]["matched_aspects"] == 0
    assert summary["sentiment"]["macro_f1"] == 0.0
    assert summary["responses"]["empty_answer"] == 1


def test_empty_denominators_are_safe():
    assert precision_recall_f1(0, 0, 0) == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "predicted": 0, "gold": 0,
    }
    assert sentiment_scores([], ("positive", "negative"))["macro_f1"] == 0.0


# --- Colab baseline examples -----------------------------------------------------


def test_colab_example_empty_answer():
    rec = record("It runs very quiet too which is a plus.", ("runs", "positive"))
    score = score_review(rec, "[]")
    assert (score["status"], score["gold"], score["predicted"], score["term_tp"]) == ("valid", 1, 0, 0)
    assert score["empty_answer"]


@pytest.mark.parametrize(
    "text, gold, response",
    [
        (
            "The Computer itself is a good product but the repair depot stinks.",
            ("repair depot", "negative"),
            '[{"aspect": "repair depot", "sentiment": "negative"}, {"aspect": "Computer itself", "sentiment": "positive"}}]',
        ),
        (
            "The battery is really long.",
            ("battery", "positive"),
            '[{"aspect": "battery", "sentiment": "positive"}}]',
        ),
    ],
)
def test_colab_examples_with_extra_brace_are_invalid_json_and_score_nothing(text, gold, response):
    score = score_review(record(text, gold), response)
    assert score["status"] == "invalid_json"
    assert (score["predicted"], score["term_tp"], score["pair_tp"]) == (0, 0, 0)


def test_colab_examples_without_extra_brace_are_scored():
    rec = record("The Computer itself is a good product but the repair depot stinks.", ("repair depot", "negative"))
    score = score_review(rec, answer(("repair depot", "negative"), ("Computer itself", "positive")))
    assert (score["status"], score["predicted"], score["term_tp"], score["pair_tp"]) == ("valid", 2, 1, 1)

    rec = record("The battery is really long.", ("battery", "positive"))
    summary = evaluate([rec], {rec["review_id"]: answer(("battery", "positive"))})
    assert summary["aspect_sentiment"]["f1"] == 1.0


def test_gold_field_name_term_is_not_accepted_as_answer_key():
    # The answer schema uses "aspect"; "term" is only the gold data's field name.
    rec = record("The battery is really long.", ("battery", "positive"))
    score = score_review(rec, '[{"term": "battery", "sentiment": "positive"}]')
    assert score["status"] == "malformed"
    assert (score["predicted"], score["term_tp"], score["malformed_items"]) == (1, 0, 1)


# --- Pairing rules -------------------------------------------------------------


def test_repeated_terms_are_paired_in_order():
    rec = record(
        "The touch-pad is fine, but the touch-pad buttons are stiff.",
        ("touch-pad", "neutral"),
        ("touch-pad", "negative"),
    )
    same = score_review(rec, answer(("touch-pad", "neutral"), ("touch-pad", "negative")))
    swapped = score_review(rec, answer(("touch-pad", "negative"), ("touch-pad", "neutral")))
    once = score_review(rec, answer(("touch-pad", "neutral")))

    assert (same["term_tp"], same["pair_tp"]) == (2, 2)
    assert (swapped["term_tp"], swapped["pair_tp"]) == (2, 0)
    assert (once["predicted"], once["term_tp"], once["pair_tp"]) == (1, 1, 1)


def test_term_matching_ignores_case_but_counts_it():
    rec = record("Battery life is great.", ("Battery life", "positive"))
    score = score_review(rec, answer(("battery life", "positive")))
    assert (score["term_tp"], score["pair_tp"], score["case_mismatches"]) == (1, 1, 1)


def test_invented_aspect_is_a_false_positive():
    rec = record("Skip the cord.", ("cord", "neutral"))
    summary = evaluate([rec], {rec["review_id"]: answer(("cord", "neutral"), ("keyboard", "positive"))})
    assert summary["aspect"]["precision"] == 0.5
    assert summary["aspect"]["recall"] == 1.0
    assert summary["invented_aspects"] == 1


def test_invalid_label_matches_term_but_never_sentiment():
    rec = record("Skip the cord.", ("cord", "positive"))
    summary = evaluate([rec], {rec["review_id"]: answer(("cord", "Positive"))})
    assert summary["aspect"]["f1"] == 1.0
    assert summary["aspect_sentiment"]["f1"] == 0.0
    assert summary["sentiment"]["accuracy"] == 0.0
    assert summary["invalid_labels"] == 1
    assert summary["responses"]["malformed"] == 1


def test_unreadable_items_count_as_false_positives():
    rec = record("Skip the cord.", ("cord", "neutral"))
    score = score_review(rec, json.dumps([{"aspect": "cord", "sentiment": "neutral"}, {"aspect": ""}, "cord"]))
    assert (score["predicted"], score["term_tp"], score["malformed_items"]) == (3, 1, 2)


def test_code_fence_is_strict_by_default():
    rec = record("Skip the cord.", ("cord", "neutral"))
    fenced = "```json\n" + answer(("cord", "neutral")) + "\n```"
    assert score_review(rec, fenced)["status"] == "invalid_json"
    lenient = score_review(rec, fenced, allow_code_fence=True)
    assert (lenient["status"], lenient["pair_tp"], lenient["code_fence"]) == ("valid", 1, True)


# --- Sentiment macro F1 -------------------------------------------------------------


def test_sentiment_macro_f1_with_and_without_conflict():
    pairs = [("positive", "positive"), ("negative", "negative"), ("neutral", "neutral"), ("conflict", "positive")]
    with_conflict = sentiment_scores(pairs, ("positive", "negative", "neutral", "conflict"))
    without = sentiment_scores(pairs, ("positive", "negative", "neutral"))

    # positive: P=1/2, R=1 -> F1=2/3; negative and neutral 1.0; conflict 0.
    assert with_conflict["macro_f1"] == pytest.approx((2 / 3 + 1 + 1 + 0) / 4)
    assert with_conflict["matched_aspects"] == 4
    # The gold-conflict pair is left out, so every remaining prediction is right.
    assert without["macro_f1"] == 1.0
    assert without["matched_aspects"] == 3


def test_predicted_conflict_is_a_miss_when_conflict_is_excluded():
    pairs = [("positive", "conflict"), ("negative", "negative"), ("neutral", "neutral")]
    scores = sentiment_scores(pairs, ("positive", "negative", "neutral"))
    assert scores["per_label"]["positive"]["recall"] == 0.0
    assert scores["accuracy"] == pytest.approx(2 / 3)


# --- Integrity guards -------------------------------------------------------------


def test_every_review_needs_exactly_one_response():
    records = [record("Skip the cord.", ("cord", "neutral")), record("Nice screen.", ("screen", "positive"))]
    with pytest.raises(ValueError, match="1 missing"):
        score_reviews(records, {records[0]["review_id"]: "[]"})
    with pytest.raises(ValueError, match="1 unexpected"):
        score_reviews(records[:1], {records[0]["review_id"]: "[]", "lap-other": "[]"})


def test_duplicate_saved_responses_are_rejected(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text('{"review_id": "a", "response": "[]"}\n{"review_id": "a", "response": "[]"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_responses(path)


def test_test_split_requires_explicit_final_test_flag():
    assert resolve_split("validation", final_test=False) == "validation"
    with pytest.raises(ValueError, match="held out"):
        resolve_split("test", final_test=False)
    assert resolve_split("test", final_test=True) == "test"
    with pytest.raises(ValueError, match="unknown split"):
        resolve_split("dev", final_test=False)
