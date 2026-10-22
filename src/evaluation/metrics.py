"""Metrics for joint aspect extraction and sentiment classification.

Predictions are paired with gold aspects by term and order: within a review,
the k-th predicted occurrence of a term is paired with the k-th gold
occurrence of the same term. Terms are compared ignoring case and extra
whitespace; case differences are still counted in ``case_mismatches``.

Every prediction counts. Invented aspects and unreadable answer items are
false positives, invalid-JSON answers contribute no predictions (so all their
gold aspects are false negatives), and unknown sentiment labels never match.

Usage (re-score saved raw outputs; the test split requires --final-test):
    python -m src.evaluation.metrics --responses outputs/baseline_validation.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support

from src.data.build_sft_dataset import load_jsonl
from src.data.prepare_dataset import (
    DEFAULT_CONFIG,
    LABELS,
    PROJECT_ROOT,
    SPLIT_NAMES,
    load_config,
    text_key,
    write_jsonl,
)
from src.evaluation.parsing import PredictedAspect, parse_and_match

LABELS_WITHOUT_CONFLICT = tuple(label for label in LABELS if label != "conflict")


def resolve_split(split: str, final_test: bool) -> str:
    """Guard against scoring the held-out test split by accident."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    if split == "test" and not final_test:
        raise ValueError(
            "the test split is held out for the final evaluation; pass --final-test (final_test=True) to use it"
        )
    return split


def align(gold_aspects: list[dict], predicted: list[PredictedAspect]) -> list[tuple[dict, PredictedAspect]]:
    """Pair the k-th gold occurrence of each term with its k-th prediction."""
    predicted_by_term: dict[str, list[PredictedAspect]] = defaultdict(list)
    for aspect in predicted:
        predicted_by_term[text_key(aspect.aspect)].append(aspect)

    pairs, used = [], Counter()
    for gold in gold_aspects:
        key = text_key(gold["term"])
        if used[key] < len(predicted_by_term[key]):
            pairs.append((gold, predicted_by_term[key][used[key]]))
            used[key] += 1
    return pairs


def score_review(record: dict, response: str, allow_code_fence: bool = False) -> dict:
    result = parse_and_match(response, record["text"], allow_code_fence=allow_code_fence)
    pairs = align(record["aspects"], result.aspects)
    return {
        "review_id": record["review_id"],
        "status": result.status,
        "errors": result.errors,
        "code_fence": result.code_fence,
        "gold": len(record["aspects"]),
        # Unreadable items are predictions too; they can never be correct.
        "predicted": len(result.aspects) + result.malformed_items,
        "term_tp": len(pairs),
        "pair_tp": sum(gold["sentiment"] == pred.sentiment for gold, pred in pairs),
        "sentiment_pairs": [(gold["sentiment"], pred.sentiment) for gold, pred in pairs],
        "invented": result.invented_count,
        "malformed_items": result.malformed_items,
        "invalid_labels": sum(not aspect.label_valid for aspect in result.aspects),
        "case_mismatches": sum(not aspect.invented and not aspect.exact_case for aspect in result.aspects),
        "empty_answer": result.status == "valid" and not result.aspects,
    }


def precision_recall_f1(tp: int, predicted: int, gold: int) -> dict:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "predicted": predicted, "gold": gold}


def sentiment_scores(pairs: list[tuple[str, str]], labels: tuple[str, ...]) -> dict:
    """Macro F1 over ``labels`` for aspects whose term was matched.

    Pairs whose gold label is outside ``labels`` are left out. A predicted
    label outside ``labels`` (e.g. "conflict" when excluded, or an invalid
    label) still counts as a miss for the gold class.
    """
    pairs = [(gold, pred) for gold, pred in pairs if gold in labels]
    scores = {"matched_aspects": len(pairs), "accuracy": 0.0, "macro_f1": 0.0, "per_label": {}}
    if not pairs:
        return scores

    gold, pred = zip(*pairs)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=list(labels), average=None, zero_division=0
    )
    scores["accuracy"] = sum(g == p for g, p in pairs) / len(pairs)
    scores["macro_f1"] = float(f1.mean())
    scores["per_label"] = {
        label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    }
    return scores


def summarize(review_scores: list[dict]) -> dict:
    total = Counter()
    for score in review_scores:
        total.update(
            {
                key: score[key]
                for key in ("gold", "predicted", "term_tp", "pair_tp", "invented", "malformed_items",
                            "invalid_labels", "case_mismatches")
            }
        )
    statuses = Counter(score["status"] for score in review_scores)
    sentiment_pairs = [pair for score in review_scores for pair in score["sentiment_pairs"]]

    return {
        "reviews": len(review_scores),
        "aspect": precision_recall_f1(total["term_tp"], total["predicted"], total["gold"]),
        "aspect_sentiment": precision_recall_f1(total["pair_tp"], total["predicted"], total["gold"]),
        "sentiment": sentiment_scores(sentiment_pairs, LABELS),
        "sentiment_without_conflict": sentiment_scores(sentiment_pairs, LABELS_WITHOUT_CONFLICT),
        "responses": {
            "valid": statuses["valid"],
            "malformed": statuses["malformed"],
            "invalid_json": statuses["invalid_json"],
            "empty_answer": sum(score["empty_answer"] for score in review_scores),
            "code_fence": sum(score["code_fence"] for score in review_scores),
        },
        "invented_aspects": total["invented"],
        "malformed_items": total["malformed_items"],
        "invalid_labels": total["invalid_labels"],
        "case_mismatches": total["case_mismatches"],
    }


def score_reviews(records: list[dict], responses: dict[str, str], allow_code_fence: bool = False) -> list[dict]:
    """Score one raw response per review. Every review must have exactly one response."""
    expected = {record["review_id"] for record in records}
    missing, extra = expected - set(responses), set(responses) - expected
    if missing or extra:
        raise ValueError(f"responses do not match the split: {len(missing)} missing, {len(extra)} unexpected")
    return [score_review(record, responses[record["review_id"]], allow_code_fence) for record in records]


def evaluate(records: list[dict], responses: dict[str, str], allow_code_fence: bool = False) -> dict:
    return summarize(score_reviews(records, responses, allow_code_fence))


def load_responses(path: Path) -> dict[str, str]:
    responses = {}
    for row in load_jsonl(path):
        if row["review_id"] in responses:
            raise ValueError(f"duplicate response for {row['review_id']}")
        responses[row["review_id"]] = row["response"]
    return responses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", type=Path, required=True, help="JSONL with review_id and raw response")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--final-test", action="store_true", help="allow scoring the held-out test split")
    parser.add_argument("--allow-code-fence", action="store_true")
    parser.add_argument("--per-review", type=Path, help="optional JSONL path for per-review scores")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    split = resolve_split(args.split, args.final_test)
    splits_dir = PROJECT_ROOT / load_config(args.config)["output"]["splits_dir"]
    records = load_jsonl(splits_dir / f"{split}.jsonl")

    scores = score_reviews(records, load_responses(args.responses), args.allow_code_fence)
    if args.per_review:
        write_jsonl(scores, args.per_review)
    print(json.dumps({"split": split, **summarize(scores)}, indent=2))


if __name__ == "__main__":
    main()
