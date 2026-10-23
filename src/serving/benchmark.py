"""Measure the inference API from a client: latency, and optionally validation scores.

Reads CFI_API_URL and CFI_API_TOKEN. Latency is reported for the first request
and the following ones separately, as seen by the client (including the
network and tunnel) and by the server (model time only).

    python -m src.serving.benchmark --requests 6
    python -m src.serving.benchmark --validation \
        --reference outputs/eval/checkpoint-195_validation.jsonl

--validation sends every validation review through the API and scores the
answers with src.evaluation.metrics. With --reference (the saved predictions
of the same checkpoint from checkpoint selection) it also reports how many raw
answers are identical. The test split is never used.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from src.app import remote, service
from src.data.build_sft_dataset import load_jsonl
from src.data.prepare_dataset import DEFAULT_CONFIG, PROJECT_ROOT, load_config, review_id
from src.evaluation.metrics import evaluate, load_responses


def timed_requests(predictor, texts: list[str], clock=time.perf_counter) -> list[dict]:
    results = []
    for text in texts:
        started = clock()
        prediction = predictor.predict([{"review_id": review_id(text), "text": text}])[0]
        results.append({
            "client_seconds": round(clock() - started, 3),
            "server_seconds": prediction["server_seconds"],
            "generated_tokens": prediction["generated_tokens"],
            "status": prediction["status"],
        })
    return results


def latency_summary(results: list[dict]) -> dict:
    first, rest = results[0], results[1:]
    summary = {"first_request": first, "requests": len(results)}
    if rest:
        client = [r["client_seconds"] for r in rest]
        server = [r["server_seconds"] for r in rest]
        summary["subsequent_requests"] = {
            "count": len(rest),
            "client_median_seconds": round(statistics.median(client), 3),
            "client_max_seconds": round(max(client), 3),
            "server_median_seconds": round(statistics.median(server), 3),
            "network_overhead_median_seconds": round(statistics.median(c - s for c, s in zip(client, server)), 3),
        }
    return summary


def validation_report(predictor, records: list[dict], reference: dict[str, str] | None) -> dict:
    responses = {}
    for record in records:
        responses[record["review_id"]] = predictor.predict([record])[0]["response"]
    report = {"split": "validation", "scores": evaluate(records, responses)}
    if reference is not None:
        same = sum(responses[rid] == reference.get(rid) for rid in responses)
        report["identical_to_reference"] = {"identical": same, "reviews": len(responses)}
        report["reference_scores"] = evaluate(records, reference)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", type=int, default=6, help="latency requests (cycling the example reviews)")
    parser.add_argument("--validation", action="store_true", help="score the validation split through the API")
    parser.add_argument("--reference", type=Path, help="saved validation predictions of the same checkpoint")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    try:
        settings = remote.remote_settings()
        if settings is None:
            raise service.AppError(f"Set {remote.URL_ENV} and {remote.TOKEN_ENV} first.")
        predictor = remote.RemotePredictor(settings)
        health = predictor.health()
        if health["status"] != "ready":
            raise service.AppError(f"The API is not ready: {health['status']} {health.get('error') or ''}")
        print(json.dumps({"health": health}, indent=2))

        texts = [service.EXAMPLE_REVIEWS[i % len(service.EXAMPLE_REVIEWS)] for i in range(args.requests)]
        print(json.dumps({"latency": latency_summary(timed_requests(predictor, texts))}, indent=2))

        if args.validation:
            splits_dir = PROJECT_ROOT / load_config(args.config)["output"]["splits_dir"]
            records = load_jsonl(splits_dir / "validation.jsonl")
            reference = load_responses(args.reference) if args.reference else None
            print(json.dumps({"validation": validation_report(predictor, records, reference)}, indent=2))
    except service.AppError as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
