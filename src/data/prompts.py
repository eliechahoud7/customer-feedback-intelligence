"""Shared prompt and answer format for aspect-based sentiment analysis.

Training data, the zero-shot baseline, evaluation and the Streamlit app all
build prompts from this module so the wording can never drift between them.
Any change to the prompt or answer format must bump PROMPT_VERSION.
"""

from __future__ import annotations

import json

from src.data.prepare_dataset import LABELS

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are an aspect-based sentiment analysis system for laptop reviews.

Identify every aspect term in the review: a word or phrase naming a feature, component, or service of the laptop (for example its hardware, software, price, or customer support) that the review mentions.
For each aspect term, classify the sentiment the review expresses toward it:
- positive: the review is favorable toward the aspect.
- negative: the review is unfavorable toward the aspect.
- neutral: the aspect is mentioned without a positive or negative opinion.
- conflict: the review expresses both positive and negative opinions about the aspect.

Rules:
- Copy each aspect term exactly as it appears in the review, keeping its spelling, capitalization, and punctuation.
- List aspects in the order they appear in the review. If the same term appears more than once, list each occurrence separately.
- Respond with a JSON array only, with no other text: [{"aspect": "<term>", "sentiment": "<label>"}]
- If the review mentions no aspect terms, respond with []."""

# The prompt text spells out the labels; fail fast if the label set changes.
if not all(f"- {label}:" in SYSTEM_PROMPT for label in LABELS):
    raise RuntimeError("SYSTEM_PROMPT does not define every label in LABELS")


def format_user_message(text: str) -> str:
    return f"Review: {text}"


def build_prompt(text: str) -> list[dict]:
    """Chat messages that precede the model's answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_user_message(text)},
    ]


def format_answer(aspects: list[dict]) -> str:
    """Reference answer: JSON array of {aspect, sentiment} in text order."""
    return json.dumps(
        [{"aspect": aspect["term"], "sentiment": aspect["sentiment"]} for aspect in aspects],
        ensure_ascii=False,
    )
