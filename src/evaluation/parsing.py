"""Parse and validate model answers for aspect-based sentiment analysis.

The parser never repairs or drops a prediction silently. Every problem is
recorded on the result so that metrics can count it:

- invalid_json: the response is not valid JSON; no predictions are recovered.
- malformed: the JSON is valid but does not follow the answer format. Items
  that cannot be read at all (missing or empty "aspect", non-string
  "sentiment") are counted in ``malformed_items``; items with an unknown
  sentiment label are kept with ``label_valid=False`` so they count as wrong.
- Aspect terms that cannot be found in the review are kept with
  ``invented=True``. A term predicted more often than it occurs in the review
  is invented for the extra predictions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.data.prepare_dataset import LABELS

ANSWER_KEYS = ("aspect", "sentiment")
CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


@dataclass
class PredictedAspect:
    aspect: str
    sentiment: str
    label_valid: bool
    start: int | None = None
    end: int | None = None
    # False when the term was found only by ignoring letter case.
    exact_case: bool = True

    @property
    def invented(self) -> bool:
        return self.start is None


@dataclass
class ParseResult:
    status: str  # "valid", "malformed" or "invalid_json"
    aspects: list[PredictedAspect] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    malformed_items: int = 0
    code_fence: bool = False

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"

    @property
    def invented_count(self) -> int:
        return sum(aspect.invented for aspect in self.aspects)


def parse_response(response: str, allow_code_fence: bool = False) -> ParseResult:
    """Parse a raw model answer into predicted aspects (not yet matched to text).

    The answer must be a bare JSON array. With ``allow_code_fence`` a single
    surrounding Markdown code fence is accepted and flagged on the result.
    """
    body = response.strip()
    code_fence = False
    if allow_code_fence:
        fenced = CODE_FENCE.match(body)
        if fenced:
            body, code_fence = fenced.group(1), True

    try:
        data = json.loads(body)
    except json.JSONDecodeError as error:
        return ParseResult(status="invalid_json", errors=[f"invalid JSON: {error.msg}"], code_fence=code_fence)

    result = ParseResult(status="valid", code_fence=code_fence)
    if not isinstance(data, list):
        result.status = "malformed"
        result.errors.append(f"expected a JSON array, got {type(data).__name__}")
        return result

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            result.malformed_items += 1
            result.errors.append(f"item {index}: expected an object, got {type(item).__name__}")
            continue

        extra = sorted(set(item) - set(ANSWER_KEYS))
        if extra:
            result.errors.append(f"item {index}: unexpected keys {extra}")

        aspect, sentiment = item.get("aspect"), item.get("sentiment")
        if not isinstance(aspect, str) or not aspect.strip():
            result.malformed_items += 1
            result.errors.append(f"item {index}: missing or empty 'aspect'")
            continue
        if not isinstance(sentiment, str):
            result.malformed_items += 1
            result.errors.append(f"item {index}: missing or non-string 'sentiment'")
            continue

        label_valid = sentiment in LABELS
        if not label_valid:
            result.errors.append(f"item {index}: invalid sentiment label {sentiment!r}")
        result.aspects.append(PredictedAspect(aspect=aspect.strip(), sentiment=sentiment, label_valid=label_valid))

    if result.errors:
        result.status = "malformed"
    return result


def match_aspects(text: str, aspects: list[PredictedAspect]) -> None:
    """Locate each predicted term in the review, in place.

    Each prediction claims the first occurrence of its term that does not
    overlap text already claimed by another aspect, so repeated terms map to
    successive occurrences. Longer terms are matched first, so a term inside a
    longer one ("mic" in "mic jack") is matched to its own occurrence.

    Matching runs in passes, each over all still-unmatched terms: exact case
    as a whole word, exact case anywhere (e.g. inside a longer word), then the
    same two ignoring case. Case-insensitive matches are accepted but flagged
    with ``exact_case=False``. Finally, the occurrences found for a repeated
    term are handed out in text order, matching the order of the answer.
    """
    claimed: list[tuple[int, int]] = []
    by_length = sorted(aspects, key=lambda a: -len(a.aspect))

    # lower() can change the length of a few characters, which would shift
    # offsets, so the case-insensitive passes only run when it does not.
    lowered = text.lower()
    passes = [(text, False, True), (text, False, False)]
    if len(lowered) == len(text):
        passes += [(lowered, True, True), (lowered, True, False)]

    for haystack, ignore_case, whole_word in passes:
        for aspect in by_length:
            if not aspect.invented:
                continue
            term = aspect.aspect.lower() if ignore_case else aspect.aspect
            span = _first_free_occurrence(haystack, term, claimed, whole_word)
            if span is not None:
                aspect.start, aspect.end = span
                aspect.exact_case = not ignore_case
                claimed.append(span)

    # Passes can match a later occurrence first (e.g. a whole-word "weight"
    # before the one inside "lightweight"); reorder per term to follow the text.
    by_term: dict[str, list[PredictedAspect]] = {}
    for aspect in aspects:
        if not aspect.invented:
            by_term.setdefault(aspect.aspect, []).append(aspect)
    for group in by_term.values():
        matches = sorted((a.start, a.end, a.exact_case) for a in group)
        for aspect, (start, end, exact_case) in zip(group, matches):
            aspect.start, aspect.end, aspect.exact_case = start, end, exact_case


def parse_and_match(response: str, text: str, allow_code_fence: bool = False) -> ParseResult:
    """Parse an answer and ground its aspects in the review text.

    ``status`` describes the answer format only; aspects that are not in the
    review are reported through ``PredictedAspect.invented``.
    """
    result = parse_response(response, allow_code_fence=allow_code_fence)
    match_aspects(text, result.aspects)
    return result


def _first_free_occurrence(
    text: str, term: str, claimed: list[tuple[int, int]], whole_word: bool
) -> tuple[int, int] | None:
    start = text.find(term)
    while start != -1:
        end = start + len(term)
        free = all(end <= s or start >= e for s, e in claimed)
        if free and (not whole_word or _is_whole_word(text, start, end)):
            return start, end
        start = text.find(term, start + 1)
    return None


def _is_whole_word(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not before.isalnum() and not after.isalnum()
