import json

import pytest

from src.data.prepare_dataset import LABELS
from src.data.prompts import format_answer
from src.evaluation.parsing import parse_and_match, parse_response


def answer(*pairs):
    return json.dumps([{"aspect": a, "sentiment": s} for a, s in pairs])


def spans(result, text):
    return [(text[a.start : a.end] if not a.invented else None, a.sentiment) for a in result.aspects]


def test_multiple_aspects_are_parsed_and_located():
    text = "I charge it at night and skip taking the cord with me because of the good battery life."
    result = parse_and_match(answer(("cord", "neutral"), ("battery life", "positive")), text)

    assert result.is_valid
    assert result.errors == []
    assert [(a.aspect, a.start, a.end, a.sentiment) for a in result.aspects] == [
        ("cord", 41, 45, "neutral"),
        ("battery life", 74, 86, "positive"),
    ]


def test_empty_array_is_a_valid_answer():
    result = parse_and_match("[]", "Nothing to see here.")
    assert result.is_valid
    assert result.aspects == []


@pytest.mark.parametrize("label", LABELS)
def test_all_four_labels_are_accepted(label):
    result = parse_and_match(answer(("screen", label)), "The screen.")
    assert result.is_valid
    assert result.aspects[0].label_valid
    assert result.aspects[0].sentiment == label


def test_repeated_term_with_different_sentiments_maps_to_successive_occurrences():
    text = "The touch-pad is fine, but the touch-pad buttons are stiff."
    result = parse_and_match(answer(("touch-pad", "neutral"), ("touch-pad", "negative")), text)

    assert result.is_valid
    assert [(a.start, a.sentiment) for a in result.aspects] == [(4, "neutral"), (31, "negative")]


def test_term_inside_a_longer_term_gets_its_own_occurrence_in_any_order():
    text = "The mic jack is loose and the mic is quiet."
    for pairs in ([("mic jack", "negative"), ("mic", "negative")], [("mic", "negative"), ("mic jack", "negative")]):
        result = parse_and_match(answer(*pairs), text)
        located = {a.aspect: (a.start, a.end) for a in result.aspects}
        assert located == {"mic jack": (4, 12), "mic": (30, 33)}


def test_repeated_term_offsets_follow_text_order_across_passes():
    # The whole-word "weight" is matched in the first pass and the one inside
    # "lightweight" in a later pass; offsets must still follow answer order.
    text = "It is lightweight and adds no weight."
    result = parse_and_match(answer(("weight", "positive"), ("weight", "neutral")), text)
    assert [(a.start, a.sentiment) for a in result.aspects] == [(11, "positive"), (30, "neutral")]


def test_repeated_term_skips_occurrence_inside_longer_aspect():
    text = "The battery life is short, the battery is old, and the battery case broke."
    result = parse_and_match(
        answer(("battery life", "negative"), ("battery", "negative"), ("battery case", "negative")), text
    )
    assert [(text[a.start : a.end], a.start) for a in result.aspects] == [
        ("battery life", 4),
        ("battery", 31),
        ("battery case", 55),
    ]


def test_whole_word_occurrence_is_preferred():
    text = "It is lightweight and adds no weight."
    result = parse_and_match(answer(("weight", "positive")), text)
    assert (result.aspects[0].start, result.aspects[0].end) == (30, 36)


def test_special_characters_round_trip_through_json():
    text = 'I called the "sales" team about the 13" screen\'s C:\\ drive / OS (Win7).'
    aspects = [
        {"term": '"sales" team', "sentiment": "negative"},
        {"term": '13" screen', "sentiment": "neutral"},
        {"term": "C:\\ drive", "sentiment": "neutral"},
        {"term": "OS (Win7)", "sentiment": "conflict"},
    ]
    result = parse_and_match(format_answer(aspects), text)

    assert result.is_valid
    assert spans(result, text) == [(a["term"], a["sentiment"]) for a in aspects]


@pytest.mark.parametrize(
    "response",
    [
        "",
        "cord: neutral",
        '[{"aspect": "cord", "sentiment": "neutral"}',
        "[{'aspect': 'cord', 'sentiment': 'neutral'}]",
        '[{"aspect": "cord", "sentiment": "neutral"}] Hope this helps!',
    ],
)
def test_invalid_json_yields_no_predictions(response):
    result = parse_and_match(response, "Skip the cord.")
    assert result.status == "invalid_json"
    assert result.aspects == []
    assert result.errors


def test_code_fence_is_rejected_by_default_and_flagged_when_allowed():
    response = '```json\n[{"aspect": "cord", "sentiment": "neutral"}]\n```'
    assert parse_response(response).status == "invalid_json"

    result = parse_response(response, allow_code_fence=True)
    assert result.is_valid
    assert result.code_fence
    assert result.aspects[0].aspect == "cord"


def test_top_level_object_is_malformed():
    result = parse_response('{"aspect": "cord", "sentiment": "neutral"}')
    assert result.status == "malformed"
    assert result.aspects == []


@pytest.mark.parametrize("label", ["Positive", "mixed", "", "POS"])
def test_invalid_label_is_kept_and_flagged_not_dropped(label):
    result = parse_and_match(answer(("cord", label), ("screen", "positive")), "The cord and screen.")

    assert result.status == "malformed"
    assert [(a.aspect, a.sentiment, a.label_valid) for a in result.aspects] == [
        ("cord", label, False),
        ("screen", "positive", True),
    ]


def test_unreadable_items_are_counted():
    response = json.dumps(
        [
            "cord",
            {"sentiment": "neutral"},
            {"aspect": "  ", "sentiment": "neutral"},
            {"aspect": "cord", "sentiment": 1},
            {"aspect": "screen", "sentiment": "positive"},
        ]
    )
    result = parse_and_match(response, "The cord and screen.")

    assert result.status == "malformed"
    assert result.malformed_items == 4
    assert [a.aspect for a in result.aspects] == ["screen"]


def test_extra_keys_are_reported_but_item_is_kept():
    result = parse_response('[{"aspect": "cord", "sentiment": "neutral", "reason": "x"}]')
    assert result.status == "malformed"
    assert [a.aspect for a in result.aspects] == ["cord"]


def test_invented_aspect_is_kept_and_flagged():
    result = parse_and_match(answer(("cord", "neutral"), ("keyboard", "positive")), "Skip the cord.")

    assert result.is_valid  # the format is fine; the prediction is wrong
    assert [a.invented for a in result.aspects] == [False, True]
    assert result.invented_count == 1


def test_term_predicted_more_often_than_it_occurs_is_invented():
    result = parse_and_match(answer(("cord", "neutral"), ("cord", "negative")), "Skip the cord.")
    assert [a.invented for a in result.aspects] == [False, True]


def test_case_insensitive_match_is_flagged():
    result = parse_and_match(answer(("battery life", "positive")), "Battery life is great.")
    aspect = result.aspects[0]
    assert not aspect.invented
    assert not aspect.exact_case
    assert (aspect.start, aspect.end) == (0, 12)
