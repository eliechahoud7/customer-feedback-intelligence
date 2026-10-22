"""Inference tests with a character-level fake tokenizer and model.

No model weights, torch or transformers are needed: the fakes record how they
are called, and a stub ``torch`` module provides ``inference_mode``.
"""

import contextlib
import sys
import types

import pytest

import src.modeling.inference as inference
from src.data.prepare_dataset import review_id
from src.data.prompts import SYSTEM_PROMPT
from src.modeling.inference import GREEDY_GENERATION, AbsaPredictor

PAD, EOS = 0, 1


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    """Characters are tokens (their code points); 0 is padding, 1 ends a turn."""

    def __init__(self):
        self.padding_side = "right"
        self.pad_token, self.pad_token_id = "<pad>", PAD
        self.eos_token = "<eos>"
        self.template_calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.template_calls.append({"messages": messages, "tokenize": tokenize,
                                    "add_generation_prompt": add_generation_prompt})
        rendered = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, texts, return_tensors, padding, add_special_tokens):
        assert self.padding_side == "left"
        assert add_special_tokens is False
        width = max(len(t) for t in texts)
        ids = [[PAD] * (width - len(t)) + [ord(c) for c in t] for t in texts]
        return FakeBatch(input_ids=ids, attention_mask=[[int(i != PAD) for i in row] for row in ids])

    def batch_decode(self, rows, skip_special_tokens):
        assert skip_special_tokens
        return ["".join(chr(t) for t in row if t not in (PAD, EOS)) for row in rows]


class FakeModel:
    """Answers each prompt with a canned response looked up by review text."""

    device = "cpu"

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def generate(self, input_ids, attention_mask, max_new_tokens, pad_token_id, **kwargs):
        self.calls.append({"batch_size": len(input_ids), "max_new_tokens": max_new_tokens, **kwargs})
        outputs = []
        for row in input_ids:
            prompt = "".join(chr(t) for t in row if t != PAD)
            text = prompt.split("Review: ", 1)[1].removesuffix("<assistant>")
            new = [ord(c) for c in self.responses[text]][:max_new_tokens]
            if len(new) < max_new_tokens:
                new.append(EOS)
            outputs.append(list(row) + new)
        width = max(len(row) for row in outputs)
        return [row + [pad_token_id] * (width - len(row)) for row in outputs]


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(inference_mode=contextlib.nullcontext))


def make_record(text, *aspects):
    located = []
    for term, sentiment in aspects:
        start = text.index(term)
        located.append({"term": term, "start": start, "end": start + len(term), "sentiment": sentiment})
    return {"review_id": review_id(text), "text": text, "aspects": located}


RECORDS = [
    make_record("The battery is really long.", ("battery", "positive")),
    make_record("It runs very quiet too which is a plus.", ("runs", "positive")),
    make_record("The Computer itself is a good product but the repair depot stinks.", ("repair depot", "negative")),
]
RESPONSES = {
    RECORDS[0]["text"]: '[{"aspect": "battery", "sentiment": "positive"}]',
    RECORDS[1]["text"]: "[]",
    RECORDS[2]["text"]: '[{"aspect": "repair depot", "sentiment": "negative"}}]',
}


def make_predictor(**kwargs):
    return AbsaPredictor(FakeModel(RESPONSES), FakeTokenizer(), **kwargs)


def test_prompt_uses_shared_messages_and_generation_prompt():
    predictor = make_predictor()
    rendered = predictor.render_prompt("The battery is really long.")

    call = predictor.tokenizer.template_calls[-1]
    assert call["add_generation_prompt"] is True
    assert call["tokenize"] is False
    assert call["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Review: The battery is really long."},
    ]
    assert rendered.endswith("<assistant>")


def test_tokenizer_is_left_padded():
    assert make_predictor().tokenizer.padding_side == "left"


def test_only_new_tokens_are_decoded():
    outputs = make_predictor().generate([r["text"] for r in RECORDS])
    assert [o["response"] for o in outputs] == [RESPONSES[r["text"]] for r in RECORDS]
    assert not any("Review:" in o["response"] or SYSTEM_PROMPT[:20] in o["response"] for o in outputs)


def test_decoding_is_greedy_and_overrides_qwen_defaults():
    predictor = make_predictor()
    predictor.generate([RECORDS[0]["text"]])
    call = predictor.model.calls[0]
    for key, value in GREEDY_GENERATION.items():
        assert call[key] == value
    assert call["do_sample"] is False
    assert call["repetition_penalty"] == 1.0


def test_model_is_reused_across_batches():
    predictor = make_predictor(batch_size=2)
    predictor.predict(RECORDS)
    predictor.predict(RECORDS)
    assert [c["batch_size"] for c in predictor.model.calls] == [2, 1, 2, 1]


def test_predict_keeps_raw_text_next_to_parsed_result():
    predictions = make_predictor(batch_size=2).predict(RECORDS)

    assert [p["review_id"] for p in predictions] == [r["review_id"] for r in RECORDS]
    valid, empty, broken = predictions
    assert valid["status"] == "valid"
    assert valid["aspects"] == [{"aspect": "battery", "sentiment": "positive", "start": 4, "end": 11,
                                 "label_valid": True, "invented": False}]
    assert (empty["status"], empty["aspects"], empty["response"]) == ("valid", [], "[]")
    assert broken["status"] == "invalid_json"
    assert broken["response"] == RESPONSES[RECORDS[2]["text"]]
    assert broken["errors"]


def test_truncated_generation_is_flagged():
    predictor = make_predictor(max_new_tokens=10)
    output = predictor.generate([RECORDS[0]["text"]])[0]
    assert output["truncated"] is True
    assert output["generated_tokens"] == 10
    assert output["response"] == RESPONSES[RECORDS[0]["text"]][:10]

    complete = make_predictor().generate([RECORDS[0]["text"]])[0]
    assert complete["truncated"] is False


def test_heavy_dependencies_are_not_imported_at_module_level():
    source = open(inference.__file__, encoding="utf-8").read().split("class AbsaPredictor")[0]
    assert "import torch" not in source
    assert "from transformers" not in source
