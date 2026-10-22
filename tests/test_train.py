"""Training-script tests that run without torch, transformers, peft or trl."""

import dataclasses
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

import src.modeling.train as train_module
from src.data.prepare_dataset import PROJECT_ROOT
from src.data.prompts import PROMPT_VERSION, build_prompt
from src.modeling.train import (
    DEFAULT_TRAIN_CONFIG,
    TRAINING_SPLITS,
    check_token_lengths,
    check_trl_support,
    load_sft_split,
    load_train_config,
    sft_config_kwargs,
    token_lengths,
)

REAL_SFT_DIR = PROJECT_ROOT / "data" / "processed" / "sft"


def write_config(tmp_path, mutate):
    config = yaml.safe_load(DEFAULT_TRAIN_CONFIG.read_text(encoding="utf-8"))
    mutate(config)
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def example_row(split_name, review="The battery is really long.", **overrides):
    row = {
        "review_id": "lap-0000000001",
        "split": split_name,
        "prompt_version": PROMPT_VERSION,
        "prompt": build_prompt(review),
        "completion": [{"role": "assistant", "content": '[{"aspect": "battery", "sentiment": "positive"}]'}],
    }
    row.update(overrides)
    return row


def write_split(directory: Path, split, rows):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# --- Config ------------------------------------------------------------------------


def test_project_config_is_valid_for_a_t4():
    config = load_train_config(DEFAULT_TRAIN_CONFIG)
    assert config["model"]["name"] == "Qwen/Qwen2.5-3B-Instruct"
    assert config["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert config["quantization"]["bnb_4bit_compute_dtype"] == "float16"
    assert config["training"]["fp16"] is True and config["training"]["bf16"] is False
    assert config["training"]["completion_only_loss"] is True
    assert config["model"]["attn_implementation"] != "flash_attention_2"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.pop("lora"), "missing sections"),
        (lambda c: c["training"].update(completion_only_loss=False), "completion_only_loss"),
        (lambda c: c["training"].pop("max_length"), "max_length"),
        (lambda c: c["training"].update(bf16=True), "cannot both be true"),
        (lambda c: c["quantization"].update(bnb_4bit_compute_dtype="bfloat16"), "compute_dtype"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path, mutate, message):
    with pytest.raises(ValueError, match=message):
        load_train_config(write_config(tmp_path, mutate))


def test_bf16_config_needs_bfloat16_compute(tmp_path):
    def to_bf16(config):
        config["training"].update(fp16=False, bf16=True)
        config["quantization"]["bnb_4bit_compute_dtype"] = "bfloat16"

    assert load_train_config(write_config(tmp_path, to_bf16))["training"]["bf16"] is True


def test_sft_config_kwargs_pass_training_section_through():
    training = load_train_config(DEFAULT_TRAIN_CONFIG)["training"]
    kwargs = sft_config_kwargs(training)

    assert kwargs["output_dir"] == str(PROJECT_ROOT / training["output_dir"])
    assert {k: v for k, v in kwargs.items() if k != "output_dir"} == {
        k: v for k, v in training.items() if k != "output_dir"
    }


def test_smoke_run_overrides_steps_and_disables_eval_and_checkpoints():
    training = load_train_config(DEFAULT_TRAIN_CONFIG)["training"]
    kwargs = sft_config_kwargs(training, max_steps=5)

    assert (kwargs["max_steps"], kwargs["eval_strategy"], kwargs["save_strategy"]) == (5, "no", "no")
    assert kwargs["output_dir"].endswith("-smoke")
    assert training["eval_strategy"] == "epoch"  # the loaded config is not mutated


def test_training_outputs_go_to_ignored_directory():
    output_dir = Path(sft_config_kwargs(load_train_config(DEFAULT_TRAIN_CONFIG)["training"])["output_dir"])
    assert output_dir.relative_to(PROJECT_ROOT).parts[0] == "outputs"


# --- Data -----------------------------------------------------------------------------


def test_only_train_and_validation_can_be_loaded(tmp_path):
    write_split(tmp_path, "test", [example_row("test")])
    assert TRAINING_SPLITS == ("train", "validation")
    with pytest.raises(ValueError, match="training may only read"):
        load_sft_split(tmp_path, "test")


def test_metadata_columns_are_dropped(tmp_path):
    write_split(tmp_path, "train", [example_row("train")])
    examples = load_sft_split(tmp_path, "train")
    assert list(examples[0]) == ["prompt", "completion"]
    assert examples[0]["prompt"][1]["content"] == "Review: The battery is really long."


@pytest.mark.parametrize(
    "overrides, message",
    [({"split": "test"}, "from split 'test'"), ({"prompt_version": "v0"}, "uses prompt v0")],
)
def test_rows_from_another_split_or_prompt_version_are_rejected(tmp_path, overrides, message):
    write_split(tmp_path, "train", [example_row("train", **overrides)])
    with pytest.raises(ValueError, match=message):
        load_sft_split(tmp_path, "train")


@pytest.mark.skipif(not (REAL_SFT_DIR / "train.jsonl").exists(), reason="SFT files not built")
def test_real_training_splits_load():
    for split, expected in (("train", 1035), ("validation", 222)):
        examples = load_sft_split(REAL_SFT_DIR, split)
        assert len(examples) == expected
        assert all([m["role"] for m in e["prompt"]] == ["system", "user"] for e in examples)
        assert all([m["role"] for m in e["completion"]] == ["assistant"] for e in examples)


# --- Token lengths ----------------------------------------------------------------------


class CharTokenizer:
    """One token per character of each message, like a chat template would render it."""

    def __init__(self, as_dict=False):
        self.as_dict = as_dict
        self.calls = []

    def apply_chat_template(self, messages, tokenize):
        self.calls.append(messages)
        ids = list("".join(m["content"] for m in messages))
        return {"input_ids": ids} if self.as_dict else ids


@pytest.mark.parametrize("as_dict", [False, True])
def test_token_lengths_cover_prompt_and_completion(as_dict):
    tokenizer = CharTokenizer(as_dict=as_dict)
    example = example_row("train")
    expected = sum(len(m["content"]) for m in example["prompt"] + example["completion"])

    assert token_lengths(tokenizer, [example]) == [expected]
    assert [m["role"] for m in tokenizer.calls[0]] == ["system", "user", "assistant"]


def test_examples_longer_than_max_length_stop_training():
    assert check_token_lengths([10, 20, 30], 30, "train")["max"] == 30
    with pytest.raises(ValueError, match="1 train examples exceed max_length=25"):
        check_token_lengths([10, 20, 30], 25, "train")


@pytest.mark.skipif(not (REAL_SFT_DIR / "manifest.json").exists(), reason="SFT files not built")
def test_check_data_run_never_reads_the_test_split(monkeypatch):
    opened = []
    real_load_jsonl = train_module.load_jsonl

    def recording_load_jsonl(path):
        opened.append(Path(path).name)
        return real_load_jsonl(path)

    class WordTokenizer:
        # Character tokens would overstate real lengths about fourfold; real
        # token counts are checked with the Qwen tokenizer via --check-data.
        def apply_chat_template(self, messages, tokenize):
            return " ".join(m["content"] for m in messages).split()

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda name: WordTokenizer())
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(train_module, "load_jsonl", recording_load_jsonl)

    result = train_module.train(load_train_config(DEFAULT_TRAIN_CONFIG), check_data=True)

    assert sorted(opened) == ["train.jsonl", "validation.jsonl"]
    assert set(result["token_lengths"]) == {"train", "validation"}
    assert result["token_lengths"]["train"]["examples"] == 1035


# --- TRL support -----------------------------------------------------------------------------


def test_trl_without_answer_only_loss_is_rejected():
    @dataclasses.dataclass
    class OldSFTConfig:
        max_seq_length: int = 1024

    @dataclasses.dataclass
    class NewSFTConfig:
        max_length: int = 1024
        completion_only_loss: bool = True

    check_trl_support(NewSFTConfig)
    with pytest.raises(RuntimeError, match="upgrade trl"):
        check_trl_support(OldSFTConfig)


def test_heavy_dependencies_are_not_imported_at_module_level():
    header = Path(train_module.__file__).read_text(encoding="utf-8").split("def load_train_config")[0]
    for name in ("import torch", "from transformers", "from peft", "from trl", "from datasets"):
        assert name not in header
