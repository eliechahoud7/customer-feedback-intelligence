"""QLoRA fine-tuning of Qwen2.5-3B-Instruct with TRL's SFTTrainer.

Trains on data/processed/sft/train.jsonl and reports validation loss on
validation.jsonl. The test split is never read. Before training, the SFT files
are checked against the split checksums in their manifest, and every example
is tokenized with the real chat template to make sure none exceeds max_length.

torch, transformers, peft and trl are imported only when training runs, so this
module can be imported and tested without them.

Usage (Colab):
    python -m src.data.build_sft_dataset
    python -m src.modeling.train --check-data       # data and token lengths only
    python -m src.modeling.train --max-steps 5      # short smoke run
    python -m src.modeling.train                    # full run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
from importlib import metadata
from pathlib import Path

import yaml

from src.data.build_sft_dataset import load_jsonl, verify_manifest
from src.data.prepare_dataset import DEFAULT_CONFIG as DATA_CONFIG
from src.data.prepare_dataset import PROJECT_ROOT, load_config
from src.data.prompts import PROMPT_VERSION

DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"

# The only splits training may read. Test is reserved for the final evaluation.
TRAINING_SPLITS = ("train", "validation")
REQUIRED_SECTIONS = ("model", "quantization", "lora", "training")
REQUIRED_SFT_CONFIG_FIELDS = ("completion_only_loss", "max_length")
COMPUTE_DTYPE_FOR_PRECISION = {"fp16": "float16", "bf16": "bfloat16"}
LIBRARIES = ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets")


def load_train_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    missing = [section for section in REQUIRED_SECTIONS if section not in (config or {})]
    if missing:
        raise ValueError(f"training config is missing sections: {missing}")

    training = config["training"]
    if training.get("completion_only_loss") is not True:
        raise ValueError("training.completion_only_loss must be true: the loss must cover only the answer")
    if not isinstance(training.get("max_length"), int) or training["max_length"] <= 0:
        raise ValueError("training.max_length must be a positive integer")
    if "warmup_ratio" in training:
        raise ValueError(
            "training.warmup_ratio was removed in transformers 5; use warmup_steps with a value below 1 "
            "(e.g. warmup_steps: 0.03 for 3% of total steps)"
        )

    precisions = [name for name in COMPUTE_DTYPE_FOR_PRECISION if training.get(name)]
    if len(precisions) > 1:
        raise ValueError("training.fp16 and training.bf16 cannot both be true")
    compute_dtype = config["quantization"].get("bnb_4bit_compute_dtype")
    if precisions and compute_dtype != COMPUTE_DTYPE_FOR_PRECISION[precisions[0]]:
        raise ValueError(
            f"quantization.bnb_4bit_compute_dtype is {compute_dtype!r} but training uses {precisions[0]}; "
            f"use {COMPUTE_DTYPE_FOR_PRECISION[precisions[0]]!r}"
        )
    return config


def load_sft_split(sft_dir: Path, split: str) -> list[dict]:
    """Prompt/completion pairs for one training split, metadata columns removed."""
    if split not in TRAINING_SPLITS:
        raise ValueError(f"training may only read {TRAINING_SPLITS}, not {split!r}")

    examples = []
    for row in load_jsonl(sft_dir / f"{split}.jsonl"):
        if row["split"] != split:
            raise ValueError(f"{split}.jsonl contains a row from split {row['split']!r} ({row['review_id']})")
        if row["prompt_version"] != PROMPT_VERSION:
            raise ValueError(f"{row['review_id']} uses prompt {row['prompt_version']}, code uses {PROMPT_VERSION}")
        examples.append({"prompt": row["prompt"], "completion": row["completion"]})
    return examples


def token_lengths(tokenizer, examples: list[dict]) -> list[int]:
    """Tokens per example as the chat template renders the full conversation."""
    lengths = []
    for example in examples:
        tokens = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=True)
        # Some transformers versions return an encoding dict instead of a list of ids.
        if isinstance(tokens, dict) or hasattr(tokens, "keys"):
            tokens = tokens["input_ids"]
        lengths.append(len(tokens))
    return lengths


def check_token_lengths(lengths: list[int], max_length: int, split: str) -> dict:
    """Fail if any example would be truncated; return length statistics."""
    too_long = sum(length > max_length for length in lengths)
    if too_long:
        raise ValueError(f"{too_long} {split} examples exceed max_length={max_length} (longest {max(lengths)})")
    ordered = sorted(lengths)
    return {
        "examples": len(lengths),
        "max": ordered[-1],
        "mean": round(sum(lengths) / len(lengths), 1),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "max_length": max_length,
    }


def check_trl_support(sft_config_class, training: dict) -> None:
    """The installed SFTConfig must support answer-only loss and every configured key.

    Runs before the model is loaded, so a renamed or removed setting fails in
    seconds with the offending keys named, not deep inside SFTConfig.
    """
    fields = {field.name for field in dataclasses.fields(sft_config_class)}
    missing = [name for name in REQUIRED_SFT_CONFIG_FIELDS if name not in fields]
    if missing:
        raise RuntimeError(f"installed trl does not support SFTConfig fields {missing}; upgrade trl")
    unsupported = sorted(key for key in training if key not in fields)
    if unsupported:
        raise RuntimeError(
            f"configs/train.yaml training keys not accepted by the installed SFTConfig: {unsupported}; "
            "check the pinned trl/transformers versions in requirements.txt"
        )


def check_transformers_version(version: str, training: dict) -> None:
    """A fractional warmup_steps is a ratio only in transformers 5 and later."""
    warmup = training.get("warmup_steps", 0)
    if 0 < warmup < 1 and int(version.split(".")[0]) < 5:
        raise RuntimeError(
            f"warmup_steps={warmup} is a fraction of total steps only in transformers>=5 "
            f"(installed {version}); install the versions pinned in requirements.txt"
        )


def sft_config_kwargs(training: dict, max_steps: int | None = None) -> dict:
    """SFTConfig arguments; a smoke run trains max_steps steps and saves nothing."""
    kwargs = dict(training)
    output_dir = PROJECT_ROOT / kwargs["output_dir"]
    if max_steps is not None:
        kwargs.update(max_steps=max_steps, eval_strategy="no", save_strategy="no", logging_steps=1)
        output_dir = output_dir.with_name(output_dir.name + "-smoke")
    kwargs["output_dir"] = str(output_dir)
    return kwargs


def library_versions() -> dict:
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout

    try:
        return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(git("status", "--porcelain").strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def train(config: dict, check_data: bool = False, max_steps: int | None = None) -> dict:
    data_config = load_config(DATA_CONFIG)
    splits_dir = PROJECT_ROOT / data_config["output"]["splits_dir"]
    sft_dir = PROJECT_ROOT / data_config["sft"]["output_dir"]
    verify_manifest(sft_dir / "manifest.json", splits_dir)
    datasets = {split: load_sft_split(sft_dir, split) for split in TRAINING_SPLITS}

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    max_length = config["training"]["max_length"]
    length_stats = {
        split: check_token_lengths(token_lengths(tokenizer, examples), max_length, split)
        for split, examples in datasets.items()
    }
    if check_data:
        return {"token_lengths": length_stats}

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    sft_kwargs = sft_config_kwargs(config["training"], max_steps)
    check_trl_support(SFTConfig, sft_kwargs)
    check_transformers_version(metadata.version("transformers"), sft_kwargs)
    set_seed(config["training"]["seed"])

    quantization = dict(config["quantization"])
    compute_dtype = getattr(torch, quantization.pop("bnb_4bit_compute_dtype"))
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        quantization_config=BitsAndBytesConfig(**quantization, bnb_4bit_compute_dtype=compute_dtype),
        dtype=compute_dtype,
        attn_implementation=config["model"]["attn_implementation"],
        device_map="auto",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config["training"].get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs=config["training"].get("gradient_checkpointing_kwargs"),
    )
    model = get_peft_model(model, LoraConfig(**config["lora"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    args = SFTConfig(**sft_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(datasets["train"]),
        eval_dataset=Dataset.from_list(datasets["validation"]),
        processing_class=tokenizer,
    )
    train_result = trainer.train()

    output_dir = Path(args.output_dir)
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    with (sft_dir / "manifest.json").open(encoding="utf-8") as f:
        sft_manifest = json.load(f)
    run_info = {
        "config": config,
        "max_steps_override": max_steps,
        "prompt_version": PROMPT_VERSION,
        "sft_manifest": sft_manifest,
        "token_lengths": length_stats,
        "train_metrics": train_result.metrics,
        "log_history": trainer.state.log_history,
        "libraries": library_versions(),
        "git": git_commit(),
    }
    with (output_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, default=str)
    return run_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--check-data", action="store_true", help="verify data and token lengths, then stop")
    parser.add_argument("--max-steps", type=int, help="short smoke run; saves nothing but the final adapter")
    args = parser.parse_args()

    result = train(load_train_config(args.config), check_data=args.check_data, max_steps=args.max_steps)
    summary = result if args.check_data else {k: result[k] for k in ("token_lengths", "train_metrics", "git")}
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
