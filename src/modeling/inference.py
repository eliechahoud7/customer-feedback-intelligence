"""Batched Qwen inference for aspect-based sentiment analysis.

The model and tokenizer are loaded once into an ``AbsaPredictor`` and reused
for every review. Prompts come from ``src.data.prompts`` and are rendered with
the tokenizer's chat template, ending with the assistant turn header
(``add_generation_prompt=True``). Decoding is greedy and deterministic, and
only the newly generated tokens are decoded. Each prediction keeps the raw
text next to the parsed result so failures can be inspected.

torch, transformers and peft are imported only when a model is loaded, so this
module can be imported and tested without them.

Usage (Colab; writes raw outputs that src.evaluation.metrics can re-score):
    python -m src.modeling.inference --split validation --output outputs/baseline_validation.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.build_sft_dataset import load_jsonl
from src.data.prepare_dataset import DEFAULT_CONFIG, PROJECT_ROOT, load_config, write_jsonl
from src.data.prompts import PROMPT_VERSION, build_prompt
from src.evaluation.metrics import evaluate, resolve_split
from src.evaluation.parsing import parse_and_match

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Greedy decoding. Qwen's default generation_config enables sampling and sets
# repetition_penalty=1.05, which also applies to greedy search, so every
# setting that affects token choice is overridden explicitly.
GREEDY_GENERATION = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "repetition_penalty": 1.0,
}


class AbsaPredictor:
    def __init__(self, model, tokenizer, max_new_tokens: int = 256, batch_size: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        # Decoder-only models must be left-padded for batched generation, so
        # that every prompt ends right where generation starts.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_MODEL,
        adapter_path: str | None = None,
        load_in_4bit: bool = True,
        **kwargs,
    ) -> "AbsaPredictor":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        quantization = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            if load_in_4bit
            else None
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization, torch_dtype=dtype, device_map="auto"
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        return cls(model, tokenizer, **kwargs)

    def render_prompt(self, text: str) -> str:
        return self.tokenizer.apply_chat_template(build_prompt(text), tokenize=False, add_generation_prompt=True)

    def generate(self, texts: list[str]) -> list[dict]:
        """Raw completions for each review text, in order."""
        import torch

        outputs = []
        for offset in range(0, len(texts), self.batch_size):
            prompts = [self.render_prompt(text) for text in texts[offset : offset + self.batch_size]]
            # The chat template already contains every special token it needs.
            batch = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            batch = batch.to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **batch,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **GREEDY_GENERATION,
                )
            prompt_length = len(batch["input_ids"][0])
            new_tokens = [row[prompt_length:] for row in generated]
            texts_out = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for tokens, text in zip(new_tokens, texts_out):
                length = sum(int(token) != self.tokenizer.pad_token_id for token in tokens)
                outputs.append({"response": text, "generated_tokens": length,
                                "truncated": length >= self.max_new_tokens})
        return outputs

    def predict(self, records: list[dict], allow_code_fence: bool = False) -> list[dict]:
        """Raw response plus parsed prediction for each review record."""
        generations = self.generate([record["text"] for record in records])
        predictions = []
        for record, generation in zip(records, generations):
            result = parse_and_match(generation["response"], record["text"], allow_code_fence=allow_code_fence)
            predictions.append(
                {
                    "review_id": record["review_id"],
                    "prompt_version": PROMPT_VERSION,
                    **generation,
                    "status": result.status,
                    "errors": result.errors,
                    "aspects": [
                        {
                            "aspect": aspect.aspect,
                            "sentiment": aspect.sentiment,
                            "start": aspect.start,
                            "end": aspect.end,
                            "label_valid": aspect.label_valid,
                            "invented": aspect.invented,
                        }
                        for aspect in result.aspects
                    ],
                }
            )
        return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--final-test", action="store_true", help="allow running on the held-out test split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", help="optional LoRA adapter directory")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, help="only the first N reviews (debugging)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    split = resolve_split(args.split, args.final_test)
    splits_dir = PROJECT_ROOT / load_config(args.config)["output"]["splits_dir"]
    records = load_jsonl(splits_dir / f"{split}.jsonl")[: args.limit]

    predictor = AbsaPredictor.from_pretrained(
        args.model,
        adapter_path=args.adapter,
        load_in_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    predictions = predictor.predict(records)
    write_jsonl(predictions, args.output)

    summary = evaluate(records, {p["review_id"]: p["response"] for p in predictions})
    print(json.dumps({"split": split, **summary}, indent=2))


if __name__ == "__main__":
    main()
