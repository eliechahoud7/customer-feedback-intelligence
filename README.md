# Customer Feedback Intelligence

Fine-tuning a small open LLM to understand customer feedback at scale.

## Objective

Customer feedback arrives as unstructured text, and one review often praises one part of a product while criticizing another. This project performs **aspect-based sentiment analysis (ABSA)**: it fine-tunes **Qwen2.5-3B-Instruct** with **QLoRA** to find the aspects a laptop review mentions and the sentiment toward each one, and exposes the model through a **Streamlit** interface.

> "I charge it at night and skip taking the cord with me because of the good battery life."
> → `cord`: neutral, `battery life`: positive

Goals:

- Beat a zero-shot baseline of the same model, with the same prompt, on held-out data.
- Report **macro F1** over the four sentiment labels, alongside aspect extraction and joint aspect + sentiment F1.
- Keep training feasible on a single consumer GPU via 4-bit quantization + LoRA adapters.

## Tech stack

| Area | Tools |
|------|-------|
| Language | Python |
| Base model | Qwen2.5-3B-Instruct |
| Fine-tuning | QLoRA — Hugging Face Transformers, PEFT, TRL (`SFTTrainer`), bitsandbytes |
| Data | Hugging Face Datasets, pandas |
| Evaluation | scikit-learn (macro F1, classification report, confusion matrix) |
| UI | Streamlit |

## Project structure

```
customer-feedback-intelligence/
├── data/
│   ├── raw/          # Original, untouched datasets (never edited by hand)
│   ├── processed/    # Cleaned, normalized data and prompt-formatted examples
│   └── splits/       # Fixed train / validation / test splits for reproducibility
├── notebooks/        # Exploration: EDA, label distribution, error analysis
├── src/
│   ├── data/         # Loading, cleaning, label mapping, prompt formatting, splitting
│   ├── modeling/     # Model/tokenizer loading, QLoRA config, training & inference
│   ├── evaluation/   # Metrics (macro F1 etc.), output parsing, reports
│   └── app/          # Streamlit app logic (service.py: config, input checks, results)
├── tests/            # Unit tests for data processing, parsing, metrics and the app
├── configs/          # YAML configs for data, training hyperparameters, evaluation
├── models/           # Local copy of the trained LoRA adapter (git-ignored)
├── streamlit_app.py  # Streamlit user interface
├── requirements.txt
└── README.md
```

Code, configs and the prepared splits in `data/splits/` are versioned. `data/raw/`, `data/processed/` (including the generated SFT files) and model artifacts are git-ignored and regenerated from the code.

## Development stages

1. **Project setup** ✅ — folder structure, dependencies, git hygiene.
2. **Data acquisition & EDA** ✅ — SemEval-2014 laptops ABSA dataset; label schema and class balance inspected.
3. **Data preprocessing** ✅ — validation, review grouping, leakage-free stratified splits, instruction-tuning examples, answer parser.
4. **Baseline evaluation** ✅ — zero-shot Qwen2.5-3B-Instruct with the shared prompt as the reference.
5. **QLoRA fine-tuning** ✅ — 4-bit NF4 base model + LoRA adapters trained with TRL's `SFTTrainer` on a Colab T4.
6. **Evaluation & error analysis** ✅ — checkpoint chosen on validation, then scored once on the held-out test split.
7. **Streamlit app** ✅ — paste a review, get each aspect with its sentiment and a sentiment summary.
8. **Packaging & documentation** — final results, model card, reproducibility instructions.

## Dataset preparation

Source: [`tomaarsen/setfit-absa-semeval-laptops`](https://huggingface.co/datasets/tomaarsen/setfit-absa-semeval-laptops) (SemEval-2014 laptops). Each row is one aspect annotation (`text`, `span`, `label`, `ordinal`). Labels: `positive`, `negative`, `neutral`, `conflict`.

```bash
python -m src.data.prepare_dataset          # settings in configs/data.yaml
```

The pipeline (`src/data/prepare_dataset.py`):

1. Validates rows: missing values, unknown labels, spans not found in the text, duplicates, and conflicting labels for the same span occurrence. Bad rows are dropped and counted, not repaired.
2. Normalizes whitespace. Three training reviews appear twice, once prefixed with a non-breaking space, and are merged.
3. Groups rows into one record per review, keeping every aspect with its character offsets.
4. Treats the original `test` split as **unlabeled**, because all its sentiment labels are empty. One of its reviews also appears in the labeled data; that copy is flagged with `in_labeled_data: true`.
5. Splits the labeled reviews 70/15/15 at the **review level** (seed 42), stratified by each review's rarest label so `conflict` appears in every split.

Outputs:

| File | Content |
|------|---------|
| `data/processed/labeled_reviews.jsonl` | All 1,479 cleaned labeled reviews (2,342 aspects) |
| `data/processed/unlabeled_reviews.jsonl` | Original test split: 422 reviews with aspect terms, no sentiment |
| `data/processed/data_report.json` | Validation counts, overlap, per-split label distributions |
| `data/splits/{train,validation,test}.jsonl` | Leakage-free splits used for training and evaluation |

Record format:

```json
{"review_id": "lap-…", "text": "I charge it at night and skip taking the cord with me because of the good battery life.",
 "aspects": [{"term": "cord", "start": 41, "end": 45, "sentiment": "neutral"},
             {"term": "battery life", "start": 74, "end": 86, "sentiment": "positive"}],
 "source_split": "train"}
```

## Instruction-tuning (SFT) data

```bash
python -m src.data.build_sft_dataset        # reads data/splits/, writes data/processed/sft/
```

The model performs **aspect extraction and sentiment classification jointly**: given only the review, it answers with every aspect term and its sentiment. Each review becomes one example in TRL's conversational **prompt/completion** format, so the training loss covers only the answer:

```json
{"review_id": "lap-76ef38e511", "split": "test", "prompt_version": "v1",
 "prompt": [{"role": "system", "content": "You are an aspect-based sentiment analysis system for laptop reviews. ..."},
            {"role": "user", "content": "Review: I charge it at night and skip taking the cord with me because of the good battery life."}],
 "completion": [{"role": "assistant", "content": "[{\"aspect\": \"cord\", \"sentiment\": \"neutral\"}, {\"aspect\": \"battery life\", \"sentiment\": \"positive\"}]"}]}
```

- **Prompt** (`src/data/prompts.py`): one system prompt defining the four labels and the rules. Terms are copied verbatim from the review and listed in text order, repeated terms are listed once per occurrence, and the reply is a JSON array only (`[]` if there are no aspects). Training, the baseline, evaluation and the app all import it from this module. Any change bumps `PROMPT_VERSION`.
- **Answer**: a JSON array of `{"aspect", "sentiment"}` objects. JSON escapes terms that contain quotes, such as `"sales" team`. The answer has no character offsets; evaluation pairs predictions with gold aspects by term and order.
- **Class balance**: training uses the natural label distribution, including the rare `conflict` class (32 training aspects). Oversampling will only be considered if validation results justify it, and then only within train.

### Answer parsing (`src/evaluation/parsing.py`)

The parser never repairs or silently drops a prediction:

| Problem | How it is reported |
|---------|--------------------|
| Not valid JSON, extra text around the array | `status="invalid_json"`, no predictions |
| Markdown code fence | Rejected by default; with `allow_code_fence=True` accepted and flagged `code_fence=True` |
| Not an array, unreadable items, extra keys | `status="malformed"`, unreadable items counted in `malformed_items` |
| Unknown sentiment label (e.g. `"Positive"`, `"mixed"`) | Item kept with `label_valid=False`, so it counts as wrong |
| Aspect not in the review, or predicted more often than it occurs | Item kept with `invented=True` |
| Aspect found only by ignoring case | Item kept, flagged `exact_case=False` |

Predicted terms are grounded in the review text. Repeated terms map to successive occurrences, whole-word matches are preferred ("weight" over "light**weight**"), and longer terms are matched first ("mic" vs "mic jack").

### Leakage prevention and integrity checks

`build_sft_dataset.py` stops with an error if any check fails:

- **The splits are the source of truth.** They are read, never re-split or modified. Each SFT file has exactly the same review IDs, in the same order, as its source split.
- **Disjoint splits.** No review ID or review text (ignoring case and spacing) appears in more than one split.
- **Labeled data only.** Every source review must come from the labeled source split with one of the four labels and valid offsets. The unlabeled original test split is never read.
- **Round trip.** Every reference answer is parsed back with the evaluation parser and must reproduce the original terms and labels exactly, all found verbatim in the review.
- **Checksums.** `data/processed/sft/manifest.json` records SHA-256 checksums of the source splits and generated files, the prompt version and a hash of the system prompt. `verify_manifest()` fails if any of them changed since the build. `.gitattributes` keeps `.jsonl` files LF-only so checksums match across operating systems.
- **Train-only changes.** Validation and test are never used for training, augmentation or oversampling. Prompt and hyperparameter choices are made on validation, and test is scored once at the end.

## QLoRA fine-tuning (Colab T4)

The notebook [`notebooks/colab_train_eval.ipynb`](notebooks/colab_train_eval.ipynb) runs the whole workflow below on a Colab T4: setup, the version check, tests, data checks, a 5-step test run, training, the baseline, checkpoint selection and the final test.
- Outputs are kept on Google Drive.
- The private repo is cloned with a `GITHUB_TOKEN` Colab secret.
- Full training and the final test evaluation are off by default.

The same steps as commands:

```bash
git clone https://github.com/eliechahoud7/customer-feedback-intelligence.git && cd customer-feedback-intelligence
pip install -r requirements.txt               # Colab's preinstalled CUDA PyTorch already satisfies torch
python -m src.data.build_sft_dataset          # SFT files are generated, not committed
python -m src.modeling.train --check-data     # verify data + real token lengths, no training
python -m src.modeling.train --max-steps 5    # smoke run (outputs/...-smoke)
python -m src.modeling.train                  # full run, settings in configs/train.yaml
```

`src/modeling/train.py`:
- **Data checks:** it checks the SFT manifest against the current splits, and loads **only** `train.jsonl` and `validation.jsonl`. Metadata columns are dropped, and every row's `split` and `prompt_version` are checked.
- **Length check:** it stops if any example, rendered with the real chat template, is longer than `max_length`.
- **Training:** it loads the model in 4-bit NF4 with float16 compute (T4 has no bf16), then applies `prepare_model_for_kbit_training` and LoRA on all attention and MLP projections. It trains with TRL's `SFTTrainer` with `completion_only_loss`, so only the JSON answer is learned, plus gradient checkpointing, `paged_adamw_8bit` and SDPA attention.
- **TRL check:** it stops at startup if the installed TRL lacks `completion_only_loss` or `max_length`.
- **Outputs:** a checkpoint each epoch, the final adapter in `outputs/qwen2.5-3b-absa-qlora/final/`, and `run_info.json`. That file records the config, data checksums, token-length statistics, the training log, library versions and the git commit.

**Choosing a checkpoint**, on validation only:

```bash
python -m src.evaluation.select_checkpoint --run-dir outputs/qwen2.5-3b-absa-qlora
```

- **Generating:** each `checkpoint-<step>` gets validation predictions from `src.modeling.inference`, run as a separate process so GPU memory is freed. Existing prediction files in `outputs/eval/` are reused; `--regenerate` re-runs them, and `--score-only` never runs the model.
- **Choosing:** the checkpoint with the highest aspect + sentiment F1 wins. Ties go to the higher aspect F1, then to the earlier checkpoint.
- **Stopping:** missing predictions or invalid metrics stop the selection.
- **Report:** `outputs/eval/checkpoint_selection.json` records the selected checkpoint, every candidate's metrics, and checksums of the prediction files and validation split. It has no timestamps, so the same predictions give the same report.
- **Test split:** never read. It is scored once, on the selected checkpoint, with `--final-test`.

## Inference and evaluation

Run inference and scoring through the tested modules, not ad-hoc notebook code. The validation split is the default. The test split is refused unless `--final-test` is passed, and it is reserved for the single final evaluation.

```bash
# Colab (GPU): generate raw outputs for the validation split and score them
python -m src.modeling.inference --split validation --output outputs/baseline_validation.jsonl

# Anywhere: re-score saved raw outputs without re-running the model
python -m src.evaluation.metrics --responses outputs/baseline_validation.jsonl --per-review outputs/baseline_validation_scores.jsonl
```

**Inference** (`src/modeling/inference.py`): `AbsaPredictor` loads the model once, in 4-bit by default with an optional LoRA adapter, and reuses it for batched generation.
- Prompts are rendered from the shared messages with `apply_chat_template(..., add_generation_prompt=True)`, using left padding.
- Decoding is greedy. Qwen's default sampling settings and `repetition_penalty=1.05` are overridden explicitly, because a repetition penalty also applies to greedy search.
- Only newly generated tokens are decoded.
- Each output row keeps the raw `response` next to the parse `status`, errors, grounded aspects, and a `truncated` flag.

**Metrics** (`src/evaluation/metrics.py`): answers are parsed with `src/evaluation/parsing.py`. Within a review, the k-th prediction of a term is paired with the k-th gold occurrence of that term, ignoring case and extra whitespace.

| Metric | Definition |
|--------|------------|
| Aspect P/R/F1 | Matched terms vs. all predicted and all gold aspects |
| Aspect + sentiment P/R/F1 | Matched terms whose sentiment is also correct |
| Sentiment macro F1 | Over the four labels, on matched aspects; also reported without `conflict` (gold-`conflict` aspects left out) |
| Response diagnostics | Valid / malformed / invalid-JSON / empty-answer / code-fence counts, invented aspects, unreadable items, invalid labels, case mismatches |

Every prediction counts:
- Invented aspects and unreadable items are false positives.
- Invalid-JSON answers contribute no predictions, so their gold aspects are missed.
- Invalid labels never match.
- Empty denominators give 0.0, and the raw counts are reported alongside.

As a sanity check, scoring the reference answers on validation gives 1.0 for every metric (`tests/test_metrics.py`).

### Known limitations

- **Every labeled review has at least one aspect**, because the source dataset excludes aspect-free sentences. The model never sees a `[]` answer during training and may invent aspects for reviews without any. Evaluation tracks this through the invented-aspect rate.
- **Ambiguous occurrences.** Because answers carry terms, not offsets, one training aspect ("keyboard", first used as a verb in the review) is grounded to a different occurrence than the annotated one. The manifest records this count (`aspects_grounded_to_other_occurrence`).
- **`conflict` is rare** (6 test aspects), so its F1 is noisy. Macro F1 will also be reported without it.

## Streamlit app

`streamlit_app.py` analyzes one review at a time. It shows the review with each aspect highlighted, a table of aspects and sentiments, counts for the four labels, and the raw model output when the answer is not valid. It reuses `AbsaPredictor` and the shared prompt and parser; `src/app/service.py` holds the rest of the logic (configuration, input checks, warnings) and is unit-tested without a model. The model is loaded once per process (`st.cache_resource`) and reused for every click.

Held-out test results of the fine-tuned adapter (`checkpoint-195`, 222 test reviews), shown in the app's sidebar:

| Metric | Fine-tuned |
|---|---|
| Aspect F1 | 85.55% |
| Aspect + sentiment F1 | 73.77% |
| Sentiment accuracy (matched aspects) | 86.23% |
| Sentiment macro F1 | 79.95% |
| Valid structured responses | 222/222 |

### Where the model runs

| `CFI_DEVICE` | Behavior |
|---|---|
| `auto` (default) | 4-bit NF4 on a CUDA GPU with at least 6 GB; otherwise CPU |
| `cpu` | CPU, bfloat16, no quantization (bitsandbytes 4-bit is GPU-only) |
| `cuda` | 4-bit on the GPU even if it is small; fails clearly if PyTorch has no CUDA |

The sidebar shows the device that was chosen and why. On CPU the bfloat16 weights alone take about 6 GB of RAM. On a laptop i7-1165G7 with 16 GB, a short review took about 30–40 s, and loading the cached model took between under a minute and a few minutes. A 2 GB laptop GPU (e.g. MX450) is too small for the 3B model even in 4-bit, so `auto` uses the CPU there.

### Windows setup (CPU)

From the project folder in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# CPU build of PyTorch (smaller download; the CUDA build is only useful with a large GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Adapter location

The adapter is not in Git. Copy these two files from the training run on Google Drive (`outputs/qwen2.5-3b-absa-qlora/checkpoint-195/`) into `models\checkpoint-195\`:

```
models\checkpoint-195\adapter_config.json
models\checkpoint-195\adapter_model.safetensors
```

The optimizer and scheduler files in the checkpoint are not needed. `models/` is git-ignored, as are `*.safetensors` files anywhere. To keep the adapter somewhere else, point `CFI_ADAPTER_PATH` at its folder (relative paths are resolved from the project root):

```powershell
$env:CFI_ADAPTER_PATH = "D:\models\absa-checkpoint-195"   # current terminal only
setx CFI_ADAPTER_PATH "D:\models\absa-checkpoint-195"     # future terminals
```

The app checks that the folder contains both files and that `adapter_config.json` names `Qwen/Qwen2.5-3B-Instruct` before it loads anything. No access token is needed.

### Launch

```powershell
.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Streamlit opens http://localhost:8501. The first analysis downloads the base model (about 6 GB) into the Hugging Face cache (`%USERPROFILE%\.cache\huggingface`) and loads it, which can take several minutes; later analyses reuse the loaded model. The model was trained on single review sentences from SemEval-2014, so short reviews work best; input is limited to 1,500 characters.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install the CUDA build of PyTorch matching your driver first, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

A CUDA-capable NVIDIA GPU is required for QLoRA training (bitsandbytes 4-bit quantization).
