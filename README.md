# Customer Feedback Intelligence

Fine-tuning a small open LLM to understand customer feedback at scale.

## Objective

Customer feedback (reviews, support tickets, survey answers) arrives as unstructured text. This project fine-tunes **Qwen2.5-3B-Instruct** with **QLoRA** to turn each piece of feedback into structured, actionable labels (for example sentiment, topic/category, and urgency), and exposes the model through a simple **Streamlit** interface.

Goals:

- Beat a zero-shot baseline of the same model on held-out data.
- Report **macro F1** as the primary metric (robust to class imbalance), along with per-class precision/recall/F1, accuracy, and a confusion matrix.
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
│   └── app/          # Streamlit application (later stage)
├── tests/            # Unit tests for data processing, parsing, and metrics
├── configs/          # YAML configs for data, training hyperparameters, evaluation
├── requirements.txt
└── README.md
```

Data folders and model artifacts are git-ignored; only the code and configs are versioned.

## Development stages

1. **Project setup** ✅ — folder structure, dependencies, git hygiene.
2. **Data acquisition & EDA** ✅ — SemEval-2014 laptops ABSA dataset; label schema and class balance inspected.
3. **Data preprocessing** — ✅ validation, review grouping, leakage-free stratified splits; ⏳ convert examples to Qwen chat-format prompts.
4. **Baseline evaluation** — run zero-shot (and optionally few-shot) Qwen2.5-3B-Instruct on the test split to establish a reference macro F1.
5. **QLoRA fine-tuning** — 4-bit NF4 base model + LoRA adapters trained with TRL's `SFTTrainer`; hyperparameters in `configs/`.
6. **Evaluation & error analysis** — macro F1, per-class metrics, confusion matrix, invalid-output rate; compare against the baseline.
7. **Streamlit app** — enter or upload feedback, get predicted labels, view batch summaries.
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
