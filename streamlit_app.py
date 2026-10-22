"""Streamlit front end for Customer Feedback Intelligence.

Run from the project root:
    streamlit run streamlit_app.py

Model loading, prompting and parsing live in src/app/service.py and the
modules it reuses; this file only lays out the page.
"""

from __future__ import annotations

import html
import traceback

import pandas as pd
import streamlit as st

from src.app import service

st.set_page_config(page_title="Customer Feedback Intelligence", page_icon="💻", layout="centered")

SENTIMENT_COLORS = {
    "positive": "#2da44e",
    "negative": "#cf222e",
    "neutral": "#8c959f",
    "conflict": "#bf8700",
}
SENTIMENT_ICONS = {"positive": "🟢", "negative": "🔴", "neutral": "⚪", "conflict": "🟠"}


@st.cache_resource(show_spinner=False)
def get_predictor(adapter_path: str, device: str):
    """Loaded once per adapter and device, then shared by every session and click."""
    from pathlib import Path

    return service.load_predictor(Path(adapter_path), device)


def load_example() -> None:
    index = st.session_state.get("example_index", -1) + 1
    st.session_state.example_index = index
    st.session_state.review = service.EXAMPLE_REVIEWS[index % len(service.EXAMPLE_REVIEWS)]


def clear_review() -> None:
    st.session_state.review = ""
    st.session_state.pop("analysis", None)


def highlighted_review(segments: list[tuple[str, str | None]]) -> str:
    parts = []
    for text, sentiment in segments:
        text = html.escape(text)
        if sentiment is None:
            parts.append(text)
        else:
            color = SENTIMENT_COLORS[sentiment]
            parts.append(
                f'<span title="{sentiment}" style="background:{color}26;border-bottom:2px solid {color};'
                f'border-radius:3px;padding:1px 3px;">{text}</span>'
            )
    return f'<div style="line-height:1.9;font-size:1.05rem;">{"".join(parts)}</div>'


def show_error(message: str, error: BaseException) -> None:
    st.error(message)
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(error)), language="text")


def render_sidebar(runtime: service.Runtime | None, adapter_error: str | None) -> None:
    with st.sidebar:
        st.header("Model")
        st.markdown(f"**Base model:** `{service.DEFAULT_MODEL}`  \n**Fine-tuning:** QLoRA")
        if runtime is not None:
            st.markdown(f"**Adapter:** `{runtime.adapter_path.name}`  \n**Runs on:** {runtime.description}")
            st.caption(f"Because {runtime.reason}. Adapter folder: {runtime.adapter_path}")
        if adapter_error:
            st.error(adapter_error)

        st.header("Held-out test results")
        st.caption("Final evaluation on 222 test reviews that were never used for training or checkpoint selection.")
        for label, value in service.TEST_RESULTS:
            st.markdown(f"{label}: **{value}**")


def render_analysis(analysis: service.Analysis) -> None:
    st.subheader("Results")
    st.markdown(highlighted_review(analysis.segments), unsafe_allow_html=True)
    st.write("")

    columns = st.columns(len(service.LABELS))
    for column, label in zip(columns, service.LABELS):
        column.metric(f"{SENTIMENT_ICONS[label]} {label.capitalize()}", analysis.counts[label])

    for warning in analysis.warnings:
        st.warning(warning)

    if analysis.rows:
        table = pd.DataFrame(analysis.rows)
        table["sentiment"] = [f"{SENTIMENT_ICONS.get(s, '❔')} {s}" for s in table["sentiment"]]
        if not table["note"].any():
            table = table.drop(columns="note")
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "aspect": st.column_config.TextColumn("Aspect"),
                "sentiment": st.column_config.TextColumn("Sentiment"),
                "note": st.column_config.TextColumn("Note"),
            },
        )
    elif analysis.status == "valid":
        st.info("The model found no laptop aspects in this review.")

    st.caption(f"Analyzed in {analysis.seconds:.1f} s · {analysis.generated_tokens} tokens generated")
    with st.expander("Raw model output", expanded=analysis.status != "valid"):
        st.code(analysis.raw_response or "(empty response)", language="json")
        st.caption(f"Parse status: {analysis.status} · prompt version {analysis.prompt_version}")
        for error in analysis.errors:
            st.markdown(f"- {error}")


st.title("Customer Feedback Intelligence")
st.markdown(
    "Aspect-based sentiment analysis for laptop reviews. A fine-tuned "
    "**Qwen2.5-3B-Instruct** model finds the product aspects a review mentions "
    "(battery, screen, support, ...) and classifies the sentiment toward each one as "
    "**positive**, **negative**, **neutral** or **conflict**."
)

runtime, adapter_error = None, None
try:
    runtime = service.resolve_runtime()
    service.check_adapter(runtime.adapter_path)
except service.AppError as error:
    adapter_error = str(error)
except Exception as error:  # e.g. PyTorch missing or broken
    adapter_error = service.describe_error(error, service.Runtime(service.adapter_path(), "cpu", ""))
render_sidebar(runtime, adapter_error)

st.text_area(
    "Laptop review",
    key="review",
    height=140,
    max_chars=service.MAX_REVIEW_CHARS,
    placeholder="Paste a laptop review, e.g. “Great display, but the battery barely lasts three hours.”",
    help="The model was trained on review sentences from SemEval-2014; short reviews work best.",
)
example_column, clear_column, _ = st.columns([1.3, 1, 3])
example_column.button("Try an example", on_click=load_example, width="stretch")
clear_column.button("Clear", on_click=clear_review, width="stretch")

if adapter_error:
    st.error("The model is not available: " + adapter_error)

if st.button("Analyze Review", type="primary", disabled=adapter_error is not None, width="stretch"):
    try:
        review = service.validate_review(st.session_state.get("review"))
    except service.AppError as error:
        st.warning(str(error))
    else:
        try:
            with st.spinner("Loading the model. The first run downloads about 6 GB and can take several minutes..."):
                predictor = get_predictor(str(runtime.adapter_path), runtime.device)
        except Exception as error:
            show_error(service.describe_error(error, runtime), error)
        else:
            try:
                with st.spinner(f"Analyzing review on {runtime.device.upper()}..."):
                    st.session_state.analysis = service.analyze(predictor, review)
            except Exception as error:
                st.session_state.pop("analysis", None)
                show_error(service.describe_error(error, runtime), error)

if "analysis" in st.session_state:
    render_analysis(st.session_state.analysis)
