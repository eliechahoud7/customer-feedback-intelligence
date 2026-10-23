"""Streamlit front end for Customer Feedback Intelligence.

Run from the project root:
    streamlit run streamlit_app.py

Model loading, prompting and parsing live in src/app/service.py and the
modules it reuses; this file only lays out the page.

By default reviews are sent to the GPU inference API (src/serving/api.py)
configured by CFI_API_URL and CFI_API_TOKEN, read from environment variables
or Streamlit secrets. If the API is not configured or fails, the app shows the
error; it never falls back to local CPU inference. Local inference runs only
with CFI_INFERENCE_MODE=local.
"""

from __future__ import annotations

import html
import os
import traceback

import pandas as pd
import requests
import streamlit as st
from streamlit.errors import StreamlitAPIException

from src.app import remote, service

st.set_page_config(page_title="Customer Feedback Intelligence", page_icon="💻", layout="centered")

SENTIMENT_COLORS = {
    "positive": "#2da44e",
    "negative": "#cf222e",
    "neutral": "#8c959f",
    "conflict": "#bf8700",
}
SENTIMENT_ICONS = {"positive": "🟢", "negative": "🔴", "neutral": "⚪", "conflict": "🟠"}
CONFIG_KEYS = (service.MODE_ENV, remote.URL_ENV, remote.TOKEN_ENV, service.ADAPTER_ENV, service.DEVICE_ENV)


def app_config() -> dict[str, str]:
    """Environment variables, completed from Streamlit secrets (e.g. on Community Cloud)."""
    config = dict(os.environ)
    try:
        for key in CONFIG_KEYS:
            if not config.get(key) and key in st.secrets:
                config[key] = str(st.secrets[key])
    except (FileNotFoundError, StreamlitAPIException):
        pass  # no (readable) secrets.toml: environment variables only
    return config


@st.cache_resource(show_spinner=False)
def get_predictor(adapter_path: str, device: str):
    """Loaded once per adapter and device, then shared by every session and click."""
    from pathlib import Path

    return service.load_predictor(Path(adapter_path), device)


@st.cache_resource(show_spinner=False)
def http_session() -> requests.Session:
    """One connection pool, shared by every session and rerun."""
    return requests.Session()


def get_remote_predictor(settings: remote.RemoteSettings):
    # Deliberately not cached: after a code update Streamlit re-imports src.app, and a
    # cached predictor would keep raising the old AppError class, which the new code
    # no longer catches.
    return remote.RemotePredictor(settings, http_session())


@st.cache_data(ttl=15, show_spinner=False)
def remote_health(settings: remote.RemoteSettings) -> tuple[dict | None, str | None]:
    try:
        return get_remote_predictor(settings).health(), None
    except service.AppError as error:
        return None, str(error)
    except Exception as error:  # a status check must never take down the page
        return None, f"Unexpected error while checking the API ({type(error).__name__})."


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
    if isinstance(error, service.AppError):
        return
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(error)), language="text")


def render_remote_status(settings: remote.RemoteSettings) -> None:
    st.markdown("**Runs on:** Remote inference API")
    health, error = remote_health(settings)
    if error:
        st.warning(f"Remote inference API unavailable. {error}")
    elif health.get("status") == "ready":
        st.success(f"Model ready on {health.get('gpu') or 'a CUDA GPU'} ({health.get('precision')})")
        st.caption("Fine-tuned adapter loaded successfully.")
    elif health.get("status") == "loading":
        st.info("The remote model is still loading. This takes a few minutes after the service starts.")
    else:
        st.error(f"The remote model failed to load: {health.get('error')}")
    st.button("Refresh status", on_click=remote_health.clear)


def render_sidebar(runtime: service.Runtime | None, settings: remote.RemoteSettings | None,
                   setup_error: str | None) -> None:
    with st.sidebar:
        st.header("Model")
        st.markdown(f"**Base model:** `{service.DEFAULT_MODEL}`  \n**Fine-tuning:** QLoRA")
        if settings is not None:
            render_remote_status(settings)
        elif runtime is not None:
            st.markdown(f"**Adapter:** `{runtime.adapter_path.name}`  \n**Runs on:** {runtime.description}")
            st.caption(f"Because {runtime.reason}. Adapter folder: {runtime.adapter_path}")
        if setup_error:
            st.error(setup_error)

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

    server = f" ({analysis.server_seconds:.1f} s on the server)" if analysis.server_seconds is not None else ""
    st.caption(f"Analyzed in {analysis.seconds:.1f} s{server} · {analysis.generated_tokens} tokens generated")
    with st.expander("Raw model output", expanded=analysis.status != "valid"):
        st.code(analysis.raw_response or "(empty response)", language="json")
        st.caption(f"Parse status: {analysis.status} · prompt version {analysis.prompt_version}")
        for error in analysis.errors:
            st.markdown(f"- {error}")


st.title("Customer Feedback Intelligence")
st.markdown("Share your experience. Discover the sentiment behind every review.")

runtime, settings, setup_error = None, None, None
try:
    config = app_config()
    if service.inference_mode(config) == "remote":
        settings = remote.require_remote_settings(config)
    else:
        runtime = service.resolve_runtime(config)
        service.check_adapter(runtime.adapter_path)
except service.AppError as error:
    setup_error = str(error)
except Exception as error:  # e.g. PyTorch missing or broken
    setup_error = service.describe_error(error, None)
render_sidebar(runtime, settings, setup_error)

st.text_area(
    "Review",  # hidden; kept for screen readers
    key="review",
    height=140,
    max_chars=service.MAX_REVIEW_CHARS,
    placeholder="Paste your review here...",
    label_visibility="collapsed",
)
example_column, clear_column, _ = st.columns([1.3, 1, 3])
example_column.button("Try an example", on_click=load_example, width="stretch")
clear_column.button("Clear", on_click=clear_review, width="stretch")

if setup_error:
    st.error("The model is not available: " + setup_error)

if st.button("Analyze Review", type="primary", disabled=setup_error is not None, width="stretch"):
    try:
        review = service.validate_review(st.session_state.get("review"))
    except service.AppError as error:
        st.warning(str(error))
    else:
        try:
            if settings is not None:
                predictor, where = get_remote_predictor(settings), "the remote GPU"
            else:
                with st.spinner("Loading the model. The first run downloads about 6 GB and can take several minutes..."):
                    predictor = get_predictor(str(runtime.adapter_path), runtime.device)
                where = runtime.device.upper()
            with st.spinner(f"Analyzing review on {where}..."):
                st.session_state.analysis = service.analyze(predictor, review)
        except Exception as error:
            # No fallback: a remote failure is reported, never retried on the local CPU.
            st.session_state.pop("analysis", None)
            show_error(service.describe_error(error, runtime), error)

if "analysis" in st.session_state:
    render_analysis(st.session_state.analysis)
