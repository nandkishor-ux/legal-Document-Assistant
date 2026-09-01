"""Streamlit chat UI for the RTI Legal Document Assistant.

Runs ask.py's retrieval pipeline (retrieve -> grade -> rewrite -> generate ->
verify) DIRECTLY in-process — no FastAPI layer. Heavy resources (embedding
model, Qdrant vector store, BM25 index, Groq client) load exactly once via
st.cache_resource.

Run with:
    venv\\Scripts\\python.exe -m streamlit run streamlit_app.py
"""

import contextlib
import io
import os
import re
import sys

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

import ask

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

PAGE_TITLE = "RTI Legal Document Assistant"
PAGE_DESC = (
    "Cited, verified answers from the Indian Right to Information corpus "
    "(Central RTI Act 2005, Delhi RTI Act 2001, and related case law)."
)
EXAMPLE_QUESTIONS = [
    "What does Section 8(1)(j) of the RTI Act 2005 protect?",
    "How many days does a CPIO have to respond to an RTI request?",
    "What fees apply to an RTI application?",
    "How do I file a first appeal under the RTI Act?",
    "What is the role of the Central Information Commission?",
]


# --------------------------------------------------------------------------
# Startup resources: loaded exactly once per process via st.cache_resource.
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_resources():
    """Load embedding model + Qdrant + BM25 + chunks once and cache them."""
    return ask.load_resources()  # (embedding_model, qdrant_client, bm25, chunks)


@st.cache_resource(show_spinner=False)
def get_groq_client():
    """Reuse a single Groq client across questions."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in .env")
    return Groq(api_key=api_key)


# --------------------------------------------------------------------------
# Pipeline helpers (mirror api.py's /ask flow, staying in-process).
# --------------------------------------------------------------------------

def _detect_graph_expansion(fn):
    """Run fn while capturing stdout; report whether graph expansion fired."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn()
    return result, any(ln.startswith("Graph expansion:") for ln in buf.getvalue().splitlines())


def source_detail(chunks, cid, src):
    """Break a cited source into document/section/subsection/clause fields."""
    c = chunks[cid]
    info = {"label": src, "document": c.get("source_document") or ""}
    if c.get("document_type") == "act":
        info["section"] = str(c.get("section_number") or "")
        info["subsection"] = str(c.get("subsection") or "")
        info["clause"] = str(c.get("clause") or "")
    else:
        info["section"] = ""
        info["subsection"] = ""
        info["clause"] = ""
    if c.get("paragraph_number"):
        info["paragraph"] = c.get("paragraph_number")
    return info


def run_pipeline(question):
    """Full retrieve -> grade -> (rewrite) -> generate -> verify flow.

    Returns a dict: {question, answer, verified, unsupported, refused,
    graph_triggered, retrieval_attempts, passages}.
    """
    model, _client, bm25, chunks = load_resources()
    client_groq = get_groq_client()

    query = question
    passages = None
    grade = None
    failed_queries = []
    graph_triggered = False
    attempts = 0

    for attempt in range(1, ask.MAX_RETRIEVAL_ATTEMPTS + 1):
        attempts = attempt
        passages, trig = _detect_graph_expansion(
            lambda: ask.retrieve(model, bm25, chunks, query)
        )
        graph_triggered = graph_triggered or trig
        grade = ask.grade_sufficiency(client_groq, question, passages)
        if grade == "sufficient":
            break
        if attempt == ask.MAX_RETRIEVAL_ATTEMPTS:
            break
        failed_queries.append(query)
        query = ask.rewrite_query(client_groq, question, failed_queries)

    out = {
        "question": question,
        "retrieval_attempts": attempts,
        "graph_triggered": graph_triggered,
        "passages": passages,
    }

    if not passages or grade != "sufficient":
        # Graceful refusal: neutral informational behaviour, not an error.
        out["refused"] = True
        out["answer"] = (
            "I don't have enough information in my knowledge base "
            "to answer this confidently."
        )
        out["verified"] = None
        out["unsupported"] = []
        return out

    out["refused"] = False

    reply = ask.answer(client_groq, question, passages)
    reply = re.sub(r"【(\d{1,2})[^】]*】", r"[\1]", reply)

    passed, unsupported = ask.verify_answer(client_groq, question, reply, passages)
    if passed:
        verified = True
    else:
        # One restorative regeneration, exactly like the CLI / API path.
        strict = ask.answer(client_groq, question, passages, strict=True)
        strict = re.sub(r"【(\d{1,2})[^】]*】", r"[\1]", strict)
        strict_ok, strict_unsupported = ask.verify_answer(
            client_groq, question, strict, passages
        )
        if strict_ok:
            reply = strict
            passed = True
            unsupported = strict_unsupported
        verified = passed

    out["answer"] = reply
    out["verified"] = verified
    out["unsupported"] = unsupported or []
    return out


# --------------------------------------------------------------------------
# Rendering helpers.
# --------------------------------------------------------------------------

def _chip(text, bg, fg):
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:999px;font-size:13px;font-weight:500;'
        f'display:inline-block">{text}</span>'
    )


def render_answer(result):
    if result.get("refused"):
        st.info(result["answer"])
        return

    # Verification + graph-expansion badges.
    chips = []
    if result["verified"]:
        chips.append(_chip("✓ Verified — claims grounded in sources", "#d1fae5", "#065f46"))
    else:
        chips.append(_chip("⚠️ Review needed — some claims not fully verified", "#fef3c7", "#92400e"))
    if result["graph_triggered"]:
        chips.append(_chip("📎 Related case law found", "#e0f2fe", "#075985"))
    if chips:
        st.markdown(
            f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">'
            + "".join(chips)
            + "</div>",
            unsafe_allow_html=True,
        )

    # Answer text.
    st.markdown(result["answer"])

    # Unsupported claims (if verification was not clean).
    if not result["verified"] and result.get("unsupported"):
        with st.expander(
            f"⚠️ {len(result['unsupported'])} claim(s) flagged as not fully supported"
        ):
            for claim in result["unsupported"]:
                st.markdown(f"- {claim}")

    # Cited sources.
    chunks = load_resources()[3]
    passages = result["passages"]
    with st.expander(f"📚 Cited sources ({len(passages)})"):
        for i, (cid, src, _text) in enumerate(passages, 1):
            detail = source_detail(chunks, cid, src)
            parts = []
            if detail["document"]:
                parts.append(detail["document"])
            if detail["section"]:
                sec = f"Section {detail['section']}"
                if detail["subsection"]:
                    sec += f"({detail['subsection']})"
                    if detail["clause"]:
                        sec += f"({detail['clause']})"
                parts.append(sec)
            st.markdown(f"**[{i}] {src}**")
            if parts:
                st.caption(" · ".join(parts))
            st.write("")

    st.caption(
        f"Retrieval attempts: {result['retrieval_attempts']} · Model: {ask.GROQ_MODEL}"
    )


# --------------------------------------------------------------------------
# App.
# --------------------------------------------------------------------------

st.set_page_config(page_title=PAGE_TITLE, page_icon="⚖️", layout="wide")

with st.sidebar:
    st.title("⚖️ " + PAGE_TITLE)
    st.caption(PAGE_DESC)
    st.divider()
    st.subheader("Example questions")
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if st.button(q, use_container_width=True, key=f"example_{i}"):
            st.session_state["queued_question"] = q
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()
    st.caption(
        f"Pipeline: {ask.GROQ_MODEL} · hybrid retrieval + cross-encoder "
        "re-ranking + hallucination verification"
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render prior conversation.
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_answer(msg["data"])
        else:
            st.markdown(msg["content"])

# Get the next question: from the chat input or a sidebar example button.
prompt = st.chat_input("Ask a question about the RTI Act, procedures, or case law…")
queued = st.session_state.pop("queued_question", None)
question = (prompt or queued or "").strip()

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving, grading, generating and verifying…"):
            try:
                result = run_pipeline(question)
            except Exception as exc:  # noqa: BLE001 - friendly, non-breaking error
                st.error(
                    "Something went wrong while running the pipeline. "
                    f"Details: {exc}"
                )
                # Drop only the failed turn so history stays intact.
                if st.session_state["messages"] and st.session_state["messages"][-1].get("role") == "assistant":
                    st.session_state["messages"].pop()
                st.session_state["messages"].pop()  # the user turn that failed
                result = None
        if result is not None:
            render_answer(result)
    if result is not None:
        st.session_state["messages"].append({"role": "assistant", "data": result})