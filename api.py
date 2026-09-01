"""FastAPI backend exposing the existing RAG pipeline (ask.py) as a web API.

Endpoints:
  GET  /health  -> {"status": "ok"}
  POST /ask     -> {question: "..."} -> final answer, cited sources,
                   verification, graph-expansion and retrieval-attempt info.

The heavyweight resources (embedding model, Qdrant vector store, BM25 index,
cross-encoder, Groq client) are loaded exactly once at server startup via
ask.load_resources(), not per request.

Run with:
    venv\\Scripts\\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
Interactive docs (OpenAPI) are served at http://127.0.0.1:8000/docs
"""

import contextlib
import io
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv
from groq import Groq

import ask
from index import label_for

load_dotenv()


app = FastAPI(
    title="RTI Legal Document Assistant API",
    description=(
        "Hybrid retrieval + re-ranking + generation + hallucination verification "
        "over the Indian Right to Information corpus (Central RTI Act 2005, "
        "Delhi RTI Act 2001, case law)."
    ),
    version="1.0.0",
)

# Allow the browser-based React frontend (Vite dev server) to call the API.
# The dev server defaults to http://localhost:5173; also permit 127.0.0.1.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- single-time startup resources ------------------------------------------
_resources = None
_client_groq = None


def _get_resources():
    """Load the heavy retrieval stack once and cache it for the process lifetime."""
    global _resources
    if _resources is None:
        _resources = ask.load_resources()  # (embedding_model, qdrant_client, bm25, chunks)
    return _resources


def _get_groq_client():
    """Reuse a single Groq client across requests."""
    global _client_groq
    if _client_groq is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        _client_groq = Groq(api_key=api_key)
    return _client_groq


# ---- request/response schemas ------------------------------------------------

class AskRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    document: str
    section: str = ""
    subsection: str = ""
    clause: str = ""
    label: str


class AskResponse(BaseModel):
    answer: str
    sources: list
    verified: bool | None = None
    unsupported: list = []
    graph_expansion_triggered: bool
    retrieval_attempts: int
    refused: bool
    model: str


# ---- pipeline helpers ---------------------------------------------------------

def _chunk_source(chunks, cid, src):
    """Break a cited source into document-name and section/clause fields."""
    c = chunks[cid]
    if c.get("document_type") == "act":
        return SourceInfo(
            document=c.get("source_document") or "",
            section=str(c.get("section_number") or ""),
            subsection=str(c.get("subsection") or ""),
            clause=str(c.get("clause") or ""),
            label=src,
        )
    if c.get("paragraph_number"):
        return SourceInfo(
            document=c.get("source_document") or "",
            section="",
            subsection="",
            clause="",
            label=src,
        )
    return SourceInfo(document=c.get("source_document") or "", label=src)


def _detect_graph_expansion(fn):
    """Run fn while capturing stdout; report whether graph expansion fired."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn()
    lines = buf.getvalue().splitlines()
    triggered = any(ln.startswith("Graph expansion:") for ln in lines)
    return result, triggered


# ---- endpoints ----------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="'question' must not be empty")

    model, _tok, bm25, chunks = _get_resources()
    client_groq = _get_groq_client()
    question = req.question.strip()

    try:
        query = question
        passages = None
        grade = None
        failed_queries = []
        graph_triggered = False
        attempts = 0

        # --- retrieval loop: retrieve -> grade -> (rewrite if insufficient) ---
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

        if not passages or grade != "sufficient":
            # graceful stop: no confident answer
            return AskResponse(
                answer=("I don't have enough information in my knowledge base "
                        "to answer this confidently."),
                sources=[],
                verified=None,
                unsupported=[],
                graph_expansion_triggered=graph_triggered,
                retrieval_attempts=attempts,
                refused=True,
                model=ask.GROQ_MODEL,
            )

        # --- generation -----------------------------------------------------
        reply = ask.answer(client_groq, question, passages)
        reply = re.sub(r"【(\d{1,2})[^】]*】", r"[\1]", reply)

        # --- verification (hallucination check) ------------------------------
        passed, unsupported = ask.verify_answer(client_groq, question, reply, passages)
        if passed:
            verified = True
        else:
            # one restorative regeneration, exactly like the CLI path
            strict = ask.answer(client_groq, question, passages, strict=True)
            strict = re.sub(r"【(\d{1,2})[^】]*】", r"[\1]", strict)
            strict_ok, strict_unsupported = ask.verify_answer(
                client_groq, question, strict, passages)
            if strict_ok:
                reply = strict
                passed = True
                unsupported = strict_unsupported
            verified = passed

        sources = [_chunk_source(chunks, cid, src) for (cid, src, _t) in passages]

        return AskResponse(
            answer=reply,
            sources=[s.dict() if hasattr(s, "dict") else s for s in sources],
            verified=verified,
            unsupported=unsupported or [],
            graph_expansion_triggered=graph_triggered,
            retrieval_attempts=attempts,
            refused=False,
            model=ask.GROQ_MODEL,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a clean JSON error
        return JSONResponse(
            status_code=500,
            content={
                "error": "pipeline_error",
                "message": str(exc)[:500],
            },
        )
