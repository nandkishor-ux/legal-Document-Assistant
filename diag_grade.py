"""Diagnostic: inspect retrieval + sufficiency grading in isolation.

Runs ONLY retrieve() then the grading call (no answer/verify/rewrite) for a set
of question ids, printing the question, retrieved chunk previews, and the raw
grading response (content + reasoning) parsed with the same logic ask.py uses.

Default models: opensai/gpt-oss-120b (the pipeline model that refused) and
openai/gpt-oss-20b (the model that answered) for contrast.
"""

import argparse
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("RAGAS_TELEMETRY_DISABLED", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv  # noqa: E402
from groq import Groq  # noqa: E402

import ask  # noqa: E402
from ask import GRADE_CHUNK_LIMIT, load_resources, retrieve  # noqa: E402
from eval_set import EVAL_SET  # noqa: E402

CHUNK_PREVIEW = 200

_MODEL_ALIASES = {
    "20b": "openai/gpt-oss-20b",
    "120b": "openai/gpt-oss-120b",
    "full": "openai/gpt-oss-120b",
}


def grade_with_raw(client_groq, model_id, question, passages):
    """Same wire format + parse as ask.grade_sufficiency, but keeps raw response."""
    numbered = "\n\n".join(
        f"[{i + 1}] Source: {src}\n{text[:GRADE_CHUNK_LIMIT]}"
        for i, (_cid, src, text) in enumerate(passages)
    )
    resp = client_groq.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": ask.GRADE_PROMPT},
            {"role": "user",
             "content": f"QUESTION:\n{question}\n\nRETRIEVED PASSAGES:\n{numbered}"},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    msg = resp.choices[0].message
    content = (msg.content or "")
    reasoning = getattr(msg, "reasoning_content", None) or ""
    low = content.lower()
    if re.search(r"insufficient|not\s*s?ufficient|not enough|does not", low):
        verdict = "insufficient"
    elif "sufficient" in low:
        verdict = "sufficient"
    else:
        verdict = "insufficient"
    return verdict, content, reasoning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=str,
                        default="F1,F3,G2,M1,M2,M3",
                        help="comma-separated eval ids to diagnose")
    parser.add_argument("--models", type=str, default="120b,20b",
                        help="comma-separated model aliases to grade with")
    parser.add_argument("--preview", type=int, default=CHUNK_PREVIEW,
                        help="chars of each passage to preview")
    parser.add_argument("--keywords", type=str, default="",
                        help="comma-separated evidence keywords to locate per "
                             "passage, reporting absolute position and whether "
                             "it falls inside the grading window")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found.")
        sys.exit(1)

    wanted = {s.strip().upper() for s in args.ids.split(",") if s.strip()}
    items = [it for it in EVAL_SET if it["id"] in wanted]
    missing = wanted - {it["id"] for it in items}
    if missing:
        print(f"unknown ids (ignored): {sorted(missing)}")
    if not items:
        print("no questions selected.")
        sys.exit(1)

    models = [_MODEL_ALIASES.get(s.strip().lower(), s.strip())
              for s in args.models.split(",") if s.strip()]
    keywords = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]
    ask.GROQ_MODEL = models[0]
    client_groq = Groq(api_key=api_key)

    emb_model, _client, bm25, chunks = load_resources()
    print(f"Models for grading: {models}\n" + "=" * 100)

    for it in items:
        qid, question = it["id"], it["question"]
        print(f"\n{'=' * 100}")
        print(f"{qid} [{it['category']}] expected_graph_expansion={it['expected_graph_expansion']}")
        print(f"QUESTION: {question}")
        passages = retrieve(emb_model, bm25, chunks, question)
        print(f"RETRIEVED {len(passages)} passages "
              f"(grade window: first {GRADE_CHUNK_LIMIT} chars each):")
        for i, (_cid, src, text) in enumerate(passages, 1):
            preview = text[:args.preview].replace("\n", "  ")
            print(f"  [{i}] {src}\n      {preview}")
            if keywords:
                low = text.lower()
                hits = []
                for kw in keywords:
                    pos = low.find(kw)
                    hits.append(f"\"{kw}\"@{pos}/{'IN-WINDOW' if 0 <= pos < GRADE_CHUNK_LIMIT else 'cut-off' if pos >= 0 else 'absent'}")
                print(f"      evidence: {', '.join(hits)} (len={len(text)})")
        for model in models:
            try:
                verdict, content, reasoning = grade_with_raw(
                    client_groq, model, question, passages)
            except Exception as e:
                print(f"  -- {model} grading FAILED: {type(e).__name__}: {str(e)[:140]}")
                continue
            print(f"\n  GRADE with {model}: {verdict}")
            print(f"    raw content: {json.dumps(content)}")
            if reasoning:
                trunc = reasoning if len(reasoning) <= 1500 else reasoning[:1500] + "…[truncated]"
                print(f"    reasoning ({len(reasoning)} ch):\n    {trunc}")
            else:
                print("    reasoning: (none returned)")
    try:
        ask._client.close()
        client_groq.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()