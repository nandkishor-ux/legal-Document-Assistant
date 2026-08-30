"""Diagnostic: Delhi RTI Act 2001 indexing + retrieval quality (M1/M2/M3)."""

import io
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from index import CHUNKS_JSON, COLLECTION, label_for  # noqa: E402
from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: E402
from ask import load_resources, tokenize  # noqa: E402

DELHI = "Delhi RTI Act 2001"
M1_Q = ("Compare the time limit within which information must be furnished "
        "under the Delhi RTI Act 2001 with the time limit under the Central "
        "RTI Act 2005.")

buf = []


def out(s=""):
    buf.append(str(s))


def preview(text, n):
    return text[:n].replace("\n", "  ")


def find_snippets(chunks, delhi_idx, keywords, radius=110):
    out("")
    out("=" * 100)
    out("DELHI SECTION TEXTS: keyword sweep across all Delhi chunks")
    out("=" * 100)
    for kw in keywords:
        out("")
        out(f"-- keyword: {kw!r}")
        hits = 0
        for i in delhi_idx:
            text = chunks[i]["text"]
            for m in re.finditer(re.escape(kw.lower()), text.lower()):
                s = max(0, m.start() - radius)
                e = min(len(text), m.end() + radius)
                out(f"  [{i}] {label_for(chunks[i])}")
                out(f"      ...{text[s:e]!r}...")
                hits += 1
                break
        out(f"  ({hits} chunk(s) hit)")
    return


def main():
    chunks = json.load(open(CHUNKS_JSON, encoding="utf-8"))
    emb_model, client, bm25, _ = load_resources()

    # ---- 1) per-source counts ----
    counts = {}
    delhi_idx = []
    for i, c in enumerate(chunks):
        src = c.get("source_document", "")
        counts[src] = counts.get(src, 0) + 1
        if src == DELHI:
            delhi_idx.append(i)
    out("PER-SOURCE CHUNK COUNTS")
    for src, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        out(f"  {n:>4}  {src}")
    out(f"\nDelhi RTI Act 2001: {len(delhi_idx)} chunks")

    out("")
    out("DELHI CHUNK INVENTORY (section / paragraph)")
    sec_tally = {}
    for i in delhi_idx:
        c = chunks[i]
        if c.get("document_type") == "act":
            k = (c.get("section_number"), c.get("chunk_type"))
            sec_tally[k] = sec_tally.get(k, 0) + 1
            if c.get("chunk_type") == "parent":
                out(f"  sec {c.get('section_number'):>3} parent: {preview(c['text'], 60)} ...")
        else:
            out(f"  para {c.get('paragraph_number'):>4}: {preview(c['text'], 60)} ...")
    out(f"act chunk breakdown (section,type)->count: "
        f"{ {f'{k[0]}/{k[1]}': v for k, v in sorted(sec_tally.items(), key=lambda kv: (kv[0][0] or '0', kv[0][1]))} }")

    # ---- 2) vector search: Delhi-only vs combined ----
    qvec = emb_model.encode([M1_Q], normalize_embeddings=True)[0]
    f = Filter(must=[FieldCondition(key="source_document",
                                    match=MatchValue(value=DELHI))])

    def q(flt, limit):
        return client.query_points(
            collection_name=COLLECTION,
            query=qvec.tolist(),
            query_filter=flt,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        ).points

    out("")
    out("=" * 100)
    out("VECTOR SEARCH (M1 query) — DELHI-ONLY, top 8 with cosine scores")
    out("=" * 100)
    hits = q(f, 8)
    for p in hits:
        out(f"  score={p.score:.4f}  {label_for(p.payload)}")

    out("")
    out("VECTOR SEARCH (M1 query) — COMBINED (all docs), top 10")
    hits = q(None, 10)
    for p in hits:
        out(f"  score={p.score:.4f}  {label_for(p.payload)}")

    # ---- BM25 cross-check on Delhi only ----
    q_toks = tokenize(M1_Q)
    scores = [(i, bm25.get_scores(q_toks)[i]) for i in delhi_idx if bm25.get_scores(q_toks)[i] > 0]
    scores.sort(key=lambda kv: kv[1], reverse=True)
    out("")
    out("BM25 (M1 query) — DELHI-ONLY, top 8")
    for i, s in scores[:8]:
        out(f"  score={s:>8.2f}  {label_for(chunks[i])}")

    # ---- 3/4) keyword sweep on Delhi sections ----
    find_snippets(chunks, delhi_idx, [
        "furnish", "within thirty", "time limit", "subject to", "penalty",
        "fine", "exceeding", "commercial", "trade secret", "exemption",
    ])

    open("delhi_diag.txt", "w", encoding="utf-8").write("\n".join(buf))
    print("wrote", len(buf), "lines")


if __name__ == "__main__":
    main()