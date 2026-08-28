import json
import os
import re
import sys

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from index import COLLECTION, EMBED_DIM, MODEL_NAME, QDRANT_PATH, CHUNKS_JSON, STOPWORDS, label_for

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
TOP_K = 8

SYSTEM_PROMPT = (
    "You are a legal research assistant specialising in Indian Right to "
    "Information (RTI) law. Answer the user's question using ONLY the provided "
    "source passages. If the passages do not contain enough information, say so "
    "explicitly.\n"
    "CITATION FORMAT: cite the source number in square brackets immediately "
    "after the sentence that relies on it, e.g. \"...exempt under Section 8(1)(d). "
    "[1]\" Never use other citation styles such as footnotes or 【 】.\n"
    "Do not state section numbers or facts that the passages do not support. "
    "Be precise and concise; quote statutory text where relevant."
)

_tokenizer = None
_client = None
_bm25 = None
_chunks = None


def tokenize(text):
    toks = re.findall(r"[A-Za-z]+", text.lower())
    return [t for t in toks if len(t) >= 3 and t not in STOPWORDS]


def load_resources():
    global _tokenizer, _client, _bm25, _chunks
    if _tokenizer is not None:
        return _tokenizer, _client, _bm25, _chunks

    with open(CHUNKS_JSON, encoding="utf-8") as f:
        _chunks = json.load(f)
    corpus_tokens = [tokenize(c["text"]) for c in _chunks]
    _bm25 = BM25Okapi(corpus_tokens)

    _tokenizer = SentenceTransformer(MODEL_NAME)
    _client = QdrantClient(path=QDRANT_PATH)
    return _tokenizer, _client, _bm25, _chunks


def vector_hits(client, query_vec, limit=15):
    res = client.query_points(
        collection_name=COLLECTION,
        query=query_vec.tolist(),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [(hit.id, hit.score) for hit in res.points]


def rrf(ranked_lists, k=60):
    fused = {}
    for lst in ranked_lists:
        for pos, (doc_id, _score) in enumerate(lst):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + pos + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def parse_section_query(question):
    m = re.search(r"(?:section|sec\.?|s\.)\s*(\d{1,2})", question, re.IGNORECASE)
    return m.group(1) if m else None


def retrieve(model, bm25, chunks, question):
    query_vec = model.encode([question], normalize_embeddings=True)[0]
    q_tokens = tokenize(question)

    vec = vector_hits(_client, query_vec)
    kw_hits = [(i, s) for i, s in enumerate(bm25.get_scores(q_tokens)) if s > 0]
    kw_hits.sort(key=lambda kv: kv[1], reverse=True)
    kw_hits = kw_hits[:15]

    fused = dict(rrf([vec, kw_hits]))
    sec = parse_section_query(question)
    if sec:
        for cid, score in list(fused.items()):
            c = chunks[cid]
            if c.get("document_type") == "act" and c.get("section_number") == sec:
                fused[cid] = score + 1.5
                if c.get("chunk_type") == "parent":
                    fused[cid] += 0.5
    order = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    if sec:
        parents = [i for i, c in enumerate(chunks)
                   if c.get("document_type") == "act" and c.get("section_number") == sec
                   and c.get("chunk_type") == "parent"]
        parents.sort(key=lambda i: "consolidated" in chunks[i].get("source_document", ""))
        for i in parents:
            if i not in [p for p, _ in order[:TOP_K]] and len(out) < 2:
                out.append(i)
    for cid, _ in order:
        if cid in out:
            continue
        out.append(cid)
        if len(out) >= TOP_K:
            break
    return [(cid, label_for(chunks[cid]), chunks[cid]["text"]) for cid in out]


def answer(client_groq, question, passages):
    numbered = "\n\n".join(
        f"[{i + 1}] Source: {src}\n{text}" for i, (_cid, src, text) in enumerate(passages)
    )
    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"SOURCE PASSAGES:\n{numbered}\n\n"
        "Answer the question using only these passages, citing sources as [1], [2], ..."
    )
    resp = client_groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    return resp.choices[0].message.content


def main():
    if len(sys.argv) < 2:
        print("usage: python ask.py \"your question\"")
        return

    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found. Check .env")
        return

    model, _, bm25, chunks = load_resources()
    question = " ".join(sys.argv[1:])

    try:
        print(f"Retrieving for: {question}\n")
        passages = retrieve(model, bm25, chunks, question)
        print(f"Top {len(passages)} passages:")
        for i, (cid, src, text) in enumerate(passages, 1):
            snippet = " ".join(text.split()[:28])
            print(f"  [{i}] ({src}) {snippet}...")

        print("\nGenerating answer with", GROQ_MODEL, "...\n")
        client_groq = Groq(api_key=api_key)
        reply = answer(client_groq, question, passages)
        reply = re.sub(r"【(\d{1,2})】", r"[\1]", reply)
        print("=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(reply)

        print("\n" + "=" * 80)
        print("SOURCES")
        print("=" * 80)
        for i, (cid, src, text) in enumerate(passages, 1):
            print(f"[{i}] {src}")
    finally:
        try:
            _client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()