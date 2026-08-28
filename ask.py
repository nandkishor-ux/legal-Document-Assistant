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
MAX_RETRIEVAL_ATTEMPTS = 3

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

GRADE_PROMPT = (
    "You are grading whether a set of retrieved legal source passages contains "
    "enough information to answer a user's question.\n"
    "Respond with exactly one word, nothing else: either \"sufficient\" or "
    "\"insufficient\".\n"
    "\"sufficient\" means the passages together cover what the question asks "
    "(e.g. the relevant statutory provision and its content). "
    "\"insufficient\" means the passages are unrelated, too vague, or clearly "
    "lack the information needed to answer the question."
)

REWRITE_PROMPT = (
    "You are improving a search query for a legal document retrieval system "
    "about Indian Right to Information (RTI) law.\n"
    "Rephrase the user's question into a single, more targeted search query: "
    "use precise legal terminology, spell out relevant concepts, or break a "
    "vague question into a focused one.\n"
    "Respond with ONLY the rewritten query, nothing else."
)

VERIFY_PROMPT = (
    "You are verifying a legal answer generated from retrieved source chunks.\n"
    "The answer cites sources as [1], [2], etc. The retrieved source chunks are "
    "numbered below with the same numbers.\n"
    "Check whether EVERY factual claim in the answer (facts, figures, amounts, "
    "dates, section numbers, legal rules) actually appears in the cited source "
    "chunk(s). A claim is unsupported if the cited source chunk does not state "
    "it.\n"
    "If every claim is supported, respond with exactly:\n"
    "VERDICT: SUPPORTED\n"
    "If any claim is not supported, respond with:\n"
    "VERDICT: UNSUPPORTED\n"
    "<one bullet line per unsupported claim, quoting the claim and naming the "
    "source number it was not found in>\n"
    "Do not add anything else."
)

STRICT_TAIL = (
    "\n\nIMPORTANT: Stick STRICTLY and ONLY to what is stated in the passages. "
    "Do not add any fee amounts, sums of money, dates, section numbers, or other "
    "details that are not present in the passages. If a requested detail is not "
    "stated in the passages, say it is not specified in the sources."
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


def answer(client_groq, question, passages, strict=False):
    numbered = "\n\n".join(
        f"[{i + 1}] Source: {src}\n{text}" for i, (_cid, src, text) in enumerate(passages)
    )
    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"SOURCE PASSAGES:\n{numbered}\n\n"
        "Answer the question using only these passages, citing sources as [1], [2], ..."
    )
    if strict:
        user_prompt += STRICT_TAIL
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


def parse_verdict(content):
    low = content.lower()
    if re.search(r"\bunsupported\b", low):
        if re.search(r"no\s+unsupported", low):
            return True, []
        claims = []
        for line in content.splitlines():
            ls = line.strip().lstrip("*•-").strip()
            if not ls or ls.lower().startswith("verdict"):
                continue
            claims.append(ls)
        return False, claims
    if re.search(r"\bsupported\b", low):
        return True, []
    return False, []


def verify_answer(client_groq, question, reply, passages):
    numbered = "\n\n".join(
        f"[{i + 1}] Source: {src}\n{text}" for i, (_cid, src, text) in enumerate(passages)
    )
    user_prompt = (
        f"QUESTION:\n{question}\n\nANSWER TO VERIFY:\n{reply}\n\n"
        f"RETRIEVED SOURCE CHUNKS:\n{numbered}"
    )
    resp = client_groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": VERIFY_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    return parse_verdict(resp.choices[0].message.content or "")


def grade_sufficiency(client_groq, question, passages):
    numbered = "\n\n".join(
        f"[{i + 1}] Source: {src}\n{text[:600]}" for i, (_cid, src, text) in enumerate(passages)
    )
    resp = client_groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": GRADE_PROMPT},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nRETRIEVED PASSAGES:\n{numbered}"},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    content = (resp.choices[0].message.content or "").lower()
    if re.search(r"insufficient|not\s*s?ufficient|not enough|does not", content):
        return "insufficient"
    if "sufficient" in content:
        return "sufficient"
    return "insufficient"


def rewrite_query(client_groq, question, prev_queries=None):
    lines = [f"ORIGINAL QUESTION: {question}", ""]
    if prev_queries:
        lines.append("The retrieval queries below did NOT find enough information to answer it:")
        lines.extend(f"- {q}" for q in prev_queries)
        lines.append("")
        lines.append(
            "Write a NEW single search query that is DIFFERENT from every query "
            "listed above: use precise legal terminology from the Indian Right to "
            "Information Act, 2005, name the relevant section and concept, and "
            "focus it so a legal document retrieval system finds the right statute "
            "text."
        )
    else:
        lines.append(
            "Write a single targeted search query using precise legal terminology "
            "from the Indian Right to Information Act, 2005, naming the relevant "
            "section and concept, so a legal document retrieval system finds the "
            "right statute text."
        )
    lines.append("Respond with ONLY the rewritten query.")
    resp = client_groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ],
        temperature=0.4,
        max_tokens=2048,
    )
    content = (resp.choices[0].message.content or "").strip().strip("\"'")
    return content or question


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
        client_groq = Groq(api_key=api_key)

        query = question
        passages = None
        grade = None
        failed_queries = []

        for attempt in range(1, MAX_RETRIEVAL_ATTEMPTS + 1):
            print(f"\nRetrieving (attempt {attempt}) for: {query}\n")
            passages = retrieve(model, bm25, chunks, query)
            print(f"Top {len(passages)} passages:")
            for i, (_cid, src, text) in enumerate(passages, 1):
                snippet = " ".join(text.split()[:24])
                print(f"  [{i}] ({src}) {snippet}...")

            print(f"\nAttempt {attempt}: grading retrieved chunks...")
            grade = grade_sufficiency(client_groq, question, passages)
            if grade == "sufficient":
                print(
                    f"Attempt {attempt}: retrieval graded SUFFICIENT, proceeding to generation"
                )
                break

            if attempt == MAX_RETRIEVAL_ATTEMPTS:
                print(
                    f"Attempt {attempt}: retrieval graded INSUFFICIENT; "
                    "no retries left, stopping"
                )
                break
            print(
                f"Attempt {attempt}: retrieval graded INSUFFICIENT, rewriting query..."
            )
            failed_queries.append(query)
            query = rewrite_query(client_groq, question, failed_queries)
            print(f"Rewritten query: {query}")

        if not passages or grade != "sufficient":
            print(
                "\nI don't have enough information in my knowledge base to answer "
                "this confidently."
            )
            return

        print("\nGenerating answer with", GROQ_MODEL, "...\n")
        reply = answer(client_groq, question, passages)
        reply = re.sub(r"【(\d{1,2})】", r"[\1]", reply)

        print("\nVerifying answer against sources...")
        passed, unsupported = verify_answer(client_groq, question, reply, passages)
        caveat = False
        if passed:
            print("Verification passed — all claims supported")
        else:
            if unsupported:
                print(
                    f"Verification found {len(unsupported)} unsupported claim(s), "
                    "regenerating..."
                )
                for i, claim in enumerate(unsupported, 1):
                    print(f"  Unsupported #{i}: {claim}")
            else:
                print("Verification flagged unsupported claims, regenerating...")
            reply = answer(client_groq, question, passages, strict=True)
            reply = re.sub(r"【(\d{1,2})】", r"[\1]", reply)
            print("Re-running verification on regenerated answer...")
            passed, unsupported = verify_answer(client_groq, question, reply, passages)
            if passed:
                print("Verification passed — all claims supported")
            else:
                caveat = True
                print(
                    "Verification still flagged unsupported claims after "
                    "regeneration; showing answer with caveat."
                )
                for i, claim in enumerate(unsupported, 1):
                    print(f"  Unsupported #{i}: {claim}")

        print("=" * 80)
        print("ANSWER")
        print("=" * 80)
        if caveat:
            print(
                "\nNote: some parts of this answer could not be fully verified "
                "against the source documents.\n"
            )
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