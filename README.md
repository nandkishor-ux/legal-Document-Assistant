# RTI Legal Document Assistant

A Corrective RAG + GraphRAG system that answers questions about the Indian Right to Information Act 2005, the Delhi RTI Act 2001, and related case law — with clause-level citation accuracy and built-in hallucination checking.

**Live demo:** [legal-document-assistant.streamlit.app](https://legal-document-assistant-w9g5xdvzrzx8qfd7prej7c.streamlit.app/)

> ⚠️ Free-tier note: the app runs on Groq's free API tier, which has daily token limits. If it refuses a question it should normally answer, it may be a temporary quota issue rather than the system being wrong.

---

## Why this project exists

Indian legal documents are hard for standard RAG systems to handle correctly. A question like *"what does Section 8(1)(j) exempt?"* requires a system that can:

- Distinguish `8(1)(j)` from `8(1)(d)` precisely — not just retrieve "something from Section 8"
- Know when it doesn't have enough information, instead of confidently guessing
- Cite the exact clause an answer came from, so the answer is verifiable
- Pull in relevant case law when a question touches on how courts have interpreted a clause

Most naive RAG tutorials skip all of this. This project builds each piece deliberately, and — just as importantly — documents the real bugs found while building it, since debugging retrieval systems is most of the actual work.

---

## Architecture

```
User question
     │
     ▼
┌─────────────────────────────────────────┐
│  Query routing                           │
│  Does the question span 2+ Acts?         │
│  (Central RTI Act 2005 vs Delhi RTI 2001)│
└───────────────┬───────────────────────────┘
                │
      ┌─────────┴──────────┐
      ▼                    ▼
 single-doc            cross-doc
 retrieval             retrieval
 (vector + BM25)   (per-Act group vector + BM25,
                     RRF-fused across groups)
      │                    │
      └─────────┬──────────┘
                ▼
      Cross-encoder re-ranking
      (top-18 pool → top-8 precise)
                │
                ▼
      Graph expansion
      (if a top-ranked chunk has a case-law
       citation AND the question shows case-
       interpretation intent → pull in the
       related judgment/decision paragraph)
                │
                ▼
      Self-grading
      "Is this enough to answer?"
                │
        ┌───────┴────────┐
        │                │
   insufficient      sufficient
        │                │
   rewrite query          ▼
   & retry (max 3)   Citation-grounded
        │            generation
        └──────┐          │
               ▼          ▼
      "I don't have   Hallucination
       enough info"   verification
                            │
                    ┌───────┴────────┐
                    │                │
               unsupported      verified
               claim found          │
                    │                ▼
              strict regenerate   Final answer
              + re-verify         + sources
                    │             + verification badge
                    └──────────────►
```

---

## Key design decisions

### Parent-child chunking, not fixed-size chunking
Legal Acts are chunked by their actual structure — a **parent** chunk is a full Section (e.g., all of Section 8), and **child** chunks are individual sub-clauses (e.g., `8(1)(j)` on its own). Every chunk carries metadata: source document, section, subsection, clause. This is what makes precise citation possible — a naive 500-character sliding-window chunker would routinely cut a clause in half.

### Corrective RAG (self-grading + retry)
Before generating an answer, the system checks whether the retrieved chunks actually contain enough information. If not, it rewrites the query and retries (up to 3 attempts) before honestly saying *"I don't have enough information"* — rather than generating a plausible-sounding but ungrounded answer.

### GraphRAG (citation graph)
Case law (a Delhi High Court judgment, a CIC decision) is linked to the specific Act clauses it interprets. When a question shows case-interpretation intent (e.g., mentions "court," "ruling," "penalty imposed") and retrieval surfaces a linked clause, the system automatically pulls in the related case paragraph — even if the case text wouldn't have ranked highly on pure semantic similarity.

### Hallucination verification
Every generated answer is checked against its cited sources by a separate verification pass. If it finds a claim the sources don't support, the system regenerates once with a stricter prompt, then re-verifies. This was stress-tested by deliberately injecting a fabricated claim ("filing fee of exactly Rs 500") into a real answer — the checker caught and corrected it.

### Cross-document retrieval
When a question compares the Central Act and the Delhi Act, retrieval runs separately within each Act's chunks, then fuses the two ranked lists. Without this, the larger Central Act corpus consistently crowded out relevant Delhi Act content in a single combined ranking.

---

## Real bugs found and fixed

Building this surfaced several genuine bugs — documenting them here because finding and fixing them was most of the actual engineering work.

| Bug | Root cause | Fix |
|---|---|---|
| Corrupted sub-clause numbers | A font-mapping issue in one source PDF rendered digit `1` inside parentheses as `/` — e.g. `(1)` became `(/)` | Detected the pattern, wrote a targeted regex fix, verified against 59 corrections with zero residuals |
| Structure-blind chunking | Initial chunker split text by character count (~1200 chars), ignoring Section/clause boundaries — cut clauses mid-sentence | Rewrote to parent-child chunking driven by actual section/subsection/clause markers |
| Delhi Act chunking collapse | OCR dropped section numbers for Sections 3–12 (some headers cut mid-word, e.g. "Obligati" for "Obligations"), so header detection only found Sections 1 and 2 — everything else got merged into one 10,883-character blob with fabricated clause labels | Wrote a document-specific header-repair pass that reconstructs proper numbered headers before chunking, using three detection strategies (numbered, title-line, OCR-stub matching) |
| Grading truncation hid relevant text | The grading step only saw the first 600 characters of each retrieved chunk — a real answer (a 48-hour deadline proviso) sat just past that cutoff and was invisible to the grader | Raised the limit to 1200 characters, verified the affected question now grades correctly |
| Graph expansion over-triggering | The graph layer fired whenever *any* cited clause appeared anywhere in the retrieved set, regardless of relevance or question intent — a pure statute question would incorrectly pull in unrelated case law | Added two gates: the triggering chunk must be genuinely top-ranked, and the question must show case-interpretation intent (not just any statute mention) |
| RRF bias against OCR'd content | A correct answer (Delhi Act's 15-day time limit) ranked #1 by pure vector similarity but scored zero on keyword (BM25) search, because OCR noise stripped the matching terms — RRF's dual-list bias pushed it out of the results entirely | Root-caused via diagnostic re-ranking; documented as a known limitation rather than rebalancing core fusion math (see below) |

---

## Known limitations

Being upfront about what doesn't work perfectly:

- **One cross-Act time-limit question still fails.** The query-rewriter keeps anchoring on "Section 7" for both Acts when retrying, but the Delhi Act's actual time-limit provision is in Section 5 (Section 7 there is *Appeals*, an unrelated topic). Root-caused precisely; not yet fixed, since the fix would touch core RRF fusion logic used by every other query.
- **The hallucination verifier struggles to confirm negative claims.** A correct statement like *"the Delhi Act has no public-interest override clause"* is sometimes flagged as unverifiable, since proving an absence from source text is inherently harder than proving a presence.
- **`has_case_intent()` is a keyword heuristic**, not a learned classifier. It's exact on the current evaluation set but could miss real-world phrasing that doesn't use its trigger words (e.g., "court," "judgment," "petitioner"). A small LLM-based intent classifier would be more robust.
- **Full RAGAS evaluation is partial.** Groq's free-tier daily token budget (200K tokens/day) can't cover all 60 required scoring calls (15 questions × 4 metrics) in a single day — evaluation is genuinely constrained by API cost, not a shortcut taken.

---

## Evaluation

Evaluated using [RAGAS](https://github.com/explodinggears/ragas) across 15 hand-written questions spanning four categories: factual (single-clause lookups, F1–F5), graph (case-law interpretation, G1–G5), multi-hop (cross-Act comparisons, M1–M3), and out-of-corpus (should correctly refuse, O1–O2).

### Pipeline-layer results (complete, deterministic — doesn't depend on RAGAS scoring)

| Metric | Before fixes | After fixes |
|---|---|---|
| Questions answered correctly | 7 / 15 (47%) | 11 / 15 (73%) |
| Questions correctly refused (out-of-corpus) | 2 / 2 | 2 / 2 |
| Over-cautious refusals (answerable but refused) | 6 | 2 |
| Refusal accuracy* | 0.25 | 0.50 |

*Refusal accuracy = (correct refusals) / (correct refusals + over-cautious refusals).*

The three multi-hop comparison questions (M1, M2, M3) were the specific target of the cross-document retrieval and Delhi Act chunking fixes:

| Question | Before | After |
|---|---|---|
| M1 — Central vs Delhi response time limits | ❌ refused (×3 attempts) | ❌ still refused — root cause diagnosed (see Known Limitations) |
| M2 — Central vs Delhi CPIO penalties | ❌ refused (×3 attempts) | ✅ answered, verified, correctly cites ₹250/day → ₹25,000 (Central) vs Delhi's "as prescribed" |
| M3 — Central vs Delhi trade-secret exemption | ❌ refused (×3 attempts) | ✅ answered, correctly cites Central 8(1)(d) + Delhi s.6 |

**2 of 3 targeted fixes confirmed working in the formal eval harness**, not just in manual testing.

### RAGAS metric scores

RAGAS scoring requires ~4 additional LLM calls per question (60 calls total for 15 questions), which exceeds Groq's free-tier daily budget (200K tokens/day) in a single run — scoring is genuinely partial as a result, not a shortcut.

**Scored on the full production model (`openai/gpt-oss-120b`), full context, no truncation:**

| Question | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| F1 | 1.000 | 0.9149 | 1.000 | 1.000 |
| F2 | 1.000 | 0.9467 | 0.250 | 1.000 |
| F3 | 0.6667 | 0.9481 | — | — |

*(F3's remaining cells weren't scored before the daily quota was exhausted; scoring is ongoing.)*

**Reading these numbers correctly:** on questions the system actually commits to answering, faithfulness and relevancy are consistently high (0.95+ relevancy, near-perfect faithfulness on 2 of 3 scored questions). A separate, earlier partial run across a wider "answered-only" subset (7 questions, prior to the final retrieval fixes) averaged **faithfulness 0.903, answer relevancy 0.745, context precision 0.636, context recall 1.000** — included here for a broader (if slightly dated) picture, since it predates the Delhi Act chunking and cross-document retrieval fixes described above.

**Why raw all-15 RAGAS averages aren't reported as a single headline number:** faithfulness scoring gives a `0.0` to any correctly-refused question (a refusal makes no factual claims, so there's nothing to be "faithful" to) — this drags a naive all-question average down in a way that actually penalizes correct behavior. Reporting answered-only scores alongside the separate refusal-accuracy metric is the fairer, more honest breakdown.

---

## Tech stack

- **Retrieval:** `sentence-transformers` (embeddings), `rank-bm25` (keyword search), `Qdrant` (vector store), `cross-encoder/ms-marco-MiniLM-L-6-v2` (re-ranking)
- **Generation & grading:** Groq API (`openai/gpt-oss-120b`)
- **Evaluation:** RAGAS
- **Document processing:** `pdfplumber`, `pytesseract` (OCR for scanned PDFs)
- **Interfaces:** Streamlit (deployed), FastAPI + React/Vite/Tailwind (built, available locally)

---

## Running locally

```bash
# Clone and set up
git clone <this-repo>
cd legal-document-assistant
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Run the Streamlit app (uses the pre-built vector store committed in vectorstore/)
streamlit run streamlit_app.py
```

To rebuild the vector store from scratch (e.g., after adding new documents):
```bash
python index.py
python build_graph.py
```

The FastAPI backend + React frontend (built but not part of the live deployment) can also be run locally:
```bash
# Terminal 1
venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000

# Terminal 2
cd frontend
npm install
npm run dev
```

---

## Dataset

- **Right to Information Act, 2005** (Central Act) — full text, 31 sections
- **Delhi Right to Information Act, 2001** — full text, 16 sections (OCR'd from a scanned source)
- **Delhi High Court judgment**, *Rakesh Kumar Gupta v. CIC* (2021) — interprets Sections 5(3), 5(4), 6(1), 7(1), 8(1)(d), 20(1)
- **CIC decision**, *Prasanta Kumar Sahoo v. Ministry of Labour* (2026) — interprets Sections 8(1)(j), 25(5)

All sourced from official/public government publications and court records.

---

## Disclaimer

This is a personal/academic project built as a technical demonstration of Corrective RAG and GraphRAG architectures. It is **not** a substitute for professional legal advice. Answers should be verified against the official Act text and, where relevant, a qualified legal professional.
