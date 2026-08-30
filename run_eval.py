"""Run the RTI RAG system through RAGAS and emit a metrics report.

Pipeline per question (mirrors ask.py):
  retrieve (hybrid + graph expansion) -> grade sufficiency (retry) ->
  generate -> verify (strict regenerate if flagged)

The generated answer plus the retrieved context chunks are then fed to RAGAS to
compute faithfulness, answer relevancy, context precision and context recall.

Results are printed as a summary table and written to eval_results.json.

Usage:
  python run_eval.py                 # all 15 questions
  python run_eval.py --limit 2       # first 2 questions (smoke test)
  python run_eval.py --ids F1,G4     # specific question ids
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("RAGAS_TELEMETRY_DISABLED", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv  # noqa: E402
from groq import Groq  # noqa: E402

import ask  # noqa: E402
from ask import (  # noqa: E402
    GROQ_MODEL,
    MAX_RETRIEVAL_ATTEMPTS,
    TOP_K,
    answer,
    grade_sufficiency,
    load_resources,
    retrieve,
    rewrite_query,
    verify_answer,
)
from eval_set import EVAL_SET  # noqa: E402

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
EMBED_MODEL = "all-MiniLM-L6-v2"
# Evaluator model for RAGAS: qwen3.8-27b emits ~35 completion tokens/call vs
# ~268+ for the pipeline's gpt-oss-120b reasoning model, keeping us far under
# Groq's 8k TPM free-tier limit (the smoke run 429'd with 120b).
RAGAS_MODEL = "qwen/qwen3.8-27b"
RAGAS_CHUNK_TRIM = 900  # chars kept per retrieved chunk for RAGAS scoring prompts
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
LOW_SCORE_THRESHOLD = 0.5
GROQ_API_KEY_ENV = "GROQ_API_KEY"


def build_ragas_llm(api_key):
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    # Groq is OpenAI-compatible. RAGAS 0.4.3's "groq" provider is buggy
    # (it hardcodes the Anthropic `messages.create`), so we use provider="openai".
    # Metrics score via `await llm.agenerate(...)`, which requires an async client;
    # instructor.from_openai() auto-detects AsyncOpenAI and returns AsyncInstructor.
    client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=api_key)
    _pace_openai_completions(client)  # stay under Groq's 8k TPM free-tier wall
    # max_tokens=2048 is plenty for the evaluator: RAGAS structured outputs are
    # small and qwen3.8-27b is not a reasoning model (no long thinking traces).
    return llm_factory(
        RAGAS_MODEL,
        provider="openai",
        client=client,
        max_tokens=2048,
    )


_RAGAS_TPM_BUDGET = 7200  # stay under Groq's 8000 TPM free-tier limit


def _pace_openai_completions(client):
    """Wrap client.chat.completions.create with a rolling 60s token budget.

    RAGAS decides the LLM call path with `isinstance(metric.llm, InstructorBaseRagasLLM)`,
    so we cannot wrap the LLM object itself. Instead we wrap the underlying async
    OpenAI call; instructor.from_openai() picks this function up as its "create".
    """
    import asyncio
    from collections import deque

    real_create = client.chat.completions.create
    window = deque()
    lock = asyncio.Lock()

    def _est_tokens(messages):
        body = sum(len(m.get("content") or "") for m in messages)
        return max(1, body // 4) + 200  # ~4 chars/token prompt + completion headroom

    async def paced_create(*args, **kwargs):
        tokens = _est_tokens(kwargs.get("messages") or (args[0] if args else []))
        while True:
            async with lock:
                now = time.time()
                while window and now - window[0][0] >= 60.0:
                    window.popleft()
                used = sum(t for _, t in window)
                if used + tokens <= _RAGAS_TPM_BUDGET:
                    window.append((now, tokens))
                    break
            await asyncio.sleep(1.0)
        return await real_create(*args, **kwargs)

    client.chat.completions.create = paced_create


def build_ragas_embeddings():
    from ragas.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model=EMBED_MODEL)


def run_pipeline(client_groq, model, bm25, chunks, question,
                 contexts=None, attempts=None):
    """Run the ask.py pipeline for one question, capturing outputs.

    `contexts` caps how many retrieved chunks are sent into each sufficiency
    grading call (token saving). Generation and verification always see the
    full retrieved set: verification must check the answer against the same
    evidence the answer was generated from, or it would spuriously flag claims
    citing chunks beyond the cap. `attempts` caps the retrieval loop (None =
    ask.py's MAX_RETRIEVAL_ATTEMPTS).
    """
    if attempts is None:
        attempts = MAX_RETRIEVAL_ATTEMPTS
    query = question
    passages = None
    grade = None
    failed_queries = []
    attempts_log = []
    final_graph_lines = []

    for attempt in range(1, attempts + 1):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            passages = retrieve(model, bm25, chunks, query)
        graph_lines = [ln for ln in buf.getvalue().splitlines()
                       if ln.startswith("Graph expansion:")]
        if graph_lines:
            final_graph_lines = graph_lines

        graded = passages[:contexts] if contexts else passages
        grade = grade_sufficiency(client_groq, question, graded)
        attempts_log.append({"attempt": attempt, "query": query, "grade": grade})
        print(f"    [pipeline] attempt {attempt}: {grade} "
              f"({len(passages)} passages, "
              f"{'graph-expanded' if graph_lines else 'no graph'}, "
              f"graded on {len(graded)})")

        if grade == "sufficient":
            break
        if attempt == attempts:
            break
        failed_queries.append(query)
        query = rewrite_query(client_groq, question, failed_queries)
        print(f"    [pipeline] rewritten query: {query}")

    graceful_stop = (not passages) or grade != "sufficient"
    sources = [src for (_cid, src, _t) in (passages or [])]

    if graceful_stop:
        reply = None
        verification_passed = None
        response = ("I don't have enough information in my knowledge base "
                    "to answer this confidently.")
    else:
        reply = answer(client_groq, question, passages)
        reply = re.sub(r"【(\d{1,2})】", r"[\1]", reply)
        verification_passed, _unsupported = verify_answer(
            client_groq, question, reply, passages)
        if not verification_passed:
            strict = answer(client_groq, question, passages, strict=True)
            strict = re.sub(r"【(\d{1,2})】", r"[\1]", strict)
            strict_ok, _ = verify_answer(client_groq, question, strict, passages)
            if strict_ok:
                reply = strict
                verification_passed = True
        response = reply

    retrieved_contexts = [text for (_cid, _src, text) in (passages or [])]
    return {
        "attempts": len(attempts_log),
        "attempts_log": attempts_log,
        "final_grade": grade,
        "graceful_stop": graceful_stop,
        "graph_expansion_triggered": len(final_graph_lines) > 0,
        "graph_lines": final_graph_lines,
        "verification_passed": verification_passed,
        "sources": sources,
        "num_retrieved": len(passages or []),
        "response": response,
        "retrieved_contexts": retrieved_contexts,
    }


def average(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def collect_rows(selected, checkpoint_path=None, done_ids=None,
                 contexts=None, attempts=None, initial=None):
    """Run the ask.py pipeline for each pending item.

    Skip questions whose ids are already in `done_ids` (from an earlier checkpoint).
    Persists each completed (item, out, row) triple to checkpoint_path so a crash /
    Groq 429 mid-run does not lose work. A question that raises RateLimitError is
    skipped (its id stays pending) so a daily-quota hit doesn't abort the run.

    `initial` carries the already-resumed triples so a checkpoint write keeps them
    (checkpoint files are overwritten wholesale, not appended).
    """
    load_dotenv()
    api_key = os.environ.get(GROQ_API_KEY_ENV)
    if not api_key:
        print(f"{GROQ_API_KEY_ENV} not found. Check .env")
        sys.exit(1)

    model, _tok, bm25, chunks = load_resources()
    client_groq = Groq(api_key=api_key)

    pending = [it for it in selected if not done_ids or it["id"] not in done_ids]
    print(f"\nRunning pipeline for {len(pending)} question(s) "
          f"({len(selected) - len(pending)} already in checkpoint)...\n")

    rows = []
    for i, item in enumerate(pending, 1):
        print(f"  ({i}/{len(pending)}) {item['id']} [{item['category']}] "
              f"{item['question'][:70]}")
        t0 = time.time()
        try:
            out = run_pipeline(client_groq, model, bm25, chunks, item["question"],
                               contexts=contexts, attempts=attempts)
        except Exception as e:
            if "429" in str(e) or "Rate limit reached" in str(e):
                print(f"      ! skipped (Groq rate limit): {str(e)[:120]}\n")
                continue
            raise
        out["elapsed_s"] = round(time.time() - t0, 1)
        row = {
            "eval_id": item["id"],
            "category": item["category"],
            "user_input": item["question"],
            "reference": item["ground_truth"],
            "response": out["response"],
            "retrieved_contexts": out["retrieved_contexts"],
        }
        rows.append((item, out, row))
        if checkpoint_path:
            _write_checkpoint(checkpoint_path, (initial or []) + rows)
        print(f"      -> {out['final_grade']}, "
              f"{'graceful-stop' if out['graceful_stop'] else 'answered'} "
              f"in {out['elapsed_s']}s\n")
    try:
        _close_client(client_groq)
    except Exception:
        pass
    return rows


def _write_checkpoint(path, triples):
    payload = [
        {"item": item, "out": out, "row": row}
        for (item, out, row) in triples
    ]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_checkpoint(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return [(d["item"], d["out"], d["row"]) for d in payload]


def _close_client(client_groq):
    # ask.py keeps its own qdrant client open; close it and the Groq client.
    ask._client.close()
    client_groq.close()


def _fmt(v):
    return "  -  " if v is None else f"{v:5.3f}"


def run_ragas_scoring(rows, api_key):
    """Run the RAGAS phase over pipeline rows; returns (scores_by_row, elapsed)."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._context_recall import context_recall
    from ragas.metrics._faithfulness import faithfulness
    from ragas.run_config import RunConfig

    groq_llm = build_ragas_llm(api_key)
    emb = build_ragas_embeddings()

    # evaluate() only accepts legacy `ragas.metrics.base.Metric` instances; the
    # `ragas.metrics.collections` classes are a different (v2) hierarchy that it
    # rejects. The underscore-module singletons below extend the legacy Metric and
    # default to llm=None/embeddings=None, so evaluate() wires them via its
    # llm=/embeddings= arguments. They call the LLM through PydanticPrompt, which
    # loops `await llm.agenerate(...)` for InstructorLLM instances.
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # answer_relevancy's `calculate_similarity` calls embeddings.embed_query() /
    # embed_documents() (the LangChain embedding protocol), which ragas's modern
    # HuggingFaceEmbeddings wrapper does not expose. Pre-set it to a raw LangChain
    # embedding so evaluate() (which only fills embeddings that are None) keeps it.
    from langchain_community.embeddings import SentenceTransformerEmbeddings

    answer_relevancy.embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    # strictness has no effect on InstructorLLM (a single agenerate is produced
    # regardless); keep it at 1 to avoid the misleading n-mismatch warnings.
    answer_relevancy.strictness = 1

    ds_rows = []
    ds_by_id = {}
    for (_item, _out, row) in rows:
        r = dict(row)
        # Score on the FULL retrieved set (every chunk, trimmed to
        # RAGAS_CHUNK_TRIM chars so scoring prompts stay bounded). This is the
        # "real" evaluation; keep a copy per id so missing cells can be retried
        # against exactly the same contexts evaluate() saw.
        r["retrieved_contexts"] = [c[:RAGAS_CHUNK_TRIM] for c in (row["retrieved_contexts"] or [])]
        ds_rows.append(r)
        ds_by_id[_item["id"]] = r
    ds = Dataset.from_list(ds_rows)
    print(f"\nEvaluating with RAGAS ({RAGAS_MODEL}) on {len(ds)} rows ...\n")
    t0 = time.time()
    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=groq_llm,
        embeddings=emb,
        run_config=RunConfig(max_workers=6, max_retries=8, max_wait=25, timeout=600),
        show_progress=True,
    )
    elapsed = round(time.time() - t0, 1)
    df = result.to_pandas() if hasattr(result, "to_pandas") else result.to_dataset().to_pandas()

    def _score_at(idx):
        s = {}
        for name in METRIC_NAMES:
            val = df[name].iloc[idx]
            s[name] = None if val != val else round(float(val), 4)
        return s

    scores_by_row = {}
    for pos, (item, _out, _row) in enumerate(rows):
        idx = pos
        if "eval_id" in df.columns:
            matches = df.index[df["eval_id"] == item["id"]]
            if len(matches):
                idx = matches[0]
        scores_by_row[item["id"]] = _score_at(idx)

    # ---- retry missing cells (timeouts / transient failures) up to 2x ----
    missing = [(pos, item["id"], name)
               for pos, (item, _out, _row) in enumerate(rows)
               for name in METRIC_NAMES
               if scores_by_row[item["id"]].get(name) is None]
    if missing:
        import queue as _queue
        import threading

        from ragas.metrics.base import SingleTurnSample

        by_name = {m.name: m for m in metrics}
        print(f"\n{len(missing)} scoring cell(s) missing in the main pass; "
              f"retrying each up to 2 times ...")

        def _run_with_timeout(fn, timeout):
            # single_turn_score() can hang (an underlying async call that never
            # returns), so run it on a daemon thread and bound the wait. The
            # daemon survives a hang without blocking interpreter shutdown.
            q = _queue.SimpleQueue()

            def _worker():
                try:
                    q.put(("ok", fn()))
                except BaseException as e:  # noqa: BLE001
                    q.put(("err", e))

            threading.Thread(target=_worker, daemon=True).start()
            try:
                _kind, val = q.get(timeout=timeout)
            except _queue.Empty:
                raise TimeoutError
            if _kind == "err":
                raise val
            return val

        for pos, qid, name in missing:
            row = ds_by_id[qid]
            sample = SingleTurnSample(
                user_input=row.get("user_input") or "",
                response=row.get("response") or "",
                retrieved_contexts=row.get("retrieved_contexts") or [],
                reference=row.get("reference"),
            )
            got = None
            metric = by_name[name]
            metric.llm = groq_llm
            if getattr(metric, "embeddings", None) is not None:
                metric.embeddings = emb
            for attempt in range(1, 3):
                try:
                    raw = _run_with_timeout(
                        lambda: metric.single_turn_score(sample), 300)
                    got = None if raw != raw else round(float(raw), 4)
                    if got is not None:
                        break
                except TimeoutError:
                    print(f"  {qid}/{name}: attempt {attempt} timed out")
                except Exception as e:
                    print(f"  {qid}/{name}: attempt {attempt} failed "
                          f"({type(e).__name__})")
            scores_by_row[qid][name] = got
            print(f"  {qid}/{name}: "
                  f"{'recovered as ' + str(got) if got is not None else 'still missing after retries'}")
    return scores_by_row, elapsed


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for the RTI RAG system")
    parser.add_argument("--limit", type=int, default=0, help="only first N questions")
    parser.add_argument("--ids", type=str, default="", help="comma-separated question ids")
    parser.add_argument("--out", type=str, default="eval_results.json",
                        help="output json path")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore/overwrite an existing checkpoint")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b",
                        help="pipeline LLM; accept 20b/120b/full shorthands or a "
                             "full model id like openai/gpt-oss-20b")
    parser.add_argument("--contexts", type=int, default=0,
                        help="cap on retrieved chunks sent into each sufficiency "
                             "grading call (0 = no cap, i.e. full context)")
    parser.add_argument("--attempts", type=int, default=None,
                        help="max retrieval attempts for this eval run "
                             "(None = ask.py default, 3)")
    parser.add_argument("--no-ragas", action="store_true",
                        help="skip the RAGAS scoring phase (pipeline only)")
    args = parser.parse_args()

    selected = list(EVAL_SET)
    if args.ids.strip():
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        selected = [q for q in selected if q["id"] in wanted]
    if args.limit > 0:
        selected = selected[:args.limit]
    if not selected:
        print("No questions selected.")
        sys.exit(1)

    # Point the pipeline's LLM calls (grade/rewrite/answer/verify) at the
    # requested model. ask functions read the module-global GROQ_MODEL at call
    # time, so this swap works without touching ask.py.
    _MODEL_ALIASES = {
        "20b": "openai/gpt-oss-20b",
        "120b": "openai/gpt-oss-120b",
        "full": "openai/gpt-oss-120b",
    }
    model = _MODEL_ALIASES.get(args.model.strip().lower(), args.model.strip())
    ask.GROQ_MODEL = model
    contexts = args.contexts or None  # 0/None => no cap
    print(f"Pipeline model: {model}  (grading context cap: "
          f"{contexts or 'none'}, attempts: {args.attempts or MAX_RETRIEVAL_ATTEMPTS})")

    checkpoint_path = args.out + ".checkpoint.json"
    resumed = []
    if not args.fresh:
        resumed = load_checkpoint(checkpoint_path)
        if resumed:
            print(f"Resuming: {len(resumed)} question(s) already in checkpoint "
                  f"({checkpoint_path})")
    done_ids = {item["id"] for (item, _out, _row) in resumed}

    rows = list(resumed)
    rows.extend(collect_rows(selected, checkpoint_path=checkpoint_path,
                             done_ids=done_ids, contexts=contexts,
                             attempts=args.attempts, initial=resumed))

    pending_ids = {item["id"] for (item, _out, _row) in rows}
    if len(pending_ids) < len(selected):
        print(f"\nWARNING: only {len(pending_ids)}/{len(selected)} questions available "
              f"({sorted(set(item['id'] for item in selected) - pending_ids)} skipped). "
              f"Re-run with --ids to finish them after the Groq quota resets.")

    # ---- run RAGAS ----
    eval_elapsed = None
    scores_by_row = {item["id"]: {n: None for n in METRIC_NAMES}
                     for (item, _out, _row) in rows}
    if not args.no_ragas:
        api_key = os.environ.get(GROQ_API_KEY_ENV)
        scores_by_row, eval_elapsed = run_ragas_scoring(rows, api_key)


    final_report(rows, scores_by_row, eval_elapsed, args.out)


def final_report(rows, scores_by_row, eval_elapsed, out_path):
    """Assemble the results dict, write it out, and print the
    summary table plus the answered/refused breakdown."""
    # ---- assemble results ----
    in_corpus_rows = []
    questions_out = []
    for item, out, _row in rows:
        s = scores_by_row.get(item["id"], {n: None for n in METRIC_NAMES})
        qout = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "expected_graph_expansion": item["expected_graph_expansion"],
            "expected_graceful_stop": item["expected_graceful_stop"],
            "final_grade": out["final_grade"],
            "attempts": out["attempts"],
            "graceful_stop": out["graceful_stop"],
            "graph_expansion_triggered": out["graph_expansion_triggered"],
            "verification_passed": out["verification_passed"],
            "num_retrieved": out["num_retrieved"],
            "elapsed_s": out["elapsed_s"],
            "retrieved_sources": out["sources"],
            "response": out["response"],
            "scores": s,
        }
        questions_out.append(qout)
        if item["category"] != "out_of_corpus":
            in_corpus_rows.append(s)

    def overall(group):
        return {n: average([g[n] for g in group]) for n in METRIC_NAMES}

    all_scores = [q["scores"] for q in questions_out]
    overall_all = overall(all_scores)
    overall_in_corpus = overall(in_corpus_rows)

    behavior = []
    for q in questions_out:
        if q["category"] == "out_of_corpus":
            behavior.append({
                "id": q["id"],
                "expected_graceful_stop": q["expected_graceful_stop"],
                "graceful_stop": q["graceful_stop"],
                "pass": q["graceful_stop"] == q["expected_graceful_stop"],
                "response": q["response"],
            })

    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": ask.GROQ_MODEL,
        "embeddings": EMBED_MODEL,
        "metrics": METRIC_NAMES,
        "n_questions": len(questions_out),
        "n_in_corpus": len(in_corpus_rows),
        "ragas_eval_elapsed_s": eval_elapsed,
        "questions": questions_out,
        "overall_all": overall_all,
        "overall_in_corpus": overall_in_corpus,
        "out_of_corpus_behavior": behavior,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- print summary table ----
    print("\n" + "=" * 100)
    print("RAGAS SUMMARY")
    print("=" * 100)
    header = (f"{'id':<4}{'category':<15}{'faith':>7}{'anrel':>7}"
              f"{'cprec':>7}{'crecal':>7}   {'stop':<9}{'graph':<6}")
    print(header)
    print("-" * 100)
    for q in questions_out:
        s = q["scores"]
        stop = "yes" if q["graceful_stop"] else "no"
        graph = "yes" if q["graph_expansion_triggered"] else "no"
        print(f"{q['id']:<4}{q['category']:<15}"
              f"{_fmt(s['faithfulness']):>7}{_fmt(s['answer_relevancy']):>7}"
              f"{_fmt(s['context_precision']):>7}{_fmt(s['context_recall']):>7}"
              f"   {stop:<9}{graph:<6}")
    print("-" * 100)
    print(f"{'OVERALL':<4}{'all-15':<15}"
          f"{_fmt(overall_all['faithfulness']):>7}{_fmt(overall_all['answer_relevancy']):>7}"
          f"{_fmt(overall_all['context_precision']):>7}{_fmt(overall_all['context_recall']):>7}"
          f"   {'':<9}{'':<6}")
    print(f"{'OVERALL':<4}{'in-corpus':<15}"
          f"{_fmt(overall_in_corpus['faithfulness']):>7}{_fmt(overall_in_corpus['answer_relevancy']):>7}"
          f"{_fmt(overall_in_corpus['context_precision']):>7}{_fmt(overall_in_corpus['context_recall']):>7}"
          f"   {'':<9}{'':<6}")

    # out-of-corpus behavioral checks
    print("\nOut-of-corpus behavior (should graceful-stop, not hallucinate):")
    for b in behavior:
        status = "PASS" if b["pass"] else "FAIL"
        print(f"  [{status}] {b['id']}: graceful_stop={b['graceful_stop']} "
              f"(expected {b['expected_graceful_stop']})")

    # flag low-scoring in-corpus questions
    print(f"\nFlagged in-corpus questions (any metric < {LOW_SCORE_THRESHOLD}):")
    any_flag = False
    for q in questions_out:
        if q["category"] == "out_of_corpus":
            continue
        lows = [f"{n}={q['scores'][n]}" for n in METRIC_NAMES
                if q["scores"][n] is not None and q["scores"][n] < LOW_SCORE_THRESHOLD]
        if lows:
            any_flag = True
            print(f"  - {q['id']} ({q['category']}): {', '.join(lows)}")
    if not any_flag:
        print("  (none)")

    # graph-expansion expectation check
    print("\nGraph-expansion expectation check:")
    any_graph_issue = False
    for q in questions_out:
        exp = q["expected_graph_expansion"]
        got = q["graph_expansion_triggered"]
        if exp != got:
            any_graph_issue = True
            print(f"  - {q['id']}: expected graph expansion={exp}, got {got}")
    if not any_graph_issue:
        print("  (all matched)")

    # ---- answered vs refused breakdown (fairer "answer quality" split) ----
    def _avg_metric(grp, name):
        vals = [s for q in grp if (s := q["scores"][name]) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    answered = [q for q in questions_out if not q["graceful_stop"]]
    refused = [q for q in questions_out if q["graceful_stop"]]
    answered_avg = {n: _avg_metric(answered, n) for n in METRIC_NAMES}

    correct_refusals = [q for q in refused if q["expected_graceful_stop"]]
    overcautious = [q for q in refused if not q["expected_graceful_stop"]]
    refusal_accuracy = (round(len(correct_refusals) / len(refused), 4)
                        if refused else None)

    breakdown = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": ask.GROQ_MODEL,
        "n_questions": len(questions_out),
        "overall_raw_all": overall_all,
        "answered": {
            "n": len(answered),
            "ids": [q["id"] for q in answered],
            "averages": answered_avg,
        },
        "refused": {
            "n": len(refused),
            "ids": [q["id"] for q in refused],
            "refusal_accuracy": refusal_accuracy,
            "correct_refusals": [q["id"] for q in correct_refusals],
            "overcautious_refusals": [q["id"] for q in overcautious],
        },
    }
    with open("eval_summary_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(breakdown, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY (answered vs correctly-refused split)")
    print("=" * 100)
    print(f"Overall raw RAGAS averages (all {len(questions_out)}, incl. refusals):")
    print(f"    faithfulness={_fmt(overall_all['faithfulness'])} "
          f"answer_relevancy={_fmt(overall_all['answer_relevancy'])} "
          f"context_precision={_fmt(overall_all['context_precision'])} "
          f"context_recall={_fmt(overall_all['context_recall'])}")
    print(f"Answered-only RAGAS averages ({len(answered)} committed answers, "
          f"ids: {', '.join(q['id'] for q in answered)}):")
    print(f"    faithfulness={_fmt(answered_avg['faithfulness'])} "
          f"answer_relevancy={_fmt(answered_avg['answer_relevancy'])} "
          f"context_precision={_fmt(answered_avg['context_precision'])} "
          f"context_recall={_fmt(answered_avg['context_recall'])}")
    if refused:
        print(f"Refusal accuracy ({len(refused)} refusals, "
              f"ids: {', '.join(q['id'] for q in refused)}):")
        print(f"    {_fmt(refusal_accuracy)} "
              f"({len(correct_refusals)}/{len(refused)} refusals appropriate)")
    else:
        print("Refusal accuracy: n/a (no refusals this run)")
    if overcautious:
        print("  FLAG — refused but should have answered (over-cautious): "
              + ", ".join(f"{q['id']} [{q['category']}]" for q in overcautious))
    if answered:
        answered_low = [
            (q["id"], f"{n}={q['scores'][n]}")
            for q in answered
            for n in METRIC_NAMES
            if q["scores"][n] is not None and q["scores"][n] < LOW_SCORE_THRESHOLD
        ]
        if answered_low:
            by_id = {}
            for qid, tag in answered_low:
                by_id.setdefault(qid, []).append(tag)
            print("  FLAG — committed answers with low metric(s): "
                  + "; ".join(f"{qid} ({', '.join(t)})" for qid, t in by_id.items()))
    print(f"\nBreakdown saved to eval_summary_breakdown.json")

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
