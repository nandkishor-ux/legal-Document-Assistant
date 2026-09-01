"""Resume-capable, cell-by-cell RAGAS scorer for the v2 eval.

Problem: run_eval.run_ragas_scoring() uses ragas.evaluate(), which re-runs the
entire 60-cell workload each invocation and discards already-scored cells if the
Groq daily TPD quota exhausts mid-pass (on-demand tier is shared across models).
On this free tier that means a run that dies at ~198k/200k tokens loses every
cell it already scored.

This script instead scores each (question, metric) cell independently via
single_turn_score() with a hard per-cell timeout, and -- critically -- persists
each successful cell immediately to a state JSON, so an interrupt only loses the
in-flight cell. Re-running resumes from whatever cells are already complete.

It also paces around Groq's on-demand TPD refill: on a 429 it parses the
"try again in Xs" window and sleeps instead of hammering the endpoint, letting
the slow token trickle top us back up incrementally.

State file schema (scores_v2_state.json):
    scores: { qid: { metric: value-or-null } }
    order:  [qid,...]  (iteration order)

Usage:
    python score_v2.py                 # score all missing cells, resume-aware
    python score_v2.py --cells F1      # only specific ids (comma list)
"""

import argparse
import json
import os
import queue as _queue
import re
import sys
import threading
import time
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

os.environ.setdefault("RAGAS_TELEMETRY_DISABLED", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv  # noqa: E402

from run_eval import (  # noqa: E402
    EMBED_MODEL,
    GROQ_API_KEY_ENV,
    METRIC_NAMES,
    RAGAS_CHUNK_TRIM,
    build_ragas_llm,
)

CHECKPOINT_JSON = "eval_results_v2.json.checkpoint.json"
STATE_JSON = "scores_v2_state.json"
PER_CELL_TIMEOUT = 300
MAX_ATTEMPTS_PER_CELL = 6

# Re-raise a hard probe 429 as a sentinel so the outer loop knows the pool is
# exhausted; _chat_completion paths else wrap it in retries we don't want here.
def _parse_wait(msg):
    """Return seconds to wait from a Groq 429 'try again in X' message."""
    m = re.search(r"try again in ([0-9.]+)([sm])", msg or "")
    if not m:
        return 45.0
    val = float(m.group(1))
    unit = m.group(2)
    return val * 60.0 if unit == "m" else val


def _run_with_timeout(fn, timeout):
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


def _load_state():
    if os.path.exists(STATE_JSON):
        with open(STATE_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {"scores": {}, "order": []}


def _save_state(state):
    state["last_saved"] = datetime.now().isoformat(timespec="seconds")
    tmp = STATE_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_JSON)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=str, default="",
                        help="comma ids to (re)score; empty = all missing")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get(GROQ_API_KEY_ENV)
    if not api_key:
        print(f"{GROQ_API_KEY_ENV} not found. Check .env")
        sys.exit(1)

    if not os.path.exists(CHECKPOINT_JSON):
        print(f"{CHECKPOINT_JSON} missing.")
        sys.exit(1)

    ckpt = json.load(open(CHECKPOINT_JSON, encoding="utf-8"))
    triples = [(d["item"], d["out"], d["row"]) for d in ckpt]
    items = {it["id"]: (it, out, row) for (it, out, row) in triples}
    order = [it["id"] for it, _o, _r in triples]
    print(f"loaded {len(order)} pipeline rows from {CHECKPOINT_JSON}")

    forced = {s.strip() for s in args.cells.split(",") if s.strip()} if args.cells.strip() else set()
    if forced - set(order):
        print(f"WARNING: asked for unknown ids: {sorted(forced - set(order))}")

    state = _load_state()
    scores = state.setdefault("scores", {})
    state.setdefault("order", order)
    _save_state(state)

    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._context_recall import context_recall
    from ragas.metrics._faithfulness import faithfulness
    from ragas.metrics.base import SingleTurnSample

    groq_llm = build_ragas_llm(api_key)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    answer_relevancy.embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    answer_relevancy.strictness = 1
    by_name = {m.name: m for m in metrics}

    def _pending_cells():
        out = []
        for qid in order:
            if forced and qid not in forced:
                continue
            for name in METRIC_NAMES:
                cur = scores.get(qid, {}).get(name)
                if cur is None:
                    out.append((qid, name))
        return out

    pending = _pending_cells()
    print(f"pending cells to score: {len(pending)}")
    if not pending:
        print("nothing to do.")
        _emit_report(order, scores)
        return

    t0 = time.time()
    filled = 0
    skipped_quota = False
    for (qid, name) in pending:
        it, out, row = items[qid]
        workers = [c[:RAGAS_CHUNK_TRIM] for c in (row["retrieved_contexts"] or [])]
        sample = SingleTurnSample(
            user_input=row["user_input"] or "",
            response=row["response"] or "",
            retrieved_contexts=workers,
            reference=row["reference"],
        )
        metric = by_name[name]
        metric.llm = groq_llm
        if getattr(metric, "embeddings", None) is not None:
            metric.embeddings = answer_relevancy.embeddings

        done = False
        for attempt in range(1, MAX_ATTEMPTS_PER_CELL + 1):
            try:
                raw = _run_with_timeout(lambda: metric.single_turn_score(sample),
                                        PER_CELL_TIMEOUT)
                val = None if raw != raw else round(float(raw), 4)
                if val is not None:
                    scores.setdefault(qid, {})[name] = val
                    _save_state(state)
                    filled += 1
                    print(f"  {qid}/{name}: {val}  (t={round(time.time()-t0,1)}s)")
                    done = True
                    break
                print(f"  {qid}/{name}: attempt {attempt} returned NaN")
            except TimeoutError:
                print(f"  {qid}/{name}: attempt {attempt} timed out ({PER_CELL_TIMEOUT}s)")
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate_limit" in msg.lower() or "Rate limit" in msg:
                    wait = _parse_wait(msg)
                    print(f"  {qid}/{name}: 429 on attempt {attempt}; "
                          f"waiting {wait:.0f}s for TPD refill ...")
                    skipped_quota = True
                    try:
                        time.sleep(min(wait, 1100))
                    except KeyboardInterrupt:
                        pass
                else:
                    print(f"  {qid}/{name}: attempt {attempt} failed "
                          f"({type(e).__name__}: {msg[:80]})")
            if done:
                break
        if not done:
            print(f"  {qid}/{name}: still missing after {MAX_ATTEMPTS_PER_CELL} attempts; left for later")
        # brief pause so each accepted call is not immediately followed by a
        # fresh request that could punch through the on-demand ceiling
        time.sleep(3)

    print(f"\nfill pass done: {filled} cell(s) scored, "
          f"{len(_pending_cells())} still pending in {round(time.time()-t0,1)}s")
    _emit_report(order, scores)


def _emit_report(order, scores):
    n_ok = {n: sum(1 for q in order if scores.get(q, {}).get(n) is not None)
            for n in METRIC_NAMES}
    print("\ncoverage by metric over all 15:")
    for n in METRIC_NAMES:
        print(f"  {n}: {n_ok[n]}/15")
    for q in order:
        print(f"  {q}: " + ", ".join(
            f"{n}={scores.get(q, {}).get(n)}" for n in METRIC_NAMES))
    print(f"\nstate saved to {STATE_JSON}")


if __name__ == "__main__":
    main()
