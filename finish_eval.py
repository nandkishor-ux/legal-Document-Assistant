"""Finish the production eval run after the pipeline checkpoint was clobbered.

collect_rows() used to rewrite the checkpoint file from a freshly created list,
so a resume that only ran O1 left a checkpoint containing only that row. The
other 13 completed rows survive in the last results file (eval_results_final.json),
so we rebuild them here, merge the fresh O1 row, run any still-missing pipeline
questions (O2), RAGAS-score only the rows lacking stored scores (O1 + O2), merge
those with the 13 stored scores, then emit the full 15-question report plus the
answered-vs-refused breakdown.

One-off recovery util; plain `python run_eval.py --out eval_results_final.json`
is the normal entry point for a full re-run.
"""

import json
import os
import queue as _queue
import sys
import threading
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("RAGAS_TELEMETRY_DISABLED", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv  # noqa: E402
from groq import Groq  # noqa: E402

import ask  # noqa: E402
from eval_set import EVAL_SET  # noqa: E402
from index import CHUNKS_JSON, label_for  # noqa: E402
from run_eval import (  # noqa: E402
    EMBED_MODEL,
    GROQ_API_KEY_ENV,
    METRIC_NAMES,
    RAGAS_CHUNK_TRIM,
    build_ragas_llm,
    collect_rows,
    final_report,
    load_checkpoint,
    _write_checkpoint,
)

RESULTS_JSON = "eval_results_final.json"
CHECKPOINT_JSON = RESULTS_JSON + ".checkpoint.json"
PIPELINE_MODEL = "openai/gpt-oss-120b"


def _label_to_chunks():
    chunks = json.load(open(CHUNKS_JSON, encoding="utf-8"))
    by_label = {}
    for c in chunks:
        by_label.setdefault(label_for(c), []).append(c)
    return by_label


def rebuild_from_results(results_path):
    """Reconstruct (item, out, row) triples stored in a results file.

    The saved rows carry everything the report needs plus their retrieved source
    labels; the full chunk texts are recovered by reversing label_for().
    """
    by_label = _label_to_chunks()
    data = json.load(open(results_path, encoding="utf-8"))
    items = {it["id"]: it for it in EVAL_SET}
    triples = []
    for q in data["questions"]:
        it = items[q["id"]]
        ctxs = []
        seen = set()
        for src in q["retrieved_sources"]:
            for c in by_label.get(src, []):
                if c["text"] not in seen:
                    seen.add(c["text"])
                    ctxs.append(c["text"])
                    break
        out = {
            "attempts": q["attempts"],
            "attempts_log": [],
            "final_grade": q["final_grade"],
            "graceful_stop": q["graceful_stop"],
            "graph_expansion_triggered": q["graph_expansion_triggered"],
            "graph_lines": [],
            "verification_passed": q["verification_passed"],
            "sources": q["retrieved_sources"],
            "num_retrieved": q["num_retrieved"],
            "elapsed_s": q["elapsed_s"],
            "response": q["response"],
            "retrieved_contexts": ctxs,
        }
        row = {
            "eval_id": it["id"],
            "category": it["category"],
            "user_input": it["question"],
            "reference": it["ground_truth"],
            "response": out["response"],
            "retrieved_contexts": ctxs,
        }
        triples.append((it, out, row))
    return triples


def _run_with_timeout(fn, timeout):
    # single_turn_score() can hang (an underlying async call that never
    # returns), so run it on a daemon thread and bound the wait. The daemon
    # survives a hang without blocking interpreter shutdown.
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


def score_rows_per_cell(rows, api_key):
    """RAGAS-score rows one cell at a time with a hard per-cell timeout.

    Built for small remaining sets (e.g. the two out-of-corpus rows after a
    quota interruption). Avoids ragas.evaluate(), which can grind through very
    long per-call timeouts/retries when the evaluator model is throttled.
    Mirrors run_ragas_scoring()'s metric wiring (legacy single-turn metrics).
    Returns (scores_by_row, elapsed).
    """
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._context_recall import context_recall
    from ragas.metrics._faithfulness import faithfulness
    from ragas.metrics.base import SingleTurnSample

    groq_llm = build_ragas_llm(api_key)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # answer_relevancy's similarity step needs the LangChain embedding protocol.
    answer_relevancy.embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    answer_relevancy.strictness = 1

    by_name = {m.name: m for m in metrics}
    t0 = time.time()
    scores_by_row = {}
    for (_item, _out, row) in rows:
        workers = [c[:RAGAS_CHUNK_TRIM] for c in (row["retrieved_contexts"] or [])]
        sample = SingleTurnSample(
            user_input=row["user_input"] or "",
            response=row["response"] or "",
            retrieved_contexts=workers,
            reference=row["reference"],
        )
        print(f"  {_item['id']} [{_item['category']}]")
        sc = {name: None for name in METRIC_NAMES}
        for name in METRIC_NAMES:
            m = by_name[name]
            m.llm = groq_llm
            got = None
            for attempt in range(1, 3):
                try:
                    raw = _run_with_timeout(
                        lambda: m.single_turn_score(sample), 300)
                    got = None if raw != raw else round(float(raw), 4)
                    if got is not None:
                        break
                except TimeoutError:
                    print(f"      {name}: attempt {attempt} timed out")
                except Exception as e:
                    print(f"      {name}: attempt {attempt} failed "
                          f"({type(e).__name__}: {str(e)[:80]})")
            sc[name] = got
            print(f"      {name}: {got}")
        scores_by_row[_item["id"]] = sc
    elapsed = round(time.time() - t0, 1)
    return scores_by_row, elapsed


def main():
    load_dotenv()
    api_key = os.environ.get(GROQ_API_KEY_ENV)
    if not api_key:
        print(f"{GROQ_API_KEY_ENV} not found.")
        sys.exit(1)
    if not os.path.exists(RESULTS_JSON):
        print(f"{RESULTS_JSON} missing; nothing to rebuild from.")
        sys.exit(1)

    ask.GROQ_MODEL = PIPELINE_MODEL

    # 1) rebuild the completed rows from the results file
    rebuilt = rebuild_from_results(RESULTS_JSON)
    print(f"rebuilt {len(rebuilt)} completed row(s) from {RESULTS_JSON}: "
          f"{sorted(it['id'] for it, _o, _r in rebuilt)}")

    # 2) checkpoint rows are fresher (just-piped pipeline runs) -> they win
    merged = {it["id"]: (it, out, row)
              for (it, out, row) in rebuilt}
    ckpt = load_checkpoint(CHECKPOINT_JSON)
    for it, out, row in ckpt:
        merged[it["id"]] = (it, out, row)
    ordered = [merged[it["id"]] for it in EVAL_SET if it["id"] in merged]
    print(f"merged rows available: {sorted(merged)}")

    # 3) run the still-missing pipeline questions (only O2)
    present = set(merged)
    missing = [it for it in EVAL_SET if it["id"] not in present]
    if missing:
        print(f"\nRunning pipeline for {len(missing)} question(s): "
              f"{[it['id'] for it in missing]}")
        new_rows = collect_rows(missing, checkpoint_path=CHECKPOINT_JSON,
                                done_ids=present, contexts=None,
                                attempts=None, initial=ordered)
        for it, out, row in new_rows:
            merged[it["id"]] = (it, out, row)
        ordered = [merged[it["id"]] for it in EVAL_SET if it["id"] in merged]

    final_present = sorted(it["id"] for it, _o, _r in ordered)
    print(f"\nrows now available: {final_present}")
    if len(final_present) < len(EVAL_SET):
        gone = sorted(it["id"] for it in EVAL_SET
                      if it["id"] not in final_present)
        print(f"WARNING: {gone} still skipped (Groq quota?). Re-run "
              f"finish_eval.py after the quota resets to pick them up.")
    else:
        _write_checkpoint(CHECKPOINT_JSON, ordered)
        print(f"checkpoint updated: {CHECKPOINT_JSON}")

    # 4) stored RAGAS scores for previously-scored rows
    stored = {q["id"]: q["scores"]
              for q in json.load(open(RESULTS_JSON, encoding="utf-8"))["questions"]}

    # 5) score only the rows lacking stored scores (O1, O2) via bounded
    # single-turn calls -- evaluate() can grind forever when throttled
    to_score = [t for t in ordered if t[0]["id"] not in stored]
    print(f"\nScoring {len(to_score)} new row(s): "
          f"{[t[0]['id'] for t in to_score]}")
    scores_by_row = {qid: dict(s) for qid, s in stored.items()}
    eval_elapsed = None
    if to_score:
        _scores, eval_elapsed = score_rows_per_cell(to_score, api_key)
        for qid, s in _scores.items():
            scores_by_row[qid] = s
    for it, _o, _r in ordered:
        scores_by_row.setdefault(it["id"], {n: None for n in METRIC_NAMES})

    # 6) assemble + write + print the full report and breakdown
    final_report(ordered, scores_by_row, eval_elapsed, RESULTS_JSON)


if __name__ == "__main__":
    main()