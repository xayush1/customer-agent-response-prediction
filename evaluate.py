"""End-to-end evaluation of the Agent Response Predictor.

Setup
-----
The dataset is split 80/20 at the turn-pair level (see ``split.py``). Only the
80% TRAIN pairs live in Milvus; the 20% TEST pairs are held out and used here as
evaluation objectives (1 conversation pair = 1 objective).

For every held-out test pair (history -> customer message -> GOLD agent reply)
we measure two layers independently:

1. RETRIEVAL QUALITY (how good is the chunk retriever?)
   For the test customer message we call ``retriever.search`` against the
   train-only collection. A retrieved train pair is RELEVANT if its agent reply
   is >= REL_THRESHOLD cosine-similar to the gold agent reply (i.e. it points the
   model at the right kind of answer). From this we derive, per query and in
   aggregate: TP / FP / FN, Precision@k, Recall@k, F1@k, Hit@k, MRR, nDCG@k.

2. GENERATION QUALITY (how good is the final predicted reply?)
   We run the full ReAct agent and compare its predicted reply to the gold reply
   with (a) embedding cosine similarity and (b) an LLM-as-judge score (1-5).
   We also report a confusion matrix that treats the model's own CONFIDENCE as a
   binary predictor of correctness (judge >= JUDGE_PASS) -> TP/TN/FP/FN, which
   tells you whether "high confidence" actually means "correct".

Usage
-----
    PY=/Users/ayusraj/Desktop/gptmerger/TAF-GPT/.venv/bin/python3
    $PY evaluate.py                 # full held-out test set
    $PY evaluate.py --limit 15      # quick run on first 15 objectives
    $PY evaluate.py --top_k 10      # change retrieval cutoff k
"""
import argparse
import json
import math
import statistics

import numpy as np

import react_agent
import retriever
from config import get_config
from embeddings import embed_documents, embed_query
from split import pair_id, split_pairs

cfg = get_config()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def cosine(a, b) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(va @ vb / denom) if denom else 0.0


def llm_judge(customer: str, context: str, predicted: str, actual: str) -> dict:
    """Score how good the predicted reply is vs the gold reply (1-5)."""
    client = react_agent._client()
    prompt = f"""You are evaluating a customer-support reply predictor.

CONVERSATION CONTEXT:
{context or '(none)'}

CUSTOMER MESSAGE:
{customer}

ACTUAL AGENT REPLY (ground truth):
{actual}

PREDICTED AGENT REPLY:
{predicted}

Rate how good the PREDICTED reply is as a substitute for the ACTUAL reply on a
1-5 scale, considering correctness of intent, appropriateness, and tone.
5 = essentially equivalent / would be fine to send
3 = reasonable but misses some specifics
1 = wrong intent or unhelpful
Return STRICT JSON: {{"score": <int>, "reason": "<short>"}}"""
    r = client.chat.completions.create(
        model=cfg.AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content)


def dcg(rels: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(rel_flags: list[int]) -> float:
    ideal = sorted(rel_flags, reverse=True)
    idcg = dcg(ideal)
    return dcg(rel_flags) / idcg if idcg else 0.0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Evaluate retrieval + generation quality.")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only first N test pairs (0 = all).")
    ap.add_argument("--top_k", type=int, default=10, help="Retrieval cutoff k.")
    ap.add_argument("--no_judge", action="store_true", help="Skip LLM-as-judge (faster).")
    args = ap.parse_args()
    k = args.top_k

    train, test = split_pairs()
    if args.limit:
        test = test[: args.limit]

    print(f"{'='*72}")
    print(f"EVALUATION  |  train(in Milvus)={len(train)}  test(held-out objectives)={len(test)}")
    print(f"k={k}  rel_threshold={cfg.REL_THRESHOLD}  judge_pass>={cfg.JUDGE_PASS}")
    print(f"{'='*72}")

    # Index train -> position, and embed all train agent replies once so we can
    # compute the full RELEVANT set per query cheaply (for recall/nDCG).
    train_pos = {p["pair_id"]: i for i, p in enumerate(train)}
    train_thread_idx: dict[str, list[int]] = {}
    for i, p in enumerate(train):
        train_thread_idx.setdefault(p["thread_id"], []).append(i)
    print("Embedding train agent replies for relevance ground-truth ...")
    train_reply_vecs = np.asarray(embed_documents([p["agent_reply"] for p in train]), dtype=float)
    train_reply_norms = np.linalg.norm(train_reply_vecs, axis=1) + 1e-9

    # Accumulators
    precisions, recalls, f1s, hits, rrs, ndcgs = [], [], [], [], [], []
    tp_tot = fp_tot = fn_tot = 0
    thread_overlap = 0  # queries where a same-thread train pair was retrieved
    cos_scores, judge_scores = [], []
    # Confidence-vs-correctness confusion matrix (generation)
    g_tp = g_tn = g_fp = g_fn = 0

    for n, t in enumerate(test, 1):
        query = t["customer_msg"]
        gold = t["agent_reply"]
        gold_vec = np.asarray(embed_query(gold), dtype=float)

        # ----- RELEVANCE GROUND TRUTH (full corpus) ----------------------- #
        # Relevant = differently-worded-but-equivalent reply OR same thread.
        sims = train_reply_vecs @ gold_vec / (train_reply_norms * (np.linalg.norm(gold_vec) + 1e-9))
        relevant_idx = set(np.where(sims >= cfg.REL_THRESHOLD)[0].tolist())
        relevant_idx.update(train_thread_idx.get(t["thread_id"], []))
        n_relevant = len(relevant_idx)

        # ----- RETRIEVAL -------------------------------------------------- #
        hits_list = retriever.search(query, top_k=k)
        rel_flags = []
        same_thread = False
        for h in hits_list:
            pid = f"{h.get('thread_id')}::{h.get('turn_position')}"
            idx = train_pos.get(pid)
            rel_flags.append(1 if (idx is not None and idx in relevant_idx) else 0)
            if h.get("thread_id") == t["thread_id"]:
                same_thread = True

        tp = sum(rel_flags)
        fp = len(rel_flags) - tp
        fn = max(n_relevant - tp, 0)
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn

        precision = tp / len(rel_flags) if rel_flags else 0.0
        recall = tp / n_relevant if n_relevant else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if recall not in (None, 0.0) and precision
            else (0.0 if recall is not None else None)
        )
        hit = 1 if tp > 0 else 0
        rr = next((1.0 / (i + 1) for i, r in enumerate(rel_flags) if r), 0.0)

        precisions.append(precision)
        if recall is not None:
            recalls.append(recall)
            if f1 is not None:
                f1s.append(f1)
        hits.append(hit)
        rrs.append(rr)
        ndcgs.append(ndcg_at_k(rel_flags))
        if same_thread:
            thread_overlap += 1

        # ----- GENERATION ------------------------------------------------- #
        history = t.get("history") or []  # held-out pairs have no stored history
        result = react_agent.predict(history, query)
        predicted = result.get("predicted_response", "")
        confidence = result.get("confidence", "medium")
        cos = cosine(embed_query(predicted), gold_vec)
        cos_scores.append(cos)

        judge_score = None
        if not args.no_judge:
            ctx = "\n".join(f"{m['role'].upper()}: {m['text'][:300]}" for m in history)
            try:
                judge_score = int(llm_judge(query, ctx, predicted, gold).get("score", 0))
            except Exception as e:  # noqa: BLE001
                print(f"  [judge error: {e}]")
            if judge_score is not None:
                judge_scores.append(judge_score)
                correct = judge_score >= cfg.JUDGE_PASS
                confident = confidence in ("high", "medium")
                if confident and correct:
                    g_tp += 1
                elif confident and not correct:
                    g_fp += 1
                elif not confident and correct:
                    g_fn += 1
                else:
                    g_tn += 1

        js = f"{judge_score}/5" if judge_score is not None else "-"
        print(
            f"[{n:>3}/{len(test)}] P@{k}={precision:.2f} R={('%.2f'%recall) if recall is not None else '  -'} "
            f"hit={hit} rr={rr:.2f} | cos={cos:.3f} judge={js} conf={confidence} "
            f"| {t['subject'][:40]}"
        )

    # ----------------------------------------------------------------- #
    # Aggregate report
    # ----------------------------------------------------------------- #
    def avg(xs):
        return statistics.mean(xs) if xs else float("nan")

    micro_p = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) else 0.0
    micro_r = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    )

    print(f"\n{'='*72}\nRETRIEVAL METRICS (chunk retriever quality)\n{'='*72}")
    print(f"  Objectives evaluated : {len(test)}")
    print(f"  Confusion (micro)    : TP={tp_tot}  FP={fp_tot}  FN={fn_tot}")
    print(f"  Precision@{k} (micro) : {micro_p:.3f}")
    print(f"  Recall@{k}    (micro) : {micro_r:.3f}")
    print(f"  F1@{k}        (micro) : {micro_f1:.3f}")
    print(f"  Precision@{k} (macro) : {avg(precisions):.3f}")
    print(f"  Recall@{k}    (macro) : {avg(recalls):.3f}")
    print(f"  Hit@{k}              : {avg(hits):.3f}")
    print(f"  MRR                  : {avg(rrs):.3f}")
    print(f"  nDCG@{k}             : {avg(ndcgs):.3f}")
    print(f"  Same-thread overlap  : {thread_overlap}/{len(test)} ({thread_overlap/len(test):.0%})")

    print(f"\n{'='*72}\nGENERATION METRICS (final predicted reply quality)\n{'='*72}")
    print(f"  Avg cosine (pred vs gold) : {avg(cos_scores):.3f}")
    if judge_scores:
        passes = sum(1 for s in judge_scores if s >= cfg.JUDGE_PASS)
        print(f"  Avg LLM-judge             : {avg(judge_scores):.2f}/5")
        print(f"  Pass rate (judge>={cfg.JUDGE_PASS})      : {passes}/{len(judge_scores)} "
              f"({passes/len(judge_scores):.0%})")
        total = g_tp + g_tn + g_fp + g_fn
        acc = (g_tp + g_tn) / total if total else 0.0
        cp = g_tp / (g_tp + g_fp) if (g_tp + g_fp) else 0.0
        cr = g_tp / (g_tp + g_fn) if (g_tp + g_fn) else 0.0
        print(f"\n  Confidence-calibration confusion (confidence vs judge-correct):")
        print(f"    TP={g_tp}  FP={g_fp}  FN={g_fn}  TN={g_tn}")
        print(f"    Accuracy  = {acc:.3f}  (confident&correct + unsure&wrong)")
        print(f"    Precision = {cp:.3f}  (of confident replies, fraction correct)")
        print(f"    Recall    = {cr:.3f}  (of correct replies, fraction confident)")
    else:
        print("  LLM-judge skipped (--no_judge).")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
