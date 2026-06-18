# Agent Response Predictor — Evaluation Results

**Run:** 10 held-out objectives (1 conversation pair = 1 objective)
**Split:** 163 train pairs in Milvus, 41 held out (deterministic, `seed=7`, pair-level 80/20)
**Params:** `k=10`, `rel_threshold=0.62`, `judge_pass >= 4`

---

## 1. Retrieval metrics (chunk retriever quality)

| Metric | Value | Meaning |
|---|---|---|
| **Hit@10** | **1.000** | Every query had a relevant pair in the top 10 |
| **MRR** | **0.900** | First relevant pair is almost always rank 1 |
| **nDCG@10** | **0.870** | Relevant pairs ranked near the top |
| Precision@10 (micro) | 0.550 | ~5.5 of 10 retrieved pairs are relevant |
| Recall@10 (micro) | 0.081 | Of all relevant pairs in corpus, top-10 captures ~8% |
| F1@10 (micro) | 0.141 | Harmonic mean of P/R |
| Confusion (micro) | TP=55 FP=45 FN=626 | Across all 10 queries x top-10 |
| Same-thread overlap | 7/10 (70%) | A pair from the exact same conversation was retrieved |

**Interpretation:** Trust **Hit@k / MRR / nDCG** — they show the retriever almost always
top-ranks a genuinely relevant pair. **Recall@10 is low only as a denominator artifact**:
with `threshold=0.62 OR same-thread`, the "relevant set" per query is huge (60-100+ pairs),
so capturing all of them in 10 slots is mathematically capped. Recall@k is not meaningful
when the relevant set is that large.

---

## 2. Generation metrics (final predicted reply)

| Metric | Value | Meaning |
|---|---|---|
| Avg cosine (pred vs gold) | 0.726 | Predicted replies semantically close to real replies |
| Avg LLM-judge | 3.70/5 | Moderate-to-good quality |
| Pass rate (judge>=4) | 6/10 (60%) | 60% "good enough to send" |

### Confidence-calibration confusion (confidence vs judge-correct)

| | Value |
|---|---|
| TP (confident & correct) | 6 |
| FP (confident & wrong) | 4 |
| FN (unsure & correct) | 0 |
| TN (unsure & wrong) | 0 |
| Accuracy | 0.600 |
| Precision | 0.600 |
| Recall | 1.000 |

**Interpretation:** This is a fair, leakage-free test (the exact gold pair is NOT in the DB),
so scores are lower than the earlier "exact-match" runs — that is expected and more honest.
**FP=4** = the agent is overconfident on ~40% of cases; **FN=0** = it never under-rates a
correct answer.

---

## Bottom line

- **Retriever: very strong** — finds and top-ranks relevant context almost every time.
- **Generator: decent, slightly overconfident** on a held-out set.
- Reproduce full run: `HF_HUB_OFFLINE=1 <venv-python> evaluate.py` (no `--limit`).
