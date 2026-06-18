# Customer Agent Response Prediction

> **Predict what a support agent should reply** — given a customer message and past conversation history — using a ReAct agent backed by a Milvus vector store and Azure OpenAI gpt-4.1.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Project structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Quick-start setup](#quick-start-setup)
6. [Using the TUI](#using-the-tui)
7. [Running evaluations](#running-evaluations)
8. [Evaluation results](#evaluation-results)
9. [Configuration reference](#configuration-reference)
10. [Good to have / roadmap](#good-to-have--roadmap)

---

## What it does

Given a new customer message (and optionally a multi-turn conversation history), the system:

1. **Retrieves** the most semantically similar past agent–customer turn-pairs from a Milvus vector database (BGE embeddings, 768-dim).
2. **Reasons** over those pairs using a ReAct loop (reason → tool-call → observe → repeat ≤ 3 rounds).
3. **Generates** a grounded agent reply via Azure OpenAI gpt-4.1.

Every turn-pair in the database was extracted from real email support threads and stored as `(customer_message, agent_reply)` units — the agent never sees the entire thread, only the *most relevant* back-and-forth (bounded by `THREAD_MAX_PAIRS`).

---

## Architecture

```
Customer message
      │
      ▼
FastAPI  /predict
      │
      ▼
 ReAct Agent  (gpt-4.1 + function calling)
      │
      ├── search_similar_conversations ──▶ Milvus semantic search
      │                                        (BGE 768-dim, HNSW index)
      ├── get_full_thread ──────────────▶ Milvus thread drill-down
      │                                    (top-k relevant pairs, NOT full thread)
      └── finish ───────────────────────▶ Final agent reply  ──▶ caller
```

**Data pipeline (one-time setup):**

```
conversations.json
      │
      ▼
preprocess.py  (strip quotes / signatures, extract turn-pairs)
      │  80/20 split (pair-level, deterministic seed)
      ├─ 80% train pairs ──▶ ingest.py ──▶ Milvus collection
      └─ 20% test pairs  ──▶ test_split.json  (used by evaluate.py)
```

**Key design choices:**

- 1 conversation pair = 1 independent training/evaluation objective (not 1 thread).
- The exact gold pair is **never** in the database during evaluation (leakage-free).
- Thread context is bounded (`THREAD_MAX_PAIRS=4`) and ranked by relevance to the query — no full-thread dumps.

---

## Project structure

```
agent_predict/
├── .env.example          # Template — copy to .env and fill in your keys
├── requirements.txt      # Python dependencies
├── run.sh                # One-liner: start backend + TUI
│
├── config.py             # All settings (loaded from .env)
├── preprocess.py         # Clean emails, extract turn-pairs
├── embeddings.py         # Local BGE-base embedder (HuggingFace, CPU)
├── split.py              # Deterministic 80/20 pair-level train/test split
├── ingest.py             # Embed train pairs → Milvus collection
│
├── retriever.py          # Semantic search + bounded thread drill-down
├── react_agent.py        # ReAct loop (reasoning + tool calls)
│
├── api.py                # FastAPI backend  (/predict, /health, /stats)
├── tui.py                # Rich terminal UI (multi-turn chat)
│
├── evaluate.py           # Full eval: retrieval metrics + generation metrics
├── test_accuracy.py      # Quick holdout accuracy check (cosine + LLM-judge)
│
├── evaluation_results.md # Latest evaluation numbers + interpretation
└── result.md             # Earlier accuracy snapshot
```

---

## Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 / 3.12 recommended |
| Milvus | 2.4+ | Run locally via Docker (see below) |
| Azure OpenAI | — | `gpt-4.1` deployment required |
| RAM | 4 GB+ | BGE model loads on CPU (~450 MB) |
| Disk | ~2 GB | Model cache + Milvus volumes |

### Start Milvus with Docker

```bash
# Standalone Milvus (simplest)
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh | bash
# or with docker compose:
wget https://github.com/milvus-io/milvus/releases/download/v2.4.1/milvus-standalone-docker-compose.yml \
     -O docker-compose.yml
docker compose up -d
```

Milvus will be reachable at `localhost:19530`.

---

## Quick-start setup

```bash
# 1. Clone the repo
git clone https://github.com/sigmacoder1/customer-agent-response-prediction.git
cd customer-agent-response-prediction

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — fill in AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, DATASET_PATH

# 5. Pre-download the BGE embedding model (one-time, needs internet)
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

# 6. Ingest the dataset into Milvus (one-time; also writes test_split.json)
python3 ingest.py

# 7. Start the backend + TUI
./run.sh
# or with an explicit python path:
# PY=/path/to/.venv/bin/python3 ./run.sh
```

> **Tip:** After the first `ingest.py` run, steps 5–6 are not needed again unless you change the dataset or want to re-split.

---

## Using the TUI

```
./run.sh
```

The terminal UI opens a multi-turn chat session. You act as the **customer**; the system predicts the **agent** reply.

| Command | What it does |
|---|---|
| *(type any message)* | Send as customer — system predicts agent reply |
| `/load <thread_id>` | Pre-load a real conversation thread up to the last customer turn |
| `/threads` | List available thread IDs from the dataset |
| `/trace` | Toggle the ReAct reasoning trace (shows tool calls + observations) |
| `/reset` | Clear conversation history and start fresh |
| `/quit` | Exit the TUI |

### Example session

```
You: My order hasn't arrived and it's been 2 weeks.

Agent: I'm sorry to hear about the delay! Could you please provide your order
       number so I can look into this for you right away?
```

Enable `/trace` to see the agent's reasoning steps:

```
[THOUGHT]  The customer has a delayed order. I should search for similar cases.
[TOOL]     search_similar_conversations("delayed order 2 weeks")
[OBSERVE]  3 similar cases found — agents asked for order number first.
[ANSWER]   I'm sorry to hear about the delay! ...
```

---

## Running evaluations

### Full evaluation (retrieval + generation + LLM judge)

```bash
# Full held-out test set (~41 objectives; takes ~10-15 min due to LLM calls)
python3 evaluate.py

# Quick run — first 10 objectives only
python3 evaluate.py --limit 10

# Change retrieval cutoff k
python3 evaluate.py --limit 10 --top_k 5

# Skip LLM-as-judge (much faster, no generation cost)
python3 evaluate.py --limit 10 --no_judge
```

### Quick accuracy check (small holdout)

```bash
python3 test_accuracy.py
```

---

## Evaluation results

> Full numbers are in [`evaluation_results.md`](evaluation_results.md).

**10-objective run** (`k=10`, `rel_threshold=0.62`, `judge_pass=4`):

### Retrieval quality

| Metric | Value | Meaning |
|---|---|---|
| Hit@10 | **1.000** | Every query found a relevant pair in the top-10 |
| MRR | **0.900** | First relevant hit is almost always rank 1 |
| nDCG@10 | **0.870** | Relevant pairs ranked near the top |
| Precision@10 | 0.550 | ~5.5 out of 10 retrieved pairs are relevant |
| Recall@10 | 0.081 | Low denominator artifact (see note below) |

> **Note on Recall@10:** With `threshold=0.62 OR same-thread` the "relevant set" per query spans 60–100+ pairs, so it's mathematically impossible to capture all of them in 10 slots. Trust **Hit@k / MRR / nDCG** as the primary retriever quality signal.

### Generation quality

| Metric | Value | Meaning |
|---|---|---|
| Avg cosine similarity | 0.726 | Predicted replies are semantically close to gold |
| Avg LLM-judge score | 3.70 / 5 | Moderate-to-good quality |
| Pass rate (judge ≥ 4) | 60% | 6 of 10 replies "good enough to send" |

### Confidence calibration

| | Value |
|---|---|
| TP (confident & correct) | 6 |
| FP (confident & wrong) | 4 |
| FN (unsure & correct) | 0 |
| Accuracy | 0.600 |

> The agent is slightly **overconfident** (FP=4): when it says it's confident it is only right 60% of the time. No false negatives — it never under-rates a correct answer.

---

## Configuration reference

All settings live in `.env`. Full list with defaults:

| Key | Default | Description |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | *(required)* | Your Azure OpenAI resource URL |
| `AZURE_OPENAI_API_KEY` | *(required)* | API key |
| `AZURE_OPENAI_API_VERSION` | `2024-08-01-preview` | API version |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4.1` | Model deployment name |
| `MILVUS_HOST` | `localhost` | Milvus server host |
| `MILVUS_PORT` | `19530` | Milvus server port |
| `MILVUS_COLLECTION` | `agent_response_pairs` | Collection name |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace model ID |
| `EMBEDDING_DIM` | `768` | Embedding dimension |
| `DATASET_PATH` | *(required)* | Absolute path to `conversations.json` |
| `REACT_MAX_DEPTH` | `3` | Max ReAct reasoning rounds |
| `CONTEXT_WINDOW` | `3` | Past turns fed as conversation context |
| `THREAD_MAX_PAIRS` | `4` | Max related pairs fetched per thread drill-down |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8888` | FastAPI port |
| `TEST_RATIO` | `0.2` | Fraction of pairs held out for evaluation |
| `SPLIT_SEED` | `7` | Random seed for deterministic split |
| `REL_THRESHOLD` | `0.62` | Cosine similarity threshold for "relevant" in retrieval eval |
| `JUDGE_PASS` | `4` | LLM-judge score (1–5) at/above which a reply "passes" |

---

## Good to have / roadmap

- [ ] **Support your own dataset** — provide any `conversations.json` with `[{"thread_id": ..., "messages": [{"role": "customer"|"agent", "body": "..."}]}]`  format.
- [ ] **GPU inference** — set `CUDA_VISIBLE_DEVICES=0` and install a CUDA-enabled PyTorch build for faster BGE embeddings.
- [ ] **Attu UI** — browse the Milvus collection visually at `http://localhost:8000` with [Attu](https://github.com/zilliztech/attu).
- [ ] **Streaming replies** — wire SSE streaming from the `/predict` endpoint to the TUI for real-time token display.
- [ ] **Re-ranking** — add a cross-encoder re-ranker on the top-k retrieved pairs before feeding the agent.
- [ ] **Confidence calibration fix** — post-process the agent's stated confidence with a Platt scaler trained on the calibration set.
- [ ] **Web UI** — replace the TUI with a lightweight React frontend for broader accessibility.
- [ ] **Docker Compose all-in-one** — bundle Milvus + the API server in a single `docker-compose.yml`.
