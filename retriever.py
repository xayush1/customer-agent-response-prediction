"""Semantic retrieval against the Milvus turn-pair collection."""
from functools import lru_cache

import numpy as np
from pymilvus import Collection, connections, utility

from config import get_config
from embeddings import embed_documents, embed_query

_OUTPUT_FIELDS = [
    "customer_msg",
    "agent_reply",
    "context",
    "subject",
    "thread_id",
    "turn_position",
]


@lru_cache
def _collection() -> Collection:
    cfg = get_config()
    if not connections.has_connection("default"):
        connections.connect(alias="default", host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT)
    if not utility.has_collection(cfg.MILVUS_COLLECTION):
        raise RuntimeError(
            f"Collection '{cfg.MILVUS_COLLECTION}' not found. Run ingest.py first."
        )
    col = Collection(cfg.MILVUS_COLLECTION)
    col.load()
    return col


def _trim(text: str, limit: int = 1200) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def search(query: str, top_k: int = 5) -> list[dict]:
    """Find past turn-pairs whose customer message is similar to `query`."""
    col = _collection()
    vec = embed_query(query)
    results = col.search(
        data=[vec],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 16}},
        limit=top_k,
        output_fields=_OUTPUT_FIELDS,
    )
    hits = []
    for hit in results[0]:
        e = hit.entity
        hits.append(
            {
                "score": round(float(hit.score), 4),
                "thread_id": e.get("thread_id"),
                "subject": e.get("subject"),
                "turn_position": e.get("turn_position"),
                "customer_msg": _trim(e.get("customer_msg")),
                "agent_reply": _trim(e.get("agent_reply"), 1800),
                "context": _trim(e.get("context"), 800),
            }
        )
    return hits


def get_thread_context(
    thread_id: str, query: str | None = None, max_pairs: int | None = None
) -> list[dict]:
    """Fetch a BOUNDED set of agent-customer pairs from a thread.

    Instead of dumping the entire thread, we return at most `max_pairs`
    (default `THREAD_MAX_PAIRS`) turn-pairs. When `query` is provided, the pairs
    are selected by relevance (cosine of their customer message to the query) so
    the agent only sees the related back-and-forth, not the whole conversation.
    The returned pairs are re-sorted by position to preserve readability.
    """
    cfg = get_config()
    if max_pairs is None:
        max_pairs = cfg.THREAD_MAX_PAIRS
    col = _collection()
    safe_id = thread_id.replace('"', "")
    rows = col.query(
        expr=f'thread_id == "{safe_id}"',
        output_fields=_OUTPUT_FIELDS,
        limit=50,
    )

    if query and len(rows) > max_pairs:
        qv = np.asarray(embed_query(query), dtype=float)
        msg_vecs = np.asarray(embed_documents([r.get("customer_msg", "") for r in rows]), dtype=float)
        scores = msg_vecs @ qv  # embeddings are L2-normalized -> dot = cosine
        top_idx = np.argsort(scores)[::-1][:max_pairs]
        rows = [rows[i] for i in top_idx]
    else:
        rows = rows[:max_pairs]

    rows.sort(key=lambda r: r.get("turn_position", 0))
    return [
        {
            "thread_id": r.get("thread_id"),
            "subject": r.get("subject"),
            "turn_position": r.get("turn_position"),
            "customer_msg": _trim(r.get("customer_msg")),
            "agent_reply": _trim(r.get("agent_reply"), 1800),
        }
        for r in rows
    ]


def stats() -> dict:
    col = _collection()
    return {"collection": col.name, "entities": col.num_entities}


if __name__ == "__main__":
    import json

    res = search("My account was suspended, please help urgently", top_k=3)
    print(json.dumps(res, indent=2, ensure_ascii=False))
