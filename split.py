"""Deterministic 80/20 train/test split at the turn-pair level.

The user's chosen unit is "1 conversation pair = 1 objective": every
customer->agent turn-pair is an independent evaluation objective. We shuffle all
extracted pairs with a fixed seed and hold out TEST_RATIO of them. Only the train
pairs are ingested into Milvus; the held-out test pairs are used by evaluate.py.

Both ingest.py and evaluate.py import this module so they always agree on the
split (deterministic for a fixed SPLIT_SEED / TEST_RATIO).
"""
import json

import random

from config import get_config
from preprocess import extract_turn_pairs, load_conversations


def pair_id(p: dict) -> str:
    """Stable identifier for a turn-pair (thread + position in thread)."""
    return f"{p['thread_id']}::{p['turn_position']}"


def split_pairs() -> tuple[list[dict], list[dict]]:
    """Return (train_pairs, test_pairs) using a deterministic pair-level split."""
    cfg = get_config()
    convos = load_conversations(cfg.DATASET_PATH)
    pairs = extract_turn_pairs(convos, window=cfg.CONTEXT_WINDOW, max_varchar=cfg.MAX_VARCHAR)
    for p in pairs:
        p["pair_id"] = pair_id(p)

    order = list(range(len(pairs)))
    random.Random(cfg.SPLIT_SEED).shuffle(order)
    n_test = int(round(len(pairs) * cfg.TEST_RATIO))
    test_set = set(order[:n_test])

    train = [pairs[i] for i in range(len(pairs)) if i not in test_set]
    test = [pairs[i] for i in range(len(pairs)) if i in test_set]
    return train, test


def save_test_split(test: list[dict], path: str | None = None) -> str:
    cfg = get_config()
    path = path or cfg.TEST_SPLIT_PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)
    return path


def load_test_split(path: str | None = None) -> list[dict]:
    cfg = get_config()
    path = path or cfg.TEST_SPLIT_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    tr, te = split_pairs()
    print(f"Total pairs: {len(tr) + len(te)}")
    print(f"Train (ingested): {len(tr)}")
    print(f"Test  (held out): {len(te)}")
