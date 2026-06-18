"""Embed turn-pairs and load them into a Milvus collection."""
import sys

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from config import get_config
from embeddings import embed_documents
from split import save_test_split, split_pairs


def connect():
    cfg = get_config()
    connections.connect(alias="default", host=cfg.MILVUS_HOST, port=cfg.MILVUS_PORT)


def build_collection(drop: bool = True) -> Collection:
    cfg = get_config()
    name = cfg.MILVUS_COLLECTION
    if utility.has_collection(name):
        if drop:
            utility.drop_collection(name)
        else:
            return Collection(name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=cfg.EMBEDDING_DIM),
        FieldSchema(name="customer_msg", dtype=DataType.VARCHAR, max_length=cfg.MAX_VARCHAR),
        FieldSchema(name="agent_reply", dtype=DataType.VARCHAR, max_length=cfg.MAX_VARCHAR),
        FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=cfg.MAX_VARCHAR),
        FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="thread_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="turn_position", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="Customer->Agent turn pairs for response prediction")
    col = Collection(name=name, schema=schema)
    col.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}},
    )
    return col


def main():
    cfg = get_config()
    print(f"Loading conversations from {cfg.DATASET_PATH} ...")
    print(f"Splitting pairs {int((1 - cfg.TEST_RATIO) * 100)}/{int(cfg.TEST_RATIO * 100)} "
          f"(pair-level, seed={cfg.SPLIT_SEED}) ...")
    train, test = split_pairs()
    print(f"Train pairs (ingested): {len(train)}  |  Test pairs (held out): {len(test)}")
    if not train:
        print("No train pairs to ingest. Aborting.")
        sys.exit(1)

    # Persist the held-out 20% so evaluate.py uses exactly the same objectives.
    path = save_test_split(test)
    print(f"Wrote held-out test split -> {path}")

    print("Connecting to Milvus ...")
    connect()
    print("Building collection (dropping any existing data) ...")
    col = build_collection(drop=True)

    print(f"Embedding {len(train)} customer messages (BGE, CPU) ...")
    embeddings = embed_documents([p["customer_msg"] for p in train])

    data = [
        embeddings,
        [p["customer_msg"] for p in train],
        [p["agent_reply"] for p in train],
        [p["context"] for p in train],
        [p["subject"] for p in train],
        [p["thread_id"] for p in train],
        [p["turn_position"] for p in train],
    ]
    print("Inserting into Milvus ...")
    col.insert(data)
    col.flush()
    col.load()
    print(f"Done. Collection '{cfg.MILVUS_COLLECTION}' now has {col.num_entities} entities "
          f"(train only; {len(test)} test pairs excluded).")


if __name__ == "__main__":
    main()
