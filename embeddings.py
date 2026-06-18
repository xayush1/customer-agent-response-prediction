"""Local BGE embedding model (cached, offline)."""
from functools import lru_cache

from config import get_config


@lru_cache
def get_embedder():
    """Load the local BAAI/bge-base-en-v1.5 model from cache (offline)."""
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings

    cfg = get_config()
    return HuggingFaceBgeEmbeddings(
        model_name=cfg.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_query(text: str) -> list[float]:
    return get_embedder().embed_query(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return get_embedder().embed_documents(texts)
