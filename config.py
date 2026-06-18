"""Central configuration loaded from .env."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Force offline HF so we use the locally cached BGE model (avoids SSL issues).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(_ENV_PATH)


class Config:
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    # Milvus
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "agent_response_pairs")

    # Embedding
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

    # Data
    DATASET_PATH = os.getenv("DATASET_PATH", "")

    # Agent
    REACT_MAX_DEPTH = int(os.getenv("REACT_MAX_DEPTH", "3"))
    CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "3"))
    # When drilling into a thread, feed at most this many RELATED agent-customer
    # pairs (ranked by relevance to the query) instead of the entire thread.
    THREAD_MAX_PAIRS = int(os.getenv("THREAD_MAX_PAIRS", "4"))

    # API
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8888"))
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8888")

    # Milvus VARCHAR cap
    MAX_VARCHAR = 65000

    # Evaluation / train-test split (pair-level: 1 conversation pair = 1 objective)
    TEST_RATIO = float(os.getenv("TEST_RATIO", "0.2"))
    SPLIT_SEED = int(os.getenv("SPLIT_SEED", "7"))
    # A retrieved train pair counts as RELEVANT to a test objective if it comes
    # from the SAME conversation thread OR its agent reply is at least this
    # cosine-similar to the held-out gold agent reply (differently-worded but
    # equivalent answers still count).
    REL_THRESHOLD = float(os.getenv("REL_THRESHOLD", "0.62"))
    # LLM-judge score (1-5) at/above which a predicted reply is "acceptable".
    JUDGE_PASS = int(os.getenv("JUDGE_PASS", "4"))
    TEST_SPLIT_PATH = os.getenv(
        "TEST_SPLIT_PATH", str(Path(__file__).parent / "test_split.json")
    )


@lru_cache
def get_config() -> Config:
    return Config()
