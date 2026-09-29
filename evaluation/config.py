"""
Cấu hình cho hệ thống đánh giá RAG Pipeline.
"""
import os
from pathlib import Path

# ============================================================
# ĐƯỜNG DẪN
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
EVALUATION_DIR = Path(__file__).parent
TEST_DATASET_PATH = EVALUATION_DIR / "test_dataset.json"
RESULTS_DIR = EVALUATION_DIR / "results"

# Storage paths
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DB_PATH = str(STORAGE_DIR / "chroma_db")
BM25_INDEX_PATH = STORAGE_DIR / "bm25_index.pkl"
DOCSTORE_PATH = str(STORAGE_DIR / "docstore.json")

# Data paths
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "final"

# ============================================================
# PIPELINE CONFIG
# ============================================================
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE = "cpu"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retriever config
RETRIEVER_TOP_K = 5
RETRIEVER_FETCH_K = 20
HYBRID_WEIGHTS = [0.7, 0.3]  # [semantic, bm25]

# LLM config
LLM_PROVIDER = "groq"
LLM_MODEL = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1500

# ============================================================
# NGƯỠNG ĐÁNH GIÁ (THRESHOLDS)
# ============================================================
THRESHOLDS = {
    # Retrieval
    "recall_at_5": 0.80,
    "mrr_at_5": 0.70,
    "ndcg_at_5": 0.75,
    "precision_at_5": 0.60,
    "hit_rate_at_5": 0.90,
    # Generation
    "faithfulness": 0.90,
    "answer_relevancy": 0.80,
    "answer_correctness": 0.75,
    "answer_completeness": 0.70,
    # System
    "citation_accuracy": 0.95,
    "hallucination_rate": 0.05,  # Threshold là upper bound (≤)
    "average_latency": 10.0,     # Seconds, upper bound (≤)
    "context_utilization": 0.50,
    "rejection_accuracy": 0.90,
}

# Các metric mà giá trị thấp hơn threshold là tốt (ngược với các metric khác)
LOWER_IS_BETTER = {"hallucination_rate", "average_latency"}

# ============================================================
# LLM-AS-JUDGE CONFIG
# ============================================================
JUDGE_LLM_PROVIDER = "groq"
JUDGE_LLM_MODEL = "openai/gpt-oss-120b"
JUDGE_LLM_TEMPERATURE = 0.0
JUDGE_LLM_MAX_TOKENS = 1000

# ============================================================
# EVALUATION CONFIG
# ============================================================
DEFAULT_K = 5  # Mặc định @K cho retrieval metrics

# Metric groups cho CLI filtering
METRIC_GROUPS = {
    "retrieval": [
        "recall_at_5", "mrr_at_5", "ndcg_at_5",
        "precision_at_5", "hit_rate_at_5"
    ],
    "generation": [
        "faithfulness", "answer_relevancy",
        "answer_correctness", "answer_completeness"
    ],
    "system": [
        "citation_accuracy", "hallucination_rate",
        "average_latency", "context_utilization", "rejection_accuracy"
    ],
}
