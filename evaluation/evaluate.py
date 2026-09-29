"""
Main Evaluation Runner cho RAG Pipeline - Chatbot Luật Việt Nam.

Chạy đánh giá toàn bộ pipeline với các chỉ số:
- Retrieval: Recall@5, MRR@5, NDCG@5, Precision@5, HitRate@5
- Generation: Faithfulness, Answer Relevancy, Answer Correctness, Answer Completeness
- System: Citation Accuracy, Hallucination Rate, Latency, Context Utilization, Rejection Accuracy

Cách chạy:
    python -m evaluation.evaluate
    python -m evaluation.evaluate --group retrieval
    python -m evaluation.evaluate --group generation
    python -m evaluation.evaluate --category lookup
    python -m evaluation.evaluate --limit 5
"""
import sys
import os
import json
import time
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Thêm project root vào sys.path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluation.config import (
    TEST_DATASET_PATH, CHROMA_DB_PATH, BM25_INDEX_PATH, DOCSTORE_PATH,
    EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE,
    RETRIEVER_TOP_K, RETRIEVER_FETCH_K,
    LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    DEFAULT_K, METRIC_GROUPS
)
from evaluation.metrics.retrieval_metrics import RetrievalEvaluator
from evaluation.metrics.generation_metrics import GenerationEvaluator
from evaluation.metrics.system_metrics import SystemEvaluator
from evaluation.llm_judge import LLMJudge
from evaluation.report import print_console_report, save_json_report, save_markdown_report


def load_test_dataset(path: str = None, category: str = None, limit: int = None) -> List[Dict]:
    """Tải bộ dữ liệu test."""
    dataset_path = path or str(TEST_DATASET_PATH)
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if category:
        dataset = [tc for tc in dataset if tc.get("category") == category]
        print(f"[*] Lọc theo category '{category}': {len(dataset)} test cases")

    if limit:
        dataset = dataset[:limit]
        print(f"[*] Giới hạn: {limit} test cases")

    return dataset


def initialize_pipeline():
    """
    Khởi tạo toàn bộ pipeline RAG (Embedding, VectorDB, BM25, Retriever, LLM).
    Trả về tuple (retriever, llm).
    """
    from dotenv import load_dotenv
    load_dotenv()

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key == "your_groq_api_key_here":
        print("[!] LỖI: Vui lòng cung cấp GROQ_API_KEY trong file .env")
        sys.exit(1)

    # 1. Embedding
    print("[1/4] Đang tải mô hình Embedding...")
    from src.embedding.providers import HuggingFaceEmbeddingProvider
    provider = HuggingFaceEmbeddingProvider(
        model_name=EMBEDDING_MODEL_NAME,
        device=EMBEDDING_DEVICE
    )
    embedder = provider.get_embedding()

    # 2. VectorDB
    print("[2/4] Đang kết nối VectorDB (ChromaDB)...")
    from src.vectordb.vector_store import ChromaDBStore
    vector_store = ChromaDBStore(
        embedding_model=embedder,
        persist_directory=CHROMA_DB_PATH
    )

    # 3. BM25 + Parent Store
    print("[3/4] Đang tải BM25 Index và Parent Store...")
    bm25_retriever = None
    if Path(BM25_INDEX_PATH).exists():
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_retriever = pickle.load(f)
    else:
        print("[!] CẢNH BÁO: Không tìm thấy BM25 index. Sẽ chỉ dùng Semantic Search.")

    from src.retrieval.parent_child_store import ParentChildStore
    parent_store = ParentChildStore(DOCSTORE_PATH)

    # 4. Retriever
    print("[4/4] Đang khởi tạo Advanced Retriever...")
    from src.retrieval.retriever import AdvancedRetriever
    retriever = AdvancedRetriever(
        vector_db_client=vector_store,
        bm25_retriever=bm25_retriever,
        embedding_model=embedder,
        top_k=RETRIEVER_TOP_K,
        fetch_k=RETRIEVER_FETCH_K,
        parent_store=parent_store
    )

    # 5. LLM
    from src.llm.llm_client import LLMFactory
    llm = LLMFactory.create_llm(
        provider=LLM_PROVIDER,
        api_key=groq_key,
        model_name=LLM_MODEL
    )

    return retriever, llm


def run_pipeline_for_query(retriever, llm, question: str) -> Dict[str, Any]:
    """
    Chạy pipeline RAG cho 1 câu hỏi.
    Trả về: {
        "retrieved_docs": [...],
        "answer": "...",
        "context_chunks": ["...", ...],
        "retrieval_latency": float,
        "generation_latency": float,
        "total_latency": float
    }
    """
    from src.prompts.promt_templates import RAG_QA_PROMPT, SYSTEM_PROMPT_DEFAULT

    total_start = time.time()

    # Step 1: Retrieval
    retrieval_start = time.time()
    retrieved_docs = retriever.retrieve(query=question)
    retrieval_latency = time.time() - retrieval_start

    # Chuẩn bị context
    context_chunks = []
    context_text = ""
    for i, doc in enumerate(retrieved_docs):
        content = doc.get('parent_content') or doc['content']
        context_chunks.append(content)
        source = doc['metadata'].get('law', doc['metadata'].get('file_name', 'Unknown'))
        article = doc['metadata'].get('article', '')
        context_text += f"\n--- Trích đoạn {i+1} ({source} - {article}) ---\n{content}\n"

    # Step 2: Generation
    generation_start = time.time()
    if retrieved_docs:
        prompt = RAG_QA_PROMPT.format(context=context_text, question=question)
        answer = llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT_DEFAULT,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
    else:
        answer = "Dựa trên các tài liệu hiện có, tôi chưa tìm thấy thông tin cụ thể để trả lời câu hỏi này."
    generation_latency = time.time() - generation_start

    total_latency = time.time() - total_start

    return {
        "retrieved_docs": retrieved_docs,
        "answer": answer,
        "context_chunks": context_chunks,
        "retrieval_latency": retrieval_latency,
        "generation_latency": generation_latency,
        "total_latency": total_latency,
    }


def evaluate_single_case(
    test_case: Dict,
    pipeline_result: Dict,
    retrieval_evaluator: RetrievalEvaluator,
    generation_evaluator: GenerationEvaluator,
    system_evaluator: SystemEvaluator,
    groups: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Đánh giá 1 test case với tất cả các metrics.
    """
    result = {
        "id": test_case["id"],
        "question": test_case["question"],
        "category": test_case.get("category", "unknown"),
        "difficulty": test_case.get("difficulty", "unknown"),
        "answer": pipeline_result["answer"],
        "num_retrieved": len(pipeline_result["retrieved_docs"]),
        "retrieval_metrics": {},
        "generation_metrics": {},
        "system_metrics": {},
    }

    run_retrieval = groups is None or "retrieval" in groups
    run_generation = groups is None or "generation" in groups
    run_system = groups is None or "system" in groups

    # --- Retrieval Metrics ---
    if run_retrieval:
        ground_truth_contexts = test_case.get("ground_truth_contexts", [])
        expected_source_docs = test_case.get("expected_source_docs", [])
        result["retrieval_metrics"] = retrieval_evaluator.evaluate_all(
            retrieved_docs=pipeline_result["retrieved_docs"],
            ground_truth_contexts=ground_truth_contexts,
            expected_source_docs=expected_source_docs
        )

    # --- Generation Metrics ---
    faithfulness_score = 0.0
    if run_generation:
        generation_metrics = generation_evaluator.evaluate_all(
            question=test_case["question"],
            answer=pipeline_result["answer"],
            context_chunks=pipeline_result["context_chunks"],
            ground_truth_answer=test_case.get("ground_truth_answer", "")
        )
        result["generation_metrics"] = generation_metrics
        faithfulness_score = generation_metrics.get("faithfulness", 0.0)

    # --- System Metrics ---
    if run_system:
        expected_rejection = test_case.get("expected_rejection", False)
        system_metrics = system_evaluator.evaluate_all(
            answer=pipeline_result["answer"],
            context_chunks=pipeline_result["context_chunks"],
            faithfulness_score=faithfulness_score,
            latency=pipeline_result["total_latency"],
            expected_rejection=expected_rejection
        )
        result["system_metrics"] = system_metrics

    return result


def aggregate_results(per_question_results: List[Dict]) -> Dict[str, float]:
    """
    Tính trung bình các metrics từ tất cả test cases.
    """
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)

    for result in per_question_results:
        for group in ["retrieval_metrics", "generation_metrics", "system_metrics"]:
            for metric, value in result.get(group, {}).items():
                if isinstance(value, (int, float)):
                    metric_sums[metric] += value
                    metric_counts[metric] += 1

    aggregated = {}
    for metric in metric_sums:
        if metric_counts[metric] > 0:
            aggregated[metric] = metric_sums[metric] / metric_counts[metric]

    # Rename latency key nếu cần
    if "latency" in aggregated:
        aggregated["average_latency"] = aggregated.pop("latency")

    return aggregated


def analyze_by_category(per_question_results: List[Dict]) -> Dict[str, Dict]:
    """
    Phân tích kết quả theo category (lookup, comparison, reasoning, etc.)
    """
    category_data = defaultdict(lambda: {
        "count": 0,
        "recall_sum": 0.0,
        "faithfulness_sum": 0.0,
        "latency_sum": 0.0,
    })

    for result in per_question_results:
        cat = result.get("category", "unknown")
        category_data[cat]["count"] += 1
        category_data[cat]["recall_sum"] += result.get("retrieval_metrics", {}).get("recall_at_5", 0.0)
        category_data[cat]["faithfulness_sum"] += result.get("generation_metrics", {}).get("faithfulness", 0.0)
        category_data[cat]["latency_sum"] += result.get("system_metrics", {}).get("latency", 0.0)

    category_results = {}
    for cat, data in category_data.items():
        count = data["count"]
        if count > 0:
            category_results[cat] = {
                "count": count,
                "avg_recall_at_5": data["recall_sum"] / count,
                "avg_faithfulness": data["faithfulness_sum"] / count,
                "avg_latency": data["latency_sum"] / count,
            }

    return category_results


def main():
    parser = argparse.ArgumentParser(description="Đánh giá RAG Pipeline - Chatbot Luật Việt Nam")
    parser.add_argument("--group", type=str, choices=["retrieval", "generation", "system"],
                        help="Chỉ chạy 1 nhóm metric")
    parser.add_argument("--category", type=str,
                        help="Lọc test cases theo category (lookup, comparison, reasoning, multi_doc, negative, ambiguous)")
    parser.add_argument("--limit", type=int,
                        help="Giới hạn số lượng test cases")
    parser.add_argument("--dataset", type=str,
                        help="Đường dẫn tới file test dataset (mặc định: evaluation/test_dataset.json)")
    parser.add_argument("--no-report", action="store_true",
                        help="Không lưu báo cáo ra file")
    args = parser.parse_args()

    print("=" * 60)
    print(" ĐÁNH GIÁ RAG PIPELINE - CHATBOT LUẬT VIỆT NAM ".center(60, "="))
    print("=" * 60)

    # 1. Load test dataset
    print("\n[STEP 1] Đang tải bộ dữ liệu test...")
    dataset = load_test_dataset(
        path=args.dataset,
        category=args.category,
        limit=args.limit
    )
    print(f"[*] Đã tải {len(dataset)} test cases")

    if not dataset:
        print("[!] Không có test case nào. Thoát.")
        return

    # 2. Khởi tạo pipeline
    print("\n[STEP 2] Đang khởi tạo pipeline RAG...")
    retriever, llm = initialize_pipeline()
    print("[+] Pipeline đã sẵn sàng!")

    # 3. Khởi tạo evaluators
    print("\n[STEP 3] Đang khởi tạo bộ đánh giá...")
    retrieval_evaluator = RetrievalEvaluator(k=DEFAULT_K)
    llm_judge = LLMJudge()
    generation_evaluator = GenerationEvaluator(llm_judge=llm_judge)
    system_evaluator = SystemEvaluator(llm_judge=llm_judge)
    print("[+] Evaluators đã sẵn sàng!")

    # Xác định groups cần chạy
    groups = [args.group] if args.group else None

    # 4. Chạy evaluation
    print(f"\n[STEP 4] Bắt đầu đánh giá {len(dataset)} test cases...")
    print("=" * 60)

    per_question_results = []
    for i, test_case in enumerate(dataset):
        test_id = test_case["id"]
        question = test_case["question"]
        category = test_case.get("category", "unknown")

        print(f"\n{'─' * 60}")
        print(f"  [{i+1}/{len(dataset)}] {test_id} ({category})")
        print(f"  Q: {question[:80]}{'...' if len(question) > 80 else ''}")
        print(f"{'─' * 60}")

        try:
            # Chạy pipeline
            pipeline_result = run_pipeline_for_query(retriever, llm, question)
            print(f"  📥 Retrieved: {len(pipeline_result['retrieved_docs'])} docs | ⏱️ {pipeline_result['total_latency']:.2f}s")
            print(f"  📝 Answer: {pipeline_result['answer'][:100]}...")

            # Đánh giá
            eval_result = evaluate_single_case(
                test_case=test_case,
                pipeline_result=pipeline_result,
                retrieval_evaluator=retrieval_evaluator,
                generation_evaluator=generation_evaluator,
                system_evaluator=system_evaluator,
                groups=groups
            )
            per_question_results.append(eval_result)

            # In tóm tắt nhanh
            if eval_result.get("retrieval_metrics"):
                r = eval_result["retrieval_metrics"]
                print(f"  📦 Recall={r.get('recall_at_5', 0):.2f} MRR={r.get('mrr_at_5', 0):.2f} NDCG={r.get('ndcg_at_5', 0):.2f}")
            if eval_result.get("generation_metrics"):
                g = eval_result["generation_metrics"]
                print(f"  🤖 Faith={g.get('faithfulness', 0):.2f} Relev={g.get('answer_relevancy', 0):.2f} Correct={g.get('answer_correctness', 0):.2f}")
            if eval_result.get("system_metrics"):
                s = eval_result["system_metrics"]
                print(f"  🏗️  Citation={s.get('citation_accuracy', 0):.2f} Halluc={s.get('hallucination_rate', 0):.2f} Latency={s.get('latency', 0):.2f}s")

        except Exception as e:
            print(f"  [!] LỖI: {e}")
            per_question_results.append({
                "id": test_id,
                "question": question,
                "category": category,
                "error": str(e),
                "retrieval_metrics": {},
                "generation_metrics": {},
                "system_metrics": {},
            })

    # 5. Tổng hợp kết quả
    print(f"\n\n[STEP 5] Đang tổng hợp kết quả...")
    aggregated = aggregate_results(per_question_results)
    category_results = analyze_by_category(per_question_results)

    # 6. Xuất báo cáo
    print_console_report(aggregated, per_question_results, category_results)

    if not args.no_report:
        save_json_report(aggregated, per_question_results, category_results)
        save_markdown_report(aggregated, per_question_results, category_results)

    print("\n✅ Đánh giá hoàn tất!")


if __name__ == "__main__":
    main()
