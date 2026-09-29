import math
import numpy as np
from typing import List, Dict, Any, Optional

class RetrievalEvaluator:
    """
    Lớp đánh giá chất lượng của hệ thống truy xuất tài liệu (Retrieval)
    cho chatbot pháp luật Việt Nam.
    """

    def __init__(self, k: int = 5):
        """
        Khởi tạo RetrievalEvaluator.

        Args:
            k (int): Số lượng tài liệu top-k để đánh giá. Mặc định là 5.
        """
        self.k = k

    def _is_relevant(self, doc: Dict[str, Any], ground_truth_contexts: List[str], expected_source_docs: Optional[List[str]] = None) -> bool:
        """
        Kiểm tra xem một tài liệu truy xuất có liên quan đến các ngữ cảnh thực tế (ground truth) hay không.
        
        Args:
            doc: Tài liệu được truy xuất, định dạng dict.
            ground_truth_contexts: Danh sách các ngữ cảnh/câu đúng.
            expected_source_docs: Danh sách các văn bản nguồn dự kiến (ví dụ: ["Luat_31_2024_QH15"]).

        Returns:
            bool: True nếu tài liệu liên quan, False nếu không.
        """
        content = str(doc.get("content", "")).lower()
        metadata = doc.get("metadata", {})

        # Kiểm tra sự xuất hiện của ngữ cảnh trong nội dung hoặc siêu dữ liệu
        for context in ground_truth_contexts:
            context_lower = context.lower()
            if context_lower in content:
                return True
            
            # Kiểm tra trong các trường của metadata
            for key, value in metadata.items():
                if isinstance(value, str) and context_lower in value.lower():
                    return True

        # Kiểm tra theo văn bản nguồn (law) nếu có
        if expected_source_docs:
            law = str(metadata.get("law", "")).lower()
            for expected in expected_source_docs:
                if expected.lower() in law:
                    return True

        return False

    def _get_relevance_labels(self, retrieved_docs: List[Dict[str, Any]], ground_truth_contexts: List[str], expected_source_docs: Optional[List[str]] = None) -> List[int]:
        """
        Lấy danh sách nhãn nhị phân (1/0) cho top-k tài liệu.
        """
        k_docs = retrieved_docs[:self.k]
        return [1 if self._is_relevant(doc, ground_truth_contexts, expected_source_docs) else 0 for doc in k_docs]

    def recall_at_k(self, retrieved_docs: List[Dict[str, Any]], ground_truth_contexts: List[str], expected_source_docs: Optional[List[str]] = None) -> float:
        """
        Tính Recall@k: Tỷ lệ tài liệu liên quan được truy xuất so với tổng số tài liệu liên quan.
        (Giả sử tổng số tài liệu liên quan bằng số lượng ground_truth_contexts).
        """
        if not ground_truth_contexts and not expected_source_docs:
            return 0.0
            
        relevance = self._get_relevance_labels(retrieved_docs, ground_truth_contexts, expected_source_docs)
        retrieved_relevant = sum(relevance)
        # Số tài liệu liên quan lý thuyết (có thể được điều chỉnh tuỳ theo cách định nghĩa cụ thể)
        total_relevant = len(ground_truth_contexts) if ground_truth_contexts else len(expected_source_docs)
        
        if total_relevant == 0:
            return 0.0
            
        return min(float(retrieved_relevant) / total_relevant, 1.0)

    def mrr_at_k(self, retrieved_docs: List[Dict[str, Any]], ground_truth_contexts: List[str], expected_source_docs: Optional[List[str]] = None) -> float:
        """
        Tính Mean Reciprocal Rank (MRR@k): 1 / thứ hạng của tài liệu liên quan đầu tiên.
        """
        relevance = self._get_relevance_labels(retrieved_docs, ground_truth_contexts, expected_source_docs)
        for i, is_rel in enumerate(relevance):
            if is_rel:
                return 1.0 / (i + 1)
        return 0.0

    def ndcg_at_k(self, retrieved_docs: List[Dict[str, Any]], ground_truth_contexts: List[str], expected_source_docs: Optional[List[str]] = None) -> float:
        """
        Tính Normalized Discounted Cumulative Gain (NDCG@k).
        """
        relevance = self._get_relevance_labels(retrieved_docs, ground_truth_contexts, expected_source_docs)
        if not any(relevance):
            return 0.0

        # DCG@k
        dcg = sum((rel / math.log2(i + 2)) for i, rel in enumerate(relevance))

        # IDCG@k: Giả sử xếp hạng hoàn hảo (các tài liệu liên quan nằm trên cùng)
        total_relevant = len(ground_truth_contexts) if ground_truth_contexts else (len(expected_source_docs) if expected_source_docs else 0)
        # Đảm bảo IDCG >= DCG bằng cách dùng max giữa ground truth count và actual relevant count
        actual_relevant = sum(relevance)
        ideal_count = max(total_relevant, actual_relevant)
        ideal_relevance = [1] * min(ideal_count, self.k)
        if not ideal_relevance:
            ideal_relevance = sorted(relevance, reverse=True)
            
        idcg = sum((rel / math.log2(i + 2)) for i, rel in enumerate(ideal_relevance))

        return min(dcg / idcg, 1.0) if idcg > 0 else 0.0

    def precision_at_k(self, retrieved_docs: List[Dict[str, Any]], ground_truth_contexts: List[str], expected_source_docs: Optional[List[str]] = None) -> float:
        """
        Tính Precision@k: Tỷ lệ tài liệu liên quan trong số các tài liệu được truy xuất.
        """
        relevance = self._get_relevance_labels(retrieved_docs, ground_truth_contexts, expected_source_docs)
        retrieved_count = len(relevance)
        if retrieved_count == 0:
            return 0.0
        return float(sum(relevance)) / retrieved_count

    def hit_rate_at_k(self, retrieved_docs: List[Dict[str, Any]], ground_truth_contexts: List[str], expected_source_docs: Optional[List[str]] = None) -> float:
        """
        Tính Hit Rate@k: 1.0 nếu tìm thấy ít nhất 1 tài liệu liên quan, ngược lại là 0.0.
        """
        relevance = self._get_relevance_labels(retrieved_docs, ground_truth_contexts, expected_source_docs)
        return 1.0 if any(relevance) else 0.0

    def evaluate_all(self, retrieved_docs: List[Dict[str, Any]], ground_truth_contexts: List[str], expected_source_docs: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Tính toán tất cả 5 chỉ số cho kết quả truy xuất.
        
        Returns:
            Dict[str, float]: Một dictionary chứa các chỉ số recall_at_k, mrr_at_k, ndcg_at_k, precision_at_k, hit_rate_at_k.
        """
        return {
            f"recall_at_{self.k}": self.recall_at_k(retrieved_docs, ground_truth_contexts, expected_source_docs),
            f"mrr_at_{self.k}": self.mrr_at_k(retrieved_docs, ground_truth_contexts, expected_source_docs),
            f"ndcg_at_{self.k}": self.ndcg_at_k(retrieved_docs, ground_truth_contexts, expected_source_docs),
            f"precision_at_{self.k}": self.precision_at_k(retrieved_docs, ground_truth_contexts, expected_source_docs),
            f"hit_rate_at_{self.k}": self.hit_rate_at_k(retrieved_docs, ground_truth_contexts, expected_source_docs)
        }
