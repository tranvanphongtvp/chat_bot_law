"""
Generation Metrics - Đánh giá chất lượng câu trả lời sinh ra bởi LLM.

Bao gồm:
- Faithfulness: Câu trả lời có bịa thông tin ngoài context không?
- Answer Relevancy: Câu trả lời có đúng trọng tâm câu hỏi không?
- Answer Correctness: Câu trả lời chính xác bao nhiêu so với đáp án chuẩn?
- Answer Completeness: Câu trả lời có đầy đủ các ý chính không?
"""
import re
from typing import List, Dict


class GenerationEvaluator:
    """
    Lớp đánh giá chất lượng câu trả lời sinh ra bởi LLM
    cho chatbot pháp luật Việt Nam.
    """

    def __init__(self, llm_judge):
        """
        Khởi tạo GenerationEvaluator.

        Args:
            llm_judge: Đối tượng LLMJudge dùng để đánh giá bằng LLM.
        """
        self.llm_judge = llm_judge

    def faithfulness(self, answer: str, context_chunks: List[str]) -> float:
        """
        Đánh giá độ trung thành (faithfulness) của câu trả lời với context.
        Kiểm tra từng claim trong câu trả lời có được hỗ trợ bởi context không.

        Args:
            answer: Câu trả lời của hệ thống.
            context_chunks: Danh sách các đoạn ngữ cảnh đã retrieve.

        Returns:
            float: Tỷ lệ claims được hỗ trợ (0.0 - 1.0).
        """
        if not answer or not context_chunks:
            return 0.0
        return self.llm_judge.judge_faithfulness(answer, context_chunks)

    def answer_relevancy(self, question: str, answer: str) -> float:
        """
        Đánh giá mức độ câu trả lời giải quyết đúng trọng tâm câu hỏi.

        Args:
            question: Câu hỏi của người dùng.
            answer: Câu trả lời của hệ thống.

        Returns:
            float: Điểm relevancy (0.0 - 1.0).
        """
        if not answer or not question:
            return 0.0
        return self.llm_judge.judge_answer_relevancy(question, answer)

    def answer_correctness(self, answer: str, ground_truth_answer: str) -> float:
        """
        Đánh giá độ chính xác câu trả lời so với đáp án chuẩn.
        Kết hợp 2 sub-scores:
          a) Token-level F1 score (trọng số 0.6)
          b) Jaccard similarity trên tập từ (trọng số 0.4)

        Args:
            answer: Câu trả lời của hệ thống.
            ground_truth_answer: Đáp án chuẩn (ground truth).

        Returns:
            float: Điểm correctness tổng hợp (0.0 - 1.0).
        """
        if not answer or not ground_truth_answer:
            return 0.0

        # a) Token-level F1 Score
        f1 = self._token_f1_score(answer, ground_truth_answer)

        # b) Jaccard Similarity (word-level)
        jaccard = self._jaccard_similarity(answer, ground_truth_answer)

        # Tổng hợp: 60% F1 + 40% Jaccard
        return 0.6 * f1 + 0.4 * jaccard

    def answer_completeness(self, answer: str, ground_truth_answer: str) -> float:
        """
        Đánh giá tính đầy đủ — câu trả lời bao phủ bao nhiêu % ý chính
        từ đáp án chuẩn.

        Args:
            answer: Câu trả lời của hệ thống.
            ground_truth_answer: Đáp án chuẩn (ground truth).

        Returns:
            float: Tỷ lệ ý chính được bao phủ (0.0 - 1.0).
        """
        if not answer or not ground_truth_answer:
            return 0.0
        return self.llm_judge.judge_answer_completeness(answer, ground_truth_answer)

    def evaluate_all(
        self,
        question: str,
        answer: str,
        context_chunks: List[str],
        ground_truth_answer: str
    ) -> Dict[str, float]:
        """
        Chạy tất cả 4 metrics đánh giá generation.

        Args:
            question: Câu hỏi của người dùng.
            answer: Câu trả lời của hệ thống.
            context_chunks: Danh sách các đoạn ngữ cảnh.
            ground_truth_answer: Đáp án chuẩn.

        Returns:
            Dict chứa 4 metrics: faithfulness, answer_relevancy,
            answer_correctness, answer_completeness.
        """
        return {
            "faithfulness": self.faithfulness(answer, context_chunks),
            "answer_relevancy": self.answer_relevancy(question, answer),
            "answer_correctness": self.answer_correctness(answer, ground_truth_answer),
            "answer_completeness": self.answer_completeness(answer, ground_truth_answer),
        }

    # =============================================
    # HELPER METHODS
    # =============================================

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tách text thành danh sách tokens (từ) bằng regex.
        Hỗ trợ tiếng Việt có dấu.
        """
        return re.findall(r'\w+', text.lower())

    @staticmethod
    def _token_f1_score(prediction: str, reference: str) -> float:
        """
        Tính Token-level F1 Score giữa prediction và reference.

        F1 = 2 * Precision * Recall / (Precision + Recall)
        - Precision = |pred_tokens ∩ ref_tokens| / |pred_tokens|
        - Recall    = |pred_tokens ∩ ref_tokens| / |ref_tokens|
        """
        pred_tokens = re.findall(r'\w+', prediction.lower())
        ref_tokens = re.findall(r'\w+', reference.lower())

        if not pred_tokens or not ref_tokens:
            return 0.0

        # Dùng multiset (bag) để đếm trùng lặp chính xác
        from collections import Counter
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)

        # Giao giữa 2 multisets
        overlap = sum((pred_counter & ref_counter).values())

        if overlap == 0:
            return 0.0

        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """
        Tính Jaccard Similarity giữa 2 tập từ.

        Jaccard = |A ∩ B| / |A ∪ B|
        """
        set_a = set(re.findall(r'\w+', text_a.lower()))
        set_b = set(re.findall(r'\w+', text_b.lower()))

        if not set_a and not set_b:
            return 0.0

        intersection = set_a & set_b
        union = set_a | set_b

        return len(intersection) / len(union) if union else 0.0
