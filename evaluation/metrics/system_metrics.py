import re
from typing import List, Dict, Optional

class SystemEvaluator:
    """
    Lớp đánh giá các số đo cấp hệ thống cho mô hình chatbot pháp luật.
    """
    def __init__(self, llm_judge):
        """
        Khởi tạo SystemEvaluator.
        
        Args:
            llm_judge: Đối tượng LLMJudge dùng để xác định tính chất của câu trả lời.
        """
        self.llm_judge = llm_judge

    def citation_accuracy(self, answer: str, context_chunks: List[str]) -> float:
        """
        Trích xuất và kiểm chứng trích dẫn pháp lý có trong câu trả lời so với ngữ cảnh.
        
        Args:
            answer: Câu trả lời của hệ thống.
            context_chunks: Danh sách các đoạn văn bản ngữ cảnh được truy xuất.
            
        Returns:
            Tỷ lệ trích dẫn hợp lệ (số trích dẫn có trong ngữ cảnh / tổng số trích dẫn).
            Trả về 0.0 nếu không có trích dẫn nào trong câu trả lời.
        """
        # Các biểu thức chính quy để trích xuất trích dẫn (Điều, Khoản, Luật, Nghị định...)
        patterns = [
            r'Điều\s+\d+',
            r'Khoản\s+\d+',
            r'Luật\s+[^,\.\n]+',
            r'Bộ luật\s+[^,\.\n]+',
            r'Nghị định\s+[^,\.\n]+',
            r'Thông tư\s+[^,\.\n]+'
        ]
        
        extracted_citations = []
        for pattern in patterns:
            # Sử dụng re.IGNORECASE để bắt các trường hợp viết hoa/thường
            matches = re.findall(pattern, answer, re.IGNORECASE)
            extracted_citations.extend(matches)
            
        # Làm sạch các trích dẫn trích xuất được
        extracted_citations = [c.strip() for c in extracted_citations if c.strip()]
        
        if not extracted_citations:
            return 0.0
            
        combined_context = " ".join(context_chunks).lower()
        verified_count = 0
        
        for citation in extracted_citations:
            if citation.lower() in combined_context:
                verified_count += 1
                
        return verified_count / len(extracted_citations)

    def hallucination_rate(self, faithfulness_score: float) -> float:
        """
        Tính tỷ lệ ảo giác (hallucination) dựa trên điểm số trung thành (faithfulness).
        
        Args:
            faithfulness_score: Điểm số trung thành của câu trả lời (0.0 -> 1.0).
            
        Returns:
            Tỷ lệ ảo giác (1.0 - faithfulness_score).
        """
        return 1.0 - faithfulness_score

    def context_utilization(self, answer: str, context_chunks: List[str]) -> float:
        """
        Tính tỷ lệ sử dụng ngữ cảnh, xác định xem bao nhiêu đoạn ngữ cảnh được dùng trong câu trả lời.
        Một đoạn được coi là có sử dụng nếu có cụm từ (3 từ trở lên) của nó xuất hiện trong câu trả lời.
        
        Args:
            answer: Câu trả lời của hệ thống.
            context_chunks: Danh sách các đoạn văn bản ngữ cảnh.
            
        Returns:
            Tỷ lệ đoạn ngữ cảnh được sử dụng. Trả về 0.0 nếu không có ngữ cảnh.
        """
        if not context_chunks:
            return 0.0
            
        utilized_count = 0
        answer_lower = answer.lower()
        
        for chunk in context_chunks:
            words = chunk.split()
            utilized = False
            
            # Nếu đoạn văn bản quá ngắn, kiểm tra toàn bộ
            if len(words) < 3:
                if chunk.lower().strip() in answer_lower and chunk.strip():
                    utilized = True
            else:
                # Trích xuất các chuỗi 3 từ liên tiếp để kiểm tra
                for i in range(len(words) - 2):
                    phrase = " ".join(words[i:i+3]).lower()
                    if phrase in answer_lower:
                        utilized = True
                        break
                        
            if utilized:
                utilized_count += 1
                
        return utilized_count / len(context_chunks)

    def rejection_accuracy(self, answer: str, expected_rejection: bool) -> float:
        """
        Đánh giá độ chính xác của việc từ chối trả lời (đối với câu hỏi ngoài phạm vi).
        
        Args:
            answer: Câu trả lời của hệ thống.
            expected_rejection: Cờ xác định câu hỏi có mong đợi bị từ chối hay không.
            
        Returns:
            1.0 nếu hệ thống thực hiện đúng (từ chối khi cần và ngược lại), 0.0 nếu sai.
        """
        is_rejection = self.llm_judge.judge_rejection(answer)
        return 1.0 if (is_rejection == expected_rejection) else 0.0

    def evaluate_all(self, answer: str, context_chunks: List[str], faithfulness_score: float, latency: float, expected_rejection: bool = False) -> Dict[str, float]:
        """
        Đánh giá toàn diện câu trả lời với tất cả các số đo.
        
        Args:
            answer: Câu trả lời của hệ thống.
            context_chunks: Danh sách các đoạn văn bản ngữ cảnh.
            faithfulness_score: Điểm số trung thành.
            latency: Thời gian phản hồi của hệ thống.
            expected_rejection: Cờ báo hiệu câu hỏi có nằm ngoài phạm vi hay không.
            
        Returns:
            Từ điển chứa kết quả của các số đo.
        """
        citation_acc = self.citation_accuracy(answer, context_chunks)
        hallucination = self.hallucination_rate(faithfulness_score)
        context_util = self.context_utilization(answer, context_chunks)
        rejection_acc = self.rejection_accuracy(answer, expected_rejection)
        
        return {
            "citation_accuracy": citation_acc,
            "hallucination_rate": hallucination,
            "context_utilization": context_util,
            "rejection_accuracy": rejection_acc,
            "latency": latency
        }
