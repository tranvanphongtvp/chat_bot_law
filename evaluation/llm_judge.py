import os
import json
import re
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.llm.llm_client import LLMFactory
from evaluation.config import (
    JUDGE_LLM_PROVIDER,
    JUDGE_LLM_MODEL,
    JUDGE_LLM_TEMPERATURE,
    JUDGE_LLM_MAX_TOKENS
)

class LLMJudge:
    """
    LLM Judge class that uses an LLM to evaluate generation quality.
    """
    
    def __init__(self):
        """Khởi tạo LLM client sử dụng cấu hình từ evaluation.config"""
        api_key = os.getenv("GROQ_API_KEY", "")
        self.llm_client = LLMFactory.create_llm(
            provider=JUDGE_LLM_PROVIDER,
            api_key=api_key,
            model_name=JUDGE_LLM_MODEL
        )

    def _parse_json_response(self, response: str) -> dict:
        """Trích xuất và parse JSON từ phản hồi của LLM"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Tìm kiếm khối markdown json
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Tìm kiếm JSON dictionary {...}
        match = re.search(r"(\{.*\})", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        raise ValueError("Không thể parse JSON từ phản hồi của LLM")

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 2) -> dict:
        """Gọi LLM và parse kết quả JSON, có cơ chế retry"""
        prompt += "\n\nHãy chỉ trả về duy nhất một chuỗi JSON hợp lệ theo yêu cầu, không kèm theo bất kỳ văn bản giải thích nào khác."
        for attempt in range(max_retries + 1):
            try:
                response_text = self.llm_client.generate(
                    prompt=prompt,
                    temperature=JUDGE_LLM_TEMPERATURE,
                    max_tokens=JUDGE_LLM_MAX_TOKENS
                )
                return self._parse_json_response(response_text)
            except Exception as e:
                if attempt == max_retries:
                    print(f"Lỗi khi gọi LLM hoặc parse JSON sau {max_retries} retries: {e}")
                    return {}
        return {}

    def judge_faithfulness(self, answer: str, context_chunks: list) -> float:
        """
        Đánh giá độ chính xác của câu trả lời dựa trên ngữ cảnh cung cấp.
        
        Args:
            answer (str): Câu trả lời của hệ thống.
            context_chunks (list): Danh sách các đoạn văn bản ngữ cảnh.
            
        Returns:
            float: Tỷ lệ các khẳng định được hỗ trợ (0.0 - 1.0)
        """
        context_str = "\n".join(context_chunks) if isinstance(context_chunks, list) else str(context_chunks)
        
        prompt = f"""Bạn là một giám khảo công bằng đánh giá độ chính xác (faithfulness) của câu trả lời dựa trên ngữ cảnh cung cấp.

Ngữ cảnh:
{context_str}

Câu trả lời cần đánh giá:
{answer}

Nhiệm vụ của bạn:
1. Trích xuất các khẳng định (claims) từ câu trả lời.
2. Kiểm tra xem mỗi khẳng định có được hỗ trợ bởi ngữ cảnh hay không.
3. Tính điểm = số khẳng định được hỗ trợ / tổng số khẳng định (nằm trong khoảng 0.0 đến 1.0).

Yêu cầu đầu ra bắt buộc phải là định dạng JSON sau:
{{"claims": [{{"claim": "khẳng định 1...", "supported": true}}, ...], "score": 0.85}}
"""
        try:
            result = self._call_llm_with_retry(prompt)
            score = float(result.get("score", 0.0))
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0

    def judge_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Đánh giá mức độ câu trả lời giải quyết câu hỏi.
        
        Args:
            question (str): Câu hỏi của người dùng.
            answer (str): Câu trả lời của hệ thống.
            
        Returns:
            float: Điểm số chuẩn hóa (0.0 - 1.0)
        """
        prompt = f"""Bạn là một chuyên gia đánh giá mức độ liên quan của câu trả lời so với câu hỏi.

Câu hỏi: {question}
Câu trả lời: {answer}

Nhiệm vụ:
Đánh giá mức độ câu trả lời giải quyết câu hỏi trên thang điểm từ 0 đến 10.

Yêu cầu đầu ra bắt buộc phải là định dạng JSON sau (với "score" là số từ 0 đến 10):
{{"score": 8, "reasoning": "lý do đánh giá..."}}
"""
        try:
            result = self._call_llm_with_retry(prompt)
            score = float(result.get("score", 0.0))
            # Chuẩn hóa về 0.0 - 1.0
            if score > 1.0:
                score = score / 10.0
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0

    def judge_answer_completeness(self, answer: str, ground_truth_answer: str) -> float:
        """
        Đánh giá tính đầy đủ của câu trả lời so với câu trả lời chuẩn (ground truth).
        
        Args:
            answer (str): Câu trả lời của hệ thống.
            ground_truth_answer (str): Câu trả lời chuẩn xác.
            
        Returns:
            float: Tỷ lệ bao phủ các ý chính (0.0 - 1.0)
        """
        prompt = f"""Bạn là một chuyên gia đánh giá tính đầy đủ của câu trả lời so với một câu trả lời chuẩn.

Câu trả lời chuẩn (ground truth): {ground_truth_answer}
Câu trả lời cần đánh giá: {answer}

Nhiệm vụ:
1. Xác định các ý chính cần có từ câu trả lời chuẩn.
2. Kiểm tra xem câu trả lời cần đánh giá bao phủ được bao nhiêu phần trăm (%) các ý chính đó.
3. Tính toán tổng số ý chính, số ý chính đã bao phủ, và điểm số bao phủ (từ 0.0 đến 1.0).

Yêu cầu đầu ra bắt buộc phải là định dạng JSON sau:
{{"key_points_total": 5, "key_points_covered": 4, "score": 0.8}}
"""
        try:
            result = self._call_llm_with_retry(prompt)
            score = float(result.get("score", 0.0))
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0

    def judge_rejection(self, answer: str) -> bool:
        """
        Kiểm tra xem câu trả lời có phải là lời từ chối (không tìm thấy thông tin) hay không.
        
        Args:
            answer (str): Câu trả lời của hệ thống.
            
        Returns:
            bool: True nếu là câu trả lời từ chối, False nếu ngược lại.
        """
        rejection_phrases = [
            "chưa tìm thấy",
            "không có thông tin",
            "ngoài phạm vi",
            "không tìm thấy",
            "chưa có thông tin",
            "không thể trả lời"
        ]
        
        answer_lower = answer.lower()
        for phrase in rejection_phrases:
            if phrase in answer_lower:
                return True
        return False
