from typing import Optional
from abc import ABC, abstractmethod
import os
import requests
try:
    from openai import OpenAI
except ImportError:
    pass

class BaseLLM(ABC):
    """
    Interface cơ sở cho mọi LLM. 
    Điều này giúp chúng ta dễ dàng thay đổi Model mà không phải sửa lại code ở những nơi khác.
    """
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        pass

class GroqLLM(BaseLLM):
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})

        # print(f"[*] Gọi Groq ({self.model_name}) thực thi prompt...")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 1500)
        )
        return response.choices[0].message.content

class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})

        # print(f"[*] Gọi OpenAI ({self.model_name}) thực thi prompt...")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.0), # RAG nên set temperature thấp để tránh bịa chuyện
            max_tokens=kwargs.get("max_tokens", 1500)
        )
        return response.choices[0].message.content

class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        # TODO: Import và Khởi tạo client của Google GenAI

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        # TODO: Viết code gọi API Gemini tại đây
        print(f"[*] Gọi Gemini ({self.model_name}) thực thi prompt...")
        return "[Gemini Result Mock] Nội dung trả về từ Gemini."

class ClaudeLLM(BaseLLM):
    def __init__(self, api_key: str, model_name: str = "claude-3-haiku-20240307"):
        self.api_key = api_key
        self.model_name = model_name
        # TODO: Import và Khởi tạo client của Anthropic

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        # TODO: Viết code gọi API Claude tại đây
        print(f"[*] Gọi Claude ({self.model_name}) thực thi prompt...")
        return "[Claude Result Mock] Nội dung trả về từ Claude."


class LLMFactory:
    """
    Sử dụng Factory pattern để cung cấp hàm tạo LLM linh hoạt dựa trên lựa chọn của bạn.
    """
    @staticmethod
    def create_llm(provider: str, api_key: str = "", model_name: Optional[str] = None) -> BaseLLM:
        provider = provider.lower()
        if provider == "openai":
            return OpenAILLM(api_key, model_name or "gpt-4o-mini")
        elif provider == "gemini":
            return GeminiLLM(api_key, model_name or "gemini-1.5-flash")
        elif provider == "claude":
            return ClaudeLLM(api_key, model_name or "claude-3-haiku-20240307")
        elif provider == "groq":
            return GroqLLM(api_key, model_name or "llama-3.3-70b-versatile")
        else:
            raise ValueError(f"Không hỗ trợ provider LLM: {provider}")
