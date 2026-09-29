import sys
import os
from pathlib import Path
import json
import requests

# Fix encoding cho Windows terminal (cp1258 không hỗ trợ đầy đủ Unicode tiếng Việt)
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Cấu hình đường dẫn
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

from src.embedding.providers import HuggingFaceEmbeddingProvider
from src.vectordb.vector_store import ChromaDBStore
from src.retrieval.retriever import AdvancedRetriever
from src.retrieval.parent_child_store import ParentChildStore
from src.llm.llm_client import LLMFactory
from src.prompts.promt_templates import RAG_QA_PROMPT, SYSTEM_PROMPT_DEFAULT

app = FastAPI(title="Vietnamese Legal RAG Chatbot")

# --- Khởi tạo Hệ thống RAG (Load 1 lần duy nhất) ---
print("[*] Đang khởi tạo Hệ thống RAG...")
try:
    provider = HuggingFaceEmbeddingProvider(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )
    embedder = provider.get_embedding()

    db_path = str(project_root / "storage" / "chroma_db")
    vector_store = ChromaDBStore(embedding_model=embedder, persist_directory=db_path)

    bm25_path = project_root / "storage" / "bm25_index.pkl"
    bm25_retriever = None
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)

    docstore_path = project_root / "storage" / "docstore.json"
    parent_store = ParentChildStore(str(docstore_path))

    retriever = AdvancedRetriever(
        vector_db_client=vector_store,
        bm25_retriever=bm25_retriever,
        embedding_model=embedder,
        top_k=5,
        fetch_k=20,
        parent_store=parent_store
    )
    print("[+] Khởi tạo thành công!")
except Exception as e:
    print(f"[!] Lỗi khi khởi tạo RAG: {e}")
    retriever = None

# --- API Models ---
class ChatRequest(BaseModel):
    query: str
    model: str = "openai/gpt-oss-120b" # Default model Groq

@app.get("/api/llm-status")
def check_llm_status():
    """Kiểm tra trạng thái cấu hình Groq."""
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key or groq_key == "your_groq_api_key_here":
            raise ValueError("Chưa cấu hình GROQ_API_KEY")
        return {"status": "running", "models": ["openai/gpt-oss-120b"]}
    except Exception as e:
        return {"status": "error", "message": str(e), "models": []}

# --- Cấu hình Frontend tĩnh ---
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def read_root():
    return FileResponse(str(static_dir / "index.html"))

@app.post("/api/chat")
def chat_with_bot(request: ChatRequest):
    if not retriever:
        raise HTTPException(status_code=500, detail="RAG System chưa được khởi tạo thành công.")

    query = request.query
    
    # 1. Tìm kiếm context từ VectorDB
    try:
        search_results = retriever.retrieve(query=query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tìm kiếm: {str(e)}")

    # Gộp context lại
    context_text = ""
    citations = []
    
    for i, res in enumerate(search_results):
        meta = res['metadata']
        law = meta.get('law', 'Unknown')
        article = meta.get('article', 'Unknown')
        
        # Ưu tiên lấy parent_content nếu có (Parent Expansion)
        content = res.get('parent_content') or res['content']
        context_text += f"\n--- Trích đoạn {i+1} ({law} - {article}) ---\n{content}\n"
        
        citations.append({
            "law": law,
            "article": article,
            "content": content[:200] + "..." # Chỉ gửi 1 ít preview về frontend để làm source
        })

    # 2. Xây dựng Prompt cho LLM
    prompt = RAG_QA_PROMPT.format(context=context_text, question=query)

    # 3. Gọi LLM thông qua LLMFactory
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key or groq_key == "your_groq_api_key_here":
            raise HTTPException(status_code=500, detail="Vui lòng cấu hình GROQ_API_KEY trong file .env")
            
        llm = LLMFactory.create_llm(provider="groq", api_key=groq_key, model_name=request.model)
        answer = llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT_DEFAULT)
        
        return {
            "answer": answer,
            "citations": citations
        }
    except Exception as e:
        error_msg = str(e)
        print(f"[!] {error_msg}")
        answer_text = f"❌ Lỗi: {error_msg}"
        
        if "Không thể kết nối" in error_msg or "Vui lòng cấu hình" in error_msg:
            answer_text = f"❌ Lỗi LLM: {error_msg}\n\n📄 Tuy nhiên, đây là các đoạn luật tôi tìm được:\n\n" + context_text
        elif "không phản hồi" in error_msg:
            answer_text = f"⏱️ Timeout: {error_msg}"
            
        return {
            "answer": answer_text,
            "citations": citations,
            "error": error_msg
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
