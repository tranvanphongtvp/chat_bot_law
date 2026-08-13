import os
from pathlib import Path
from dotenv import load_dotenv
from embedding.providers import HuggingFaceEmbeddingProvider
from vectordb.vector_store import ChromaDBStore
from retrieval.retriever import AdvancedRetriever
from llm.llm_client import LLMFactory
from prompts.promt_templates import SYSTEM_PROMPT_DEFAULT, RAG_QA_PROMPT
def main():
    print("="*50)
    print(" KHỞI ĐỘNG HỆ THỐNG CHATBOT LUẬT ".center(50, "="))
    print("="*50)
    # 1. Load Biến Môi Trường (API Keys)
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key == "your_groq_api_key_here":
        print("[!] LỖI: Vui lòng cung cấp GROQ_API_KEY trong file .env")
        return
    # 2. Khởi tạo LLM Client (Groq)
    print("[1/3] Đang kết nối tới Groq (Llama 3.3)...")
    llm = LLMFactory.create_llm(provider="groq", api_key=groq_key, model_name="llama-3.3-70b-versatile")
    # 3. Khởi tạo Embedding & VectorDB (Chroma)
    print("[2/3] Đang tải mô hình Embedding và VectorDB...")
    provider = HuggingFaceEmbeddingProvider(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )
    embedder = provider.get_embedding()
    project_root = Path(__file__).parent.parent
    db_path = str(project_root / "storage" / "chroma_db")
    
    vector_store = ChromaDBStore(
        embedding_model=embedder,
        persist_directory=db_path
    )
    # 4. Khởi tạo Retriever
    print("[3/3] Đang khởi tạo bộ Tìm kiếm (Retriever)...")
    retriever = AdvancedRetriever(vector_db_client=vector_store, top_k=3)
    print("\n" + "="*50)
    print(" HỆ THỐNG ĐÃ SẴN SÀNG ".center(50, "="))
    print(" Gõ 'exit' hoặc 'quit' để thoát ".center(50, " "))
    print("="*50 + "\n")
    # 5. Vòng lặp Chat (Chat Loop)
    while True:
        try:
            query = input("\n👤 Bạn: ")
            if query.lower() in ["exit", "quit", "thoát"]:
                print("Tạm biệt!")
                break
            
            if not query.strip():
                continue
            # BƯỚC A: Retrieval (Kéo dữ liệu)
            print("⏳ Đang tra cứu tài liệu luật...")
            retrieved_docs = retriever.retrieve(query=query)
            
            if not retrieved_docs:
                print("🤖 Bot: Không tìm thấy tài liệu luật nào liên quan trong kho dữ liệu.")
                continue
            # Chuẩn bị context (Ngữ cảnh)
            context_text = ""
            for i, doc in enumerate(retrieved_docs):
                source = doc['metadata'].get('title', doc['metadata'].get('file_name', 'Unknown'))
                context_text += f"\n--- Trích xuất {i+1} (Từ: {source}) ---\n"
                context_text += f"{doc['content']}\n"
            # BƯỚC B: Áp dụng Prompt Templates
            final_prompt = RAG_QA_PROMPT.format(
                context=context_text,
                question=query
            )
            # BƯỚC C: Generation (Tạo câu trả lời bằng LLM)
            print("⏳ Đang phân tích và trả lời...")
            answer = llm.generate(
                prompt=final_prompt,
                system_prompt=SYSTEM_PROMPT_DEFAULT,
                temperature=0.0
            )
            # In câu trả lời
            print("\n🤖 Bot:")
            print(answer)
            print("-" * 50)
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"\n[!] Lỗi xảy ra: {e}")
if __name__ == "__main__":
    main()