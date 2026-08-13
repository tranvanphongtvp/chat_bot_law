# Vietnamese Legal RAG Chatbot

Dự án này là một chatbot pháp lý dành cho luật pháp Việt Nam, sử dụng kiến trúc **Retrieval-Augmented Generation (RAG)** để cung cấp các câu trả lời chính xác, đáng tin cậy dựa trên các văn bản luật thực tế.

## Tính năng

- **Truy xuất thông tin pháp lý chính xác:** Sử dụng vector database (ChromaDB) và mô hình embedding đa ngôn ngữ (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) kết hợp BM25 (Hybrid Search) để tìm kiếm các trích đoạn luật liên quan đến câu hỏi.
- **Hỗ trợ đa dạng LLM:** Hỗ trợ nhiều mô hình LLM thông qua thiết kế linh hoạt (mặc định sử dụng Groq LLaMA 3.3, hỗ trợ mở rộng thêm OpenAI, Gemini, Claude thông qua API Key).
- **Giao diện Web:** Cung cấp giao diện web tương tác trực tiếp sử dụng FastAPI.

## Cấu trúc dự án chính

- `src/api/`: Chứa mã nguồn cho FastAPI server và thư mục `static` phục vụ frontend.
- `src/llm/`: Modules quản lý kết nối tới các dịch vụ/mô hình LLM. Sử dụng Factory Pattern để dễ dàng chuyển đổi giữa Groq, OpenAI, Gemini,...
- `src/embedding/`: Chứa logic xử lý embedding.
- `src/vectordb/`: Quản lý thao tác với ChromaDB.
- `src/retrieval/`: Xử lý logic tìm kiếm và truy xuất tài liệu (Advanced Retriever).
- `src/prompts/`: Chứa các prompt templates chuyên dụng cho hệ thống RAG và Chatbot.
- `storage/`: Thư mục chứa cơ sở dữ liệu sau khi ingest (ChromaDB, BM25 Index, Docstore).

## Cài đặt

### 1. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv .venv
```

Kích hoạt môi trường:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường

- Sao chép file `.env.example` thành `.env`.
- Điền API Key của Groq vào biến `GROQ_API_KEY`. Cấu hình này là BẮT BUỘC để bot có thể hoạt động.
- (Tùy chọn) Điền thêm các API Key khác nếu bạn đổi mô hình sang OpenAI, Gemini hoặc Claude.

## Chạy ứng dụng

### 1. Khởi động Web Server

Chạy trực tiếp:

```bash
python src/api/app.py
```

Hoặc sử dụng Uvicorn:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

## Truy cập ứng dụng

Sau khi khởi động thành công, mở trình duyệt và truy cập:

```text
http://localhost:8000
```
