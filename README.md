# 🔧 Gemini RAG - Tool-Calling Q&A on PDF

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline powered by **Google Gemini**, built using **LangChain tools**, **FAISS vector search**, and a **Streamlit UI**.

You can upload a PDF, and the app will extract, chunk, embed, and retrieve relevant sections to answer your questions using Gemini.

---

## 📌 Features

- 📄 Upload PDF documents
- ✂️ Chunk and embed text using Sentence Transformers
- 🔍 Vector-based semantic search with FAISS
- 💬 Question-answering via Google Gemini (`gemini-pro`)
- 🧠 Modular, tool-based design using LangChain's `@tool` decorator
- ⚡ Interactive frontend using Streamlit

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that improves LLM responses by retrieving relevant information from external documents and feeding it into the model as context.

Instead of relying on the model’s internal memory alone, it dynamically pulls relevant content to generate more accurate, grounded answers.

---

## 🛠️ How It Works

1. **PDF Upload** – Load your document.
2. **Text Chunking** – Break it into manageable segments.
3. **Embedding** – Convert text into vector representations.
4. **Semantic Search** – Match your question to the most relevant chunks.
5. **Gemini Q&A** – Provide the context to Gemini and get a natural language response.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/gemini-rag-pdf.git
cd gemini-rag-pdf
