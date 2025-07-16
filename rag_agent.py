import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
# from langchain.tools import tool  # No longer needed

# 📌 Set Gemini API key here
GEMINI_API_KEY = "AIzaSyDR938WLM0_Ni5QRWB6xx2NrFoNH4rJa3I"  # Replace with your actual key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# 🧠 Memory store
memory = {}

# 🛠️ Tool: Load PDF
# @tool

def load_pdf(file_path: str) -> str:
    """Loads a PDF document from the given file path and stores it in memory."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    memory["docs"] = docs
    return f"{len(docs)} pages loaded."

# 🛠️ Tool: Chunk Text
# @tool

def chunk_text(chunk_size: int = 500, chunk_overlap: int = 100) -> str:
    """Chunks the loaded document into smaller segments."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(memory["docs"])
    memory["chunks"] = chunks
    return f"Chunked into {len(chunks)} pieces."

# 🛠️ Tool: Embed Chunks
# @tool

def embed_chunks() -> str:
    """Embeds the document chunks using a sentence transformer model and stores in FAISS index."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [chunk.page_content for chunk in memory["chunks"]]
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    memory["texts"] = texts
    memory["index"] = index
    memory["embed_model"] = model
    return f"Embedded {len(texts)} chunks and stored in FAISS."

# 🛠️ Tool: Query Chunks
# @tool

def query_chunks(query: str, top_k: int = 3) -> str:
    """Finds the top relevant document chunks for the given query."""
    model = memory["embed_model"]
    index = memory["index"]
    texts = memory["texts"]
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)
    results = [texts[i] for i in indices[0]]
    context = "\n".join(results)
    memory["context"] = context
    return context

# 🛠️ Tool: Answer Question with Gemini
# @tool

def answer_question(question: str) -> str:
    """Uses Gemini to answer a question based on retrieved context."""
    model = genai.GenerativeModel("gemini-2.5-flash")
    context = memory.get("context", "")
    # Efficient, focused prompt
    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the answer is not in the context, say 'I don't know based on the document.'\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer: "
    )
    response = model.generate_content(prompt)
    return response.text

# 🌐 Streamlit UI
st.set_page_config(page_title="Gemini RAG Q&A (Tool Calling)", layout="centered")
st.title("🔧 Gemini RAG - Tool Calling Q&A on PDF")

# 📄 Upload PDF
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file is not None:
    file_path = os.path.join("temp.pdf")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Process PDF (call tools directly)
    st.success(load_pdf(file_path))
    st.info(chunk_text())
    st.info(embed_chunks())

    # --- Chat Interface ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("---")
    st.header("💬 Chat with your document")

    for chat in st.session_state.chat_history:
        st.chat_message(chat["role"]).write(chat["content"])

    user_input = st.chat_input("Ask a question about the document...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        st.info("🔍 Retrieving context...")
        context = query_chunks(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": "(Context retrieved)"})
        st.chat_message("assistant").write("(Context retrieved)")
        st.success("💬 Answering with Gemini...")
        answer = answer_question(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
