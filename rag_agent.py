import os
import google.generativeai as genai
from pinecone import Pinecone

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==== STEP 1: Set your API Keys and Host ====
os.environ["GOOGLE_API_KEY"] = "AIzaSyB0YmJ6rP_dxkBamxq_ueMwGoppIFW7A2U"
os.environ["PINECONE_API_KEY"] = "pcsk_4aAC2L_5zgaeGk1wbn6cmDYbdtJSA3M4vnbTFNmhAMSmPQM7F365RRQw6uVBnnhFfahcNm"
PINECONE_HOST = "https://rag-p0g64lw.svc.aped-4627-b74a.pinecone.io"
PINECONE_INDEX_NAME = "rag"
PINECONE_NAMESPACE = "nips2017"

# ==== STEP 2: Initialize Gemini ====
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ==== RAG System Class ====
class SimpleRAGSystem:
    def __init__(self):
        self.docs = []
        self.chunks = []
        self.index = None
        
    def load_pdf(self, file_path: str) -> str:
        """Load a PDF file from the given path."""
        if not os.path.exists(file_path):
            return f"❌ File not found: {file_path}"
        
        loader = PyPDFLoader(file_path)
        self.docs = loader.load()
        return f"✅ Loaded {len(self.docs)} pages from PDF."
    
    def chunk_documents(self, chunk_size: int = 500, chunk_overlap: int = 50) -> str:
        """Split documents into chunks."""
        if not self.docs:
            return "❌ No documents loaded."
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        self.chunks = splitter.split_documents(self.docs)
        
        # Add metadata
        for i, chunk in enumerate(self.chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["source"] = "nips2017"
        
        return f"✅ Chunked into {len(self.chunks)} pieces."
    
    def embed_to_pinecone(self) -> str:
        """Embed chunks and store in Pinecone."""
        if not self.chunks:
            return "❌ No chunks to embed."
        
        try:
            # Initialize Pinecone client
            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            
            # Use an embedding model that produces 1024 dimensions to match your Pinecone index
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
            
            # Check if index exists
            if PINECONE_INDEX_NAME not in pc.list_indexes().names():
                return f"❌ Index '{PINECONE_INDEX_NAME}' not found in Pinecone. Please create it first."
            
            # Connect to existing index
            index = pc.Index(PINECONE_INDEX_NAME)
            
            # Store in Pinecone using LangChain
            self.index = LangchainPinecone(
                index=index,
                embedding=embeddings,
                text_key="text",
                namespace=PINECONE_NAMESPACE
            )
            
            # Add documents to the index
            self.index.add_documents(self.chunks, namespace=PINECONE_NAMESPACE)
            
            return f"✅ Stored {len(self.chunks)} chunks into Pinecone index '{PINECONE_INDEX_NAME}' under namespace '{PINECONE_NAMESPACE}'."
        
        except Exception as e:
            return f"❌ Error embedding to Pinecone: {str(e)}"
    
    def query(self, question: str) -> str:
        """Query the system using Gemini."""
        if not self.index:
            return "❌ Index not found. Please embed data first."
        
        try:
            # Retrieve relevant documents
            retriever = self.index.as_retriever(
                search_kwargs={"k": 3, "namespace": PINECONE_NAMESPACE}
            )
            docs = retriever.get_relevant_documents(question)
            
            if not docs:
                return "⚠️ No relevant documents found."
            
            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt for Gemini
            prompt = f"""
Use the following context to answer the question:

{context}

Question: {question}
Answer:"""
            
            # Generate response using Gemini
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        
        except Exception as e:
            return f"❌ Error querying: {str(e)}"

# ==== Main Execution ====
if __name__ == "__main__":
    # Initialize RAG system
    rag_system = SimpleRAGSystem()
    
    # File path
    file_path = r"data\NIPS-2017-attention-is-all-you-need-Paper.pdf"
    
    # Step 1: Load PDF
    print("\n📂 Loading PDF...")
    result = rag_system.load_pdf(file_path)
    print(result)
    
    # Step 2: Chunk documents
    print("\n✂️ Chunking text...")
    result = rag_system.chunk_documents()
    print(result)
    
    # Step 3: Embed to Pinecone
    print("\n🧠 Embedding into Pinecone...")
    result = rag_system.embed_to_pinecone()
    print(result)
    
    # Step 4: Query the system
    print("\n❓ Asking a question...")
    question = "What is the main idea of the document?"
    answer = rag_system.query(question)
    print(f"\n🧠 Gemini's Answer:\n{answer}")
    
    # Additional questions
    print("\n❓ Follow-up questions...")
    questions = [
        "What is the transformer architecture?",
        "What are the key components of the attention mechanism?",
        "How does the model perform compared to previous approaches?"
    ]
    
    for q in questions:
        print(f"\n🔍 Question: {q}")
        answer = rag_system.query(q)
        print(f"📝 Answer: {answer}")