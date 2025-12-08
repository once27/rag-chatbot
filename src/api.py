import os
import sys
import time
from typing import List
from contextlib import asynccontextmanager

# --- NEW IMPORTS FOR SECURITY ---
from fastapi import FastAPI, HTTPException, Security, status, Depends
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv  # Needs: pip install python-dotenv

from pydantic import BaseModel

# Make project imports work regardless of how we run
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .embeddings import load_vector_store, create_vector_store
from .document_loader import load_chunks_from_json, load_documents, chunk_documents, save_chunks_to_json
from .llm_handler import LLMHandler
from .rag_pipeline import RAGPipeline

# --- SECURITY SETUP ---
# 1. Load the secrets from the .env file
load_dotenv() 

# 2. Get the key from the environment
SERVER_API_KEY = os.getenv("RAG_API_KEY")

# 3. Define the header key (User must send "X-API-Key" in headers)
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# 4. Security Logic Function
async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Checks if the header sent by the user matches the SERVER_API_KEY in our .env file.
    """
    if SERVER_API_KEY is None:
        # Safety net: If you forgot to set the key on GCP, don't let anyone in.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server Security Configuration Error: API Key missing on server."
        )
    
    if api_key_header != SERVER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key. Access Denied."
        )
    return api_key_header


# --- PYDANTIC MODELS ---
class ChatRequest(BaseModel):
    query: str

class SourceItem(BaseModel):
    source: str
    preview: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    model_used: str
    sources: List[SourceItem]
    elapsed_seconds: float


# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting RAG API with lifespan...")

    vector_db = load_vector_store()

    if vector_db is None:
        print("No FAISS index found, building from chunks/documents...")

        chunks = load_chunks_from_json()

        if not chunks:
            print("No chunks.json found, loading raw documents...")
            docs = load_documents()
            chunks = chunk_documents(docs)

            if chunks:
                save_chunks_to_json(chunks)
            else:
                raise RuntimeError("No chunks available. Add documents in `documents/`.")

        vector_db = create_vector_store(chunks)

    llm_handler = LLMHandler()
    rag = RAGPipeline(vector_db=vector_db, llm_handler=llm_handler)

    app.state.vector_db = vector_db
    app.state.llm_handler = llm_handler
    app.state.rag = rag

    print("✅ RAG API ready.")
    yield
    print("🛑 Shutting down RAG API...")


app = FastAPI(
    title="RAG Chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)

# --- ENDPOINTS ---

# 1. Health Check - LEFT OPEN (No Security)
# This allows GCP or Uptime monitors to ping your server without needing a key.
@app.get("/health")
def health_check():
    """
    Health and readiness info:
      - whether RAG pipeline is initialized
      - whether chunks.json and FAISS index exist
    """
    rag = getattr(app.state, "rag", None)
    vector_db = getattr(app.state, "vector_db", None)

    faiss_path = os.path.join(PROJECT_ROOT, "faiss_index")
    json_path = os.path.join(PROJECT_ROOT, "json", "chunks.json")

    return {
        "status": "ok" if rag is not None else "initializing",
        "vector_store_loaded": vector_db is not None,
        "faiss_index_present": os.path.isdir(faiss_path),
        "chunks_json_present": os.path.exists(json_path),
    }


# 2. Chat Endpoint - PROTECTED
# We add `dependencies=[Depends(get_api_key)]` to lock this door.
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(get_api_key)])
def chat(request: ChatRequest):
    """
    Main chat endpoint.
    """
    rag: RAGPipeline = getattr(app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized yet.")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        t0 = time.time()
        answer, docs, model_used = rag.query(query)
        elapsed = time.time() - t0

        sources: List[SourceItem] = []
        for d in docs or []:
            src = d.metadata.get("source", d.metadata.get("source_file", "unknown"))
            # Simple cleanup for preview
            preview = d.page_content[:240].replace("\n", " ")
            sources.append(SourceItem(source=src, preview=preview))

        return ChatResponse(
            question=query,
            answer=answer,
            model_used=model_used,
            sources=sources,
            elapsed_seconds=elapsed,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 3. Reload Endpoint - PROTECTED
@app.post("/admin/reload-index", dependencies=[Depends(get_api_key)])
def reload_index():
    """
    Rebuilds chunks and index.
    """
    rag: RAGPipeline = getattr(app.state, "rag", None)
    llm_handler: LLMHandler = getattr(app.state, "llm_handler", None)

    docs = load_documents()
    chunks = chunk_documents(docs)

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks found to rebuild index. Add documents first.")

    save_chunks_to_json(chunks)

    vector_db = create_vector_store(chunks)

    app.state.vector_db = vector_db
    app.state.rag = RAGPipeline(vector_db=vector_db, llm_handler=llm_handler)

    return {"status": "ok", "message": "Chunks and index rebuilt successfully."}