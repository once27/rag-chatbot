import os
import sys
import time
from datetime import datetime
from typing import List

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
from embeddings import load_vector_store, create_vector_store
from document_loader import (
    load_chunks_from_json,
    load_documents,
    chunk_documents,
    save_chunks_to_json,
)
from llm_handler import LLMHandler
from rag_pipeline import RAGPipeline

#

# pipeline
def init_pipeline():
    """
    Initialize the RAG pipeline once per Streamlit session.
    Uses existing FAISS index / chunks.json when available.
    """
    if "rag_pipeline" in st.session_state and st.session_state["rag_pipeline"] is not None:
        return  # already initialized

    with st.spinner("Initializing RAG pipeline..."):
        # 1) Load or build vector store
        vector_db = load_vector_store()
        if vector_db is None:
            st.write("No FAISS index found. Building from existing chunks or documents...")

            chunks = load_chunks_from_json()
            if not chunks:
                st.write("No chunks.json found. Loading raw documents from documents/...")
                docs = load_documents()
                chunks = chunk_documents(docs)
                if chunks:
                    save_chunks_to_json(chunks)
                else:
                    st.error("No chunks could be created. Add documents to the documents/ folder.")
                    st.warning("No chunks available. Please upload documents using the sidebar and click 'Save & Rebuild index'.")
                    st.session_state["rag_pipeline"] = None
                    st.session_state["vector_db"] = None
                    return

            vector_db = create_vector_store(chunks)

        # 2) LLM handler
        llm_handler = LLMHandler()

        # 3) RAG pipeline
        rag = RAGPipeline(vector_db=vector_db, llm_handler=llm_handler)

        st.session_state["vector_db"] = vector_db
        st.session_state["rag_pipeline"] = rag


def rebuild_index_from_documents():
    """
    Rebuild both chunks.json and FAISS index from documents folder.
    Updates the RAG pipeline in session state.
    """
    documents_dir = config.DOCUMENTS_PATH

    with st.spinner("Rebuilding chunks and FAISS index from documents..."):
        if not os.path.isdir(documents_dir):
            st.error(f"Documents directory not found: {documents_dir}")
            return

        st.write(f"Using documents from: `{documents_dir}`")

        docs = load_documents()
        if not docs:
            st.error("No documents found to index. Please upload or add files to the documents folder.")
            return

        # Basic progress bar stages
        progress = st.progress(0, text="Chunking documents...")
        chunks = chunk_documents(docs)
        progress.progress(40, text="Saving chunks to json/chunks.json...")

        if not chunks:
            st.error("Chunking failed. Cannot rebuild index.")
            progress.empty()
            return

        save_chunks_to_json(chunks)
        progress.progress(70, text="Creating FAISS index from chunks...")

        vector_db = create_vector_store(chunks)

        # Update pipeline in session_state
        # Check if pipeline exists to reuse llm_handler, else create new
        if st.session_state.get("rag_pipeline"):
            llm_handler = st.session_state["rag_pipeline"].llm_handler
        else:
            llm_handler = LLMHandler()
            
        rag = RAGPipeline(vector_db=vector_db, llm_handler=llm_handler)
        st.session_state["vector_db"] = vector_db
        st.session_state["rag_pipeline"] = rag

        progress.progress(100, text="Index rebuild complete.")
        time.sleep(0.5)
        progress.empty()
        st.success("Chunks and FAISS index rebuilt successfully.")


def save_uploaded_files(uploaded_files: List):
    """
    Save uploaded files into the configured DOCUMENTS_PATH folder.
    """
    if not uploaded_files:
        return

    documents_dir = config.DOCUMENTS_PATH
    os.makedirs(documents_dir, exist_ok=True)

    for f in uploaded_files:
        dest_path = os.path.join(documents_dir, f.name)
        with open(dest_path, "wb") as out_f:
            out_f.write(f.getbuffer())


# UI

def main():
    st.set_page_config(page_title="RAG Chatbot (FAISS)", page_icon="🤖", layout="wide")

    st.title("🤖 RAG Chatbot (FAISS + Gemini/Ollama)")
    st.caption("Ask questions about your local documents. Backend: FAISS-based RAG pipeline.")

    # Sidebar
    with st.sidebar:
        st.header("Document Management")

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            help="Upload files to be indexed. They will be saved into the documents folder.",
            key="documents_uploader",
        )

        if uploaded_files:
            st.info(f"{len(uploaded_files)} file(s) selected. They will be saved and indexed if you click 'Save & Rebuild index'.")

        if st.button("Save & Rebuild index"):
            save_uploaded_files(uploaded_files)
            rebuild_index_from_documents()

        st.markdown("---")
        st.subheader("System status")

        faiss_path = os.path.join(PROJECT_ROOT, "faiss_index")
        json_path = os.path.join(PROJECT_ROOT, "json", "chunks.json")

        st.write(f"**FAISS index present:** `{os.path.isdir(faiss_path)}`")
        st.write(f"**chunks.json present:** `{os.path.exists(json_path)}`")

        # If pipeline already initialized, try to show vector count
        if "rag_pipeline" in st.session_state and st.session_state["rag_pipeline"] is not None:
            try:
                db = st.session_state["rag_pipeline"].vector_db
                ntotal = getattr(getattr(db, "index", None), "ntotal", None)
                if ntotal is not None:
                    st.write(f"**Indexed vectors:** `{ntotal}`")
            except Exception as e:
                st.write(f"Status check error: {e}")
        documents_dir = config.DOCUMENTS_PATH
        st.write(f"**Documents path:** `{documents_dir}`")

    # Initialize Pipeline
    init_pipeline()
    
    # Retrieve pipeline from session state (safely)
    rag = st.session_state.get("rag_pipeline", None)

    # If no pipeline, tell user and STOP here.
    if rag is None:
        st.info(
            "RAG pipeline is not initialized yet. "
            "Please upload documents in the sidebar and click 'Save & Rebuild index'."
        )
        return

    st.markdown("### Ask a Question")

    query = st.text_input("Enter your question about the documents:")
    ask_button = st.button("Ask")

    if ask_button and query.strip():
        question = query.strip()

        with st.spinner("Running RAG pipeline..."):
            t0 = time.time()
            try:
                answer, docs, model_used = rag.query(question)
            except Exception as e:
                st.error(f"Error during RAG query: {e}")
                return
            elapsed = time.time() - t0
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.markdown("### Response")
        st.markdown(f"**Question:** {question}")
        st.markdown(f"**Answer:**\n\n{answer}")
        st.markdown(f"**Model used:** `{model_used}`")
        st.markdown(f"**Time taken:** `{elapsed:.2f} seconds`")
        st.markdown(f"**Generated at:** `{timestamp}`")

        # Show sources
        if docs:
            st.markdown("---")
            st.markdown("#### 📚 Sources")
            for i, d in enumerate(docs, start=1):
                src = d.metadata.get("source", d.metadata.get("source_file", "unknown"))
                preview = d.page_content[:250].replace("\n", " ")
                st.markdown(f"**[{i}] {src}**")
                st.markdown(f"> {preview}...")
        else:
            st.info("No relevant source chunks were found for this question.")


if __name__ == "__main__":
    main()