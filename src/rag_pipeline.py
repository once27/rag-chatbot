from typing import List, Tuple
import os
import sys

try:
    from .embeddings import load_vector_store, create_vector_store, retrieve_relevant_docs
    from .document_loader import load_chunks_from_json, load_documents, chunk_documents, save_chunks_to_json
    from .llm_handler import LLMHandler
    import config as config # to run either as an individual module or externally from an api
except ImportError:
    # Fallback path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(current_dir)
    sys.path.append(parent_dir)

    from embeddings import load_vector_store, create_vector_store, retrieve_relevant_docs
    from document_loader import load_chunks_from_json, load_documents, chunk_documents, save_chunks_to_json
    import config 
    from llm_handler import LLMHandler


class RAGPipeline:
    """
    - Retrieve relevant chunks from FAISS
    - Building a RAG-style prompt
    - Use LLMHandler (Gemini + Ollama fallback) to answer
    """

    def __init__(self, vector_db, llm_handler: LLMHandler):
        if vector_db is None:
            raise ValueError("RAGPipeline requires a non-null vector_db.")
        self.vector_db = vector_db
        self.llm_handler = llm_handler

    def build_context(self, docs: List) -> str:
        """
        Turn retrieved docs into a context string for the prompt.
        """
        if not docs:
            return "No relevant context found in the documents."

        blocks = []
        for i, doc in enumerate(docs, start=1):
            # Clean up source path for display
            src = doc.metadata.get("source", "unknown").split(os.sep)[-1]
            blocks.append(f"[Source {i}: {src}]\n{doc.page_content}")

        return "\n\n".join(blocks)

    def build_prompt(self, question: str, docs: List) -> str:
        context_text = self.build_context(docs)

        prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the context below.

        Context:
        {context_text}

        Question:
        {question}

        Instructions:
        - Use only the information from the context.
        - If the answer is not in the context, say: "I don't have enough information from the documents to answer that."
        - Be concise and clear.
        - If multiple sources mention similar information, synthesize them.

        Answer:"""
        return prompt

    def query(self, user_query: str) -> Tuple[str, List, str]:
        """
        1. Retrieve relevant docs from FAISS
        2. Build RAG prompt
        3. Ask LLM handler
        """
        user_query = user_query.strip()
        if not user_query:
            raise ValueError("Query cannot be empty.")

        # Use your existing retrieval with relevance threshold
        docs = retrieve_relevant_docs(
            self.vector_db,
            user_query,
            k=config.TOP_K_RESULTS,
            score_threshold=0.0, # ) because our model give results based on cosine similarity and we want value between 0 and 1
        )

        prompt = self.build_prompt(user_query, docs)
        answer_text, model_used = self.llm_handler.generate_response(prompt)
        return answer_text, docs, model_used


# test
if __name__ == "__main__":

    # Priority: 1. Load Existing FAISS -> 2. Load JSON Cache -> 3. Load Raw Docs
    
    vector_db = load_vector_store()
    
    if not vector_db:
        print("No existing FAISS index found. Checking for JSON cache...")

        chunks = load_chunks_from_json()

        if not chunks:
            print("No JSON cache found. Reading source documents from folder...")
            docs = load_documents()
            chunks = chunk_documents(docs)
        
            if chunks: # if no json cache found then build it
                save_chunks_to_json(chunks) 

        if chunks:
            # This creates the index AND saves it to disk (inside embeddings.py logic)
            vector_db = create_vector_store(chunks)
        else:
            print("Error: No documents found to index.")
            sys.exit(1)

    # 2. Initialize LLM 
    print("\nInitializing LLM Handler...")
    llm_handler = LLMHandler()

    # 3. Start Pipeline
    rag = RAGPipeline(vector_db=vector_db, llm_handler=llm_handler)

    print("\nSystem Ready! Type 'exit' to quit.")

    try:
        while True:
            user_q = input("\nYour Question: ").strip()
            
            if not user_q:
                continue
                
            if user_q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            print("   Thinking...", end="\r")
            
            # Run Query
            answer, sources, model = rag.query(user_q)

            # Display Results
            print("\n")
            print(f"Assistant ({model}):")
            print(answer)

            if sources:
                print(f"Context used ({len(sources)} chunks):")
                for i, doc in enumerate(sources, start=1):
                    src = doc.metadata.get('source', 'unknown').split(os.sep)[-1]
                    page = doc.metadata.get('page', 'N/A')
                    print(f"{src} ")#(Page {page})")
            else:
                print(" No relevant documents found in the database.")
            print("\n")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")