import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import src.config as config
FAISS_PATH = os.path.join(os.getcwd(), "faiss_index")
LOCAL_MODEL_PATH = os.path.join(os.getcwd(), "models", "all-MiniLM-L6-v2")
def get_embedding_model():
    if os.path.isdir(LOCAL_MODEL_PATH):
        print(f"Using local embedding model at: {LOCAL_MODEL_PATH}")
        return HuggingFaceEmbeddings(model_name=LOCAL_MODEL_PATH)

    # otherwise use the HF hub name (may attempt network)
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

def create_vector_store(chunks):
    """Create a new vector store from chunks and SAVE it to disk."""
    if not chunks:
        return None

    print(f"Generating embeddings using: {config.EMBEDDING_MODEL}...")
    embeddings = get_embedding_model()

    try:
        # Create the store
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # SAVE to disk 
        vector_store.save_local(FAISS_PATH)
        print(f"Vector store saved to: {FAISS_PATH}")
        print(f"Index contains {vector_store.index.ntotal} items.")
        return vector_store
    except Exception as e:
        print(f"Failed to create/save vector store: {e}")
        return None
    

def load_vector_store():
    """Load an existing vector store from disk."""
    if not os.path.exists(FAISS_PATH):
        print("No local vector store found. Please create one first.")
        return None
    
    print("Loading vector store from disk...")
    embeddings = get_embedding_model()
    try:
        # Allow dangerous deserialization is required for local pickle files
        vector_store = FAISS.load_local(
            FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True 
        )
        print("Vector store loaded successfully.")
        return vector_store
    except Exception as e:
        print(f"Failed to load vector store: {e}")
        return None
    
def retrieve_relevant_docs(vector_store, query, k=config.TOP_K_RESULTS, score_threshold=0.5):
    """Retrieve top-k relevant documents with filtering."""
    if not vector_store:
        return []
        
    print(f"Searching for: '{query}'")
    docs_and_scores = vector_store.similarity_search_with_relevance_scores(query, k=k)
    
    filtered_docs = []
    for doc, score in docs_and_scores:
        if score >= score_threshold:
            filtered_docs.append(doc)
            
    return filtered_docs


# Testing
# if __name__ == "__main__":
#     # Import loader logic
#     try:
#         from document_loader import load_documents, chunk_documents, load_chunks_from_json
#     except ImportError:
#         import sys
#         sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#         from document_loader import load_documents, chunk_documents, load_chunks_from_json

#     print("--- Starting Embeddings Pipeline ---")
    
#     if os.path.exists(FAISS_PATH): # Check for a saved Vector Store
#         print("Found existing FAISS index. Loading directly...")
#         vector_db = load_vector_store()
#     else:
#         print("No existing index found. Starting fresh build...")
    
#         chunks = load_chunks_from_json() # loading chunks from JSON first
        
#         if not chunks: # If no JSON, Load from folder
#             print("\nNo JSON chunks found, reading source documents...")
#             docs = load_documents()
#             chunks = chunk_documents(docs)
        
#         # Build and Save
#         if chunks:
#             vector_db = create_vector_store(chunks)
#         else:
#             vector_db = None

# try:
#     # Make sure vector DB exists
#     if not vector_db:
#         print("No vector DB available, exiting.")
#     else:
#         # Ask user for query
#         test_query = input("\nEnter a test query: ").strip()

#         if not test_query:
#             print("Empty query, exiting.")
#         else:
#             # Perform search
#             results = retrieve_relevant_docs(vector_db, test_query, score_threshold=0.3)

#             print(f"\nFound {len(results)} result(s).")

#             if not results:
#                 print("No results found.")
#             else:
#                 # Print results
#                 for i, d in enumerate(results, start=1):
#                     print(f"\n--- Result {i} ---")
#                     print(d.page_content[:150])
#                     print("Metadata:", d.metadata)

# except KeyboardInterrupt:
#     print("\n\nUser cancelled operation.")


