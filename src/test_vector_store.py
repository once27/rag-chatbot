from document_loader import load_documents, chunk_documents
from embeddings import create_vector_store, retrieve_relevant_docs
from config import DOCUMENTS_PATH, TOP_K_RESULTS


def main():
    print(f"Loading documents from: {DOCUMENTS_PATH}")
    docs = load_documents()
    print(f"Loaded {len(docs)} document objects")

    if not docs:
        print("No documents to index. Add some files in the documents/ folder.")
        return

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    if not chunks:
        print("No chunks created. Check your document's content.")
        return

    print("Creating FAISS vector store")
    vector_store = create_vector_store(chunks)
    print("Vector store ready.")

    query = input("\nEnter a test query: ").strip()
    if not query:
        print("Empty query, exiting.")
        return

    print(f"\n Retrieving top {TOP_K_RESULTS} results.")
    results = retrieve_relevant_docs(vector_store, query)

    if not results:
        print("No results found.")
        return

    for i, d in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(d.page_content[:150])
        print("Metadata:", d.metadata)


if __name__ == "__main__":
    main()
