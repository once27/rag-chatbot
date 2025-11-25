from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import config

def create_vector_store(chunks):

    """Create and populate vector store"""

    embeddings = HuggingFaceEmbeddings(

        model_name=config.EMBEDDING_MODEL

    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

def retrieve_relevant_docs(vector_store, query, k=1):

    """Retrieve top-k relevant documents"""

    docs = vector_store.similarity_search(query, k=k)

    return docs
