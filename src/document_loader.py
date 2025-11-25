import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

try:
    from .config import DOCUMENTS_PATH, CHUNK_SIZE, CHUNK_OVERLAP
except ImportError:

    from config import DOCUMENTS_PATH, CHUNK_SIZE, CHUNK_OVERLAP



def load_documents(folder_path: str = DOCUMENTS_PATH):

    """Load all supported documents from folder."""

    documents: List = []

    if not os.path.isdir(folder_path):
        print(f" Documents folder not found: {folder_path}")
        return documents

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue  # skip subfolders etc.
        lower = filename.lower()
        loader = None
        if lower.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif lower.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
        elif lower.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        try:

            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

    return documents



def chunk_documents(documents,chunk_size: int = CHUNK_SIZE,overlap: int = CHUNK_OVERLAP):

    """Split documents into chunks"""

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=chunk_size,

        chunk_overlap=overlap

    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# testing
if __name__ == "__main__":
    print(f"Scanning folder: {DOCUMENTS_PATH}...")
    
    loaded_docs = load_documents() # load the document
    
    if loaded_docs:
        print(f"Found and loaded {len(loaded_docs)} document(s).")
        
        chunks = chunk_documents(loaded_docs) # chunk the documents
        
        print(f"Successfully split into {len(chunks)} text chunks.")
        
        print("Preview of Chunk ") 
        print(chunks[0].page_content[:100] + "...") 
        print(f"\n[Source: {chunks[0].metadata.get('source', 'unknown')}]")
        print()
        
    else:
        print("  No documents found! Please check:")