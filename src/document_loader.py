import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
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

def save_chunks_to_json(chunks, filename="chunks.json"):
    """
    Save chunks to a JSON file in the 'json' directory of the project root.
    """

    project_root = os.getcwd() # get the current directory
    raw_dir = os.path.join(project_root, "json") 
    
    if not os.path.exists(raw_dir): # Create the directory if it doesn't exist
        os.makedirs(raw_dir)
        print(f"Created directory: {raw_dir}")

    output_path = os.path.join(raw_dir, filename) # File path

    serializable = [] # Data for JSON serialization
    for chunk in chunks:
        serializable.append({
            "content": chunk.page_content,
            "metadata": chunk.metadata
        })

    try: # Saving to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(chunks)} chunks into: {output_path}")
    except Exception as e:
        print(f" Failed to save JSON: {e}")

def load_chunks_from_json(filename="chunks.json"):
    """
    Load chunks from a JSON file and convert them back to LangChain Documents.
    """
    project_root = os.getcwd()
    file_path = os.path.join(project_root, "raw", filename)

    if not os.path.exists(file_path):
        print(f"JSON file not found: {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert list of dicts back to list of Document objects
        documents = []
        for item in data:
            doc = Document(
                page_content=item["content"],
                metadata=item["metadata"]
            )
            documents.append(doc)
            
        print(f"Loaded {len(documents)} chunks from JSON cache.")
        return documents
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return None

# testing
if __name__ == "__main__":
    print(f"Scanning folder: {DOCUMENTS_PATH}...")
    
    loaded_docs = load_documents() # load the document
    
    if loaded_docs:
        print(f"Found and loaded {len(loaded_docs)} document(s).")
        
        chunks = chunk_documents(loaded_docs) # chunk the documents
        
        print(f"Successfully split into {len(chunks)} text chunks.")
        
        # print("Preview of Chunk ") 
        # print(chunks[0].page_content[:100] + "...") 
        # print(f"\n[Source: {chunks[0].metadata.get('source', 'unknown')}]")
        # print()
        save_chunks_to_json(chunks)
        
    else:
        print("  No documents found! Please check:")

