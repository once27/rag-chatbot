import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OLLAMA_MODEL = "llama3:8b"  # or "phi3"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DOCUMENTS_PATH = "./documents"

CHUNK_SIZE = 250        #500

CHUNK_OVERLAP = 100        #50

TOP_K_RESULTS = 3
