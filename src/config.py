import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OLLAMA_MODEL = "llama3"  # or "phi3"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DOCUMENTS_PATH = "./documents"

CHUNK_SIZE = 500

CHUNK_OVERLAP = 50

TOP_K_RESULTS = 3
