# 📚 RAG Chatbot – Local & Cloud Hybrid AI Assistant

A fully offline-capable **Retrieval-Augmented Generation (RAG)** chatbot built using:

- **FAISS** for vector search  
- **HuggingFaceEmbeddings** for semantic indexing  
- **Gemini** (primary LLM)  
- **Ollama** (fallback local LLM)  
- **Local JSON chunk cache**  
- **Terminal Chat UI**  

This project lets you chat with your **own documents** using an AI assistant that is both powerful and privacy-friendly.

---

## 🚀 Features

### 🔍 **Retrieval-Augmented Generation (RAG)**  
- Finds relevant text chunks from your documents  
- Builds a context-rich prompt  
- Sends it to an LLM with strict grounding rules  

### 🤖 **Hybrid Model Support**
- **Primary**: Google Gemini (fast, high quality)  
- **Fallback**: Local Ollama model for offline responses  

### 📦 **Smart Caching**
- Document chunks stored as `json/chunks.json`  
- FAISS vectors stored in `faiss_index/`  
- Prevents re-chunking or re-indexing  

### 📡 **Offline Mode**
Works fully offline **after one-time setup**:
- Embedding model cached locally  
- FAISS index on disk  
- Ollama model installed  

### 💬 **Terminal Chat UI**
Commands included:
```
exit | help | clear | status | sources | history | save_history
```

### 📝 **Logging & History**
- Every query/response logged to `logs/`  
- Full session history export available  

---

## 📁 Project Structure

```
rag-chatbot/
│
├── src/
│   ├── embeddings.py
│   ├── llm_handler.py
│   ├── document_loader.py
│   ├── rag_pipeline.py
│   ├── chatbot.py
│   └── config.py
│
├── documents/            # Put PDFs / DOCX / TXT here
├── json/                 # Auto-generated chunks.json
├── faiss_index/          # Auto-generated vector index
├── models/               # Optional local embedding model
├── logs/                 # Session logs
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repo
```
git clone <repo url>
cd rag-chatbot
```

### 2. Create virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Install Ollama (if using local LLM)
https://ollama.ai

Example:
```
ollama pull llama3
```

### 5. Configure `.env` file
Create `.env`:
```
GEMINI_API_KEY=YOUR_API_KEY
```

---

## 📘 How RAG Pipeline Works

### **Step 1 — Ingestion**
- Documents are loaded from `documents/`
- Chunked using `RecursiveCharacterTextSplitter`
- Saved to `json/chunks.json`

### **Step 2 — Embedding & Indexing**
- Embeddings generated using HuggingFace model (locally cached)
- Indexed using FAISS
- Stored in `faiss_index/`

### **Step 3 — Query Time**
1. User enters a query  
2. Query is converted into an embedding  
3. FAISS retrieves matching chunks  
4. RAG prompt is built  
5. LLM handler:
   - tries Gemini  
   - falls back to Ollama on fail  
6. Answer + sources returned  

---

## 🧠 Offline Mode Explained

After the first successful online run:

- `json/chunks.json` exists  
- `faiss_index/` exists  
- embedding model is cached  


You can now use:
✔ FAISS offline  
✔ Embeddings offline  
✔ Ollama offline  
✔ Full RAG pipeline offline  

Only Gemini requires the internet — and the system automatically falls back.

---

## 💻 Running the Chatbot

### Method 1 — Recommended
```
python -m src.chatbot
```

### Method 2
```
python src/chatbot.py
```

---

## 🧩 Commands in Terminal UI

| Command | Description |
|---------|-------------|
| **help** | Show all commands |
| **status** | Display model + index + chunk status |
| **sources** | View source chunks used for last answer |
| **history** | Print last 10 Q/A items |
| **save_history** | Save entire session history to logs/ |
| **clear** | Clear screen |
| **exit** | Quit the chatbot |


## ❓ Common Issues

### ❌ "Failed to resolve huggingface.co"  
Cause: embedding model not cached and offline.

Fix:
```
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2').save('./models/all-MiniLM-L6-v2')
PY
```

### ❌ "No chunks found"  
Put documents into `documents/` and rebuild:
```
python -m src.embeddings
```

---

## 🏁 You're Done!

Your system is now a fully working semantic search + AI assistant that understands and answers questions based on your personal documents — locally or online, with reliability and fallbacks.
