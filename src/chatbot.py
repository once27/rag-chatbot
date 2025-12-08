# src/chatbot.py
"""
Terminal Chatbot wrapper around your RAG pipeline.
Features:
 - interactive prompt loop
 - commands: exit, help, clear, status, sources, history, save_history
 - shows which model answered and response time
 - logs queries & responses to ./logs/queries.log
 - uses colorama for nicer output
"""

import os
import sys
import time
import json
from typing import List

# make project imports work reliably
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from colorama import init as colorama_init, Fore, Style
from rag_pipeline import RAGPipeline
from llm_handler import LLMHandler
from embeddings import load_vector_store, create_vector_store
from document_loader import load_chunks_from_json, load_documents, chunk_documents,save_chunks_to_json

colorama_init(autoreset=True)

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "queries.log")


class TerminalChatbot:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag = rag_pipeline
        self.history: List[dict] = []

    def print_welcome(self):
        print(f"\n{Fore.CYAN}{'='*64}")
        print(f"  🤖  RAG-Based Document Chatbot")
        print(f"  Project: rag-chatbot — Terminal Interface")
        print(f"{'='*64}{Style.RESET_ALL}\n")
        print("Commands: 'exit' | 'clear' | 'help' | 'status' | 'sources' | 'history' | 'save_history'\n")

    def show_help(self):
        print(f"\n{Fore.YELLOW}Available Commands:{Style.RESET_ALL}")
        print(" exit         - Quit the chatbot")
        print(" clear        - Clear conversation history from screen (not saved history)")
        print(" help         - Show this help message")
        print(" status       - Show system status (models, index presence, doc counts)")
        print(" sources      - Show the sources used by the last answer")
        print(" history      - Print recent Q/A pairs in this session")
        print(" save_history - Save session history to JSON in logs/")
        print()

    def clear_screen(self):
        # cross-platform clear
        os.system("cls" if os.name == "nt" else "clear")

    def show_status(self):
        # Basic status information
        print(f"\n{Fore.MAGENTA}System status:{Style.RESET_ALL}")
        # model info from llm handler if possible
        try:
            llm = self.rag.llm_handler
            gem = getattr(llm, "gemini", None) # if gemini fails prevents to program from crashing by setting it to null
            oll = getattr(llm, "ollama", None)
            gem_name = getattr(gem, "model_name", "gemini (unknown)") if gem else "Gemini (not init)"
            oll_name = getattr(oll, "model", "ollama (unknown)") if oll else "Ollama (not init)"
            print(f"  Primary LLM: {gem_name}")
            print(f"  Fallback LLM: {oll_name}")
        except Exception:
            print("  LLM handler status: unavailable")

        # vector store / chunks
        faiss_path = os.path.join(PROJECT_ROOT, "faiss_index")
        json_path = os.path.join(PROJECT_ROOT, "json", "chunks.json")
        print(f"  FAISS index present: {os.path.isdir(faiss_path)}")
        print(f"  Chunks JSON present: {os.path.exists(json_path)}")
        # approximate counts (if available)
        try:
            # Not all vectorstores expose ntotal; guard it
            db = self.rag.vector_store
            ntotal = getattr(getattr(db, "index", None), "ntotal", None)
            print(f"  Indexed vectors: {ntotal if ntotal is not None else 'unknown'}")
        except Exception:
            pass
        print()

    def show_sources(self, last_sources):
        if not last_sources:
            print(f"\n{Fore.YELLOW}No sources available for last answer.{Style.RESET_ALL}\n")
            return
        print(f"\n{Fore.MAGENTA}Sources used in last answer:{Style.RESET_ALL}")
        for i, d in enumerate(last_sources, start=1):
            src = d.metadata.get("source", d.metadata.get("source_file", "unknown"))
            preview = d.page_content[:240].replace("\n", " ")
            print(f"  [{i}] {src} — \"{preview}...\"")
        print()

    def save_history(self):
        if not self.history:
            print("No history to save.")
            return
        filename = os.path.join(LOG_DIR, f"history_{int(time.time())}.json")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            print(f"Saved history to {filename}")
        except Exception as e:
            print(f"Failed to save history: {e}")

    def log_interaction(self, q, a, model, sources):
        record = {
            "timestamp": int(time.time()),
            "query": q,
            "answer": a,
            "model": model,
            "sources": [d.metadata.get("source", "unknown") for d in sources] if sources else [],
        }
        self.history.append(record)
        # append to a simple log file
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def run(self):
        self.print_welcome()
        last_sources = None

        while True:
            try:
                user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()

                if not user_input:
                    continue

                cmd = user_input.lower().strip()

                # Commands
                if cmd in {"exit", "quit"}:
                    print(f"\n{Fore.CYAN}👋 Goodbye!{Style.RESET_ALL}\n")
                    break

                if cmd == "help":
                    self.show_help()
                    continue

                if cmd == "clear":
                    self.clear_screen()
                    continue

                if cmd == "status":
                    self.show_status()
                    continue

                if cmd == "sources":
                    self.show_sources(last_sources)
                    continue

                if cmd == "history":
                    print(json.dumps(self.history[-10:], indent=2, ensure_ascii=False))
                    continue

                if cmd == "save_history":
                    self.save_history()
                    continue

                # Normal query: run RAG pipeline
                print(f"{Fore.YELLOW}🔍 Searching documents...{Style.RESET_ALL}", end="\r")
                t0 = time.time()
                answer, sources, model = self.rag.query(user_input)
                duration = time.time() - t0

                # show result
                print()  # newline after the searching line
                print(f"{Fore.BLUE}🤖 Assistant ({model}) [{duration:.2f}s]:{Style.RESET_ALL}\n{answer}\n")

                if sources:
                    print(f"{Fore.MAGENTA}📚 Sources: {len(sources)} document chunk(s){Style.RESET_ALL}")
                    # show short listing (source names)
                    for i, d in enumerate(sources, start=1):
                        src = d.metadata.get("source", d.metadata.get("source_file", "unknown"))
                        print(f"  - [{i}] {src}")
                    print()

                # log & keep last sources
                self.log_interaction(user_input, answer, model, sources or [])
                last_sources = sources

            except KeyboardInterrupt:
                print(f"\n\n{Fore.CYAN}👋 Goodbye!{Style.RESET_ALL}\n")
                break
            except Exception as e:
                print(f"\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}\n")


# Helper to initialize components and start chatbot
def main():
# 1) Load or build vector store
    vector_db = load_vector_store()
    
    if vector_db is None:
        print(f"{Fore.YELLOW}🐢 No Index found. Checking cache...{Style.RESET_ALL}")
        chunks = load_chunks_from_json()
        
        if not chunks:
            print(f"{Fore.YELLOW}📂 Reading raw documents...{Style.RESET_ALL}")
            docs = load_documents()
            chunks = chunk_documents(docs)
            
            if chunks:
                save_chunks_to_json(chunks) 

        if not chunks:
            raise SystemExit("No chunks found. Place documents in the documents/ folder.")
        
        vector_db = create_vector_store(chunks)

    # 2) LLM handler and RAG pipeline
    llm = LLMHandler()
    from rag_pipeline import RAGPipeline  # local import to avoid cycle at module import time
    rag = RAGPipeline(vector_db=vector_db, llm_handler=llm)

    # 3) start interactive chat
    bot = TerminalChatbot(rag)
    bot.run()


if __name__ == "__main__":
    main()
