import google.generativeai as genai
from langchain_ollama import OllamaLLM as Ollama
import time
import config
import concurrent.futures

class LLMHandler:

    def __init__(self):
        self.setup_gemini()
        self.setup_ollama()

    
    def setup_gemini(self):
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.gemini = genai.GenerativeModel(config.GEMINI_MODEL)
            print(f"Gemini Configured: {config.GEMINI_MODEL}")
        except Exception as e:
            print(f"Gemini Setup Failed: {e}")
            self.gemini = None

    
    def setup_ollama(self):
        """Configure Local Ollama"""
        try:
            self.ollama = Ollama(model=config.OLLAMA_MODEL)
            print(f"Ollama Configured: {config.OLLAMA_MODEL}")
        except Exception as e:
            print(f"Ollama Setup Failed: {e}")
            self.ollama = None

    def _call_gemini(self, prompt):
        """Helper function to call Gemini (used inside the timer)"""
        response = self.gemini.generate_content(prompt)
        return response.text.strip()
    

    def generate_response(self, prompt):

        """Generate response with automatic fallback"""

        response_text = None
        model_name = None
        
        if self.gemini:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(self._call_gemini, prompt) # Submit the task
                
                response_text = future.result(timeout=10) # we wait for 10 seconds
                model_name = f"Gemini ({config.GEMINI_MODEL})"

            except concurrent.futures.TimeoutError:
                print("\nGemini Timed Out (took >10s). Switching to fallback...")
                # We do not wait for the stuck thread. 
                executor.shutdown(wait=False, cancel_futures=True) # wait=False makes sure that the control is shifted back to the main block
                
            except Exception as e:
                print(f"\nGemini Error: {e}")
                executor.shutdown(wait=False)

            finally:
                executor.shutdown(wait=False)

            if response_text: # If Gemini succeeded, return immediately
                return response_text, model_name

        if self.ollama: # This runs if Gemini failed OR timed out
            try:
                response = self.ollama.invoke(prompt)
                return response.strip(), f"Ollama ({config.OLLAMA_MODEL})"
            except Exception as e:
                return f"Error: Both models failed. Last error: {e}", "SYSTEM_FAILURE"
        
        return "Error: No models configured.", "SYSTEM_FAILURE"

# --- Test Block ---
# if __name__ == "__main__":
#     print("--- Initializing LLM Handler ---")
#     bot = LLMHandler()
    
#     test_prompt = "Explain in one sentence why the sky is blue."
    
#     print(f"\nTest Prompt: '{test_prompt}'")
    
#     start_time = time.time()
#     response, model_used = bot.generate_response(test_prompt)
#     end_time = time.time()
    
#     print(f"\nAnswer: {response}")
#     print(f"\nModel Used: {model_used}")
#     print(f"\nTime Taken: {end_time - start_time:.2f}s")
