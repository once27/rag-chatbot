# src/llm_handler.py
import google.generativeai as genai
import src.config as config
import time

class LLMHandler:

    def __init__(self):
        self.setup_gemini()
    
    def setup_gemini(self):
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.gemini = genai.GenerativeModel(config.GEMINI_MODEL)
            print(f"Gemini Configured: {config.GEMINI_MODEL}")
        except Exception as e:
            print(f"Gemini Setup Failed: {e}")
            self.gemini = None

    def generate_response(self, prompt):
        """Generate response using Gemini only."""
        if not self.gemini:
             return "Error: Gemini model is not configured.", "SYSTEM_FAILURE"

        try:
            # simple direct call without threading/timeout complexity
            response = self.gemini.generate_content(prompt)
            return response.text.strip(), f"Gemini ({config.GEMINI_MODEL})"
            
        except Exception as e:
            return f"Error: Gemini failed to generate response. Details: {e}", "SYSTEM_FAILURE"