import google.generativeai as genai
from langchain_ollama import OllamaLLM as Ollama
import src.config
import sys

def test_gemini():
    """Tests the connection to Google's Gemini API"""
    print(f"Testing Gemini API ({src.config.GEMINI_API_KEY[:5]}...)...")
    try:
        genai.configure(api_key=src.config.GEMINI_API_KEY) # get api key from file

        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Send a simple test message
        response = model.generate_content("Say 'Gemini is online' if you can hear me.")
        
        # Check if we got a valid text response
        if response.text:
            print(f"Gemini Success: {response.text.strip()}")
            return True
    except Exception as e:
        print(f"Gemini Failed: {e}")
        return False

def test_ollama():
    """Tests the connection to the local Ollama instance"""
    print(f"Testing Local Ollama ({src.config.OLLAMA_MODEL})...")
    try:
        llm = Ollama(model=src.config.OLLAMA_MODEL) # info about local model from file
        
        # Send a test message
        response = llm.invoke("Say 'Ollama is online' if you can hear me.")
        
        if response:
            print(f"Ollama Success: {response.strip()}")
            return True
    except Exception as e:
        print(f" Ollama Failed: {e}")
        return False

if __name__ == "__main__":
    print("--- 🔌 Connection Diagnostics ---")
    
    gemini_status = test_gemini()
    ollama_status = test_ollama()
    
    print("\n" + "="*30)
    if gemini_status and ollama_status:
        print("SUCCESS: Both LLMs are reachable! You are ready for Day 2.")
    elif gemini_status:
        print("PARTIAL: Gemini is working, but Ollama failed.")
    elif ollama_status:
        print("PARTIAL: Ollama is working, but Gemini failed.")
    else:
        print("FAILURE: Neither model is responding. Check your API key and Ollama installation.")
    print("="*30)