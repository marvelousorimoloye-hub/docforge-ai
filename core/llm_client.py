# core/llm_client.py
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_cohere import ChatCohere
from core.config import GROQ_API_KEY, GEMINI_API_KEY, COHERE_API_KEY

def get_llm(model_type: str = "groq"):
    """
    Optimized LLM loader for 8GB RAM machine.
    Prioritizes speed and stability.
    """
    
    if model_type == "local" or model_type == "ollama":
        print("Loading Ollama (qwen3:4b)... This may take a while on first load.")
        return ChatOllama(
            model="qwen3:4b",
            temperature=0.3,
            num_ctx=3584,           # Reduced from 8192 to save RAM
            num_predict=1024,       # Limit max output tokens
            top_k=40,
            top_p=0.9,
        )

    if model_type == "groq":
        try:
            groq_llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=GROQ_API_KEY,
                temperature=0.3,
                max_tokens=2048,
                request_timeout=120.0
            )
            # Light fallback only if Cohere key exists
            if COHERE_API_KEY:
                cohere_llm = ChatCohere(
                    model="command-r-plus-08-2024",
                    cohere_api_key=COHERE_API_KEY,
                    temperature=0.3
                )
                return groq_llm.with_fallbacks([cohere_llm])
            return groq_llm
        except:
            print("Groq failed. Falling back to Ollama.")
            return get_llm("local")

    if model_type == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=GEMINI_API_KEY,
            temperature=0.3,
        )

    # Default fallback
    print("Using default: Ollama (qwen3:4b)")
    return get_llm("local")