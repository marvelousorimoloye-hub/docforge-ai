#config.py
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
VOYAGE_API_KEY= os.getenv("VOYAGE_API_KEY")
OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
COHERE_API_KEY= os.getenv("COHERE_API_KEY")
DATA_DIR = "data"
UPLOAD_DIR = f"{DATA_DIR}/uploads"
VECTOR_STORE_DIR = f"{DATA_DIR}/vector_store"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)