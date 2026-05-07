# core/memory_manager.py
from typing import List, Dict, Optional
from datetime import datetime
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from pinecone import Pinecone, ServerlessSpec
from core.config import PINECONE_API_KEY

embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
pc = Pinecone(api_key=PINECONE_API_KEY)

MEMORY_INDEX_NAME = "docforge-memory"   # Separate index for past analyses


class MemoryManager:
    """Semantic Long-term Memory using Pinecone (Separate Index)"""

    def __init__(self):
        self._ensure_index_exists()
        self.vector_store = PineconeVectorStore(
            index_name=MEMORY_INDEX_NAME,
            embedding=embed_model,
            namespace="analyses"          # Namespace inside the memory index
        )

    def _ensure_index_exists(self):
        """Create memory index if it doesn't exist"""
        if MEMORY_INDEX_NAME not in pc.list_indexes().names():
            print(f"🆕 Creating Pinecone Memory Index: {MEMORY_INDEX_NAME}")
            pc.create_index(
                name=MEMORY_INDEX_NAME,
                dimension=384,                    # bge-small-en-v1.5
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

    def save_analysis(self, query: str, report: str, feedback: str = ""):
        """Save a completed analysis to semantic memory"""
        doc = Document(
            page_content=report,
            metadata={
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "feedback": feedback,
                "type": "past_analysis",
                "source": "docforge_agent"
            }
        )
        
        self.vector_store.add_documents([doc])
        print(f"💾 Saved to Long-term Memory: {query[:60]}...")

    def retrieve_relevant_memories(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve semantically similar past analyses"""
        try:
            docs = self.vector_store.similarity_search(
                query, 
                k=k,
                filter={"type": "past_analysis"}   # Optional: metadata filtering
            )
            print(f"🧠 Retrieved {len(docs)} relevant past analyses from memory")
            return docs
        except Exception as e:
            print(f"⚠️ Memory retrieval failed: {e}")
            return []

    def get_all_memories(self, limit: int = 10) -> List[Dict]:
        """Optional: Get recent memories (for Past Analyses tab)"""
        # This is a simple implementation. For full history, you might want a separate DB.
        docs = self.vector_store.similarity_search(
            "any analysis", k=limit, filter={"type": "past_analysis"}
        )
        return [doc.metadata for doc in docs]