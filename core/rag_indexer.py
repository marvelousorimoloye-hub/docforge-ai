# core/rag_indexer.py
import socket
import time
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from core.config import PINECONE_API_KEY

socket.setdefaulttimeout(30)

# Constants
INDEX_NAME = "docforge-hybrid"
BM25_PARAMS_PATH = "bm25_params.json"

# Initialize Models
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
pc = Pinecone(api_key=PINECONE_API_KEY)

def sanitize_metadata(metadata: dict) -> dict:
    """Ensures metadata values are within limits and consistent."""
    return {
        "filename": str(metadata.get("filename", "unknown"))[:300],
        "chunk_type": metadata.get("chunk_type", "child"),
        "parent_id": metadata.get("parent_id"),
        "level": metadata.get("level", "child"),
        "source": "uploaded"
    }

def get_or_create_index(documents: Optional[List[Document]] = None):
    """Hierarchical Chunking + Hybrid Retriever with Persistent BM25 State"""
    
    # 1. Handle Pinecone Index Creation
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"🆕 Creating Pinecone Index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(8)

    index = pc.Index(INDEX_NAME)
    
    # 2. Initialize BM25 Encoder with Persistence
    bm25_encoder = BM25Encoder()
    
    if os.path.exists(BM25_PARAMS_PATH):
        print(f"📖 Loading existing BM25 parameters from {BM25_PARAMS_PATH}")
        bm25_encoder = BM25Encoder().load(BM25_PARAMS_PATH)
    else:
        print("⚠️ No BM25 params found. Will initialize a new one.")

    # 3. Process Documents if Provided
    if documents:
        print(f"📥 Processing {len(documents)} documents with Hierarchical Chunking...")

        # FIT and SAVE BM25 Encoder
        # We fit on the raw documents to capture the full context vocabulary
        raw_texts = [doc.page_content for doc in documents]
        bm25_encoder.fit(raw_texts)
        bm25_encoder.dump(BM25_PARAMS_PATH)
        print(f"💾 BM25 parameters updated and saved to {BM25_PARAMS_PATH}")

        all_chunks = []
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3800, chunk_overlap=350)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=150)

        for doc in documents:
            filename = doc.metadata.get("filename", "unknown")

            parent_chunks = parent_splitter.split_documents([doc])
            for p_idx, parent in enumerate(parent_chunks):
                parent_id = f"{filename}_p_{p_idx}"
                parent.metadata = sanitize_metadata({
                    "chunk_type": "parent",
                    "parent_id": parent_id,
                    "filename": filename,
                    "level": "parent",
                    "text": parent.page_content[:500]   # Store snippet for hybrid search result
                })
                all_chunks.append(parent)

                child_chunks = child_splitter.split_documents([parent])
                for c_idx, child in enumerate(child_chunks):
                    child.metadata = sanitize_metadata({
                        "chunk_type": "child",
                        "parent_id": parent_id,
                        "filename": filename,
                        "level": "child",
                        "text": child.page_content[:500]
                    })
                    all_chunks.append(child)

        print(f"   Created {len(all_chunks)} chunks. Indexing...")

        vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embed_model)
        
        # Batch upload to Pinecone
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            try:
                vector_store.add_documents(batch)
                print(f"   ✅ Batch {i//batch_size + 1} indexed")
                time.sleep(0.7)
            except Exception as e:
                print(f"   Batch error: {e}")
                
    elif not os.path.exists(BM25_PARAMS_PATH):
        # Emergency initialization to prevent ValueError if index is empty and no file exists
        print("⚡ Performing emergency BM25 fit to prevent encoding errors.")
        bm25_encoder.fit(["initialization dummy text"])

    # 4. Return Hybrid Retriever
    # Using 'text' as text_key because you stored snippets there in metadata
    retriever = PineconeHybridSearchRetriever(
        embeddings=embed_model,
        sparse_encoder=bm25_encoder,
        index=index,
        top_k=10,
        alpha=0.75,
        text_key="text"
    )

    print(f"✅ Pinecone Hierarchical Retriever ready")
    return retriever
