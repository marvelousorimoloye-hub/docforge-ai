# 1_Upload_Docs.py
import sys
import os
from pathlib import Path
import streamlit as st
import time
from datetime import datetime

# Setup path for core imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.document_parser import parse_document
from core.rag_indexer import get_or_create_index
from langchain_core.documents import Document
from core.config import UPLOAD_DIR 

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.header("📤 Upload Documents to Pinecone")

uploaded_files = st.file_uploader(
    "Upload PDFs, Images, or Office docs", 
    accept_multiple_files=True,
    type=['pdf', 'png', 'jpg', 'jpeg', 'docx']
)

if st.button("Process & Index Documents") and uploaded_files:
    progress_bar = st.progress(0)
    status_text = st.empty()
    all_docs = []
    
    for i, file in enumerate(uploaded_files):
        file_path = Path(UPLOAD_DIR) / file.name
        
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        status_text.info(f"Parsing {file.name}...")
        
        with st.spinner(f"Parsing {file.name}..."):
            result = parse_document(str(file_path))
            
            if result and result.get("markdown"):
                doc = Document(
                    page_content=result["markdown"],
                    metadata={
                        "filename": result["filename"],
                        "source": "uploaded",
                        "upload_time": datetime.now().isoformat()
                    }
                )
                all_docs.append(doc)
                st.success(f"✅ Parsed {file.name} ({len(result['markdown'])} chars)")
            else:
                st.error(f"❌ Failed to parse {file.name}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))

    # ====================== INDEX TO PINECONE ======================
    if all_docs:
        with st.spinner(f"Indexing {len(all_docs)} documents to Pinecone... This may take a while."):
            try:
                start_time = time.time()
                
                retriever = get_or_create_index(all_docs)
                
                duration = time.time() - start_time
                
                st.success(f"🎉 Successfully indexed {len(all_docs)} document(s) to Pinecone!")
                st.info(f"⏱️ Time taken: {duration:.1f} seconds")
                
                # Clear cached components so they reload with new index
                keys_to_clear = ["query_engine", "agent_graph"]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.balloons()
                st.info("👉 You can now go to the **Query Documents** tab to test.")

            except Exception as e:
                st.error("❌ Failed to index documents to Pinecone")
                st.exception(e)
                st.info("💡 Tip: Check your internet connection / DNS settings (try Google DNS 8.8.8.8)")
    else:
        st.warning("No valid documents were processed.")

# Optional: Show current index stats
if st.button("Show Pinecone Index Stats"):
    try:
        from pinecone import Pinecone
        from core.config import PINECONE_API_KEY
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("docforge-hybrid")
        stats = index.describe_index_stats()
        st.json(stats)
    except Exception as e:
        st.error(f"Could not fetch stats: {e}")