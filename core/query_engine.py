#query_engine.py
from core.rag_indexer import get_or_create_index
from core.llm_client import get_llm
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

def create_query_engine():
    # 1. Get the Hybrid EnsembleRetriever (BM25 + FAISS)
    # This now returns the EnsembleRetriever directly from rag_indexer.py
    hybrid_retriever = get_or_create_index()
    
    if hybrid_retriever is None:
        return None
    
    # 2. Initialize the LLM (Groq)
    llm = get_llm("groq")
    
    # 3. Define the Retrieval Prompt
    system_prompt = (
        "You are a Legal & Compliance Assistant. "
        "Use the following pieces of retrieved context to answer the user's question accurately. "
        "If the answer isn't in the context, clearly state that the provided documents "
        "do not contain the information. Provide citations if possible."
        "\n\n"
        "CONTEXT:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. Build the RAG Chain
    # We use the hybrid_retriever directly
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    # The final retrieval chain (Links the Hybrid Search to the LLM)
    rag_chain = create_retrieval_chain(hybrid_retriever, combine_docs_chain)
    
    return rag_chain
