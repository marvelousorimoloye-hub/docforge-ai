#summarizer.py
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from core.llm_client import get_llm

# --- CUSTOM PROMPTS FOR LEGAL/REGULATORY CONTEXT ---

# Stage 1: Map Prompt (Identifies raw data in each chunk)
map_template = """
You are an expert legal researcher. Extract all critical information from the following document chunk:
{text}

FOCUS ON:
- Specific legal obligations or mandates.
- Potential liabilities, red flags, or high-risk clauses.
- Mentioned entities, dates, or financial thresholds.

SUMMARY:"""

MAP_PROMPT = PromptTemplate(template=map_template, input_variables=["text"])

# Stage 2: Reduce Prompt (Synthesizes everything into a Risk Assessment)
combine_template = """
You are a Senior Compliance Architect. Synthesize these intermediate summaries into a final regulatory brief:
{text}

Your final report MUST include:
1. EXECUTIVE SUMMARY: A high-level overview of the document's purpose.
2. CRITICAL RISKS: A prioritized list of legal red flags or liabilities.
3. KEY OBLIGATIONS: Specific actions or compliance mandates required.
4. IMPORTANT METRICS: Any dates, penalties, or dollar amounts found.

FINAL RESEARCH BRIEF:"""

COMBINE_PROMPT = PromptTemplate(template=combine_template, input_variables=["text"])

def summarize_long_text(text: str):
    if len(text) < 3000:
        return text

    # Standard splitter logic as before
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from langchain_core.documents import Document
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = [Document(page_content=t) for t in text_splitter.split_text(text)]

    map_llm = get_llm("local") 
    reduce_llm = get_llm("gemini") 

    # Pass the custom prompts to the chain
    chain = load_summarize_chain(
        llm=map_llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=COMBINE_PROMPT,
        reduce_llm=reduce_llm
    )
    
    return chain.run(docs)
