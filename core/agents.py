# core/agents.py
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch

from core.llm_client import get_llm
from core.config import TAVILY_API_KEY

# ====================== STRUCTURED PLAN ======================
class SubTask(BaseModel):
    type: str = Field(..., description="LOCAL_SEARCH or WEB_SCRAPE")
    content: str = Field(..., description="Search query or full URL")

class ExecutionPlan(BaseModel):
    subtasks: List[SubTask]


# ====================== PLANNER AGENT (Fast - Groq) ======================
planner_system_prompt = """You are the Lead Legal & Regulatory Architect for DocForge AI.
Your goal is to decompose a complex legal query into a high-precision Execution Plan.

### WORKFLOW:
1. **REASON**: Analyze the query. If you do not have specific, direct URLs (like EUR-Lex or official gov portals) for the topic, you MUST use the 'tavily_search_results_json' tool first.
2. **SEARCH**: Use the tool to find the most recent regulatory texts and official guidance URLs.
3. **PLAN**: Once you have gathered the necessary URLs, provide your final answer.

### FINAL RESPONSE FORMAT:
Your final response to the user must contain ONLY the following structured lines:
LOCAL_SEARCH: <specific sub-query for local RAG>
WEB_SCRAPE: <full authoritative URL found during search>

### EXAMPLE FINAL RESPONSE:
LOCAL_SEARCH: high-risk AI system transparency obligations Article 13
LOCAL_SEARCH: technical documentation requirements for AI providers
WEB_SCRAPE: https://ec.europa.eu/commission/presscorner/detail/en/ip_26_203
WEB_SCRAPE: https://ec.europa.eu/commission/presscorner/detail/en/ip_25_2891

Wait to provide this plan until you have verified the URLs using your tool."""

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", planner_system_prompt),
    ("placeholder", "{messages}"),
    ("placeholder", "{agent_scratchpad}"),
])

planner_llm = get_llm("groq")   # Fast model for planning

search_tool = TavilySearch(api_key=TAVILY_API_KEY, k=6, search_depth="advanced")

planner_executor = create_react_agent(
    model=planner_llm,
    tools=[search_tool],
    prompt=planner_prompt,
)

plan_parser = PydanticOutputParser(pydantic_object=ExecutionPlan)


# ====================== ANALYZER (Local - Ollama) ======================
analyzer_prompt = ChatPromptTemplate.from_template(
    """You are a Senior Compliance Analyst and Expert Legal Researcher.

CONTEXT (from local documents and/or web sources):
{context}

USER QUERY:
{query}

Provide a detailed, professional, and well-structured analysis.

Focus on:
- High-risk clauses, compliance gaps, and red flags
- Key obligations, deadlines, financial thresholds, and responsibilities
- Clear, precise citations (e.g., "Document X, Clause 7.3", "EUR-Lex Source", "Section 12(2) of EU AI Act")

Structure your response as:

**Key Obligations**
- Bullet list of main requirements

**Risks and Compliance Gaps**
- Highlight potential risks with severity (High/Medium/Low)

**Important Details**
- Dates, parties, thresholds, definitions, etc.

**Supporting Citations**
- Explicitly reference the sources used

Be objective, precise, and evidence-based. Only use information present in the provided context."""
)

analyzer_chain = analyzer_prompt | get_llm("gemini") | StrOutputParser()


# ====================== REPORT (Fast - Groq) ======================
report_prompt = ChatPromptTemplate.from_template(
    """You are a Professional Executive Report Writer for legal and regulatory matters.

ANALYSIS FROM COMPLIANCE EXPERT:
{analysis}

USER QUERY:
{query}

Write a clean, professional executive report using this exact structure:

**Executive Summary**
**Key Findings**
**Risk Assessment** (include High/Medium/Low ratings)
**Sources Referenced**
**Recommended Next Steps**

Guidelines:
- Use the citations provided in the analysis.
- Maintain a formal, objective tone.
- Make it easy to read with bullet points where appropriate."""
)

report_chain = report_prompt | get_llm("groq") | StrOutputParser()