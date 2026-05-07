# core/graph.py
import gc
import re
from typing import TypedDict, Annotated, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages

# Import utilities and agents
from core.graph_utils import rerank_content, ethical_scrape,with_retry,pinecone_circuit, groq_circuit, scrape_circuit
from core.rag_indexer import get_or_create_index
from core.agents import planner_executor, plan_parser, analyzer_chain, report_chain
from core.memory_manager import MemoryManager   # ← Added
from core.evaluator import evaluator
import os
from dotenv import load_dotenv

load_dotenv()

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "docforge-ai"

# ====================== STATE ======================
class AgentState(TypedDict):
    query: str
    messages: Annotated[List[BaseMessage], add_messages]
    subtasks: str
    execution_plan: Optional[dict]
    context: str
    analysis: str
    final_report: str
    thread_id: str
    past_analyses: List[dict]
    next_node: Optional[str]


# Initialize Semantic Memory Manager
memory_manager = MemoryManager()


# ====================== NODES ======================
@with_retry(max_retries=3, min_delay=2)
@groq_circuit
def planner_node(state: AgentState):
    print(f"\n🔹 [PLANNER] Starting for query: {state['query'][:80]}...")
    result = planner_executor.invoke({
        "messages": state.get("messages", []) + [HumanMessage(content=state["query"])]
    })
    plan_text = result["messages"][-1].content
    print(f"✅ [PLANNER] Completed - Generated plan ({len(plan_text)} chars)")
    return {
        "subtasks": plan_text,
        "messages": [AIMessage(content="Planner: Created execution plan.")],
    }

@with_retry(max_retries=2, min_delay=3)
@pinecone_circuit
def researcher_node(state: AgentState):
    query = state["query"]
    subtasks_text = state.get("subtasks", "")
    all_candidates = []
    retriever = get_or_create_index()
    # 1. Try Pydantic Parser first
    try:
        execution_plan = plan_parser.parse(subtasks_text)
        local_queries = [t.content for t in execution_plan.subtasks if t.type == "LOCAL_SEARCH"]
        scrape_urls = [t.content for t in execution_plan.subtasks if t.type == "WEB_SCRAPE"]
    except Exception:
        # 2. REGEX FALLBACK
        print("⚠️ Pydantic parse failed, using regex fallback...")
        local_queries = [q.strip() for q in re.findall(r"LOCAL_SEARCH:\s*(.+)", subtasks_text)]
        scrape_urls = list(set(re.findall(r"WEB_SCRAPE:\s*(https?://[^\s\)]+)", subtasks_text)))

    # 3. Last resort
    if not local_queries and not scrape_urls:
        local_queries = [query]
        scrape_urls = []

    # Local RAG
    if retriever and local_queries:
        print("   📚 Running Local RAG search...")
        for q in local_queries[:5]:
            docs = retriever.invoke(q)
            for d in docs:
                all_candidates.append(f"SOURCE [Local]: {d.metadata.get('filename', 'Unknown')}\nCONTENT: {d.page_content[:1200]}")

    # Web Scraping (Your current logic preserved)
    if scrape_urls:
        keywords = [w.lower() for w in query.split() if len(w) > 3]

        for url in list(set(scrape_urls))[:4]:
            try:
                raw_markdown = ethical_scrape(url)
                
                chunk_size = 1500
                overlap = 300
                
                for i in range(0, len(raw_markdown), chunk_size - overlap):
                    chunk = raw_markdown[i : i + chunk_size]
                    if any(k in chunk.lower() for k in keywords) or len(raw_markdown) < 8000:
                        all_candidates.append(f"SOURCE [Web]: {url}\nCONTENT: {chunk}")

            except Exception as e:
                print(f"Scrape error: {e}")
                continue

    # ==================== NEW: SEMANTIC MEMORY RETRIEVAL ====================
    past_memories = memory_manager.retrieve_relevant_memories(query, k=3)
    if past_memories:
        memory_text = "\n\n".join([
            f"--- PAST ANALYSIS ---\nQuery: {doc.metadata.get('query', '')}\n"
            f"{doc.page_content[:700]}..."
            for doc in past_memories
        ])
        all_candidates.append(f"SOURCE [Long-term Memory]:\n{memory_text}")
        print(f"🧠 Retrieved {len(past_memories)} relevant past analyses from memory")

    # Final Reranking
    final_context = rerank_content(query, all_candidates, top_k=7) if all_candidates else "No relevant context found."
    
    print(f"📊 Gathered context size: {len(final_context):,} characters")
    print("\n🔍 Context Preview:")
    print(final_context[:800] + "...")
    
    gc.collect()
    return {"context": final_context, "subtasks": subtasks_text}

@with_retry(max_retries=2, min_delay=3)
@groq_circuit
def analyzer_node(state: AgentState):
    print(f"\n🔹 [ANALYZER] Starting analysis...")
    analysis = analyzer_chain.invoke({"context": state["context"], "query": state["query"]})
    print(f"✅ [ANALYZER] Completed - Analysis length: {len(analysis):,} characters")
    print(f"   Preview: {analysis[:300]}...\n")
    return {"analysis": analysis, "messages": [AIMessage(content="Analyzer: Completed analysis.")]}

@with_retry(max_retries=3, min_delay=2)
@groq_circuit
def report_node(state: AgentState):
    print(f"\n🔹 [REPORT] Generating final report...")
    final_report = report_chain.invoke({
        "analysis": state["analysis"], 
        "query": state["query"]
    })
    
    print(f"✅ [REPORT] Completed - Report length: {len(final_report):,} characters")

    # === Run Evaluation ===
    try:
        eval_scores = evaluator.evaluate_report(
            query=state["query"],
            analysis=state["analysis"],
            final_report=final_report
        )
        print(f"📊 Report Evaluation Score: {eval_scores.get('overall_score', 'N/A')}/10")
    except Exception as e:
        print(f"⚠️ Evaluation failed: {e}")
        eval_scores = {"overall_score": None}

    # Save to memory
    memory_manager.save_analysis(
        query=state["query"],
        report=final_report,
        feedback=state.get("current_feedback", "")
    )

    memory_entry = {
        "query": state["query"],
        "report": final_report,
        "timestamp": datetime.now().isoformat(),
        "feedback": state.get("current_feedback", ""),
        "eval_score": eval_scores.get("overall_score")
    }
    
    past_analyses = state.get("past_analyses", []) + [memory_entry]

    return {
        "final_report": final_report, 
        "past_analyses": past_analyses, 
        "messages": [AIMessage(content=final_report)],
        "eval_scores": eval_scores   # Optional: pass to Streamlit
    }


# ====================== BUILD GRAPH ======================
def build_agent_graph(checkpointer=None):
    """Build graph with LangSmith observability"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("report", report_node)
    
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "analyzer")
    workflow.add_edge("analyzer", "report")
    workflow.add_edge("report", END)

    return workflow.compile(checkpointer=checkpointer)