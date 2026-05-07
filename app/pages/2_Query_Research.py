# 2_Query_Research.py
import torch
torch.classes.__path__ = []

import sys
import os
from pathlib import Path
from datetime import datetime
import streamlit as st
from langchain_core.messages import HumanMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.query_engine import create_query_engine
from core.graph import build_agent_graph
from core.graph_utils import DocForgeReportGenerator
import socket
socket.setdefaulttimeout(30)

# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="DocForge AI", layout="wide")
st.header("🔍 DocForge AI - Intelligent Document Analysis")

# ====================== MEMORY SETUP ======================
MEMORY_DIR = Path("data/memory")
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_DB_PATH = MEMORY_DIR / "checkpoints.db"

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

@st.cache_resource(show_spinner="Initializing persistent memory...")
def get_checkpointer():
    conn = sqlite3.connect(str(MEMORY_DB_PATH), check_same_thread=False)
    return SqliteSaver(conn)

checkpointer = get_checkpointer()



# ====================== INITIALIZATION ======================
if "query_engine" not in st.session_state:
    with st.spinner("Loading vector index..."):
        st.session_state.query_engine = create_query_engine()

if "agent_graph" not in st.session_state:
    with st.spinner("Initializing Multi-Agent System..."):
        st.session_state.agent_graph = build_agent_graph(checkpointer=checkpointer)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session_{int(datetime.now().timestamp())}"

if "past_analyses" not in st.session_state:
    st.session_state.past_analyses = []

if "current_feedback" not in st.session_state:
    st.session_state.current_feedback = ""

query_engine = st.session_state.query_engine
agent_graph = st.session_state.agent_graph
thread_id = st.session_state.thread_id

# ====================== TABS ======================
tab1, tab2, tab3 = st.tabs(["📊 New Analysis", "📚 Past Analyses", "⚙️ Settings"])

with tab1:
    st.success("✅ Multi-Agent System Ready")

    mode = st.radio("Mode:", ["Simple RAG", "Multi-Agent Analysis"], horizontal=True)
    query = st.text_input("Your Query:", placeholder="Analyze termination clauses, liability limits and compliance risks...")

    if st.button("🚀 Run Analysis", type="primary") and query:
        with st.spinner("Executing multi-agent workflow..."):
            try:
                if mode == "Simple RAG":
                    response = query_engine.invoke({"input": query})
                    st.markdown("### 📝 Answer")
                    st.markdown(response.get("answer", ""))

                else:
                    # Multi-Agent Analysis
                    user_feedback = st.session_state.get("current_feedback", "")

                    initial_state = {
                        "query": query,
                        "messages": [HumanMessage(content=query)],
                        "past_analyses": st.session_state.past_analyses.copy()
                    }

                    if user_feedback:
                        initial_state["messages"].append(
                            HumanMessage(content=f"User Feedback for improvement: {user_feedback}")
                        )

                    result = agent_graph.invoke(
                        initial_state, 
                        config={"configurable": {"thread_id": thread_id}}
                    )

                    final_report = result.get("final_report", "No report generated.")
                    eval_scores = result.get("eval_scores", {})

                    st.markdown("### 📋 Final Report")
                    st.markdown(final_report)

                    # PDF Export
                    if final_report:
                        pdf_bytes = DocForgeReportGenerator.generate_pdf(
                            report_text=final_report, 
                            query=query, 
                            analysis=result.get("analysis", "")
                        )
                        st.download_button(
                            label="⬇️ Download Professional PDF Report",
                            data=pdf_bytes,
                            file_name=f"DocForge_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )

                    # ==================== LANGSMITH EVALUATION & FEEDBACK ====================
                    st.markdown("### 📊 Evaluation & Feedback")

                    if eval_scores:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric("Overall Score", f"{eval_scores.get('overall_score', 'N/A')}/10")
                        with col2:
                            st.caption(eval_scores.get("feedback", ""))

                    # Thumbs Up/Down Feedback
                    feedback_option = st.radio(
                        "Was this report helpful?",
                        ["👍 Helpful", "👎 Not Helpful", "😐 Neutral"],
                        horizontal=True,
                        key=f"feedback_{thread_id}"
                    )

                    if st.button("Submit Feedback to LangSmith"):
                        try:
                            from langsmith import Client
                            ls_client = Client()
                            
                            # LangSmith automatically adds run_id in newer versions
                            run_id = result.get("__run_id__") or result.get("run_id")
                            if run_id:
                                score = 1.0 if "Helpful" in feedback_option else 0.0 if "Not Helpful" in feedback_option else 0.5
                                ls_client.create_feedback(
                                    run_id=run_id,
                                    key="user_feedback",
                                    score=score,
                                    comment=feedback_option,
                                    source="streamlit_ui"
                                )
                                st.success("✅ Feedback successfully logged to LangSmith!")
                            else:
                                st.warning("Run ID not found. Feedback not logged.")
                        except Exception as e:
                            st.error(f"Failed to log feedback: {e}")

                    # Enhanced Debug Panel
                    with st.expander("🔍 Detailed Workflow Debug", expanded=False):
                        st.info(f"Context size: {len(result.get('context','')):,} characters")
                        
                        if "execution_plan" in result:
                            st.json(result["execution_plan"], expanded=False)

                        context = result.get('context', '')
                        if "Long-term Memory" in context:
                            st.success("🧠 Long-term Memory was used")
                            st.text_area("Retrieved Memories Preview", 
                                       context[context.find("Long-term Memory"):context.find("Long-term Memory")+800],
                                       height=200)

                    # Human-in-the-loop Feedback
                    st.markdown("### 💡 Suggest Improvements")
                    feedback = st.text_area(
                        "What should be improved in the next version?",
                        placeholder="Make risk assessment more detailed...",
                        key="feedback_input"
                    )

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("✅ Accept & Save"):
                            st.session_state.past_analyses.append({
                                "query": query,
                                "report": final_report,
                                "timestamp": datetime.now().isoformat(),
                                "feedback": "Accepted"
                            })
                            st.success("Report saved to history!")
                            st.session_state.current_feedback = ""

                    with col2:
                        if st.button("🔄 Regenerate with Feedback") and feedback.strip():
                            st.session_state.current_feedback = feedback
                            st.rerun()

                    with col3:
                        if st.button("Clear Feedback"):
                            st.session_state.current_feedback = ""
                            st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

# Keep your other tabs unchanged
with tab2:
    st.header("📚 Past Analyses")
    if not st.session_state.past_analyses:
        st.info("No analyses saved yet.")
    else:
        for analysis in reversed(st.session_state.past_analyses):
            with st.expander(f"{analysis['timestamp'][:16]} — {analysis['query'][:70]}...", expanded=False):
                st.write("**Query:**", analysis['query'])
                st.write("**Report:**")
                st.write(analysis.get('report', '')[:600] + "..." if len(analysis.get('report', '')) > 600 else analysis.get('report', ''))
                if analysis.get('feedback'):
                    st.caption(f"Feedback: {analysis['feedback']}")

with tab3:
    st.header("⚙️ Settings")
    st.write(f"Thread ID: `{thread_id}`")
    st.write(f"Total Past Analyses: **{len(st.session_state.past_analyses)}**")
    if st.button("Clear History"):
        st.session_state.past_analyses = []
        st.success("History cleared.")