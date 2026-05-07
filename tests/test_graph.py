import asyncio
import os
from core.graph import build_agent_graph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def main():
    # Ensure the directory exists
    os.makedirs("data/memory", exist_ok=True)
    
    # Use 'async with' to initialize the AsyncSqliteSaver
    async with AsyncSqliteSaver.from_conn_string("data/memory/checkpoints.db") as checkpointer:
        
        # Build the app using the async checkpointer
        app = build_agent_graph(checkpointer=checkpointer)

        query = "What are the transparency requirements and obligations for high-risk AI systems under the EU AI Act?"

        inputs = {
            "query": query,
            "messages": [],
            "past_analyses": []
        }

        print("🚀 Starting End-to-End Async Graph Test...\n")
        print(f"Query: {query}")
        print("=" * 80)

        try:
            # Use astream for async execution
            async for event in app.astream(
                inputs, 
                config={"configurable": {"thread_id": "test_run_001"}}
            ):
                for node, values in event.items():
                    print(f"\n--- 🧠 Node: {node.upper()} ---")
                    
                    if node == "planner":
                        print("Plan generated")
                    elif node == "researcher":
                        print(f"Context gathered: {len(values.get('context', '')):,} characters")
                    elif node == "analyzer":
                        print("Analysis completed")
                    elif node == "report":
                        report = values.get("final_report", "")
                        print("\n📄 FINAL REPORT:")
                        print(report[:1000] + "..." if len(report) > 1000 else report)

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
