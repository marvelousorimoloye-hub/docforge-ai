import asyncio
from core.graph import researcher_node

def test_researcher_standalone():
    # 1. Define the mock state exactly as the planner would provide it
    mock_state = {
        "query": "What are the transparency requirements for high-risk AI systems under the EU AI Act?",
        "subtasks": """
LOCAL_SEARCH: high-risk AI system transparency obligations Article 13
LOCAL_SEARCH: technical documentation requirements for AI providers
WEB_SCRAPE: https://agfundernews.com/a-brussels-moat-can-european-regulation-work-for-founders
WEB_SCRAPE: https://letsdatascience.com/news/eu-talks-stall-over-exemptions-in-ai-act-89cccaf6
        """,
        "messages": [],
        "context": "",
        "past_analyses": []
    }

    print("🚀 Starting Standalone Researcher Test...")
    print("-" * 50)

    # 2. Execute the node (researcher_node is synchronous in your code)
    try:
        result = researcher_node(mock_state)

        print("\n✅ Researcher Execution Complete!")
        print(f"📊 Gathered context size: {len(result.get('context', '')):,} characters")
        
        print("\n🔍 Context Preview:")
        print(result.get('context', '')[:800] + "...")

    except Exception as e:
        print(f"\n❌ Error during research: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_researcher_standalone()
