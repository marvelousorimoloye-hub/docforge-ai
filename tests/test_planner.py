#test_planner.py
from core.agents import planner_executor
from langchain_core.messages import HumanMessage

# Test query
test_query = "What are the transparency requirements and obligations for high-risk AI systems under the EU AI Act?"


# Call the planner
try:
    # Since we used create_react_agent, we invoke it with a 'messages' list
    response = planner_executor.invoke({"messages": [HumanMessage(content=test_query)]})
    
    # Extract the last message (the agent's plan)
    plan = response["messages"][-1].content
    print("--- PLANNER OUTPUT ---")
    print(plan)
    
except Exception as e:
    print(f"Planner Failed: {e}")
