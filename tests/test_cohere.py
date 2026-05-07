# test_cohere.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from core.llm_client import get_llm
from langchain_core.messages import HumanMessage

print("=== Testing Cohere Model ===\n")

# Test 1: Get the Cohere LLM directly
try:
    cohere_llm = get_llm("cohere")        # This forces Cohere (no fallback)
    print(f"✅ LLM object created successfully")
    print(f"Model: {cohere_llm.model}")
except Exception as e:
    print(f"❌ Failed to create LLM object: {e}")
    sys.exit(1)

# Test 2: Simple invocation
try:
    print("\nSending a simple test prompt...")
    
    messages = [HumanMessage(content="Hello! Please introduce yourself in one short sentence.")]
    
    response = cohere_llm.invoke(messages)
    
    print("\n✅ Cohere Response Received:")
    print("-" * 50)
    print(response.content)
    print("-" * 50)
    
    # Show token usage if available
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        print(f"Usage: {response.usage_metadata}")

except Exception as e:
    print(f"\n❌ Error during invocation: {type(e).__name__}: {e}")
    print("Possible causes:")
    print("   - Invalid COHERE_API_KEY")
    print("   - API rate limit reached")
    print("   - Network issue")
    print("   - Model name is deprecated")