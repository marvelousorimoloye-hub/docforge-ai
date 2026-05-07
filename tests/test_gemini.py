# test_gemini.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from core.llm_client import get_llm
from langchain_core.messages import HumanMessage

print("=== Testing Gemini Model ===\n")

# Test with "gemini" mode
try:
    gemini_llm = get_llm("gemini")
    print(f"✅ LLM object created successfully")
    print(f"Model: {gemini_llm.model_name if hasattr(gemini_llm, 'model_name') else 'gemini-2.5-flash'}")
except Exception as e:
    print(f"❌ Failed to create Gemini LLM: {e}")
    sys.exit(1)

# Test actual API call
try:
    print("\nSending test prompt to Gemini...")
    
    messages = [HumanMessage(content="Hello! Please introduce yourself in one short, friendly sentence and confirm you are Gemini.")]

    response = gemini_llm.invoke(messages)
    
    print("\n✅ Gemini Response Received:")
    print("-" * 60)
    print(response.content)
    print("-" * 60)

    # Show usage info if available
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        print(f"Token Usage: {response.usage_metadata}")

except Exception as e:
    print(f"\n❌ Error during Gemini invocation: {type(e).__name__}: {e}")
    print("\nPossible causes:")
    print("   - Invalid or missing GEMINI_API_KEY")
    print("   - Network / DNS issue (same as Cohere)")
    print("   - Quota / rate limit reached")
    print("   - Incorrect model name")