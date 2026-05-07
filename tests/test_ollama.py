from core.llm_client import get_llm
from langchain_core.messages import HumanMessage
import time

print("=== Clean Ollama Test ===\n")

llm = get_llm("local")

start = time.time()

response = llm.invoke("Write a short, nice poem about coding. Keep it under 8 lines.")

end = time.time()

print("\n" + "="*60)
print(response.content)
print("="*60)
print(f"Time taken: {end - start:.1f} seconds")