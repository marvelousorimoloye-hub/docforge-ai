# test_internet.py
import socket

print("Testing DNS resolution...")
try:
    ip = socket.gethostbyname('api.cohere.ai')
    print(f"✅ DNS resolution successful → {ip}")
except Exception as e:
    print(f"❌ DNS resolution failed: {e}")

print("\nTesting general internet...")
try:
    import requests
    r = requests.get("https://www.google.com", timeout=5)
    print(f"✅ Internet working (status code: {r.status_code})")
except Exception as e:
    print(f"❌ Internet test failed: {e}")