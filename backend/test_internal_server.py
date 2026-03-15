"""
Test: Verify WAF-safe system prompt works with internal server
"""
import os, sys, json, httpx
sys.path.insert(0, os.getcwd())

# Load .env manually
env = {}
env_path = os.path.join(os.getcwd(), "../.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

BASE_URL = env.get("LLM_BASE_URL", "")
MODEL = env.get("LLM_MODEL_NAME", "")
API_KEY = env.get("LLM_API_KEY", "")

print(f"Testing server: {BASE_URL}")
print(f"Model: {MODEL}")
print(f"API Key: {'[none]' if not API_KEY or API_KEY == 'none' else '[set]'}")
print("=" * 60)

# Import the actual services to test current system prompt
from app.services.ontology_generator import ONTOLOGY_SYSTEM_PROMPT, OntologyGenerator

g = OntologyGenerator()
user_msg = g._build_user_message(
    ["VN-Index drops 30 points. Retail investors panic selling. Market opened with heavy volume."],
    "Simulate stock market expert discussion about VNM stock",
    None
)

# Build headers
headers = {"Content-Type": "application/json"}
if API_KEY and API_KEY not in ("none", "null", ""):
    headers["Authorization"] = f"Bearer {API_KEY}"

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg}
    ],
    "temperature": 0.3,
    "max_tokens": 2048
}

url = BASE_URL.rstrip("/") + "/chat/completions"

try:
    with httpx.Client(verify=False, timeout=90.0) as client:
        print(f"\nSending request to: {url}")
        print(f"System prompt length: {len(ONTOLOGY_SYSTEM_PROMPT)} chars")
        print(f"User message length: {len(user_msg)} chars")
        print("\nWaiting for response...")
        
        resp = client.post(url, json=payload, headers=headers)
        
    print(f"\n[STATUS] {resp.status_code}")
    
    content_type = resp.headers.get("content-type", "")
    is_html = resp.text.lstrip().startswith("<") or "text/html" in content_type
    
    if is_html:
        print("[FAIL] WAF blocked — server returned HTML")
        print(f"HTML snippet: {resp.text[:300]}")
    elif resp.status_code == 200:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print(f"[PASS] Server responded OK!")
        print(f"Response length: {len(content)} chars")
        print(f"Preview: {content[:200]}")
    else:
        print(f"[FAIL] Error: {resp.status_code}")
        print(resp.text[:500])
        
except Exception as e:
    print(f"[ERROR] {e}")
