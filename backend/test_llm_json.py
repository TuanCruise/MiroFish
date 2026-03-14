import os, sys, json
sys.path.insert(0, os.getcwd())

env = {}
with open("../.env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

from app.utils.llm_client import LLMClient
from app.services.ontology_generator import ONTOLOGY_SYSTEM_PROMPT, OntologyGenerator

g = OntologyGenerator()
user_msg = g._build_user_message(
    ["VN-Index drops 30 pts. Panic selling from retail investors."],
    "Simulate stock market reactions",
    None
)

client = LLMClient()
raw = client.chat(
    messages=[
        {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg}
    ],
    temperature=0.3,
    max_tokens=4096
)

print("RAW LLM RESPONSE:")
print(raw[:3000])
print(f"\n\nLength: {len(raw)}")

# Try to parse
try:
    import re
    cleaned = raw.strip()
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    cleaned = cleaned.strip()
    data = json.loads(cleaned)
    print(f"\nParsed OK. Keys: {list(data.keys())}")
    if 'entity_types' in data:
        print(f"entity_types[0]: {data['entity_types'][0]}")
except Exception as e:
    print(f"\nParse FAILED: {e}")
