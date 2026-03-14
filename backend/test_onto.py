import sys
import os
sys.path.insert(0, os.getcwd())
os.environ['PYTHONUTF8'] = '1'
import traceback

# Monkey-patch chat to print raw response before parsing
from app.utils import llm_client as llm_module

original_chat = llm_module.LLMClient.chat

def debug_chat(self, messages, temperature=0.7, max_tokens=4096, response_format=None):
    payload = {
        "model": self.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format

    headers = {"Content-Type": "application/json"}
    if self.api_key:
        headers["Authorization"] = f"Bearer {self.api_key}"

    url = self._get_url()
    print(f"[DEBUG] Calling URL: {url}")
    response = self.http_client.post(url, json=payload, headers=headers)
    print(f"[DEBUG] Status: {response.status_code}")
    print(f"[DEBUG] Raw text (first 500): {response.text[:500]}")
    response.raise_for_status()
    import json, re
    data = response.json()
    if isinstance(data, str):
        data = json.loads(data)
    message = data["choices"][0]["message"]
    content = message.get("content")
    if not content:
        content = message.get("reasoning_content", "")
    if content is None:
        content = ""
    content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
    print(f"[DEBUG] Final content (first 300): {content[:300]}")
    return content

llm_module.LLMClient.chat = debug_chat

try:
    from app.services.ontology_generator import OntologyGenerator
    g = OntologyGenerator()
    res = g.generate(['VN-Index falls 30 points, investors panic'], 'Simulate stock market reactions')
    print("[RESULT]", list(res.keys()))
except Exception as e:
    traceback.print_exc()
