import os
import re
import json
import urllib.request
import ssl
import time
import sys

def translate_text(text):
    if not text.strip():
        return text
    
    # Simple delay to prevent 429 / WAF blocks
    time.sleep(2)
    
    for attempt in range(3):
        try:
            url = "https://chatbot-test.etc.vn:9006/llm/v1/chat/completions"
            payload = {
                "model": "openai/gpt-oss-120b",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a direct translation tool. Translate the given Chinese text to English. Output ONLY the translated text without any explanation, quotes, or markdown formatting. Keep the exact same punctuation around the text if any."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                "temperature": 0.3
            }
            
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={
                'Content-Type': 'application/json'
            }, method='POST')
            
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
                result = json.loads(response.read().decode('utf-8'))
                translated = result['choices'][0]['message']['content'].strip()
                if translated.startswith('"') and translated.endswith('"'):
                    translated = translated[1:-1]
                return translated if translated else text
        except Exception as e:
            # print error but securely
            print(f"X", end='', flush=True)
            time.sleep(5) # wait longer before retry
            
    return text

zh_pattern = re.compile(r'[\u4e00-\u9fff]+[^\x00-\x7F]*[\u4e00-\u9fff]*|[\u4e00-\u9fff]+')

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    matches = set(zh_pattern.findall(content))
    if not matches:
        return False
        
    print(f"\nTranslating {os.path.basename(filepath)} : ", end='', flush=True)
    sorted_matches = sorted(list(matches), key=len, reverse=True)
    
    new_content = content
    for zh_text in sorted_matches:
        if not re.search(r'[\u4e00-\u9fff]', zh_text):
            continue
            
        en_text = translate_text(zh_text)
        print(".", end='', flush=True)
        
        if en_text and en_text != zh_text:
            new_content = new_content.replace(zh_text, en_text)

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

if __name__ == '__main__':
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    
    file_paths = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.vue') or file.endswith('.js'):
                path = os.path.join(root, file)
                if 'locales' in path or 'i18n.js' in path or 'Home.vue' in path:
                    continue
                file_paths.append(path)
                
    translated_count = 0
    for f in file_paths:
        if process_file(f):
            translated_count += 1
            print(f" [DONE]")
            
    print(f"\nDone! Translated {translated_count} files.")
