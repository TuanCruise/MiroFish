import os
import re

root_dir = r'd:\Working\DuAn\R_D\MiroFish\backend'
chinese_pattern = re.compile(r'[\u4e00-\u9fff]')

with open('chinese_files.txt', 'w', encoding='utf-8') as out:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'venv' in dirpath or '__pycache__' in dirpath:
            continue
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            if chinese_pattern.search(line):
                                out.write(f"{filepath}:{i+1} : {line.strip()}\n")
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
print("Done")
