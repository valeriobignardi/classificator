#!/usr/bin/env python3
"""
Script per sostituire yaml.safe_load() con load_config() in server.py
Autore: Valerio Bignardi
Data: 2025-11-08
"""

import re

# Leggi server.py
with open('server.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern per trovare blocchi di caricamento config.yaml con yaml.safe_load
pattern = r"with open\('config\.yaml', 'r'\) as f:\s*\n\s*config = yaml\.safe_load\(f\)"
replacement = "config = load_config()"

# Sostituisci tutte le occorrenze
new_content = re.sub(pattern, replacement, content)

# Pattern alternativo con 'file' invece di 'f'
pattern2 = r"with open\('config\.yaml', 'r'\) as file:\s*\n\s*config = yaml\.safe_load\(file\)"
new_content = re.sub(pattern2, replacement, new_content)

# Pattern con encoding
pattern3 = r"with open\('config\.yaml', 'r', encoding='utf-8'\) as f:\s*\n\s*config = yaml\.safe_load\(f\)"
new_content = re.sub(pattern3, replacement, new_content)

# Pattern con path assoluto
pattern4 = r"with open\(config_path, 'r', encoding='utf-8'\) as f:\s*\n\s*config = yaml\.safe_load\(f\)"
new_content = re.sub(pattern4, replacement, new_content)

# Pattern per cfg invece di config
pattern5 = r"with open\('config\.yaml', 'r'\) as f:\s*\n\s*cfg = yaml\.safe_load\(f\)"
replacement5 = "cfg = load_config()"
new_content = re.sub(pattern5, replacement5, new_content)

# Scrivi file aggiornato
with open('server.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ server.py aggiornato con load_config()")
print(f"📝 Pattern sostituiti: {len(re.findall(pattern, content)) + len(re.findall(pattern2, content)) + len(re.findall(pattern3, content)) + len(re.findall(pattern4, content)) + len(re.findall(pattern5, content))}")
