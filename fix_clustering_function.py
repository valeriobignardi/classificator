#!/usr/bin/env python3

import re

# Legge il file
with open('/home/ubuntu/classificatore/Pipeline/end_to_end_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern per sostituire tutto il blocco di clustering problematico
# Trova dal "try:" fino a dove inizia STEP 2
pattern = r'        try:\s*\n            n_clusters = len\(\[l for l in cluster_labels if l != -1\]\)\s*\n.*?(?=            # STEP 2: Selezione rappresentanti per ogni cluster)'

replacement = '''        try:
            # Usa clustering fornito come parametri
            n_clusters = len([l for l in cluster_labels if l != -1])
            n_outliers = sum(1 for l in cluster_labels if l == -1)
            
            print(f"   📈 Cluster trovati: {n_clusters}")
            print(f"   🔍 Outliers: {n_outliers}")
            
'''

# Sostituisce usando regex DOTALL per catturare newlines
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Scrive il file modificato
with open('/home/ubuntu/classificatore/Pipeline/end_to_end_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ Sostituzione completata")
