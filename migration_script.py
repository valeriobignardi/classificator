#!/usr/bin/env python3
"""
File: migration_script.py
Descrizione: Script per migrare dal vecchio al nuovo sistema embedding

COSA FA:
1. Sostituisce tutte le chiamate a embedding_manager.get_shared_embedder()
2. Con simple_embedding_manager.get_embedder_for_tenant()
3. Aggiorna le API per usare il nuovo sistema
"""

import os
import re

def find_embedding_manager_usages():
    """Trova tutti i file che usano il vecchio embedding_manager"""
    
    # Usa percorso relativo dinamico
    root_dir = os.path.dirname(os.path.abspath(__file__))
    files_to_check = []
    
    # Pattern per trovare i file
    for root, dirs, files in os.walk(root_dir):
        # Skip alcune directory
        skip_dirs = {'__pycache__', '.git', '.venv', 'node_modules', 'backup'}
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith(('.py',)):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'embedding_manager' in content and 'get_shared_embedder' in content:
                            files_to_check.append(file_path)
                except Exception as e:
                    print(f"⚠️  Errore leggendo {file_path}: {e}")
    
    return files_to_check

def analyze_usage_patterns():
    """Analizza i pattern di utilizzo"""
    files = find_embedding_manager_usages()
    
    print("🔍 FILE CHE USANO embedding_manager.get_shared_embedder():")
    print("=" * 60)
    
    for file_path in files:
        print(f"\n📄 {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines, 1):
                if 'get_shared_embedder' in line:
                    print(f"  Linea {i:3d}: {line.strip()}")
                    
        except Exception as e:
            print(f"  ⚠️  Errore: {e}")
    
    print(f"\n📊 TOTALE FILE DA AGGIORNARE: {len(files)}")
    return files

if __name__ == "__main__":
    analyze_usage_patterns()
