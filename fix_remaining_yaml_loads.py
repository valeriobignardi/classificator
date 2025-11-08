#!/usr/bin/env python3
"""
Script per fixare TUTTI i rimanenti yaml.safe_load() nel progetto
Autore: Valerio Bignardi
Data: 2025
"""

import re
import os
from typing import List, Tuple

# Lista COMPLETA dei file rimanenti da fixare
REMAINING_FILES = [
    # Script di utilità root
    'fix_review_queue_wopta.py',
    'create_prompt_altro_validator.py',
    'analyze_wopta_tenant_collection.py',
    'fix_clustering_direct.py',
    'debug_rappresentanti_reali.py',
    'debug_review_queue_wopta.py',
    'save_classification_function_tool.py',
    'analyze_mongodb_case.py',
    'explore_mongodb.py',
    'analyze_wopta_review_queue.py',
    'debug_representatives_complete.py',
    'debug_humanitas_review_queue.py',
    'manage_tools_table.py',
    'explore_all_mongodb_collections.py',
    'analyze_humanitas_mongodb.py',
    
    # Moduli critici
    'LettoreConversazioni/lettore.py',
    'Database/clustering_results_db.py',
    'Database/remote_tag_sync.py',
    'Database/migrate_prompts_humanitas.py',
    'Database/database_ai_config_service.py',
    'Database/scheduler_config_db.py',
    'Database/engines_schema_manager.py',
    'Models/documento_processing.py',
    'Clustering/clustering_test_service_new.py',
]


def add_import_if_missing(content: str) -> str:
    """Aggiunge import di load_config se mancante"""
    if 'from config_loader import load_config' in content:
        return content
    
    # Trova l'ultimo import
    import_pattern = r'^(import |from )'
    lines = content.split('\n')
    last_import_idx = -1
    
    for idx, line in enumerate(lines):
        if re.match(import_pattern, line.strip()):
            last_import_idx = idx
    
    # Inserisci dopo l'ultimo import
    if last_import_idx >= 0:
        lines.insert(last_import_idx + 1, 'from config_loader import load_config')
    else:
        # Se non ci sono import, inserisci dopo docstring/shebang
        insert_idx = 0
        if lines[0].startswith('#!'):
            insert_idx = 1
        if insert_idx < len(lines) and (lines[insert_idx].startswith('"""') or lines[insert_idx].startswith("'''")):
            # Trova chiusura docstring
            quote = '"""' if '"""' in lines[insert_idx] else "'''"
            for i in range(insert_idx + 1, len(lines)):
                if quote in lines[i]:
                    insert_idx = i + 1
                    break
        lines.insert(insert_idx, 'from config_loader import load_config')
    
    return '\n'.join(lines)


def fix_yaml_loads(content: str) -> Tuple[str, int]:
    """Sostituisce tutti i pattern yaml.safe_load() con load_config()"""
    replacements = 0
    
    # Pattern 1: with open(config_path...) as f: config = yaml.safe_load(f)
    pattern1 = r'with open\([^)]*config_path[^)]*\)\s+as\s+(\w+):\s*\n\s*(\w+)\s*=\s*yaml\.safe_load\(\1\)'
    replacement1 = r'\2 = load_config()'
    content, count1 = re.subn(pattern1, replacement1, content)
    replacements += count1
    
    # Pattern 2: config = yaml.safe_load(f) dopo with open
    pattern2 = r'(\w+)\s*=\s*yaml\.safe_load\((\w+)\)'
    replacement2 = r'\1 = load_config()'
    content, count2 = re.subn(pattern2, replacement2, content)
    replacements += count2
    
    # Pattern 3: return yaml.safe_load(f)
    pattern3 = r'return\s+yaml\.safe_load\(\w+\)'
    replacement3 = 'return load_config()'
    content, count3 = re.subn(pattern3, replacement3, content)
    replacements += count3
    
    # Pattern 4: self.config = yaml.safe_load(f)
    pattern4 = r'self\.config\s*=\s*yaml\.safe_load\(\w+\)'
    replacement4 = 'self.config = load_config()'
    content, count4 = re.subn(pattern4, replacement4, content)
    replacements += count4
    
    # Pattern 5: tenant_config = yaml.safe_load(file)
    pattern5 = r'tenant_config\s*=\s*yaml\.safe_load\(\w+\)'
    replacement5 = 'tenant_config = load_config()'
    content, count5 = re.subn(pattern5, replacement5, content)
    replacements += count5
    
    # Pattern 6: data = yaml.safe_load(f) or {}
    pattern6 = r'(\w+)\s*=\s*yaml\.safe_load\(\w+\)\s+or\s+\{\}'
    replacement6 = r'\1 = load_config()'
    content, count6 = re.subn(pattern6, replacement6, content)
    replacements += count6
    
    # Pattern 7: full_config = yaml.safe_load(file)
    pattern7 = r'full_config\s*=\s*yaml\.safe_load\(\w+\)'
    replacement7 = 'full_config = load_config()'
    content, count7 = re.subn(pattern7, replacement7, content)
    replacements += count7
    
    return content, replacements


def remove_config_path_open_blocks(content: str) -> Tuple[str, int]:
    """
    Rimuove i blocchi 'with open(config_path)' che diventano inutili 
    dopo la conversione a load_config()
    """
    removals = 0
    lines = content.split('\n')
    new_lines = []
    skip_until = -1
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Se siamo in modalità skip
        if i < skip_until:
            i += 1
            continue
        
        # Controlla se questa riga inizia un blocco with open(config_path)
        if re.search(r'with\s+open\([^)]*config_path[^)]*\)\s+as\s+\w+:', line):
            # Trova l'indentazione del with
            indent = len(line) - len(line.lstrip())
            
            # Controlla la riga successiva
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                next_indent = len(next_line) - len(next_line.lstrip())
                
                # Se la riga successiva è load_config() con indentazione maggiore
                if next_indent > indent and 'load_config()' in next_line:
                    # Rimuovi il blocco with e de-indenta la chiamata load_config
                    dedented = next_line[indent:]
                    new_lines.append(dedented)
                    skip_until = i + 2
                    removals += 1
                    i += 2
                    continue
        
        new_lines.append(line)
        i += 1
    
    return '\n'.join(new_lines), removals


def process_file(filepath: str) -> bool:
    """Processa un singolo file"""
    if not os.path.exists(filepath):
        print(f"⚠️  {filepath}: FILE NON TROVATO")
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Step 1: Fix yaml.safe_load()
        content, yaml_replacements = fix_yaml_loads(original_content)
        
        # Step 2: Add import if needed
        if yaml_replacements > 0:
            content = add_import_if_missing(content)
        
        # Step 3: Remove unnecessary with open blocks
        content, block_removals = remove_config_path_open_blocks(content)
        
        total_changes = yaml_replacements + block_removals
        
        if total_changes > 0:
            # Salva il file modificato
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ {filepath}: {yaml_replacements} sostituzioni, {block_removals} blocchi rimossi")
            return True
        else:
            print(f"ℹ️  {filepath}: nessuna modifica necessaria")
            return False
    
    except Exception as e:
        print(f"❌ {filepath}: ERRORE - {e}")
        return False


def main():
    """Funzione principale"""
    print("=" * 80)
    print("FIX COMPLETO DI TUTTI I RIMANENTI yaml.safe_load()")
    print("=" * 80)
    print()
    
    base_path = '/home/ubuntu/classificatore'
    os.chdir(base_path)
    
    total_files = len(REMAINING_FILES)
    fixed_files = 0
    failed_files = 0
    
    for filename in REMAINING_FILES:
        filepath = os.path.join(base_path, filename)
        success = process_file(filepath)
        
        if success:
            fixed_files += 1
        elif not os.path.exists(filepath):
            failed_files += 1
    
    print()
    print("=" * 80)
    print(f"✅ Completato! {fixed_files}/{total_files} file aggiornati")
    if failed_files > 0:
        print(f"⚠️  {failed_files} file non trovati")
    print("=" * 80)
    print()
    print("RIEPILOGO:")
    print(f"- File processati: {total_files}")
    print(f"- File modificati: {fixed_files}")
    print(f"- File non trovati: {failed_files}")
    print(f"- File già corretti: {total_files - fixed_files - failed_files}")


if __name__ == '__main__':
    main()
