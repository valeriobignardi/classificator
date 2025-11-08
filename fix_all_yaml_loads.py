#!/usr/bin/env python3
"""
Script per sostituire yaml.safe_load() con load_config() in TUTTI i file Python
Autore: Valerio Bignardi
Data: 2025-11-08
"""

import os
import re
from pathlib import Path

# File critici da aggiornare (quelli usati dal backend Docker)
CRITICAL_FILES = [
    'Classification/intelligent_classifier.py',
    'mongo_classification_reader.py',
    'server.py',
    'QualityGate/quality_gate_engine.py',
    'AIConfiguration/ai_configuration_service.py',
    'Debug/llm_debugger.py',
    'Debug/ml_ensemble_debugger.py',
    'Services/llm_configuration_service.py',
    'Preprocessing/session_aggregator.py',
    'MongoDB/connettore_mongo.py',
    'Pipeline/end_to_end_pipeline.py',
    'Clustering/clustering_test_service.py',
    'Clustering/intent_clusterer.py',
    'Clustering/hierarchical_adaptive_clusterer.py',
    'Clustering/intelligent_intent_clusterer.py',
    'Clustering/hdbscan_clusterer.py',
    'FineTuning/mistral_finetuning_manager.py',
    'Utils/tracing.py',
    'Utils/tokenization_utils.py',
    'Utils/tool_manager.py',
    'Utils/tenant_config_helper.py',
    'Classification/advanced_ensemble_classifier.py',
    'SemanticMemory/semantic_memory_manager.py',
    'HumanReview/altro_tag_validator.py',
    'TAGS/tag.py',
]

def add_import_if_missing(content, filepath):
    """Aggiunge import di load_config se manca"""
    
    # Controlla se già importa load_config
    if 'from config_loader import load_config' in content:
        return content
    
    # Trova dove aggiungere l'import
    lines = content.split('\n')
    import_index = -1
    
    # Cerca l'ultimo import
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_index = i
    
    if import_index == -1:
        # Nessun import trovato, aggiungi dopo docstring
        for i, line in enumerate(lines):
            if '"""' in line or "'''" in line:
                # Cerca la fine del docstring
                if i + 1 < len(lines):
                    import_index = i + 1
                break
    
    if import_index == -1:
        import_index = 0
    
    # Aggiungi import dopo gli altri import
    import_line = '\n# Import config_loader per caricare config.yaml con variabili ambiente'
    if filepath.startswith('Utils/') or filepath.startswith('Classification/'):
        import_line += '\nimport sys\nimport os\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))'
    import_line += '\nfrom config_loader import load_config\n'
    
    lines.insert(import_index + 1, import_line)
    return '\n'.join(lines)

def fix_yaml_loads(filepath):
    """Sostituisce yaml.safe_load() con load_config() in un file"""
    
    if not os.path.exists(filepath):
        print(f"⏭️  File non trovato: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes = 0
    
    # Pattern 1: with open(config_path, 'r') as f: config = yaml.safe_load(f)
    pattern1 = r"with open\([^)]*config[^)]*\)\s+as\s+\w+:\s*\n\s*(\w+)\s*=\s*yaml\.safe_load\([^)]+\)"
    matches1 = re.findall(pattern1, content)
    if matches1:
        content = re.sub(pattern1, r'\1 = load_config()', content)
        changes += len(matches1)
    
    # Pattern 2: config = yaml.safe_load(file)  (dentro un with già aperto)
    pattern2 = r"(\w+)\s*=\s*yaml\.safe_load\(\w+\)"
    # Ma solo se è per config.yaml
    for match in re.finditer(pattern2, content):
        var_name = match.group(1)
        # Verifica se nel contesto c'è 'config'
        start = max(0, match.start() - 200)
        context = content[start:match.end()]
        if 'config' in context.lower():
            content = content[:match.start()] + f"{var_name} = load_config()" + content[match.end():]
            changes += 1
    
    # Pattern 3: return yaml.safe_load(file)
    pattern3 = r"return\s+yaml\.safe_load\(\w+\)"
    if re.search(pattern3, content):
        content = re.sub(pattern3, 'return load_config()', content)
        changes += 1
    
    # Pattern 4: self.config = yaml.safe_load(file)
    pattern4 = r"self\.config\s*=\s*yaml\.safe_load\(\w+\)"
    if re.search(pattern4, content):
        content = re.sub(pattern4, 'self.config = load_config()', content)
        changes += 1
    
    if changes > 0:
        # Aggiungi import se manca
        content = add_import_if_missing(content, filepath)
        
        # Salva file modificato
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ {filepath}: {changes} sostituzioni")
        return True
    else:
        print(f"⏭️  {filepath}: nessuna modifica necessaria")
        return False

def main():
    print("🔧 Fix automatico yaml.safe_load() → load_config()")
    print("=" * 80)
    
    total_fixed = 0
    total_files = 0
    
    for filepath in CRITICAL_FILES:
        total_files += 1
        if fix_yaml_loads(filepath):
            total_fixed += 1
    
    print("=" * 80)
    print(f"✅ Completato! {total_fixed}/{total_files} file aggiornati")

if __name__ == '__main__':
    main()
