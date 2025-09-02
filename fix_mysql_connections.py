#!/usr/bin/env python3
"""
Script per correggere tutti i controlli di connessione MySQL nel PromptManager

Autore: Valerio Bignardi
Data: 2025-09-02
Scopo: Sostituire tutti i controlli "if not self.connection:" con la versione corretta
"""

import re

def fix_connection_checks():
    """
    Corregge tutti i controlli di connessione nel file PromptManager
    """
    file_path = '/home/ubuntu/classificatore/Utils/prompt_manager.py'
    
    print("🔧 Correzione controlli connessione MySQL in PromptManager")
    print("=" * 60)
    
    # Leggi il file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern da sostituire
    old_pattern = r'if not self\.connection:\s*\n\s*self\.connect\(\)'
    new_pattern = '''if not self.connection or not self.connection.is_connected():
            connection_result = self.connect()
            if not connection_result:
                self.logger.error("❌ Impossibile stabilire connessione MySQL")
                return []'''
    
    # Conta le occorrenze
    matches = re.findall(old_pattern, content)
    print(f"🔍 Trovate {len(matches)} occorrenze da correggere")
    
    # Sostituisci tutte le occorrenze
    corrected_content = re.sub(old_pattern, new_pattern, content)
    
    # Scrivi il file corrotto
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(corrected_content)
    
    print(f"✅ Correzioni applicate al file {file_path}")
    print("📝 Tutte le connessioni MySQL ora verificano se la connessione è attiva")

if __name__ == "__main__":
    fix_connection_checks()
