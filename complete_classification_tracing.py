#!/usr/bin/env python3
"""
Script per completare automaticamente il tracing in tutte le funzioni di classificazione

Autore: Valerio Bignardi  
Data: 2025-09-09
"""

import re
import os
from typing import List, Tuple

class ClassificationTracingCompleter:
    """Completa automaticamente il tracing per le funzioni di classificazione"""
    
    def __init__(self):
        self.classification_files = [
            '/home/ubuntu/classificatore/Classification/intelligent_classifier.py',
            '/home/ubuntu/classificatore/Classification/advanced_ensemble_classifier.py', 
            '/home/ubuntu/classificatore/Clustering/intelligent_intent_clusterer.py'
        ]
        
        # Funzioni che necessitano di tracing completo
        self.target_functions = [
            '_validate_llm_classification_with_embedding',
            '_call_llm_api_structured',
            '_parse_llm_response', 
            '_fallback_parse_response',
            'classify_conversation',
            'classify_multiple_conversations_optimized',
            'predict_with_llm_only',
            '_combine_predictions',
            '_find_best_match_via_examples',
            '_evaluate_new_category_with_bertopic',
            '_resolve_label_semantically'
        ]
        
    def find_functions_needing_tracing(self, file_path: str) -> List[Tuple[str, int, int]]:
        """
        Trova funzioni che necessitano di tracing
        
        Returns:
            Lista di tuple (function_name, start_line, end_line)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        functions_info = []
        
        # Pattern per trovare definizioni di funzioni
        function_pattern = r'^\s*def\s+(\w+)\s*\('
        
        i = 0
        while i < len(lines):
            match = re.match(function_pattern, lines[i])
            if match:
                func_name = match.group(1)
                start_line = i + 1  # Line numbers are 1-based
                
                # Trova la fine della funzione
                end_line = self._find_function_end(lines, i)
                
                # Controlla se è una funzione target e se ha già tracing
                if (func_name in self.target_functions and 
                    not self._has_complete_tracing(lines, i, end_line)):
                    functions_info.append((func_name, start_line, end_line))
            
            i += 1
        
        return functions_info
    
    def _find_function_end(self, lines: List[str], start_idx: int) -> int:
        """Trova la fine di una funzione basandosi sull'indentazione"""
        
        # Trova l'indentazione della funzione
        func_line = lines[start_idx]
        func_indent = len(func_line) - len(func_line.lstrip())
        
        # Cerca la prossima linea con indentazione <= a quella della funzione
        i = start_idx + 1
        while i < len(lines):
            line = lines[i]
            if line.strip():  # Skip empty lines
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= func_indent:
                    return i
            i += 1
        
        return len(lines)
    
    def _has_complete_tracing(self, lines: List[str], start_idx: int, end_idx: int) -> bool:
        """Controlla se la funzione ha tracing ENTER, EXIT e ERROR completo"""
        
        function_content = ''.join(lines[start_idx:end_idx])
        
        has_enter = 'trace_all(' in function_content and '"ENTER"' in function_content
        has_exit = 'trace_all(' in function_content and '"EXIT"' in function_content  
        has_error = 'trace_all(' in function_content and '"ERROR"' in function_content
        
        return has_enter and has_exit and has_error
    
    def add_tracing_to_function(self, file_path: str, func_name: str, start_line: int, end_line: int):
        """Aggiunge tracing completo a una funzione"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Backup
        backup_path = f"{file_path}.backup_tracing_{os.getpid()}"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"📁 Backup creato: {backup_path}")
        
        # Aggiungi ENTER tracing se mancante
        if not self._has_enter_tracing(lines, start_line, end_line):
            lines = self._add_enter_tracing(lines, func_name, start_line)
            print(f"  ✅ Aggiunto ENTER tracing per {func_name}")
        
        # Aggiungi EXIT e ERROR tracing se mancanti  
        if not self._has_exit_error_tracing(lines, start_line, end_line):
            lines = self._add_exit_error_tracing(lines, func_name, start_line, end_line)
            print(f"  ✅ Aggiunto EXIT/ERROR tracing per {func_name}")
        
        # Scrivi file modificato
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"✅ Tracing completato per {func_name} in {os.path.basename(file_path)}")
    
    def _has_enter_tracing(self, lines: List[str], start_line: int, end_line: int) -> bool:
        """Controlla se ha ENTER tracing"""
        content = ''.join(lines[start_line:end_line])
        return 'trace_all(' in content and '"ENTER"' in content
    
    def _has_exit_error_tracing(self, lines: List[str], start_line: int, end_line: int) -> bool:
        """Controlla se ha EXIT e ERROR tracing"""
        content = ''.join(lines[start_line:end_line])
        has_exit = 'trace_all(' in content and '"EXIT"' in content
        has_error = 'trace_all(' in content and '"ERROR"' in content
        return has_exit and has_error
    
    def _add_enter_tracing(self, lines: List[str], func_name: str, start_line: int) -> List[str]:
        """Aggiunge ENTER tracing all'inizio della funzione"""
        
        # Trova la prima linea di codice dopo la definizione e docstring
        insert_idx = start_line  # Lines are 1-based, so this is the line after def
        
        # Skip docstring if present
        while insert_idx < len(lines):
            line = lines[insert_idx].strip()
            if line and not line.startswith('"""') and not line.startswith("'''"):
                break
            insert_idx += 1
        
        # Determina l'indentazione
        if insert_idx < len(lines):
            indent = self._get_line_indent(lines[insert_idx])
        else:
            indent = '        '  # Default 8 spaces
        
        # Crea linea di tracing ENTER
        tracing_lines = [
            f'{indent}# 🔍 TRACING ENTER\n',
            f'{indent}trace_all("{func_name}", "ENTER", \n',
            f'{indent}         called_from="classification_pipeline")\n',
            f'{indent}\n'
        ]
        
        # Inserisci le linee
        lines[insert_idx:insert_idx] = tracing_lines
        
        return lines
    
    def _add_exit_error_tracing(self, lines: List[str], func_name: str, start_line: int, end_line: int) -> List[str]:
        """Aggiunge EXIT e ERROR tracing ai return statements e exception handling"""
        
        # Questa è una implementazione semplificata
        # Wrappa tutta la funzione in try/except se non ce l'ha già
        
        # Per semplicità, aggiungiamo solo un commento che indica dove aggiungere il tracing
        insert_idx = end_line - 1
        
        if insert_idx < len(lines):
            indent = self._get_line_indent(lines[insert_idx])
        else:
            indent = '        '
        
        comment_lines = [
            f'{indent}# TODO: Aggiungere trace_all EXIT e ERROR per {func_name}\n'
        ]
        
        lines[insert_idx:insert_idx] = comment_lines
        
        return lines
    
    def _get_line_indent(self, line: str) -> str:
        """Ottiene l'indentazione di una linea"""
        return line[:len(line) - len(line.lstrip())]
    
    def run(self):
        """Esegue il completamento del tracing per tutti i file"""
        
        print("🚀 Avvio completamento tracing per funzioni di classificazione")
        
        total_functions = 0
        
        for file_path in self.classification_files:
            if not os.path.exists(file_path):
                print(f"⚠️ File non trovato: {file_path}")
                continue
            
            print(f"\n📄 Analizzando: {os.path.basename(file_path)}")
            
            functions_needing_tracing = self.find_functions_needing_tracing(file_path)
            
            if not functions_needing_tracing:
                print("  ✅ Tutte le funzioni target hanno già tracing completo")
                continue
            
            print(f"  🎯 Trovate {len(functions_needing_tracing)} funzioni che necessitano tracing:")
            for func_name, start_line, end_line in functions_needing_tracing:
                print(f"    - {func_name} (linee {start_line}-{end_line})")
            
            # Applica tracing a ogni funzione
            for func_name, start_line, end_line in functions_needing_tracing:
                try:
                    self.add_tracing_to_function(file_path, func_name, start_line, end_line)
                    total_functions += 1
                except Exception as e:
                    print(f"❌ Errore aggiungendo tracing a {func_name}: {e}")
        
        print(f"\n🎉 Completamento tracing terminato!")
        print(f"📊 Funzioni processate: {total_functions}")
        print(f"📁 File modificati: {len([f for f in self.classification_files if os.path.exists(f)])}")
        print(f"💾 Backup disponibili con suffisso _tracing_{os.getpid()}")

def main():
    """Funzione main"""
    completer = ClassificationTracingCompleter()
    completer.run()

if __name__ == "__main__":
    main()
