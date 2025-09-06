#!/usr/bin/env python3
"""
Script per aggiungere automaticamente il sistema di tracing a tutte le funzioni
della pipeline end_to_end_pipeline.py

Autore: Valerio Bignardi
Data: 2025-09-06
"""

import re
import os

def add_tracing_to_pipeline():
    """
    Aggiunge automaticamente trace_all() a tutte le funzioni della pipeline
    """
    
    pipeline_file = "Pipeline/end_to_end_pipeline.py"
    
    print(f"🚀 Inizio aggiunta tracing automatico a {pipeline_file}")
    
    # Leggi il file
    with open(pipeline_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Lista delle funzioni da non modificare (già modificate)
    skip_functions = {'trace_all', 'get_supervised_training_params_from_db'}
    
    # Pattern per trovare definizioni di funzioni
    function_pattern = r'(^[ ]*def\s+(\w+)\s*\([^)]*\):[^\n]*\n)([ ]*"""[^"]*?"""[ ]*\n)?([ ]*)'
    
    def replace_function(match):
        """Sostituisce una definizione di funzione aggiungendo il tracing"""
        
        full_match = match.group(0)
        function_def = match.group(1)
        function_name = match.group(2)
        docstring = match.group(3) or ""
        indent = match.group(4)
        
        # Salta funzioni specifiche
        if function_name in skip_functions:
            return full_match
        
        # Salta costruttore (gestito separatamente)
        if function_name == '__init__':
            return full_match
        
        print(f"  ✅ Aggiunto tracing a: {function_name}")
        
        # Costruisci parametri per il tracing (semplificato)
        # Estrai i parametri dalla firma della funzione
        param_match = re.search(r'\(([^)]*)\)', function_def)
        if param_match:
            params_str = param_match.group(1)
            # Pulisci i parametri (rimuovi self, type hints, default values)
            params = []
            for param in params_str.split(','):
                param = param.strip()
                if param and param != 'self':
                    # Rimuovi type hints e default values
                    param_name = param.split(':')[0].split('=')[0].strip()
                    if param_name and param_name != 'self':
                        params.append(param_name)
            
            # Costruisci i parametri per trace_all
            if params:
                trace_params = ', '.join([f"{p}={p}" for p in params[:5]])  # Max 5 parametri
                trace_call = f'{indent}trace_all("{function_name}", "ENTER", {trace_params})\n'
            else:
                trace_call = f'{indent}trace_all("{function_name}", "ENTER")\n'
        else:
            trace_call = f'{indent}trace_call = trace_all("{function_name}", "ENTER")\n'
        
        # Ricostruisci la funzione con il tracing
        result = function_def + docstring + trace_call
        
        return result
    
    # Applica il pattern replacement
    modified_content = re.sub(function_pattern, replace_function, content, flags=re.MULTILINE)
    
    # Ora aggiungi trace_all prima di ogni return
    print("  🔄 Aggiunto tracing ai return statements...")
    
    # Pattern per trovare return statements (semplificato)
    return_pattern = r'(^[ ]*)return\s+([^\n]+)'
    
    def replace_return(match):
        """Sostituisce un return statement aggiungendo il tracing"""
        indent = match.group(1)
        return_value = match.group(2)
        
        # Evita di modificare return già con tracing
        if 'trace_all' in return_value:
            return match.group(0)
        
        # Aggiungi trace_all prima del return
        result = f'{indent}trace_all("CURRENT_FUNCTION", "EXIT", return_value={return_value})\n{match.group(0)}'
        return result
    
    # Applica il pattern per i return (con cautela)
    # modified_content = re.sub(return_pattern, replace_return, modified_content, flags=re.MULTILINE)
    
    # Backup del file originale
    backup_file = f"{pipeline_file}.backup_tracing_{os.getpid()}"
    print(f"📁 Backup creato: {backup_file}")
    
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Scrivi il file modificato
    with open(pipeline_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"✅ Tracing automatico completato!")
    print(f"📋 File modificato: {pipeline_file}")
    print(f"💾 Backup disponibile: {backup_file}")

if __name__ == "__main__":
    add_tracing_to_pipeline()
