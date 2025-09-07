#!/usr/bin/env python3
"""
Script per aggiungere automaticamente trace_all a tutte le rotte rimanenti di server.py
Author: Valerio Bignardi
Date: 2024-12-19
"""

import re
import os

def add_trace_to_remaining_routes():
    """
    Aggiunge trace_all alle rotte che non l'hanno ancora
    """
    
    server_file = "/home/ubuntu/classificatore/server.py"
    
    print(f"🚀 Aggiunta tracing automatico alle rotte rimanenti in {server_file}")
    
    # Leggi il file
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Lista delle funzioni che già hanno trace_all
    existing_traces = {
        'home', 'health_check', 'classify_all_sessions', 
        'classify_new_sessions', 'get_client_status', 'supervised_training'
    }
    
    # Pattern per trovare le rotte Flask
    # Cerca: @app.route(...) seguito da def function_name(...)
    pattern = r'(@app\.route\([^)]+\))\s*\n(def\s+(\w+)\s*\([^)]*\):[^\n]*\n)((?:[ ]*"""[^"]*?"""[ ]*\n)?)([ ]*)(.*?)(?=\n[ ]*[a-zA-Z_]|$)'
    
    modifications = 0
    
    def replace_route(match):
        nonlocal modifications
        
        route_decorator = match.group(1)
        function_def = match.group(2)
        function_name = match.group(3)
        docstring = match.group(4) or ""
        indent = match.group(5)
        first_code_line = match.group(6)
        
        # Salta se già ha trace_all
        if function_name in existing_traces or 'trace_all' in first_code_line:
            return match.group(0)
        
        print(f"  ✅ Aggiunto tracing a: {function_name}")
        modifications += 1
        
        # Estrai parametri dalla funzione
        param_match = re.search(r'\(([^)]*)\)', function_def)
        trace_params = []
        
        if param_match:
            params_str = param_match.group(1)
            for param in params_str.split(','):
                param = param.strip()
                if param and 'self' not in param:
                    param_name = param.split(':')[0].split('=')[0].strip()
                    if param_name and param_name != 'self':
                        trace_params.append(f"{param_name}={param_name}")
        
        # Costruisci la chiamata trace_all
        if trace_params:
            params_str = ', '.join(trace_params[:3])  # Max 3 parametri
            trace_call = f'{indent}trace_all("{function_name}", "ENTER", {params_str})\n'
        else:
            trace_call = f'{indent}trace_all("{function_name}", "ENTER")\n'
        
        # Ricostruisci con tracing
        return route_decorator + '\n' + function_def + docstring + trace_call + indent + first_code_line
    
    # Applica le modifiche
    modified_content = re.sub(pattern, replace_route, content, flags=re.MULTILINE | re.DOTALL)
    
    if modifications == 0:
        print("  ⚠️  Nessuna rotta trovata da modificare")
        return
    
    # Scrivi il file modificato
    with open(server_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"✅ Completato! Aggiunte {modifications} chiamate trace_all")

if __name__ == "__main__":
    add_trace_to_remaining_routes()
