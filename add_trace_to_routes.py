#!/usr/bin/env python3
"""
Script per aggiungere trace_all a tutte le rotte di server.py
Author: Valerio Bignardi
Date: 2024-12-19
"""

import re
import os

def add_trace_to_server_routes():
    """
    Aggiunge automaticamente trace_all() a tutte le rotte di server.py
    """
    
    server_file = "/home/ubuntu/classificatore/server.py"
    
    print(f"🚀 Inizio aggiunta tracing automatico a tutte le rotte di {server_file}")
    
    # Leggi il file
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern per trovare le rotte Flask che non hanno già trace_all
    route_pattern = r'(@app\.route\([^)]+\)\s*\n)(def\s+(\w+)\s*\([^)]*\):[^\n]*\n)([ ]*"""[^"]*?"""[ ]*\n)?([ ]*)(.*?)(\n[ ]*try:|\n[ ]*[^t][^r][^y].*?|\n[ ]*return|\n[ ]*if|\n[ ]*#)'
    
    def replace_route(match):
        """Sostituisce una definizione di rotta aggiungendo il tracing"""
        
        route_decorator = match.group(1)  # @app.route(...)
        function_def = match.group(2)     # def function_name(...):
        function_name = match.group(3)    # nome della funzione
        docstring = match.group(4) or ""  # docstring se presente
        indent = match.group(5)           # indentazione
        first_line = match.group(6)       # prima riga del codice
        terminator = match.group(7)       # terminatore (\n try: o altro)
        
        # Verifica se già ha trace_all
        if 'trace_all' in first_line:
            print(f"  ⏭️  {function_name} ha già trace_all - skipping")
            return match.group(0)
        
        print(f"  ✅ Aggiunto tracing a rotta: {function_name}")
        
        # Estrai parametri dalla firma della funzione
        param_match = re.search(r'\(([^)]*)\)', function_def)
        trace_params = []
        
        if param_match:
            params_str = param_match.group(1)
            # Pulisci i parametri (rimuovi type hints, default values)
            for param in params_str.split(','):
                param = param.strip()
                if param and param != 'self':
                    # Rimuovi type hints e default values
                    param_name = param.split(':')[0].split('=')[0].strip()
                    if param_name and param_name not in ['self']:
                        trace_params.append(f"{param_name}={param_name}")
        
        # Costruisci la chiamata a trace_all
        if trace_params:
            trace_params_str = ', '.join(trace_params[:3])  # Max 3 parametri
            trace_call = f'{indent}trace_all("{function_name}", "ENTER", {trace_params_str})\n'
        else:
            trace_call = f'{indent}trace_all("{function_name}", "ENTER")\n'
        
        # Ricostruisci la rotta con il tracing
        result = route_decorator + function_def + docstring + trace_call + indent + first_line + terminator
        
        return result
    
    # Applica il pattern replacement
    print("  🔄 Analisi e modifica delle rotte...")
    modified_content = re.sub(route_pattern, replace_route, content, flags=re.MULTILINE | re.DOTALL)
    
    # Controlla se ci sono stati cambiamenti
    if modified_content == content:
        print("  ⚠️  Nessuna modifica necessaria - tutte le rotte hanno già trace_all")
        return
    
    # Scrivi il file modificato
    with open(server_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"✅ Tracing aggiunto con successo a tutte le rotte di {server_file}")
    print("📝 Le seguenti modifiche sono state applicate:")
    print("   - Aggiunto import trace_all da end_to_end_pipeline")
    print("   - Aggiunto trace_all(\"function_name\", \"ENTER\", params) a ogni rotta")
    print("   - Parametri tracciati: client_name, tenant_id, case_id, ecc.")

if __name__ == "__main__":
    add_trace_to_server_routes()
