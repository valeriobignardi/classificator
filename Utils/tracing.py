#!/usr/bin/env python3
"""
Modulo centralizzato per il sistema di tracing
Autore: Valerio Bignardi
Data: 2025-09-13
Scopo: Evitare import circolari centralizzando la funzione trace_all
"""

import yaml
import os
import json
from datetime import datetime
from typing import Any

# Import config_loader per caricare config.yaml con variabili ambiente
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config



def trace_all(function_name: str, action: str = "ENTER", called_from: str = None, **kwargs: Any) -> None:
    """
    Sistema di tracing completo per tracciare il flusso della pipeline
    
    Scopo della funzione: Tracciare ingresso, uscita ed errori di tutte le funzioni
    Parametri di input: function_name, action, called_from, **kwargs (parametri da tracciare)
    Parametri di output: None (scrive su file)
    Valori di ritorno: None
    Tracciamento aggiornamenti: 2025-09-06 - Valerio Bignardi - Sistema tracing pipeline
    
    Args:
        function_name (str): Nome della funzione da tracciare
        action (str): "ENTER", "EXIT", "ERROR"
        called_from (str): Nome della funzione chiamante (per tracciare chiamate annidate)
        **kwargs: Parametri da tracciare (input, return_value, exception, etc.)
        
    Autore: Valerio Bignardi
    Data: 2025-09-06
    """
    try:
        # Carica configurazione tracing dal config.yaml
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        if not os.path.exists(config_path):
            return  # Tracing disabilitato se config non esiste
            
        config = load_config()
            
        tracing_config = config.get('tracing', {})
        if not tracing_config.get('enabled', False):
            return  # Tracing disabilitato
            
        # Configurazioni tracing
        log_file = tracing_config.get('log_file', 'tracing.log')
        include_parameters = tracing_config.get('include_parameters', True)
        include_return_values = tracing_config.get('include_return_values', True)
        include_exceptions = tracing_config.get('include_exceptions', True)
        max_file_size_mb = tracing_config.get('max_file_size_mb', 100)
        
        # Path assoluto per il file di log
        log_path = os.path.join(os.path.dirname(__file__), '..', log_file)
        
        # Rotazione file se troppo grande
        if os.path.exists(log_path):
            file_size_mb = os.path.getsize(log_path) / (1024 * 1024)
            if file_size_mb > max_file_size_mb:
                backup_path = f"{log_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(log_path, backup_path)
        
        # Timestamp formattato
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Costruisci messaggio di tracing con called_from
        if called_from:
            message_parts = [f"[{timestamp}]", f"{action:>5}", "->", f"{called_from}::{function_name}"]
        else:
            message_parts = [f"[{timestamp}]", f"{action:>5}", "->", function_name]
        
        # Aggiungi parametri se richiesto
        if action == "ENTER" and include_parameters and kwargs:
            params_str = []
            for key, value in kwargs.items():
                try:
                    # Converti i parametri in stringa gestendo oggetti complessi
                    if isinstance(value, (dict, list)):
                        if len(str(value)) > 200:
                            value_str = f"{type(value).__name__}(size={len(value)})"
                        else:
                            value_str = json.dumps(value, default=str, ensure_ascii=False)[:200]
                    elif hasattr(value, '__len__') and len(str(value)) > 200:
                        value_str = f"{type(value).__name__}(len={len(value)})"
                    else:
                        value_str = str(value)[:200]
                    params_str.append(f"{key}={value_str}")
                except Exception:
                    params_str.append(f"{key}=<{type(value).__name__}>")
            
            if params_str:
                message_parts.append(f"({', '.join(params_str)})")
        
        # Aggiungi valore di ritorno se richiesto
        elif action == "EXIT" and include_return_values and 'return_value' in kwargs:
            try:
                return_val = kwargs['return_value']
                if isinstance(return_val, (dict, list)):
                    if len(str(return_val)) > 300:
                        return_str = f"{type(return_val).__name__}(size={len(return_val)})"
                    else:
                        return_str = json.dumps(return_val, default=str, ensure_ascii=False)[:300]
                elif hasattr(return_val, '__len__') and len(str(return_val)) > 300:
                    return_str = f"{type(return_val).__name__}(len={len(return_val)})"
                else:
                    return_str = str(return_val)[:300]
                message_parts.append(f"RETURN: {return_str}")
            except Exception:
                message_parts.append(f"RETURN: <{type(kwargs['return_value']).__name__}>")
        
        # Aggiungi eccezione se richiesto
        elif action == "ERROR" and include_exceptions and 'exception' in kwargs:
            try:
                exc = kwargs['exception']
                exc_str = f"{type(exc).__name__}: {str(exc)}"[:500]
                message_parts.append(f"EXCEPTION: {exc_str}")
            except Exception:
                message_parts.append(f"EXCEPTION: <{type(kwargs['exception']).__name__}>")
        
        # Scrivi nel file di log
        log_message = " ".join(message_parts) + "\n"
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_message)
            
    except Exception as e:
        # Fallback silenzioso se il tracing fallisce
        # Non vogliamo che errori di tracing interrompano la pipeline
        pass


def create_trace_fallback():
    """
    Crea una funzione trace_all fallback per casi di emergenza
    
    Returns:
        function: Funzione trace_all semplificata
    """
    def trace_all_fallback(function_name: str, action: str = "ENTER", called_from: str = None, **kwargs: Any) -> None:
        """Funzione trace_all semplificata per fallback"""
        if action == "ERROR" and 'exception' in kwargs:
            print(f"🔍 TRACE {action}: {function_name} - ERROR: {kwargs['exception']}")
        else:
            print(f"🔍 TRACE {action}: {function_name}")
    
    return trace_all_fallback
