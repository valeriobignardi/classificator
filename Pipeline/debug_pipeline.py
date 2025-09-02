#!/usr/bin/env python3
"""
Pipeline Debug Utility

Autore: Valerio Bignardi
Data creazione: 2025-09-02
Storia aggiornamenti:
- 2025-09-02: Creazione utility per debug dettagliato pipeline

Scopo: Fornisce funzioni di debug standardizzate per tracciare 
       l'esecuzione completa della pipeline di classificazione
"""

import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
import json


def debug_pipeline(function_name: str, action: str, data: Dict[str, Any] = None, 
                  level: str = "INFO") -> None:
    """
    Scopo: Log standardizzato per debug della pipeline
    
    Parametri input:
    - function_name: Nome della funzione che chiama il debug
    - action: Azione che sta per essere eseguita o completata
    - data: Dati da loggare (dizionario)
    - level: Livello di debug (INFO, WARNING, ERROR, CRITICAL)
    
    Output: None (stampa su console)
    
    Ultimo aggiornamento: 2025-09-02
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    # Emoji per livello
    level_emoji = {
        "INFO": "🔍",
        "WARNING": "⚠️",
        "ERROR": "❌", 
        "CRITICAL": "🚨",
        "SUCCESS": "✅",
        "ENTRY": "🚀",
        "EXIT": "🏁",
        "DECISION": "🎯"
    }
    
    emoji = level_emoji.get(level, "🔍")
    
    print(f"{emoji} [{timestamp}] [DEBUG-{function_name.upper()}] {action}")
    
    if data:
        for key, value in data.items():
            # Gestisci diversi tipi di dati
            if isinstance(value, (dict, list)):
                try:
                    value_str = json.dumps(value, indent=2, default=str)[:500]  # Limita output
                    if len(str(value)) > 500:
                        value_str += "... [TRUNCATED]"
                except:
                    value_str = str(value)[:500]
            else:
                value_str = str(value)
                
            print(f"    📋 {key}: {value_str}")


def debug_exception(function_name: str, exception: Exception, context: Dict[str, Any] = None) -> None:
    """
    Scopo: Log dettagliato per eccezioni con traceback completo
    
    Parametri input:
    - function_name: Nome della funzione dove è avvenuta l'eccezione
    - exception: L'eccezione catturata
    - context: Contesto aggiuntivo (parametri, stato, ecc.)
    
    Output: None (stampa su console)
    
    Ultimo aggiornamento: 2025-09-02
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    print(f"🚨 [{timestamp}] [EXCEPTION-{function_name.upper()}] {str(exception)}")
    print(f"🔍 [EXCEPTION-{function_name.upper()}] Type: {type(exception).__name__}")
    
    if context:
        print(f"📋 [EXCEPTION-{function_name.upper()}] Context:")
        for key, value in context.items():
            print(f"    {key}: {str(value)[:200]}")
    
    print(f"📚 [EXCEPTION-{function_name.upper()}] Traceback:")
    traceback.print_exc()
    print("-" * 80)


def debug_flow(function_name: str, flow_step: str, inputs: Dict[str, Any] = None, 
               outputs: Dict[str, Any] = None) -> None:
    """
    Scopo: Traccia il flusso della pipeline con input/output
    
    Parametri input:
    - function_name: Nome della funzione
    - flow_step: Descrizione del passo nel flusso
    - inputs: Parametri di input
    - outputs: Risultati di output
    
    Output: None (stampa su console)
    
    Ultimo aggiornamento: 2025-09-02
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    print(f"🌊 [{timestamp}] [FLOW-{function_name.upper()}] {flow_step}")
    
    if inputs:
        print(f"    📥 INPUTS:")
        for key, value in inputs.items():
            if isinstance(value, (list, dict)):
                summary = f"Type: {type(value).__name__}, Length: {len(value)}"
            else:
                summary = str(value)[:100]
            print(f"      {key}: {summary}")
    
    if outputs:
        print(f"    📤 OUTPUTS:")
        for key, value in outputs.items():
            if isinstance(value, (list, dict)):
                summary = f"Type: {type(value).__name__}, Length: {len(value)}"
            else:
                summary = str(value)[:100]
            print(f"      {key}: {summary}")


def debug_metadata(session_id: str, cluster_metadata: Optional[Dict[str, Any]], 
                  classified_by: str, function_name: str) -> None:
    """
    Scopo: Debug specifico per cluster_metadata e classificazione
    
    Parametri input:
    - session_id: ID della sessione
    - cluster_metadata: Metadati del cluster (può essere None)
    - classified_by: Metodo di classificazione usato
    - function_name: Funzione che sta salvando
    
    Output: None (stampa su console)
    
    Ultimo aggiornamento: 2025-09-02
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    if cluster_metadata is None:
        print(f"🚨 [{timestamp}] [METADATA-{function_name.upper()}] cluster_metadata=None per {session_id}")
        print(f"    ⚠️ classified_by: {classified_by}")
        print(f"    🎯 QUESTO CAUSERÀ LLM_STRUCTURED!")
    else:
        print(f"✅ [{timestamp}] [METADATA-{function_name.upper()}] cluster_metadata OK per {session_id}")
        print(f"    📋 cluster_id: {cluster_metadata.get('cluster_id', 'N/A')}")
        print(f"    🏷️ is_representative: {cluster_metadata.get('is_representative', False)}")
        print(f"    🔄 method: {classified_by}")


def debug_prediction(session_id: str, prediction: Dict[str, Any], source_function: str) -> None:
    """
    Scopo: Debug delle predizioni generate
    
    Parametri input:
    - session_id: ID della sessione
    - prediction: Predizione generata
    - source_function: Funzione che ha generato la predizione
    
    Output: None (stampa su console)
    
    Ultimo aggiornamento: 2025-09-02
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    print(f"🎯 [{timestamp}] [PREDICTION-{source_function.upper()}] Session {session_id}")
    print(f"    🏷️ predicted_label: {prediction.get('predicted_label', 'N/A')}")
    print(f"    📊 confidence: {prediction.get('confidence', 'N/A')}")
    print(f"    🔧 method: {prediction.get('method', 'N/A')}")
    print(f"    🆔 cluster_id: {prediction.get('cluster_id', 'N/A')}")
    
    if 'cluster_metadata' in prediction:
        print(f"    ✅ HAS cluster_metadata: {bool(prediction['cluster_metadata'])}")
    else:
        print(f"    ❌ NO cluster_metadata")
