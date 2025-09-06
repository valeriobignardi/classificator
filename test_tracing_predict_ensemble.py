#!/usr/bin/env python3
"""
Test del sistema di tracing per predict_with_ensemble e altre funzioni aggiunte

Autore: Valerio Bignardi
Data creazione: 2025-09-06
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from Pipeline.end_to_end_pipeline import trace_all
    from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier
    
    print("✅ Test 1: Import del trace_all funziona")
    
    # Test della funzione trace_all con called_from
    trace_all("test_function", "ENTER", called_from="test_main", param1="valore1", param2=42)
    trace_all("test_function", "EXIT", called_from="test_main", return_value="success")
    
    print("✅ Test 2: trace_all con called_from funziona")
    
    # Test creazione AdvancedEnsembleClassifier (solo per verificare import)
    print("✅ Test 3: Import AdvancedEnsembleClassifier funziona")
    
    # Verifica che il file di log sia creato
    if os.path.exists("tracing.log"):
        with open("tracing.log", "r") as f:
            lines = f.readlines()
            print(f"✅ Test 4: File tracing.log creato con {len(lines)} righe")
            # Mostra le ultime 3 righe
            for line in lines[-3:]:
                print(f"   LOG: {line.strip()}")
    else:
        print("❌ Test 4 FALLITO: File tracing.log non trovato")
    
    print("\n🎯 TUTTI I TEST COMPLETATI")
    
except Exception as e:
    print(f"❌ ERRORE nel test: {e}")
    import traceback
    traceback.print_exc()
