#!/usr/bin/env python3
"""
Test rapido per debuggare il problema ML nel training supervisionato

Autore: Valerio Bignardi
Data: 2025-09-04
"""

import sys
import os
sys.path.append('.')

from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier
from Utils.tenant import Tenant
import numpy as np

def test_ml_debug():
    """
    Test per verificare lo stato dell'ensemble ML
    
    Scopo della funzione: Simula il problema ML prediction = N/A
    Parametri di input: None
    Parametri di output: None (stampa debug)
    Valori di ritorno: None
    Tracciamento aggiornamenti: 2025-09-04 - Creato per debug ML
    
    Autore: Valerio Bignardi
    Data: 2025-09-04
    """
    
    print("🧪 TEST DEBUG ML ENSEMBLE\n")
    
    # Simula tenant humanitas
    tenant = Tenant(
        tenant_id="16c222a9-f293-11ef-9315-96000228e7fe",
        tenant_name="Humanitas", 
        tenant_slug="humanitas",
        tenant_database="humanitas_16c222a9",
        tenant_status=1
    )
    
    try:
        # Crea ensemble classifier (come nella pipeline)
        print("🎯 Creazione AdvancedEnsembleClassifier...")
        ensemble = AdvancedEnsembleClassifier(
            client_name="humanitas",
            confidence_threshold=0.7,
            adaptive_weights=True,
            performance_tracking=True
        )
        
        print(f"✅ Ensemble creato")
        print(f"   📊 ML Ensemble disponibile: {ensemble.ml_ensemble is not None}")
        print(f"   📊 LLM Classifier disponibile: {ensemble.llm_classifier is not None}")
        
        # Test predict_with_ensemble (come nell'interfaccia)
        test_text = "Vorrei prenotare una visita cardiologica"
        test_embedding = np.random.rand(768)  # Simula embedding
        
        print(f"\n🔍 Test predict_with_ensemble...")
        print(f"   📝 Testo: '{test_text}'")
        
        result = ensemble.predict_with_ensemble(
            test_text,
            return_details=True,
            embedder=None
        )
        
        print(f"\n📊 RISULTATO PREDICT_WITH_ENSEMBLE:")
        print(f"   🎯 Final label: {result.get('final_label', 'N/A')}")
        print(f"   📈 Final confidence: {result.get('final_confidence', 0.0):.3f}")
        print(f"   🤖 ML prediction: {result.get('ml_prediction', 'N/A')}")
        print(f"   📈 ML confidence: {result.get('ml_confidence', 0.0):.3f}")
        print(f"   🧠 LLM prediction: {result.get('llm_prediction', 'N/A')}")
        print(f"   📈 LLM confidence: {result.get('llm_confidence', 0.0):.3f}")
        print(f"   🔄 Method used: {result.get('method_used', 'N/A')}")
        print(f"   ⚠️ Errors: {result.get('errors', [])}")
        
        if result.get('ml_prediction') == 'N/A':
            print(f"\n❌ PROBLEMA CONFERMATO: ML prediction = N/A")
            print(f"   🔍 Possibili cause:")
            print(f"     - ML ensemble non allenato (ml_ensemble is None)")
            print(f"     - Errore nel predict_ml_only()")
            print(f"     - Problemi con i modelli ML interni")
        
        return result
        
    except Exception as e:
        print(f"💥 ERRORE NEL TEST: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 AVVIO TEST ML DEBUG\n" + "="*50)
    
    try:
        result = test_ml_debug()
        
        print("\n" + "="*50)
        if result and result.get('ml_prediction') != 'N/A':
            print("🎉 ML FUNZIONA CORRETTAMENTE!")
        else:
            print("💥 PROBLEMA ML CONFERMATO!")
            
    except Exception as e:
        print(f"💥 ERRORE CRITICO: {e}")
        import traceback
        traceback.print_exc()
