#!/usr/bin/env python3
"""
Test rapido per debugging ensemble classifier durante training supervisionato

Autore: Valerio Bignardi
Data: 2025-09-04
Storia: Creato per debug problema ML Prediction N/A
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier
from Utils.tenant import Tenant

def test_ensemble_debug():
    """
    Test dell'ensemble classifier per debugging
    
    Scopo della funzione: Verifica cosa accade con ML e LLM durante classificazione
    Parametri di input: None
    Parametri di output: None (stampa debug)
    Valori di ritorno: None
    Tracciamento aggiornamenti: 2025-09-04 - Creato per debug ML
    
    Autore: Valerio Bignardi
    Data: 2025-09-04
    """
    
    print("🧪 TEST ENSEMBLE CLASSIFIER DEBUG\n" + "="*50)
    
    try:
        # Crea tenant fittizio per test
        test_tenant = Tenant(
            tenant_id="test-uuid-123",
            tenant_name="Test Tenant", 
            tenant_slug="test",
            tenant_database="test_database",
            tenant_status=1
        )
        
        # Crea ensemble classifier
        print("🏗️ Creazione ensemble classifier...")
        ensemble = AdvancedEnsembleClassifier()
        
        print(f"📊 Ensemble inizializzato:")
        print(f"   LLM classifier: {ensemble.llm_classifier is not None}")
        print(f"   ML ensemble: {ensemble.ml_ensemble is not None}")
        print(f"   Weights: {ensemble.weights}")
        
        # Test con testo esempio
        test_text = "[UTENTE] Buongiorno, vorrei maggiori informazioni riguardo delle analisi fecali"
        
        print(f"\n🧪 Test classificazione con testo:")
        print(f"   '{test_text[:80]}...'")
        
        # Esegue predizione con debug
        result = ensemble.predict_with_ensemble(
            text=test_text,
            return_details=True
        )
        
        print(f"\n📊 RISULTATO FINALE:")
        print(f"   Predicted label: {result.get('predicted_label')}")
        print(f"   Confidence: {result.get('confidence')}")
        print(f"   Method: {result.get('method')}")
        print(f"   LLM available: {result.get('llm_available')}")
        print(f"   ML available: {result.get('ml_available')}")
        
        if 'llm_prediction' in result and result['llm_prediction']:
            llm_pred = result['llm_prediction']
            print(f"   LLM Prediction: '{llm_pred.get('predicted_label')}' (conf: {llm_pred.get('confidence'):.3f})")
        else:
            print(f"   LLM Prediction: None")
            
        if 'ml_prediction' in result and result['ml_prediction']:
            ml_pred = result['ml_prediction']
            print(f"   ML Prediction: '{ml_pred.get('predicted_label')}' (conf: {ml_pred.get('confidence'):.3f})")
        else:
            print(f"   ML Prediction: None")
        
        print("\n" + "="*50)
        print("✅ Test completato con successo")
        
    except Exception as e:
        print(f"❌ ERRORE NEL TEST: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ensemble_debug()
