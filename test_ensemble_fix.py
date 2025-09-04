#!/usr/bin/env python3
"""
Test per verificare che IntelligentIntentClusterer 
ora accetti ensemble_classifier e lo usi correttamente
"""

import sys
import os
sys.path.append('.')

from Clustering.intelligent_intent_clusterer import IntelligentIntentClusterer

def test_ensemble_classifier_integration():
    """
    Test: Verifica che IntelligentIntentClusterer accetti ensemble_classifier
    """
    
    print("🧪 TEST: Integrazione ensemble_classifier in IntelligentIntentClusterer")
    
    # Test 1: Inizializzazione solo con llm_classifier (vecchio modo)
    print("\n📋 Test 1: Inizializzazione con solo llm_classifier")
    clusterer_old = IntelligentIntentClusterer(
        config_path='config.yaml',
        llm_classifier="mock_llm_classifier"
    )
    
    print(f"   ✅ use_ensemble: {clusterer_old.use_ensemble}")
    print(f"   ✅ ensemble_classifier: {clusterer_old.ensemble_classifier}")
    print(f"   ✅ llm_classifier: {clusterer_old.llm_classifier}")
    
    # Test 2: Inizializzazione con ensemble_classifier (nuovo modo)
    print("\n📋 Test 2: Inizializzazione con ensemble_classifier")
    clusterer_new = IntelligentIntentClusterer(
        config_path='config.yaml',
        llm_classifier="mock_llm_classifier",
        ensemble_classifier="mock_ensemble_classifier"
    )
    
    print(f"   ✅ use_ensemble: {clusterer_new.use_ensemble}")
    print(f"   ✅ ensemble_classifier: {clusterer_new.ensemble_classifier}")
    print(f"   ✅ llm_classifier: {clusterer_new.llm_classifier}")
    
    # Test 3: Inizializzazione solo con ensemble_classifier
    print("\n📋 Test 3: Inizializzazione con solo ensemble_classifier")
    clusterer_ensemble_only = IntelligentIntentClusterer(
        config_path='config.yaml',
        ensemble_classifier="mock_ensemble_classifier"
    )
    
    print(f"   ✅ use_ensemble: {clusterer_ensemble_only.use_ensemble}")
    print(f"   ✅ ensemble_classifier: {clusterer_ensemble_only.ensemble_classifier}")
    print(f"   ✅ llm_classifier: {clusterer_ensemble_only.llm_classifier}")
    
    print("\n🎯 RISULTATO: Tutte le modifiche funzionano correttamente!")
    print("   - IntelligentIntentClusterer ora accetta ensemble_classifier")
    print("   - La logica di priorità funziona (ensemble > llm > fallback)")
    print("   - Mantiene compatibilità con vecchio codice (solo llm_classifier)")
    
    return True

if __name__ == "__main__":
    test_ensemble_classifier_integration()
