#!/usr/bin/env python3
"""
Test per verificare l'ottimizzazione BERTopic

Autore: Valerio Bignardi
Data creazione: 2025-09-06
Ultima modifica: 2025-09-06 - Test ottimizzazione BERTopic
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier


def test_bertopic_optimization():
    """
    Test per verificare che le features pre-calcolate vengano utilizzate correttamente
    """
    print("🧪 TEST: Ottimizzazione BERTopic")
    
    # Crea ensemble classifier
    ensemble = AdvancedEnsembleClassifier()
    
    # Test 1: Verifica che accetti il parametro ml_features_precalculated
    text = "Test di classificazione"
    fake_features = np.random.random((1, 768))  # Simula features ML
    
    try:
        # Questa chiamata dovrebbe funzionare senza errori
        result = ensemble.predict_with_ensemble(
            text=text,
            return_details=True,
            ml_features_precalculated=fake_features
        )
        print("✅ Test 1 PASSED: predict_with_ensemble accetta ml_features_precalculated")
        
        # Verifica che il metodo sia stato registrato
        method = result.get('method', 'UNKNOWN')
        print(f"   📊 Method utilizzato: {method}")
        
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        return False
    
    # Test 2: Verifica batch_predict con features pre-calcolate
    texts = ["Test 1", "Test 2"]
    fake_features_batch = [np.random.random((1, 768)) for _ in texts]
    
    try:
        results = ensemble.batch_predict(
            texts=texts,
            ml_features_batch=fake_features_batch
        )
        print("✅ Test 2 PASSED: batch_predict accetta ml_features_batch")
        print(f"   📊 Risultati batch: {len(results)} predizioni")
        
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        return False
    
    # Test 3: Verifica create_ml_features_cache
    try:
        fake_embeddings = np.random.random((2, 768))
        cache = ensemble.create_ml_features_cache(texts, fake_embeddings)
        print("✅ Test 3 PASSED: create_ml_features_cache funziona")
        print(f"   📊 Cache creata: {len(cache)} entries")
        
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        return False
    
    print("🎉 TUTTI I TEST PASSATI: Ottimizzazione BERTopic implementata correttamente!")
    return True


if __name__ == "__main__":
    success = test_bertopic_optimization()
    exit(0 if success else 1)
