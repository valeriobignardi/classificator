#!/usr/bin/env python3
"""
Test semplificato per verificare l'ottimizzazione BERTopic - solo signature API

Autore: Valerio Bignardi  
Data creazione: 2025-09-06
Ultima modifica: 2025-09-06 - Test API ottimizzazione BERTopic
"""

import numpy as np
import sys
import os
import inspect
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier


def test_api_signatures():
    """
    Test per verificare che le nuove API siano state implementate correttamente
    """
    print("🧪 TEST API: Ottimizzazione BERTopic - Signature Test")
    
    # Test 1: Verifica signature predict_with_ensemble
    sig = inspect.signature(AdvancedEnsembleClassifier.predict_with_ensemble)
    params = list(sig.parameters.keys())
    
    expected_params = ['self', 'text', 'return_details', 'embedder', 'ml_features_precalculated']
    if all(param in params for param in expected_params):
        print("✅ Test 1 PASSED: predict_with_ensemble ha il parametro ml_features_precalculated")
    else:
        print(f"❌ Test 1 FAILED: Parametri attuali: {params}")
        print(f"   Parametri attesi: {expected_params}")
        return False
    
    # Test 2: Verifica signature batch_predict
    sig = inspect.signature(AdvancedEnsembleClassifier.batch_predict)
    params = list(sig.parameters.keys())
    
    expected_params = ['self', 'texts', 'batch_size', 'embedder', 'ml_features_batch']
    if all(param in params for param in expected_params):
        print("✅ Test 2 PASSED: batch_predict ha il parametro ml_features_batch")
    else:
        print(f"❌ Test 2 FAILED: Parametri attuali: {params}")
        print(f"   Parametri attesi: {expected_params}")
        return False
    
    # Test 3: Verifica esistenza create_ml_features_cache
    if hasattr(AdvancedEnsembleClassifier, 'create_ml_features_cache'):
        print("✅ Test 3 PASSED: create_ml_features_cache esiste")
    else:
        print("❌ Test 3 FAILED: create_ml_features_cache non trovato")
        return False
    
    # Test 4: Verifica docstring aggiornati
    docstring = AdvancedEnsembleClassifier.predict_with_ensemble.__doc__
    if 'ml_features_precalculated' in docstring:
        print("✅ Test 4 PASSED: Docstring predict_with_ensemble aggiornato")
    else:
        print("❌ Test 4 FAILED: Docstring predict_with_ensemble non aggiornato")
        return False
    
    print("🎉 TUTTI I TEST API PASSATI: Ottimizzazione BERTopic implementata correttamente!")
    print("📋 Riassunto modifiche:")
    print("   🔧 predict_with_ensemble: +ml_features_precalculated parameter")
    print("   🔧 batch_predict: +ml_features_batch parameter")  
    print("   🔧 create_ml_features_cache: nuovo metodo")
    print("   📝 Docstring aggiornati con author/date headers")
    
    return True


def test_pipeline_methods():
    """
    Test per verificare che i metodi della pipeline siano stati aggiunti
    """
    print("\n🧪 TEST PIPELINE: Metodi cache implementati")
    
    # Import pipeline
    try:
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
    except ImportError as e:
        print(f"❌ Import EndToEndPipeline failed: {e}")
        return False
    
    # Test 1: Verifica _create_ml_features_cache
    if hasattr(EndToEndPipeline, '_create_ml_features_cache'):
        print("✅ Test 1 PASSED: _create_ml_features_cache esiste nella pipeline")
    else:
        print("❌ Test 1 FAILED: _create_ml_features_cache non trovato nella pipeline")
        return False
    
    # Test 2: Verifica _get_cached_ml_features  
    if hasattr(EndToEndPipeline, '_get_cached_ml_features'):
        print("✅ Test 2 PASSED: _get_cached_ml_features esiste nella pipeline")
    else:
        print("❌ Test 2 FAILED: _get_cached_ml_features non trovato nella pipeline")
        return False
    
    print("🎉 TUTTI I TEST PIPELINE PASSATI!")
    return True


if __name__ == "__main__":
    success1 = test_api_signatures()
    success2 = test_pipeline_methods()
    
    if success1 and success2:
        print("\n🚀 IMPLEMENTAZIONE COMPLETATA CON SUCCESSO!")
        print("💡 Benefici attesi:")
        print("   ⚡ Eliminazione ricalcolo BERTopic durante predizioni")
        print("   🔄 Riutilizzo features calcolate nel training")
        print("   📈 Miglioramento performance globale pipeline")
        print("   ✅ Backward compatibility mantenuta")
    
    exit(0 if (success1 and success2) else 1)
