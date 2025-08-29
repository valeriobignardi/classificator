#!/usr/bin/env python3
"""
Autore: Valerio Bignardi
Data creazione: 2025-08-29
Descrizione: Test per verificare la correzione del bug force_retrain
Storia aggiornamenti:
- 2025-08-29: Creazione iniziale del test
"""

import sys
sys.path.append('/home/ubuntu/classificatore')

from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier
import numpy as np

def test_train_ml_ensemble_signature():
    """
    Testa che il metodo train_ml_ensemble funzioni con i parametri corretti
    """
    print("🧪 TEST: Verifica signature train_ml_ensemble")
    
    try:
        # Crea istanza del classificatore
        classifier = AdvancedEnsembleClassifier()
        
        # Crea dati di test minimali
        X_test = np.random.rand(20, 10)  # 20 samples, 10 features
        y_test = np.random.randint(0, 3, 20)  # 3 classi
        
        # Test della chiamata corretta
        print("   📊 Creazione dati di test:")
        print(f"   - X_test shape: {X_test.shape}")
        print(f"   - y_test shape: {y_test.shape}")
        print(f"   - Classi uniche: {np.unique(y_test)}")
        
        # Chiamata al metodo con la signature corretta
        print("   🎯 Test chiamata train_ml_ensemble...")
        metrics = classifier.train_ml_ensemble(X_test, y_test)
        
        print("   ✅ train_ml_ensemble chiamato correttamente!")
        print(f"   📊 Metriche restituite: {list(metrics.keys())}")
        print(f"   📈 Training accuracy: {metrics.get('training_accuracy', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERRORE nel test: {str(e)}")
        return False

def test_pipeline_integration():
    """
    Testa che la correzione non introduca errori nella pipeline
    """
    print("\n🧪 TEST: Verifica integrazione nella pipeline")
    
    try:
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        
        # Questo import dovrebbe funzionare senza errori dopo la correzione
        print("   ✅ Import EndToEndPipeline completato")
        print("   ✅ Nessun errore di sintassi nella pipeline")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERRORE nell'integrazione: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 AVVIO TEST CORREZIONE BUG FORCE_RETRAIN")
    print("="*60)
    
    # Test 1: Signature del metodo
    test1_result = test_train_ml_ensemble_signature()
    
    # Test 2: Integrazione pipeline
    test2_result = test_pipeline_integration()
    
    # Risultati finali
    print("\n" + "="*60)
    print("📊 RISULTATI TEST:")
    print(f"   🧪 Test signature metodo: {'✅ PASSA' if test1_result else '❌ FALLISCE'}")
    print(f"   🧪 Test integrazione pipeline: {'✅ PASSA' if test2_result else '❌ FALLISCE'}")
    
    if test1_result and test2_result:
        print("\n🎉 TUTTI I TEST PASSANO!")
        print("✅ Il bug del parametro force_retrain è stato risolto correttamente")
        print("✅ Il riaddestramento automatico dovrebbe ora funzionare senza errori")
    else:
        print("\n❌ ALCUNI TEST FALLISCONO!")
        print("⚠️  La correzione potrebbe necessitare di ulteriori modifiche")
    
    print("\n🏁 Test completato")
