#!/usr/bin/env python3
"""
Test del sistema di debug integrato
Verifica il funzionamento di LLM e ML debug con output graficamente strutturato
"""

import os
import sys
import yaml
import time
import numpy as np
from datetime import datetime

# Import relativi al nostro sistema
sys.path.append('.')
from Classification.intelligent_classifier import IntelligentClassifier
from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier
from EmbeddingEngine.labse_embedder import LaBSEEmbedder


def test_debug_configuration():
    """Test configurazione debug da config.yaml"""
    print("🔧 Testing Debug Configuration...")
    
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Verifica sezione debug
    debug_config = config.get('debug', {})
    llm_debug = debug_config.get('llm_debug', {})
    ml_debug = debug_config.get('ml_debug', {})
    
    print(f"   ✅ Debug section exists: {bool(debug_config)}")
    print(f"   ✅ LLM Debug enabled: {llm_debug.get('enabled', False)}")
    print(f"   ✅ ML Debug enabled: {ml_debug.get('enabled', False)}")
    print(f"   ✅ Detail level: {llm_debug.get('detail_level', 'basic')}")
    
    return llm_debug.get('enabled', False), ml_debug.get('enabled', False)


def test_llm_debug():
    """Test debug LLM con IntelligentClassifier"""
    print("\n🧠 Testing LLM Debug System...")
    
    try:
        # Inizializza classifier con debug
        classifier = IntelligentClassifier(
            model_name="mistral:7b",
            enable_logging=True,
            config_path="config.yaml"
        )
        
        # Test text
        test_text = "Buongiorno, vorrei prenotare una visita cardiologica per mia madre di 75 anni"
        
        print(f"   🔍 Testing classification with debug...")
        print(f"   📝 Text: {test_text[:50]}...")
        
        # Classifica con debug attivo
        result = classifier.classify_with_motivation(test_text)
        
        print(f"   ✅ Result: {result.predicted_label} (confidence: {result.confidence:.3f})")
        print(f"   ⏱️ Processing time: {result.processing_time:.3f}s")
        print(f"   📊 Method: {result.method}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ LLM Debug test failed: {e}")
        return False


def test_ml_debug():
    """Test debug ML Ensemble"""
    print("\n🤖 Testing ML Debug System...")
    
    try:
        # Inizializza embedder
        embedder = LaBSEEmbedder()
        
        # Inizializza ensemble classifier con debug
        ensemble = AdvancedEnsembleClassifier(
            config_path="config.yaml"
        )
        
        # Crea dati di training minimali per test
        print("   📚 Creating minimal training data...")
        X_train = np.random.rand(20, 768)  # 20 samples, 768 features (simulazione embedding)
        y_train = np.array(['prenotazione'] * 10 + ['altro'] * 10)
        
        # Test training con debug
        print("   🎓 Testing training with debug...")
        training_result = ensemble.train_ml_ensemble(X_train, y_train)
        print(f"   ✅ Training completed: accuracy {training_result['training_accuracy']:.3f}")
        
        # Test prediction con debug
        test_text = "Vorrei prenotare una visita"
        print(f"   🔍 Testing prediction with debug...")
        print(f"   📝 Text: {test_text}")
        
        prediction_result = ensemble.predict_with_ensemble(test_text, embedder=embedder)
        
        print(f"   ✅ Prediction: {prediction_result['predicted_label']} (confidence: {prediction_result['confidence']:.3f})")
        print(f"   📊 Method: {prediction_result['method']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ML Debug test failed: {e}")
        return False


def test_ensemble_debug():
    """Test debug ensemble completo"""
    print("\n🔗 Testing Complete Ensemble Debug...")
    
    try:
        # Setup completo
        embedder = LaBSEEmbedder()
        ensemble = AdvancedEnsembleClassifier(
            config_path="config.yaml"
        )
        
        # Training veloce
        X_train = np.random.rand(30, 768)
        y_train = np.array(['prenotazione'] * 15 + ['altro'] * 15)
        ensemble.train_ml_ensemble(X_train, y_train)
        
        # Test con LLM e ML insieme
        test_cases = [
            "Buongiorno, vorrei prenotare una visita cardiologica",
            "Salve, dove si trova il reparto di radiologia?",
            "Ho bisogno di informazioni sui tempi di attesa"
        ]
        
        for i, text in enumerate(test_cases, 1):
            print(f"   📋 Test case {i}: {text[:40]}...")
            
            result = ensemble.predict_with_ensemble(
                text, 
                return_details=True, 
                embedder=embedder
            )
            
            print(f"       🎯 Result: {result['predicted_label']} ({result['method']})")
            print(f"       📊 Confidence: {result['confidence']:.3f}")
            
            time.sleep(0.5)  # Small delay for readability
        
        return True
        
    except Exception as e:
        print(f"   ❌ Ensemble Debug test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 DEBUG SYSTEM INTEGRATION TEST")
    print("=" * 50)
    
    # Test configurazione
    llm_debug_enabled, ml_debug_enabled = test_debug_configuration()
    
    if not (llm_debug_enabled or ml_debug_enabled):
        print("\n⚠️ Debug is disabled in config.yaml")
        print("   Enable debug.llm_debug.enabled or debug.ml_debug.enabled to see debug output")
    
    # Test dei componenti
    tests = [
        ("LLM Debug", test_llm_debug),
        ("ML Debug", test_ml_debug), 
        ("Ensemble Debug", test_ensemble_debug)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All debug systems working correctly!")
    else:
        print("⚠️ Some debug systems need attention")
    
    print("\n💡 To see debug output:")
    print("   1. Enable debug in config.yaml")
    print("   2. Set detail_level to 'detailed' or 'verbose'")
    print("   3. Run pipeline_completa_da_zero.py or server.py")


if __name__ == "__main__":
    main()
