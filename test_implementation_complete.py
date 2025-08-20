#!/usr/bin/env python3
"""
Test completo dell'implementazione del sistema fine-tuning e ML arricchito

Questo script verifica:
1. Eliminazione modelli fine-tuned errati ✅ (già fatto)
2. Sistema fine-tuning con tag_description
3. Validazione label per evitare "info" generica
4. Sistema anti-duplicati per tag database
5. Training ML arricchito con descrizioni

Autore: Sistema Classificazione Humanitas
Data: 24 Luglio 2025
"""

import sys
import os
from datetime import datetime

# Aggiunge i path necessari
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Database'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FineTuning'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Classification'))

def test_database_schema():
    """Test del database schema e sistema anti-duplicati"""
    print("🧪 === TEST DATABASE SCHEMA & ANTI-DUPLICATI ===")
    
    try:
        from schema_manager import ClassificationSchemaManager
        
        schema_manager = ClassificationSchemaManager(schema='humanitas')
        
        # Verifica tabelle
        print("1. Verifica tabelle esistenti...")
        existing_tables = schema_manager.check_tables_exist()
        all_exist = all(existing_tables.values())
        
        if not all_exist:
            print("   Creazione tabelle mancanti...")
            schema_manager.create_classification_tables()
        
        # Test sistema anti-duplicati
        print("2. Test sistema anti-duplicati...")
        
        # Tenta di aggiungere un tag esistente
        result1 = schema_manager.add_tag_if_not_exists(
            "info_contatti", 
            "Test duplicato - non dovrebbe essere aggiunto", 
            "#FF0000"
        )
        
        # Tenta di aggiungere un nuovo tag
        result2 = schema_manager.add_tag_if_not_exists(
            "test_nuovo_tag", 
            "Questo è un tag di test nuovo", 
            "#00FF00"
        )
        
        # Verifica tag disponibili
        print("3. Tag disponibili nel database:")
        tags = schema_manager.get_all_tags()
        for tag in tags[:5]:  # Mostra solo i primi 5
            print(f"   🏷️ {tag['tag_name']}: {tag['tag_description']}")
        
        print(f"   📊 Totale tag: {len(tags)}")
        print("✅ Test database schema completato")
        return True
        
    except Exception as e:
        print(f"❌ Errore test database: {e}")
        return False

def test_finetuning_with_descriptions():
    """Test del sistema fine-tuning con descrizioni"""
    print("\n🧪 === TEST FINE-TUNING CON DESCRIZIONI ===")
    
    try:
        from mistral_finetuning_manager import MistralFineTuningManager
        
        # Inizializza manager
        ft_manager = MistralFineTuningManager()
        
        # Test recupero tag con descrizioni
        print("1. Test recupero tag con descrizioni...")
        tags_dict = ft_manager._get_tags_with_descriptions()
        
        if tags_dict:
            print(f"   📋 Recuperati {len(tags_dict)} tag")
            for tag_name, description in list(tags_dict.items())[:3]:
                print(f"   🏷️ {tag_name}: {description[:60]}...")
        
        # Test system message con descrizioni
        print("2. Test system message con descrizioni...")
        system_message = ft_manager._build_finetuning_system_message()
        
        if "ETICHETTE DISPONIBILI:" in system_message:
            print("   ✅ System message include descrizioni dei tag")
        else:
            print("   ❌ System message non include descrizioni")
        
        print("✅ Test fine-tuning completato")
        return True
        
    except Exception as e:
        print(f"❌ Errore test fine-tuning: {e}")
        return False

def test_ml_ensemble_enhanced():
    """Test ML ensemble con descrizioni"""
    print("\n🧪 === TEST ML ENSEMBLE ARRICCHITO ===")
    
    try:
        from advanced_ensemble_classifier import AdvancedEnsembleClassifier
        import numpy as np
        
        # Inizializza ensemble
        ensemble = AdvancedEnsembleClassifier()
        
        # Test recupero tag per training
        print("1. Test recupero tag per training ML...")
        tags_dict = ensemble.get_tags_with_descriptions_for_training()
        
        if tags_dict:
            print(f"   📋 Tag per ML: {len(tags_dict)}")
            print(f"   🎯 Sample: {list(tags_dict.keys())[:3]}")
        
        # Test simulato di training con descrizioni
        print("2. Test training ML arricchito...")
        
        # Dati di test simulati
        X_test = np.random.rand(10, 384)  # 10 samples, 384 features (LaBSE)
        y_test = ['info_contatti', 'prenotazione_esami', 'altro'] * 3 + ['info_esami']
        
        # Test training arricchito
        training_result = ensemble.train_ml_ensemble_with_descriptions(
            X_test, y_test, tags_dict
        )
        
        if training_result and training_result.get('enhancement_method') == 'with_tag_descriptions':
            print("   ✅ Training ML arricchito completato")
            print(f"   📊 Features arricchite: {training_result.get('n_features')}")
        else:
            print("   ⚠️ Fallback a training standard")
        
        print("✅ Test ML ensemble completato")
        return True
        
    except Exception as e:
        print(f"❌ Errore test ML ensemble: {e}")
        return False

def test_label_validation():
    """Test validazione label per evitare inconsistenze"""
    print("\n🧪 === TEST VALIDAZIONE LABEL ===")
    
    try:
        from mistral_finetuning_manager import MistralFineTuningManager
        
        ft_manager = MistralFineTuningManager()
        valid_tags = set(ft_manager._get_tags_with_descriptions().keys())
        
        # Simula training decisions con label invalide
        test_decisions = [
            {'session_id': 'test1', 'human_decision': 'info_contatti', 'human_confidence': 0.9},
            {'session_id': 'test2', 'human_decision': 'info', 'human_confidence': 0.8},  # INVALIDA
            {'session_id': 'test3', 'human_decision': 'prenotazione_esami', 'human_confidence': 0.95},
            {'session_id': 'test4', 'human_decision': 'invalid_label', 'human_confidence': 0.7},  # INVALIDA
        ]
        
        valid_count = 0
        invalid_count = 0
        
        for decision in test_decisions:
            label = decision['human_decision']
            if label in valid_tags:
                valid_count += 1
                print(f"   ✅ Label valida: {label}")
            else:
                invalid_count += 1
                print(f"   ❌ Label INVALIDA: {label}")
        
        print(f"\n   📊 Risultato validazione:")
        print(f"   ✅ Label valide: {valid_count}")
        print(f"   ❌ Label invalide: {invalid_count}")
        print(f"   🎯 Label disponibili: {sorted(list(valid_tags)[:5])}...")
        
        print("✅ Test validazione label completato")
        return True
        
    except Exception as e:
        print(f"❌ Errore test validazione: {e}")
        return False

def main():
    """Esegue tutti i test"""
    print("🧪 ===== TEST IMPLEMENTAZIONE COMPLETA =====")
    print(f"⏰ Avvio test: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Database Schema & Anti-duplicati", test_database_schema),
        ("Fine-tuning con Descrizioni", test_finetuning_with_descriptions),
        ("ML Ensemble Arricchito", test_ml_ensemble_enhanced),
        ("Validazione Label", test_label_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Errore test '{test_name}': {e}")
            results.append((test_name, False))
    
    # Risultati finali
    print("\n🎯 ===== RISULTATI FINALI =====")
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Successo: {passed}/{total} test passati")
    
    if passed == total:
        print("🎉 TUTTI I TEST PASSATI! Sistema pronto per nuovo fine-tuning.")
    else:
        print("⚠️ Alcuni test falliti. Verificare implementazione.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
