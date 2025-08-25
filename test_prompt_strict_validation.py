#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
TEST PROMPT STRICT VALIDATION - SISTEMA SENZA FALLBACK
=====================================================================
Autore: Sistema di Classificazione AI  
Data: 2025-08-24
Descrizione: Test per verificare il funzionamento del sistema strict
             senza fallback hardcoded
=====================================================================
"""

import sys
import os

# Aggiungi path per i moduli
sys.path.append(os.path.join(os.path.dirname(__file__), 'Classification'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Utils'))

from Classification.intelligent_classifier import IntelligentClassifier
from Utils.prompt_manager import PromptManager
import json

def test_prompt_validation_strict():
    """
    Test per validazione strict dei prompt
    """
    print("🧪 Test 1: Validazione STRICT prompt per tenant Humanitas")
    print("-" * 60)
    
    try:
        # Test con tenant Humanitas (dovrebbe avere i prompt configurati)
        classifier = IntelligentClassifier(
            config_path="config.yaml",
            client_name="humanitas",
            enable_logging=True
        )
        
        print("✅ Classificatore inizializzato con successo per tenant Humanitas")
        
        # Test classificazione
        test_text = "Mi serve un oculista che cura la retinopatia diabetica"
        
        print(f"\n📝 Test classificazione con testo: '{test_text}'")
        
        result = classifier.classify_with_motivation(test_text)
        
        print(f"✅ Classificazione completata:")
        print(f"   Label: {result.predicted_label}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Method: {result.method}")
        print(f"   Processing time: {result.processing_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test tenant Humanitas: {e}")
        return False

def test_missing_prompts_tenant():
    """
    Test per tenant senza prompt configurati (dovrebbe fallire)
    """
    print("\n🧪 Test 2: Tenant senza prompt configurati (dovrebbe FALLIRE)")
    print("-" * 60)
    
    try:
        # Test con tenant inesistente
        classifier = IntelligentClassifier(
            config_path="config.yaml",
            client_name="test_tenant_missing",
            enable_logging=True
        )
        
        print("❌ ERRORE: Classificatore non dovrebbe inizializzarsi per tenant senza prompt")
        return False
        
    except Exception as e:
        print(f"✅ CORRETTO: Sistema ha fallito come previsto per tenant senza prompt")
        print(f"   Errore: {str(e)[:100]}...")
        return True

def test_api_validation_endpoints():
    """
    Test per gli endpoint di validazione API
    """
    print("\n🧪 Test 3: API Validation Endpoints")
    print("-" * 60)
    
    try:
        from APIServer.prompt_validation_api import PromptValidationAPI
        
        api = PromptValidationAPI(config_path="config.yaml")
        
        # Test validazione tenant Humanitas
        print("📡 Test validazione API per tenant Humanitas...")
        result = api.validate_tenant_prompts("humanitas")
        
        if result['success'] and result['validation_result']['valid']:
            print("✅ API validazione: tenant Humanitas VALIDO")
        else:
            print("❌ API validazione: tenant Humanitas NON VALIDO")
            print(f"   Dettagli: {result}")
            return False
        
        # Test report prompt mancanti
        print("\n📊 Test report prompt mancanti...")
        report = api.get_missing_prompts_report("humanitas")
        
        if report['success']:
            summary = report['summary']
            print(f"✅ Report generato: {summary['available']}/{summary['total_required']} prompt disponibili")
            
            if summary['missing'] > 0:
                print(f"⚠️  Prompt mancanti: {summary['missing']}")
                for missing in report['missing_prompts']:
                    print(f"   - {missing['engine']}/{missing['prompt_type']}/{missing['prompt_name']}")
            else:
                print("✅ Tutti i prompt richiesti sono disponibili")
        else:
            print("❌ Errore generazione report")
            return False
        
        # Test tenant inesistente
        print("\n🔍 Test validazione tenant inesistente...")
        result_missing = api.validate_tenant_prompts("missing_tenant")
        
        if not result_missing['success']:
            print("✅ CORRETTO: Validazione fallita per tenant inesistente")
        else:
            print("❌ ERRORE: Validazione dovrebbe fallire per tenant inesistente")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test API: {e}")
        return False

def test_prompt_manager_strict():
    """
    Test per il PromptManager con modalità strict
    """
    print("\n🧪 Test 4: PromptManager Strict Mode")
    print("-" * 60)
    
    try:
        pm = PromptManager(config_path="config.yaml")
        
        # Test caricamento prompt strict esistente
        print("📥 Test caricamento prompt STRICT esistente...")
        prompt_content = pm.get_prompt_strict(
            tenant_id="humanitas",
            engine="LLM",
            prompt_type="SYSTEM", 
            prompt_name="intelligent_classifier_system"
        )
        
        if prompt_content and len(prompt_content) > 100:
            print(f"✅ Prompt caricato correttamente ({len(prompt_content)} caratteri)")
        else:
            print("❌ Errore caricamento prompt strict")
            return False
        
        # Test caricamento prompt strict inesistente (dovrebbe fallire)
        print("\n📥 Test caricamento prompt STRICT inesistente (dovrebbe FALLIRE)...")
        try:
            prompt_missing = pm.get_prompt_strict(
                tenant_id="humanitas",
                engine="LLM",
                prompt_type="SYSTEM",
                prompt_name="non_existent_prompt"
            )
            print("❌ ERRORE: Caricamento dovrebbe fallire per prompt inesistente")
            return False
            
        except Exception as e:
            print(f"✅ CORRETTO: Caricamento fallito come previsto")
            print(f"   Errore: {str(e)[:80]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test PromptManager strict: {e}")
        return False

def run_all_tests():
    """
    Esegue tutti i test di validazione
    """
    print("🎯 SISTEMA PROMPT STRICT VALIDATION - SUITE DI TEST")
    print("=" * 70)
    
    tests = [
        ("Validazione prompt tenant Humanitas", test_prompt_validation_strict),
        ("Tenant senza prompt (fallimento)", test_missing_prompts_tenant),
        ("API Validation Endpoints", test_api_validation_endpoints),
        ("PromptManager Strict Mode", test_prompt_manager_strict)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Errore critico in test '{test_name}': {e}")
            results.append((test_name, False))
    
    # Riassunto risultati
    print("\n" + "=" * 70)
    print("📋 RIASSUNTO RISULTATI TEST")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status:10} | {test_name}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 70)
    print(f"✅ Test passati: {passed}")
    print(f"❌ Test falliti: {failed}")
    print(f"📊 Totali: {len(results)}")
    
    if failed == 0:
        print("\n🎉 TUTTI I TEST SONO PASSATI! Sistema prompt strict operativo.")
        return True
    else:
        print(f"\n💥 {failed} test falliti! Sistema richiede correzioni.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
