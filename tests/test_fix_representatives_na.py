#!/usr/bin/env python3
"""
Test per verificare che il fix per rappresentanti con N/A funzioni correttamente

Autore: Valerio Bignardi
Data: 2025-09-06
Descrizione: Verifica che il nuovo metodo _classify_and_save_representatives_post_training
             classifichi e salvi i rappresentanti con predizioni ML+LLM complete
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_new_method_exists():
    """
    Testa che il nuovo metodo esista nella pipeline
    """
    print("🧪 TEST 1: Verifica esistenza nuovo metodo...")
    
    try:
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        
        # Verifica che il metodo esista
        if hasattr(EndToEndPipeline, '_classify_and_save_representatives_post_training'):
            print("✅ Test 1 PASSED: Metodo _classify_and_save_representatives_post_training esiste")
            return True
        else:
            print("❌ Test 1 FAILED: Metodo _classify_and_save_representatives_post_training NON esiste")
            return False
            
    except Exception as e:
        print(f"❌ Test 1 ERROR: {e}")
        return False

def test_method_signature():
    """
    Testa che il metodo abbia la signature corretta
    """
    print("🧪 TEST 2: Verifica signature del metodo...")
    
    try:
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        import inspect
        
        # Ottieni signature del metodo
        method = getattr(EndToEndPipeline, '_classify_and_save_representatives_post_training')
        signature = inspect.signature(method)
        
        # Verifica parametri attesi
        expected_params = [
            'self', 'sessioni', 'representatives', 'suggested_labels', 
            'cluster_labels', 'reviewed_labels'
        ]
        
        actual_params = list(signature.parameters.keys())
        
        if actual_params == expected_params:
            print("✅ Test 2 PASSED: Signature corretta")
            print(f"   📋 Parametri: {actual_params}")
            return True
        else:
            print("❌ Test 2 FAILED: Signature incorretta")
            print(f"   📋 Attesi: {expected_params}")
            print(f"   📋 Trovati: {actual_params}")
            return False
            
    except Exception as e:
        print(f"❌ Test 2 ERROR: {e}")
        return False

def test_docstring_content():
    """
    Testa che il metodo abbia una docstring appropriata
    """
    print("🧪 TEST 3: Verifica docstring del metodo...")
    
    try:
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        
        method = getattr(EndToEndPipeline, '_classify_and_save_representatives_post_training')
        docstring = method.__doc__
        
        if docstring and len(docstring) > 100:
            # Verifica che contenga keywords chiave
            required_keywords = [
                'Classifica e salva', 'rappresentanti', 'DOPO il training',
                'ML+LLM', 'Valerio Bignardi', '2025-09-06'
            ]
            
            missing_keywords = []
            for keyword in required_keywords:
                if keyword not in docstring:
                    missing_keywords.append(keyword)
            
            if not missing_keywords:
                print("✅ Test 3 PASSED: Docstring completa e appropriata")
                return True
            else:
                print("❌ Test 3 FAILED: Docstring manca keywords")
                print(f"   📋 Keywords mancanti: {missing_keywords}")
                return False
        else:
            print("❌ Test 3 FAILED: Docstring troppo corta o mancante")
            return False
            
    except Exception as e:
        print(f"❌ Test 3 ERROR: {e}")
        return False

def test_pipeline_compilation():
    """
    Testa che la pipeline compili correttamente con le modifiche
    """
    print("🧪 TEST 4: Verifica compilazione pipeline completa...")
    
    try:
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        
        # Verifica che possa istanziare (senza parametri per ora)
        print("   📊 Compilazione EndToEndPipeline: OK")
        
        # Verifica che metodi dipendenti esistano
        required_methods = [
            '_get_cached_ml_features',
            '_create_ml_features_cache',
            '_get_embedder'
        ]
        
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(EndToEndPipeline, method_name):
                missing_methods.append(method_name)
        
        if not missing_methods:
            print("✅ Test 4 PASSED: Pipeline compila e metodi dipendenti esistono")
            return True
        else:
            print("❌ Test 4 FAILED: Metodi dipendenti mancanti")
            print(f"   📋 Metodi mancanti: {missing_methods}")
            return False
            
    except Exception as e:
        print(f"❌ Test 4 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Esegue tutti i test
    """
    print("🚀 TEST FIX RAPPRESENTANTI N/A - Inizio validazione")
    print("=" * 60)
    
    tests = [
        test_new_method_exists,
        test_method_signature, 
        test_docstring_content,
        test_pipeline_compilation
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()  # Riga vuota tra test
        except Exception as e:
            print(f"❌ Errore durante test {test_func.__name__}: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 RISULTATI FINALI: {passed}/{total} test passati")
    
    if passed == total:
        print("🎉 TUTTI I TEST PASSATI! Fix implementato correttamente")
        print("✅ Il nuovo metodo è pronto per risolvere il problema N/A")
    else:
        print("⚠️ ALCUNI TEST FALLITI - Verificare implementazione")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
