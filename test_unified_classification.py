#!/usr/bin/env python3
"""
Script di test per la logica unificata di classificazione post-training

Autore: Valerio Bignardi  
Data di creazione: 2025-08-29
Storia degli aggiornamenti:
- 2025-08-29: Creazione iniziale per test logica unificata
"""

import sys
import os
import yaml
from datetime import datetime

# Aggiungi percorsi
sys.path.append('.')
sys.path.append('./Pipeline')

def test_pipeline_import():
    """
    Scopo: Testa l'importazione della pipeline modificata
    
    Output:
        - bool: True se import riuscito
        
    Ultimo aggiornamento: 2025-08-29
    """
    try:
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        print("✅ Import della pipeline riuscito")
        return True
    except Exception as e:
        print(f"❌ Errore import pipeline: {e}")
        return False

def test_mongo_reader_import():
    """
    Scopo: Testa l'importazione del MongoClassificationReader
    
    Output:
        - bool: True se import riuscito
        
    Ultimo aggiornamento: 2025-08-29
    """
    try:
        from mongo_classification_reader import MongoClassificationReader
        print("✅ Import del MongoClassificationReader riuscito")
        return True
    except Exception as e:
        print(f"❌ Errore import MongoReader: {e}")
        return False

def test_config_loading():
    """
    Scopo: Testa il caricamento della configurazione
    
    Output:
        - dict: Configurazione caricata o None
        
    Ultimo aggiornamento: 2025-08-29
    """
    try:
        with open('config.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        print("✅ Configurazione caricata correttamente")
        print(f"   - Tenant principale: {config.get('tenant_slug', 'NON TROVATO')}")
        return config
    except Exception as e:
        print(f"❌ Errore caricamento config: {e}")
        return None

def test_pipeline_initialization(config):
    """
    Scopo: Testa l'inizializzazione della pipeline
    
    Parametri input:
        - config: Configurazione YAML
        
    Output:
        - EndToEndPipeline: Istanza inizializzata o None
        
    Ultimo aggiornamento: 2025-08-29
    """
    try:
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        
        tenant_slug = config.get('tenant_slug', 'wopta')
        pipeline = EndToEndPipeline(config_path='config.yaml', tenant_slug=tenant_slug)
        
        print(f"✅ Pipeline inizializzata per tenant: {tenant_slug}")
        return pipeline
    except Exception as e:
        print(f"❌ Errore inizializzazione pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_classifica_e_salva_sessioni_signature(pipeline):
    """
    Scopo: Testa la signature della funzione modificata
    
    Parametri input:
        - pipeline: Istanza della pipeline
        
    Output:
        - bool: True se signature corretta
        
    Ultimo aggiornamento: 2025-08-29
    """
    try:
        import inspect
        
        # Ottieni signature della funzione
        sig = inspect.signature(pipeline.classifica_e_salva_sessioni)
        params = list(sig.parameters.keys())
        
        print("✅ Signature della funzione classifica_e_salva_sessioni:")
        print(f"   Parametri: {params}")
        
        # Verifica presenza del parametro force_review
        if 'force_review' in params:
            print("✅ Parametro force_review presente")
            
            # Verifica default value
            force_review_param = sig.parameters['force_review']
            if force_review_param.default == False:
                print("✅ Valore default force_review = False corretto")
                return True
            else:
                print(f"⚠️ Valore default force_review = {force_review_param.default}")
        else:
            print("❌ Parametro force_review mancante")
            
        return False
    except Exception as e:
        print(f"❌ Errore test signature: {e}")
        return False

def test_mongo_clear_method():
    """
    Scopo: Testa il metodo clear_tenant_collection
    
    Output:
        - bool: True se metodo disponibile
        
    Ultimo aggiornamento: 2025-08-29
    """
    try:
        from mongo_classification_reader import MongoClassificationReader
        
        mongo_reader = MongoClassificationReader()
        
        # Verifica presenza del metodo
        if hasattr(mongo_reader, 'clear_tenant_collection'):
            print("✅ Metodo clear_tenant_collection disponibile")
            
            # Testa signature
            import inspect
            sig = inspect.signature(mongo_reader.clear_tenant_collection)
            params = list(sig.parameters.keys())
            print(f"   Parametri: {params}")
            
            return True
        else:
            print("❌ Metodo clear_tenant_collection non trovato")
            return False
            
    except Exception as e:
        print(f"❌ Errore test mongo clear method: {e}")
        return False

def run_all_tests():
    """
    Scopo: Esegue tutti i test della logica unificata
    
    Output:
        - dict: Risultati dei test
        
    Ultimo aggiornamento: 2025-08-29
    """
    print("🧪 INIZIO TEST LOGICA UNIFICATA CLASSIFICAZIONE")
    print("=" * 60)
    
    results = {
        'pipeline_import': False,
        'mongo_import': False,
        'config_loading': False,
        'pipeline_init': False,
        'function_signature': False,
        'mongo_clear_method': False
    }
    
    # Test 1: Import pipeline
    print("\n1️⃣ TEST IMPORT PIPELINE")
    results['pipeline_import'] = test_pipeline_import()
    
    # Test 2: Import mongo reader  
    print("\n2️⃣ TEST IMPORT MONGO READER")
    results['mongo_import'] = test_mongo_reader_import()
    
    # Test 3: Caricamento config
    print("\n3️⃣ TEST CARICAMENTO CONFIGURAZIONE")
    config = test_config_loading()
    results['config_loading'] = config is not None
    
    if not config:
        print("❌ Impossibile proseguire senza configurazione")
        return results
    
    # Test 4: Inizializzazione pipeline
    print("\n4️⃣ TEST INIZIALIZZAZIONE PIPELINE")
    pipeline = test_pipeline_initialization(config)
    results['pipeline_init'] = pipeline is not None
    
    if not pipeline:
        print("❌ Impossibile proseguire senza pipeline")
        return results
    
    # Test 5: Signature funzione
    print("\n5️⃣ TEST SIGNATURE FUNZIONE")
    results['function_signature'] = test_classifica_e_salva_sessioni_signature(pipeline)
    
    # Test 6: Metodo clear MongoDB
    print("\n6️⃣ TEST METODO CLEAR MONGODB")
    results['mongo_clear_method'] = test_mongo_clear_method()
    
    # Riepilogo
    print("\n" + "=" * 60)
    print("📊 RIEPILOGO TEST")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 RISULTATO FINALE: {passed}/{total} test passati")
    
    if passed == total:
        print("🎉 TUTTI I TEST SUPERATI - LOGICA UNIFICATA PRONTA!")
    else:
        print("⚠️ Alcuni test falliti - Verificare implementazione")
    
    return results

if __name__ == "__main__":
    run_all_tests()
