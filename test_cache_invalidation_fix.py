#!/usr/bin/env python3
"""
File: test_cache_invalidation_fix.py
Autore: Sistema AI
Data Creazione: 2025-01-27
Ultima Modifica: 2025-01-27

Scopo:
Test per verificare la correzione del bug critico di invalidazione cache
nelle configurazioni AI. Verifica che i cambi di configurazione siano
immediatamente visibili senza dover riavviare il server.

Test del bug critico risolto:
- Configurazione viene letta e cachata
- Configurazione viene modificata 
- Cache deve essere completamente invalidata
- Rilettura deve mostrare la nuova configurazione
"""

import os
import sys
import time
from datetime import datetime

# Aggiungi path progetto
sys.path.append('/home/ubuntu/classificazione_discussioni_bck_23_08_2025')

from Database.database_ai_config_service import DatabaseAIConfigService

def test_cache_invalidation_fix():
    """
    Test principale per verificare correzione bug cache
    
    Scopo:
    Simula il scenario del bug originale e verifica che sia risolto
    
    Parametri:
    Nessuno
    
    Returns:
    bool: True se test passa, False se fallisce
    
    Ultima modifica: 2025-01-27
    """
    print("🧪 === TEST CORREZIONE BUG CACHE INVALIDATION ===\n")
    
    service = DatabaseAIConfigService()
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"  # Humanitas
    
    try:
        # FASE 1: Lettura iniziale (carica cache)
        print("📖 FASE 1: Lettura iniziale configurazione...")
        config1 = service.get_tenant_configuration(tenant_id)
        initial_engine = config1.get('embedding_engine', 'unknown')
        print(f"Configurazione iniziale: {initial_engine}")
        print(f"Cache timestamp: {service._cache_timestamp}")
        print(f"Cache keys: {list(service._tenant_configs_cache.keys())}")
        
        # FASE 2: Modifica configurazione
        print(f"\n🔄 FASE 2: Modifica configurazione embedding engine...")
        # Cambia da qualunque sia l'engine attuale all'altro
        new_engine = "openai_large" if initial_engine != "openai_large" else "bge_m3"
        print(f"Cambio da '{initial_engine}' a '{new_engine}'")
        
        result = service.set_embedding_engine(tenant_id, new_engine)
        if not result.get('success'):
            print(f"❌ Errore modifica configurazione: {result}")
            return False
        
        print(f"Cache timestamp dopo modifica: {service._cache_timestamp}")
        print(f"Cache keys dopo modifica: {list(service._tenant_configs_cache.keys())}")
        
        # FASE 3: Lettura immediata (deve mostrare nuova configurazione)
        print(f"\n✅ FASE 3: Lettura immediata per verificare cambio...")
        config2 = service.get_tenant_configuration(tenant_id)
        current_engine = config2.get('embedding_engine', 'unknown')
        
        print(f"Configurazione dopo modifica: {current_engine}")
        print(f"Source: {config2.get('source', 'unknown')}")
        
        # VERIFICA: La configurazione deve essere cambiata
        if current_engine == new_engine:
            print(f"✅ SUCCESS: Configurazione aggiornata correttamente!")
            print(f"   Prima: {initial_engine}")
            print(f"   Dopo:  {current_engine}")
            
            # FASE 4: Test cache refresh
            print(f"\n🔄 FASE 4: Test cache refresh dopo qualche secondo...")
            time.sleep(2)
            
            config3 = service.get_tenant_configuration(tenant_id)
            final_engine = config3.get('embedding_engine', 'unknown')
            
            if final_engine == new_engine:
                print(f"✅ SUCCESS: Cache refresh OK - configurazione persistente!")
                return True
            else:
                print(f"❌ FAIL: Cache refresh fallita")
                print(f"   Atteso: {new_engine}")
                print(f"   Trovato: {final_engine}")
                return False
            
        else:
            print(f"❌ FAIL: Configurazione NON aggiornata!")
            print(f"   Atteso: {new_engine}")
            print(f"   Trovato: {current_engine}")
            print(f"   BUG: Cache invalidation non funziona ancora!")
            return False
            
    except Exception as e:
        print(f"❌ ERRORE durante test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_tenants_isolation():
    """
    Test isolamento cache tra tenant multipli
    
    Scopo:
    Verifica che la modifica di un tenant non influenzi gli altri
    
    Ultima modifica: 2025-01-27
    """
    print(f"\n🏢 === TEST ISOLAMENTO MULTI-TENANT ===\n")
    
    service = DatabaseAIConfigService()
    
    # Due tenant diversi
    tenant1 = "015007d9-d413-11ef-86a5-96000228e7fe"  # Humanitas  
    tenant2 = "0c1231de-a58c-11ef-8c7f-96000228e7fe"  # Demo BluPantheon
    
    try:
        # Leggi configurazioni iniziali
        print("📖 Lettura configurazioni iniziali...")
        config1_before = service.get_tenant_configuration(tenant1)
        config2_before = service.get_tenant_configuration(tenant2)
        
        engine1_before = config1_before.get('embedding_engine')
        engine2_before = config2_before.get('embedding_engine')
        
        print(f"Tenant1 ({config1_before.get('tenant_name')}): {engine1_before}")
        print(f"Tenant2 ({config2_before.get('tenant_name')}): {engine2_before}")
        
        # Modifica solo tenant1
        new_engine1 = "openai_large" if engine1_before != "openai_large" else "bge_m3"
        print(f"\n🔄 Modifica solo Tenant1 a '{new_engine1}'...")
        
        result = service.set_embedding_engine(tenant1, new_engine1)
        if not result.get('success'):
            print(f"❌ Errore modifica: {result}")
            return False
        
        # Verifica modifiche
        print(f"\n✅ Verifica dopo modifica...")
        config1_after = service.get_tenant_configuration(tenant1)
        config2_after = service.get_tenant_configuration(tenant2)
        
        engine1_after = config1_after.get('embedding_engine')
        engine2_after = config2_after.get('embedding_engine')
        
        print(f"Tenant1 dopo: {engine1_after}")
        print(f"Tenant2 dopo: {engine2_after}")
        
        # Tenant1 deve essere cambiato, Tenant2 deve rimanere uguale
        if engine1_after == new_engine1 and engine2_after == engine2_before:
            print(f"✅ SUCCESS: Isolamento multi-tenant OK!")
            return True
        else:
            print(f"❌ FAIL: Problemi isolamento multi-tenant")
            return False
            
    except Exception as e:
        print(f"❌ ERRORE test multi-tenant: {e}")
        return False

if __name__ == "__main__":
    print("🧪 INIZIO TEST SUITE CORREZIONE BUG CACHE\n")
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Cache invalidation fix
    if test_cache_invalidation_fix():
        success_count += 1
    
    # Test 2: Multi-tenant isolation  
    if test_multiple_tenants_isolation():
        success_count += 1
    
    # Risultato finale
    print(f"\n{'='*50}")
    print(f"🏁 RISULTATI TEST SUITE")
    print(f"{'='*50}")
    print(f"Test passati: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print(f"✅ TUTTI I TEST PASSATI - BUG CACHE RISOLTO!")
        sys.exit(0)
    else:
        print(f"❌ ALCUNI TEST FALLITI - BUG ANCORA PRESENTE")
        sys.exit(1)
