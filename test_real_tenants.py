#!/usr/bin/env python3
"""
============================================================================
Test con Tenant Reali Esistenti
============================================================================

Autore: Valerio Bignardi
Data creazione: 2025-01-31
Ultima modifica: 2025-01-31

Descrizione:
    Test del flusso React → Database usando i tenant reali esistenti
    per verificare se il problema è solo nei tenant fake che stavamo usando.

============================================================================
"""

import sys
import os
import json
sys.path.append('/home/ubuntu/classificatore')
sys.path.append('/home/ubuntu/classificatore/Database')
sys.path.append('/home/ubuntu/classificatore/Services')

from Services.llm_configuration_service import LLMConfigurationService
from Database.database_ai_config_service import DatabaseAIConfigService


def test_existing_tenant_flow(tenant_id: str):
    """
    Test con tenant reale esistente
    
    Args:
        tenant_id: ID del tenant esistente
        
    Data ultima modifica: 2025-01-31
    """
    print(f"\n🏢 TESTING TENANT ESISTENTE: {tenant_id}")
    print("=" * 80)
    
    # Test 1: Verifica configurazione attuale
    print("🔍 TEST 1: Lettura configurazione attuale")
    print("-" * 50)
    
    db_service = DatabaseAIConfigService()
    current_config = db_service.get_tenant_configuration(tenant_id)
    
    if current_config:
        print(f"✅ Tenant trovato nel database!")
        print(f"   LLM Engine: {current_config.get('llm_engine')}")
        print(f"   Embedding Engine: {current_config.get('embedding_engine')}")
        print(f"   Is Active: {current_config.get('is_active')}")
        
        if current_config.get('llm_config'):
            print(f"   LLM Config attuale: {len(str(current_config['llm_config']))} chars")
            print(f"   Keys: {list(current_config['llm_config'].keys())}")
        else:
            print(f"   LLM Config: NULL")
    else:
        print(f"❌ Tenant NON trovato!")
        return
    
    # Test 2: Aggiornamento tramite LLMConfigurationService
    print(f"\n🔄 TEST 2: Aggiornamento parametri LLM")
    print("-" * 50)
    
    llm_service = LLMConfigurationService()
    
    # Parametri di test
    test_parameters = {
        'tokenization': {
            'max_tokens': 4200,
            'temperature': 0.85
        },
        'generation': {
            'max_tokens': 2200,
            'temperature': 0.65,
            'top_k': 55,
            'top_p': 0.97
        },
        'connection': {
            'timeout': 450,
            'url': 'http://localhost:11434'
        },
        'test_metadata': {
            'test_type': 'existing_tenant_flow',
            'timestamp': '2025-01-31T14:00:00',
            'original_tenant': tenant_id[:8]
        }
    }
    
    print(f"📤 Aggiornamento parametri...")
    result = llm_service.update_tenant_parameters(
        tenant_id=tenant_id,
        parameters=test_parameters,
        model_name='gpt-4o'
    )
    
    print(f"✅ Risultato aggiornamento:")
    print(f"   Success: {result.get('success')}")
    print(f"   Saved to: {result.get('saved_to')}")
    print(f"   Message: {result.get('message')}")
    if not result.get('success'):
        print(f"   Error: {result.get('error')}")
    
    # Test 3: Verifica persistenza
    if result.get('success'):
        print(f"\n🔍 TEST 3: Verifica persistenza")
        print("-" * 50)
        
        # Re-read from database
        updated_config = db_service.get_tenant_configuration(tenant_id, force_no_cache=True)
        
        if updated_config and updated_config.get('llm_config'):
            saved_config = updated_config['llm_config']
            
            if 'test_metadata' in saved_config:
                test_meta = saved_config['test_metadata']
                print(f"✅ Configurazione salvata e verificata!")
                print(f"   Test type: {test_meta.get('test_type')}")
                print(f"   Timestamp: {test_meta.get('timestamp')}")
                print(f"   Saved to: {result.get('saved_to')}")
                
                return {
                    'success': True,
                    'tenant_id': tenant_id,
                    'saved_to': result.get('saved_to'),
                    'verified': True
                }
            else:
                print(f"⚠️ Configurazione salvata ma test metadata mancanti")
                return {
                    'success': True,
                    'tenant_id': tenant_id,
                    'saved_to': result.get('saved_to'),
                    'verified': False,
                    'warning': 'Test metadata missing'
                }
        else:
            print(f"❌ Configurazione NON trovata dopo salvataggio!")
            return {
                'success': False,
                'tenant_id': tenant_id,
                'error': 'Config not found after save'
            }
    
    return {
        'success': False,
        'tenant_id': tenant_id,
        'error': result.get('error', 'Update failed')
    }


def main():
    """
    Test con tenant reali esistenti
    
    Data ultima modifica: 2025-01-31
    """
    print("🧪 TEST CON TENANT REALI ESISTENTI")
    print("=" * 80)
    
    # Tenant che hanno già llm_config (dovrebbero funzionare)
    tenants_with_config = [
        '015007d9-d413-11ef-86a5-96000228e7fe',  # 341 bytes
        'a0fd7600-f4f7-11ef-9315-96000228e7fe'   # 338 bytes  
    ]
    
    # Tenant senza llm_config (per vedere se il problema è questo)
    tenants_without_config = [
        '0c1231de-a58c-11ef-8c7f-96000228e7fe',  # NULL config
        '0f9d6e90-d319-11ef-86a5-96000228e7fe'   # NULL config
    ]
    
    all_results = []
    
    print("🔥 TESTING TENANT CON CONFIG ESISTENTE:")
    for tenant_id in tenants_with_config:
        result = test_existing_tenant_flow(tenant_id)
        all_results.append(result)
    
    print("\n🆕 TESTING TENANT SENZA CONFIG:")
    for tenant_id in tenants_without_config:
        result = test_existing_tenant_flow(tenant_id) 
        all_results.append(result)
    
    # Report finale
    print("\n" + "=" * 80)
    print("📊 REPORT FINALE")
    print("=" * 80)
    
    successful = [r for r in all_results if r.get('success')]
    
    for result in all_results:
        tenant_short = result['tenant_id'][:8]
        status = "✅ PASS" if result.get('success') else "❌ FAIL"
        saved_to = result.get('saved_to', 'unknown')
        
        print(f"{status} {tenant_short} → {saved_to}")
        if not result.get('success') and 'error' in result:
            print(f"   Error: {result['error']}")
        if result.get('verified'):
            print(f"   ✅ Database verified")
    
    print(f"\n🎯 RISULTATI: {len(successful)}/{len(all_results)} tenant funzionanti")
    
    # Analisi patterns
    database_saves = [r for r in successful if r.get('saved_to') == 'database']
    config_saves = [r for r in successful if r.get('saved_to') == 'config.yaml']
    
    print(f"💾 Salvati su DATABASE: {len(database_saves)}")
    print(f"📄 Salvati su CONFIG.YAML: {len(config_saves)}")
    
    if len(database_saves) > 0:
        print("\n🎉 IL SALVATAGGIO SU DATABASE FUNZIONA!")
        print("   Il problema era nei tenant ID fake che stavamo usando.")
    elif len(config_saves) > 0:
        print("\n⚠️ Solo fallback su config.yaml")
        print("   Indagare perché il database non accetta i salvataggi")
    else:
        print("\n🚨 NESSUN SALVATAGGIO RIUSCITO")


if __name__ == "__main__":
    main()