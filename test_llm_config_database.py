#!/usr/bin/env python3
"""
Script di test per LLMConfigService con database-first
Autore: Valerio Bignardi
Data: 2025-11-04

Testa:
1. Lettura parametri LLM da database
2. Salvataggio parametri LLM su database
3. Verifica persistenza
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from Services.llm_configuration_service import LLMConfigurationService
from Database.database_ai_config_service import DatabaseAIConfigService

def test_database_read_write():
    """
    Test completo lettura/scrittura parametri LLM
    """
    print("=" * 80)
    print("TEST LLM CONFIG SERVICE - DATABASE FIRST")
    print("=" * 80)
    
    # Inizializza servizi
    llm_service = LLMConfigurationService()
    db_service = DatabaseAIConfigService()
    
    # Tenant di test (Alleanza)
    test_tenant_id = "a0fd7600-f4f7-11ef-9315-96000228e7fe"
    
    print(f"\n📋 Test 1: Lettura parametri attuali per {test_tenant_id}")
    print("-" * 80)
    current_params = llm_service.get_tenant_parameters(test_tenant_id)
    print(f"✅ Parametri recuperati:")
    print(f"   Source: {current_params.get('source')}")
    print(f"   Model: {current_params.get('current_model')}")
    print(f"   Last modified: {current_params.get('last_modified')}")
    
    params_data = current_params.get('parameters', {})
    if params_data:
        print(f"   Tokenization max_tokens: {params_data.get('tokenization', {}).get('max_tokens', 'N/A')}")
        print(f"   Generation max_tokens: {params_data.get('generation', {}).get('max_tokens', 'N/A')}")
        print(f"   Generation temperature: {params_data.get('generation', {}).get('temperature', 'N/A')}")
    
    print(f"\n📝 Test 2: Aggiornamento parametri (salvataggio su database)")
    print("-" * 80)
    
    # Nuovi parametri di test
    new_params = {
        'tokenization': {
            'max_tokens': 128000,
            'model_name': 'cl100k_base',
            'truncation_strategy': 'start'
        },
        'generation': {
            'max_tokens': 3000,  # Aumento per test
            'temperature': 0.05,  # Cambio per test
            'top_k': 50,
            'top_p': 0.95,
            'repeat_penalty': 1.05
        },
        'connection': {
            'timeout': 350,
            'url': 'http://localhost:11434'
        },
        'test_metadata': {
            'test_run': True,
            'test_timestamp': '2025-11-04T20:00:00'
        }
    }
    
    update_result = llm_service.update_tenant_parameters(
        tenant_id=test_tenant_id,
        parameters=new_params,
        model_name='gpt-4o'
    )
    
    print(f"✅ Risultato aggiornamento:")
    print(f"   Success: {update_result.get('success')}")
    print(f"   Saved to: {update_result.get('saved_to')}")
    print(f"   Message: {update_result.get('message')}")
    
    if not update_result.get('success'):
        print(f"   ❌ ERRORE: {update_result.get('error')}")
        return False
    
    print(f"\n🔍 Test 3: Verifica persistenza su database")
    print("-" * 80)
    
    # Leggi direttamente dal database
    db_config = db_service.get_tenant_configuration(test_tenant_id, force_no_cache=True)
    
    if db_config and db_config.get('llm_config'):
        llm_config_db = db_config['llm_config']
        print(f"✅ Configurazione LLM trovata nel database:")
        print(f"   Tokenization max_tokens: {llm_config_db.get('tokenization', {}).get('max_tokens')}")
        print(f"   Generation max_tokens: {llm_config_db.get('generation', {}).get('max_tokens')}")
        print(f"   Generation temperature: {llm_config_db.get('generation', {}).get('temperature')}")
        print(f"   Test metadata presente: {llm_config_db.get('test_metadata') is not None}")
        
        # Verifica corrispondenza
        if llm_config_db.get('generation', {}).get('max_tokens') == 3000:
            print(f"\n✅ VERIFICA SUPERATA: max_tokens salvato correttamente (3000)")
        else:
            print(f"\n❌ VERIFICA FALLITA: max_tokens non corrisponde")
            return False
            
        if llm_config_db.get('generation', {}).get('temperature') == 0.05:
            print(f"✅ VERIFICA SUPERATA: temperature salvata correttamente (0.05)")
        else:
            print(f"❌ VERIFICA FALLITA: temperature non corrisponde")
            return False
    else:
        print(f"❌ Configurazione LLM NON trovata nel database")
        return False
    
    print(f"\n🔄 Test 4: Rilettura tramite LLMConfigService")
    print("-" * 80)
    
    # Rileggi tramite servizio (dovrebbe usare database)
    reloaded_params = llm_service.get_tenant_parameters(test_tenant_id)
    
    print(f"✅ Parametri riletti:")
    print(f"   Source: {reloaded_params.get('source')}")
    
    reloaded_data = reloaded_params.get('parameters', {})
    if reloaded_data:
        gen_max_tokens = reloaded_data.get('generation', {}).get('max_tokens')
        gen_temperature = reloaded_data.get('generation', {}).get('temperature')
        
        print(f"   Generation max_tokens: {gen_max_tokens}")
        print(f"   Generation temperature: {gen_temperature}")
        
        if gen_max_tokens == 3000 and gen_temperature == 0.05:
            print(f"\n✅ VERIFICA SUPERATA: Parametri corrispondono dopo reload")
        else:
            print(f"\n❌ VERIFICA FALLITA: Parametri non corrispondono dopo reload")
            return False
    else:
        print(f"❌ Nessun parametro trovato dopo reload")
        return False
    
    print("\n" + "=" * 80)
    print("✅ TUTTI I TEST SUPERATI!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    try:
        success = test_database_read_write()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERRORE TEST: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
