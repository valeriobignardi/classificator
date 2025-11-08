#!/usr/bin/env python3
"""
============================================================================
Test del Flusso React → Database per Configurazione LLM
============================================================================

Autore: Valerio Bignardi
Data creazione: 2025-01-31
Ultima modifica: 2025-01-31

Descrizione:
    Simula il flusso esatto che dovrebbe avvenire quando l'interfaccia React
    invia parametri LLM al backend per verificare dove si blocca la persistenza
    nel database.

Flusso testato:
    1. React Frontend → llmConfigService.ts
    2. API PUT /api/llm/parameters/{tenant_id}
    3. LLMConfigurationService.update_tenant_parameters()
    4. DatabaseAIConfigService.set_llm_engine()
    5. Database TAG.engines (llm_config field)

============================================================================
"""

import sys
import os
import json
import requests
import traceback
from typing import Dict, Any

# Aggiungi path per importi
sys.path.append('/home/ubuntu/classificatore')
sys.path.append('/home/ubuntu/classificatore/Database')
sys.path.append('/home/ubuntu/classificatore/Services')

from Services.llm_configuration_service import LLMConfigurationService
from Database.database_ai_config_service import DatabaseAIConfigService


def test_direct_database_save(tenant_id: str) -> Dict[str, Any]:
    """
    Test diretto del salvataggio su database senza passare per API
    
    Args:
        tenant_id: ID del tenant da testare
        
    Returns:
        Risultato del test
        
    Data ultima modifica: 2025-01-31
    """
    print("=" * 80)
    print("🧪 TEST 1: Salvataggio DIRETTO su Database")
    print("=" * 80)
    
    try:
        # Inizializza servizio database
        db_service = DatabaseAIConfigService()
        
        # Parametri di test
        test_parameters = {
            'tokenization': {
                'max_tokens': 4000,
                'temperature': 0.7
            },
            'generation': {
                'max_tokens': 2000,
                'temperature': 0.6,
                'top_k': 50,
                'top_p': 0.95
            },
            'connection': {
                'timeout': 300,
                'url': 'http://localhost:11434'
            },
            'test_metadata': {
                'test_type': 'direct_database_save',
                'timestamp': '2025-01-31T12:00:00'
            }
        }
        
        print(f"📤 Tentativo salvataggio parametri per tenant: {tenant_id}")
        print(f"📋 Parametri da salvare: {json.dumps(test_parameters, indent=2)}")
        
        # Salva usando set_llm_engine
        result = db_service.set_llm_engine(
            tenant_id=tenant_id,
            model_name='gpt-4o',
            **test_parameters
        )
        
        print(f"\n✅ Risultato salvataggio:")
        print(f"   Success: {result.get('success')}")
        print(f"   Message: {result.get('message', 'N/A')}")
        print(f"   Error: {result.get('error', 'N/A')}")
        
        # Verifica lettura
        if result.get('success'):
            print(f"\n🔍 Verifica lettura da database...")
            config = db_service.get_tenant_configuration(tenant_id, force_no_cache=True)
            
            if config and config.get('llm_config'):
                print(f"✅ Configurazione LLM trovata nel database!")
                print(f"   Dimensione llm_config: {len(str(config['llm_config']))} caratteri")
                print(f"   Keys presenti: {list(config['llm_config'].keys())}")
                return {
                    'success': True,
                    'test': 'direct_database_save',
                    'database_written': True,
                    'database_readable': True
                }
            else:
                print(f"❌ Configurazione NON trovata nel database!")
                return {
                    'success': False,
                    'test': 'direct_database_save',
                    'database_written': result.get('success'),
                    'database_readable': False,
                    'error': 'Config non leggibile dal database'
                }
        
        return {
            'success': False,
            'test': 'direct_database_save',
            'database_written': False,
            'error': result.get('error', 'Salvataggio fallito')
        }
        
    except Exception as e:
        error_msg = f"Errore test database diretto: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return {
            'success': False,
            'test': 'direct_database_save',
            'error': error_msg
        }


def test_llm_service_flow(tenant_id: str) -> Dict[str, Any]:
    """
    Test del flusso tramite LLMConfigurationService
    
    Args:
        tenant_id: ID del tenant da testare
        
    Returns:
        Risultato del test
        
    Data ultima modifica: 2025-01-31
    """
    print("\n" + "=" * 80)
    print("🧪 TEST 2: Flusso tramite LLMConfigurationService")
    print("=" * 80)
    
    try:
        # Inizializza servizio LLM
        llm_service = LLMConfigurationService()
        
        # Parametri di test (come inviati da React)
        react_parameters = {
            'tokenization': {
                'max_tokens': 3500,
                'temperature': 0.8
            },
            'generation': {
                'max_tokens': 1500,
                'temperature': 0.5,
                'top_k': 40,
                'top_p': 0.9
            },
            'connection': {
                'timeout': 350,
                'url': 'http://localhost:11434'
            },
            'test_metadata': {
                'test_type': 'llm_service_flow',
                'timestamp': '2025-01-31T12:30:00'
            }
        }
        
        print(f"📤 Chiamata update_tenant_parameters per tenant: {tenant_id}")
        print(f"📋 Parametri: {json.dumps(react_parameters, indent=2)}")
        
        # Chiama il metodo che dovrebbe usare React
        result = llm_service.update_tenant_parameters(
            tenant_id=tenant_id,
            parameters=react_parameters,
            model_name='gpt-4o'
        )
        
        print(f"\n✅ Risultato LLMConfigurationService:")
        print(f"   Success: {result.get('success')}")
        print(f"   Saved to: {result.get('saved_to')}")
        print(f"   Message: {result.get('message')}")
        print(f"   Error: {result.get('error', 'N/A')}")
        
        # Se salvato su database, verifica
        if result.get('success') and result.get('saved_to') == 'database':
            print(f"\n🔍 Verifica persistenza su database...")
            db_service = DatabaseAIConfigService()
            config = db_service.get_tenant_configuration(tenant_id, force_no_cache=True)
            
            if config and config.get('llm_config'):
                saved_config = config['llm_config']
                print(f"✅ Configurazione persistita correttamente!")
                
                # Verifica presenza test_metadata
                if 'test_metadata' in saved_config:
                    print(f"✅ Test metadata trovata: {saved_config['test_metadata']}")
                    return {
                        'success': True,
                        'test': 'llm_service_flow',
                        'saved_to': result.get('saved_to'),
                        'database_verified': True
                    }
                else:
                    print(f"⚠️ Test metadata NON trovata, ma configurazione salvata")
                    return {
                        'success': True,
                        'test': 'llm_service_flow',
                        'saved_to': result.get('saved_to'),
                        'database_verified': True,
                        'warning': 'Metadata mancanti'
                    }
            else:
                return {
                    'success': False,
                    'test': 'llm_service_flow',
                    'saved_to': result.get('saved_to'),
                    'database_verified': False,
                    'error': 'Config non trovata in database dopo salvataggio'
                }
        
        return {
            'success': result.get('success'),
            'test': 'llm_service_flow',
            'saved_to': result.get('saved_to'),
            'error': result.get('error')
        }
        
    except Exception as e:
        error_msg = f"Errore test LLM service: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return {
            'success': False,
            'test': 'llm_service_flow',
            'error': error_msg
        }


def test_api_endpoint_simulation(tenant_id: str, server_port: int = 5000) -> Dict[str, Any]:
    """
    Test simulando chiamata HTTP come fa React
    
    Args:
        tenant_id: ID del tenant da testare
        server_port: Porta del server Flask
        
    Returns:
        Risultato del test
        
    Data ultima modifica: 2025-01-31
    """
    print("\n" + "=" * 80)
    print("🧪 TEST 3: Simulazione chiamata API HTTP (come React)")
    print("=" * 80)
    
    try:
        # URL endpoint
        url = f"http://localhost:{server_port}/api/llm/parameters/{tenant_id}"
        
        # Payload identico a quello di React
        payload = {
            'parameters': {
                'tokenization': {
                    'max_tokens': 3800,
                    'temperature': 0.9
                },
                'generation': {
                    'max_tokens': 1800,
                    'temperature': 0.4,
                    'top_k': 45,
                    'top_p': 0.92
                },
                'connection': {
                    'timeout': 400,
                    'url': 'http://localhost:11434'
                },
                'test_metadata': {
                    'test_type': 'api_endpoint_simulation',
                    'timestamp': '2025-01-31T13:00:00'
                }
            },
            'model_name': 'gpt-4o'
        }
        
        print(f"📤 POST {url}")
        print(f"📋 Payload: {json.dumps(payload, indent=2)}")
        
        # Headers come React
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # Esegui chiamata
        response = requests.put(url, json=payload, headers=headers, timeout=30)
        
        print(f"\n📨 Risposta HTTP:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        
        try:
            response_data = response.json()
            print(f"   Body: {json.dumps(response_data, indent=2)}")
            
            if response.status_code == 200 and response_data.get('success'):
                print(f"\n✅ API call riuscita!")
                
                # Verifica database
                print(f"🔍 Verifica finale su database...")
                db_service = DatabaseAIConfigService()
                config = db_service.get_tenant_configuration(tenant_id, force_no_cache=True)
                
                if config and config.get('llm_config'):
                    saved_config = config['llm_config']
                    if 'test_metadata' in saved_config:
                        print(f"✅ API → Database flow COMPLETAMENTE FUNZIONANTE!")
                        return {
                            'success': True,
                            'test': 'api_endpoint_simulation',
                            'http_status': response.status_code,
                            'database_verified': True,
                            'api_response': response_data
                        }
                
                return {
                    'success': False,
                    'test': 'api_endpoint_simulation',
                    'http_status': response.status_code,
                    'database_verified': False,
                    'error': 'Config non persistita in database',
                    'api_response': response_data
                }
            else:
                return {
                    'success': False,
                    'test': 'api_endpoint_simulation',
                    'http_status': response.status_code,
                    'error': response_data.get('error', 'API call failed'),
                    'api_response': response_data
                }
                
        except json.JSONDecodeError:
            print(f"❌ Risposta non JSON: {response.text}")
            return {
                'success': False,
                'test': 'api_endpoint_simulation',
                'http_status': response.status_code,
                'error': f'Risposta non JSON: {response.text[:200]}'
            }
        
    except requests.exceptions.ConnectionError:
        error_msg = f"Server non raggiungibile su porta {server_port}"
        print(f"❌ {error_msg}")
        return {
            'success': False,
            'test': 'api_endpoint_simulation',
            'error': error_msg,
            'suggestion': 'Avviare il server Flask prima di eseguire il test'
        }
    except Exception as e:
        error_msg = f"Errore chiamata API: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return {
            'success': False,
            'test': 'api_endpoint_simulation',
            'error': error_msg
        }


def main():
    """
    Esegue tutti i test per verificare il flusso React → Database
    
    Data ultima modifica: 2025-01-31
    """
    print("🔬 ANALISI COMPLETA FLUSSO REACT → DATABASE")
    print("=" * 80)
    
    # Tenant di test
    test_tenants = [
        'f47ac10b-58cc-4372-a567-0e02b2c3d479',  # Humanitas
        'a1b2c3d4-e5f6-7890-1234-567890abcdef'   # Alleanza (se esiste)
    ]
    
    results = []
    
    for tenant_id in test_tenants:
        print(f"\n🏢 TESTING TENANT: {tenant_id}")
        print("-" * 80)
        
        # Test 1: Database diretto
        result1 = test_direct_database_save(tenant_id)
        results.append(result1)
        
        # Test 2: LLM Service
        result2 = test_llm_service_flow(tenant_id)
        results.append(result2)
        
        # Test 3: API HTTP (solo se server disponibile)
        result3 = test_api_endpoint_simulation(tenant_id)
        results.append(result3)
    
    # Report finale
    print("\n" + "=" * 80)
    print("📊 REPORT FINALE")
    print("=" * 80)
    
    for i, result in enumerate(results):
        test_name = result.get('test', f'test_{i}')
        success = result.get('success', False)
        status = "✅ PASS" if success else "❌ FAIL"
        
        print(f"{status} {test_name}")
        if not success and 'error' in result:
            print(f"   Error: {result['error']}")
        if 'saved_to' in result:
            print(f"   Saved to: {result['saved_to']}")
    
    # Conclusioni
    successful_tests = [r for r in results if r.get('success')]
    total_tests = len(results)
    
    print(f"\n🎯 SUCCESSI: {len(successful_tests)}/{total_tests}")
    
    if len(successful_tests) == 0:
        print("🚨 TUTTI I TEST FALLITI - Problema critico nel flusso!")
    elif len(successful_tests) < total_tests:
        print("⚠️ ALCUNI TEST FALLITI - Indagare i punti di failure")
    else:
        print("🎉 TUTTI I TEST RIUSCITI - Il flusso React → Database funziona!")


if __name__ == "__main__":
    main()