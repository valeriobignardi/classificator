#!/usr/bin/env python3
"""
============================================================================
Test Script Configurazione Batch Processing - Sistema Completo
============================================================================

Autore: Valerio Bignardi
Data creazione: 2025-09-07
Ultima modifica: 2025-09-07

Descrizione:
    Script completo per testare l'intera pipeline di configurazione
    batch processing da React frontend al database MySQL:
    
    1. Test database persistence layer
    2. Test service layer con validazione
    3. Test REST API endpoints
    4. Test integrazione completa
    5. Simulazione frontend calls

Dipendenze:
    - requests (pip install requests)
    - mysql-connector-python
    - AIConfiguration/ai_configuration_service.py
    - Database/database_ai_config_service.py

============================================================================
"""

import sys
import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, Any, Optional

# Aggiungi path progetto per import
sys.path.append('/home/ubuntu/classificatore')

try:
    from AIConfiguration.ai_configuration_service import AIConfigurationService
    from Database.database_ai_config_service import DatabaseAIConfigService
    import yaml
except ImportError as e:
    print(f"❌ Errore import: {e}")
    print("🔧 Assicurarsi che i moduli siano nel PYTHONPATH")
    sys.exit(1)

class BatchProcessingTestSuite:
    """
    Test suite completa per configurazione batch processing
    
    Scopo:
        Valida funzionamento end-to-end del sistema di configurazione
        batch processing dal frontend React al database MySQL
        
    Metodi:
        test_database_layer: Test persistenza database
        test_service_layer: Test service con validazione
        test_api_endpoints: Test REST API
        test_integration: Test integrazione completa
        
    Data ultima modifica: 2025-09-07
    """
    
    def __init__(self, tenant_id: str = "test_tenant_001"):
        """
        Inizializza test suite
        
        Args:
            tenant_id: ID tenant per test
            
        Data ultima modifica: 2025-09-07
        """
        self.tenant_id = tenant_id
        self.server_url = "http://localhost:8001"
        self.config = self._load_config()
        
        # Istanze servizi
        self.db_service = DatabaseAIConfigService()
        self.ai_service = AIConfigurationService()
        
        print("🚀 [BatchProcessingTest] Test suite inizializzata")
        print(f"📊 Tenant ID: {tenant_id}")
        print(f"🌐 Server URL: {self.server_url}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Carica configurazione da config.yaml
        
        Returns:
            Dizionario configurazione
            
        Data ultima modifica: 2025-09-07
        """
        try:
            with open('/home/ubuntu/classificatore/config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Errore caricamento config: {e}")
            return {}
    
    def test_database_layer(self) -> bool:
        """
        Test del layer database per persistenza configurazione
        
        Returns:
            True se test superato, False altrimenti
            
        Data ultima modifica: 2025-09-07
        """
        print("\n" + "="*60)
        print("🗄️  TEST 1: DATABASE PERSISTENCE LAYER")
        print("="*60)
        
        try:
            # Test 1: Salvataggio configurazione
            test_config = {
                "classification_batch_size": 50,
                "max_parallel_calls": 150,
                "source": "test",
                "test_timestamp": datetime.now().isoformat()
            }
            
            print(f"💾 Salvando configurazione test: {test_config}")
            result = self.db_service.save_batch_processing_config(
                self.tenant_id, 
                test_config
            )
            
            if not result.get('success'):
                print(f"❌ Errore salvataggio: {result.get('error')}")
                return False
            
            print("✅ Configurazione salvata nel database")
            
            # Test 2: Recupero configurazione
            print(f"📥 Recuperando configurazione per tenant {self.tenant_id}")
            retrieved_config = self.db_service.get_batch_processing_config(
                self.tenant_id
            )
            
            if not retrieved_config.get('success'):
                print(f"❌ Errore recupero: {retrieved_config.get('error')}")
                return False
            
            saved_config = retrieved_config['batch_config']
            print(f"✅ Configurazione recuperata: {saved_config}")
            
            # Verifica corrispondenza
            if (saved_config['classification_batch_size'] != test_config['classification_batch_size'] or
                saved_config['max_parallel_calls'] != test_config['max_parallel_calls']):
                print("❌ Configurazione recuperata non corrisponde a quella salvata")
                return False
            
            print("✅ Test database layer SUPERATO")
            return True
            
        except Exception as e:
            print(f"❌ Errore test database: {e}")
            return False
    
    def test_service_layer(self) -> bool:
        """
        Test del service layer con validazione parametri
        
        Returns:
            True se test superato, False altrimenti
            
        Data ultima modifica: 2025-09-07
        """
        print("\n" + "="*60)
        print("⚙️  TEST 2: SERVICE LAYER CON VALIDAZIONE")
        print("="*60)
        
        try:
            # Test 1: Configurazione valida
            valid_config = {
                "classification_batch_size": 25,
                "max_parallel_calls": 100
            }
            
            print(f"🔍 Test configurazione valida: {valid_config}")
            result = self.ai_service.save_batch_processing_config(
                self.tenant_id,
                valid_config
            )
            
            if not result.get('success'):
                print(f"❌ Errore salvataggio configurazione valida: {result.get('error')}")
                return False
            
            print("✅ Configurazione valida salvata correttamente")
            
            # Test 2: Configurazione non valida - batch size troppo alto
            invalid_config = {
                "classification_batch_size": 2000,  # Troppo alto
                "max_parallel_calls": 100
            }
            
            print(f"🔍 Test configurazione non valida: {invalid_config}")
            result = self.ai_service.save_batch_processing_config(
                self.tenant_id,
                invalid_config
            )
            
            if result.get('success'):
                print("❌ Configurazione non valida accettata erroneamente")
                return False
            
            print(f"✅ Configurazione non valida respinta: {result.get('error')}")
            
            # Test 3: Recupero con fallback intelligente
            print(f"📥 Test recupero configurazione con fallback")
            retrieved = self.ai_service.get_batch_processing_config(self.tenant_id)
            
            if not retrieved.get('success'):
                print(f"❌ Errore recupero: {retrieved.get('error')}")
                return False
            
            config = retrieved['batch_config']
            print(f"✅ Configurazione recuperata con fallback: {config}")
            
            # Verifica presenza source
            if 'source' not in config:
                print("❌ Campo 'source' mancante nella configurazione")
                return False
            
            print("✅ Test service layer SUPERATO")
            return True
            
        except Exception as e:
            print(f"❌ Errore test service: {e}")
            return False
    
    def test_api_endpoints(self) -> bool:
        """
        Test degli endpoint REST API
        
        Returns:
            True se test superato, False altrimenti
            
        Data ultima modifica: 2025-09-07
        """
        print("\n" + "="*60)
        print("🌐 TEST 3: REST API ENDPOINTS")
        print("="*60)
        
        try:
            # Test 1: GET configurazione
            print(f"📥 GET /api/ai-config/{self.tenant_id}/batch-config")
            response = requests.get(
                f"{self.server_url}/api/ai-config/{self.tenant_id}/batch-config",
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"❌ GET fallito: {response.status_code}")
                return False
            
            get_data = response.json()
            print(f"✅ GET response: {get_data}")
            
            # Test 2: POST configurazione valida
            valid_payload = {
                "classification_batch_size": 40,
                "max_parallel_calls": 180
            }
            
            print(f"💾 POST configurazione valida: {valid_payload}")
            response = requests.post(
                f"{self.server_url}/api/ai-config/{self.tenant_id}/batch-config",
                json=valid_payload,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"❌ POST fallito: {response.status_code} - {response.text}")
                return False
            
            post_data = response.json()
            print(f"✅ POST response: {post_data}")
            
            # Test 3: POST configurazione non valida
            invalid_payload = {
                "classification_batch_size": -5,  # Valore negativo
                "max_parallel_calls": 1000       # Troppo alto
            }
            
            print(f"🚫 POST configurazione non valida: {invalid_payload}")
            response = requests.post(
                f"{self.server_url}/api/ai-config/{self.tenant_id}/batch-config",
                json=invalid_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print("❌ Configurazione non valida accettata erroneamente")
                return False
            
            print(f"✅ Configurazione non valida respinta: {response.status_code}")
            
            # Test 4: Validazione endpoint
            validation_payload = {
                "classification_batch_size": 60,
                "max_parallel_calls": 250
            }
            
            print(f"🔍 POST validation: {validation_payload}")
            response = requests.post(
                f"{self.server_url}/api/ai-config/{self.tenant_id}/batch-config/validate",
                json=validation_payload,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"❌ Validation fallito: {response.status_code}")
                return False
            
            validation_data = response.json()
            print(f"✅ Validation response: {validation_data}")
            
            print("✅ Test API endpoints SUPERATO")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Errore connessione API: {e}")
            print("🔧 Assicurarsi che il server sia in esecuzione su porta 8001")
            return False
        except Exception as e:
            print(f"❌ Errore test API: {e}")
            return False
    
    def test_integration_flow(self) -> bool:
        """
        Test integrazione completa end-to-end
        
        Returns:
            True se test superato, False altrimenti
            
        Data ultima modifica: 2025-09-07
        """
        print("\n" + "="*60)
        print("🔄 TEST 4: INTEGRAZIONE END-TO-END")
        print("="*60)
        
        try:
            # Simula flusso completo React -> API -> Service -> Database
            
            # 1. Frontend invia configurazione
            frontend_config = {
                "classification_batch_size": 35,
                "max_parallel_calls": 220,
                "user_action": "frontend_save",
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"🎯 Simulazione frontend: invio configurazione {frontend_config}")
            
            # 2. API riceve e processa richiesta
            api_response = requests.post(
                f"{self.server_url}/api/ai-config/{self.tenant_id}/batch-config",
                json=frontend_config,
                timeout=10
            )
            
            if api_response.status_code != 200:
                print(f"❌ API non ha accettato configurazione: {api_response.status_code}")
                return False
            
            api_data = api_response.json()
            print(f"✅ API ha processato configurazione: {api_data}")
            
            # 3. Verifica persistenza database
            time.sleep(1)  # Piccola pausa per permettere il salvataggio
            
            db_config = self.db_service.get_batch_processing_config(self.tenant_id)
            if not db_config.get('success'):
                print(f"❌ Configurazione non persistita in database: {db_config.get('error')}")
                return False
            
            saved_config = db_config['batch_config']
            print(f"✅ Configurazione persistita in database: {saved_config}")
            
            # 4. Frontend recupera configurazione aggiornata
            print("📥 Frontend recupera configurazione aggiornata")
            get_response = requests.get(
                f"{self.server_url}/api/ai-config/{self.tenant_id}/batch-config",
                timeout=10
            )
            
            if get_response.status_code != 200:
                print(f"❌ Frontend non riesce a recuperare configurazione: {get_response.status_code}")
                return False
            
            frontend_retrieved = get_response.json()
            print(f"✅ Frontend ha recuperato: {frontend_retrieved}")
            
            # 5. Verifica corrispondenza end-to-end
            if (saved_config['classification_batch_size'] != frontend_config['classification_batch_size'] or
                saved_config['max_parallel_calls'] != frontend_config['max_parallel_calls']):
                print("❌ Configurazione end-to-end non corrisponde")
                return False
            
            print("✅ Test integrazione end-to-end SUPERATO")
            return True
            
        except Exception as e:
            print(f"❌ Errore test integrazione: {e}")
            return False
    
    def run_comprehensive_test(self) -> bool:
        """
        Esegue test suite completa
        
        Returns:
            True se tutti i test superati, False altrimenti
            
        Data ultima modifica: 2025-09-07
        """
        print("🧪 " + "="*60)
        print("🧪 SUITE TEST CONFIGURAZIONE BATCH PROCESSING")
        print("🧪 " + "="*60)
        print(f"🕐 Inizio test: {datetime.now()}")
        
        test_results = {
            "database_layer": False,
            "service_layer": False,
            "api_endpoints": False,
            "integration_flow": False
        }
        
        # Esegui tutti i test
        test_results["database_layer"] = self.test_database_layer()
        test_results["service_layer"] = self.test_service_layer()
        test_results["api_endpoints"] = self.test_api_endpoints()
        test_results["integration_flow"] = self.test_integration_flow()
        
        # Report finale
        print("\n" + "🏁" + "="*60)
        print("🏁 REPORT FINALE TEST SUITE")
        print("🏁" + "="*60)
        
        all_passed = True
        for test_name, result in test_results.items():
            status = "✅ SUPERATO" if result else "❌ FALLITO"
            print(f"📊 {test_name.replace('_', ' ').title()}: {status}")
            if not result:
                all_passed = False
        
        print(f"\n🎯 RISULTATO FINALE: {'✅ TUTTI I TEST SUPERATI' if all_passed else '❌ ALCUNI TEST FALLITI'}")
        print(f"🕐 Fine test: {datetime.now()}")
        
        if all_passed:
            print("\n🎉 CONFIGURAZIONE BATCH PROCESSING COMPLETAMENTE FUNZIONALE!")
            print("🚀 Il sistema React -> API -> Service -> Database è operativo")
        else:
            print("\n🔧 ALCUNI COMPONENTI NECESSITANO CORREZIONI")
            print("📋 Controllare i log di errore sopra per dettagli")
        
        return all_passed

def main():
    """
    Funzione principale per esecuzione test
    
    Data ultima modifica: 2025-09-07
    """
    if len(sys.argv) > 1:
        tenant_id = sys.argv[1]
    else:
        tenant_id = "test_tenant_001"
    
    test_suite = BatchProcessingTestSuite(tenant_id=tenant_id)
    success = test_suite.run_comprehensive_test()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
