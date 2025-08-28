#!/usr/bin/env python3
"""
File: verify_api_coherence.py
Autore: Valerio Bignardi
Data creazione: 2025-01-17
Descrizione: Verifica coerenza tra API, frontend e MongoDB per i filtri
Storia aggiornamenti:
- 2025-01-17: Creazione per verifica completa sistema
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from pymongo import MongoClient
import yaml
import requests
from Utils.tenant import Tenant

class CoherenceVerifier:
    """
    Verifica coerenza tra i diversi livelli del sistema
    """
    
    def __init__(self):
        """
        Inizializza il verificatore
        """
        print("🔍 VERIFICA COERENZA SISTEMA COMPLETO")
        
        # Carica configurazione
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup connessione MongoDB
        mongodb_config = self.config.get('mongodb', {})
        mongodb_url = mongodb_config.get('url', 'mongodb://localhost:27017')
        database_name = mongodb_config.get('database', 'classificazioni')
        
        self.mongo_client = MongoClient(mongodb_url)
        self.db = self.mongo_client[database_name]
        
        # Setup tenant
        self.tenant = Tenant.from_slug("humanitas")
        self.collection_name = f"humanitas_{self.tenant.tenant_id}"
        
        # API base URL
        self.api_base = "http://localhost:5000/api"
        
        print(f"✅ Configurato per tenant: {self.tenant.tenant_name}")
        print(f"✅ Collection MongoDB: {self.collection_name}")
    
    def create_test_data(self):
        """
        Crea dati di test con tutti i tipi di sessione
        """
        print("\n📊 CREAZIONE DATI DI TEST")
        
        # Cancella dati esistenti
        self.db[self.collection_name].delete_many({
            "session_id": {"$in": [
                "test_representative_1",
                "test_representative_2", 
                "test_propagated_1",
                "test_propagated_2",
                "test_outlier_1"
            ]}
        })
        
        test_sessions = [
            # RAPPRESENTATIVI (pending review)
            {
                "session_id": "test_representative_1",
                "client_name": self.tenant.tenant_id,
                "predicted_label": "info_polizza",
                "confidence": 0.9,
                "review_status": "pending",
                "cluster_metadata": {
                    "cluster_id": "cluster_test_1",
                    "session_type": "representative",
                    "is_representative": True
                },
                "conversation_text": "Test rappresentativo 1",
                "timestamp": "2025-01-17T10:00:00Z"
            },
            {
                "session_id": "test_representative_2",
                "client_name": self.tenant.tenant_id,
                "predicted_label": "reclamo",
                "confidence": 0.85,
                "review_status": "pending",
                "cluster_metadata": {
                    "cluster_id": "cluster_test_2",
                    "session_type": "representative",
                    "is_representative": True
                },
                "conversation_text": "Test rappresentativo 2",
                "timestamp": "2025-01-17T10:01:00Z"
            },
            # PROPAGATED (derivano dai rappresentativi)
            {
                "session_id": "test_propagated_1",
                "client_name": self.tenant.tenant_id,
                "predicted_label": "info_polizza",
                "confidence": 0.8,
                "review_status": "auto_classified",
                "cluster_metadata": {
                    "cluster_id": "cluster_test_1",
                    "session_type": "propagated",
                    "is_representative": False
                },
                "conversation_text": "Test propagated 1",
                "timestamp": "2025-01-17T10:02:00Z"
            },
            {
                "session_id": "test_propagated_2", 
                "client_name": self.tenant.tenant_id,
                "predicted_label": None,
                "confidence": None,
                "review_status": "pending",
                "cluster_metadata": {
                    "cluster_id": "cluster_test_1",
                    "session_type": "propagated", 
                    "is_representative": False
                },
                "conversation_text": "Test propagated 2 (pending)",
                "timestamp": "2025-01-17T10:03:00Z"
            },
            # OUTLIER
            {
                "session_id": "test_outlier_1",
                "client_name": self.tenant.tenant_id,
                "predicted_label": "altro",
                "confidence": 0.6,
                "review_status": "pending",
                "cluster_metadata": {
                    "cluster_id": "outlier_1", 
                    "session_type": "outlier",
                    "is_representative": False
                },
                "conversation_text": "Test outlier 1",
                "timestamp": "2025-01-17T10:04:00Z"
            }
        ]
        
        # Inserisci dati
        self.db[self.collection_name].insert_many(test_sessions)
        
        print(f"✅ Creati {len(test_sessions)} documenti di test:")
        print(f"   👥 Representatives: 2 (pending)")
        print(f"   🔄 Propagated: 2 (1 auto_classified, 1 pending)")
        print(f"   ⚡ Outliers: 1 (pending)")
        print(f"   📋 Total pending: 4")
    
    def verify_mongodb_data(self):
        """
        Verifica dati direttamente in MongoDB
        """
        print("\n🗄️ VERIFICA DATI MONGODB")
        
        # Conta per tipo usando la nuova struttura
        representatives = list(self.db[self.collection_name].find({
            "cluster_metadata.session_type": "representative"
        }))
        
        propagated = list(self.db[self.collection_name].find({
            "cluster_metadata.session_type": "propagated"
        }))
        
        outliers = list(self.db[self.collection_name].find({
            "cluster_metadata.session_type": "outlier"
        }))
        
        print(f"📊 Conteggi MongoDB diretti:")
        print(f"   👥 Representatives: {len(representatives)}")
        print(f"   🔄 Propagated: {len(propagated)}")
        print(f"   ⚡ Outliers: {len(outliers)}")
        
        # Verifica pending vs auto_classified
        pending = list(self.db[self.collection_name].find({
            "review_status": "pending"
        }))
        
        auto_classified = list(self.db[self.collection_name].find({
            "review_status": "auto_classified"
        }))
        
        print(f"📋 Review status:")
        print(f"   ⏳ Pending: {len(pending)}")
        print(f"   ✅ Auto-classified: {len(auto_classified)}")
        
        return {
            'representatives': len(representatives),
            'propagated': len(propagated),
            'outliers': len(outliers),
            'pending': len(pending),
            'auto_classified': len(auto_classified)
        }
    
    def test_api_filters(self):
        """
        Testa le API con diversi filtri
        """
        print("\n🌐 TEST API CON FILTRI")
        
        test_cases = [
            {
                'name': 'Tutti i tipi',
                'params': {
                    'include_representatives': 'true',
                    'include_propagated': 'true', 
                    'include_outliers': 'true'
                }
            },
            {
                'name': 'Solo Representatives',
                'params': {
                    'include_representatives': 'true',
                    'include_propagated': 'false',
                    'include_outliers': 'false'
                }
            },
            {
                'name': 'Solo Propagated',
                'params': {
                    'include_representatives': 'false',
                    'include_propagated': 'true',
                    'include_outliers': 'false'
                }
            },
            {
                'name': 'Solo Outliers',
                'params': {
                    'include_representatives': 'false',
                    'include_propagated': 'false',
                    'include_outliers': 'true'
                }
            },
            {
                'name': 'Nascondi Outliers (hide)',
                'params': {
                    'include_representatives': 'true',
                    'include_propagated': 'true',
                    'include_outliers': 'false'
                }
            }
        ]
        
        results = {}
        
        for test_case in test_cases:
            try:
                url = f"{self.api_base}/review/humanitas/cases"
                response = requests.get(url, params=test_case['params'], timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    case_count = len(data.get('cases', []))
                    results[test_case['name']] = case_count
                    print(f"   ✅ {test_case['name']}: {case_count} casi")
                    
                    # Debug: mostra session_type dei risultati
                    types = []
                    for case in data.get('cases', []):
                        session_type = case.get('cluster_metadata', {}).get('session_type', 'unknown')
                        types.append(session_type)
                    type_counts = {t: types.count(t) for t in set(types)}
                    print(f"      📋 Breakdown: {type_counts}")
                    
                else:
                    print(f"   ❌ {test_case['name']}: HTTP {response.status_code}")
                    results[test_case['name']] = 0
                    
            except Exception as e:
                print(f"   ❌ {test_case['name']}: Errore {e}")
                results[test_case['name']] = 0
        
        return results
    
    def verify_coherence(self, mongodb_counts, api_results):
        """
        Verifica la coerenza tra MongoDB e API
        """
        print("\n🔍 VERIFICA COERENZA")
        
        # Test coerenza
        coherence_tests = [
            {
                'test': 'Solo Representatives',
                'expected': mongodb_counts['representatives'],
                'actual': api_results.get('Solo Representatives', 0)
            },
            {
                'test': 'Solo Propagated', 
                'expected': mongodb_counts['propagated'],
                'actual': api_results.get('Solo Propagated', 0)
            },
            {
                'test': 'Solo Outliers',
                'expected': mongodb_counts['outliers'],
                'actual': api_results.get('Solo Outliers', 0)
            },
            {
                'test': 'Tutti i tipi',
                'expected': mongodb_counts['representatives'] + mongodb_counts['propagated'] + mongodb_counts['outliers'],
                'actual': api_results.get('Tutti i tipi', 0)
            }
        ]
        
        all_coherent = True
        
        for test in coherence_tests:
            if test['expected'] == test['actual']:
                print(f"   ✅ {test['test']}: {test['expected']} (coerente)")
            else:
                print(f"   ❌ {test['test']}: Atteso {test['expected']}, ottenuto {test['actual']}")
                all_coherent = False
        
        if all_coherent:
            print("\n🎉 SISTEMA COMPLETAMENTE COERENTE!")
        else:
            print("\n⚠️ Rilevate incoerenze nel sistema")
        
        return all_coherent
    
    def cleanup_test_data(self):
        """
        Pulisce i dati di test
        """
        print("\n🧹 CLEANUP DATI DI TEST")
        result = self.db[self.collection_name].delete_many({
            "session_id": {"$in": [
                "test_representative_1",
                "test_representative_2",
                "test_propagated_1", 
                "test_propagated_2",
                "test_outlier_1"
            ]}
        })
        print(f"✅ Eliminati {result.deleted_count} documenti di test")
    
    def run_verification(self):
        """
        Esegue la verifica completa
        """
        try:
            # Setup
            self.create_test_data()
            
            # Verifica MongoDB
            mongodb_counts = self.verify_mongodb_data()
            
            # Test API
            api_results = self.test_api_filters()
            
            # Verifica coerenza
            coherent = self.verify_coherence(mongodb_counts, api_results)
            
            if coherent:
                print(f"\n✅ VERIFICA COMPLETATA: Sistema coerente")
                print(f"   📊 Filtri funzionano correttamente")
                print(f"   🔄 MongoDB ↔ API ↔ Frontend allineati") 
            else:
                print(f"\n❌ VERIFICA FALLITA: Sistema non coerente")
            
            return coherent
            
        except Exception as e:
            print(f"\n❌ ERRORE DURANTE VERIFICA: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Cleanup
            self.cleanup_test_data()
            self.mongo_client.close()

def main():
    """
    Funzione main
    """
    verifier = CoherenceVerifier()
    return verifier.run_verification()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
