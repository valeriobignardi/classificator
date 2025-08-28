#!/usr/bin/env python3
"""
Author: Valerio Bignardi
Date: 2025-08-28
Description: Test sistematico completo per verifica coerenza sistema
Last Update: 2025-08-28

SCOPO: Verifica sistematica completa del sistema di classificazione
dopo il training, includendo:
1. Verifica MongoDB data structure
2. Test API coherence
3. Test filtri frontend
4. Test sistema di consensus intelligente
5. Validazione end-to-end workflow

Ultimo aggiornamento: 2025-08-28
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mongo_classification_reader import MongoClassificationReader
from typing import Dict, List, Any, Optional
import requests
import json
from datetime import datetime

class SystematicVerificationTester:
    """
    Scopo: Test sistematico completo del sistema di classificazione
    
    Parametri input:
        - client_name: Nome del cliente (es. 'wopta')
        - tenant_id: ID del tenant
        - api_base_url: URL base delle API
        
    Output:
        - Report completo stato sistema
        - Verifica coerenza MongoDB-API-Frontend
        
    Ultimo aggiornamento: 2025-08-28
    """
    
    def __init__(self, client_name: str, tenant_id: str, api_base_url: str = "http://localhost:5000"):
        """
        Inizializzazione tester sistematico
        
        Args:
            client_name: Nome cliente (wopta)
            tenant_id: ID tenant (16c222a9-f293-11ef-9315-96000228e7fe)
            api_base_url: URL base API server
        """
        self.client_name = client_name
        self.tenant_id = tenant_id
        self.api_base_url = api_base_url
        
        # Inizializza MongoDB reader
        self.mongo_reader = MongoClassificationReader(
            mongodb_url='mongodb://localhost:27017',
            database_name='classificazioni',
            tenant_name=client_name
        )
        
        # Risultati test
        self.test_results = {}
        
    def connect_mongodb(self) -> bool:
        """
        Connessione a MongoDB
        """
        try:
            success = self.mongo_reader.connect()
            if success:
                self.collection = self.mongo_reader.db[self.mongo_reader.collection_name]
                print(f"✅ MongoDB connesso - Collection: {self.mongo_reader.collection_name}")
                return True
            else:
                print("❌ Errore connessione MongoDB")
                return False
        except Exception as e:
            print(f"❌ Errore connessione MongoDB: {e}")
            return False
    
    def test_mongodb_data_structure(self) -> Dict[str, Any]:
        """
        Scopo: Verifica struttura dati MongoDB post-training
        
        Returns:
            Dizionario con risultati verifica MongoDB
            
        Ultimo aggiornamento: 2025-08-28
        """
        print("\\n🔍 TEST 1: VERIFICA STRUTTURA DATI MONGODB")
        print("="*50)
        
        results = {
            "total_documents": 0,
            "with_cluster_metadata": 0,
            "representatives": 0,
            "propagated": 0,
            "outliers": 0,
            "auto_classified": 0,
            "pending_review": 0,
            "cluster_coverage": {},
            "status": "UNKNOWN"
        }
        
        try:
            # Conteggi base
            results["total_documents"] = self.collection.count_documents({})
            results["with_cluster_metadata"] = self.collection.count_documents(
                {"cluster_metadata": {"$exists": True}}
            )
            
            # Conteggi per tipo
            results["representatives"] = self.collection.count_documents(
                {"cluster_metadata.is_representative": True}
            )
            results["propagated"] = self.collection.count_documents(
                {"cluster_metadata.session_type": "propagated"}
            )
            results["outliers"] = self.collection.count_documents(
                {"cluster_metadata.session_type": "outlier"}
            )
            
            # Conteggi per review status
            results["pending_review"] = self.collection.count_documents(
                {"review_status": "pending"}
            )
            results["auto_classified"] = self.collection.count_documents(
                {"review_status": "auto_classified"}
            )
            
            # Verifica copertura cluster
            cluster_pipeline = [
                {"$match": {"cluster_metadata": {"$exists": True}}},
                {"$group": {
                    "_id": "$cluster_metadata.cluster_id",
                    "count": {"$sum": 1},
                    "representatives": {
                        "$sum": {"$cond": [{"$eq": ["$cluster_metadata.is_representative", True]}, 1, 0]}
                    }
                }},
                {"$sort": {"count": -1}}
            ]
            
            cluster_coverage = list(self.collection.aggregate(cluster_pipeline))
            results["cluster_coverage"] = cluster_coverage[:10]  # Top 10 cluster
            
            # Determina status
            if results["total_documents"] == 0:
                results["status"] = "EMPTY"
            elif results["with_cluster_metadata"] == 0:
                results["status"] = "NO_TRAINING"
            elif results["representatives"] == 0:
                results["status"] = "NO_REPRESENTATIVES"
            elif results["representatives"] > 0 and results["cluster_coverage"]:
                results["status"] = "COMPLETE"
            else:
                results["status"] = "PARTIAL"
            
            # Report
            print(f"📊 Totale documenti: {results['total_documents']}")
            print(f"🏷️  Con cluster_metadata: {results['with_cluster_metadata']}")
            print(f"👤 Rappresentanti: {results['representatives']}")
            print(f"📈 Propagated: {results['propagated']}")
            print(f"🔍 Outliers: {results['outliers']}")
            print(f"⏳ Pending review: {results['pending_review']}")
            print(f"✅ Auto-classified: {results['auto_classified']}")
            print(f"📋 Top cluster: {len(results['cluster_coverage'])}")
            print(f"🎯 Status: {results['status']}")
            
        except Exception as e:
            print(f"❌ Errore test MongoDB: {e}")
            results["status"] = "ERROR"
            results["error"] = str(e)
        
        self.test_results["mongodb"] = results
        return results
    
    def test_api_coherence(self) -> Dict[str, Any]:
        """
        Scopo: Test coerenza API con MongoDB
        
        Returns:
            Risultati test coerenza API
            
        Ultimo aggiornamento: 2025-08-28
        """
        print("\\n🌐 TEST 2: VERIFICA COERENZA API")
        print("="*40)
        
        results = {
            "api_accessible": False,
            "review_queue_working": False,
            "filter_tests": {},
            "status": "UNKNOWN"
        }
        
        try:
            # Test 1: API accessibilità
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            results["api_accessible"] = response.status_code == 200
            print(f"🔌 API accessible: {results['api_accessible']}")
            
            if results["api_accessible"]:
                # Test 2: Review queue
                queue_url = f"{self.api_base_url}/api/review/{self.client_name}/cases"
                queue_params = {
                    "limit": 5,
                    "include_representatives": True,
                    "include_propagated": True,
                    "include_outliers": True
                }
                
                queue_response = requests.get(queue_url, params=queue_params, timeout=10)
                results["review_queue_working"] = queue_response.status_code == 200
                
                if results["review_queue_working"]:
                    queue_data = queue_response.json()
                    queue_count = len(queue_data.get("cases", []))
                    print(f"📋 Review queue: {queue_count} sessioni recuperate")
                    
                    # Test 3: Filtri specifici
                    filter_tests = {
                        "only_representatives": {"include_representatives": True, "include_propagated": False, "include_outliers": False},
                        "only_propagated": {"include_representatives": False, "include_propagated": True, "include_outliers": False},
                        "only_outliers": {"include_representatives": False, "include_propagated": False, "include_outliers": True}
                    }
                    
                    for test_name, params in filter_tests.items():
                        params["limit"] = 10
                        filter_response = requests.get(queue_url, params=params, timeout=10)
                        
                        if filter_response.status_code == 200:
                            filter_data = filter_response.json()
                            filter_count = len(filter_data.get("cases", []))
                            results["filter_tests"][test_name] = {
                                "success": True,
                                "count": filter_count
                            }
                            print(f"  🔸 {test_name}: {filter_count} risultati")
                        else:
                            results["filter_tests"][test_name] = {
                                "success": False,
                                "error": f"HTTP {filter_response.status_code}"
                            }
                            print(f"  ❌ {test_name}: HTTP {filter_response.status_code}")
                else:
                    print(f"❌ Review queue non funziona: HTTP {queue_response.status_code}")
            
            # Determina status finale
            if not results["api_accessible"]:
                results["status"] = "API_OFFLINE"
            elif not results["review_queue_working"]:
                results["status"] = "QUEUE_ERROR"
            elif all(test["success"] for test in results["filter_tests"].values()):
                results["status"] = "COMPLETE"
            else:
                results["status"] = "PARTIAL"
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Errore connessione API: {e}")
            results["status"] = "CONNECTION_ERROR"
            results["error"] = str(e)
        except Exception as e:
            print(f"❌ Errore test API: {e}")
            results["status"] = "ERROR"
            results["error"] = str(e)
        
        self.test_results["api"] = results
        return results
    
    def test_consensus_system(self) -> Dict[str, Any]:
        """
        Scopo: Test sistema di consensus intelligente
        
        Returns:
            Risultati test sistema consensus
            
        Ultimo aggiornamento: 2025-08-28
        """
        print("\\n🧠 TEST 3: VERIFICA SISTEMA CONSENSUS")
        print("="*45)
        
        results = {
            "clusters_with_consensus": 0,
            "auto_classified_clusters": 0,
            "pending_review_clusters": 0,
            "consensus_distribution": {},
            "status": "UNKNOWN"
        }
        
        try:
            # Pipeline per analizzare consensus per cluster
            consensus_pipeline = [
                {"$match": {"cluster_metadata": {"$exists": True}}},
                {"$group": {
                    "_id": "$cluster_metadata.cluster_id",
                    "review_status_counts": {
                        "$push": "$review_status"
                    },
                    "total_sessions": {"$sum": 1},
                    "representative_count": {
                        "$sum": {"$cond": [{"$eq": ["$cluster_metadata.is_representative", True]}, 1, 0]}
                    }
                }},
                {"$project": {
                    "cluster_id": "$_id",
                    "total_sessions": 1,
                    "representative_count": 1,
                    "auto_classified": {
                        "$size": {
                            "$filter": {
                                "input": "$review_status_counts",
                                "cond": {"$eq": ["$$this", "auto_classified"]}
                            }
                        }
                    },
                    "pending_review": {
                        "$size": {
                            "$filter": {
                                "input": "$review_status_counts",
                                "cond": {"$eq": ["$$this", "pending"]}
                            }
                        }
                    }
                }}
            ]
            
            consensus_data = list(self.collection.aggregate(consensus_pipeline))
            results["clusters_with_consensus"] = len(consensus_data)
            
            # Analizza distribuzione consensus
            auto_classified_clusters = 0
            pending_review_clusters = 0
            
            for cluster in consensus_data:
                if cluster["auto_classified"] > cluster["pending_review"]:
                    auto_classified_clusters += 1
                else:
                    pending_review_clusters += 1
            
            results["auto_classified_clusters"] = auto_classified_clusters
            results["pending_review_clusters"] = pending_review_clusters
            
            # Distribuzione percentuale
            if results["clusters_with_consensus"] > 0:
                auto_pct = (auto_classified_clusters / results["clusters_with_consensus"]) * 100
                pending_pct = (pending_review_clusters / results["clusters_with_consensus"]) * 100
                
                results["consensus_distribution"] = {
                    "auto_classified_pct": round(auto_pct, 1),
                    "pending_review_pct": round(pending_pct, 1)
                }
            
            # Determina status
            if results["clusters_with_consensus"] == 0:
                results["status"] = "NO_CLUSTERS"
            elif auto_classified_clusters > 0 and pending_review_clusters > 0:
                results["status"] = "BALANCED"
            elif auto_classified_clusters > pending_review_clusters:
                results["status"] = "AUTO_DOMINATED"
            else:
                results["status"] = "REVIEW_DOMINATED"
            
            print(f"🏷️  Cluster con consensus: {results['clusters_with_consensus']}")
            print(f"✅ Auto-classified: {auto_classified_clusters} ({results['consensus_distribution'].get('auto_classified_pct', 0)}%)")
            print(f"⏳ Pending review: {pending_review_clusters} ({results['consensus_distribution'].get('pending_review_pct', 0)}%)")
            print(f"🎯 Status consensus: {results['status']}")
            
        except Exception as e:
            print(f"❌ Errore test consensus: {e}")
            results["status"] = "ERROR"
            results["error"] = str(e)
        
        self.test_results["consensus"] = results
        return results
    
    def generate_final_report(self) -> str:
        """
        Scopo: Genera report finale sistematico
        
        Returns:
            Report finale completo
            
        Ultimo aggiornamento: 2025-08-28
        """
        print("\\n📊 REPORT FINALE SISTEMATICO")
        print("="*60)
        
        # Calcola status finale
        mongodb_ok = self.test_results.get("mongodb", {}).get("status") == "COMPLETE"
        api_ok = self.test_results.get("api", {}).get("status") == "COMPLETE"
        consensus_ok = self.test_results.get("consensus", {}).get("status") in ["BALANCED", "AUTO_DOMINATED"]
        
        overall_status = "COMPLETE" if (mongodb_ok and api_ok and consensus_ok) else "PARTIAL"
        if not mongodb_ok:
            overall_status = "MONGODB_ISSUES"
        elif not api_ok:
            overall_status = "API_ISSUES"
        elif not consensus_ok:
            overall_status = "CONSENSUS_ISSUES"
        
        report = f"""
🎯 STATUS FINALE SISTEMA: {overall_status}
{'✅' if overall_status == 'COMPLETE' else '⚠️'}

📋 MONGODB DATA STRUCTURE:
   Status: {self.test_results.get('mongodb', {}).get('status', 'N/A')}
   Documenti totali: {self.test_results.get('mongodb', {}).get('total_documents', 0)}
   Rappresentanti: {self.test_results.get('mongodb', {}).get('representatives', 0)}
   Auto-classified: {self.test_results.get('mongodb', {}).get('auto_classified', 0)}

🌐 API COHERENCE:
   Status: {self.test_results.get('api', {}).get('status', 'N/A')}
   API accessible: {self.test_results.get('api', {}).get('api_accessible', False)}
   Review queue: {self.test_results.get('api', {}).get('review_queue_working', False)}
   Filtri funzionanti: {len([t for t in self.test_results.get('api', {}).get('filter_tests', {}).values() if t.get('success')])}

🧠 CONSENSUS SYSTEM:
   Status: {self.test_results.get('consensus', {}).get('status', 'N/A')}
   Cluster con consensus: {self.test_results.get('consensus', {}).get('clusters_with_consensus', 0)}
   Auto-classified: {self.test_results.get('consensus', {}).get('consensus_distribution', {}).get('auto_classified_pct', 0)}%
   
⏰ Test completato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{"🎉 SISTEMA COMPLETAMENTE FUNZIONALE!" if overall_status == 'COMPLETE' else "⚠️  Sistema richiede attenzione"}
        """
        
        print(report)
        return report
    
    def run_complete_verification(self) -> Dict[str, Any]:
        """
        Scopo: Esegue verifica sistematica completa
        
        Returns:
            Risultati completi di tutti i test
            
        Ultimo aggiornamento: 2025-08-28
        """
        print("🚀 AVVIO VERIFICA SISTEMATICA COMPLETA")
        print("="*60)
        print(f"🏢 Cliente: {self.client_name}")
        print(f"🆔 Tenant: {self.tenant_id}")
        print(f"⏰ Inizio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Connessione MongoDB
        if not self.connect_mongodb():
            return {"status": "CONNECTION_ERROR", "error": "MongoDB connection failed"}
        
        # Esegui tutti i test
        try:
            self.test_mongodb_data_structure()
            self.test_api_coherence()
            self.test_consensus_system()
            
            # Report finale
            final_report = self.generate_final_report()
            
            return {
                "status": "SUCCESS",
                "results": self.test_results,
                "report": final_report
            }
            
        except Exception as e:
            print(f"❌ Errore durante verifica: {e}")
            return {"status": "ERROR", "error": str(e)}


def main():
    """
    Main function per test sistematico
    """
    tester = SystematicVerificationTester(
        client_name="wopta",
        tenant_id="16c222a9-f293-11ef-9315-96000228e7fe"
    )
    
    results = tester.run_complete_verification()
    
    if results["status"] == "SUCCESS":
        print("\\n✅ Verifica sistematica completata con successo")
        return 0
    else:
        print(f"\\n❌ Verifica fallita: {results.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    exit(main())
