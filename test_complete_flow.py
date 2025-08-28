#!/usr/bin/env python3
"""
File: test_complete_flow.py
Autore: Valerio Bignardi
Data creazione: 2025-01-17
Descrizione: Test completo del flusso di supervisione con logica intelligente propagated
Storia aggiornamenti:
- 2025-01-17: Creazione iniziale per test completo supervised training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from pymongo import MongoClient
from Pipeline.end_to_end_pipeline import EndToEndPipeline
from mongo_classification_reader import MongoClassificationReader
import yaml
from Utils.tenant import Tenant

class CompleteFlowTester:
    """
    Test completo del flusso supervised training con logica propagated intelligente
    """
    
    def __init__(self):
        """
        Inizializza il tester
        """
        print("🧪 TEST FLUSSO COMPLETO SUPERVISIONE INTELLIGENTE")
        
        # Carica configurazione
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup connessione MongoDB
        mongodb_config = self.config.get('mongodb', {})
        mongodb_url = mongodb_config.get('url', 'mongodb://localhost:27017')
        database_name = mongodb_config.get('database', 'classificazioni')
        
        self.mongo_client = MongoClient(mongodb_url)
        self.db = self.mongo_client[database_name]
        
        # Setup reader
        self.reader = MongoClassificationReader()
        
        # Setup tenant
        self.tenant = Tenant.from_slug("humanitas")
        print(f"✅ Tenant risolto: {self.tenant.tenant_name} ({self.tenant.tenant_id})")
    
    def setup_test_data(self):
        """
        Crea dati di test nel database
        """
        print("\n📊 SETUP DATI DI TEST")
        
        # Crea cluster con rappresentativi e propagated
        test_cluster_id = "test_cluster_consensus"
        
        # Cancella dati esistenti
        self.db.classifications.delete_many({
            "cluster_metadata.cluster_id": test_cluster_id
        })
        
        # Crea rappresentativi con diverse etichette
        representatives = [
            {
                "session_id": "rep_1",
                "client_name": self.tenant.tenant_id,
                "predicted_label": "info_polizza",
                "confidence": 0.9,
                "review_status": "pending",  # Da revieware
                "cluster_metadata": {
                    "cluster_id": test_cluster_id,
                    "session_type": "representative",
                    "is_representative": True
                }
            },
            {
                "session_id": "rep_2", 
                "client_name": self.tenant.tenant_id,
                "predicted_label": "info_polizza",
                "confidence": 0.85,
                "review_status": "pending",
                "cluster_metadata": {
                    "cluster_id": test_cluster_id,
                    "session_type": "representative",
                    "is_representative": True
                }
            },
            {
                "session_id": "rep_3",
                "client_name": self.tenant.tenant_id,
                "predicted_label": "reclamo",
                "confidence": 0.8,
                "review_status": "pending",
                "cluster_metadata": {
                    "cluster_id": test_cluster_id,
                    "session_type": "representative", 
                    "is_representative": True
                }
            }
        ]
        
        # Crea propagated
        propagated_sessions = [
            {
                "session_id": "prop_1",
                "client_name": self.tenant.tenant_id,
                "predicted_label": None,
                "confidence": None,
                "review_status": "pending",
                "cluster_metadata": {
                    "cluster_id": test_cluster_id,
                    "session_type": "propagated",
                    "is_representative": False
                }
            },
            {
                "session_id": "prop_2",
                "client_name": self.tenant.tenant_id,
                "predicted_label": None,
                "confidence": None,
                "review_status": "pending",
                "cluster_metadata": {
                    "cluster_id": test_cluster_id,
                    "session_type": "propagated",
                    "is_representative": False
                }
            }
        ]
        
        # Inserisci i dati
        self.db.classifications.insert_many(representatives)
        self.db.classifications.insert_many(propagated_sessions)
        
        print(f"✅ Creati {len(representatives)} rappresentativi")
        print(f"✅ Creati {len(propagated_sessions)} propagated")
        print(f"🔍 Cluster ID: {test_cluster_id}")
        
        return test_cluster_id
    
    def simulate_human_review(self, cluster_id: str):
        """
        Simula review umana dei rappresentativi
        """
        print("\n👤 SIMULAZIONE REVIEW UMANA")
        
        # Review rappresentativo 1: conferma info_polizza
        self.db.classifications.update_one(
            {"session_id": "rep_1"},
            {
                "$set": {
                    "review_status": "auto_classified",
                    "human_label": "info_polizza",
                    "reviewed_by": "test_user",
                    "reviewed_at": "2025-01-17T10:00:00Z"
                }
            }
        )
        print("✅ Rep 1: Confermato 'info_polizza'")
        
        # Review rappresentativo 2: conferma info_polizza  
        self.db.classifications.update_one(
            {"session_id": "rep_2"},
            {
                "$set": {
                    "review_status": "auto_classified",
                    "human_label": "info_polizza", 
                    "reviewed_by": "test_user",
                    "reviewed_at": "2025-01-17T10:01:00Z"
                }
            }
        )
        print("✅ Rep 2: Confermato 'info_polizza'")
        
        # Review rappresentativo 3: conferma reclamo
        self.db.classifications.update_one(
            {"session_id": "rep_3"},
            {
                "$set": {
                    "review_status": "auto_classified",
                    "human_label": "reclamo",
                    "reviewed_by": "test_user", 
                    "reviewed_at": "2025-01-17T10:02:00Z"
                }
            }
        )
        print("✅ Rep 3: Confermato 'reclamo'")
        
        # Verifica consenso: 2/3 = 66.7% < 70% → Review necessaria
        print("📊 Consenso atteso: 66.7% (2/3) < 70% → Review necessaria")
    
    def test_propagated_update(self, cluster_id: str):
        """
        Testa l'aggiornamento automatico dei propagated
        """
        print("\n🔄 TEST AGGIORNAMENTO PROPAGATED")
        
        # Inizializza pipeline
        pipeline = EndToEndPipeline(
            tenant_slug="humanitas",
            confidence_threshold=0.7,
            auto_mode=False
        )
        
        # Testa update dopo review
        result = pipeline.update_propagated_after_review("rep_1")
        print(f"📤 Risultato update: {result}")
        
        # Verifica stato propagated
        propagated = list(self.db.classifications.find({
            "cluster_metadata.cluster_id": cluster_id,
            "cluster_metadata.session_type": "propagated"
        }))
        
        print(f"\n📋 STATO PROPAGATED DOPO UPDATE:")
        for prop in propagated:
            session_id = prop['session_id']
            review_status = prop['review_status']
            predicted_label = prop.get('predicted_label')
            print(f"  {session_id}: {review_status}, label: {predicted_label}")
            
            # Dovrebbero essere ancora pending (consenso < 70%)
            if review_status == "pending":
                print(f"  ✅ {session_id}: Correttamente in pending (consenso < 70%)")
            else:
                print(f"  ❌ {session_id}: Errore - dovrebbe essere pending!")
    
    def verify_react_interface_data(self, cluster_id: str):
        """
        Verifica che i dati siano corretti per l'interfaccia React
        """
        print("\n🔍 VERIFICA DATI PER INTERFACCIA REACT")
        
        # Conta per tipo
        representatives = self.db.classifications.count_documents({
            "cluster_metadata.cluster_id": cluster_id,
            "cluster_metadata.session_type": "representative"
        })
        
        propagated = self.db.classifications.count_documents({
            "cluster_metadata.cluster_id": cluster_id, 
            "cluster_metadata.session_type": "propagated"
        })
        
        outliers = self.db.classifications.count_documents({
            "cluster_metadata.cluster_id": cluster_id,
            "cluster_metadata.session_type": "outlier"
        })
        
        print(f"📊 Conteggi per filtri React:")
        print(f"  👥 Representatives: {representatives}")
        print(f"  🔄 Propagated: {propagated}")  
        print(f"  ⚡ Outliers: {outliers}")
        
        # Verifica etichette "Ereditata"
        pending_propagated = list(self.db.classifications.find({
            "cluster_metadata.cluster_id": cluster_id,
            "cluster_metadata.session_type": "propagated",
            "review_status": "pending"
        }))
        
        print(f"\n🏷️ VERIFICA ETICHETTE 'EREDITATA':")
        for prop in pending_propagated:
            session_id = prop['session_id']
            has_predicted = prop.get('predicted_label') is not None
            
            if not has_predicted:
                print(f"  ✅ {session_id}: Nessuna etichetta → Non mostrerà 'Ereditata'")
            else:
                print(f"  ⚠️ {session_id}: Ha etichetta → Potrebbe mostrare 'Ereditata'")
    
    def cleanup(self, cluster_id: str):
        """
        Pulisce i dati di test
        """
        print("\n🧹 CLEANUP DATI DI TEST")
        deleted = self.db.classifications.delete_many({
            "cluster_metadata.cluster_id": cluster_id
        })
        print(f"✅ Eliminati {deleted.deleted_count} record di test")
    
    async def run_complete_test(self):
        """
        Esegue il test completo
        """
        try:
            # Setup
            cluster_id = self.setup_test_data()
            
            # Simula review umana
            self.simulate_human_review(cluster_id)
            
            # Testa aggiornamento propagated
            self.test_propagated_update(cluster_id)
            
            # Verifica dati interfaccia
            self.verify_react_interface_data(cluster_id)
            
            print("\n🎉 TEST COMPLETO COMPLETATO CON SUCCESSO!")
            
        except Exception as e:
            print(f"\n❌ ERRORE DURANTE IL TEST: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            self.cleanup(cluster_id)
            self.mongo_client.close()

async def main():
    """
    Funzione main per eseguire i test
    """
    tester = CompleteFlowTester()
    await tester.run_complete_test()

if __name__ == "__main__":
    asyncio.run(main())
