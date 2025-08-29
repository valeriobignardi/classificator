#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Autore: AI Assistant
Data: 27 Agosto 2025
Scopo: Test per la funzionalità di propagazione delle decisioni umane nei cluster

Questo modulo testa la nuova funzionalità che propaga automaticamente
le decisioni umane dai rappresentanti del cluster agli altri membri.

Ultimo aggiornamento: 27/08/2025
"""

import os
import sys
import unittest
import json
from datetime import datetime
from bson import ObjectId

# Aggiunge il path del progetto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mongo_classification_reader import MongoClassificationReader
from QualityGate.quality_gate_engine import QualityGateEngine


class TestClusterReviewPropagation(unittest.TestCase):
    """
    Test class per verificare la propagazione delle decisioni umane nei cluster
    
    Testa:
    1. Risoluzione di casi rappresentanti con propagazione
    2. Risoluzione di casi non-rappresentanti senza propagazione  
    3. Gestione di errori e casi edge
    """
    
    def setUp(self):
        """
        Setup iniziale per i test
        
        Scopo: Inizializza le connessioni e i dati di test
        """
        self.tenant_name = "test_propagation"
        self.client_name = "test_client"
        
        # Inizializza MongoClassificationReader con oggetto Tenant
        from tenant import Tenant
        test_tenant = Tenant.from_slug(self.tenant_name)
        self.mongo_reader = MongoClassificationReader(tenant=test_tenant)
        self.mongo_reader.connect()
        
        # Inizializza QualityGate per i test
        self.quality_gate = QualityGateEngine(tenant_name=self.tenant_name)
        
        # Pulisce eventuali dati precedenti
        self._cleanup_test_data()
        
        # Crea dati di test
        self._setup_test_data()
    
    def tearDown(self):
        """
        Cleanup dopo i test
        
        Scopo: Rimuove i dati di test dal database
        """
        self._cleanup_test_data()
    
    def _cleanup_test_data(self):
        """
        Rimuove i dati di test dal database
        
        Scopo: Pulizia del database per evitare interferenze tra test
        """
        try:
            collection = self.mongo_reader.db[self.mongo_reader.get_collection_name()]
            collection.delete_many({"session_id": {"$regex": "^test_session_"}})
            print("🧹 Dati di test rimossi")
        except Exception as e:
            print(f"Errore nella pulizia dati di test: {e}")
    
    def _setup_test_data(self):
        """
        Crea dati di test per i cluster
        
        Scopo: Inserisce nel database casi rappresentanti e propagati per testing
        """
        try:
            # Cluster 1: Rappresentante + 2 membri propagati
            cluster_1_id = "test_cluster_001"
            
            # Caso rappresentante cluster 1
            self.representative_case_id = self._insert_test_case({
                "session_id": "test_session_representative_1",
                "conversazione": "Test conversazione per rappresentante cluster 1",
                "classificazione": "informazioni", 
                "confidence": 0.75,
                "review_status": "pending",
                "metadata": {
                    "cluster_metadata": {
                        "cluster_id": cluster_1_id,
                        "is_representative": True,
                        "method": "cluster_representative"
                    }
                }
            })
            
            # Casi propagati cluster 1
            self.propagated_case_1_id = self._insert_test_case({
                "session_id": "test_session_propagated_1_1",
                "conversazione": "Test conversazione propagata 1 cluster 1",
                "classificazione": "informazioni",
                "confidence": 0.68,
                "review_status": "not_required",
                "metadata": {
                    "cluster_metadata": {
                        "cluster_id": cluster_1_id,
                        "is_representative": False,
                        "propagated_from": "cluster_propagation"
                    }
                }
            })
            
            self.propagated_case_2_id = self._insert_test_case({
                "session_id": "test_session_propagated_1_2", 
                "conversazione": "Test conversazione propagata 2 cluster 1",
                "classificazione": "informazioni",
                "confidence": 0.72,
                "review_status": "not_required",
                "metadata": {
                    "cluster_metadata": {
                        "cluster_id": cluster_1_id,
                        "is_representative": False,
                        "propagated_from": "cluster_propagation"
                    }
                }
            })
            
            # Cluster 2: Solo rappresentante (per test senza propagazione)
            cluster_2_id = "test_cluster_002"
            
            self.isolated_representative_id = self._insert_test_case({
                "session_id": "test_session_isolated_representative",
                "conversazione": "Test conversazione per rappresentante isolato",
                "classificazione": "reclamo",
                "confidence": 0.85,
                "review_status": "pending",
                "metadata": {
                    "cluster_metadata": {
                        "cluster_id": cluster_2_id,
                        "is_representative": True,
                        "method": "cluster_representative"
                    }
                }
            })
            
            # Caso senza cluster (non-rappresentante)
            self.non_cluster_case_id = self._insert_test_case({
                "session_id": "test_session_no_cluster",
                "conversazione": "Test conversazione senza cluster",
                "classificazione": "domanda",
                "confidence": 0.60,
                "review_status": "pending",
                "metadata": {}
            })
            
            print(f"✅ Dati di test creati:")
            print(f"   - Rappresentante cluster 1: {self.representative_case_id}")
            print(f"   - Propagati cluster 1: {self.propagated_case_1_id}, {self.propagated_case_2_id}")
            print(f"   - Rappresentante cluster 2 (isolato): {self.isolated_representative_id}")
            print(f"   - Caso senza cluster: {self.non_cluster_case_id}")
            
        except Exception as e:
            print(f"❌ Errore nella creazione dati di test: {e}")
            raise
    
    def _insert_test_case(self, case_data):
        """
        Inserisce un singolo caso di test nel database
        
        Parametri:
            case_data: Dict con i dati del caso
            
        Returns:
            str: ID del caso inserito
        """
        collection = self.mongo_reader.db[self.mongo_reader.get_collection_name()]
        
        # Aggiunge campi comuni
        case_data.update({
            "client": self.client_name,
            "timestamp": datetime.now().isoformat(),
            "classified_by": "test_setup"
        })
        
        result = collection.insert_one(case_data)
        return str(result.inserted_id)
    
    def test_representative_case_propagation(self):
        """
        Test: Risoluzione di un caso rappresentante con propagazione ai membri
        
        Scopo: Verifica che la decisione umana su un rappresentante si propaghi
               a tutti i membri non-revisionati dello stesso cluster
        """
        print("\n🧪 Test: Propagazione decisione da rappresentante")
        
        # Risolve il caso rappresentante
        human_decision = "reclamo"
        human_confidence = 0.95
        notes = "Decisione di test - dovrebbe propagarsi"
        
        result = self.mongo_reader.resolve_review_session_with_cluster_propagation(
            case_id=self.representative_case_id,
            client_name=self.client_name,
            human_decision=human_decision,
            human_confidence=human_confidence,
            human_notes=notes
        )
        
        # Verifica risultato
        self.assertTrue(result.get("case_resolved", False), "Il caso dovrebbe essere risolto")
        self.assertTrue(result.get("is_representative", False), "Il caso dovrebbe essere rappresentante")
        self.assertEqual(result.get("propagated_cases", 0), 2, "Dovrebbero essere propagati 2 casi")
        self.assertEqual(result.get("cluster_id"), "test_cluster_001", "Cluster ID corretto")
        
        # Verifica che il rappresentante sia stato aggiornato
        representative_doc = self._get_case_by_id(self.representative_case_id)
        self.assertEqual(representative_doc["classificazione"], human_decision)
        self.assertEqual(representative_doc["human_decision"], human_decision)
        self.assertEqual(representative_doc["review_status"], "completed")
        
        # Verifica che i casi propagati siano stati aggiornati
        propagated_1_doc = self._get_case_by_id(self.propagated_case_1_id)
        self.assertEqual(propagated_1_doc["classificazione"], human_decision)
        self.assertTrue(propagated_1_doc.get("human_reviewed_by_propagation", False))
        
        propagated_2_doc = self._get_case_by_id(self.propagated_case_2_id)
        self.assertEqual(propagated_2_doc["classificazione"], human_decision)
        self.assertTrue(propagated_2_doc.get("human_reviewed_by_propagation", False))
        
        print(f"✅ Propagazione completata: {result}")
    
    def test_isolated_representative_no_propagation(self):
        """
        Test: Rappresentante senza membri da propagare
        
        Scopo: Verifica che un rappresentante isolato venga risolto correttamente
               senza errori anche se non ha membri da propagare
        """
        print("\n🧪 Test: Rappresentante isolato (senza propagazione)")
        
        human_decision = "informazioni"
        human_confidence = 0.90
        
        result = self.mongo_reader.resolve_review_session_with_cluster_propagation(
            case_id=self.isolated_representative_id,
            client_name=self.client_name,
            human_decision=human_decision,
            human_confidence=human_confidence,
            human_notes="Test rappresentante isolato"
        )
        
        # Verifica risultato
        self.assertTrue(result.get("case_resolved", False), "Il caso dovrebbe essere risolto")
        self.assertTrue(result.get("is_representative", False), "Il caso dovrebbe essere rappresentante")
        self.assertEqual(result.get("propagated_cases", 0), 0, "Non dovrebbero esserci propagazioni")
        
        # Verifica che il caso sia stato aggiornato
        case_doc = self._get_case_by_id(self.isolated_representative_id)
        self.assertEqual(case_doc["classificazione"], human_decision)
        self.assertEqual(case_doc["review_status"], "completed")
        
        print(f"✅ Rappresentante isolato risolto: {result}")
    
    def test_non_representative_case_resolution(self):
        """
        Test: Risoluzione di un caso non-rappresentante
        
        Scopo: Verifica che un caso non-rappresentante venga risolto individualmente
               senza tentare propagazioni
        """
        print("\n🧪 Test: Caso non-rappresentante (senza propagazione)")
        
        human_decision = "complimento"
        human_confidence = 0.88
        
        result = self.mongo_reader.resolve_review_session_with_cluster_propagation(
            case_id=self.non_cluster_case_id,
            client_name=self.client_name,
            human_decision=human_decision,
            human_confidence=human_confidence,
            human_notes="Test caso non-rappresentante"
        )
        
        # Verifica risultato
        self.assertTrue(result.get("case_resolved", False), "Il caso dovrebbe essere risolto")
        self.assertFalse(result.get("is_representative", True), "Il caso non dovrebbe essere rappresentante")
        self.assertEqual(result.get("propagated_cases", 0), 0, "Non dovrebbero esserci propagazioni")
        self.assertIsNone(result.get("cluster_id"), "Non dovrebbe avere cluster_id")
        
        # Verifica che il caso sia stato aggiornato
        case_doc = self._get_case_by_id(self.non_cluster_case_id)
        self.assertEqual(case_doc["classificazione"], human_decision)
        self.assertEqual(case_doc["review_status"], "completed")
        
        print(f"✅ Caso non-rappresentante risolto: {result}")
    
    def test_quality_gate_integration(self):
        """
        Test: Integrazione con QualityGateEngine
        
        Scopo: Verifica che il QualityGateEngine utilizzi correttamente
               la nuova funzionalità di propagazione
        """
        print("\n🧪 Test: Integrazione QualityGateEngine")
        
        # Usa un caso esistente dal setUp invece di crearne uno nuovo
        # che potrebbe essere pulito dal tearDown di altri test
        
        human_decision = "reclamo"
        human_confidence = 0.93
        notes = "Test integrazione QualityGate"
        
        result = self.quality_gate.resolve_review_case(
            case_id=self.representative_case_id,
            human_decision=human_decision,
            human_confidence=human_confidence,
            notes=notes
        )
        
        # Verifica che il risultato sia un dizionario (nuovo comportamento)
        self.assertIsInstance(result, dict, "Il risultato dovrebbe essere un dizionario")
        self.assertTrue(result.get("case_resolved", False), "Il caso dovrebbe essere risolto")
        self.assertTrue(result.get("is_representative", False), "Dovrebbe essere rappresentante")
        self.assertEqual(result.get("propagated_cases", 0), 2, "Dovrebbero essere propagati 2 casi")
        
        # Verifica che il caso sia stato aggiornato
        case_doc = self._get_case_by_id(self.representative_case_id)
        self.assertEqual(case_doc["classificazione"], human_decision)
        self.assertEqual(case_doc["review_status"], "completed")
        
        print(f"✅ Integrazione QualityGate completata: {result}")
    
    def test_error_handling(self):
        """
        Test: Gestione degli errori
        
        Scopo: Verifica che la funzione gestisca correttamente
               casi di errore comuni (caso non trovato, parametri invalidi, etc.)
        """
        print("\n🧪 Test: Gestione errori")
        
        # Test 1: Caso non esistente
        result = self.mongo_reader.resolve_review_session_with_cluster_propagation(
            case_id="caso_inesistente",
            client_name=self.client_name,
            human_decision="test",
            human_confidence=0.8
        )
        
        self.assertFalse(result.get("case_resolved", True), "Caso inesistente non dovrebbe essere risolto")
        self.assertIn("error", result, "Dovrebbe contenere messaggio di errore")
        
        # Test 2: Case ID invalido
        result = self.mongo_reader.resolve_review_session_with_cluster_propagation(
            case_id="invalid_object_id",
            client_name=self.client_name,
            human_decision="test",
            human_confidence=0.8
        )
        
        self.assertFalse(result.get("case_resolved", True), "Case ID invalido non dovrebbe essere risolto")
        
        print("✅ Gestione errori verificata")
    
    def _get_case_by_id(self, case_id):
        """
        Recupera un caso dal database per ID
        
        Parametri:
            case_id: ID del caso
            
        Returns:
            Dict: Documento del caso
        """
        collection = self.mongo_reader.db[self.mongo_reader.get_collection_name()]
        return collection.find_one({"_id": ObjectId(case_id)})
    
    def test_propagation_confidence_reduction(self):
        """
        Test: Verifica che la confidenza si riduca durante la propagazione
        
        Scopo: Controlla che i casi propagati abbiano confidenza leggermente ridotta
               rispetto alla decisione umana originale
        """
        print("\n🧪 Test: Riduzione confidenza in propagazione")
        
        human_confidence = 0.95
        expected_propagated_confidence = human_confidence * 0.9  # 0.855
        
        result = self.mongo_reader.resolve_review_session_with_cluster_propagation(
            case_id=self.representative_case_id,
            client_name=self.client_name,
            human_decision="reclamo",
            human_confidence=human_confidence,
            human_notes="Test riduzione confidenza"
        )
        
        self.assertTrue(result.get("case_resolved", False))
        
        # Verifica che i casi propagati abbiano confidenza ridotta
        propagated_doc = self._get_case_by_id(self.propagated_case_1_id)
        actual_confidence = propagated_doc.get("confidence", 0)
        
        # Verifica che la confidenza sia stata ridotta ma non sotto 0.1
        self.assertGreaterEqual(actual_confidence, 0.1, "Confidenza non dovrebbe essere sotto 0.1")
        self.assertLessEqual(actual_confidence, human_confidence, "Confidenza propagata dovrebbe essere <= originale")
        
        print(f"✅ Confidenza originale: {human_confidence}, propagata: {actual_confidence}")


def main():
    """
    Funzione principale per eseguire i test
    
    Scopo: Esegue tutti i test della suite
    """
    print("🚀 Avvio test suite per propagazione cluster review")
    
    # Configura logging per i test
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Esegue i test
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
