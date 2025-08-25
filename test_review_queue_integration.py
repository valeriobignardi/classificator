#!/usr/bin/env python3
"""
File Header:
    - Autore: GitHub Copilot Assistant
    - Data creazione: 2025-01-27
    - Scopo: Test integrazione completa Review Queue a 3 livelli
    - Aggiornamenti: 
        * 2025-01-27: Creazione iniziale test API + MongoDB integration

Test di verifica dell'integrazione completa Review Queue:
1. Test metodo save_classification_result() con cluster_metadata
2. Test metodo get_review_queue_sessions() con filtri 3 livelli
3. Test API endpoint /api/review/<client>/cases con nuovi parametri
4. Verifica formattazione output con nuovi campi Review Queue
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

# Aggiungi path progetto
sys.path.append('/home/ubuntu/classificazione_discussioni_bck_23_08_2025')

# Import componenti sistema
from mongo_classification_reader import MongoClassificationReader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewQueueIntegrationTest:
    """
    Scopo: Test completo integrazione Review Queue a 3 livelli
    
    Metodi:
        - setup_test_data(): Crea dati di test con cluster metadata
        - test_save_with_cluster_metadata(): Test salvataggio con metadata
        - test_review_queue_filtering(): Test filtri 3 livelli 
        - test_api_integration(): Simula chiamate API con nuovi parametri
        
    Data ultima modifica: 2025-01-27
    """
    
    def __init__(self, client_name: str = "test_client"):
        """
        Scopo: Inizializza test environment
        
        Parametri input:
            - client_name: Nome cliente per test
            
        Data ultima modifica: 2025-01-27
        """
        self.client_name = client_name
        self.reader = MongoClassificationReader()
        self.test_sessions = []
        
    def setup_test_data(self) -> List[Dict[str, Any]]:
        """
        Scopo: Crea dati di test per Review Queue con cluster metadata
        
        Returns:
            - List[Dict]: Sessioni di test con diversi tipi (representative/propagated/outlier)
            
        Data ultima modifica: 2025-01-27
        """
        test_data = [
            # 1. RAPPRESENTANTE CLUSTER - pending review
            {
                'session_id': 'test_repr_001',
                'conversation_text': 'Conversazione rappresentante cluster 1',
                'classification': 'test_label_1',
                'confidence': 0.85,
                'review_status': 'pending',
                'cluster_metadata': {
                    'cluster_id': 5,
                    'is_representative': True,
                    'cluster_size': 15,
                    'confidence': 0.85
                }
            },
            
            # 2. SESSIONE PROPAGATA - da cluster
            {
                'session_id': 'test_prop_001',
                'conversation_text': 'Conversazione propagata dal cluster 5',
                'classification': 'test_label_1',
                'confidence': 0.75,
                'review_status': 'not_required',
                'cluster_metadata': {
                    'cluster_id': 5,
                    'is_representative': False,
                    'propagated_from': 'test_repr_001',
                    'propagation_confidence': 0.75
                }
            },
            
            # 3. OUTLIER - cluster -1
            {
                'session_id': 'test_outlier_001', 
                'conversation_text': 'Conversazione outlier non in cluster',
                'classification': 'test_label_2',
                'confidence': 0.45,
                'review_status': 'pending',
                'cluster_metadata': {
                    'cluster_id': -1,
                    'is_representative': False,
                    'outlier_score': 0.95
                }
            }
        ]
        
        self.test_sessions = test_data
        return test_data
    
    def test_save_with_cluster_metadata(self) -> bool:
        """
        Scopo: Test salvataggio classification result con cluster_metadata
        
        Returns:
            - bool: True se test passa, False altrimenti
            
        Data ultima modifica: 2025-01-27
        """
        logger.info("🔄 Test: save_classification_result() con cluster_metadata")
        
        try:
            for session_data in self.test_sessions:
                # Test nuovo metodo con cluster_metadata - FIRMA CORRETTA
                final_decision = {
                    'classification': session_data['classification'],
                    'confidence': session_data['confidence'],
                    'review_status': session_data['review_status']
                }
                
                result = self.reader.save_classification_result(
                    session_id=session_data['session_id'],
                    client_name=self.client_name,
                    final_decision=final_decision,
                    conversation_text=session_data['conversation_text'],
                    needs_review=(session_data['review_status'] == 'pending'),
                    cluster_metadata=session_data['cluster_metadata']
                )
                
                if result:
                    logger.info(f"✅ Salvato: {session_data['session_id']} con cluster_metadata")
                else:
                    logger.error(f"❌ Errore salvataggio: {session_data['session_id']}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"❌ Errore test save_classification_result: {e}")
            return False
    
    def test_review_queue_filtering(self) -> bool:
        """
        Scopo: Test filtri Review Queue a 3 livelli
        
        Returns:
            - bool: True se tutti i filtri funzionano correttamente
            
        Data ultima modifica: 2025-01-27
        """
        logger.info("🔄 Test: get_review_queue_sessions() filtri 3 livelli")
        
        try:
            # Test 1: Solo rappresentanti
            representatives = self.reader.get_review_queue_sessions(
                client_name=self.client_name,
                show_representatives=True,
                show_propagated=False,
                show_outliers=False
            )
            logger.info(f"📊 Rappresentanti trovati: {len(representatives)}")
            
            # Test 2: Solo propagate  
            propagated = self.reader.get_review_queue_sessions(
                client_name=self.client_name,
                show_representatives=False,
                show_propagated=True,
                show_outliers=False
            )
            logger.info(f"📊 Propagate trovate: {len(propagated)}")
            
            # Test 3: Solo outliers
            outliers = self.reader.get_review_queue_sessions(
                client_name=self.client_name,
                show_representatives=False,
                show_propagated=False, 
                show_outliers=True
            )
            logger.info(f"📊 Outliers trovati: {len(outliers)}")
            
            # Test 4: Tutti insieme
            all_sessions = self.reader.get_review_queue_sessions(
                client_name=self.client_name,
                show_representatives=True,
                show_propagated=True,
                show_outliers=True
            )
            logger.info(f"📊 Totale sessioni: {len(all_sessions)}")
            
            # Validazione: totale dovrebbe essere somma delle parti
            expected_total = len(representatives) + len(propagated) + len(outliers)
            if len(all_sessions) >= expected_total:
                logger.info("✅ Test filtri Review Queue PASSATO")
                return True
            else:
                logger.error("❌ Inconsistenza nei filtri Review Queue")
                return False
                
        except Exception as e:
            logger.error(f"❌ Errore test review_queue_filtering: {e}")
            return False
    
    def test_api_integration_simulation(self) -> bool:
        """
        Scopo: Simula chiamate API con nuovi parametri Review Queue
        
        Returns:
            - bool: True se simulazione API passa
            
        Data ultima modifica: 2025-01-27
        """
        logger.info("🔄 Test: Simulazione API integration")
        
        try:
            # Simula comportamento API endpoint
            test_scenarios = [
                {'show_representatives': True, 'show_propagated': False, 'show_outliers': False},
                {'show_representatives': False, 'show_propagated': True, 'show_outliers': False},
                {'show_representatives': False, 'show_propagated': False, 'show_outliers': True},
                {'show_representatives': True, 'show_propagated': True, 'show_outliers': True}
            ]
            
            for i, scenario in enumerate(test_scenarios, 1):
                logger.info(f"📡 Test scenario API {i}: {scenario}")
                
                # Recupera sessioni con parametri scenario
                sessions = self.reader.get_review_queue_sessions(
                    client_name=self.client_name,
                    **scenario
                )
                
                # Simula formattazione API response
                formatted_cases = []
                for session in sessions:
                    case_item = {
                        'session_id': session.get('session_id', ''),
                        'cluster_id': session.get('cluster_id'),
                        'session_type': session.get('session_type', 'unknown'),
                        'is_representative': session.get('is_representative', False),
                        'propagated_from': session.get('propagated_from'),
                        'review_status': session.get('review_status', 'not_required')
                    }
                    formatted_cases.append(case_item)
                
                logger.info(f"✅ Scenario {i}: {len(formatted_cases)} casi formattati")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Errore test API integration: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """
        Scopo: Esegue tutti i test Review Queue integration
        
        Returns:
            - Dict[str, bool]: Risultati di tutti i test
            
        Data ultima modifica: 2025-01-27
        """
        logger.info("🚀 INIZIO TEST INTEGRATION REVIEW QUEUE")
        
        results = {}
        
        # Setup test data
        logger.info("📋 Setup dati test...")
        self.setup_test_data()
        
        # Test 1: Salvataggio con cluster metadata
        results['save_cluster_metadata'] = self.test_save_with_cluster_metadata()
        
        # Test 2: Filtri Review Queue
        results['review_queue_filtering'] = self.test_review_queue_filtering()
        
        # Test 3: API integration simulation
        results['api_integration'] = self.test_api_integration_simulation()
        
        # Report finale
        logger.info("📊 RISULTATI TEST:")
        for test_name, passed in results.items():
            status = "✅ PASSATO" if passed else "❌ FALLITO"
            logger.info(f"   {test_name}: {status}")
        
        all_passed = all(results.values())
        final_status = "✅ TUTTI I TEST PASSATI" if all_passed else "❌ ALCUNI TEST FALLITI"
        logger.info(f"🏁 RISULTATO FINALE: {final_status}")
        
        return results

def main():
    """
    Scopo: Entry point per test Review Queue integration
    
    Data ultima modifica: 2025-01-27
    """
    print("=" * 60)
    print("TEST INTEGRATION REVIEW QUEUE COMPLETO")
    print("=" * 60)
    
    # Esegui test
    tester = ReviewQueueIntegrationTest("test_humanitas")
    results = tester.run_all_tests()
    
    # Exit code
    exit_code = 0 if all(results.values()) else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
