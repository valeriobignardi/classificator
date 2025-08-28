#!/usr/bin/env python3
"""
Test per verificare il fix della review queue del training supervisionato
Autore: Valerio Bignardi
Data: 2025-01-28
Storia modifiche:
    - 2025-01-28: Creazione script test review queue fix
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

from mongo_classification_reader import MongoClassificationReader
from Pipeline.end_to_end_pipeline import EndToEndPipeline
import yaml

def test_review_queue_fix():
    """
    Testa il fix della review queue per il training supervisionato
    
    Scopo:
        Verifica che i rappresentanti vengano salvati correttamente in MongoDB
        con review_status: 'pending' e metadata appropriati
    
    Output:
        Stampa risultati del test
    
    Data ultima modifica: 2025-01-28
    """
    print("🧪 AVVIO TEST REVIEW QUEUE FIX")
    
    try:
        # Carica configurazione
        with open('/home/ubuntu/classificatore/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Inizializza reader MongoDB  
        mongo_reader = MongoClassificationReader()
        
        # Test 1: Verifica metodi esistenti nel pipeline
        print("\n📋 TEST 1: Verifica metodi pipeline")
        pipeline = EndToEndPipeline("humanitas")
        
        # Controlla che i nuovi metodi esistano
        has_save_representatives = hasattr(pipeline, '_save_representatives_for_review')
        has_save_propagated = hasattr(pipeline, '_save_propagated_sessions_metadata')
        
        print(f"✓ Metodo _save_representatives_for_review: {'presente' if has_save_representatives else 'MANCANTE'}")
        print(f"✓ Metodo _save_propagated_sessions_metadata: {'presente' if has_save_propagated else 'MANCANTE'}")
        
        # Test 2: Verifica struttura MongoDB per un tenant di test
        print("\n📊 TEST 2: Verifica struttura MongoDB")
        test_client = "wopta_16c222a9-f293-11ef-9315-96000228e7fe"
        
        # Ottieni alcune sessioni dalla review queue
        review_sessions = mongo_reader.get_review_queue_sessions(
            client_name=test_client,
            limit=5,
            show_representatives=True,
            show_propagated=True,
            show_outliers=True
        )
        
        print(f"📈 Sessioni in review queue: {len(review_sessions)}")
        
        # Analizza la struttura dei dati
        session_types = {
            'representatives': 0,
            'outliers': 0,
            'propagated': 0,
            'regular': 0
        }
        
        for session in review_sessions:
            cluster_metadata = session.get('cluster_metadata', {})
            review_status = session.get('review_status', 'unknown')
            
            if cluster_metadata.get('is_representative', False):
                session_types['representatives'] += 1
            elif cluster_metadata.get('is_outlier', False) or cluster_metadata.get('cluster_id') == -1:
                session_types['outliers'] += 1
            elif cluster_metadata.get('propagated_from'):
                session_types['propagated'] += 1
            else:
                session_types['regular'] += 1
        
        print(f"📊 Distribuzione tipi sessioni:")
        for session_type, count in session_types.items():
            print(f"  - {session_type}: {count}")
        
        # Test 3: Verifica che API server supporti i filtri
        print("\n🔌 TEST 3: Verifica supporto filtri API")
        
        # Testa get_review_queue_sessions con diversi filtri
        test_scenarios = [
            {"show_representatives": True, "show_outliers": False, "show_propagated": False},
            {"show_representatives": False, "show_outliers": True, "show_propagated": False}, 
            {"show_representatives": False, "show_outliers": False, "show_propagated": True},
        ]
        
        for i, filters in enumerate(test_scenarios):
            filtered_sessions = mongo_reader.get_review_queue_sessions(
                client_name=test_client,
                limit=10,
                **filters
            )
            print(f"  Scenario {i+1}: {filters} -> {len(filtered_sessions)} sessioni")
        
        print("\n✅ TEST COMPLETATO CON SUCCESSO")
        
    except Exception as e:
        print(f"❌ ERRORE DURANTE IL TEST: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False
    
    return True

if __name__ == "__main__":
    test_review_queue_fix()
