#!/usr/bin/env python3
"""
Script per testare il training supervisionato con il nuovo fix
Autore: Valerio Bignardi
Data: 2025-01-28
Storia modifiche:
    - 2025-01-28: Creazione script test training supervisionato
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

from mongo_classification_reader import MongoClassificationReader
from Pipeline.end_to_end_pipeline import EndToEndPipeline
import yaml

def test_supervised_training():
    """
    Esegue un test del training supervisionato per verificare il fix
    
    Scopo:
        Esegue alcune sessioni del training supervisionato e verifica
        che i rappresentanti vengano salvati correttamente
        
    Output:
        Stampa risultati del training e verifica review queue
    
    Data ultima modifica: 2025-01-28
    """
    print("🎯 AVVIO TEST TRAINING SUPERVISIONATO")
    
    try:
        # Inizializza pipeline 
        pipeline = EndToEndPipeline("humanitas")
        print(f"✅ Pipeline inizializzata per tenant: {pipeline.tenant.tenant_slug}")
        
        # Controlla se ci sono già dati
        mongo_reader = MongoClassificationReader()
        client_name = pipeline.tenant.tenant_id
        
        print(f"\n🔍 VERIFICA DATI ESISTENTI per {client_name}")
        
        # Verifica quante sessioni ci sono totali 
        all_sessions = mongo_reader.get_all_sessions(client_name, limit=100)
        print(f"📊 Sessioni totali: {len(all_sessions)}")
        
        # Verifica quante sono auto-classificate vs pending
        auto_classified = [s for s in all_sessions if s.get('review_status') == 'auto_classified']
        pending_review = [s for s in all_sessions if s.get('review_status') == 'pending']
        
        print(f"🤖 Auto-classificate: {len(auto_classified)}")
        print(f"👤 Pending review: {len(pending_review)}")
        
        # Se non ci sono dati pending, proviamo a fare un micro-training
        if len(pending_review) == 0 and len(all_sessions) >= 10:
            print(f"\n🧪 ESEGUIAMO MICRO-TRAINING SUPERVISIONATO")
            print(f"   Utilizzeremo prime 10 sessioni per creare rappresentanti")
            
            # Prendiamo prime 10 sessioni per test
            test_sessions = all_sessions[:10]
            
            # Convertiamo in formato atteso dal pipeline
            conversations_data = []
            for session in test_sessions:
                conversations_data.append({
                    'session_id': session.get('session_id', session.get('id')),
                    'conversation_text': session.get('conversation_text', session.get('testo_completo', '')),
                    'timestamp': session.get('timestamp', session.get('data_conversazione'))
                })
            
            print(f"📝 Preparate {len(conversations_data)} conversazioni per clustering")
            
            # Eseguiamo solo il clustering per test
            try:
                print(f"\n🔄 ESECUZIONE CLUSTERING...")
                clustering_result = pipeline.esegui_clustering(conversations_data)
                
                print(f"✅ Clustering completato!")
                print(f"📊 Numero cluster trovati: {clustering_result.get('n_clusters', 'N/A')}")
                print(f"📊 Outliers: {clustering_result.get('n_outliers', 'N/A')}")
                
                # Ora verifichiamo se la review queue è stata popolata
                print(f"\n🔍 VERIFICA REVIEW QUEUE DOPO CLUSTERING")
                
                review_sessions_after = mongo_reader.get_review_queue_sessions(
                    client_name, 
                    limit=50,
                    show_representatives=True,
                    show_propagated=True, 
                    show_outliers=True
                )
                
                print(f"📈 Sessioni in review queue: {len(review_sessions_after)}")
                
                # Analisi tipi
                session_types = {
                    'representatives': 0,
                    'outliers': 0,
                    'propagated': 0,
                    'regular': 0
                }
                
                for session in review_sessions_after:
                    cluster_metadata = session.get('cluster_metadata', {})
                    
                    if cluster_metadata.get('is_representative', False):
                        session_types['representatives'] += 1
                    elif cluster_metadata.get('is_outlier', False) or cluster_metadata.get('cluster_id') == -1:
                        session_types['outliers'] += 1  
                    elif cluster_metadata.get('propagated_from'):
                        session_types['propagated'] += 1
                    else:
                        session_types['regular'] += 1
                
                print(f"📊 Distribuzione dopo clustering:")
                for session_type, count in session_types.items():
                    print(f"  - {session_type}: {count}")
                
                if session_types['representatives'] > 0:
                    print(f"✅ FIX FUNZIONA! Trovati {session_types['representatives']} rappresentanti")
                else:
                    print(f"⚠️ Nessun rappresentante trovato - possibile problema")
                    
            except Exception as clustering_error:
                print(f"❌ Errore durante clustering: {str(clustering_error)}")
                import traceback
                print(traceback.format_exc())
                
        else:
            print(f"ℹ️ Dati insufficienti per test (sessioni: {len(all_sessions)})")
            print(f"   O review queue già popolata (pending: {len(pending_review)})")
        
        print(f"\n✅ TEST COMPLETATO")
        
    except Exception as e:
        print(f"❌ ERRORE DURANTE IL TEST: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False
    
    return True

if __name__ == "__main__":
    test_supervised_training()
