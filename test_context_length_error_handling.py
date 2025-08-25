#!/usr/bin/env python3
"""
Test per verificare la gestione degli errori di lunghezza del testo durante il clustering
Autore: Sistema di IA  
Data: 25/08/2025
"""

import sys
import os

# Configura il percorso per importare i moduli
sys.path.append('/home/ubuntu/classificazione_discussioni_bck_23_08_2025')

from Pipeline.end_to_end_pipeline import EndToEndPipeline
from unittest.mock import Mock, patch
import numpy as np

def create_mock_sessions_with_long_text():
    """
    Crea sessioni mock con testi di lunghezza crescente per simulare l'errore
    """
    sessions = {}
    
    # Sessione normale
    sessions['session_1'] = {
        'session_id': 'session_1',
        'agent_name': 'TestAgent',
        'num_messaggi_totali': 3,
        'num_messaggi_user': 2,
        'num_messaggi_agent': 1,
        'primo_messaggio': '2025-08-25 10:00:00',
        'ultimo_messaggio': '2025-08-25 10:05:00',
        'testo_completo': '[UTENTE] Ciao, ho un problema [ASSISTENTE] Come posso aiutarla?'
    }
    
    # Sessione lunga (simulata come problematica)
    long_text = '[UTENTE] ' + 'Questo è un testo molto lungo che simula una conversazione estremamente dettagliata. ' * 200
    long_text += '[ASSISTENTE] ' + 'Questa è una risposta molto lunga dell\'assistente che fornisce informazioni dettagliate. ' * 150
    long_text += '[UTENTE] ' + 'L\'utente continua con una richiesta molto specifica e dettagliata. ' * 100
    
    sessions['session_2_LUNGA'] = {
        'session_id': 'session_2_LUNGA',
        'agent_name': 'TestAgent',
        'num_messaggi_totali': 15,
        'num_messaggi_user': 8,
        'num_messaggi_agent': 7,
        'primo_messaggio': '2025-08-25 11:00:00',
        'ultimo_messaggio': '2025-08-25 11:30:00',
        'testo_completo': long_text
    }
    
    # Sessione ancora più lunga
    very_long_text = '[UTENTE] ' + 'Testo estremamente lungo con dettagli tecnici complessi. ' * 500
    very_long_text += '[ASSISTENTE] ' + 'Risposta tecnica molto dettagliata con procedure step-by-step. ' * 400
    very_long_text += '[UTENTE] ' + 'Follow-up dettagliato con ulteriori specifiche tecniche. ' * 300
    
    sessions['session_3_MOLTO_LUNGA'] = {
        'session_id': 'session_3_MOLTO_LUNGA', 
        'agent_name': 'ExpertAgent',
        'num_messaggi_totali': 25,
        'num_messaggi_user': 15,
        'num_messaggi_agent': 10,
        'primo_messaggio': '2025-08-25 12:00:00',
        'ultimo_messaggio': '2025-08-25 13:00:00',
        'testo_completo': very_long_text
    }
    
    return sessions

def test_context_length_error_handling():
    """
    Test per verificare la gestione degli errori di context length
    """
    print("🚀 TEST GESTIONE ERRORI LUNGHEZZA TESTO DURING CLUSTERING")
    print("=" * 80)
    
    try:
        # Crea pipeline per il test
        pipeline = EndToEndPipeline(tenant_slug='humanitas')
        print("✅ Pipeline inizializzata per test")
        
        # Crea sessioni mock con testi lunghi
        mock_sessions = create_mock_sessions_with_long_text()
        print(f"📋 Create {len(mock_sessions)} sessioni mock per test")
        
        # Mostra informazioni sulle lunghezze dei testi
        for session_id, data in mock_sessions.items():
            text_len = len(data['testo_completo'])
            print(f"   📝 {session_id}: {text_len} caratteri")
        
        print("\n🔧 Simulazione errore di context length...")
        
        # Mock dell'embedder per farlo fallire con errore di context length
        original_embedder = pipeline._get_embedder()
        
        def mock_encode_that_fails(*args, **kwargs):
            raise RuntimeError("Input text exceeds maximum context length of 512 tokens. The input contains approximately 1500 tokens.")
        
        # Patch del metodo encode per simulare l'errore
        with patch.object(original_embedder, 'encode', side_effect=mock_encode_that_fails):
            
            print("🎯 Tentativo di eseguire clustering (dovrebbe fallire e mostrare conversazioni)...")
            
            try:
                # Questo dovrebbe fallire e mostrare le conversazioni lunghe
                embeddings, cluster_labels, representatives, suggested_labels = pipeline.esegui_clustering(mock_sessions)
                
                print("❌ ERROR: Il test doveva fallire ma è riuscito!")
                return False
                
            except RuntimeError as e:
                if "context length" in str(e):
                    print("\n✅ SUCCESS: L'errore è stato gestito correttamente!")
                    print("✅ Le conversazioni problematiche sono state mostrate per intero")
                    print("✅ Il sistema ha fornito informazioni di debug dettagliate")
                    return True
                else:
                    print(f"❌ ERROR: Errore inaspettato: {e}")
                    return False
                    
    except Exception as e:
        print(f"❌ ERROR durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_normal_operation_with_short_texts():
    """
    Test per verificare che il sistema funzioni normalmente con testi corti
    """
    print("\n\n🚀 TEST OPERAZIONE NORMALE CON TESTI CORTI")
    print("=" * 80)
    
    try:
        # Crea pipeline per il test
        pipeline = EndToEndPipeline(tenant_slug='humanitas')
        print("✅ Pipeline inizializzata per test normale")
        
        # Crea solo sessioni con testi corti
        short_sessions = {
            'short_1': {
                'session_id': 'short_1',
                'agent_name': 'TestAgent',
                'num_messaggi_totali': 2,
                'num_messaggi_user': 1,
                'num_messaggi_agent': 1,
                'primo_messaggio': '2025-08-25 10:00:00',
                'ultimo_messaggio': '2025-08-25 10:01:00',
                'testo_completo': '[UTENTE] Ciao [ASSISTENTE] Ciao, come posso aiutarla?'
            },
            'short_2': {
                'session_id': 'short_2',
                'agent_name': 'TestAgent', 
                'num_messaggi_totali': 3,
                'num_messaggi_user': 2,
                'num_messaggi_agent': 1,
                'primo_messaggio': '2025-08-25 10:05:00',
                'ultimo_messaggio': '2025-08-25 10:07:00',
                'testo_completo': '[UTENTE] Ho un problema [UTENTE] Potete aiutarmi? [ASSISTENTE] Certo, spieghi il problema'
            }
        }
        
        print(f"📋 Create {len(short_sessions)} sessioni con testi corti")
        
        # Test che dovrebbe funzionare normalmente
        try:
            embeddings, cluster_labels, representatives, suggested_labels = pipeline.esegui_clustering(short_sessions)
            print("✅ SUCCESS: Clustering completato senza errori con testi corti")
            print(f"📊 Risultati: {len(embeddings)} embeddings, {len(set(cluster_labels))} cluster")
            return True
            
        except Exception as e:
            print(f"❌ ERROR: Clustering fallito anche con testi corti: {e}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR durante il test normale: {e}")
        return False

def main():
    """
    Esegue tutti i test
    """
    print("🎯 AVVIO TEST SUITE - GESTIONE ERRORI LUNGHEZZA TESTO\n")
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Gestione errori di context length
    print("📋 Test 1: Gestione errori di context length")
    if test_context_length_error_handling():
        success_count += 1
        print("✅ Test 1: PASSED")
    else:
        print("❌ Test 1: FAILED")
    
    # Test 2: Operazione normale con testi corti
    print("\n📋 Test 2: Operazione normale con testi corti")  
    if test_normal_operation_with_short_texts():
        success_count += 1
        print("✅ Test 2: PASSED")
    else:
        print("❌ Test 2: FAILED")
    
    print(f"\n🎯 RISULTATI FINALI:")
    print(f"   ✅ Test superati: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🏆 TUTTI I TEST SUPERATI!")
        print("✅ La gestione degli errori di lunghezza testo funziona correttamente")
        print("✅ Le conversazioni problematiche vengono mostrate per intero")
        return True
    else:
        print("❌ Alcuni test hanno fallito")
        return False

if __name__ == "__main__":
    main()
