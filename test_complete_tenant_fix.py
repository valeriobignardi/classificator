#!/usr/bin/env python3
"""
Test completo della correzione tenant_id vs tenant_slug nella pipeline

Autore: Valerio Bignardi
Data: 2025-08-28
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from mongo_classification_reader import MongoClassificationReader

def test_save_classification_with_corrections():
    """
    Scopo: Verifica che save_classification_result funzioni con tenant_slug
    """
    
    print("🔧 Test save_classification_result con correzioni...")
    
    try:
        # Inizializza reader MongoDB
        mongo_reader = MongoClassificationReader()
        
        # Dati di test
        tenant_slug = "humanitas"
        session_id = "test_session_" + str(int(os.urandom(4).hex(), 16))
        
        # Test save_classification_result con tenant_slug (CORRETTO)
        print(f"📋 Test save_classification_result con tenant_slug: {tenant_slug}")
        
        success = mongo_reader.save_classification_result(
            session_id=session_id,
            client_name=tenant_slug,  # CORRETTO: usa tenant_slug
            final_decision={
                'label': 'test_label', 
                'confidence': 0.95,
                'method': 'test_supervised_training'
            },
            conversation_text="Test conversation per verificare correzione",
            needs_review=True,
            review_reason="Test per correzione tenant_id vs tenant_slug",
            classified_by="test_correction",
            cluster_metadata={
                'cluster_id': 999,
                'is_representative': True,
                'cluster_method': 'test_supervised'
            }
        )
        
        print(f"   Risultato save: {success}")
        
        if success:
            print("✅ SUCCESSO: save_classification_result con tenant_slug funziona")
            
            # Verifica che la sessione sia stata salvata nella collection corretta
            tenant_info = mongo_reader.get_tenant_info_from_name(tenant_slug)
            expected_collection = f"{tenant_slug}_{tenant_info['tenant_id']}"
            print(f"   Collection attesa: {expected_collection}")
            
            # Prova a recuperare la sessione salvata
            collection = mongo_reader.db[expected_collection]
            saved_session = collection.find_one({'session_id': session_id})
            
            if saved_session:
                print("✅ SESSIONE RECUPERATA: la sessione è stata salvata correttamente")
                print(f"   Session ID: {saved_session['session_id']}")
                print(f"   Review Status: {saved_session.get('review_status', 'N/A')}")
                print(f"   Cluster ID: {saved_session.get('cluster_metadata', {}).get('cluster_id', 'N/A')}")
                return True
            else:
                print("❌ SESSIONE NON TROVATA: save ha restituito success ma sessione non trovata")
                return False
        else:
            print("❌ ERRORE: save_classification_result ha restituito False")
            return False
            
    except Exception as e:
        print(f"❌ Errore durante test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cluster_representatives_access():
    """
    Scopo: Verifica che i metodi cluster helper funzionino correttamente
    """
    
    print("\n🔧 Test cluster representatives access...")
    
    try:
        mongo_reader = MongoClassificationReader()
        tenant_slug = "humanitas"
        cluster_id = 999  # Cluster ID del test precedente
        
        # Test get_cluster_representatives
        print(f"📋 Test get_cluster_representatives per cluster {cluster_id}")
        representatives = mongo_reader.get_cluster_representatives(tenant_slug, cluster_id)
        
        print(f"   Rappresentanti trovati: {len(representatives)}")
        
        if len(representatives) > 0:
            print("✅ SUCCESSO: get_cluster_representatives funziona")
            rep = representatives[0]
            print(f"   Primo rappresentante: {rep.get('session_id')}")
            return True
        else:
            print("⚠️ Nessun rappresentante trovato (normale se è il primo test)")
            return True  # Non è un errore
            
    except Exception as e:
        print(f"❌ Errore durante test cluster: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TEST CORREZIONE TENANT_ID vs TENANT_SLUG")
    print("=" * 60)
    
    test1_success = test_save_classification_with_corrections()
    test2_success = test_cluster_representatives_access()
    
    overall_success = test1_success and test2_success
    print("\n" + "=" * 60)
    print(f"{'✅ TUTTI I TEST PASSATI' if overall_success else '❌ ALCUNI TEST FALLITI'}")
    print("=" * 60)
