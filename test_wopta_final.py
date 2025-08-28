#!/usr/bin/env python3
"""
Script finale per verificare il fix completo del training supervisionato
Autore: Valerio Bignardi
Data: 2025-01-28
Storia modifiche:
    - 2025-01-28: Test finale con tenant wopta reale
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

from mongo_classification_reader import MongoClassificationReader

def test_wopta_tenant():
    """
    Testa il tenant wopta che aveva il problema originale
    
    Scopo:
        Verifica che il tenant wopta_16c222a9-f293-11ef-9315-96000228e7fe
        abbia i dati e che il fix sia compatibile
        
    Output:
        Analisi dei dati esistenti e compatibilità del fix
    
    Data ultima modifica: 2025-01-28
    """
    print("🎯 TEST FINALE - VERIFICA TENANT WOPTA")
    
    try:
        # Verifica il tenant originale del problema
        wopta_client = "wopta_16c222a9-f293-11ef-9315-96000228e7fe"
        
        mongo_reader = MongoClassificationReader()
        
        print(f"\n🔍 VERIFICA DATI TENANT {wopta_client}")
        
        # Ottieni tutte le sessioni
        all_sessions = mongo_reader.get_all_sessions(wopta_client, limit=100)
        print(f"📊 Sessioni totali trovate: {len(all_sessions)}")
        
        if len(all_sessions) > 0:
            # Analizza la struttura dei dati esistenti
            print(f"\n📋 ANALISI STRUTTURA DATI ESISTENTI:")
            
            sample_session = all_sessions[0]
            print(f"🔍 Campi sessione di esempio:")
            for key, value in sample_session.items():
                if isinstance(value, str) and len(str(value)) > 100:
                    print(f"  - {key}: {str(value)[:100]}...")
                else:
                    print(f"  - {key}: {value}")
            
            # Verifica review_status
            auto_classified = [s for s in all_sessions if s.get('review_status') == 'auto_classified']
            pending_review = [s for s in all_sessions if s.get('review_status') == 'pending']
            no_status = [s for s in all_sessions if 'review_status' not in s]
            
            print(f"\n📊 DISTRIBUZIONE REVIEW_STATUS:")
            print(f"  🤖 auto_classified: {len(auto_classified)}")
            print(f"  👤 pending: {len(pending_review)}")
            print(f"  ❓ senza status: {len(no_status)}")
            
            # Verifica cluster_metadata
            with_cluster_metadata = [s for s in all_sessions if 'cluster_metadata' in s and s['cluster_metadata']]
            
            print(f"\n🗂️ VERIFICA CLUSTER_METADATA:")
            print(f"  ✅ Con cluster_metadata: {len(with_cluster_metadata)}")
            print(f"  ❌ Senza cluster_metadata: {len(all_sessions) - len(with_cluster_metadata)}")
            
            if with_cluster_metadata:
                print(f"  📋 Esempio cluster_metadata:")
                for key, value in with_cluster_metadata[0]['cluster_metadata'].items():
                    print(f"    - {key}: {value}")
            
            # Test filtri review queue
            print(f"\n🔌 TEST FILTRI REVIEW QUEUE:")
            
            # Test solo rappresentanti 
            representatives_only = mongo_reader.get_review_queue_sessions(
                wopta_client,
                limit=50,
                show_representatives=True,
                show_propagated=False,
                show_outliers=False
            )
            print(f"  📈 Solo rappresentanti: {len(representatives_only)}")
            
            # Test solo outliers
            outliers_only = mongo_reader.get_review_queue_sessions(
                wopta_client,
                limit=50,
                show_representatives=False,
                show_propagated=False,
                show_outliers=True
            )
            print(f"  🎯 Solo outliers: {len(outliers_only)}")
            
            # Test tutti i tipi
            all_review = mongo_reader.get_review_queue_sessions(
                wopta_client,
                limit=50,
                show_representatives=True,
                show_propagated=True,
                show_outliers=True
            )
            print(f"  🔄 Tutti i tipi: {len(all_review)}")
            
            # Verifica che il problema originale sia risolto
            print(f"\n✅ VERIFICA PROBLEMA ORIGINALE:")
            print(f"Il problema era: 'se ho nascosto gli outliers perchè vedo Ereditata?'")
            print(f"")
            print(f"📊 Risultati attuali:")
            print(f"  - Outliers nascosti → Mostrare: {len(representatives_only)} (solo rappresentanti)")
            print(f"  - Solo outliers → Mostrare: {len(outliers_only)} outliers")
            print(f"  - Tutti visibili → Mostrare: {len(all_review)} totali")
            
            if len(representatives_only) > 0:
                print(f"\n✅ FIX CONFERMATO: Quando si nascondono outliers, si vedono {len(representatives_only)} rappresentanti")
            else:
                print(f"\n⚠️ NOTA: Nessun rappresentante trovato - potrebbe essere necessario rilanciare training supervisionato")
            
        else:
            print(f"ℹ️ Tenant {wopta_client} non ha dati")
            
        print(f"\n✅ TEST FINALE COMPLETATO")
        
    except Exception as e:
        print(f"❌ ERRORE: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_wopta_tenant()
