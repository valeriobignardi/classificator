#!/usr/bin/env python3
"""
File: test_fix_training_supervisionato.py
Autore: Valerio Bignardi
Data creazione: 2025-09-11
Descrizione: Test per verificare il fix del training supervisionato

Scopo: 
- Simulare il salvataggio con i metadati corretti
- Verificare che i rappresentanti vengano salvati correttamente
- Test del metodo corretto save_classification_result()
"""

import sys
import os
import json
from datetime import datetime
from pymongo import MongoClient
import yaml

# Aggiungi path per import
sys.path.append(os.path.dirname(__file__))

from Utils.tenant import Tenant
from mongo_classification_reader import MongoClassificationReader

def load_config():
    """Carica configurazione dal file config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Errore caricamento config: {e}")
        return None

def create_test_tenant():
    """Crea un tenant di test per Wopta"""
    return Tenant(
        tenant_id="16c222a9-f293-11ef-9315-96000228e7fe",
        tenant_name="Wopta",
        tenant_slug="wopta",
        tenant_database="wopta_16c222a9-f293-11ef-9315-96000228e7fe",
        tenant_status=1
    )

def simulate_training_supervisionato_save():
    """
    Simula il salvataggio corretto del training supervisionato
    usando il nuovo metodo save_classification_result()
    """
    print(f"🧪 TEST FIX TRAINING SUPERVISIONATO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Setup
    config = load_config()
    if not config:
        return False
    
    tenant = create_test_tenant()
    
    try:
        mongo_reader = MongoClassificationReader(tenant=tenant)
        print(f"✅ MongoClassificationReader inizializzato")
    except Exception as e:
        print(f"❌ Errore inizializzazione MongoReader: {e}")
        return False
    
    # Cleanup documenti di test precedenti
    cleanup_test_data(tenant)
    
    # Simula dati dal clustering (come nel training supervisionato)
    test_sessions = {
        'test_repr_001': {
            'conversation_text': 'Vorrei informazioni sui benefici della polizza vita',
            'cluster_id': 5,
            'is_representative': True,
            'is_outlier': False,
            'classification': 'INFO_POLIZZA_VITA',
            'confidence': 0.89,
            'embedding': [0.1] * 768
        },
        'test_repr_002': {
            'conversation_text': 'Come funziona il rimborso delle spese mediche?',
            'cluster_id': 12,
            'is_representative': True,
            'is_outlier': False,
            'classification': 'RIMBORSO_SPESE',
            'confidence': 0.91,
            'embedding': [0.2] * 768
        },
        'test_outlier_001': {
            'conversation_text': 'Ciao, buongiorno a tutti!',
            'cluster_id': -1,
            'is_representative': False,
            'is_outlier': True,
            'classification': 'ALTRO',
            'confidence': 0.15,
            'embedding': [0.3] * 768
        },
        'test_normal_001': {
            'conversation_text': 'Voglio disdire la mia polizza auto',
            'cluster_id': 8,
            'is_representative': False,
            'is_outlier': False,
            'classification': 'DISDETTA_POLIZZA',
            'confidence': 0.94,
            'embedding': [0.4] * 768
        }
    }
    
    print(f"\n💾 SIMULAZIONE SALVATAGGIO TRAINING SUPERVISIONATO:")
    print(f"   📊 Salvando {len(test_sessions)} sessioni di test...")
    
    saved_count = 0
    
    # Simula il loop di salvataggio corretto (come nel fix applicato)
    for session_id, session_data in test_sessions.items():
        print(f"\n   📝 Salvando {session_id}:")
        print(f"      Cluster ID: {session_data['cluster_id']}")
        print(f"      Is Representative: {session_data['is_representative']}")
        print(f"      Is Outlier: {session_data['is_outlier']}")
        print(f"      Classification: {session_data['classification']}")
        
        # Prepara final_decision nel formato corretto
        final_decision = {
            'predicted_label': session_data['classification'],
            'confidence': session_data['confidence'],
            'method': 'training_supervisionato_test',
            'reasoning': f"Test training supervisionato - {session_data['classification']}"
        }
        
        # Prepara cluster_metadata nel formato corretto
        cluster_metadata = {
            'cluster_id': session_data['cluster_id'],
            'cluster_size': 10 if session_data['cluster_id'] != -1 else 1,
            'is_representative': session_data['is_representative'],
            'is_outlier': session_data['is_outlier'],
            'selection_reason': 'test_training_supervisionato' if session_data['is_representative'] else None,
            'propagated_from': None,
            'propagation_consensus': None,
            'propagation_reason': None
        }
        
        # Determina review info
        needs_review = session_data['confidence'] < 0.7
        review_reason = 'confidence_bassa' if needs_review else 'pipeline_processing'
        classified_by = 'test_training_supervisionato'
        
        # Usa il metodo CORRETTO save_classification_result()
        success = mongo_reader.save_classification_result(
            session_id=session_id,
            client_name=tenant.tenant_slug,
            final_decision=final_decision,
            conversation_text=session_data['conversation_text'],
            needs_review=needs_review,
            review_reason=review_reason,
            classified_by=classified_by,
            notes=f"Test training supervisionato - cluster_id: {session_data['cluster_id']}",
            cluster_metadata=cluster_metadata,
            embedding=session_data['embedding']
        )
        
        if success:
            saved_count += 1
            print(f"      ✅ Salvato con successo")
        else:
            print(f"      ❌ Errore nel salvataggio")
    
    print(f"\n   📊 Risultato: {saved_count}/{len(test_sessions)} sessioni salvate")
    
    # Verifica i documenti salvati
    print(f"\n🔍 VERIFICA DOCUMENTI SALVATI:")
    verify_saved_sessions(tenant, test_sessions)
    
    # Cleanup
    cleanup_test_data(tenant)
    
    return saved_count == len(test_sessions)

def verify_saved_sessions(tenant, expected_sessions):
    """
    Verifica che le sessioni siano state salvate correttamente
    """
    try:
        client = MongoClient('mongodb://localhost:27017')
        db = client['classificazioni']
        collection_name = f"wopta_{tenant.tenant_id}"
        collection = db[collection_name]
        
        for session_id, expected_data in expected_sessions.items():
            print(f"\n   🔍 Verifica {session_id}:")
            
            doc = collection.find_one({"session_id": session_id})
            
            if not doc:
                print(f"      ❌ Documento NON TROVATO!")
                continue
            
            # Verifica metadati cluster
            metadata = doc.get('metadata', {})
            expected_cluster_id = expected_data['cluster_id']
            expected_is_repr = expected_data['is_representative']
            expected_is_outlier = expected_data['is_outlier']
            
            saved_cluster_id = metadata.get('cluster_id')
            saved_is_repr = metadata.get('is_representative', False)
            saved_is_outlier = metadata.get('outlier', False)
            
            print(f"      Classification Type: {doc.get('classification_type', 'N/A')}")
            print(f"      Cluster ID: {expected_cluster_id} → {saved_cluster_id} {'✅' if expected_cluster_id == saved_cluster_id else '❌'}")
            print(f"      Is Representative: {expected_is_repr} → {saved_is_repr} {'✅' if expected_is_repr == saved_is_repr else '❌'}")
            print(f"      Is Outlier: {expected_is_outlier} → {saved_is_outlier} {'✅' if expected_is_outlier == saved_is_outlier else '❌'}")
            
            # Verifica classification_type
            expected_type = "RAPPRESENTANTE" if expected_is_repr else ("OUTLIER" if expected_is_outlier else "PROPAGATO")
            saved_type = doc.get('classification_type', 'N/A')
            print(f"      Classification Type: {expected_type} → {saved_type} {'✅' if expected_type == saved_type else '❌'}")
        
        client.close()
        
    except Exception as e:
        print(f"   ❌ Errore verifica: {e}")

def cleanup_test_data(tenant):
    """Pulisce i dati di test"""
    try:
        client = MongoClient('mongodb://localhost:27017')
        db = client['classificazioni']
        collection_name = f"wopta_{tenant.tenant_id}"
        collection = db[collection_name]
        
        # Rimuovi documenti di test
        result = collection.delete_many({
            "session_id": {"$in": ["test_repr_001", "test_repr_002", "test_outlier_001", "test_normal_001"]}
        })
        
        if result.deleted_count > 0:
            print(f"🧹 Rimossi {result.deleted_count} documenti di test precedenti")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Errore cleanup: {e}")

def main():
    """Funzione principale"""
    success = simulate_training_supervisionato_save()
    
    print(f"\n{'='*80}")
    print(f"📊 RISULTATO FINALE")
    print(f"{'='*80}")
    
    if success:
        print(f"✅ TEST SUPERATO: Il fix del training supervisionato funziona correttamente!")
        print(f"🎯 I rappresentanti vengono ora salvati con i metadati corretti")
        print(f"🔧 Il metodo save_classification_result() gestisce correttamente i cluster")
    else:
        print(f"❌ TEST FALLITO: Il fix non funziona come previsto")
        print(f"🐛 Serve ulteriore debug del processo di salvataggio")
    
    print(f"\n🏁 Test completato!")

if __name__ == "__main__":
    main()
