#!/usr/bin/env python3
"""
File: test_metadata_fix.py
Autore: Valerio Bignardi
Data: 2025-09-04
Scopo: Test dei fix per metadati booleani (representative/outlier/propagated)

Storia aggiornamenti:
2025-09-04: Creazione script di test per verificare i metadati booleani
"""

import sys
import os
import yaml
from datetime import datetime
from pymongo import MongoClient

# Caricamento configurazione
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def test_metadata_structure():
    """
    Scopo: Testa la struttura dei metadati booleani nei documenti MongoDB
    
    Parametri input: Nessuno
    
    Output: Report sui metadati trovati
    
    Ultimo aggiornamento: 2025-09-04 - Valerio Bignardi
    """
    print("🔍 TEST STRUTTURA METADATI BOOLEANI")
    print("="*50)
    
    try:
        # Connessione MongoDB
        mongo_config = config['mongodb']
        client = MongoClient(mongo_config['url'])
        db = client[mongo_config['database']]
        
        # Usa collection Humanitas
        collection_name = "humanitas_015007d9-d413-11ef-86a5-96000228e7fe"
        collection = db[collection_name]
        
        print(f"📊 Collection: {collection_name}")
        print(f"📊 Totale documenti: {collection.count_documents({})}")
        
        # Test 1: Verifica presenza metadati booleani
        print("\n1️⃣ VERIFICA PRESENZA METADATI BOOLEANI:")
        
        # Conta documenti con metadati booleani
        with_representative = collection.count_documents({"metadata.representative": {"$exists": True}})
        with_outlier = collection.count_documents({"metadata.outlier": {"$exists": True}})
        with_propagated = collection.count_documents({"metadata.propagated": {"$exists": True}})
        
        print(f"   ✅ Con metadata.representative: {with_representative}")
        print(f"   ✅ Con metadata.outlier: {with_outlier}")
        print(f"   ✅ Con metadata.propagated: {with_propagated}")
        
        # Test 2: Verifica coerenza con classification_type
        print("\n2️⃣ VERIFICA COERENZA METADATI vs CLASSIFICATION_TYPE:")
        
        # RAPPRESENTANTI
        rep_by_type = collection.count_documents({"classification_type": "RAPPRESENTANTE"})
        rep_by_metadata = collection.count_documents({"metadata.representative": True})
        print(f"   👑 RAPPRESENTANTI - classification_type: {rep_by_type}, metadata.representative=True: {rep_by_metadata}")
        
        # OUTLIERS  
        out_by_type = collection.count_documents({"classification_type": "OUTLIER"})
        out_by_metadata = collection.count_documents({"metadata.outlier": True})
        print(f"   🔴 OUTLIERS - classification_type: {out_by_type}, metadata.outlier=True: {out_by_metadata}")
        
        # PROPAGATI
        prop_by_type = collection.count_documents({"classification_type": "PROPAGATO"})
        prop_by_metadata = collection.count_documents({"metadata.propagated": True})
        print(f"   🔗 PROPAGATI - classification_type: {prop_by_type}, metadata.propagated=True: {prop_by_metadata}")
        
        # Test 3: Esempi di documenti con metadati
        print("\n3️⃣ ESEMPI DOCUMENTI CON METADATI:")
        
        # Esempio rappresentante
        rep_sample = collection.find_one({"metadata.representative": True})
        if rep_sample:
            print(f"   👑 RAPPRESENTANTE sample:")
            print(f"      session_id: {rep_sample.get('session_id', 'N/A')}")
            print(f"      classification_type: {rep_sample.get('classification_type', 'N/A')}")
            print(f"      metadata: {rep_sample.get('metadata', {})}")
        else:
            print(f"   👑 NESSUN RAPPRESENTANTE trovato")
        
        # Esempio outlier
        out_sample = collection.find_one({"metadata.outlier": True})
        if out_sample:
            print(f"   🔴 OUTLIER sample:")
            print(f"      session_id: {out_sample.get('session_id', 'N/A')}")
            print(f"      classification_type: {out_sample.get('classification_type', 'N/A')}")
            print(f"      metadata: {out_sample.get('metadata', {})}")
        else:
            print(f"   🔴 NESSUN OUTLIER con metadata.outlier=True trovato")
            
        # Esempio propagato
        prop_sample = collection.find_one({"metadata.propagated": True})
        if prop_sample:
            print(f"   🔗 PROPAGATO sample:")
            print(f"      session_id: {prop_sample.get('session_id', 'N/A')}")
            print(f"      classification_type: {prop_sample.get('classification_type', 'N/A')}")
            print(f"      metadata: {prop_sample.get('metadata', {})}")
        else:
            print(f"   🔗 NESSUN PROPAGATO con metadata.propagated=True trovato")
        
        # Test 4: Verifica integrità metadati booleani
        print("\n4️⃣ VERIFICA INTEGRITÀ METADATI BOOLEANI:")
        
        # Documenti con esattamente un flag True
        single_flag_docs = []
        
        pipeline = [
            {
                "$project": {
                    "session_id": 1,
                    "classification_type": 1,
                    "representative": "$metadata.representative",
                    "outlier": "$metadata.outlier", 
                    "propagated": "$metadata.propagated",
                    "flag_count": {
                        "$add": [
                            {"$cond": [{"$eq": ["$metadata.representative", True]}, 1, 0]},
                            {"$cond": [{"$eq": ["$metadata.outlier", True]}, 1, 0]},
                            {"$cond": [{"$eq": ["$metadata.propagated", True]}, 1, 0]}
                        ]
                    }
                }
            },
            {"$match": {"flag_count": {"$ne": 1}}}  # Documenti con != 1 flag True
        ]
        
        problematic_docs = list(collection.aggregate(pipeline))
        print(f"   ⚠️ Documenti con problemi di integrità: {len(problematic_docs)}")
        
        if problematic_docs[:3]:  # Mostra primi 3 esempi
            for doc in problematic_docs[:3]:
                print(f"      session_id: {doc.get('session_id', 'N/A')}")
                print(f"      flag_count: {doc.get('flag_count', 'N/A')}")
                print(f"      representative: {doc.get('representative', 'N/A')}")
                print(f"      outlier: {doc.get('outlier', 'N/A')}")
                print(f"      propagated: {doc.get('propagated', 'N/A')}")
                print(f"      ---")
        
        print("\n✅ TEST COMPLETATO!")
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()

def test_save_classification_with_metadata():
    """
    Scopo: Testa il salvataggio di una classificazione con metadati booleani
    
    Parametri input: Nessuno
    
    Output: Risultato del test di salvataggio
    
    Ultimo aggiornamento: 2025-09-04 - Valerio Bignardi  
    """
    print("\n🧪 TEST SALVATAGGIO CLASSIFICAZIONE CON METADATI")
    print("="*50)
    
    try:
        from mongo_classification_reader import MongoClassificationReader
        from tenant import Tenant
        
        # Carica tenant Humanitas direttamente con UUID
        HUMANITAS_TENANT_ID = "015007d9-d413-11ef-86a5-96000228e7fe"
        tenant = Tenant.from_uuid(HUMANITAS_TENANT_ID)
        
        if not tenant:
            print("❌ Tenant Humanitas non trovato")
            return
            
        # Crea reader MongoDB
        reader = MongoClassificationReader(tenant=tenant)
        if not reader.connect():
            print("❌ Errore connessione MongoDB")
            return
            
        print(f"✅ Connesso a MongoDB per tenant: {tenant.tenant_name}")
        
        # Test 1: Salvataggio rappresentante
        print("\n1️⃣ TEST SALVATAGGIO RAPPRESENTANTE:")
        
        test_session_id = f"test_representative_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        success = reader.save_classification_result(
            session_id=test_session_id,
            client_name=tenant.tenant_slug,
            ml_result={
                'predicted_label': 'test_label',
                'confidence': 0.9,
                'method': 'test_ml'
            },
            llm_result={
                'predicted_label': 'test_label', 
                'confidence': 0.95,
                'method': 'test_llm',
                'reasoning': 'Test reasoning'
            },
            final_decision={
                'predicted_label': 'test_label',
                'confidence': 0.95,
                'method': 'ensemble',
                'reasoning': 'Test final decision'
            },
            conversation_text="[UTENTE] Test conversation [ASSISTENTE] Test response",
            needs_review=True,
            classified_by='supervised_training_pipeline',
            notes='Test representative',
            cluster_metadata={
                'cluster_id': 0,
                'is_representative': True,  # RAPPRESENTANTE
                'method': 'test'
            }
        )
        
        if success:
            print(f"   ✅ Rappresentante salvato con session_id: {test_session_id}")
            
            # Verifica metadati
            collection = reader.db[reader.get_collection_name()]
            doc = collection.find_one({"session_id": test_session_id})
            
            if doc:
                metadata = doc.get('metadata', {})
                print(f"   📋 classification_type: {doc.get('classification_type', 'N/A')}")
                print(f"   📋 metadata.representative: {metadata.get('representative', 'N/A')}")
                print(f"   📋 metadata.outlier: {metadata.get('outlier', 'N/A')}")
                print(f"   📋 metadata.propagated: {metadata.get('propagated', 'N/A')}")
                
                # Verifica integrità
                rep = metadata.get('representative', False)
                out = metadata.get('outlier', False)
                prop = metadata.get('propagated', False)
                
                if rep and not out and not prop:
                    print(f"   ✅ Metadati booleani corretti per rappresentante")
                else:
                    print(f"   ❌ Metadati booleani ERRATI: rep={rep}, out={out}, prop={prop}")
        else:
            print(f"   ❌ Errore salvataggio rappresentante")
        
        # Test 2: Salvataggio propagato
        print("\n2️⃣ TEST SALVATAGGIO PROPAGATO:")
        
        test_session_id = f"test_propagated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        success = reader.save_classification_result(
            session_id=test_session_id,
            client_name=tenant.tenant_slug,
            ml_result=None,
            llm_result={
                'predicted_label': 'test_label_propagated',
                'confidence': 0.85,
                'method': 'cluster_propagation',
                'reasoning': 'Propagated from cluster'
            },
            final_decision={
                'predicted_label': 'test_label_propagated',
                'confidence': 0.85,
                'method': 'CLUSTER_PROPAGATION',
                'reasoning': 'Propagated from cluster'
            },
            conversation_text="[UTENTE] Test propagated [ASSISTENTE] Test propagated response",
            needs_review=False,
            classified_by='cluster_propagation',
            notes='Test propagated',
            cluster_metadata={
                'cluster_id': 1,
                'is_representative': False,  # NON rappresentante
                'propagated_from': 'cluster_propagation',
                'propagation_confidence': 0.85
            }
        )
        
        if success:
            print(f"   ✅ Propagato salvato con session_id: {test_session_id}")
            
            # Verifica metadati
            doc = collection.find_one({"session_id": test_session_id})
            
            if doc:
                metadata = doc.get('metadata', {})
                print(f"   📋 classification_type: {doc.get('classification_type', 'N/A')}")
                print(f"   📋 metadata.representative: {metadata.get('representative', 'N/A')}")
                print(f"   📋 metadata.outlier: {metadata.get('outlier', 'N/A')}")
                print(f"   📋 metadata.propagated: {metadata.get('propagated', 'N/A')}")
                
                # Verifica integrità
                rep = metadata.get('representative', False)
                out = metadata.get('outlier', False)
                prop = metadata.get('propagated', False)
                
                if not rep and not out and prop:
                    print(f"   ✅ Metadati booleani corretti per propagato")
                else:
                    print(f"   ❌ Metadati booleani ERRATI: rep={rep}, out={out}, prop={prop}")
        else:
            print(f"   ❌ Errore salvataggio propagato")
        
        print("\n✅ TEST SALVATAGGIO COMPLETATO!")
        
    except Exception as e:
        print(f"❌ Errore durante il test salvataggio: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 TEST METADATI BOOLEANI (representative/outlier/propagated)")
    print("="*70)
    
    # Test 1: Struttura metadati esistenti
    test_metadata_structure()
    
    # Test 2: Salvataggio con metadati
    test_save_classification_with_metadata()
    
    print("\n🎯 TUTTI I TEST COMPLETATI!")
