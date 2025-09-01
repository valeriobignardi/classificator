#!/usr/bin/env python3
"""
Popola database MongoDB con casi di test per la UI di revisione
Autore: Valerio Bignardi  
Data: 2025-09-01
"""

import sys
import os
from datetime import datetime, timedelta
import uuid

# Aggiungi path del progetto
sys.path.append('/home/ubuntu/classificatore')

def create_test_cases():
    """
    Crea casi di test nel database MongoDB per testare la UI
    """
    print("🔧 CREAZIONE CASI DI TEST - DATABASE MONGODB")
    print("="*80)
    
    try:
        from pymongo import MongoClient
        from Utils.tenant import Tenant
        
        # Connessione al tenant humanitas
        print("🔧 Inizializzazione tenant...")
        tenant = Tenant.from_uuid('015007d9-d413-11ef-86a5-96000228e7fe')
        
        # Connessione MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['classificazioni']
        collection_name = f"humanitas_{tenant.tenant_id.replace('-', '_')}"
        collection = db[collection_name]
        
        print(f"📊 Collection: {collection_name}")
        
        # Crea casi di test diversificati
        test_cases = []
        
        # Caso 1: Prenotazione visita (pending)
        test_cases.append({
            "_id": "case_001_prenotazione_visita",
            "conversation_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "conversation_text": "[UTENTE] Vorrei prenotare una visita cardiologica [ASSISTENTE] Ti aiuto a prenotare la visita cardiologica. Che giorno preferisci?",
            "predicted_label": "prenotazione",
            "confidence": 0.85,
            "classification_type": "REPRESENTATIVE",
            "cluster_id": 1,
            "review_status": "pending",
            "timestamp": datetime.now() - timedelta(hours=2),
            "tenant_id": tenant.tenant_id,
            "needs_review": True,
            "human_feedback": None,
            "propagated_from": None
        })
        
        # Caso 2: Problema referto (pending)  
        test_cases.append({
            "_id": "case_002_problema_referto",
            "conversation_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "conversation_text": "[UTENTE] Non riesco a scaricare il mio referto dal portale [ASSISTENTE] Ti aiuto con il problema del referto. Hai provato a controllare la sezione download?",
            "predicted_label": "problema_ritiro_referti_cartella_clinica",
            "confidence": 0.72,
            "classification_type": "OUTLIER",
            "cluster_id": -1,
            "review_status": "pending", 
            "timestamp": datetime.now() - timedelta(hours=4),
            "tenant_id": tenant.tenant_id,
            "needs_review": True,
            "human_feedback": None,
            "propagated_from": None
        })
        
        # Caso 3: Info generica (pending)
        test_cases.append({
            "_id": "case_003_info_orari",
            "conversation_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "conversation_text": "[UTENTE] Quali sono gli orari di apertura? [ASSISTENTE] Gli orari di apertura sono dal lunedì al venerdì dalle 8:00 alle 20:00",
            "predicted_label": "altro",
            "confidence": 0.45,
            "classification_type": "PROPAGATED",
            "cluster_id": 2,
            "review_status": "pending",
            "timestamp": datetime.now() - timedelta(hours=1),
            "tenant_id": tenant.tenant_id,
            "needs_review": True,
            "human_feedback": None,
            "propagated_from": "case_001_prenotazione_visita"
        })
        
        # Caso 4: Già risolto (completed)
        test_cases.append({
            "_id": "case_004_completed_esempio",
            "conversation_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "conversation_text": "[UTENTE] Grazie per l'aiuto [ASSISTENTE] Prego, è stato un piacere aiutarti!",
            "predicted_label": "altro",
            "confidence": 0.95,
            "classification_type": "REPRESENTATIVE",
            "cluster_id": 3,
            "review_status": "completed",
            "timestamp": datetime.now() - timedelta(days=1),
            "tenant_id": tenant.tenant_id,
            "needs_review": False,
            "human_feedback": {
                "correct_label": "altro",
                "feedback_type": "correct",
                "reviewer": "sistema_test",
                "review_timestamp": datetime.now() - timedelta(days=1)
            },
            "propagated_from": None
        })
        
        # Inserisci i casi
        print(f"📥 Inserimento {len(test_cases)} casi di test...")
        result = collection.insert_many(test_cases)
        print(f"✅ Inseriti {len(result.inserted_ids)} documenti")
        
        # Crea indici per performance
        print(f"🔧 Creazione indici...")
        collection.create_index("review_status")
        collection.create_index([("timestamp", -1)])
        collection.create_index("classification_type")
        collection.create_index("cluster_id")
        print(f"✅ Indici creati")
        
        # Verifica risultati
        pending_count = collection.count_documents({"review_status": "pending"})
        completed_count = collection.count_documents({"review_status": "completed"})
        total_count = collection.count_documents({})
        
        print(f"\n📊 RISULTATI FINALI:")
        print(f"   📈 Totale documenti: {total_count}")
        print(f"   📈 Casi pending: {pending_count}")
        print(f"   📈 Casi completed: {completed_count}")
        
        # Chiudi connessione
        client.close()
        print(f"\n✅ Database popolato con successo!")
        print(f"🌐 La UI dovrebbe ora mostrare {pending_count} casi pending")
        
    except Exception as e:
        print(f"❌ Errore durante la creazione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_test_cases()
