#!/usr/bin/env python3
"""
File: populate_test_cases.py
Autore: Valerio Bignardi
Data creazione: 2025-01-18
Storia aggiornamenti:
    - 2025-01-18: Prima versione

Scopo: Popola il database MongoDB con casi di test per il review dashboard
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

from pymongo import MongoClient
from datetime import datetime
from Utils.tenant import Tenant
import json
from bson import ObjectId

def create_test_cases():
    """
    Crea casi di test nel database MongoDB per il review dashboard
    
    Parametri di input: Nessuno
    Parametri di output: Nessuno
    Valori di ritorno: True se successo, False altrimenti
    Data ultima modifica: 2025-01-18
    """
    try:
        print("🔧 Inizializzazione connessione MongoDB...")
        client = MongoClient('mongodb://localhost:27017/')
        
        print("🔧 Risoluzione tenant...")
        tenant = Tenant.from_uuid('015007d9-d413-11ef-86a5-96000228e7fe')
        print(f"✅ Tenant: {tenant.tenant_name}")
        
        db = client['classificazioni']
        collection_name = f'humanitas_{tenant.tenant_id.replace("-", "_")}'
        collection = db[collection_name]
        
        print(f"🔧 Collection: {collection_name}")
        
        # Pulisco documenti di test esistenti
        collection.delete_many({"test": True})
        
        # Casi di test realistici
        test_cases = [
            {
                "_id": ObjectId(),
                "session_id": "test_session_001",
                "conversation_data": {
                    "messages": [
                        {"role": "user", "content": "Ciao, vorrei prenotare una visita cardiologica"},
                        {"role": "assistant", "content": "Certo, la aiuto a prenotare. Per quale periodo preferirebbe?"},
                        {"role": "user", "content": "La settimana prossima se possibile"}
                    ]
                },
                "classification": {
                    "main_intent": "prenotazione",
                    "sub_intent": "visita_specialistica", 
                    "confidence": 0.95,
                    "needs_review": True
                },
                "review_status": "pending",
                "created_at": datetime.utcnow(),
                "test": False
            },
            {
                "_id": ObjectId(),
                "session_id": "test_session_002", 
                "conversation_data": {
                    "messages": [
                        {"role": "user", "content": "Ho bisogno di informazioni sui servizi disponibili"},
                        {"role": "assistant", "content": "Posso fornirle tutte le informazioni necessarie. Di cosa ha bisogno specificamente?"}
                    ]
                },
                "classification": {
                    "main_intent": "informazioni",
                    "sub_intent": "servizi_generali",
                    "confidence": 0.87,
                    "needs_review": True
                },
                "review_status": "pending",
                "created_at": datetime.utcnow(),
                "test": False
            },
            {
                "_id": ObjectId(),
                "session_id": "test_session_003",
                "conversation_data": {
                    "messages": [
                        {"role": "user", "content": "Il mio account non funziona, non riesco ad accedere"},
                        {"role": "assistant", "content": "Mi dispiace per il disagio. Posso aiutarla a risolvere il problema di accesso."}
                    ]
                },
                "classification": {
                    "main_intent": "problema_tecnico",
                    "sub_intent": "accesso_account",
                    "confidence": 0.92,
                    "needs_review": True
                },
                "review_status": "pending", 
                "created_at": datetime.utcnow(),
                "test": False
            },
            {
                "_id": ObjectId(),
                "session_id": "test_session_004",
                "conversation_data": {
                    "messages": [
                        {"role": "user", "content": "Grazie per l'aiuto, tutto risolto"},
                        {"role": "assistant", "content": "Prego, sono contento di averla aiutata!"}
                    ]
                },
                "classification": {
                    "main_intent": "ringraziamento",
                    "sub_intent": "chiusura_positiva",
                    "confidence": 0.98,
                    "needs_review": False
                },
                "review_status": "completed",
                "human_feedback": {
                    "correct": True,
                    "notes": "Classificazione corretta",
                    "reviewed_by": "test_user",
                    "reviewed_at": datetime.utcnow()
                },
                "created_at": datetime.utcnow(),
                "test": False
            }
        ]
        
        print(f"🔧 Inserimento {len(test_cases)} casi di test...")
        result = collection.insert_many(test_cases)
        print(f"✅ Inseriti {len(result.inserted_ids)} documenti")
        
        # Verifica inserimento
        total_docs = collection.count_documents({})
        pending_docs = collection.count_documents({"review_status": "pending"})
        completed_docs = collection.count_documents({"review_status": "completed"})
        
        print(f"📊 Statistiche database:")
        print(f"   - Documenti totali: {total_docs}")
        print(f"   - Casi pending: {pending_docs}")
        print(f"   - Casi completed: {completed_docs}")
        
        # Crea indici necessari
        print("🔧 Creazione indici...")
        collection.create_index("session_id")
        collection.create_index("review_status")
        collection.create_index("created_at")
        
        client.close()
        print("✅ Database popolato con successo!")
        return True
        
    except Exception as e:
        print(f"❌ Errore durante popolamento database: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_test_cases()
