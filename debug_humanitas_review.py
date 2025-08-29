#!/usr/bin/env python3
"""
Debug script per verificare i casi in review queue per Humanitas
Autore: AI Assistant
Data: 23-08-2025
Descrizione: Verifica se ci sono casi nella review queue per il tenant Humanitas
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mongo_classification_reader import MongoClassificationReader
import yaml

def main():
    """
    Funzione principale per il debug della review queue Humanitas
    """
    print("🔍 Debug Review Queue Humanitas - Inizio analisi...")
    
    # Carico la configurazione
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ Configurazione caricata correttamente")
    except Exception as e:
        print(f"❌ Errore nel caricamento configurazione: {e}")
        return
    
    # Inizializzo il reader MongoDB
    try:
        mongodb_config = config.get('mongodb', {})
        reader = MongoClassificationReader(
            mongodb_url=mongodb_config.get('url', 'mongodb://localhost:27017'),
            database_name=mongodb_config.get('database', 'classificazioni')
        )
        print("✅ MongoClassificationReader inizializzato")
        
        # Connetto al database
        if reader.connect():
            print("✅ Connessione MongoDB stabilita")
        else:
            print("❌ Errore nella connessione MongoDB")
            return
            
    except Exception as e:
        print(f"❌ Errore inizializzazione MongoDB: {e}")
        return
    
    # Cerco il client Humanitas
    client_name = "humanitas"
    print(f"\n📊 Analisi dati per client: {client_name}")
    
    try:
        # 1. Verifico se esiste la collezione
        reader.set_tenant(client_name)  # Imposta il tenant
        collection_name = reader.get_collection_name()  # Ora usa l'oggetto tenant interno
        print(f"📁 Nome collezione: {collection_name}")
        
        # Accedo alla collezione tramite il database
        collection = reader.db[collection_name]
        
        # 2. Conto tutti i documenti
        total_docs = collection.count_documents({})
        print(f"📄 Totale documenti in collezione: {total_docs}")
        
        # 3. Controllo documenti con human_label empty/null
        pending_review = collection.count_documents({
            "$or": [
                {"human_label": {"$exists": False}},
                {"human_label": None},
                {"human_label": ""}
            ]
        })
        print(f"🔄 Documenti in attesa di review (human_label vuoto): {pending_review}")
        
        # 4. Controllo documenti con human_label definito
        reviewed_docs = collection.count_documents({
            "human_label": {"$exists": True, "$ne": None, "$ne": ""}
        })
        print(f"✅ Documenti già reviewati: {reviewed_docs}")
        
        # 5. Controllo presenza di ml_prediction e llm_prediction
        docs_with_ml = collection.count_documents({
            "ml_prediction": {"$exists": True, "$ne": None, "$ne": ""}
        })
        docs_with_llm = collection.count_documents({
            "llm_prediction": {"$exists": True, "$ne": None, "$ne": ""}
        })
        print(f"🤖 Documenti con ml_prediction: {docs_with_ml}")
        print(f"🧠 Documenti con llm_prediction: {docs_with_llm}")
        
        # 6. Verifico la struttura di alcuni documenti
        print(f"\n🔬 Campione di documenti (primi 3):")
        sample_docs = list(collection.find({}).limit(3))
        for i, doc in enumerate(sample_docs):
            print(f"\nDocumento {i+1}:")
            print(f"  - _id: {doc.get('_id')}")
            print(f"  - session_id: {doc.get('session_id', 'N/A')}")
            print(f"  - human_label: {doc.get('human_label', 'VUOTO')}")
            print(f"  - ml_prediction: {doc.get('ml_prediction', 'VUOTO')}")
            print(f"  - llm_prediction: {doc.get('llm_prediction', 'VUOTO')}")
            print(f"  - is_representative: {doc.get('is_representative', 'N/A')}")
            print(f"  - cluster_id: {doc.get('cluster_id', 'N/A')}")
        
        # 7. Testo il metodo get_pending_review_sessions direttamente
        print(f"\n🧪 Test metodo get_pending_review_sessions:")
        try:
            cases = reader.get_pending_review_sessions(client_name, limit=5)
            print(f"📋 Casi trovati dal metodo: {len(cases)}")
            
            if cases:
                print("Dettagli primi 2 casi:")
                for i, case in enumerate(cases[:2]):
                    print(f"\nCaso {i+1}:")
                    print(f"  - session_id: {case.get('session_id')}")
                    print(f"  - ml_prediction: {case.get('ml_prediction')}")
                    print(f"  - llm_prediction: {case.get('llm_prediction')}")
                    print(f"  - human_label: {case.get('human_label')}")
            else:
                print("❌ Nessun caso trovato dal metodo get_pending_review_sessions")
                
        except Exception as e:
            print(f"❌ Errore nel test get_pending_review_sessions: {e}")
        
        # 8. Testo il nuovo metodo get_representative_sessions_only
        print(f"\n🧪 Test metodo get_representative_sessions_only:")
        try:
            rep_cases = reader.get_representative_sessions_only(client_name, limit=5)
            print(f"👑 Casi rappresentanti trovati: {len(rep_cases)}")
            
            if rep_cases:
                print("Dettagli primi 2 casi rappresentanti:")
                for i, case in enumerate(rep_cases[:2]):
                    print(f"\nCaso rappresentante {i+1}:")
                    print(f"  - session_id: {case.get('session_id')}")
                    print(f"  - is_representative: {case.get('is_representative')}")
                    print(f"  - cluster_id: {case.get('cluster_id')}")
            else:
                print("❌ Nessun caso rappresentante trovato")
                
        except Exception as e:
            print(f"❌ Errore nel test get_representative_sessions_only: {e}")
    
    except Exception as e:
        print(f"❌ Errore durante l'analisi: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Debug completato!")

if __name__ == "__main__":
    main()
