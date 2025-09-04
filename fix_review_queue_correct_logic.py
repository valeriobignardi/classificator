#!/usr/bin/env python3
"""
Correzione logica Review Queue secondo le specifiche corrette:

REGOLE CORRETTE:
1. RAPPRESENTANTI con bassa confidenza/disaccordo → review_status: "pending" 
2. OUTLIERS con bassa confidenza/disaccordo → review_status: "pending"
3. PROPAGATI → SEMPRE review_status: "auto_classified" (mai in queue automaticamente)
4. PROPAGATI → Solo se aggiunti manualmente da UI "Tutte le Sessioni"

CRITERI per "bassa confidenza/disaccordo":
- Confidenza < soglia (es. 0.8)
- Disaccordo tra ML e LLM predictions
- Scores di incertezza elevati

Autore: Valerio Bignardi
Data: 2025-09-04
"""

import yaml
from pymongo import MongoClient
from datetime import datetime
import json

def apply_correct_review_logic():
    """
    Applica la logica corretta per Review Queue:
    - Solo rappresentanti e outliers con problemi di confidenza/disaccordo
    - Mai propagati (a meno di aggiunta manuale)
    """
    try:
        # Carica configurazione MongoDB
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        mongo_url = config.get('mongodb', {}).get('url', 'mongodb://localhost:27017')
        mongo_db_name = config.get('mongodb', {}).get('database', 'classificazioni')
        
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        db = client[mongo_db_name]
        
        # Trova la collezione di Humanitas
        collections = db.list_collection_names()
        humanitas_collection = None
        for coll_name in collections:
            if 'humanitas' in coll_name.lower():
                humanitas_collection = coll_name
                break
        
        if not humanitas_collection:
            print("❌ Nessuna collezione Humanitas trovata")
            return
        
        print(f"🎯 Applicazione logica corretta alla collection: {humanitas_collection}")
        collection = db[humanitas_collection]
        
        print("\n" + "="*80)
        print("🔧 LOGICA CORRETTA REVIEW QUEUE")
        print("="*80)
        
        # STEP 1: RESET - Tutti i documenti diventano 'auto_classified'
        print("\n🔄 STEP 1: Reset di tutti i documenti → 'auto_classified'")
        
        reset_result = collection.update_many(
            {},  # Tutti i documenti
            {
                "$set": {
                    "review_status": "auto_classified",
                    "review_reason": "system_reset"
                }
            }
        )
        print(f"   ✅ Reset completato: {reset_result.modified_count} documenti")
        
        # STEP 2: Soglie per confidenza e disaccordo
        CONFIDENCE_THRESHOLD = 0.8
        DISAGREEMENT_THRESHOLD = 0.3  # Differenza tra confidenze ML/LLM
        
        print(f"\n⚙️ PARAMETRI:")
        print(f"   📊 Soglia confidenza: {CONFIDENCE_THRESHOLD}")
        print(f"   ⚖️ Soglia disaccordo: {DISAGREEMENT_THRESHOLD}")
        
        # STEP 3: RAPPRESENTANTI con problemi → 'pending'
        print(f"\n1️⃣ RAPPRESENTANTI con problemi → 'pending'")
        
        # Query per rappresentanti con problemi
        repr_problematic_query = {
            "classification_type": "RAPPRESENTANTE",
            "$or": [
                # Bassa confidenza generale
                {"confidence": {"$lt": CONFIDENCE_THRESHOLD}},
                # Bassa confidenza ML
                {"ml_confidence": {"$lt": CONFIDENCE_THRESHOLD}},
                # Bassa confidenza LLM  
                {"llm_confidence": {"$lt": CONFIDENCE_THRESHOLD}},
                # Disaccordo tra ML e LLM (predictions diverse)
                {"$expr": {"$ne": ["$ml_prediction", "$llm_prediction"]}},
                # Differenza significativa tra confidenze ML/LLM
                {"$expr": {"$gt": [{"$abs": {"$subtract": ["$ml_confidence", "$llm_confidence"]}}, DISAGREEMENT_THRESHOLD]}}
            ]
        }
        
        repr_count = collection.count_documents(repr_problematic_query)
        print(f"   📊 Rappresentanti con problemi: {repr_count}")
        
        if repr_count > 0:
            repr_result = collection.update_many(
                repr_problematic_query,
                {
                    "$set": {
                        "review_status": "pending",
                        "review_reason": "representative_low_confidence_or_disagreement",
                        "marked_for_review_at": datetime.now().isoformat()
                    }
                }
            )
            print(f"   ✅ Rappresentanti → pending: {repr_result.modified_count}")
        
        # STEP 4: OUTLIERS con problemi → 'pending'
        print(f"\n2️⃣ OUTLIERS con problemi → 'pending'")
        
        # Query per outliers con problemi (stessi criteri dei rappresentanti)
        outlier_problematic_query = {
            "classification_type": "OUTLIER",
            "$or": [
                {"confidence": {"$lt": CONFIDENCE_THRESHOLD}},
                {"ml_confidence": {"$lt": CONFIDENCE_THRESHOLD}},
                {"llm_confidence": {"$lt": CONFIDENCE_THRESHOLD}},
                {"$expr": {"$ne": ["$ml_prediction", "$llm_prediction"]}},
                {"$expr": {"$gt": [{"$abs": {"$subtract": ["$ml_confidence", "$llm_confidence"]}}, DISAGREEMENT_THRESHOLD]}}
            ]
        }
        
        outlier_count = collection.count_documents(outlier_problematic_query)
        print(f"   📊 Outliers con problemi: {outlier_count}")
        
        if outlier_count > 0:
            outlier_result = collection.update_many(
                outlier_problematic_query,
                {
                    "$set": {
                        "review_status": "pending",
                        "review_reason": "outlier_low_confidence_or_disagreement", 
                        "marked_for_review_at": datetime.now().isoformat()
                    }
                }
            )
            print(f"   ✅ Outliers → pending: {outlier_result.modified_count}")
        
        # STEP 5: PROPAGATI → sempre 'auto_classified' (regola ferrea)
        print(f"\n3️⃣ PROPAGATI → sempre 'auto_classified' (regola ferrea)")
        
        propagated_count = collection.count_documents({"classification_type": "PROPAGATO"})
        propagated_result = collection.update_many(
            {"classification_type": "PROPAGATO"},
            {
                "$set": {
                    "review_status": "auto_classified",
                    "review_reason": "propagated_never_in_queue_automatically"
                }
            }
        )
        print(f"   📊 Propagati totali: {propagated_count}")
        print(f"   ✅ Propagati → auto_classified: {propagated_result.modified_count}")
        
        print("\n" + "="*80)
        print("📊 RISULTATI FINALI")
        print("="*80)
        
        # Verifica risultati finali
        final_pending = collection.count_documents({"review_status": "pending"})
        final_auto_classified = collection.count_documents({"review_status": "auto_classified"})
        total = collection.count_documents({})
        
        print(f"✅ DOPO LA CORREZIONE LOGICA:")
        print(f"   🟡 Pending (Review Queue): {final_pending}")
        print(f"   🟢 Auto-classified: {final_auto_classified}")
        print(f"   📈 Totale: {total}")
        
        # Distribuzione dettagliata per pending
        if final_pending > 0:
            print(f"\n📂 DETTAGLIO REVIEW QUEUE ({final_pending} documenti):")
            
            pending_details = list(collection.aggregate([
                {"$match": {"review_status": "pending"}},
                {"$group": {
                    "_id": {
                        "classification_type": "$classification_type",
                        "review_reason": "$review_reason"
                    },
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id.classification_type": 1, "count": -1}}
            ]))
            
            for item in pending_details:
                class_type = item['_id']['classification_type'] or 'UNDEFINED'
                reason = item['_id']['review_reason'] or 'no_reason'
                count = item['count']
                print(f"   📄 {class_type:12} | {reason:35} | {count:3} docs")
        
        # Verifica nessun propagato in pending
        propagated_in_pending = collection.count_documents({
            "classification_type": "PROPAGATO",
            "review_status": "pending"
        })
        
        if propagated_in_pending == 0:
            print(f"\n✅ REGOLA RISPETTATA: Nessun propagato in Review Queue")
        else:
            print(f"\n❌ ERRORE: {propagated_in_pending} propagati ancora in pending!")
        
        client.close()
        
        return {
            "success": True,
            "final_pending": final_pending,
            "representatives_pending": repr_count if repr_count > 0 else 0,
            "outliers_pending": outlier_count if outlier_count > 0 else 0,
            "propagated_auto_classified": propagated_count
        }
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def test_review_queue_logic():
    """
    Testa la logica corretta senza modificare i dati
    """
    print("🧪 TEST DELLA LOGICA (senza modifiche)")
    print("="*50)
    
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        mongo_url = config.get('mongodb', {}).get('url', 'mongodb://localhost:27017')
        mongo_db_name = config.get('mongodb', {}).get('database', 'classificazioni')
        
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        db = client[mongo_db_name]
        
        collections = db.list_collection_names()
        humanitas_collection = None
        for coll_name in collections:
            if 'humanitas' in coll_name.lower():
                humanitas_collection = coll_name
                break
        
        collection = db[humanitas_collection]
        
        CONFIDENCE_THRESHOLD = 0.8
        DISAGREEMENT_THRESHOLD = 0.3
        
        # Test rappresentanti con problemi
        repr_problematic = collection.count_documents({
            "classification_type": "RAPPRESENTANTE",
            "$or": [
                {"confidence": {"$lt": CONFIDENCE_THRESHOLD}},
                {"ml_confidence": {"$lt": CONFIDENCE_THRESHOLD}},
                {"llm_confidence": {"$lt": CONFIDENCE_THRESHOLD}},
                {"$expr": {"$ne": ["$ml_prediction", "$llm_prediction"]}},
                {"$expr": {"$gt": [{"$abs": {"$subtract": ["$ml_confidence", "$llm_confidence"]}}, DISAGREEMENT_THRESHOLD]}}
            ]
        })
        
        # Test outliers con problemi
        outlier_problematic = collection.count_documents({
            "classification_type": "OUTLIER",
            "$or": [
                {"confidence": {"$lt": CONFIDENCE_THRESHOLD}},
                {"ml_confidence": {"$lt": CONFIDENCE_THRESHOLD}},
                {"llm_confidence": {"$lt": CONFIDENCE_THRESHOLD}},
                {"$expr": {"$ne": ["$ml_prediction", "$llm_prediction"]}},
                {"$expr": {"$gt": [{"$abs": {"$subtract": ["$ml_confidence", "$llm_confidence"]}}, DISAGREEMENT_THRESHOLD]}}
            ]
        })
        
        total_propagated = collection.count_documents({"classification_type": "PROPAGATO"})
        
        print(f"📊 SIMULAZIONE RISULTATI:")
        print(f"   🟡 Rappresentanti → pending: {repr_problematic}")
        print(f"   🟡 Outliers → pending: {outlier_problematic}")
        print(f"   🟢 Propagati → auto_classified: {total_propagated}")
        print(f"   📈 Totale in Review Queue: {repr_problematic + outlier_problematic}")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Errore test: {e}")

if __name__ == "__main__":
    print("🔧 CORREZIONE LOGICA REVIEW QUEUE")
    print("="*40)
    
    # Prima mostra la simulazione
    test_review_queue_logic()
    
    print(f"\n" + "="*40)
    confirm = input("\n❓ Applicare la correzione logica? (y/N): ").lower()
    
    if confirm == 'y':
        result = apply_correct_review_logic()
        if result.get("success"):
            print(f"\n🎉 CORREZIONE COMPLETATA!")
            print(f"   Review Queue ora contiene {result['final_pending']} documenti")
        else:
            print(f"\n❌ Errore: {result.get('error')}")
    else:
        print("❌ Operazione annullata.")
