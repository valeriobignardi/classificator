#!/usr/bin/env python3
"""
Script per implementare una correzione al problema Review Queue.

Questa correzione ripristina alcuni documenti da 'auto_classified' a 'pending'
basandosi su criteri intelligenti per il sistema di review umano.

Criteri per tornare in 'pending':
1. Rappresentanti di cluster (sempre da rivedere)
2. Outliers con bassa confidenza
3. Documenti con classificazioni controverse

Autore: Valerio Bignardi
Data: 2025-09-04
"""

import yaml
from pymongo import MongoClient
from datetime import datetime
import json

def restore_pending_status():
    """
    Ripristina lo status 'pending' per documenti che dovrebbero essere in Review Queue
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
        
        print(f"🎯 Correzione collection: {humanitas_collection}")
        collection = db[humanitas_collection]
        
        print("\n" + "="*80)
        print("🔧 STRATEGIA DI CORREZIONE")
        print("="*80)
        
        # STRATEGIA 1: Ripristina TUTTI i rappresentanti di cluster
        print("\n1️⃣ RAPPRESENTANTI DI CLUSTER → 'pending'")
        
        # Prima contiamo
        repr_count = collection.count_documents({
            "review_status": "auto_classified",
            "classification_type": "RAPPRESENTANTE"  # Questi sono i veri rappresentanti
        })
        
        print(f"   📊 Trovati {repr_count} rappresentanti auto-classificati")
        
        if repr_count > 0:
            repr_result = collection.update_many(
                {
                    "review_status": "auto_classified",
                    "classification_type": "RAPPRESENTANTE"
                },
                {
                    "$set": {
                        "review_status": "pending",
                        "review_reason": "representative_needs_human_review",
                        "marked_for_review_at": datetime.now().isoformat()
                    }
                }
            )
            print(f"   ✅ Ripristinati {repr_result.modified_count} rappresentanti → 'pending'")
        
        # STRATEGIA 2: Ripristina OUTLIER con bassa confidenza
        print("\n2️⃣ OUTLIERS CON BASSA CONFIDENZA → 'pending'")
        
        # Outlier con confidenza < 0.8
        outlier_count = collection.count_documents({
            "review_status": "auto_classified",
            "classification_type": "OUTLIER",
            "confidence": {"$lt": 0.8}
        })
        
        print(f"   📊 Trovati {outlier_count} outlier con confidenza < 0.8")
        
        if outlier_count > 0:
            outlier_result = collection.update_many(
                {
                    "review_status": "auto_classified",
                    "classification_type": "OUTLIER",
                    "confidence": {"$lt": 0.8}
                },
                {
                    "$set": {
                        "review_status": "pending",
                        "review_reason": "low_confidence_outlier",
                        "marked_for_review_at": datetime.now().isoformat()
                    }
                }
            )
            print(f"   ✅ Ripristinati {outlier_result.modified_count} outlier → 'pending'")
        
        # STRATEGIA 3: Campione casuale di PROPAGATI per supervisione
        print("\n3️⃣ CAMPIONE PROPAGATI PER SUPERVISIONE → 'pending'")
        
        # Prendi un 5% casuale dei propagati per supervisione umana
        total_propagated = collection.count_documents({
            "review_status": "auto_classified",
            "classification_type": "PROPAGATO"
        })
        
        sample_size = max(10, int(total_propagated * 0.05))  # Minimo 10, massimo 5%
        print(f"   📊 Totale propagati: {total_propagated}, campione: {sample_size}")
        
        # Usa aggregation per campione casuale
        if total_propagated > 0:
            sample_pipeline = [
                {"$match": {
                    "review_status": "auto_classified",
                    "classification_type": "PROPAGATO"
                }},
                {"$sample": {"size": sample_size}}
            ]
            
            sample_docs = list(collection.aggregate(sample_pipeline))
            sample_ids = [doc['_id'] for doc in sample_docs]
            
            if sample_ids:
                sample_result = collection.update_many(
                    {"_id": {"$in": sample_ids}},
                    {
                        "$set": {
                            "review_status": "pending",
                            "review_reason": "supervised_training_representative",
                            "marked_for_review_at": datetime.now().isoformat()
                        }
                    }
                )
                print(f"   ✅ Ripristinati {sample_result.modified_count} propagati (campione) → 'pending'")
        
        print("\n" + "="*80)
        print("📊 VERIFICA RISULTATI")
        print("="*80)
        
        # Controlla la nuova distribuzione
        new_pending = collection.count_documents({"review_status": "pending"})
        new_auto_classified = collection.count_documents({"review_status": "auto_classified"})
        total = collection.count_documents({})
        
        print(f"✅ DOPO LA CORREZIONE:")
        print(f"   🟡 Pending (Review Queue): {new_pending}")
        print(f"   🟢 Auto-classified: {new_auto_classified}")
        print(f"   📈 Totale: {total}")
        
        if new_pending > 0:
            print(f"\n🎉 SUCCESSO! La Review Queue ora contiene {new_pending} documenti")
            
            # Mostra distribuzione per tipo
            pending_by_type = list(collection.aggregate([
                {"$match": {"review_status": "pending"}},
                {"$group": {"_id": "$classification_type", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]))
            
            print(f"   📂 Distribuzione per tipo:")
            for item in pending_by_type:
                type_name = item['_id'] or 'UNDEFINED'
                count = item['count']
                print(f"      {type_name:15}: {count:3} documenti")
        else:
            print(f"\n⚠️ Nessun documento ripristinato. Potrebbe essere necessario un approccio diverso.")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 CORREZIONE REVIEW QUEUE - Ripristino documenti 'pending'")
    print("="*60)
    
    # Chiedi conferma all'utente
    confirm = input("\n❓ Vuoi procedere con la correzione? (y/N): ").lower()
    if confirm == 'y':
        restore_pending_status()
        print("\n✅ Correzione completata!")
    else:
        print("❌ Operazione annullata.")
