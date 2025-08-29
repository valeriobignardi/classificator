#!/usr/bin/env python3
"""
Autore: Valerio Bignardi
Data creazione: 2025-08-29
Descrizione: Script per aggiornare i record esistenti con il campo classification_type
Storia aggiornamenti:
- 2025-08-29: Creazione iniziale
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mongo_classification_reader import MongoClassificationReader
import yaml

def update_existing_records():
    """
    Aggiorna tutti i record esistenti per aggiungere il campo classification_type
    basandosi sui metadati esistenti
    """
    print("🔄 AGGIORNAMENTO RECORDS ESISTENTI CON CLASSIFICATION_TYPE")
    print("="*60)
    
    # Carica configurazione
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Crea reader MongoDB
    reader = MongoClassificationReader(config)
    
    if not reader.ensure_connection():
        print("❌ Impossibile connettersi al database")
        return False
    
    # Lista tutte le collections wopta
    db = reader.db
    collections = db.list_collection_names()
    wopta_collections = [c for c in collections if 'wopta' in c]
    
    print(f"📋 Trovate {len(wopta_collections)} collections wopta:")
    for col in wopta_collections:
        print(f"   - {col}")
    
    total_updated = 0
    total_processed = 0
    
    for collection_name in wopta_collections:
        print(f"\n🎯 Processando collection: {collection_name}")
        collection = db[collection_name]
        
        # Conta documenti totali
        total_docs = collection.count_documents({})
        print(f"   📊 Documenti totali: {total_docs}")
        
        # Conta documenti senza classification_type
        missing_type = collection.count_documents({
            "classification_type": {"$exists": False}
        })
        print(f"   🔍 Documenti senza classification_type: {missing_type}")
        
        if missing_type == 0:
            print(f"   ✅ Tutti i documenti hanno già classification_type")
            continue
        
        # Aggiorna documenti in batch
        print(f"   🔄 Aggiornamento in corso...")
        
        # Batch 1: Documenti con metadata.is_representative = true -> RAPPRESENTANTE
        result1 = collection.update_many(
            {
                "classification_type": {"$exists": False},
                "metadata.is_representative": True
            },
            {"$set": {"classification_type": "RAPPRESENTANTE"}}
        )
        print(f"      ✅ RAPPRESENTANTI: {result1.modified_count}")
        
        # Batch 2: Documenti con metadata.propagated_from -> PROPAGATO
        result2 = collection.update_many(
            {
                "classification_type": {"$exists": False},
                "metadata.propagated_from": {"$exists": True, "$ne": None}
            },
            {"$set": {"classification_type": "PROPAGATO"}}
        )
        print(f"      ✅ PROPAGATI: {result2.modified_count}")
        
        # Batch 3: Outliers - cluster_id con "outlier" o = -1
        result3 = collection.update_many(
            {
                "classification_type": {"$exists": False},
                "$or": [
                    {"metadata.cluster_id": {"$regex": "outlier", "$options": "i"}},
                    {"metadata.cluster_id": -1},
                    {"metadata.cluster_id": "-1"},
                    {"metadata.outlier_score": {"$gt": 0.5}},
                    {"session_type": "outlier"}
                ]
            },
            {"$set": {"classification_type": "OUTLIER"}}
        )
        print(f"      ✅ OUTLIERS: {result3.modified_count}")
        
        # Batch 4: Tutti i rimanenti -> NORMALE
        result4 = collection.update_many(
            {"classification_type": {"$exists": False}},
            {"$set": {"classification_type": "NORMALE"}}
        )
        print(f"      ✅ NORMALI: {result4.modified_count}")
        
        collection_updated = (result1.modified_count + result2.modified_count + 
                            result3.modified_count + result4.modified_count)
        
        print(f"   📊 Totale aggiornati in {collection_name}: {collection_updated}")
        
        total_updated += collection_updated
        total_processed += total_docs
    
    print(f"\n🎉 AGGIORNAMENTO COMPLETATO!")
    print(f"   📊 Documenti processati: {total_processed}")
    print(f"   ✅ Documenti aggiornati: {total_updated}")
    
    # Verifica finale
    print(f"\n🔍 VERIFICA FINALE:")
    for collection_name in wopta_collections:
        collection = db[collection_name]
        
        # Conta per tipo
        rappresentanti = collection.count_documents({"classification_type": "RAPPRESENTANTE"})
        propagati = collection.count_documents({"classification_type": "PROPAGATO"})
        outliers = collection.count_documents({"classification_type": "OUTLIER"})
        normali = collection.count_documents({"classification_type": "NORMALE"})
        
        print(f"   {collection_name}:")
        print(f"      🎯 RAPPRESENTANTI: {rappresentanti}")
        print(f"      🔗 PROPAGATI: {propagati}")
        print(f"      ⚠️  OUTLIERS: {outliers}")
        print(f"      📝 NORMALI: {normali}")
    
    reader.disconnect()
    return True

if __name__ == "__main__":
    success = update_existing_records()
    if success:
        print("\n✅ Script completato con successo!")
        print("🚀 L'interfaccia Human Review ora dovrebbe mostrare correttamente i tipi di classificazione!")
    else:
        print("\n❌ Script completato con errori")
        sys.exit(1)
