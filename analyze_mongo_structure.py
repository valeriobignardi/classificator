#!/usr/bin/env python3
"""
Analizza la struttura dei documenti nella collezione MongoDB di Humanitas
per capire i campi disponibili per le statistiche
"""

import yaml
from pymongo import MongoClient
import json
from pprint import pprint

def analyze_mongo_collection():
    """
    Analizza la collezione MongoDB di Humanitas per capire la struttura dei documenti
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
        print(f"📋 Collezioni disponibili: {collections}")
        
        humanitas_collection = None
        for coll_name in collections:
            if 'humanitas' in coll_name.lower():
                humanitas_collection = coll_name
                break
        
        if not humanitas_collection:
            print("❌ Nessuna collezione Humanitas trovata")
            return
        
        print(f"\n🎯 Analizzo collezione: {humanitas_collection}")
        collection = db[humanitas_collection]
        
        # Conta totale documenti
        total_docs = collection.count_documents({})
        print(f"📊 Totale documenti: {total_docs}")
        
        if total_docs == 0:
            print("❌ Collezione vuota")
            return
        
        print("\n" + "="*80)
        print("🔍 ANALISI CAMPIONI DI DOCUMENTI")
        print("="*80)
        
        # Cerca documenti rappresentanti
        print("\n👑 RAPPRESENTANTI:")
        representatives = list(collection.find(
            {"$or": [
                {"is_representative": True},
                {"classification_type": "RAPPRESENTANTE"},
                {"type": "representative"}
            ]}
        ).limit(2))
        
        if representatives:
            print(f"✅ Trovati {len(representatives)} rappresentanti")
            for i, doc in enumerate(representatives):
                print(f"\n--- RAPPRESENTANTE #{i+1} ---")
                # Rimuovi _id per leggibilità
                if '_id' in doc:
                    del doc['_id']
                pprint(doc, width=100)
        else:
            print("❌ Nessun rappresentante trovato con i criteri: is_representative=True, classification_type='RAPPRESENTANTE', type='representative'")
        
        # Cerca documenti outlier
        print("\n🚨 OUTLIERS:")
        outliers = list(collection.find(
            {"$or": [
                {"is_outlier": True},
                {"classification_type": "OUTLIER"},
                {"type": "outlier"}
            ]}
        ).limit(2))
        
        if outliers:
            print(f"✅ Trovati {len(outliers)} outliers")
            for i, doc in enumerate(outliers):
                print(f"\n--- OUTLIER #{i+1} ---")
                if '_id' in doc:
                    del doc['_id']
                pprint(doc, width=100)
        else:
            print("❌ Nessun outlier trovato con i criteri: is_outlier=True, classification_type='OUTLIER', type='outlier'")
        
        # Cerca documenti propagati
        print("\n🔗 PROPAGATI:")
        propagated = list(collection.find(
            {"$or": [
                {"is_representative": False},
                {"classification_type": "PROPAGATO"},
                {"type": "propagated"}
            ]}
        ).limit(2))
        
        if propagated:
            print(f"✅ Trovati {len(propagated)} propagati")
            for i, doc in enumerate(propagated):
                print(f"\n--- PROPAGATO #{i+1} ---")
                if '_id' in doc:
                    del doc['_id']
                pprint(doc, width=100)
        else:
            print("❌ Nessun propagato trovato con i criteri: is_representative=False, classification_type='PROPAGATO', type='propagated'")
        
        # Analisi generale dei campi
        print("\n" + "="*80)
        print("🔍 ANALISI STRUTTURA GENERALE")
        print("="*80)
        
        # Prendi alcuni documenti casuali per analizzare tutti i campi
        sample_docs = list(collection.find().limit(5))
        all_fields = set()
        
        for doc in sample_docs:
            all_fields.update(doc.keys())
        
        print(f"\n📋 Tutti i campi trovati nei documenti ({len(all_fields)} campi):")
        for field in sorted(all_fields):
            print(f"   - {field}")
        
        # Analizza i campi che potrebbero contenere classificazioni
        print(f"\n🔍 ANALISI CAMPI CLASSIFICAZIONE:")
        classification_fields = [field for field in all_fields if any(keyword in field.lower() for keyword in ['classif', 'tag', 'label', 'prediction', 'decision'])]
        
        if classification_fields:
            for field in classification_fields:
                # Conta valori unici per questo campo
                pipeline = [
                    {"$match": {field: {"$exists": True, "$ne": None}}},
                    {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": 10}
                ]
                
                try:
                    values = list(collection.aggregate(pipeline))
                    print(f"\n   📊 {field}:")
                    for value in values:
                        print(f"      '{value['_id']}': {value['count']} occorrenze")
                except Exception as e:
                    print(f"   ❌ Errore analizzando {field}: {e}")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_mongo_collection()
