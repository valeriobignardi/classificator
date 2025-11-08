#!/usr/bin/env python3
"""
File: debug_review_queue_wopta.py
Autore: Valerio Bignardi
Data creazione: 2025-01-31
Descrizione: Analisi della collection wopta in MongoDB per debug review queue

Scopo: Identificare perché non si vedono casi da revisionare e verificare presenza rappresentanti
"""

import sys
import os
import json
from datetime import datetime
from pymongo import MongoClient
import yaml

def load_config():
    """
    Carica configurazione dal file config.yaml
    
    Returns:
        dict: Configurazione caricata
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    try:
        config = load_config()
        return config
    except Exception as e:
        print(f"❌ Errore caricamento config: {e}")
        return None

def connect_mongodb(config):
    """
    Connessione a MongoDB
    
    Args:
        config: Configurazione sistema
        
    Returns:
        Database connection
    """
    try:
        mongodb_config = config.get('mongodb', {})
        url = mongodb_config.get('url', 'mongodb://localhost:27017')
        database_name = mongodb_config.get('database', 'classificazioni')
        
        print(f"🔗 Connessione a MongoDB: {url}")
        client = MongoClient(url)
        db = client[database_name]
        
        # Test connessione
        db.command('ping')
        print(f"✅ Connesso al database: {database_name}")
        
        return db, client
    except Exception as e:
        print(f"❌ Errore connessione MongoDB: {e}")
        return None, None

def analyze_wopta_collection(db):
    """
    Analizza la collection wopta per debug review queue
    
    Args:
        db: Database MongoDB
    """
    print(f"\n{'='*80}")
    print(f"📊 ANALISI COLLECTION WOPTA")
    print(f"{'='*80}")
    
    try:
        collection = db['wopta']
        
        # 1. Conteggio totale documenti
        total_count = collection.count_documents({})
        print(f"\n1️⃣ TOTALE DOCUMENTI: {total_count}")
        
        if total_count == 0:
            print("⚠️  La collection wopta è vuota!")
            return
        
        # 2. Analisi struttura documenti
        print(f"\n2️⃣ STRUTTURA DOCUMENTI:")
        sample_doc = collection.find_one({})
        if sample_doc:
            print(f"📋 Campi presenti nel documento:")
            for key in sample_doc.keys():
                if key != '_id':
                    value = sample_doc[key]
                    if isinstance(value, (str, int, float, bool)):
                        print(f"   • {key}: {type(value).__name__} = {value}")
                    else:
                        print(f"   • {key}: {type(value).__name__}")
        
        # 3. Verifica campi per review queue
        print(f"\n3️⃣ CAMPI REVIEW QUEUE:")
        review_fields = [
            'review_status', 'requires_review', 'human_review_needed',
            'is_reviewed', 'review_pending', 'needs_human_review'
        ]
        
        for field in review_fields:
            count = collection.count_documents({field: {"$exists": True}})
            if count > 0:
                print(f"   ✅ {field}: {count} documenti")
                # Mostra valori distinti
                distinct_values = collection.distinct(field)
                print(f"      Valori: {distinct_values}")
            else:
                print(f"   ❌ {field}: campo non presente")
        
        # 4. Verifica rappresentanti
        print(f"\n4️⃣ VERIFICA RAPPRESENTANTI:")
        representative_fields = [
            'is_representative', 'representative', 'cluster_representative',
            'is_cluster_representative'
        ]
        
        representative_count = 0
        for field in representative_fields:
            count = collection.count_documents({field: True})
            if count > 0:
                print(f"   ✅ {field}=True: {count} documenti")
                representative_count += count
            else:
                count_exists = collection.count_documents({field: {"$exists": True}})
                if count_exists > 0:
                    print(f"   ⚠️  {field}: {count_exists} documenti (ma nessuno True)")
                else:
                    print(f"   ❌ {field}: campo non presente")
        
        print(f"   📊 TOTALE RAPPRESENTANTI TROVATI: {representative_count}")
        
        # 5. Analisi cluster_id
        print(f"\n5️⃣ ANALISI CLUSTER:")
        cluster_fields = ['cluster_id', 'cluster', 'cluster_label']
        
        for field in cluster_fields:
            count = collection.count_documents({field: {"$exists": True}})
            if count > 0:
                distinct_clusters = collection.distinct(field)
                print(f"   ✅ {field}: {count} documenti, {len(distinct_clusters)} cluster distinti")
                if len(distinct_clusters) < 20:  # Mostra solo se pochi cluster
                    print(f"      Cluster: {distinct_clusters}")
            else:
                print(f"   ❌ {field}: campo non presente")
        
        # 6. Analisi date e timestamp
        print(f"\n6️⃣ ANALISI TEMPORALE:")
        date_fields = [
            'created_at', 'updated_at', 'timestamp', 'date_created',
            'last_modified', 'processed_at'
        ]
        
        for field in date_fields:
            count = collection.count_documents({field: {"$exists": True}})
            if count > 0:
                # Trova date min/max
                pipeline = [
                    {"$match": {field: {"$exists": True}}},
                    {"$group": {
                        "_id": None,
                        "min_date": {"$min": f"${field}"},
                        "max_date": {"$max": f"${field}"}
                    }}
                ]
                result = list(collection.aggregate(pipeline))
                if result:
                    min_date = result[0]['min_date']
                    max_date = result[0]['max_date']
                    print(f"   ✅ {field}: {count} documenti, da {min_date} a {max_date}")
                else:
                    print(f"   ✅ {field}: {count} documenti")
            else:
                print(f"   ❌ {field}: campo non presente")
        
        # 7. Ricerca pattern review queue
        print(f"\n7️⃣ PATTERN REVIEW QUEUE:")
        
        # Query possibili per review queue
        review_queries = [
            {"review_status": "pending"},
            {"review_status": "requires_review"},
            {"requires_review": True},
            {"human_review_needed": True},
            {"needs_human_review": True},
            {"is_reviewed": False},
            {"review_pending": True}
        ]
        
        for query in review_queries:
            count = collection.count_documents(query)
            if count > 0:
                print(f"   ✅ Query {query}: {count} documenti trovati")
                # Mostra esempio
                example = collection.find_one(query)
                if example:
                    print(f"      Esempio: {example.get('_id')}")
            else:
                print(f"   ❌ Query {query}: nessun documento")
        
        # 8. Documenti recenti (ultimi 7 giorni)
        print(f"\n8️⃣ DOCUMENTI RECENTI:")
        from datetime import datetime, timedelta
        
        recent_threshold = datetime.now() - timedelta(days=7)
        
        for field in ['created_at', 'timestamp', 'updated_at']:
            try:
                recent_count = collection.count_documents({
                    field: {"$gte": recent_threshold}
                })
                if recent_count > 0:
                    print(f"   ✅ {field} >= {recent_threshold.strftime('%Y-%m-%d')}: {recent_count} documenti")
                else:
                    print(f"   ❌ {field}: nessun documento recente")
            except:
                print(f"   ⚠️  {field}: errore nel controllo date")
        
        # 9. Esempi di documenti per debug
        print(f"\n9️⃣ ESEMPI DOCUMENTI:")
        
        # Mostra 3 documenti di esempio
        examples = list(collection.find({}).limit(3))
        for i, doc in enumerate(examples, 1):
            print(f"\n   📄 DOCUMENTO {i}:")
            doc_copy = dict(doc)
            if '_id' in doc_copy:
                print(f"      _id: {doc_copy.pop('_id')}")
            
            for key, value in doc_copy.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"      {key}: {value[:100]}... (troncato)")
                else:
                    print(f"      {key}: {value}")
        
    except Exception as e:
        print(f"❌ Errore analisi collection wopta: {e}")
        import traceback
        traceback.print_exc()

def check_review_queue_filters(db):
    """
    Controlla i filtri specifici della review queue
    
    Args:
        db: Database MongoDB
    """
    print(f"\n{'='*80}")
    print(f"🔍 CONTROLLO FILTRI REVIEW QUEUE")
    print(f"{'='*80}")
    
    try:
        collection = db['wopta']
        
        # Controlla i filtri tipici della review queue
        filters_to_check = [
            # Filtro 1: Documenti non revisionati
            {"$or": [
                {"review_status": {"$in": ["pending", "requires_review"]}},
                {"requires_review": True},
                {"human_review_needed": True},
                {"is_reviewed": False}
            ]},
            
            # Filtro 2: Rappresentanti non revisionati
            {"$and": [
                {"is_representative": True},
                {"$or": [
                    {"review_status": {"$ne": "completed"}},
                    {"is_reviewed": {"$ne": True}}
                ]}
            ]},
            
            # Filtro 3: Documenti con bassa confidence
            {"confidence": {"$lt": 0.7}},
            
            # Filtro 4: Documenti senza etichetta o con "altro"
            {"$or": [
                {"predicted_label": {"$exists": False}},
                {"predicted_label": "altro"},
                {"predicted_label": "ALTRO"},
                {"predicted_label": None}
            ]}
        ]
        
        for i, filter_query in enumerate(filters_to_check, 1):
            print(f"\n🔎 FILTRO {i}: {filter_query}")
            
            try:
                count = collection.count_documents(filter_query)
                print(f"   📊 Risultati: {count} documenti")
                
                if count > 0:
                    # Mostra alcuni esempi
                    examples = list(collection.find(filter_query).limit(2))
                    for j, doc in enumerate(examples, 1):
                        print(f"   📄 Esempio {j}:")
                        print(f"      _id: {doc.get('_id')}")
                        print(f"      review_status: {doc.get('review_status')}")
                        print(f"      is_reviewed: {doc.get('is_reviewed')}")
                        print(f"      requires_review: {doc.get('requires_review')}")
                        print(f"      predicted_label: {doc.get('predicted_label')}")
                        print(f"      confidence: {doc.get('confidence')}")
                        print(f"      is_representative: {doc.get('is_representative')}")
                
            except Exception as e:
                print(f"   ❌ Errore esecuzione filtro: {e}")
    
    except Exception as e:
        print(f"❌ Errore controllo filtri: {e}")

def main():
    """
    Funzione principale per debug review queue MongoDB
    """
    print(f"🔍 DEBUG REVIEW QUEUE WOPTA - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Carica configurazione
    config = load_config()
    if not config:
        return
    
    # 2. Connetti a MongoDB
    db, client = connect_mongodb(config)
    if db is None:
        return
    
    try:
        # 3. Analizza collection wopta
        analyze_wopta_collection(db)
        
        # 4. Controlla filtri review queue
        check_review_queue_filters(db)
        
        print(f"\n{'='*80}")
        print(f"✅ ANALISI COMPLETATA")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ Errore durante analisi: {e}")
        import traceback
from config_loader import load_config
        traceback.print_exc()
    
    finally:
        if client:
            client.close()
            print(f"🔌 Connessione MongoDB chiusa")

if __name__ == "__main__":
    main()
