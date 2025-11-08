#!/usr/bin/env python3
"""
File: analyze_wopta_tenant_collection.py
Autore: Valerio Bignardi
Data creazione: 2025-01-31
Descrizione: Analisi dettagliata della collection wopta con tenant_id

Scopo: Identificare perché non si vedono casi da revisionare e verificare rappresentanti
"""

import sys
import os
import json
from datetime import datetime, timedelta
from pymongo import MongoClient
import yaml

def load_config():
    """Carica configurazione dal file config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    try:
        config = load_config()
        return config
    except Exception as e:
        print(f"❌ Errore caricamento config: {e}")
        return None

def connect_mongodb(config):
    """Connessione a MongoDB"""
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

def analyze_wopta_tenant_collection(db):
    """
    Analizza la collection wopta con tenant_id
    
    Args:
        db: Database MongoDB
    """
    print(f"\n{'='*80}")
    print(f"📊 ANALISI DETTAGLIATA COLLECTION WOPTA TENANT")
    print(f"{'='*80}")
    
    # Trova la collection wopta corretta
    collections = db.list_collection_names()
    wopta_collection_name = None
    
    for collection_name in collections:
        if 'wopta' in collection_name.lower():
            wopta_collection_name = collection_name
            break
    
    if not wopta_collection_name:
        print("❌ Nessuna collection wopta trovata!")
        return
    
    print(f"🎯 Collection trovata: {wopta_collection_name}")
    collection = db[wopta_collection_name]
    
    try:
        # 1. Statistiche generali
        total_count = collection.count_documents({})
        print(f"\n1️⃣ STATISTICHE GENERALI:")
        print(f"   📊 Totale documenti: {total_count}")
        
        # 2. Analisi review_status
        print(f"\n2️⃣ ANALISI REVIEW STATUS:")
        review_statuses = collection.distinct('review_status')
        print(f"   📋 Valori review_status: {review_statuses}")
        
        for status in review_statuses:
            count = collection.count_documents({'review_status': status})
            print(f"   • {status}: {count} documenti")
        
        # 3. Analisi classificazioni
        print(f"\n3️⃣ ANALISI CLASSIFICAZIONI:")
        classifications = collection.distinct('classification')
        print(f"   📋 Classificazioni: {classifications}")
        
        for classification in classifications:
            count = collection.count_documents({'classification': classification})
            print(f"   • {classification}: {count} documenti")
        
        # 4. Analisi confidence
        print(f"\n4️⃣ ANALISI CONFIDENCE:")
        
        confidence_ranges = [
            ("Molto bassa (< 0.3)", {"confidence": {"$lt": 0.3}}),
            ("Bassa (0.3-0.7)", {"confidence": {"$gte": 0.3, "$lt": 0.7}}),
            ("Alta (0.7-0.9)", {"confidence": {"$gte": 0.7, "$lt": 0.9}}),
            ("Molto alta (>= 0.9)", {"confidence": {"$gte": 0.9}})
        ]
        
        for range_name, query in confidence_ranges:
            count = collection.count_documents(query)
            print(f"   • {range_name}: {count} documenti")
        
        # 5. Ricerca rappresentanti
        print(f"\n5️⃣ VERIFICA RAPPRESENTANTI:")
        
        representative_fields = [
            'is_representative', 'representative', 'cluster_representative',
            'is_cluster_representative', 'is_outlier_representative'
        ]
        
        total_representatives = 0
        for field in representative_fields:
            # Controlla presenza campo
            field_exists = collection.count_documents({field: {"$exists": True}})
            if field_exists > 0:
                print(f"   📋 Campo {field}: {field_exists} documenti lo hanno")
                
                # Controlla valori True
                true_count = collection.count_documents({field: True})
                if true_count > 0:
                    print(f"      ✅ {field}=True: {true_count} documenti")
                    total_representatives += true_count
                    
                    # Mostra esempio
                    example = collection.find_one({field: True})
                    if example:
                        print(f"         📄 Esempio: {example.get('session_id', 'N/A')}")
                else:
                    # Mostra valori diversi da True
                    distinct_values = collection.distinct(field)
                    print(f"      ⚠️  Valori trovati: {distinct_values}")
            else:
                print(f"   ❌ Campo {field}: non presente")
        
        print(f"   📊 TOTALE RAPPRESENTANTI: {total_representatives}")
        
        # 6. Analisi clustering
        print(f"\n6️⃣ ANALISI CLUSTERING:")
        
        cluster_fields = ['cluster_id', 'cluster', 'cluster_label']
        for field in cluster_fields:
            field_exists = collection.count_documents({field: {"$exists": True}})
            if field_exists > 0:
                distinct_clusters = collection.distinct(field)
                print(f"   ✅ {field}: {field_exists} documenti, {len(distinct_clusters)} cluster")
                if len(distinct_clusters) <= 10:
                    print(f"      Cluster: {distinct_clusters}")
            else:
                print(f"   ❌ {field}: non presente")
        
        # 7. Controllo filtri review queue
        print(f"\n7️⃣ FILTRI REVIEW QUEUE:")
        
        review_filters = [
            ("Pending review", {"review_status": "pending"}),
            ("Requires review", {"review_status": "requires_review"}),
            ("Not completed", {"review_status": {"$ne": "completed"}}),
            ("Confidence < 0.7", {"confidence": {"$lt": 0.7}}),
            ("Classification ALTRO", {"classification": {"$in": ["ALTRO", "altro"]}}),
            ("Recent (last 7 days)", {"classified_at": {"$gte": (datetime.now() - timedelta(days=7)).isoformat()}}),
        ]
        
        candidates_found = 0
        for filter_name, filter_query in review_filters:
            try:
                count = collection.count_documents(filter_query)
                print(f"   • {filter_name}: {count} documenti")
                if count > 0:
                    candidates_found += count
                    
                    # Mostra esempio per filtri importanti
                    if "confidence" in filter_name.lower() or "altro" in filter_name.lower():
                        example = collection.find_one(filter_query)
                        if example:
                            print(f"      📄 Esempio:")
                            print(f"         session_id: {example.get('session_id')}")
                            print(f"         classification: {example.get('classification')}")
                            print(f"         confidence: {example.get('confidence')}")
                            print(f"         review_status: {example.get('review_status')}")
            except Exception as e:
                print(f"      ❌ Errore filtro {filter_name}: {e}")
        
        print(f"   📊 TOTALE CANDIDATI POTENZIALI: {candidates_found} (con sovrapposizioni)")
        
        # 8. Analisi date
        print(f"\n8️⃣ ANALISI TEMPORALE:")
        
        # Trova range date
        try:
            pipeline = [
                {"$match": {"classified_at": {"$exists": True}}},
                {"$group": {
                    "_id": None,
                    "min_date": {"$min": "$classified_at"},
                    "max_date": {"$max": "$classified_at"},
                    "count": {"$sum": 1}
                }}
            ]
            result = list(collection.aggregate(pipeline))
            if result:
                data = result[0]
                print(f"   📅 Date classificazione:")
                print(f"      Da: {data['min_date']}")
                print(f"      A: {data['max_date']}")
                print(f"      Documenti con data: {data['count']}")
            
            # Documenti recenti
            recent_threshold = datetime.now() - timedelta(days=7)
            recent_count = collection.count_documents({
                "classified_at": {"$gte": recent_threshold.isoformat()}
            })
            print(f"   📊 Documenti ultimi 7 giorni: {recent_count}")
            
        except Exception as e:
            print(f"   ❌ Errore analisi date: {e}")
        
        # 9. Esempi documenti per debug
        print(f"\n9️⃣ ESEMPI DOCUMENTI:")
        
        # Documento con confidence bassa
        low_confidence = collection.find_one({"confidence": {"$lt": 0.5}})
        if low_confidence:
            print(f"   📄 DOCUMENTO CONFIDENCE BASSA:")
            relevant_fields = ['session_id', 'classification', 'confidence', 'review_status', 
                             'classified_at', 'classification_method', 'classification_type']
            for field in relevant_fields:
                if field in low_confidence:
                    print(f"      {field}: {low_confidence[field]}")
        
        # Documento ALTRO
        altro_doc = collection.find_one({"classification": "ALTRO"})
        if altro_doc:
            print(f"   📄 DOCUMENTO CLASSIFICATO ALTRO:")
            for field in relevant_fields:
                if field in altro_doc:
                    print(f"      {field}: {altro_doc[field]}")
        
        # 10. Proposta filtro review queue
        print(f"\n🔟 PROPOSTA FILTRO REVIEW QUEUE:")
        
        # Filtro combinato per trovare documenti che necessitano review
        review_query = {
            "$or": [
                {"confidence": {"$lt": 0.7}},
                {"classification": {"$in": ["ALTRO", "altro"]}},
                {"review_status": {"$in": ["pending", "requires_review"]}}
            ]
        }
        
        potential_reviews = collection.count_documents(review_query)
        print(f"   🎯 DOCUMENTI CHE NECESSITANO REVIEW: {potential_reviews}")
        
        if potential_reviews > 0:
            print(f"   💡 Query suggerita:")
            print(f"      {json.dumps(review_query, indent=6)}")
            
            # Mostra alcuni esempi
            examples = list(collection.find(review_query).limit(3))
            for i, doc in enumerate(examples, 1):
                print(f"      📄 Esempio {i}: {doc.get('session_id')} - {doc.get('classification')} (conf: {doc.get('confidence')})")
        
    except Exception as e:
        print(f"❌ Errore analisi: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Funzione principale per analisi collection wopta tenant
    """
    print(f"🔍 ANALISI WOPTA TENANT COLLECTION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Carica configurazione
    config = load_config()
    if not config:
        return
    
    # 2. Connetti a MongoDB
    db, client = connect_mongodb(config)
    if db is None:
        return
    
    try:
        # 3. Analizza collection wopta tenant
        analyze_wopta_tenant_collection(db)
        
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
