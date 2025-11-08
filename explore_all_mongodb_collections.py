#!/usr/bin/env python3
"""
File: explore_all_mongodb_collections.py
Autore: Valerio Bignardi
Data creazione: 2025-01-31
Descrizione: Esplora tutte le collection MongoDB per trovare i dati

Scopo: Identificare dove sono salvati i documenti e i rappresentanti
"""

import sys
import os
import json
from datetime import datetime
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

def explore_all_collections(db):
    """
    Esplora tutte le collection nel database
    
    Args:
        db: Database MongoDB
    """
    print(f"\n{'='*80}")
    print(f"📂 TUTTE LE COLLECTION NEL DATABASE")
    print(f"{'='*80}")
    
    try:
        # Lista tutte le collection
        collections = db.list_collection_names()
        print(f"📊 TOTALE COLLECTION: {len(collections)}")
        
        if not collections:
            print("⚠️  Nessuna collection trovata nel database!")
            return
        
        print(f"📋 ELENCO COLLECTION:")
        for collection_name in sorted(collections):
            print(f"   • {collection_name}")
        
        print(f"\n{'='*80}")
        print(f"📊 ANALISI DETTAGLIATA COLLECTION")
        print(f"{'='*80}")
        
        # Analizza ogni collection
        for collection_name in sorted(collections):
            analyze_collection(db, collection_name)
            
    except Exception as e:
        print(f"❌ Errore esplorazione collection: {e}")

def analyze_collection(db, collection_name):
    """
    Analizza una singola collection
    
    Args:
        db: Database MongoDB
        collection_name: Nome della collection
    """
    try:
        collection = db[collection_name]
        count = collection.count_documents({})
        
        print(f"\n🗂️  COLLECTION: {collection_name}")
        print(f"   📊 Documenti: {count}")
        
        if count == 0:
            print(f"   ⚠️  Collection vuota")
            return
        
        # Ottieni documento di esempio
        sample_doc = collection.find_one({})
        if sample_doc:
            print(f"   📋 Campi principali:")
            for key in list(sample_doc.keys())[:10]:  # Mostra primi 10 campi
                if key != '_id':
                    value = sample_doc[key]
                    if isinstance(value, (str, int, float, bool)):
                        if isinstance(value, str) and len(value) > 50:
                            print(f"      • {key}: {type(value).__name__} = {value[:50]}...")
                        else:
                            print(f"      • {key}: {type(value).__name__} = {value}")
                    else:
                        print(f"      • {key}: {type(value).__name__}")
        
        # Cerca campi correlati alla review queue
        review_fields_found = []
        review_fields = [
            'review_status', 'requires_review', 'human_review_needed',
            'is_reviewed', 'review_pending', 'needs_human_review'
        ]
        
        for field in review_fields:
            field_count = collection.count_documents({field: {"$exists": True}})
            if field_count > 0:
                review_fields_found.append(f"{field}({field_count})")
        
        if review_fields_found:
            print(f"   🔍 Review fields: {', '.join(review_fields_found)}")
        
        # Cerca rappresentanti
        representative_fields_found = []
        representative_fields = [
            'is_representative', 'representative', 'cluster_representative',
            'is_cluster_representative'
        ]
        
        for field in representative_fields:
            field_count = collection.count_documents({field: True})
            if field_count > 0:
                representative_fields_found.append(f"{field}({field_count})")
        
        if representative_fields_found:
            print(f"   👑 Rappresentanti: {', '.join(representative_fields_found)}")
        
        # Cerca campi di clustering
        cluster_fields_found = []
        cluster_fields = ['cluster_id', 'cluster', 'cluster_label']
        
        for field in cluster_fields:
            field_count = collection.count_documents({field: {"$exists": True}})
            if field_count > 0:
                distinct_count = len(collection.distinct(field))
                cluster_fields_found.append(f"{field}({field_count}docs,{distinct_count}clusters)")
        
        if cluster_fields_found:
            print(f"   🎯 Clustering: {', '.join(cluster_fields_found)}")
        
        # Cerca documenti con tenant specifico
        tenant_fields = ['tenant_id', 'tenant_name', 'tenant_slug']
        tenant_info = []
        
        for field in tenant_fields:
            try:
                distinct_tenants = collection.distinct(field)
                if distinct_tenants:
                    tenant_info.append(f"{field}: {distinct_tenants}")
            except:
                pass
        
        if tenant_info:
            print(f"   🏢 Tenant info: {'; '.join(tenant_info)}")
        
    except Exception as e:
        print(f"   ❌ Errore analisi {collection_name}: {e}")

def search_review_candidates(db):
    """
    Cerca documenti candidati per review queue in tutte le collection
    
    Args:
        db: Database MongoDB
    """
    print(f"\n{'='*80}")
    print(f"🔍 RICERCA CANDIDATI REVIEW QUEUE")
    print(f"{'='*80}")
    
    collections = db.list_collection_names()
    
    for collection_name in collections:
        try:
            collection = db[collection_name]
            count = collection.count_documents({})
            
            if count == 0:
                continue
            
            print(f"\n🗂️  COLLECTION: {collection_name}")
            
            # Query per possibili candidati review
            candidates_queries = [
                ("Review pending", {"review_status": {"$in": ["pending", "requires_review"]}}),
                ("Requires review", {"requires_review": True}),
                ("Not reviewed", {"is_reviewed": False}),
                ("Low confidence", {"confidence": {"$lt": 0.7}}),
                ("Predicted as 'altro'", {"predicted_label": {"$in": ["altro", "ALTRO"]}}),
                ("Representatives", {"is_representative": True}),
                ("Has cluster_id", {"cluster_id": {"$exists": True}}),
                ("Recent docs (7 days)", {"created_at": {"$gte": datetime.now().replace(hour=0, minute=0, second=0) - 
                                                        __import__('datetime').timedelta(days=7)}})
            ]
            
            found_candidates = False
            for query_name, query in candidates_queries:
                try:
                    candidate_count = collection.count_documents(query)
                    if candidate_count > 0:
                        print(f"   ✅ {query_name}: {candidate_count} documenti")
                        found_candidates = True
                        
                        # Mostra esempio per query interessanti
                        if "review" in query_name.lower() or "representative" in query_name.lower():
                            example = collection.find_one(query)
                            if example:
                                print(f"      📄 Esempio ID: {example.get('_id')}")
                                relevant_fields = ['review_status', 'is_reviewed', 'requires_review', 
                                                 'is_representative', 'predicted_label', 'confidence']
                                for field in relevant_fields:
                                    if field in example:
                                        print(f"         {field}: {example[field]}")
                except Exception as e:
                    pass
            
            if not found_candidates:
                print(f"   ❌ Nessun candidato trovato")
                
        except Exception as e:
            print(f"   ❌ Errore ricerca in {collection_name}: {e}")

def main():
    """
    Funzione principale per esplorazione MongoDB
    """
    print(f"🔍 ESPLORAZIONE COMPLETA MONGODB - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Carica configurazione
    config = load_config()
    if not config:
        return
    
    # 2. Connetti a MongoDB
    db, client = connect_mongodb(config)
    if db is None:
        return
    
    try:
        # 3. Esplora tutte le collection
        explore_all_collections(db)
        
        # 4. Cerca candidati per review queue
        search_review_candidates(db)
        
        print(f"\n{'='*80}")
        print(f"✅ ESPLORAZIONE COMPLETATA")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ Errore durante esplorazione: {e}")
        import traceback
from config_loader import load_config
        traceback.print_exc()
    
    finally:
        if client:
            client.close()
            print(f"🔌 Connessione MongoDB chiusa")

if __name__ == "__main__":
    main()
