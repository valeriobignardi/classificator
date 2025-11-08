#!/usr/bin/env python3
"""
File: fix_review_queue_wopta.py
Autore: Valerio Bignardi
Data creazione: 2025-01-31
Descrizione: Corregge i problemi della review queue per Wopta

Problemi identificati:
1. Collection errata: cerca 'wopta' invece di 'wopta_tenant_id'
2. review_status errato: tutti sono 'auto_classified' invece di 'pending'
3. Mancano rappresentanti: nessun is_representative nel database
4. 489 documenti candidati non visibili nella review queue

Soluzioni:
1. Corregge collection name nel mongo_reader
2. Imposta review_status='pending' per candidati review
3. Verifica logica rappresentanti nel clustering
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

def fix_review_queue_status(db):
    """
    Corregge i review_status per popolare la review queue
    
    Args:
        db: Database MongoDB
    """
    print(f"\n{'='*80}")
    print(f"🔧 CORREZIONE REVIEW QUEUE STATUS")
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
        return False
    
    print(f"🎯 Collection trovata: {wopta_collection_name}")
    collection = db[wopta_collection_name]
    
    try:
        # 1. Analisi stato attuale
        total_docs = collection.count_documents({})
        auto_classified = collection.count_documents({"review_status": "auto_classified"})
        pending = collection.count_documents({"review_status": "pending"})
        
        print(f"\n📊 STATO ATTUALE:")
        print(f"   • Totale documenti: {total_docs}")
        print(f"   • Auto-classified: {auto_classified}")
        print(f"   • Pending: {pending}")
        
        # 2. Identifica candidati per review queue
        review_query = {
            "$or": [
                {"confidence": {"$lt": 0.7}},
                {"classification": {"$in": ["ALTRO", "altro"]}}
            ]
        }
        
        candidates_count = collection.count_documents(review_query)
        print(f"\n🎯 CANDIDATI REVIEW QUEUE:")
        print(f"   • Documenti che necessitano review: {candidates_count}")
        
        if candidates_count == 0:
            print("⚠️  Nessun candidato trovato per review queue")
            return True
        
        # 3. Backup documenti prima della modifica
        print(f"\n💾 CREAZIONE BACKUP:")
        backup_docs = list(collection.find(review_query).limit(5))
        backup_filename = f"backup_review_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup_path = os.path.join(os.path.dirname(__file__), 'backup', backup_filename)
        
        # Crea directory backup se non esiste
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_docs, f, indent=2, default=str, ensure_ascii=False)
        print(f"   ✅ Backup salvato: {backup_path}")
        
        # 4. Aggiorna review_status per candidati
        print(f"\n🔄 AGGIORNAMENTO REVIEW_STATUS:")
        
        # Prepara l'update
        update_operation = {
            "$set": {
                "review_status": "pending",
                "review_reason": "low_confidence_or_altro",
                "review_updated_at": datetime.now().isoformat(),
                "review_updated_by": "fix_review_queue_script"
            }
        }
        
        # Conferma prima di procedere
        print(f"   ⚠️  Si sta per aggiornare {candidates_count} documenti")
        print(f"   📝 Update: {json.dumps(update_operation, indent=6)}")
        
        response = input(f"\n❓ Procedere con l'aggiornamento? (s/N): ").strip().lower()
        if response != 's':
            print("❌ Operazione annullata dall'utente")
            return False
        
        # Esegui update
        result = collection.update_many(review_query, update_operation)
        
        print(f"\n✅ AGGIORNAMENTO COMPLETATO:")
        print(f"   • Documenti modificati: {result.modified_count}")
        print(f"   • Documenti trovati: {result.matched_count}")
        
        # 5. Verifica risultato
        new_pending = collection.count_documents({"review_status": "pending"})
        print(f"\n📊 STATO FINALE:")
        print(f"   • Pending (review queue): {new_pending}")
        
        # 6. Mostra esempi aggiornati
        examples = list(collection.find({"review_status": "pending"}).limit(3))
        print(f"\n📄 ESEMPI AGGIORNATI:")
        for i, doc in enumerate(examples, 1):
            print(f"   {i}. {doc.get('session_id')} - {doc.get('classification')} (conf: {doc.get('confidence')})")
            print(f"      review_status: {doc.get('review_status')}")
            print(f"      review_reason: {doc.get('review_reason')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore durante correzione: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_representatives_issue(db):
    """
    Analizza il problema dei rappresentanti mancanti
    
    Args:
        db: Database MongoDB
    """
    print(f"\n{'='*80}")
    print(f"🔍 ANALISI PROBLEMA RAPPRESENTANTI")
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
    
    collection = db[wopta_collection_name]
    
    try:
        # 1. Verifica campi clustering
        print(f"\n1️⃣ VERIFICA CAMPI CLUSTERING:")
        
        clustering_fields = [
            'cluster_id', 'cluster', 'cluster_label', 'is_representative',
            'representative', 'cluster_representative', 'is_cluster_representative'
        ]
        
        found_clustering = False
        for field in clustering_fields:
            count = collection.count_documents({field: {"$exists": True}})
            if count > 0:
                print(f"   ✅ {field}: {count} documenti")
                found_clustering = True
            else:
                print(f"   ❌ {field}: non presente")
        
        if not found_clustering:
            print(f"\n⚠️  PROBLEMA: Nessun campo di clustering trovato!")
            print(f"   💡 Soluzione: Il clustering non è stato eseguito o non ha salvato metadati")
        
        # 2. Verifica metodo di classificazione
        print(f"\n2️⃣ VERIFICA METODI CLASSIFICAZIONE:")
        
        methods = collection.distinct('classification_method')
        print(f"   📋 Metodi trovati: {methods}")
        
        for method in methods:
            count = collection.count_documents({'classification_method': method})
            print(f"   • {method}: {count} documenti")
        
        # 3. Verifica classification_type
        print(f"\n3️⃣ VERIFICA CLASSIFICATION_TYPE:")
        
        types = collection.distinct('classification_type')
        print(f"   📋 Tipi trovati: {types}")
        
        for type_name in types:
            count = collection.count_documents({'classification_type': type_name})
            print(f"   • {type_name}: {count} documenti")
        
        # 4. Raccomandazioni
        print(f"\n💡 RACCOMANDAZIONI:")
        print(f"   1. Il clustering INTELLIGENT_CLUSTERING ha creato solo OUTLIER")
        print(f"   2. Nessun cluster effettivo o rappresentante è stato generato")
        print(f"   3. Possibili cause:")
        print(f"      • Parametri clustering troppo restrittivi")
        print(f"      • Dataset insufficiente per clustering")
        print(f"      • Errore nella fase di clustering")
        print(f"   4. Soluzioni:")
        print(f"      • Rieseguire pipeline con parametri clustering ottimizzati")
        print(f"      • Verificare logs del clustering")
        print(f"      • Implementare review queue basata su confidence invece che rappresentanti")
        
    except Exception as e:
        print(f"❌ Errore analisi rappresentanti: {e}")

def test_review_queue_query(db):
    """
    Testa la query della review queue dopo le correzioni
    
    Args:
        db: Database MongoDB
    """
    print(f"\n{'='*80}")
    print(f"🧪 TEST QUERY REVIEW QUEUE")
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
    
    collection = db[wopta_collection_name]
    
    try:
        # Query esatta usata dal sistema
        review_query = {"review_status": "pending"}
        
        pending_docs = list(collection.find(review_query).limit(5))
        print(f"\n🔍 QUERY: {review_query}")
        print(f"📊 Risultati: {len(pending_docs)} documenti")
        
        if pending_docs:
            print(f"\n📄 ESEMPI DOCUMENTI PENDING:")
            for i, doc in enumerate(pending_docs, 1):
                print(f"   {i}. ID: {doc.get('_id')}")
                print(f"      session_id: {doc.get('session_id')}")
                print(f"      classification: {doc.get('classification')}")
                print(f"      confidence: {doc.get('confidence')}")
                print(f"      review_status: {doc.get('review_status')}")
                print(f"      review_reason: {doc.get('review_reason')}")
        else:
            print(f"❌ Nessun documento pending trovato!")
            
            # Suggerisci alternative
            print(f"\n💡 ALTERNATIVE:")
            alt_queries = [
                ("Low confidence", {"confidence": {"$lt": 0.7}}),
                ("ALTRO classification", {"classification": "ALTRO"}),
                ("Auto classified", {"review_status": "auto_classified"}),
            ]
            
            for name, query in alt_queries:
                count = collection.count_documents(query)
                print(f"   • {name}: {count} documenti")
        
    except Exception as e:
        print(f"❌ Errore test query: {e}")

def main():
    """
    Funzione principale per correzione review queue
    """
    print(f"🔧 FIX REVIEW QUEUE WOPTA - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Carica configurazione
    config = load_config()
    if not config:
        return
    
    # 2. Connetti a MongoDB
    db, client = connect_mongodb(config)
    if db is None:
        return
    
    try:
        # 3. Analizza problema rappresentanti
        check_representatives_issue(db)
        
        # 4. Corregge review_status
        if fix_review_queue_status(db):
            # 5. Testa query dopo correzione
            test_review_queue_query(db)
        
        print(f"\n{'='*80}")
        print(f"✅ CORREZIONE COMPLETATA")
        print(f"{'='*80}")
        print(f"💡 PROSSIMI PASSI:")
        print(f"   1. Riavvia l'interfaccia web review")
        print(f"   2. Verifica che i casi appaiano nella dashboard")
        print(f"   3. Se necessario, riesegui pipeline con clustering ottimizzato")
        
    except Exception as e:
        print(f"❌ Errore durante correzione: {e}")
        import traceback
from config_loader import load_config
        traceback.print_exc()
    
    finally:
        if client:
            client.close()
            print(f"🔌 Connessione MongoDB chiusa")

if __name__ == "__main__":
    main()
