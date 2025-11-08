#!/usr/bin/env python3
"""
============================================================================
Esplorazione MongoDB - Collezioni e Contenuti
============================================================================

Autore: Valerio Bignardi
Data creazione: 2025-01-31

Descrizione:
    Script per esplorare le collezioni MongoDB e trovare dove sono i dati.

============================================================================
"""

import os
import yaml
import pymongo
from pymongo import MongoClient
from typing import Dict, List, Any
import json
from config_loader import load_config

def load_config() -> Dict[str, Any]:
    """Carica configurazione da config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    return load_config()

def explore_mongodb():
    """Esplora MongoDB per trovare dove sono i dati"""
    try:
        # Carica configurazione
        config = load_config()
        mongodb_config = config.get('mongodb', {})
        url = mongodb_config.get('url', 'mongodb://localhost:27017')
        database_name = mongodb_config.get('database', 'classificazioni')
        
        print(f"🔗 Connessione a MongoDB: {url}")
        client = MongoClient(url)
        
        # Lista tutti i database
        print(f"\n📊 DATABASE DISPONIBILI:")
        dbs = client.list_database_names()
        for db_name in dbs:
            if not db_name.startswith('admin') and not db_name.startswith('config') and not db_name.startswith('local'):
                print(f"   📁 {db_name}")
        
        # Esplora database principale
        db = client[database_name]
        print(f"\n📊 COLLEZIONI IN '{database_name}':")
        collections = db.list_collection_names()
        
        if not collections:
            print(f"   ❌ Nessuna collezione trovata in '{database_name}'")
            
            # Prova altri database comuni
            common_dbs = ['humanitas', 'classification', 'sessions', 'reviews']
            for db_name in common_dbs:
                if db_name in dbs:
                    print(f"\n🔍 Controllo database '{db_name}':")
                    test_db = client[db_name]
                    test_collections = test_db.list_collection_names()
                    for coll in test_collections:
                        count = test_db[coll].count_documents({})
                        print(f"   📄 {coll}: {count} documenti")
        else:
            for collection_name in collections:
                try:
                    collection = db[collection_name]
                    count = collection.count_documents({})
                    print(f"   📄 {collection_name}: {count} documenti")
                    
                    # Se la collezione ha documenti, mostra un esempio
                    if count > 0:
                        sample = collection.find_one()
                        if sample:
                            sample.pop('_id', None)
                            print(f"      📋 Esempio: {list(sample.keys())[:5]}...")
                            
                            # Cerca il nostro caso specifico
                            if 'session_id' in sample:
                                case_found = collection.find_one({"session_id": "68bcec2b34ea2bca071d45be"})
                                if case_found:
                                    print(f"      ✅ CASO TROVATO in {collection_name}!")
                            
                            # Cerca altri campi che potrebbero contenere l'ID
                            for field in ['_id', 'id', 'session_id', 'review_id']:
                                if field in sample:
                                    case_found = collection.find_one({field: "68bcec2b34ea2bca071d45be"})
                                    if case_found:
                                        print(f"      ✅ CASO TROVATO in {collection_name} campo {field}!")
                
                except Exception as e:
                    print(f"      ❌ Errore: {e}")
        
        # Cerca in tutte le collezioni per il caso specifico
        print(f"\n🔍 RICERCA GLOBALE CASO: 68bcec2b34ea2bca071d45be")
        case_id = "68bcec2b34ea2bca071d45be"
        
        for collection_name in collections:
            try:
                collection = db[collection_name]
                
                # Cerca in diversi campi possibili
                search_fields = ['session_id', '_id', 'id', 'review_id', 'case_id']
                for field in search_fields:
                    result = collection.find_one({field: case_id})
                    if result:
                        print(f"   ✅ Trovato in {collection_name}.{field}")
                        result.pop('_id', None)
                        print(f"      📄 Dati: {json.dumps(result, indent=2, default=str)[:500]}...")
                        break
                
                # Ricerca anche come substring
                text_search = collection.find_one({"$text": {"$search": case_id[:10]}}) if collection.count_documents({}) < 1000 else None
                if text_search:
                    print(f"   ✅ Trovato (text search) in {collection_name}")
                    
            except Exception as e:
                pass  # Ignora errori di ricerca
        
        # Controlla anche file system per modelli
        print(f"\n📁 CONTROLLO FILE SYSTEM:")
        
        # Controlla cartella models
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        print(f"   📂 models/: {os.path.exists(models_dir)}")
        if os.path.exists(models_dir):
            files = os.listdir(models_dir)
            print(f"      File: {files}")
        
        # Controlla altre cartelle comuni
        common_dirs = ['finetuned_models', 'backup', 'debug_logs', 'semantic_cache']
        for dir_name in common_dirs:
            dir_path = os.path.join(os.path.dirname(__file__), dir_name)
            if os.path.exists(dir_path):
                files = os.listdir(dir_path)
                print(f"   📂 {dir_name}/: {len(files)} file")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Errore: {e}")

if __name__ == "__main__":
    explore_mongodb()
