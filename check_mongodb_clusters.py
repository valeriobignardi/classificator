#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verifica diretta MongoDB per cluster_id nelle sessioni

Autore: AI Assistant
Data: 2025-01-27
Scopo: Controllare direttamente MongoDB per sessioni con cluster_id
"""

import yaml
from pymongo import MongoClient

def check_mongodb_cluster_data():
    """
    Controlla direttamente MongoDB per verificare presenza cluster_id
    """
    print("🔍 VERIFICA DIRETTA MONGODB - Cluster ID")
    print("="*60)
    
    try:
        # Carica configurazione
        with open('/home/ubuntu/classificatore/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Prova diverse configurazioni MongoDB possibili
        mongo_configs = [
            # Config da database.mongodb se esiste
            config.get('database', {}).get('mongodb', {}),
            # Config diretta da mongodb se esiste
            config.get('mongodb', {}),
            # Fallback default
            {'uri': 'mongodb://localhost:27017', 'database': 'classificazioni'}
        ]
        
        client = None
        db = None
        
        for i, mongo_config in enumerate(mongo_configs):
            if not mongo_config:
                continue
                
            print(f"\n🔗 Tentativo {i+1}: {mongo_config}")
            
            try:
                mongo_uri = mongo_config.get('uri', 'mongodb://localhost:27017')
                database_name = mongo_config.get('database', 'classificazioni')
                
                client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
                client.server_info()  # Test connessione
                db = client[database_name]
                
                print(f"✅ Connessione riuscita!")
                print(f"   📊 Database: {database_name}")
                break
                
            except Exception as e:
                print(f"❌ Tentativo {i+1} fallito: {e}")
                if client:
                    client.close()
                client = None
                db = None
        
        if db is None:
            print("❌ Impossibile connettersi a MongoDB con nessuna configurazione")
            return
        
        # Verifica collezioni disponibili
        collections = db.list_collection_names()
        print(f"\n📚 Collezioni disponibili: {collections}")
        
        # Cerca collezioni che potrebbero contenere sessioni
        session_collections = [name for name in collections if 'session' in name.lower() or 'humanitas' in name.lower()]
        
        if not session_collections:
            # Prova con 'sessions' standard
            session_collections = ['sessions']
        
        print(f"🎯 Collezioni sessioni da analizzare: {session_collections}")
        
        total_sessions_with_cluster = 0
        total_sessions = 0
        
        for collection_name in session_collections:
            try:
                collection = db[collection_name]
                
                # Conta totale documenti nella collezione
                total_docs = collection.count_documents({})
                
                if total_docs == 0:
                    print(f"\n📂 Collezione '{collection_name}': vuota")
                    continue
                
                print(f"\n📂 Collezione '{collection_name}': {total_docs} documenti")
                
                # Filtra per tenant humanitas se presente
                humanitas_filter = [
                    {'tenant': 'humanitas'},
                    {'client_name': 'humanitas'},
                    {'tenant_slug': 'humanitas'}
                ]
                
                humanitas_sessions = 0
                for filter_query in humanitas_filter:
                    count = collection.count_documents(filter_query)
                    if count > 0:
                        humanitas_sessions = count
                        print(f"   🏥 Sessioni humanitas ({list(filter_query.keys())[0]}): {count}")
                        break
                
                if humanitas_sessions == 0:
                    print(f"   ⚠️ Nessuna sessione humanitas trovata, analizzo tutti i documenti")
                    analysis_filter = {}
                    analysis_total = total_docs
                else:
                    # Usa il filtro che ha funzionato
                    for filter_query in humanitas_filter:
                        if collection.count_documents(filter_query) > 0:
                            analysis_filter = filter_query
                            analysis_total = humanitas_sessions
                            break
                
                # Cerca cluster_id nei metadati
                cluster_filter = {**analysis_filter, 'metadata.cluster_id': {'$exists': True, '$ne': None}}
                sessions_with_cluster = collection.count_documents(cluster_filter)
                
                print(f"   📊 Sessioni con cluster_id: {sessions_with_cluster}/{analysis_total}")
                
                # Campione di cluster_id
                if sessions_with_cluster > 0:
                    sample_clusters = list(collection.aggregate([
                        {'$match': cluster_filter},
                        {'$group': {'_id': '$metadata.cluster_id'}},
                        {'$limit': 10}
                    ]))
                    
                    cluster_ids = [doc['_id'] for doc in sample_clusters]
                    print(f"   🔢 Cluster IDs campione: {sorted(cluster_ids)}")
                    
                    # Esempio completo
                    sample_session = collection.find_one(cluster_filter)
                    if sample_session:
                        print(f"\n📋 ESEMPIO SESSIONE CON CLUSTER:")
                        print(f"   Session ID: {sample_session.get('session_id', 'N/A')[:20]}...")
                        print(f"   Cluster ID: {sample_session.get('metadata', {}).get('cluster_id')}")
                        print(f"   Classification: {sample_session.get('classification', 'N/A')}")
                        print(f"   Confidence: {sample_session.get('confidence', 'N/A')}")
                
                total_sessions += analysis_total
                total_sessions_with_cluster += sessions_with_cluster
                
            except Exception as e:
                print(f"❌ Errore analisi collezione '{collection_name}': {e}")
        
        # Statistiche finali
        print(f"\n📈 RIEPILOGO MONGODB:")
        print(f"   Sessioni totali analizzate: {total_sessions}")
        print(f"   Sessioni con cluster_id: {total_sessions_with_cluster}")
        
        if total_sessions > 0:
            coverage = (total_sessions_with_cluster / total_sessions) * 100
            print(f"   📊 Coverage cluster: {coverage:.1f}%")
            
            if total_sessions_with_cluster == 0:
                print(f"\n❌ PROBLEMA IDENTIFICATO:")
                print(f"   Le sessioni NON hanno cluster_id nei metadati MongoDB")
                print(f"   Questo spiega perché non appaiono in 'Tutte le Sessioni'")
                print(f"\n💡 SOLUZIONI:")
                print(f"   1. Eseguire clustering su sessioni esistenti")
                print(f"   2. Aggiornare metadati MongoDB con cluster_id")
                print(f"   3. Re-processare le classificazioni con clustering")
            else:
                print(f"\n✅ Alcune sessioni hanno cluster_id - API dovrebbe funzionare!")
        
        if client:
            client.close()
            
    except Exception as e:
        print(f"❌ Errore generale: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🚀 DIAGNOSI MONGODB - Cluster ID")
    print("="*60)
    check_mongodb_cluster_data()

if __name__ == "__main__":
    main()