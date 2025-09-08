#!/usr/bin/env python3
"""
Verifica specifica per documenti non-outlier in MongoDB Humanitas
usando la query: { "metadata.cluster_id": { "$ne": -1 } }

Autore: Valerio Bignardi
Data: 2025-09-08
"""

import yaml
from pymongo import MongoClient

tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"

def load_config():
    """Carica la configurazione MongoDB dal file config.yaml"""
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config['mongodb']

def check_non_outliers():
    """Verifica documenti con metadata.cluster_id != -1"""
    
    try:
        # Carica configurazione MongoDB
        mongo_config = load_config()
        print(f"🔌 CONNESSIONE MONGODB")
        print(f"   URL: {mongo_config['url']}")
        print(f"   Database: {mongo_config['database']}")
        
        # Connetti a MongoDB
        client = MongoClient(mongo_config['url'])
        db = client[mongo_config['database']]
        
        # Nome collezione per Humanitas
        collection_name = f"humanitas_{tenant_id}"
        collection = db[collection_name]
        
        print(f"📋 Collezione: {collection_name}")
        print("=" * 70)
        
        # Query specifica richiesta
        query = {"metadata.cluster_id": {"$ne": -1}}
        
        print(f"🔍 QUERY: {query}")
        
        # Conta documenti con cluster_id != -1
        non_outliers = collection.count_documents(query)
        print(f"📊 Documenti con metadata.cluster_id != -1: {non_outliers}")
        
        if non_outliers > 0:
            print(f"\n✅ TROVATI {non_outliers} DOCUMENTI NON-OUTLIER!")
            
            # Mostra alcuni esempi
            print(f"\n📄 ESEMPI DI DOCUMENTI NON-OUTLIER:")
            examples = collection.find(query).limit(5)
            
            for i, doc in enumerate(examples, 1):
                cluster_id = doc.get('metadata', {}).get('cluster_id', 'N/A')
                is_rep = doc.get('metadata', {}).get('is_representative', 'N/A')
                classification = doc.get('classification', 'N/A')
                message = doc.get('message', '')[:50] + "..." if len(doc.get('message', '')) > 50 else doc.get('message', '')
                
                print(f"   {i}. Cluster: {cluster_id}, Rappresentante: {is_rep}")
                print(f"      Classificazione: {classification}")
                print(f"      Messaggio: {message}")
                print()
            
            # Conta rappresentanti tra i non-outlier
            rep_query = {
                "metadata.cluster_id": {"$ne": -1},
                "metadata.is_representative": True
            }
            representatives = collection.count_documents(rep_query)
            print(f"🎯 Rappresentanti tra i non-outlier: {representatives}")
            
            # Distribuzione per cluster_id
            print(f"\n📊 DISTRIBUZIONE PER CLUSTER_ID:")
            pipeline = [
                {"$match": query},
                {"$group": {"_id": "$metadata.cluster_id", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            
            cluster_stats = list(collection.aggregate(pipeline))
            for stat in cluster_stats[:10]:  # Top 10
                cluster_id = stat['_id']
                count = stat['count']
                print(f"   Cluster {cluster_id}: {count} documenti")
        
        else:
            print(f"❌ NESSUN DOCUMENTO CON metadata.cluster_id != -1")
            print(f"   Tutti i documenti sono outlier (cluster_id = -1)")
            
            # Verifica per debug: mostra alcuni cluster_id effettivi
            print(f"\n🔍 DEBUG - CLUSTER_ID PRESENTI:")
            pipeline = [
                {"$group": {"_id": "$metadata.cluster_id", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            
            all_clusters = list(collection.aggregate(pipeline))
            for stat in all_clusters[:5]:  # Top 5
                cluster_id = stat['_id']
                count = stat['count']
                print(f"   Cluster {cluster_id}: {count} documenti")
        
        # Totale per verifica
        total = collection.count_documents({})
        print(f"\n📊 TOTALE DOCUMENTI: {total}")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'client' in locals():
            client.close()
            print(f"\n🔒 Connessione MongoDB chiusa")

if __name__ == "__main__":
    print("🔍 VERIFICA DOCUMENTI NON-OUTLIER HUMANITAS")
    print("=" * 50)
    check_non_outliers()
