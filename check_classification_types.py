#!/usr/bin/env python3
"""
Script per verificare i problemi di classification_type
"""

import yaml
from pymongo import MongoClient

def check_classification_types():
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        mongo_url = config.get('mongodb', {}).get('url', 'mongodb://localhost:27017')
        mongo_db_name = config.get('mongodb', {}).get('database', 'classificazioni')
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        db = client[mongo_db_name]
        
        collections = db.list_collection_names()
        collection_name = [c for c in collections if 'humanitas' in c.lower()][0]
        collection = db[collection_name]
        
        print("📊 VERIFICA CLASSIFICATION_TYPE:")
        
        # Conta per classification_type
        types = ["RAPPRESENTANTE", "OUTLIER", "PROPAGATO", "NORMALE", "CLUSTER_MEMBER", None]
        for t in types:
            count = collection.count_documents({"classification_type": t})
            if count > 0:
                print(f"   {str(t):15}: {count:4} documenti")
        
        print("\n📊 VERIFICA OUTLIERS DETAILS:")
        outliers = list(collection.find(
            {"classification_type": "OUTLIER"}, 
            {"session_id": 1, "metadata": 1, "classification": 1}
        ).limit(10))
        
        for doc in outliers:
            metadata = doc.get('metadata', {})
            cluster_id = metadata.get('cluster_id', 'N/A')
            session_id = doc.get('session_id', 'N/A')[:12]
            classification = doc.get('classification', 'N/A')[:20]
            print(f"   {session_id}... | cluster: {cluster_id} | class: {classification}")
        
        print(f"\n📊 VERIFICA RAPPRESENTANTI:")
        representatives = collection.count_documents({"classification_type": "RAPPRESENTANTE"})
        print(f"   Rappresentanti: {representatives}")
        
        print(f"\n📊 VERIFICA PROPAGATI:")
        propagated = collection.count_documents({"classification_type": "PROPAGATO"})
        print(f"   Propagati: {propagated}")
        
        # Verifica metadata null
        print(f"\n📊 VERIFICA METADATA NULL:")
        null_metadata = collection.count_documents({"metadata": None})
        empty_metadata = collection.count_documents({"metadata": {}})
        print(f"   Metadata null: {null_metadata}")
        print(f"   Metadata vuoti: {empty_metadata}")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_classification_types()
