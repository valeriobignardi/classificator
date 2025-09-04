#!/usr/bin/env python3
"""
Debug script per analizzare il problema di sparizione degli elementi dalla Review Queue

Questo script analizza:
1. Come sono distribuiti i review_status nella collection
2. Cosa succede ai documenti dopo l'elaborazione
3. Chi/cosa cambia i review_status da "pending" ad altro

Autore: Valerio Bignardi
Data: 2025-09-04
"""

import yaml
from pymongo import MongoClient
from datetime import datetime, timedelta
import json

def analyze_review_status_distribution():
    """
    Analizza la distribuzione dei review_status nella collection humanitas
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
        humanitas_collection = None
        for coll_name in collections:
            if 'humanitas' in coll_name.lower():
                humanitas_collection = coll_name
                break
        
        if not humanitas_collection:
            print("❌ Nessuna collezione Humanitas trovata")
            return
        
        print(f"🎯 Analizzo collezione: {humanitas_collection}")
        collection = db[humanitas_collection]
        
        print("\n" + "="*80)
        print("📊 DISTRIBUZIONE REVIEW_STATUS")
        print("="*80)
        
        # Aggregazione per review_status
        pipeline = [
            {"$group": {"_id": "$review_status", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        status_distribution = list(collection.aggregate(pipeline))
        total_docs = sum(item['count'] for item in status_distribution)
        
        print(f"📈 Totale documenti: {total_docs}")
        print("\n📋 Distribuzione review_status:")
        for item in status_distribution:
            status = item['_id'] or 'null/undefined'
            count = item['count']
            percentage = (count / total_docs) * 100
            print(f"   {status:20} : {count:5} documenti ({percentage:5.1f}%)")
        
        print("\n" + "="*80)
        print("🔍 ANALISI DOCUMENTI 'pending' (DOVREBBERO ESSERE IN REVIEW QUEUE)")
        print("="*80)
        
        # Documenti con review_status = "pending"
        pending_docs = list(collection.find(
            {"review_status": "pending"},
            {"session_id": 1, "classification_type": 1, "classified_at": 1, "updated_at": 1, "metadata": 1}
        ).limit(10))
        
        print(f"✅ Trovati {len(pending_docs)} documenti con review_status='pending' (primi 10):")
        for doc in pending_docs:
            session_id = doc.get('session_id', 'N/A')[:12] + "..."
            class_type = doc.get('classification_type', 'N/A')
            classified_at = doc.get('classified_at', doc.get('updated_at', 'N/A'))
            metadata = doc.get('metadata', {})
            is_rep = metadata.get('is_representative', False)
            cluster_id = metadata.get('cluster_id', 'N/A')
            
            print(f"   📄 {session_id} | {class_type:12} | Rep: {is_rep} | Cluster: {cluster_id} | {classified_at}")
        
        print("\n" + "="*80)
        print("🔍 ANALISI DOCUMENTI 'auto_classified' (CHI LI HA CAMBIATI?)")
        print("="*80)
        
        # Documenti con review_status = "auto_classified" modificati di recente
        recent_auto_classified = list(collection.find(
            {
                "review_status": "auto_classified",
                "updated_at": {"$exists": True}
            },
            {"session_id": 1, "classification_type": 1, "classified_at": 1, "updated_at": 1, "classified_by": 1, "metadata": 1}
        ).sort("updated_at", -1).limit(15))
        
        print(f"🔄 Ultimi {len(recent_auto_classified)} documenti auto-classificati:")
        for doc in recent_auto_classified:
            session_id = doc.get('session_id', 'N/A')[:12] + "..."
            class_type = doc.get('classification_type', 'N/A')
            classified_by = doc.get('classified_by', 'N/A')
            updated_at = doc.get('updated_at', 'N/A')
            metadata = doc.get('metadata', {})
            cluster_id = metadata.get('cluster_id', 'N/A')
            
            print(f"   🤖 {session_id} | {class_type:12} | By: {classified_by:20} | Cluster: {cluster_id} | {updated_at}")
        
        print("\n" + "="*80)
        print("🔍 ANALISI PER CLASSIFICATION_TYPE")
        print("="*80)
        
        # Distribuzione per classification_type
        type_pipeline = [
            {"$group": {
                "_id": {"classification_type": "$classification_type", "review_status": "$review_status"}, 
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id.classification_type": 1, "_id.review_status": 1}}
        ]
        
        type_distribution = list(collection.aggregate(type_pipeline))
        
        current_type = None
        for item in type_distribution:
            class_type = item['_id']['classification_type'] or 'UNDEFINED'
            review_status = item['_id']['review_status'] or 'null'
            count = item['count']
            
            if class_type != current_type:
                print(f"\n📂 {class_type}:")
                current_type = class_type
            
            print(f"   └─ {review_status:15} : {count:4} documenti")
        
        print("\n" + "="*80)
        print("🕐 ANALISI TEMPORALE - COSA È SUCCESSO NELLE ULTIME ORE")
        print("="*80)
        
        # Documenti modificati nelle ultime 24 ore
        twenty_four_hours_ago = (datetime.now() - timedelta(hours=24)).isoformat()
        
        recent_changes = list(collection.find(
            {
                "$or": [
                    {"updated_at": {"$gte": twenty_four_hours_ago}},
                    {"classified_at": {"$gte": twenty_four_hours_ago}}
                ]
            },
            {
                "session_id": 1, "classification_type": 1, "review_status": 1, 
                "classified_at": 1, "updated_at": 1, "classified_by": 1, "metadata": 1
            }
        ).sort("updated_at", -1).limit(20))
        
        print(f"🕐 Ultimi {len(recent_changes)} documenti modificati nelle ultime 24h:")
        for doc in recent_changes:
            session_id = doc.get('session_id', 'N/A')[:10] + "..."
            class_type = doc.get('classification_type', 'N/A')[:8]
            review_status = doc.get('review_status', 'N/A')[:12]
            classified_by = doc.get('classified_by', 'N/A')[:15]
            updated_at = doc.get('updated_at', doc.get('classified_at', 'N/A'))
            metadata = doc.get('metadata', {})
            cluster_id = metadata.get('cluster_id', 'N/A')
            
            # Timestamp più leggibile
            try:
                if updated_at and updated_at != 'N/A':
                    dt = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                    time_str = dt.strftime("%m-%d %H:%M")
                else:
                    time_str = 'N/A'
            except:
                time_str = str(updated_at)[:16] if updated_at != 'N/A' else 'N/A'
                
            print(f"   ⏰ {session_id} | {class_type:8} | {review_status:12} | {classified_by:15} | C{cluster_id} | {time_str}")
        
        print("\n" + "="*80)
        print("🎯 RIASSUNTO PROBLEMA")
        print("="*80)
        
        pending_count = collection.count_documents({"review_status": "pending"})
        auto_classified_count = collection.count_documents({"review_status": "auto_classified"})
        
        print(f"📊 SITUAZIONE ATTUALE:")
        print(f"   🟡 Pending (visibili in Review Queue): {pending_count}")
        print(f"   🟢 Auto-classified (spariti dalla queue): {auto_classified_count}")
        print(f"   📈 Totale: {total_docs}")
        
        if auto_classified_count > pending_count:
            print(f"\n❗ PROBLEMA IDENTIFICATO:")
            print(f"   La maggior parte dei documenti è stata auto-classificata ({auto_classified_count})")
            print(f"   mentre solo {pending_count} sono ancora 'pending' e visibili nella Review Queue.")
            print(f"   Questo spiega perché la queue sembra vuota!")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_review_status_distribution()
