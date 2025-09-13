#!/usr/bin/env python3
"""
🔍 ANALISI HUMANITAS: Debug review queue vuota dopo training supervisionato

Analizza cosa succede nella collezione humanitas_015007d9-d413-11ef-86a5-96000228e7fe:
1. Conta rappresentanti e outlier
2. Verifica confidence e review_status 
3. Analizza documenti salvati durante training supervisionato
4. Confronta con filtri React

Autore: Valerio Bignardi
Data: 2025-01-13
"""

import os
import sys
import pymongo
import yaml
from datetime import datetime
from collections import Counter

# Aggiungi il path del progetto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_config():
    """Carica configurazione dal file config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def analyze_humanitas_collection():
    """
    Analizza la collezione Humanitas per debug review queue
    """
    print("🔍 ANALISI COLLEZIONE HUMANITAS - DEBUG REVIEW QUEUE")
    print("=" * 80)
    
    # Carica config e connessione MongoDB
    config = load_config()
    client = pymongo.MongoClient(config['mongodb']['url'])
    db = client[config['mongodb']['database']]
    
    # Nome collezione Humanitas
    collection_name = "humanitas_015007d9-d413-11ef-86a5-96000228e7fe"
    collection = db[collection_name]
    
    print(f"📊 Analizzando collezione: {collection_name}")
    
    # STEP 1: Statistiche generali
    total_docs = collection.count_documents({})
    print(f"\n📈 STATISTICHE GENERALI:")
    print(f"   📝 Totale documenti: {total_docs}")
    
    if total_docs == 0:
        print("❌ COLLEZIONE VUOTA! Nessun documento trovato")
        client.close()
        return False
    
    # STEP 2: Analisi review_status
    print(f"\n📋 ANALISI REVIEW STATUS:")
    status_counts = {}
    for status in ["pending", "auto_classified", "completed", "rejected"]:
        count = collection.count_documents({"review_status": status})
        status_counts[status] = count
        icon = "⏳" if status == "pending" else "🤖" if status == "auto_classified" else "✅" if status == "completed" else "❌"
        print(f"   {icon} {status}: {count}")
    
    # PROBLEMA CRITICO: Nessun pending
    if status_counts.get("pending", 0) == 0:
        print("❌ PROBLEMA CRITICO: Nessun documento con review_status='pending'!")
        print("💡 Questo spiega perché la review queue è vuota")
    
    # STEP 3: Analisi rappresentanti
    print(f"\n🎯 ANALISI RAPPRESENTANTI:")
    
    # Cerca rappresentanti usando diversi campi possibili
    repr_fields_to_check = [
        "metadata.is_representative",
        "is_representative", 
        "metadata.selection_reason"
    ]
    
    total_representatives = 0
    for field in repr_fields_to_check:
        if "selection_reason" in field:
            count = collection.count_documents({field: {"$regex": "rappresentante|representative"}})
        else:
            count = collection.count_documents({field: True})
        
        if count > 0:
            print(f"   🎯 {field}: {count}")
            total_representatives = max(total_representatives, count)
    
    print(f"   📊 Totale rappresentanti stimati: {total_representatives}")
    
    # STEP 4: Analisi outlier
    print(f"\n🔴 ANALISI OUTLIER:")
    
    outlier_fields_to_check = [
        "metadata.is_outlier",
        "is_outlier"
    ]
    
    total_outliers = 0
    for field in outlier_fields_to_check:
        count = collection.count_documents({field: True})
        if count > 0:
            print(f"   🔴 {field}: {count}")
            total_outliers = max(total_outliers, count)
    
    print(f"   📊 Totale outlier stimati: {total_outliers}")
    
    # STEP 5: Analisi confidence
    print(f"\n📊 ANALISI CONFIDENCE:")
    
    # Documenti con confidence
    docs_with_conf = collection.count_documents({"confidence": {"$exists": True}})
    print(f"   📝 Documenti con confidence: {docs_with_conf}")
    
    if docs_with_conf > 0:
        # Distribuzione confidence
        pipeline = [
            {"$match": {"confidence": {"$exists": True}}},
            {"$group": {
                "_id": None,
                "min_conf": {"$min": "$confidence"},
                "max_conf": {"$max": "$confidence"},
                "avg_conf": {"$avg": "$confidence"}
            }}
        ]
        
        stats = list(collection.aggregate(pipeline))
        if stats:
            stat = stats[0]
            print(f"   📉 Min confidence: {stat['min_conf']:.3f}")
            print(f"   📈 Max confidence: {stat['max_conf']:.3f}")
            print(f"   📊 Avg confidence: {stat['avg_conf']:.3f}")
        
        # Confidence sotto soglie critiche
        low_conf_085 = collection.count_documents({"confidence": {"$lt": 0.85}})
        low_conf_060 = collection.count_documents({"confidence": {"$lt": 0.60}})
        low_conf_095 = collection.count_documents({"confidence": {"$lt": 0.95}})
        
        print(f"   ⚠️ Confidence < 0.85 (soglia rappresentanti): {low_conf_085}")
        print(f"   ⚠️ Confidence < 0.60 (soglia outlier): {low_conf_060}")  
        print(f"   ⚠️ Confidence < 0.95 (soglia generica): {low_conf_095}")
        
        if low_conf_095 > 0:
            print("💡 CI SONO DOCUMENTI CON CONFIDENCE BASSA CHE DOVREBBERO ESSERE PENDING!")
    
    # STEP 6: Analisi documenti candidati per review queue
    print(f"\n🎯 CANDIDATI REVIEW QUEUE:")
    
    # Rappresentanti con confidence bassa
    repr_low_conf = collection.count_documents({
        "$and": [
            {"$or": [
                {"metadata.is_representative": True},
                {"is_representative": True}
            ]},
            {"confidence": {"$lt": 0.85}}
        ]
    })
    
    # Outlier con confidence bassa  
    outlier_low_conf = collection.count_documents({
        "$and": [
            {"$or": [
                {"metadata.is_outlier": True},
                {"is_outlier": True}
            ]},
            {"confidence": {"$lt": 0.60}}
        ]
    })
    
    # Documenti generici con confidence bassa
    generic_low_conf = collection.count_documents({
        "$and": [
            {"$or": [
                {"metadata.is_representative": {"$ne": True}},
                {"is_representative": {"$ne": True}}
            ]},
            {"$or": [
                {"metadata.is_outlier": {"$ne": True}},
                {"is_outlier": {"$ne": True}}
            ]},
            {"confidence": {"$lt": 0.95}}
        ]
    })
    
    print(f"   🎯 Rappresentanti confidence < 0.85: {repr_low_conf}")
    print(f"   🔴 Outlier confidence < 0.60: {outlier_low_conf}")
    print(f"   📝 Generici confidence < 0.95: {generic_low_conf}")
    
    total_should_be_pending = repr_low_conf + outlier_low_conf + generic_low_conf
    print(f"   ⏳ TOTALE CHE DOVREBBE ESSERE PENDING: {total_should_be_pending}")
    
    if total_should_be_pending > 0 and status_counts.get("pending", 0) == 0:
        print("🚨 CONFERMATO IL BUG: Ci sono documenti che dovrebbero essere pending ma sono auto_classified!")
    
    # STEP 7: Campioni di documenti problematici
    print(f"\n🔍 CAMPIONI DOCUMENTI PROBLEMATICI:")
    
    # Trova alcuni esempi di documenti che dovrebbero essere pending
    problematic_docs = list(collection.find({
        "$and": [
            {"confidence": {"$lt": 0.85}},
            {"review_status": "auto_classified"}
        ]
    }).limit(3))
    
    for i, doc in enumerate(problematic_docs, 1):
        print(f"\n   📄 Esempio {i}: {doc.get('session_id', 'unknown')}")
        print(f"      📊 Confidence: {doc.get('confidence', 'N/A')}")
        print(f"      🏷️  Label: {doc.get('predicted_label', 'N/A')}")
        print(f"      📋 Review Status: {doc.get('review_status', 'N/A')}")
        print(f"      🎯 Is Representative: {doc.get('metadata', {}).get('is_representative', doc.get('is_representative', 'N/A'))}")
        print(f"      🔴 Is Outlier: {doc.get('metadata', {}).get('is_outlier', doc.get('is_outlier', 'N/A'))}")
        print(f"      🕐 Classified By: {doc.get('classified_by', 'N/A')}")
        print(f"      📝 Notes: {doc.get('notes', 'N/A')[:100]}...")
    
    # STEP 8: Analisi classified_by per capire la fonte
    print(f"\n🔧 ANALISI FONTE CLASSIFICAZIONE:")
    
    pipeline_classified_by = [
        {"$group": {"_id": "$classified_by", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    
    classified_by_stats = list(collection.aggregate(pipeline_classified_by))
    for stat in classified_by_stats:
        print(f"   🔧 {stat['_id']}: {stat['count']}")
    
    client.close()
    
    # STEP 9: Riassunto e diagnosi
    print(f"\n" + "=" * 80)
    print("🩺 DIAGNOSI PROBLEMA:")
    
    if total_should_be_pending > 0 and status_counts.get("pending", 0) == 0:
        print("❌ CONFERMATO: Il bug del fix review queue persiste!")
        print("💡 CAUSE POSSIBILI:")
        print("   1. Il fix non è stato applicato a questa collezione")
        print("   2. Il training supervisionato non usa la logica DocumentoProcessing")
        print("   3. I documenti sono stati salvati prima del fix")
        print("   4. C'è un altro punto nel codice che sovrascrive needs_review")
        
        print(f"\n🎯 AZIONI IMMEDIATE:")
        print("   1. Verificare quando sono stati salvati questi documenti")
        print("   2. Controllare se il training supervisionato usa il fix")
        print("   3. Verificare filtri React per review queue")
        
        return False
    else:
        print("✅ Situazione normale o collezione senza documenti a confidence bassa")
        return True

def main():
    """
    Esegue l'analisi completa della collezione Humanitas
    """
    try:
        success = analyze_humanitas_collection()
        
        if not success:
            print("\n🚨 PROBLEMA CONFERMATO! Review queue vuota per bug nel training supervisionato")
        else:
            print("\n✅ Analisi completata")
        
        return success
        
    except Exception as e:
        print(f"❌ ERRORE DURANTE L'ANALISI: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
