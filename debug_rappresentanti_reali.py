#!/usr/bin/env python3
"""
File: debug_rappresentanti_reali.py
Autore: Valerio Bignardi
Data creazione: 2025-01-31
Descrizione: Debug dei rappresentanti reali in MongoDB

Scopo: 
- Analizzare i documenti reali della collection Wopta
- Verificare i metadati cluster esistenti
- Identificare la discrepanza tra clustering e salvataggio
"""

import sys
import os
import json
from datetime import datetime
from pymongo import MongoClient
import yaml
from collections import defaultdict

# Aggiungi path per import
sys.path.append(os.path.dirname(__file__))

from Utils.tenant import Tenant

def load_config():
    """Carica configurazione dal file config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Errore caricamento config: {e}")
        return None

def analyze_mongodb_representatives():
    """
    Analizza i rappresentanti reali in MongoDB
    """
    print(f"🔍 ANALISI RAPPRESENTANTI REALI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    try:
        # Connetti a MongoDB
        client = MongoClient('mongodb://localhost:27017')
        db = client['classificazioni']
        collection_name = "wopta_16c222a9-f293-11ef-9315-96000228e7fe"
        collection = db[collection_name]
        
        print(f"✅ Connesso alla collection: {collection_name}")
        
        # Conta totale documenti
        total_docs = collection.count_documents({})
        print(f"📊 Totale documenti: {total_docs}")
        
        # 1. Analizza classification_type
        print(f"\n🎯 ANALISI PER CLASSIFICATION_TYPE:")
        classification_types = collection.aggregate([
            {"$group": {"_id": "$classification_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ])
        
        for ct in classification_types:
            print(f"   {ct['_id']}: {ct['count']} documenti")
        
        # 2. Analizza metadati is_representative
        print(f"\n🏷️ ANALISI METADATI IS_REPRESENTATIVE:")
        
        # Documenti con is_representative = true
        repr_count = collection.count_documents({"metadata.is_representative": True})
        print(f"   is_representative = True: {repr_count} documenti")
        
        # Documenti con representative = true (campo duplicato)
        repr_dup_count = collection.count_documents({"metadata.representative": True})
        print(f"   representative = True: {repr_dup_count} documenti")
        
        # Documenti con outlier = true
        outlier_count = collection.count_documents({"metadata.outlier": True})
        print(f"   outlier = True: {outlier_count} documenti")
        
        # 3. Analisi cluster_id
        print(f"\n🔗 ANALISI CLUSTER_ID:")
        
        # Documenti con cluster_id != -1
        clustered_count = collection.count_documents({"metadata.cluster_id": {"$ne": -1}})
        print(f"   cluster_id != -1: {clustered_count} documenti")
        
        # Documenti con cluster_id = -1
        unclustered_count = collection.count_documents({"metadata.cluster_id": -1})
        print(f"   cluster_id = -1: {unclustered_count} documenti")
        
        # 4. Combina analisi
        print(f"\n🔍 ANALISI COMBINATA:")
        
        # Rappresentanti con cluster_id != -1
        valid_repr = collection.count_documents({
            "metadata.is_representative": True,
            "metadata.cluster_id": {"$ne": -1}
        })
        print(f"   Rappresentanti validi (is_representative=True & cluster_id!=-1): {valid_repr}")
        
        # Documenti classificati come RAPPRESENTANTE
        classified_repr = collection.count_documents({"classification_type": "RAPPRESENTANTE"})
        print(f"   Classificati come RAPPRESENTANTE: {classified_repr}")
        
        # 5. Esempi di rappresentanti
        print(f"\n📝 ESEMPI DI DOCUMENTI RAPPRESENTANTI:")
        
        representatives = collection.find({
            "metadata.is_representative": True
        }).limit(5)
        
        for i, doc in enumerate(representatives, 1):
            print(f"\n   {i}. Session: {doc.get('session_id', 'N/A')}")
            print(f"      Classification: {doc.get('classification', 'N/A')}")
            print(f"      Classification Type: {doc.get('classification_type', 'N/A')}")
            print(f"      Cluster ID: {doc.get('metadata', {}).get('cluster_id', 'N/A')}")
            print(f"      Is Representative: {doc.get('metadata', {}).get('is_representative', 'N/A')}")
            print(f"      Representative: {doc.get('metadata', {}).get('representative', 'N/A')}")
            print(f"      Review Status: {doc.get('review_status', 'N/A')}")
            
        # 6. Analisi review_status
        print(f"\n📋 ANALISI REVIEW_STATUS:")
        review_statuses = collection.aggregate([
            {"$group": {"_id": "$review_status", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ])
        
        for rs in review_statuses:
            print(f"   {rs['_id']}: {rs['count']} documenti")
        
        # 7. Documenti per review queue
        print(f"\n🔍 DOCUMENTI PER REVIEW QUEUE:")
        
        # Documenti che dovrebbero essere in review
        needs_review = collection.count_documents({"review_status": "needs_review"})
        print(f"   needs_review: {needs_review} documenti")
        
        # Documenti pending_review
        pending_review = collection.count_documents({"review_status": "pending_review"})
        print(f"   pending_review: {pending_review} documenti")
        
        # 8. Query per review queue (simula i filtri del frontend)
        print(f"\n🎯 SIMULAZIONE FILTRI REVIEW QUEUE:")
        
        # Filtro 1: status needs_review + rappresentanti
        filter1 = {
            "review_status": "needs_review",
            "metadata.is_representative": True
        }
        count1 = collection.count_documents(filter1)
        print(f"   needs_review + is_representative: {count1} documenti")
        
        # Filtro 2: status needs_review + outlier
        filter2 = {
            "review_status": "needs_review", 
            "metadata.outlier": True
        }
        count2 = collection.count_documents(filter2)
        print(f"   needs_review + outlier: {count2} documenti")
        
        # Filtro 3: status pending_review
        filter3 = {"review_status": "pending_review"}
        count3 = collection.count_documents(filter3)
        print(f"   pending_review: {count3} documenti")
        
        # 9. Esempi specifici per debug
        print(f"\n🐛 DEBUG: ESEMPI PER REVIEW QUEUE")
        
        if count1 > 0:
            print(f"\n   📌 Esempio needs_review + rappresentante:")
            example1 = collection.find_one(filter1)
            if example1:
                print(f"      Session: {example1.get('session_id', 'N/A')}")
                print(f"      Classification: {example1.get('classification', 'N/A')}")
                print(f"      Review Status: {example1.get('review_status', 'N/A')}")
                print(f"      Is Representative: {example1.get('metadata', {}).get('is_representative', 'N/A')}")
        
        if count2 > 0:
            print(f"\n   📌 Esempio needs_review + outlier:")
            example2 = collection.find_one(filter2)
            if example2:
                print(f"      Session: {example2.get('session_id', 'N/A')}")
                print(f"      Classification: {example2.get('classification', 'N/A')}")
                print(f"      Review Status: {example2.get('review_status', 'N/A')}")
                print(f"      Is Outlier: {example2.get('metadata', {}).get('outlier', 'N/A')}")
        
        client.close()
        
        # 10. Conclusioni
        print(f"\n{'='*80}")
        print(f"📊 CONCLUSIONI:")
        print(f"{'='*80}")
        
        if valid_repr == 0:
            print(f"🚨 PROBLEMA IDENTIFICATO: Nessun rappresentante valido trovato!")
            print(f"   - Clustering potrebbe non aver marcato correttamente i rappresentanti")
            print(f"   - O i metadati sono stati persi durante qualche aggiornamento")
        elif count1 == 0 and count2 == 0 and count3 == 0:
            print(f"🚨 PROBLEMA IDENTIFICATO: Nessun documento in review!")
            print(f"   - Tutti i documenti sono 'auto_classified'")
            print(f"   - La pipeline non sta marcando documenti per review")
        else:
            print(f"✅ Dati sembrano corretti:")
            print(f"   - {valid_repr} rappresentanti validi")
            print(f"   - {count1 + count2 + count3} documenti in review queue")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Funzione principale"""
    analyze_mongodb_representatives()

if __name__ == "__main__":
    main()
