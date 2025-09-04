#!/usr/bin/env python3
"""
Controlla se il training supervisionato viene chiamato per questo tenant

Autore: Valerio Bignardi
Data: 2025-09-04
"""

import os
import sys
import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mongo_classification_reader import MongoClassificationReader

def check_training_calls():
    """
    Controlla i log di chiamate al training supervisionato
    """
    try:
        # Leggo il config per il tenant
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        tenant_slug = "humanitas"
        
        # Crea oggetto tenant
        class TenantInfo:
            def __init__(self, slug):
                self.tenant_slug = slug
                self.tenant_name = slug
                self.tenant_id = slug
                
        tenant = TenantInfo(tenant_slug)
        
        # Connetti a MongoDB
        reader = MongoClassificationReader(tenant=tenant)
        if not reader.connect():
            print(f"❌ Impossibile connettersi a MongoDB")
            return
            
        collection = reader.db[reader.collection_name]
        
        print(f"🔍 Analisi chiamate training supervisionato per {tenant_slug}")
        print(f"📊 Collection: {reader.collection_name}")
        
        # 1. Cerca documenti con classified_by del training supervisionato
        supervised_docs = list(collection.find({
            'classified_by': 'supervised_training_pipeline'
        }).limit(3))
        
        print(f"\n📋 DOCUMENTI DA TRAINING SUPERVISIONATO: {len(supervised_docs)}")
        if supervised_docs:
            for i, doc in enumerate(supervised_docs[:2]):
                print(f"\n   📄 Documento {i+1}:")
                print(f"      session_id: {doc.get('session_id', 'N/A')}")
                print(f"      classified_by: {doc.get('classified_by', 'N/A')}")
                print(f"      metadata: {doc.get('metadata', {})}")
                print(f"      classification_type: {doc.get('classification_type', 'N/A')}")
        
        # 2. Cerca documenti con cluster_metadata
        cluster_docs = list(collection.find({
            'metadata.cluster_id': {'$exists': True}
        }).limit(3))
        
        print(f"\n🧩 DOCUMENTI CON CLUSTER METADATA: {len(cluster_docs)}")
        if cluster_docs:
            for i, doc in enumerate(cluster_docs[:2]):
                print(f"\n   📄 Documento {i+1}:")
                print(f"      session_id: {doc.get('session_id', 'N/A')}")
                print(f"      metadata.cluster_id: {doc.get('metadata', {}).get('cluster_id', 'N/A')}")
                print(f"      metadata.is_representative: {doc.get('metadata', {}).get('is_representative', 'N/A')}")
                print(f"      classification_type: {doc.get('classification_type', 'N/A')}")
        
        # 3. Controlla documenti con review_reason
        review_docs = list(collection.find({
            'review_reason': {'$exists': True, '$ne': None}
        }).limit(3))
        
        print(f"\n🔍 DOCUMENTI CON REVIEW_REASON: {len(review_docs)}")
        if review_docs:
            for i, doc in enumerate(review_docs[:2]):
                print(f"\n   📄 Documento {i+1}:")
                print(f"      session_id: {doc.get('session_id', 'N/A')}")
                print(f"      review_reason: {doc.get('review_reason', 'N/A')}")
                print(f"      needs_review: {doc.get('needs_review', 'N/A')}")
                print(f"      classification_type: {doc.get('classification_type', 'N/A')}")
        
        # 4. Verifica documenti con is_representative=True
        representative_docs = list(collection.find({
            'metadata.is_representative': True
        }).limit(3))
        
        print(f"\n👑 DOCUMENTI RAPPRESENTANTI: {len(representative_docs)}")
        if representative_docs:
            for i, doc in enumerate(representative_docs[:2]):
                print(f"\n   📄 Documento {i+1}:")
                print(f"      session_id: {doc.get('session_id', 'N/A')}")
                print(f"      metadata.cluster_id: {doc.get('metadata', {}).get('cluster_id', 'N/A')}")
                print(f"      metadata.is_representative: {doc.get('metadata', {}).get('is_representative', 'N/A')}")
                print(f"      classification_type: {doc.get('classification_type', 'N/A')}")
        
        # 5. DIAGNOSI FINALE
        total_docs = collection.count_documents({})
        print(f"\n🏁 DIAGNOSI FINALE:")
        print(f"   📊 Totale documenti: {total_docs}")
        print(f"   🎓 Training supervisionato: {len(supervised_docs)}")
        print(f"   🧩 Con cluster metadata: {len(cluster_docs)}")
        print(f"   🔍 Con review reason: {len(review_docs)}")
        print(f"   👑 Rappresentanti: {len(representative_docs)}")
        
        if len(supervised_docs) == 0:
            print(f"\n❌ PROBLEMA: Nessun documento da training supervisionato!")
            print(f"   💡 Il tenant non ha mai eseguito supervised_training_advanced")
        elif len(cluster_docs) == 0:
            print(f"\n❌ PROBLEMA: Training chiamato ma nessun cluster metadata salvato!")
            print(f"   💡 Bug nel salvataggio cluster_metadata")
        else:
            print(f"\n✅ Training supervisionato eseguito e metadata salvati")
            
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_training_calls()
