#!/usr/bin/env python3
"""
Verifica se questo tenant ha mai eseguito il training supervisionato

Autore: Valerio Bignardi
Data: 2025-09-04
"""

import os
import sys
import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mongo_classification_reader import MongoClassificationReader

def load_config():
    """Carica configurazione da config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def check_training_history():
    """
    Controlla la storia del training di questo tenant
    """
    try:
        # Carica config
        config = load_config()
        tenant_slug = "humanitas"  # Il tenant è sempre humanitas in questo progetto
        
        print(f"🏢 Analizzando tenant: {tenant_slug}")
        
        # Crea oggetto tenant semplificato
        class TenantInfo:
            def __init__(self, slug):
                self.tenant_slug = slug
                self.tenant_name = slug
                self.tenant_id = slug  # Aggiungo anche tenant_id
                
        tenant = TenantInfo(tenant_slug)
        
        # Connetti a MongoDB
        reader = MongoClassificationReader(tenant=tenant)
        
        # FONDAMENTALE: Stabilisco la connessione MongoDB
        if not reader.connect():
            print(f"❌ Impossibile connettersi a MongoDB")
            return
            
        client_name = tenant_slug
        collection = reader.db[reader.collection_name]  # Usa collection interna del reader
        
        print(f"🔍 Collection: {reader.collection_name}")
        
        # 1. Conta documenti con metadati di clustering
        docs_with_clustering = collection.count_documents({
            'metadata.cluster_id': {'$exists': True}
        })
        
        print(f"\n📊 ANALISI STORIA TRAINING:")
        print(f"   🔢 Documenti con metadati clustering: {docs_with_clustering}")
        
        # 2. Conta documenti con is_representative
        docs_with_representative_flag = collection.count_documents({
            'metadata.is_representative': {'$exists': True}
        })
        
        print(f"   👑 Documenti con flag is_representative: {docs_with_representative_flag}")
        
        # 3. Conta documenti salvati da pipeline supervisionata
        docs_supervised = collection.count_documents({
            'classified_by': 'supervised_training_pipeline'
        })
        
        print(f"   🎓 Documenti da training supervisionato: {docs_supervised}")
        
        # 4. Conta documenti con review_reason supervisionato
        docs_supervised_review = collection.count_documents({
            'review_reason': 'supervised_training_representative'
        })
        
        print(f"   🔍 Documenti per review supervisionata: {docs_supervised_review}")
        
        # 5. Cerca documenti con metodo clustering
        docs_clustering_method = collection.count_documents({
            'final_decision.method': {'$regex': 'clustering', '$options': 'i'}
        })
        
        print(f"   🧩 Documenti con metodo clustering: {docs_clustering_method}")
        
        # 6. Controlla se ci sono stati tentativi di training
        training_logs = collection.find({
            '$or': [
                {'notes': {'$regex': 'cluster', '$options': 'i'}},
                {'classified_by': {'$regex': 'training', '$options': 'i'}},
                {'method': {'$regex': 'REPRESENTATIVE|CLUSTER', '$options': 'i'}}
            ]
        }).limit(5)
        
        logs_found = list(training_logs)
        print(f"\n🔍 CAMPIONI DI DOCUMENTI DI TRAINING: {len(logs_found)}")
        
        for i, doc in enumerate(logs_found):
            print(f"\n   📄 Documento {i+1}:")
            print(f"      session_id: {doc.get('session_id', 'N/A')}")
            print(f"      classified_by: {doc.get('classified_by', 'N/A')}")
            print(f"      review_reason: {doc.get('review_reason', 'N/A')}")
            print(f"      method: {doc.get('final_decision', {}).get('method', 'N/A')}")
            print(f"      is_representative: {doc.get('metadata', {}).get('is_representative', 'N/A')}")
        
        # 7. DIAGNOSI FINALE
        print(f"\n🏁 DIAGNOSI:")
        
        if docs_supervised == 0:
            print(f"   ❌ PROBLEMA IDENTIFICATO: Questo tenant NON ha mai eseguito il training supervisionato!")
            print(f"   💡 SOLUZIONE: Esegui /api/supervised_training_advanced/{client_name}")
            print(f"   📋 Solo quello crea i rappresentanti con is_representative=True")
        elif docs_with_representative_flag == 0:
            print(f"   ❌ PROBLEMA: Training eseguito ma metadati corrotti")
            print(f"   💡 SOLUZIONE: Re-eseguire training supervisionato")
        else:
            print(f"   ✅ Training supervisionato eseguito correttamente")
            print(f"   🔍 Problema probabilmente nella logica di classificazione")
            
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_training_history()
