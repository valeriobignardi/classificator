#!/usr/bin/env python3
"""
Test della correzione _determine_classification_type

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

def test_classification_type_fix():
    """
    Testa la funzione _determine_classification_type corretta
    """
    try:
        # Carica config
        config = load_config()
        tenant_slug = "humanitas"
        
        print(f"🏢 Testing classification type fix per tenant: {tenant_slug}")
        
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
        
        print(f"🔍 Collection: {reader.collection_name}")
        
        # 1. Prendi alcuni documenti con metadata
        docs_with_metadata = list(collection.find({
            'metadata': {'$exists': True}
        }).limit(5))
        
        print(f"\n🧪 TEST SU {len(docs_with_metadata)} DOCUMENTI:")
        
        for i, doc in enumerate(docs_with_metadata):
            session_id = doc.get('session_id', 'N/A')
            metadata = doc.get('metadata', {})
            current_type = doc.get('classification_type')
            
            print(f"\n   📄 Documento {i+1}: {session_id}")
            print(f"      🏷️ Tipo attuale: {current_type}")
            print(f"      📊 Metadata: {metadata}")
            
            # Testa la funzione corretta
            new_type = reader._determine_classification_type(metadata)
            print(f"      ✨ Tipo corretto: {new_type}")
            
            if current_type != new_type:
                print(f"      🔄 CAMBIO NECESSARIO: {current_type} → {new_type}")
            else:
                print(f"      ✅ TIPO GIÀ CORRETTO")
        
        # 2. Conta documenti per tipo attuale vs corretto
        print(f"\n📊 ANALISI CORREZIONE NECESSARIA:")
        
        tipos_corretti = {'RAPPRESENTANTE': 0, 'OUTLIER': 0, 'PROPAGATO': 0, 'FALLBACK': 0}
        tipos_attuali = {'RAPPRESENTANTE': 0, 'OUTLIER': 0, 'PROPAGATO': 0, 'NORMALE': 0, 'NULL': 0}
        
        all_docs = collection.find({'metadata': {'$exists': True}})
        
        for doc in all_docs:
            metadata = doc.get('metadata', {})
            current_type = doc.get('classification_type')
            correct_type = reader._determine_classification_type(metadata)
            
            tipos_corretti[correct_type] = tipos_corretti.get(correct_type, 0) + 1
            
            if current_type is None:
                tipos_attuali['NULL'] += 1
            else:
                tipos_attuali[current_type] = tipos_attuali.get(current_type, 0) + 1
        
        print(f"   📊 TIPI ATTUALI:")
        for tipo, count in tipos_attuali.items():
            if count > 0:
                print(f"      {tipo}: {count}")
                
        print(f"   ✨ TIPI CORRETTI (con funzione aggiornata):")
        for tipo, count in tipos_corretti.items():
            if count > 0:
                print(f"      {tipo}: {count}")
                
        # 3. Controlla se servono aggiornamenti
        docs_needing_update = 0
        for doc in collection.find({'metadata': {'$exists': True}}):
            metadata = doc.get('metadata', {})
            current_type = doc.get('classification_type')
            correct_type = reader._determine_classification_type(metadata)
            
            if current_type != correct_type:
                docs_needing_update += 1
                
        print(f"\n🔧 DOCUMENTI CHE NECESSITANO AGGIORNAMENTO: {docs_needing_update}")
        
        if docs_needing_update > 0:
            print(f"💡 SOLUZIONE: Eseguire script di correzione massa per aggiornare tutti i classification_type")
        else:
            print(f"✅ TUTTI I DOCUMENTI HANNO GIÀ IL TIPO CORRETTO")
            
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_classification_type_fix()
