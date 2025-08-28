#!/usr/bin/env python3
"""
Test per verificare la risoluzione corretta del problema tenant_id vs tenant_slug

Autore: Valerio Bignardi  
Data: 2025-08-28
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from mongo_classification_reader import MongoClassificationReader

def test_tenant_resolution():
    """
    Scopo: Verifica che i metodi MongoDB accettino tenant_slug e non UUID
    """
    
    print("🔧 Test risoluzione tenant_id vs tenant_slug...")
    
    try:
        # Inizializza reader MongoDB
        mongo_reader = MongoClassificationReader()
        
        # Test con tenant_slug corretto
        tenant_slug = "humanitas"  # Nome corretto
        tenant_uuid = "16c222a9-f293-11ef-9315-96000228e7fe"  # UUID che causava errore
        
        print(f"📋 Test 1: get_tenant_info_from_name con tenant_slug '{tenant_slug}'")
        tenant_info_slug = mongo_reader.get_tenant_info_from_name(tenant_slug)
        print(f"   Risultato: {tenant_info_slug}")
        
        print(f"📋 Test 2: get_tenant_info_from_name con UUID '{tenant_uuid}'")
        tenant_info_uuid = mongo_reader.get_tenant_info_from_name(tenant_uuid)
        print(f"   Risultato: {tenant_info_uuid}")
        
        print(f"📋 Test 3: generate_collection_name con tenant_slug '{tenant_slug}'")
        collection_name_slug = mongo_reader.generate_collection_name(tenant_slug)
        print(f"   Risultato: {collection_name_slug}")
        
        print(f"📋 Test 4: generate_collection_name con UUID '{tenant_uuid}'")
        collection_name_uuid = mongo_reader.generate_collection_name(tenant_uuid)
        print(f"   Risultato: {collection_name_uuid}")
        
        # Verifica che il tenant_slug funzioni e l'UUID no
        if tenant_info_slug and not tenant_info_uuid:
            print("✅ CORRETTO: tenant_slug trova tenant, UUID non lo trova")
        else:
            print("❌ ERRORE: comportamento inaspettato")
            
        return tenant_info_slug is not None and tenant_info_uuid is None
        
    except Exception as e:
        print(f"❌ Errore durante test: {e}")
        return False

if __name__ == "__main__":
    success = test_tenant_resolution()
    print(f"\n{'✅ Test PASSATO' if success else '❌ Test FALLITO'}")
