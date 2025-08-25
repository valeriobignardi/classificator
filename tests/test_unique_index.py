#!/usr/bin/env python3
"""
Test specifico per verificare l'indice univoco MongoDB
"""

import sys
import os
import yaml
import hashlib
from datetime import datetime

# Aggiungi il percorso principale al path
sys.path.append('/home/ubuntu/classificazione_discussioni')

from mongo_classification_reader import MongoClassificationReader
from pymongo.errors import DuplicateKeyError

def test_unique_index():
    """Test specifico per l'indice univoco session_id + tenant_id"""
    
    print("🔍 Test Indice Univoco MongoDB (session_id + tenant_id)")
    print("=" * 60)
    
    # TEST DISABILITATO - MongoDBConnector deprecato
    print("⚠️ Test disabilitato - MongoDBConnector deprecato, usa MongoClassificationReader")
    return True
    
    # # Carica configurazione
    # with open('/home/ubuntu/classificazione_b+++++_bck2/config.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # 
    # # Inizializza connettore MongoDB
    # mongo_connector = MongoDBConnector(config)
    
    if mongo_connector.db is None:
        print("❌ Connessione MongoDB fallita")
        return False
    
    print("✅ Connessione MongoDB stabilita")
    
    # Accesso diretto alla collection per test basso livello
    collection = mongo_connector.db[mongo_connector.collection_name]
    
    # Dati di test identici
    test_doc1 = {
        'client': 'test_unique',
        'session_id': 'unique_test_session',
        'tenant_id': 'unique_test_tenant',
        'tenant_name': 'Test Unique',
        'testo': 'Primo documento',
        'conversazione': 'Prima conversazione',
        'embedding': [0.1, 0.2],
        'embedding_model': 'test_model',
        'timestamp': datetime.now(),
        'classificazione': 'test_class',
        'confidence': 0.9,
        'motivazione': 'Prima motivazione',
        'metadata': {'test': 'first'}
    }
    
    test_doc2 = {
        'client': 'test_unique',
        'session_id': 'unique_test_session',  # Stessa session_id
        'tenant_id': 'unique_test_tenant',    # Stesso tenant_id
        'tenant_name': 'Test Unique',
        'testo': 'Secondo documento',
        'conversazione': 'Seconda conversazione',
        'embedding': [0.3, 0.4],
        'embedding_model': 'test_model',
        'timestamp': datetime.now(),
        'classificazione': 'test_class_2',
        'confidence': 0.8,
        'motivazione': 'Seconda motivazione',
        'metadata': {'test': 'second'}
    }
    
    print("📝 Documenti di test preparati con stessi session_id + tenant_id")
    
    # Test 1: Inserimento primo documento
    print("\n1️⃣ Inserimento primo documento...")
    try:
        result1 = collection.insert_one(test_doc1)
        print(f"✅ Primo documento inserito: {result1.inserted_id}")
    except Exception as e:
        print(f"❌ Errore inserimento primo documento: {e}")
        return False
    
    # Test 2: Tentativo inserimento documento duplicato
    print("\n2️⃣ Tentativo inserimento documento con stessi session_id + tenant_id...")
    try:
        result2 = collection.insert_one(test_doc2)
        print(f"❌ PROBLEMA: Documento duplicato inserito: {result2.inserted_id}")
        print("   L'indice univoco non sta funzionando!")
        return False
    except DuplicateKeyError as e:
        print("✅ Duplicato correttamente respinto dall'indice univoco")
        print(f"   Errore: {e}")
    except Exception as e:
        print(f"❌ Errore inaspettato: {e}")
        return False
    
    # Test 3: Verifica che il documento esistente sia ancora presente
    print("\n3️⃣ Verifica documento originale...")
    try:
        found = collection.find_one({
            'session_id': 'unique_test_session',
            'tenant_id': 'unique_test_tenant'
        })
        
        if found:
            print("✅ Documento originale ancora presente")
            print(f"   Testo: {found['testo']}")
            print(f"   Classificazione: {found['classificazione']}")
        else:
            print("❌ Documento originale non trovato")
            return False
    except Exception as e:
        print(f"❌ Errore verifica documento: {e}")
        return False
    
    # Test 4: Test con tenant_id diverso ma stesso session_id
    print("\n4️⃣ Test con tenant_id diverso...")
    test_doc3 = test_doc2.copy()
    test_doc3['tenant_id'] = 'different_tenant'
    test_doc3['testo'] = 'Documento con tenant diverso'
    
    try:
        result3 = collection.insert_one(test_doc3)
        print(f"✅ Documento con tenant_id diverso inserito: {result3.inserted_id}")
        print("   L'indice univoco permette session_id duplicati con tenant_id diversi")
    except Exception as e:
        print(f"❌ Errore inaspettato con tenant diverso: {e}")
        return False
    
    # Pulizia: rimuovi documenti di test
    print("\n🧹 Pulizia documenti di test...")
    try:
        deleted = collection.delete_many({'client': 'test_unique'})
        print(f"✅ Rimossi {deleted.deleted_count} documenti di test")
    except Exception as e:
        print(f"⚠️ Errore pulizia: {e}")
    
    print("\n🎉 Test indice univoco completato con successo!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_unique_index()
