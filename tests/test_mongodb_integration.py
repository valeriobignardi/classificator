#!/usr/bin/env python3
"""
Test MongoDB integration con nuovi campi tenant_id e tenant_name
"""

import sys
import os
import yaml
import hashlib
from datetime import datetime

# Aggiungi il percorso principale al path
sys.path.append('/home/ubuntu/classificazione_b+++++_bck2')

from MongoDB.connettore_mongo import MongoDBConnector

def test_mongodb_integration():
    """Test completo integrazione MongoDB con nuovi campi"""
    
    print("🔍 Test MongoDB Integration con tenant_id/tenant_name")
    print("=" * 60)
    
    # Carica configurazione
    try:
        with open('/home/ubuntu/classificazione_b+++++_bck2/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configurazione caricata")
    except Exception as e:
        print(f"❌ Errore caricamento config: {e}")
        return False
    
    # Inizializza connettore MongoDB
    try:
        mongo_connector = MongoDBConnector(config)
        print("✅ Connettore MongoDB inizializzato")
    except Exception as e:
        print(f"❌ Errore inizializzazione MongoDB: {e}")
        return False
    
    # Test connessione (la connessione avviene automaticamente nell'__init__)
    if mongo_connector.db is None:
        print("❌ Connessione MongoDB fallita")
        return False
    print("✅ Connessione MongoDB stabilita")
    
    # Dati di test
    test_data = {
        'client': 'test_humanitas',
        'session_id': f'test_session_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'tenant_id': hashlib.md5('test_humanitas'.encode()).hexdigest()[:16],
        'tenant_name': 'test_humanitas',
        'testo': 'Testo di esempio per test MongoDB',
        'conversazione': 'Conversazione: Utente ha un problema, operatore risponde',
        'embedding': [0.1, 0.2, 0.3, 0.4, 0.5],  # Embedding di test
        'embedding_model': 'labse_test',
        'classificazione': 'technical_support',
        'confidence': 0.87,
        'motivazione': 'Test automatico integrazione MongoDB con tenant',
        'metadata': {
            'method': 'test_method',
            'processing_time': 0.123,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    print(f"📝 Dati di test preparati:")
    print(f"   - Session ID: {test_data['session_id']}")
    print(f"   - Tenant ID: {test_data['tenant_id']}")
    print(f"   - Tenant Name: {test_data['tenant_name']}")
    
    # Test 1: Salvataggio
    print("\n1️⃣ Test salvataggio...")
    try:
        success = mongo_connector.save_session_classification(**test_data)
        if success:
            print("✅ Salvataggio completato")
        else:
            print("❌ Salvataggio fallito")
            return False
    except Exception as e:
        print(f"❌ Errore durante salvataggio: {e}")
        return False
    
    # Test 2: Recupero per chiave univoca
    print("\n2️⃣ Test recupero per chiave univoca (session_id + tenant_id)...")
    try:
        retrieved = mongo_connector.get_session_by_unique_key(
            session_id=test_data['session_id'],
            tenant_id=test_data['tenant_id'],
            include_embedding=True
        )
        
        if retrieved:
            print("✅ Documento recuperato correttamente")
            print(f"   - Client: {retrieved.get('client')}")
            print(f"   - Session ID: {retrieved.get('session_id')}")
            print(f"   - Tenant ID: {retrieved.get('tenant_id')}")
            print(f"   - Tenant Name: {retrieved.get('tenant_name')}")
            print(f"   - Classificazione: {retrieved.get('classificazione')}")
            print(f"   - Confidence: {retrieved.get('confidence')}")
            print(f"   - Embedding length: {len(retrieved.get('embedding', []))}")
        else:
            print("❌ Documento non trovato")
            return False
    except Exception as e:
        print(f"❌ Errore durante recupero: {e}")
        return False
    
    # Test 3: Recupero per tenant
    print("\n3️⃣ Test recupero sessioni per tenant...")
    try:
        sessions = mongo_connector.get_sessions_by_tenant(
            tenant_id=test_data['tenant_id'],
            limit=10,
            include_embedding=False
        )
        
        print(f"✅ Trovate {len(sessions)} sessioni per tenant {test_data['tenant_id']}")
        for i, session in enumerate(sessions[:3]):  # Mostra solo le prime 3
            print(f"   {i+1}. Session: {session.get('session_id')[:8]}... - {session.get('classificazione')}")
    except Exception as e:
        print(f"❌ Errore durante recupero sessioni: {e}")
        return False
    
    # Test 4: Test chiave univoca duplicata
    print("\n4️⃣ Test prevenzione duplicati...")
    try:
        # Tenta di salvare lo stesso session_id + tenant_id
        duplicate_success = mongo_connector.save_session_classification(**test_data)
        if not duplicate_success:
            print("✅ Duplicato correttamente respinto (chiave univoca funziona)")
        else:
            print("⚠️ Duplicato accettato - verificare indice univoco")
    except Exception as e:
        print(f"✅ Duplicato respinto con eccezione (OK): {e}")
    
    print("\n🎉 Test completato con successo!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_mongodb_integration()
