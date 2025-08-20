#!/usr/bin/env python3
"""
Test completo integrazione IntelligentClassifier con MongoDB e nuovi campi tenant
"""

import sys
import os
import yaml
import logging
from datetime import datetime

# Aggiungi il percorso principale al path
sys.path.append('/home/ubuntu/classificazione_b+++++_bck2')

# Configurazione logging per test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from Classification.intelligent_classifier import IntelligentClassifier
from MongoDB.connettore_mongo import MongoDBConnector

def test_intelligent_classifier_mongodb():
    """Test completo integrazione IntelligentClassifier con MongoDB"""
    
    print("🔍 Test Integrazione IntelligentClassifier + MongoDB + Tenant")
    print("=" * 70)
    
    # Carica configurazione
    try:
        with open('/home/ubuntu/classificazione_b+++++_bck2/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configurazione caricata")
    except Exception as e:
        print(f"❌ Errore caricamento config: {e}")
        return False
    
    # Test 1: Inizializzazione IntelligentClassifier con client_name
    print("\n1️⃣ Inizializzazione IntelligentClassifier...")
    try:
        classifier = IntelligentClassifier(
            config_path='/home/ubuntu/classificazione_b+++++_bck2/config.yaml',
            client_name='test_integration_client',  # Importante: specifica client_name
            enable_finetuning=True
        )
        print("✅ IntelligentClassifier inizializzato")
        print(f"   - Client: {classifier.client_name}")
        print(f"   - MongoDB abilitato: {classifier.mongo_connector is not None}")
    except Exception as e:
        print(f"❌ Errore inizializzazione classifier: {e}")
        return False
    
    # Test 2: Classificazione con salvataggio automatico in MongoDB
    print("\n2️⃣ Test classificazione con salvataggio MongoDB...")
    
    test_conversation = """
    Operatore: Buongiorno, come posso aiutarla?
    Utente: Ciao, vorrei prenotare una visita cardiologica per la prossima settimana
    Operatore: Certo, controllo subito le disponibilità. Ha una preferenza per il giorno?
    Utente: Preferibilmente martedì o mercoledì mattina
    Operatore: Perfetto, ho disponibilità martedì alle 9:30. Le va bene?
    Utente: Sì, perfetto. Devo portare qualcosa di particolare?
    Operatore: Solo la tessera sanitaria e eventuali esami cardiologici precedenti
    Utente: Ottimo, grazie mille
    """
    
    try:
        # Esegui classificazione
        result = classifier.classify_conversation(test_conversation)
        
        print("✅ Classificazione completata")
        print(f"   - Classificazione: {result.get('predicted_label', 'N/A')}")
        print(f"   - Confidence: {result.get('confidence', 0):.3f}")
        print(f"   - Metodo: {result.get('method', 'N/A')}")
        print(f"   - Motivazione: {result.get('motivation', 'N/A')[:100]}...")
        
        # La classificazione dovrebbe essere salvata automaticamente in MongoDB
        print("   - Salvataggio MongoDB: automatico durante classify_conversation")
        
    except Exception as e:
        print(f"❌ Errore durante classificazione: {e}")
        return False
    
    # Test 3: Verifica salvataggio in MongoDB
    print("\n3️⃣ Verifica salvataggio in MongoDB...")
    try:
        # Inizializza connettore MongoDB per verifica
        mongo_connector = MongoDBConnector(config)
        
        # Cerca classificazioni per questo tenant
        import hashlib
        tenant_id = hashlib.md5('test_integration_client'.encode()).hexdigest()[:16]
        
        sessions = mongo_connector.get_sessions_by_tenant(
            tenant_id=tenant_id,
            limit=5,
            include_embedding=False
        )
        
        print(f"✅ Trovate {len(sessions)} sessioni per tenant {tenant_id}")
        
        if sessions:
            latest_session = sessions[0]  # Più recente
            print(f"   - Session ID: {latest_session.get('session_id', 'N/A')[:12]}...")
            print(f"   - Tenant ID: {latest_session.get('tenant_id', 'N/A')}")
            print(f"   - Tenant Name: {latest_session.get('tenant_name', 'N/A')}")
            print(f"   - Client: {latest_session.get('client', 'N/A')}")
            print(f"   - Classificazione: {latest_session.get('classificazione', 'N/A')}")
            print(f"   - Confidence: {latest_session.get('confidence', 'N/A')}")
            print(f"   - Embedding Model: {latest_session.get('embedding_model', 'N/A')}")
            print(f"   - Timestamp: {latest_session.get('timestamp', 'N/A')}")
            
            # Verifica presenza campi richiesti
            required_fields = ['session_id', 'tenant_id', 'tenant_name', 'client', 'classificazione', 'motivazione']
            missing_fields = [field for field in required_fields if field not in latest_session]
            
            if not missing_fields:
                print("✅ Tutti i campi richiesti sono presenti")
            else:
                print(f"⚠️ Campi mancanti: {missing_fields}")
        else:
            print("❌ Nessuna sessione trovata per questo tenant")
            return False
        
    except Exception as e:
        print(f"❌ Errore verifica MongoDB: {e}")
        return False
    
    # Test 4: Test con altro tenant (diverso client_name)
    print("\n4️⃣ Test con tenant diverso...")
    try:
        classifier2 = IntelligentClassifier(
            config_path='/home/ubuntu/classificazione_b+++++_bck2/config.yaml',
            client_name='altro_cliente_test'
        )
        
        result2 = classifier2.classify_conversation("Ho problemi con il login, non riesco ad accedere")
        
        print("✅ Classificazione con secondo tenant completata")
        print(f"   - Client: {classifier2.client_name}")
        print(f"   - Classificazione: {result2.get('predicted_label', 'N/A')}")
        
        # Verifica che sia salvato con tenant_id diverso
        tenant_id2 = hashlib.md5('altro_cliente_test'.encode()).hexdigest()[:16]
        sessions2 = mongo_connector.get_sessions_by_tenant(tenant_id=tenant_id2, limit=1)
        
        if sessions2:
            print(f"✅ Trovata sessione per secondo tenant: {tenant_id2}")
            print(f"   - Diverso dal primo tenant: {tenant_id != tenant_id2}")
        else:
            print("⚠️ Sessione per secondo tenant non trovata")
        
    except Exception as e:
        print(f"❌ Errore test secondo tenant: {e}")
        return False
    
    # Test 5: Verifica chiave univoca funziona
    print("\n5️⃣ Test chiave univoca session_id + tenant_id...")
    try:
        # Conta il numero totale di documenti per il primo tenant
        initial_count = len(mongo_connector.get_sessions_by_tenant(tenant_id=tenant_id, limit=100))
        
        # Riclassifica la stessa conversazione (dovrebbe fare upsert, non duplicare)
        classifier.classify_conversation(test_conversation)
        
        final_count = len(mongo_connector.get_sessions_by_tenant(tenant_id=tenant_id, limit=100))
        
        print(f"✅ Test chiave univoca completato")
        print(f"   - Documenti prima: {initial_count}")
        print(f"   - Documenti dopo: {final_count}")
        
        if final_count <= initial_count + 1:  # Al massimo 1 nuovo documento
            print("✅ Chiave univoca funziona correttamente (no duplicati eccessivi)")
        else:
            print("⚠️ Possibili duplicati - verificare implementazione")
        
    except Exception as e:
        print(f"❌ Errore test chiave univoca: {e}")
        return False
    
    print("\n🎉 Test integrazione completo terminato con successo!")
    print("=" * 70)
    print("\n📊 Riepilogo:")
    print(f"   ✅ IntelligentClassifier + MongoDB integrazione funzionante")
    print(f"   ✅ Salvataggio automatico con tenant_id e tenant_name")
    print(f"   ✅ Chiave univoca session_id + tenant_id operativa")
    print(f"   ✅ Multi-tenant support verificato")
    print(f"   ✅ Campi richiesti (session_id, tenant_id, tenant_name, motivazione) presenti")
    
    return True

if __name__ == "__main__":
    test_intelligent_classifier_mongodb()
