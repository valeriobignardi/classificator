#!/usr/bin/env python3
"""
Test del RemoteTagSyncService per verificare connessione e funzionalità

Scopo: Verificare se RemoteTagSyncService riesce a connettersi al database remoto
       e se il schema humanitas viene creato/popolato correttamente
Autore: Valerio Bignardi  
Data: 2025-01-15
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Database.remote_tag_sync import RemoteTagSyncService
from Utils.tenant import Tenant
import mysql.connector
import yaml

def test_connection():
    """Test della connessione al database remoto"""
    print("🔍 Test connessione database remoto...")
    
    # Leggi configurazione
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    db_cfg = config['database']
    print(f"   Host: {db_cfg['host']}:{db_cfg.get('port', 3306)}")
    print(f"   Database: {db_cfg['database']}")
    print(f"   User: {db_cfg['user']}")
    
    try:
        # Test connessione senza database specifico
        conn = mysql.connector.connect(
            host=db_cfg['host'],
            port=db_cfg.get('port', 3306),
            user=db_cfg['user'],
            password=db_cfg['password'],
            autocommit=True,
        )
        print("   ✅ Connessione al server MySQL riuscita")
        
        # Verifica database esistenti
        cur = conn.cursor()
        cur.execute("SHOW DATABASES")
        databases = [row[0] for row in cur.fetchall()]
        print(f"   📊 Database disponibili: {databases}")
        
        # Controlla se esiste schema humanitas
        if 'humanitas' in databases:
            print("   ✅ Schema 'humanitas' esiste")
            
            # Connetti allo schema humanitas e controlla tabelle
            conn.close()
            conn = mysql.connector.connect(
                host=db_cfg['host'],
                port=db_cfg.get('port', 3306),
                user=db_cfg['user'],
                password=db_cfg['password'],
                database='humanitas',
                autocommit=True,
            )
            
            cur = conn.cursor()
            cur.execute("SHOW TABLES")
            tables = [row[0] for row in cur.fetchall()]
            print(f"   📊 Tabelle in humanitas: {tables}")
            
            # Controlla contenuto tabelle se esistono
            if 'conversations_tags' in tables:
                cur.execute("SELECT COUNT(*) FROM conversations_tags")
                count = cur.fetchone()[0]
                print(f"   📊 Righe in conversations_tags: {count}")
                
                if count > 0:
                    cur.execute("SELECT tag_id, tag_description FROM conversations_tags LIMIT 5")
                    tags = cur.fetchall()
                    print(f"   📋 Esempi tag: {tags}")
            
            if 'ai_session_tags' in tables:
                cur.execute("SELECT COUNT(*) FROM ai_session_tags")
                count = cur.fetchone()[0]
                print(f"   📊 Righe in ai_session_tags: {count}")
                
                if count > 0:
                    cur.execute("SELECT session_id, tag_id, confidence_score FROM ai_session_tags LIMIT 5")
                    sessions = cur.fetchall()
                    print(f"   📋 Esempi session tag: {sessions}")
        else:
            print("   ⚠️ Schema 'humanitas' NON esiste")
        
        cur.close()
        conn.close()
        
    except mysql.connector.Error as e:
        print(f"   ❌ Errore connessione: {e}")
        return False
    
    return True

def test_remote_sync_service():
    """Test del RemoteTagSyncService"""
    print("\n🔧 Test RemoteTagSyncService...")
    
    try:
        # Crea istanza del servizio
        sync_service = RemoteTagSyncService()
        print("   ✅ RemoteTagSyncService creato con successo")
        
        # Crea tenant di test usando from_uuid (come nel pipeline reale)
        tenant = Tenant.from_uuid('015007d9-d413-11ef-86a5-96000228e7fe')
        print(f"   📋 Tenant creato: {tenant.tenant_slug}")
        
        # Test creazione schema
        sync_service._ensure_schema('humanitas')
        print("   ✅ Schema 'humanitas' creato/verificato")
        
        # Test connessione allo schema
        conn = sync_service._connect_schema('humanitas')
        print("   ✅ Connessione allo schema 'humanitas' riuscita")
        
        # Test creazione tabelle
        sync_service._ensure_tables(conn)
        print("   ✅ Tabelle create/verificate")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Errore RemoteTagSyncService: {e}")
        return False

def test_sync_with_mock_data():
    """Test sync con dati mock"""
    print("\n📊 Test sync con dati mock...")
    
    try:
        # Crea istanza del servizio
        sync_service = RemoteTagSyncService()
        
        # Crea tenant di test usando from_uuid (come nel pipeline reale)
        tenant = Tenant.from_uuid('015007d9-d413-11ef-86a5-96000228e7fe')
        
        # Crea documento mock
        class MockDocumento:
            def __init__(self, session_id, predicted_label, confidence):
                self.session_id = session_id
                self.predicted_label = predicted_label
                self.confidence = confidence
                self.classification_method = 'test_sync'
                self.classified_by = 'test_script'
        
        mock_documenti = [
            MockDocumento('test_session_001', 'PRENOTAZIONE_ESAMI', 0.95),
            MockDocumento('test_session_002', 'RITIRO_REFERTI', 0.88),
            MockDocumento('test_session_003', 'INFO_GENERALI', 0.75)
        ]
        
        # Esegui sync
        result = sync_service.sync_session_tags(tenant, mock_documenti)
        print(f"   📊 Risultato sync: {result}")
        
        if result.get('success'):
            print("   ✅ Sync completato con successo")
            return True
        else:
            print(f"   ❌ Sync fallito: {result.get('error', 'Errore sconosciuto')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Errore test sync: {e}")
        return False

if __name__ == '__main__':
    print("🚀 Test RemoteTagSyncService - Debug training supervisionato")
    print("=" * 60)
    
    success = True
    
    # Test 1: Connessione
    success &= test_connection()
    
    # Test 2: RemoteTagSyncService
    success &= test_remote_sync_service()
    
    # Test 3: Sync con dati mock
    success &= test_sync_with_mock_data()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Tutti i test completati con successo")
    else:
        print("❌ Alcuni test sono falliti")