#!/usr/bin/env python3
"""
Debug ESATTO delle credenziali lette da RemoteTagSyncService

Scopo: Verificare ESATTAMENTE che credenziali vengono lette dal config.yaml
Autore: Valerio Bignardi  
Data: 2025-01-15
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Database.remote_tag_sync import RemoteTagSyncService
import yaml

def debug_credentials():
    """Debug delle credenziali ESATTE lette dal RemoteTagSyncService"""
    print("🔍 Debug credenziali RemoteTagSyncService")
    print("=" * 50)
    
    # 1. Lettura DIRETTA dal config.yaml
    print("📋 1. Lettura DIRETTA da config.yaml:")
    with open('/home/ubuntu/classificatore/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    db_cfg = config['database']
    print(f"   host: '{db_cfg['host']}'")
    print(f"   port: {db_cfg.get('port', 3306)}")
    print(f"   user: '{db_cfg['user']}'")
    print(f"   password: '{db_cfg['password'][:3]}...{db_cfg['password'][-3:]}' (len={len(db_cfg['password'])})")
    print(f"   database: '{db_cfg['database']}'")
    
    # 2. Verifica caratteri nascosti nell'utente
    user = db_cfg['user']
    print(f"\n🔍 2. Analisi caratteri utente:")
    print(f"   Lunghezza: {len(user)}")
    print(f"   Bytes: {user.encode('utf-8')}")
    print(f"   Repr: {repr(user)}")
    
    # 3. Verifica tramite RemoteTagSyncService
    print(f"\n🔧 3. Credenziali lette da RemoteTagSyncService:")
    try:
        sync_service = RemoteTagSyncService()
        service_cfg = sync_service.db_cfg
        
        print(f"   host: '{service_cfg['host']}'")
        print(f"   port: {service_cfg.get('port', 3306)}")
        print(f"   user: '{service_cfg['user']}'")
        print(f"   password: '{service_cfg['password'][:3]}...{service_cfg['password'][-3:]}' (len={len(service_cfg['password'])})")
        print(f"   database: '{service_cfg['database']}'")
        
        # Analisi caratteri utente dal service
        service_user = service_cfg['user']
        print(f"\n🔍 4. Analisi caratteri utente dal service:")
        print(f"   Lunghezza: {len(service_user)}")
        print(f"   Bytes: {service_user.encode('utf-8')}")
        print(f"   Repr: {repr(service_user)}")
        
        # Confronto
        if user == service_user:
            print(f"   ✅ Gli utenti sono IDENTICI")
        else:
            print(f"   ❌ Gli utenti sono DIVERSI!")
            print(f"      Config diretta: {repr(user)}")
            print(f"      Service:        {repr(service_user)}")
        
    except Exception as e:
        print(f"   ❌ Errore creazione RemoteTagSyncService: {e}")
    
    # 4. Test connessione con credenziali del service
    print(f"\n🧪 5. Test connessione con credenziali del service:")
    try:
        import mysql.connector
        
        conn = mysql.connector.connect(
            host=sync_service.db_cfg['host'],
            port=sync_service.db_cfg.get('port', 3306),
            user=sync_service.db_cfg['user'],
            password=sync_service.db_cfg['password'],
            autocommit=True,
        )
        
        cur = conn.cursor()
        cur.execute("SELECT USER()")
        mysql_user = cur.fetchone()[0]
        print(f"   ✅ Connessione riuscita")
        print(f"   👤 MySQL vede utente: '{mysql_user}'")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"   ❌ Errore connessione: {e}")

if __name__ == '__main__':
    debug_credentials()