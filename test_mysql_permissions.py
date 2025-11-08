#!/usr/bin/env python3
"""
Test dettagliato dei permessi MySQL per RemoteTagSyncService

Scopo: Verificare esattamente quali permessi ha l'utente taggenerator sui vari database
Autore: Valerio Bignardi  
Data: 2025-01-15
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mysql.connector
import yaml

def test_mysql_permissions():
    """Test dettagliato dei permessi MySQL"""
    print("🔍 Test permessi utente MySQL remoto...")
    
    # Leggi configurazione
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    db_cfg = config['database']
    
    try:
        # Connessione al server senza database specifico
        conn = mysql.connector.connect(
            host=db_cfg['host'],
            port=db_cfg.get('port', 3306),
            user=db_cfg['user'],
            password=db_cfg['password'],
            autocommit=True,
        )
        cur = conn.cursor()
        
        # 1. Verifica utente corrente
        cur.execute("SELECT USER(), CURRENT_USER()")
        user_info = cur.fetchone()
        print(f"   👤 Utente connesso: {user_info[0]} / Effective user: {user_info[1]}")
        
        # 2. Verifica permessi globali
        print("   🔐 Permessi globali:")
        cur.execute("SHOW GRANTS")
        grants = cur.fetchall()
        for grant in grants:
            print(f"      {grant[0]}")
        
        # 3. Test accesso database common (dovrebbe funzionare)
        print("\n   🧪 Test accesso database 'common':")
        try:
            cur.execute("USE common")
            cur.execute("SHOW TABLES LIMIT 3")
            tables = cur.fetchall()
            print(f"      ✅ Accesso riuscito. Primi 3 tabelle: {[t[0] for t in tables]}")
        except mysql.connector.Error as e:
            print(f"      ❌ Errore accesso 'common': {e}")
        
        # 4. Test accesso database humanitas (problematico)
        print("\n   🧪 Test accesso database 'humanitas':")
        try:
            cur.execute("USE humanitas")
            print("      ✅ USE humanitas riuscito")
            
            # Test SELECT
            cur.execute("SELECT COUNT(*) FROM conversations_tags")
            count = cur.fetchone()[0]
            print(f"      ✅ SELECT riuscito: {count} righe in conversations_tags")
            
            # Test INSERT (il vero test)
            test_insert = """
            INSERT IGNORE INTO conversations_tags (tag_id, tag_description, is_system_tag) 
            VALUES ('TEST_PERMISSION', 'Test permessi', 1)
            """
            cur.execute(test_insert)
            print(f"      ✅ INSERT riuscito (affected rows: {cur.rowcount})")
            
            # Cleanup del test
            cur.execute("DELETE FROM conversations_tags WHERE tag_id = 'TEST_PERMISSION'")
            print(f"      🧹 Cleanup completato")
            
        except mysql.connector.Error as e:
            print(f"      ❌ Errore accesso 'humanitas': {e}")
        
        # 5. Verifica permessi specifici su humanitas
        print("\n   🔍 Verifica permessi specifici su humanitas:")
        try:
            cur.execute("SHOW GRANTS FOR CURRENT_USER()")
            current_grants = cur.fetchall()
            for grant in current_grants:
                grant_text = grant[0]
                if 'humanitas' in grant_text.lower() or '*.*' in grant_text:
                    print(f"      🔐 {grant_text}")
        except mysql.connector.Error as e:
            print(f"      ⚠️ Non posso verificare permessi specifici: {e}")
            
        cur.close()
        conn.close()
        
    except mysql.connector.Error as e:
        print(f"   ❌ Errore connessione generale: {e}")

def test_create_database_permissions():
    """Test se possiamo creare un database di test"""
    print("\n🧪 Test creazione database di test...")
    
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    db_cfg = config['database']
    
    try:
        conn = mysql.connector.connect(
            host=db_cfg['host'],
            port=db_cfg.get('port', 3306),
            user=db_cfg['user'],
            password=db_cfg['password'],
            autocommit=True,
        )
        cur = conn.cursor()
        
        # Test creazione database
        test_db_name = "test_permissions_classificatore"
        
        try:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{test_db_name}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"   ✅ Creazione database '{test_db_name}' riuscita")
            
            # Test uso del database
            cur.execute(f"USE `{test_db_name}`")
            print(f"   ✅ Accesso al database '{test_db_name}' riuscito")
            
            # Cleanup
            cur.execute(f"DROP DATABASE IF EXISTS `{test_db_name}`")
            print(f"   🧹 Database di test rimosso")
            
        except mysql.connector.Error as e:
            print(f"   ❌ Errore test database: {e}")
            
        cur.close()
        conn.close()
        
    except mysql.connector.Error as e:
        print(f"   ❌ Errore connessione: {e}")

if __name__ == '__main__':
    print("🔐 Test Permessi MySQL RemoteTagSyncService")
    print("=" * 50)
    
    test_mysql_permissions()
    test_create_database_permissions()
    
    print("\n" + "=" * 50)