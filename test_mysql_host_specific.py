#!/usr/bin/env python3
"""
Test specifico per verificare il problema di host/permessi MySQL

Scopo: Verificare se il problema è l'host di provenienza o i permessi del database
Autore: Valerio Bignardi  
Data: 2025-01-15
"""

import mysql.connector
import yaml
import socket

def get_our_hostname():
    """Ottiene il nostro hostname/IP"""
    try:
        # Ottieni hostname
        hostname = socket.gethostname()
        # Ottieni IP locale
        local_ip = socket.gethostbyname(hostname)
        return hostname, local_ip
    except:
        return "unknown", "unknown"

def test_mysql_host_permissions():
    """Test per verificare permessi host-specific"""
    print("🔍 Test permessi MySQL - Focus su HOST")
    print("=" * 50)
    
    # Info sul nostro host
    hostname, local_ip = get_our_hostname()
    print(f"🖥️  Nostro hostname: {hostname}")
    print(f"🌐 Nostro IP locale: {local_ip}")
    print()
    
    # Leggi configurazione
    with open('/home/ubuntu/classificatore/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    db_cfg = config['database']
    
    print(f"📋 Configurazione database:")
    print(f"   Host: {db_cfg['host']}:{db_cfg['port']}")
    print(f"   User: {db_cfg['user']}")
    print(f"   Database: {db_cfg['database']}")
    print()
    
    # Test 1: Connessione al database comune
    print("🔍 Test 1: Connessione al database 'common'...")
    try:
        conn = mysql.connector.connect(
            host=db_cfg['host'],
            port=db_cfg.get('port', 3306),
            user=db_cfg['user'],
            password=db_cfg['password'],
            database=db_cfg['database'],  # common
            autocommit=True,
        )
        
        cur = conn.cursor()
        print("✅ Connessione al database 'common' riuscita")
        
        # Verifica utente e host correnti
        cur.execute("SELECT USER(), @@hostname")
        user_host = cur.fetchone()
        print(f"   MySQL vede: utente='{user_host[0]}', server_host='{user_host[1]}'")
        
        # Verifica privilegi
        cur.execute("SHOW GRANTS FOR CURRENT_USER")
        grants = cur.fetchall()
        print("   Privilegi attuali:")
        for grant in grants:
            grant_text = grant[0]
            print(f"     {grant_text}")
            
            # Cerca privilegi su humanitas
            if 'humanitas' in grant_text.lower():
                print("     ✅ Trovato privilegio su 'humanitas'!")
        
        cur.close()
        conn.close()
        
    except mysql.connector.Error as e:
        print(f"❌ Errore connessione database 'common': {e}")
        return False
    
    # Test 2: Tentativo accesso diretto a humanitas
    print("\n🔍 Test 2: Tentativo connessione diretta a 'humanitas'...")
    try:
        conn = mysql.connector.connect(
            host=db_cfg['host'],
            port=db_cfg.get('port', 3306),
            user=db_cfg['user'],
            password=db_cfg['password'],
            database='humanitas',  # Direttamente a humanitas
            autocommit=True,
        )
        
        cur = conn.cursor()
        print("✅ Connessione diretta al database 'humanitas' riuscita!")
        
        # Test operazioni base
        cur.execute("SHOW TABLES")
        tables = cur.fetchall()
        print(f"   Tabelle trovate: {[t[0] for t in tables]}")
        
        # Test SELECT su tabelle target
        if any('conversations_tags' in str(t) for t in tables):
            cur.execute("SELECT COUNT(*) FROM conversations_tags")
            count = cur.fetchone()[0]
            print(f"   Righe in conversations_tags: {count}")
        
        if any('ai_session_tags' in str(t) for t in tables):
            cur.execute("SELECT COUNT(*) FROM ai_session_tags")
            count = cur.fetchone()[0]
            print(f"   Righe in ai_session_tags: {count}")
        
        cur.close()
        conn.close()
        return True
        
    except mysql.connector.Error as e:
        print(f"❌ Connessione diretta a 'humanitas' fallita: {e}")
        
        # Analizza il tipo di errore
        error_str = str(e)
        if "Access denied" in error_str:
            if "to database" in error_str:
                print("   🎯 PROBLEMA: L'utente non ha permessi per il database 'humanitas'")
                print("   💡 SOLUZIONE: Concedere privilegi su database humanitas")
                print("      GRANT ALL ON humanitas.* TO 'taggenerator'@'%';")
            elif "for user" in error_str:
                print("   🎯 PROBLEMA: Credenziali sbagliate o utente non autorizzato dall'host")
                print("   💡 SOLUZIONE: Verificare utente/password o autorizzare host")
        
        return False

if __name__ == '__main__':
    test_mysql_host_permissions()