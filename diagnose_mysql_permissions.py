#!/usr/bin/env python3
"""
Diagnosi e soluzione per i permessi MySQL RemoteTagSyncService

Scopo: Identificare il problema di permessi e proporre la soluzione SQL
Autore: Valerio Bignardi  
Data: 2025-01-15
"""

import mysql.connector
import yaml

def diagnose_mysql_permissions():
    """Diagnosi dei permessi MySQL per l'utente taggenerator"""
    print("🔍 Diagnosi permessi MySQL per RemoteTagSyncService")
    print("=" * 60)
    
    # Leggi configurazione
    with open('/home/ubuntu/classificatore/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    db_cfg = config['database']
    
    try:
        # Connessione come taggenerator al database common
        print(f"📋 Tentativo connessione: {db_cfg['user']}@{db_cfg['host']}:{db_cfg['port']}")
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
        
        # Controlla privilegi utente corrente
        print("\n🔍 Controllo privilegi utente taggenerator...")
        cur.execute("SHOW GRANTS FOR CURRENT_USER")
        grants = cur.fetchall()
        
        print("📋 Privilegi attuali:")
        for grant in grants:
            print(f"   {grant[0]}")
        
        # Verifica se può accedere a humanitas
        print("\n🔍 Test accesso database humanitas...")
        try:
            cur.execute("USE humanitas")
            print("✅ Accesso al database 'humanitas' riuscito")
            
            # Test permessi specifici
            print("🔍 Test permessi specifici...")
            
            # Test SELECT
            try:
                cur.execute("SELECT 1")
                print("✅ Permesso SELECT: OK")
            except Exception as e:
                print(f"❌ Permesso SELECT: {e}")
            
            # Test CREATE TABLE (temporanea)
            try:
                cur.execute("CREATE TEMPORARY TABLE test_permissions (id INT)")
                cur.execute("DROP TEMPORARY TABLE test_permissions")
                print("✅ Permesso CREATE: OK")
            except Exception as e:
                print(f"❌ Permesso CREATE: {e}")
            
            # Test INSERT (su tabella esistente se presente)
            try:
                cur.execute("SHOW TABLES LIKE 'conversations_tags'")
                if cur.fetchone():
                    # Tabella esiste, test INSERT
                    test_insert = """
                    INSERT INTO conversations_tags (tag_id, tag_description, is_system_tag) 
                    VALUES ('TEST_PERMISSION', 'Test permessi', 1)
                    ON DUPLICATE KEY UPDATE updated_at=CURRENT_TIMESTAMP
                    """
                    cur.execute(test_insert)
                    
                    # Cleanup
                    cur.execute("DELETE FROM conversations_tags WHERE tag_id = 'TEST_PERMISSION'")
                    print("✅ Permesso INSERT/UPDATE/DELETE: OK")
                else:
                    print("⚠️ Tabella conversations_tags non trovata, impossibile testare INSERT")
            except Exception as e:
                print(f"❌ Permesso INSERT/UPDATE/DELETE: {e}")
                
        except Exception as e:
            print(f"❌ Accesso database 'humanitas' fallito: {e}")
            print("\n💡 SOLUZIONE RICHIESTA:")
            print("   L'utente 'taggenerator' non ha accesso al database 'humanitas'")
            print("   Eseguire questi comandi SQL come amministratore:")
            print()
            print("   -- Concedi tutti i privilegi su database humanitas")
            print("   GRANT ALL PRIVILEGES ON humanitas.* TO 'taggenerator'@'%';")
            print("   FLUSH PRIVILEGES;")
            print()
            print("   -- Oppure privilegi specifici minimi:")
            print("   GRANT SELECT, INSERT, UPDATE, DELETE, CREATE ON humanitas.* TO 'taggenerator'@'%';")
            print("   FLUSH PRIVILEGES;")
            return False
        
        cur.close()
        conn.close()
        return True
        
    except mysql.connector.Error as e:
        print(f"❌ Errore connessione al database common: {e}")
        return False

def suggest_solution():
    """Suggerisce la soluzione per risolvere il problema"""
    print("\n" + "=" * 60)
    print("🔧 SOLUZIONE CONSIGLIATA")
    print("=" * 60)
    print()
    print("Il problema è che l'utente 'taggenerator' non ha i permessi necessari")
    print("per accedere al database 'humanitas' sul server MySQL remoto.")
    print()
    print("PER RISOLVERE IL PROBLEMA:")
    print("1. Connettersi al server MySQL remoto come amministratore:")
    print("   mysql -h 159.69.223.201 -u root -p")
    print()
    print("2. Eseguire questi comandi SQL:")
    print()
    print("   -- Concedi privilegi necessari per RemoteTagSyncService")
    print("   GRANT SELECT, INSERT, UPDATE, DELETE, CREATE ON humanitas.* TO 'taggenerator'@'%';")
    print("   FLUSH PRIVILEGES;")
    print()
    print("3. Verificare che i privilegi siano stati applicati:")
    print("   SHOW GRANTS FOR 'taggenerator'@'%';")
    print()
    print("DOPO la correzione dei permessi, il training supervisionato dovrebbe")
    print("automaticamente salvare le classificazioni nel database remoto.")

if __name__ == '__main__':
    success = diagnose_mysql_permissions()
    
    if not success:
        suggest_solution()
    else:
        print("\n✅ Tutti i permessi sono corretti!")
        print("   Il RemoteTagSyncService dovrebbe funzionare normalmente.")