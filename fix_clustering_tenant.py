#!/usr/bin/env python3
"""
Script per la correzione diretta dei parametri di clustering HDBSCAN
nel database MySQL del tenant specifico.

Autore: Valerio Bignardi
Data: 2025-01-22
Ultimo aggiornamento: 2025-01-22 - Supporto architettura multi-database
"""

import yaml
import mysql.connector
from mysql.connector import Error

# Configurazione tenant
tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"

# Parametri di clustering ottimali per permettere formazione cluster
new_params = {
    'min_cluster_size': '5',         # Era 13, troppo alto per 1360 doc
    'min_samples': '3',              # Era 16, troppo alto
    'cluster_selection_epsilon': '0.1',  # Era 0.28, troppo restrittivo
    'alpha': '1.0',                  # Era 0.4, aumentiamo per stabilità
    'cluster_selection_method': 'eom' # Manteniamo EOM
}

def get_database_config():
    """Carica la configurazione del database common dal file config.yaml"""
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config['database']

def get_tenant_database_info(tenant_id):
    """
    Ottiene le informazioni del database per il tenant specifico
    dal database common
    """
    config = get_database_config()
    
    try:
        # Connessione al database common
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
        # Ottieni info tenant e pool
        query = """
        SELECT t.tenant_name, t.tenant_database, p.pool_host, p.pool_user, 
               p.pool_password, p.pool_port, p.pool_database
        FROM tenants t
        JOIN pools p ON t.pool_id = p.pool_id
        WHERE t.tenant_id = %s
        """
        
        cursor.execute(query, (tenant_id,))
        result = cursor.fetchone()
        
        if not result:
            print(f"❌ Tenant {tenant_id} non trovato!")
            return None
        
        tenant_name, tenant_database, pool_host, pool_user, pool_password, pool_port, pool_database = result
        
        print(f"✅ Tenant trovato: {tenant_name}")
        print(f"   Database tenant: {tenant_database}")
        print(f"   Host pool: {pool_host}")
        
        # Restituisci config per il database del tenant
        tenant_config = {
            'host': pool_host,
            'user': pool_user,
            'password': pool_password,
            'port': pool_port,
            'database': tenant_database
        }
        
        return tenant_config, tenant_name
        
    except Error as e:
        print(f"❌ Errore connessione database common: {e}")
        return None
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def fix_clustering_parameters_direct():
    """
    Correzione diretta dei parametri di clustering nel database MySQL del tenant
    """
    connection = None
    try:
        # Prima ottieni le info del tenant dal database common
        tenant_info = get_tenant_database_info(tenant_id)
        if not tenant_info:
            return False
        
        tenant_config, tenant_name = tenant_info
        
        print(f"\n🔌 Connessione al database del tenant...")
        print(f"   Host: {tenant_config['host']}")
        print(f"   Database: {tenant_config['database']}")
        
        # Se l'host è localhost, usa le credenziali del database common
        if tenant_config['host'] == 'localhost':
            print(f"🔄 Host localhost rilevato, uso credenziali del database common...")
            common_config = get_database_config()
            tenant_config['host'] = common_config['host']
            tenant_config['user'] = common_config['user'] 
            tenant_config['password'] = common_config['password']
            print(f"   Host corretto: {tenant_config['host']}")
        
        connection = mysql.connector.connect(**tenant_config)
        cursor = connection.cursor()
        
        print(f"\n🔍 Ricerca parametri clustering...")
        
        # Prima controlla se la tabella soglie esiste
        cursor.execute("SHOW TABLES LIKE 'soglie'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            print(f"❌ La tabella 'soglie' non esiste nel database {tenant_config['database']}")
            
            # Mostra le tabelle disponibili per debug
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            print(f"📋 Tabelle disponibili nel database {tenant_config['database']}:")
            for i, table in enumerate(tables, 1):
                print(f"    {i:2d}. {table}")
            
            # Cerca la tabella soglie specificamente
            if 'soglie' in tables:
                print(f"✅ Tabella 'soglie' trovata!")
            else:
                print(f"❌ Tabella 'soglie' non trovata!")
            
            return False
        
        print(f"✅ Tabella 'soglie' trovata! Analizzo la struttura...")
        
        # Mostra la struttura della tabella soglie
        cursor.execute("DESCRIBE soglie")
        columns = cursor.fetchall()
        print(f"📋 Struttura tabella soglie:")
        for col in columns:
            print(f"   - {col[0]} ({col[1]}) {col[3]} {col[5] if col[5] else ''}")
        
        # Mostra tutto il contenuto della tabella soglie per capire la struttura
        cursor.execute("SELECT * FROM soglie LIMIT 10")
        rows = cursor.fetchall()
        print(f"\n� Contenuto tabella soglie (prime 10 righe):")
        for i, row in enumerate(rows, 1):
            print(f"   {i}. {row}")
        
        # Cerca parametri HDBSCAN nella tabella soglie
        hdbscan_params = {}
        for param_name in new_params.keys():
            cursor.execute("SELECT * FROM soglie WHERE nome LIKE %s OR descrizione LIKE %s", 
                          (f'%{param_name}%', f'%{param_name}%'))
            matches = cursor.fetchall()
            if matches:
                print(f"🎯 Trovato parametro {param_name}:")
                for match in matches:
                    print(f"   ✅ {match}")
                    # Assumendo che la struttura sia (id, nome, valore, descrizione, ...)
                    if len(match) >= 3:
                        hdbscan_params[param_name] = str(match[2])  # valore nella 3a colonna
        
        current_params = hdbscan_params
        
        if not current_params:
            print(f"❌ Nessun parametro clustering trovato nel database {tenant_config['database']}")
            
            # Mostra le tabelle disponibili per debug
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            print(f"📋 Tabelle disponibili nel database {tenant_config['database']}:")
            for i, table in enumerate(tables, 1):
                print(f"    {i:2d}. {table}")
            
            # Se esiste la tabella engines, mostra il contenuto
            if 'engines' in tables:
                cursor.execute("SELECT DISTINCT engine_name FROM engines")
                engines = [engine[0] for engine in cursor.fetchall()]
                print(f"🔧 Engine disponibili: {', '.join(engines)}")
            else:
                print(f"⚠️  La tabella 'engines' non esiste in questo database!")
                # Cerca tabelle simili che potrebbero contenere configurazioni
                config_tables = [t for t in tables if any(keyword in t.lower() for keyword in ['config', 'setting', 'param', 'engine'])]
                if config_tables:
                    print(f"🔍 Tabelle di configurazione possibili: {', '.join(config_tables)}")
            
            return False
        
        print(f"\n📊 PARAMETRI ATTUALI → NUOVI:")
        for param, value in current_params.items():
            if param in new_params:
                change_direction = "⬆" if float(new_params[param]) > float(value) else "⬇"
                change_percent = abs((float(new_params[param]) - float(value)) / float(value) * 100)
                if new_params[param] == value:
                    print(f"   - {param}: {value} → {new_params[param]} (✅ OK)")
                else:
                    print(f"   - {param}: {value} → {new_params[param]} ({change_direction} {change_percent:.0f}%)")
        
        # Aggiorna i parametri nella tabella soglie
        print(f"\n🔄 Aggiornamento parametri nella tabella soglie...")
        updated_count = 0
        
        for param_name, new_value in new_params.items():
            if param_name in current_params:
                # Cerca il record specifico per questo parametro
                cursor.execute("SELECT * FROM soglie WHERE nome LIKE %s", (f'%{param_name}%',))
                param_record = cursor.fetchone()
                
                if param_record and len(param_record) >= 1:
                    # Aggiorna usando l'ID del record (assumendo che la prima colonna sia l'ID)
                    record_id = param_record[0]
                    update_query = "UPDATE soglie SET valore = %s WHERE id = %s"
                    cursor.execute(update_query, (str(new_value), record_id))
                    updated_count += 1
                    print(f"   ✅ {param_name}: {current_params[param_name]} → {new_value} (ID: {record_id})")
                else:
                    # Se non esiste, crealo
                    insert_query = "INSERT INTO soglie (nome, valore, descrizione) VALUES (%s, %s, %s)"
                    cursor.execute(insert_query, (param_name, str(new_value), f"Parametro HDBSCAN {param_name}"))
                    updated_count += 1
                    print(f"   ➕ {param_name}: NUOVO → {new_value}")
            else:
                # Parametro non esistente, crealo
                insert_query = "INSERT INTO soglie (nome, valore, descrizione) VALUES (%s, %s, %s)"
                cursor.execute(insert_query, (param_name, str(new_value), f"Parametro HDBSCAN {param_name}"))
                updated_count += 1
                print(f"   ➕ {param_name}: NUOVO → {new_value}")
        
        connection.commit()
        print(f"\n🎯 CORREZIONE COMPLETATA!")
        print(f"   📊 Parametri aggiornati: {updated_count}")
        print(f"   🆔 Tenant: {tenant_name} ({tenant_id})")
        print(f"   💾 Database: {tenant_config['database']}")
        
        return True
        
    except Error as e:
        print(f"❌ Errore aggiornamento database: {e}")
        return False
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    print("🔧 AVVIO CORREZIONE DIRETTA PARAMETRI CLUSTERING")
    print("🔧 CORREZIONE DIRETTA PARAMETRI CLUSTERING - HUMANITAS")
    print("=" * 70)
    print(f"📊 PARAMETRI OTTIMALI:")
    for param, value in new_params.items():
        print(f"   - {param}: {value}")
    
    success = fix_clustering_parameters_direct()
    
    if success:
        print("\n✅ CORREZIONE COMPLETATA CON SUCCESSO!")
        print("💡 I parametri sono ora ottimizzati per permettere la formazione di cluster.")
        print("🔄 Riavvia il processo di clustering per vedere i rappresentanti.")
    else:
        print("\n❌ CORREZIONE FALLITA!")
        print("🔍 Controlla i messaggi di errore sopra per più dettagli.")
