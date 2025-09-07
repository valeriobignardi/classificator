#!/usr/bin/env python3
"""
Script per esplorare le tabelle di configurazione del tenant
per trovare dove sono memorizzati i parametri di clustering.

Autore: Valerio Bignardi
Data: 2025-01-22
"""

import yaml
import mysql.connector
from mysql.connector import Error

tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"

def get_database_config():
    """Carica la configurazione del database common dal file config.yaml"""
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config['database']

def get_tenant_database_info(tenant_id):
    """Ottiene le informazioni del database per il tenant specifico"""
    config = get_database_config()
    
    try:
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
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
            return None
        
        tenant_name, tenant_database, pool_host, pool_user, pool_password, pool_port, pool_database = result
        
        tenant_config = {
            'host': pool_host,
            'user': pool_user,
            'password': pool_password,
            'port': pool_port,
            'database': tenant_database
        }
        
        # Se l'host è localhost, usa le credenziali del database common
        if tenant_config['host'] == 'localhost':
            common_config = get_database_config()
            tenant_config['host'] = common_config['host']
            tenant_config['user'] = common_config['user'] 
            tenant_config['password'] = common_config['password']
        
        return tenant_config, tenant_name
        
    except Error as e:
        print(f"❌ Errore: {e}")
        return None
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def explore_config_tables():
    """Esplora le tabelle di configurazione per trovare i parametri clustering"""
    
    tenant_info = get_tenant_database_info(tenant_id)
    if not tenant_info:
        return
    
    tenant_config, tenant_name = tenant_info
    
    try:
        connection = mysql.connector.connect(**tenant_config)
        cursor = connection.cursor()
        
        print(f"🔍 ESPLORAZIONE DATABASE {tenant_config['database']} - {tenant_name}")
        print("=" * 70)
        
        # Lista delle tabelle da esplorare
        config_tables = ['ai_engines', 'app_config', 'project_ai_config', 'tools']
        
        for table_name in config_tables:
            print(f"\n📄 TABELLA: {table_name}")
            print("-" * 40)
            
            try:
                # Descrivi la struttura della tabella
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                print(f"📋 Struttura:")
                for col in columns:
                    print(f"   - {col[0]} ({col[1]}) {col[3]} {col[5] if col[5] else ''}")
                
                # Mostra alcuni dati di esempio
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
                rows = cursor.fetchall()
                if rows:
                    print(f"\n📊 Dati di esempio:")
                    for i, row in enumerate(rows, 1):
                        print(f"   {i}. {row}")
                else:
                    print(f"\n📊 Nessun dato nella tabella")
                
                # Cerca specificamente termini legati al clustering
                clustering_terms = ['HDBSCAN', 'clustering', 'cluster', 'min_cluster_size', 'min_samples']
                for term in clustering_terms:
                    try:
                        # Cerca nelle colonne di testo
                        text_columns = [col[0] for col in columns if 'text' in col[1] or 'varchar' in col[1]]
                        for col in text_columns:
                            cursor.execute(f"SELECT * FROM {table_name} WHERE {col} LIKE %s", (f'%{term}%',))
                            matches = cursor.fetchall()
                            if matches:
                                print(f"\n🎯 TROVATO '{term}' in {col}:")
                                for match in matches:
                                    print(f"   ✅ {match}")
                    except:
                        pass  # Ignora errori di colonne non compatibili
                
            except Error as e:
                print(f"   ❌ Errore accesso tabella {table_name}: {e}")
        
        # Cerca anche in tutte le tabelle per termini di clustering
        print(f"\n🔍 RICERCA GLOBALE TERMINI CLUSTERING")
        print("-" * 50)
        
        cursor.execute("SHOW TABLES")
        all_tables = [table[0] for table in cursor.fetchall()]
        
        clustering_found = False
        for table_name in all_tables:
            try:
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                text_columns = [col[0] for col in columns if 'text' in col[1] or 'varchar' in col[1] or 'json' in col[1]]
                
                for col in text_columns:
                    cursor.execute(f"SELECT * FROM {table_name} WHERE {col} LIKE %s LIMIT 3", ('%HDBSCAN%',))
                    matches = cursor.fetchall()
                    if matches:
                        clustering_found = True
                        print(f"🎯 HDBSCAN trovato in {table_name}.{col}:")
                        for match in matches:
                            print(f"   ✅ {match}")
            except:
                pass  # Ignora errori
        
        if not clustering_found:
            print("❌ Nessun riferimento a HDBSCAN trovato nel database del tenant")
            print("💡 I parametri potrebbero essere memorizzati altrove o gestiti diversamente")
        
    except Error as e:
        print(f"❌ Errore: {e}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    explore_config_tables()
