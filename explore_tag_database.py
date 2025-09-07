#!/usr/bin/env python3
"""
Script per cercare e modificare i parametri HDBSCAN nella tabella soglie
del database locale TAG.

Autore: Valerio Bignardi
Data: 2025-01-22
Ultimo aggiornamento: 2025-01-22 - Database TAG locale
"""

import yaml
import mysql.connector
from mysql.connector import Error

# Parametri di clustering ottimali 
new_params = {
    'min_cluster_size': '5',         # Era probabilmente troppo alto
    'min_samples': '3',              # Era probabilmente troppo alto
    'cluster_selection_epsilon': '0.1',  # Era probabilmente troppo restrittivo
    'alpha': '1.0',                  # Per stabilità
    'cluster_selection_method': 'eom' # Manteniamo EOM
}

def get_tag_database_config():
    """Carica la configurazione del database TAG dal file config.yaml"""
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config['tag_database']

def explore_tag_database():
    """Esplora il database TAG per trovare la tabella soglie e i parametri HDBSCAN"""
    
    try:
        config = get_tag_database_config()
        print(f"🔌 Connessione al database TAG...")
        print(f"   Host: {config['host']}")
        print(f"   Database: {config['database']}")
        
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
        print(f"\n📋 TABELLE NEL DATABASE TAG:")
        print("=" * 50)
        
        # Elenca tutte le tabelle
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        
        for i, table in enumerate(tables, 1):
            print(f"{i:2d}. {table}")
        
        # Cerca la tabella soglie
        if 'soglie' in tables:
            print(f"\n🎯 TABELLA SOGLIE TROVATA!")
            print("=" * 40)
            
            # Descrivi la struttura
            cursor.execute("DESCRIBE soglie")
            columns = cursor.fetchall()
            print(f"📋 Struttura tabella soglie:")
            for col in columns:
                print(f"   - {col[0]} ({col[1]}) {col[3]} {col[5] if col[5] else ''}")
            
            # Mostra tutto il contenuto
            cursor.execute("SELECT * FROM soglie")
            rows = cursor.fetchall()
            
            print(f"\n📊 CONTENUTO TABELLA SOGLIE ({len(rows)} righe):")
            print("-" * 60)
            
            if rows:
                for i, row in enumerate(rows, 1):
                    print(f"{i:3d}. {row}")
            else:
                print("❌ Nessun dato nella tabella soglie")
            
            # Cerca specificamente parametri HDBSCAN
            print(f"\n🔍 RICERCA PARAMETRI HDBSCAN:")
            print("-" * 40)
            
            clustering_terms = ['HDBSCAN', 'clustering', 'cluster', 'min_cluster_size', 'min_samples', 'epsilon', 'alpha']
            found_clustering = False
            
            for term in clustering_terms:
                # Cerca in tutte le colonne di testo
                text_columns = [col[0] for col in columns if 'text' in col[1] or 'varchar' in col[1]]
                for col in text_columns:
                    try:
                        cursor.execute(f"SELECT * FROM soglie WHERE {col} LIKE %s", (f'%{term}%',))
                        matches = cursor.fetchall()
                        if matches:
                            found_clustering = True
                            print(f"🎯 TROVATO '{term}' in colonna {col}:")
                            for match in matches:
                                print(f"   ✅ {match}")
                    except Exception as e:
                        pass  # Ignora errori di colonne incompatibili
            
            if not found_clustering:
                print("❌ Nessun parametro HDBSCAN trovato nella tabella soglie")
        
        else:
            print(f"\n❌ TABELLA SOGLIE NON TROVATA!")
            print("🔍 Tabelle disponibili con nomi simili:")
            similar_tables = [t for t in tables if any(keyword in t.lower() for keyword in ['soglia', 'param', 'config', 'setting'])]
            if similar_tables:
                for table in similar_tables:
                    print(f"   - {table}")
            else:
                print("   Nessuna tabella simile trovata")
        
        # Cerca in tutte le tabelle per termini HDBSCAN
        print(f"\n🔍 RICERCA GLOBALE HDBSCAN IN TUTTE LE TABELLE:")
        print("=" * 60)
        
        global_found = False
        for table_name in tables:
            try:
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                text_columns = [col[0] for col in columns if 'text' in col[1] or 'varchar' in col[1] or 'json' in col[1]]
                
                for col in text_columns:
                    cursor.execute(f"SELECT * FROM {table_name} WHERE {col} LIKE %s LIMIT 3", ('%HDBSCAN%',))
                    matches = cursor.fetchall()
                    if matches:
                        global_found = True
                        print(f"🎯 HDBSCAN trovato in {table_name}.{col}:")
                        for match in matches:
                            print(f"   ✅ {match}")
                    
                    # Cerca anche per 'cluster'
                    cursor.execute(f"SELECT * FROM {table_name} WHERE {col} LIKE %s LIMIT 3", ('%cluster%',))
                    matches = cursor.fetchall()
                    if matches:
                        global_found = True
                        print(f"🎯 'cluster' trovato in {table_name}.{col}:")
                        for match in matches:
                            print(f"   ✅ {match}")
            except:
                pass  # Ignora errori
        
        if not global_found:
            print("❌ Nessun riferimento a HDBSCAN o cluster trovato nel database TAG")
        
    except Error as e:
        print(f"❌ Errore database: {e}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            print("\n🔒 Connessione al database TAG chiusa.")

if __name__ == "__main__":
    print("🔍 ESPLORAZIONE DATABASE TAG - TABELLA SOGLIE")
    print("=" * 60)
    explore_tag_database()
