#!/usr/bin/env python3
"""
Script per elencare tutte le tabelle nel database MySQL
Autore: Valerio Bignardi
Data: 2025-01-22
"""

import yaml
import mysql.connector
from mysql.connector import Error

def load_config():
    """Carica la configurazione dal file config.yaml"""
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config['database']

def list_all_tables():
    """Elenca tutte le tabelle nel database"""
    try:
        config = load_config()
        print(f"🔌 Connessione al database MySQL...")
        print(f"   Host: {config['host']}")
        print(f"   Database: {config['database']}")
        
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
        # Elenca tutte le tabelle
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        print(f"\n📋 TABELLE NEL DATABASE '{config['database']}':")
        print("=" * 50)
        for i, table in enumerate(tables, 1):
            print(f"{i:2d}. {table[0]}")
        
        print(f"\n📊 Totale tabelle: {len(tables)}")
        
        # Per ogni tabella, mostra la struttura
        print("\n🔍 STRUTTURA TABELLE:")
        print("=" * 50)
        for table in tables:
            table_name = table[0]
            print(f"\n📄 TABELLA: {table_name}")
            cursor.execute(f"DESCRIBE {table_name}")
            columns = cursor.fetchall()
            for col in columns:
                print(f"   - {col[0]} ({col[1]}) {col[3]} {col[5] if col[5] else ''}")
        
    except Error as e:
        print(f"❌ Errore database: {e}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            print("\n🔒 Connessione al database chiusa.")

if __name__ == "__main__":
    print("🔍 ANALISI STRUTTURA DATABASE")
    print("=" * 40)
    list_all_tables()
