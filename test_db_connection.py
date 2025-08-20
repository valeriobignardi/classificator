#!/usr/bin/env python3
"""
Test connessione database per debug
Autore: GitHub Copilot
Creato: 2025-08-08
"""

import yaml
import mysql.connector
from mysql.connector import Error

def test_database_connection():
    """
    Testa la connessione al database e conta le sessioni
    """
    try:
        # Carica configurazione
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        db_config = config.get('database', {})
        print("🔧 Configurazione database:")
        print(f"  Host: {db_config.get('host', 'MISSING')}")
        print(f"  Port: {db_config.get('port', 'MISSING')}")
        print(f"  User: {db_config.get('user', 'MISSING')}")
        print(f"  Database: {db_config.get('database', 'MISSING')}")
        
        # Test connessione
        print("\n🔌 Tentativo connessione...")
        connection = mysql.connector.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'], 
            password=db_config['password'],
            database=db_config['database']
        )
        
        if connection.is_connected():
            print("✅ Connessione riuscita!")
            
            cursor = connection.cursor()
            
            # Verifica schema humanitas
            cursor.execute("SHOW DATABASES LIKE 'humanitas'")
            humanitas_db = cursor.fetchone()
            print(f"📋 Database 'humanitas': {'TROVATO' if humanitas_db else 'NON TROVATO'}")
            
            # Se esiste, conta le sessioni
            if humanitas_db:
                cursor.execute("USE humanitas")
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                print(f"📊 Tabelle in humanitas: {[t[0] for t in tables]}")
                
                # Cerca tabelle con sessioni/conversazioni
                for table in tables:
                    table_name = table[0]
                    if any(keyword in table_name.lower() for keyword in ['session', 'conversation', 'message', 'chat']):
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        print(f"  📈 {table_name}: {count} record")
            
            # Verifica anche schema common
            cursor.execute("USE common")
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print(f"📊 Tabelle in common: {[t[0] for t in tables]}")
            
            # Cerca tabelle con sessioni/conversazioni
            for table in tables:
                table_name = table[0]
                if any(keyword in table_name.lower() for keyword in ['session', 'conversation', 'message', 'chat']):
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    print(f"  📈 {table_name}: {count} record")
                    
        else:
            print("❌ Connessione fallita")
            
    except Error as e:
        print(f"❌ Errore MySQL: {e}")
    except Exception as e:
        print(f"❌ Errore generico: {e}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()
            print("🔌 Connessione chiusa")

if __name__ == "__main__":
    test_database_connection()
