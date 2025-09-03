#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per aggiornare schema tabella soglie con parametri unificati

Autore: Valerio Bignardi
Data: 2025-01-03
"""

import mysql.connector
from mysql.connector import Error
import yaml

def load_config():
    """Carica configurazione database"""
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config['tag_database']

def check_current_schema():
    """Controlla schema attuale tabella soglie"""
    try:
        db_config = load_config()
        connection = mysql.connector.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            autocommit=True
        )
        
        cursor = connection.cursor()
        
        # Mostra struttura attuale
        cursor.execute("DESCRIBE soglie")
        columns = cursor.fetchall()
        
        print("🔍 SCHEMA ATTUALE TABELLA 'soglie':")
        print("-" * 70)
        for col in columns:
            field, type_, null, key, default, extra = col
            print(f"  {field:35} {type_:20} {null:5} {key:5} {default}")
        
        cursor.close()
        connection.close()
        
        return [col[0] for col in columns]  # Lista nomi colonne
        
    except Error as e:
        print(f"❌ Errore MySQL: {e}")
        return []
    except Exception as e:
        print(f"❌ Errore generico: {e}")
        return []

def update_schema_unified():
    """Aggiorna schema per parametri unificati"""
    try:
        db_config = load_config()
        connection = mysql.connector.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            autocommit=True
        )
        
        cursor = connection.cursor()
        
        # Lista delle colonne da aggiungere
        new_columns = [
            # HDBSCAN Parameters
            ("min_cluster_size", "INT DEFAULT 5"),
            ("min_samples", "INT DEFAULT 3"), 
            ("cluster_selection_epsilon", "DECIMAL(10,4) DEFAULT 0.12"),
            ("metric", "VARCHAR(50) DEFAULT 'euclidean'"),
            ("cluster_selection_method", "VARCHAR(50) DEFAULT 'leaf'"),
            ("alpha", "DECIMAL(5,3) DEFAULT 0.8"),
            ("max_cluster_size", "INT DEFAULT 0"),
            ("allow_single_cluster", "BOOLEAN DEFAULT FALSE"),
            ("only_user", "BOOLEAN DEFAULT TRUE"),
            
            # UMAP Parameters
            ("use_umap", "BOOLEAN DEFAULT FALSE"),
            ("umap_n_neighbors", "INT DEFAULT 10"),
            ("umap_min_dist", "DECIMAL(10,4) DEFAULT 0.05"),
            ("umap_metric", "VARCHAR(50) DEFAULT 'euclidean'"),
            ("umap_n_components", "INT DEFAULT 3"),
            ("umap_random_state", "INT DEFAULT 42")
        ]
        
        print("\n🔧 AGGIORNAMENTO SCHEMA TABELLA 'soglie':")
        print("-" * 70)
        
        for col_name, col_definition in new_columns:
            try:
                alter_query = f"ALTER TABLE soglie ADD COLUMN {col_name} {col_definition}"
                cursor.execute(alter_query)
                print(f"  ✅ Aggiunta colonna: {col_name}")
                
            except Error as e:
                if "Duplicate column name" in str(e):
                    print(f"  ⚠️  Colonna già esistente: {col_name}")
                else:
                    print(f"  ❌ Errore aggiungendo {col_name}: {e}")
        
        print(f"\n🎯 AGGIORNAMENTO COMPLETATO")
        
        cursor.close()
        connection.close()
        
    except Error as e:
        print(f"❌ Errore MySQL: {e}")
    except Exception as e:
        print(f"❌ Errore generico: {e}")

if __name__ == "__main__":
    print("=" * 70)
    print("🛠️  AGGIORNAMENTO SCHEMA TABELLA SOGLIE")
    print("=" * 70)
    
    # 1. Mostra schema attuale
    existing_columns = check_current_schema()
    
    # 2. Aggiorna schema
    if existing_columns:
        print(f"\n📊 Trovate {len(existing_columns)} colonne esistenti")
        update_schema_unified()
        
        # 3. Verifica finale
        print("\n" + "=" * 70)
        print("🔍 VERIFICA FINALE SCHEMA:")
        check_current_schema()
    else:
        print("⚠️ Impossibile accedere al database")
