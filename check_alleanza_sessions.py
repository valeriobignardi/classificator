#!/usr/bin/env python3
"""
Verifica conteggio sessioni per tenant Alleanza

Autore: Valerio Bignardi  
Data di creazione: 2025-08-29
Storia degli aggiornamenti:
- 2025-08-29: Verifica conteggio reale sessioni Alleanza
"""

import sys
import os

# Aggiungi percorsi
sys.path.append('.')

def check_alleanza_sessions():
    """
    Scopo: Verifica il numero reale di sessioni per Alleanza
    
    Output:
        - int: Numero di sessioni trovate
        
    Ultimo aggiornamento: 2025-08-29
    """
    print("🔍 VERIFICA SESSIONI ALLEANZA")
    print("=" * 40)
    
    try:
        # Prova diversi metodi per trovare le sessioni Alleanza
        
        # Metodo 1: Pipeline
        print("1️⃣ TEST VIA PIPELINE")
        try:
            from Pipeline.end_to_end_pipeline import EndToEndPipeline
            
            pipeline = EndToEndPipeline(tenant_slug='alleanza')
            sessioni = pipeline.estrai_sessioni(limit=None)
            print(f"   📊 Via pipeline: {len(sessioni)} sessioni")
            
            if len(sessioni) > 0:
                return len(sessioni)
                
        except Exception as e:
            print(f"   ❌ Errore pipeline: {str(e)[:100]}...")
        
        # Metodo 2: Database MySQL diretto  
        print("\n2️⃣ TEST VIA DATABASE MYSQL")
        try:
            import mysql.connector
            import yaml
            
            # Carica config
            with open('config.yaml', 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            db_config = config['database']
            
            # Connessione MySQL
            conn = mysql.connector.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database']
            )
            
            cursor = conn.cursor()
            
            # Query per trovare Alleanza
            queries = [
                "SELECT COUNT(*) FROM conversations WHERE customer LIKE '%alleanza%'",
                "SELECT COUNT(*) FROM conversations WHERE customer LIKE '%Alleanza%'",
                "SELECT COUNT(*) FROM conversations WHERE customer = 'alleanza'",
                "SELECT COUNT(*) FROM conversations WHERE customer = 'Alleanza'",
                "SELECT DISTINCT customer FROM conversations WHERE customer LIKE '%alleanza%' LIMIT 10"
            ]
            
            for i, query in enumerate(queries, 1):
                try:
                    cursor.execute(query)
                    result = cursor.fetchall()
                    
                    if 'DISTINCT' in query:
                        print(f"   🔍 Query {i} (nomi clienti): {result}")
                    else:
                        count = result[0][0] if result else 0
                        print(f"   📊 Query {i}: {count} sessioni")
                        
                        if count > 0:
                            cursor.close()
                            conn.close()
                            return count
                            
                except Exception as e:
                    print(f"   ❌ Query {i} fallita: {e}")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"   ❌ Errore database: {str(e)[:100]}...")
        
        # Metodo 3: MongoDB (se disponibile)
        print("\n3️⃣ TEST VIA MONGODB")
        try:
            from mongo_classification_reader import MongoClassificationReader
            
            mongo_reader = MongoClassificationReader()
            sessioni = mongo_reader.get_all_sessions('alleanza')
            print(f"   📊 Via MongoDB: {len(sessioni)} classificazioni esistenti")
            
        except Exception as e:
            print(f"   ❌ Errore MongoDB: {str(e)[:100]}...")
        
        print("\n⚠️ Nessun metodo ha trovato sessioni per 'alleanza'")
        return 0
        
    except Exception as e:
        print(f"❌ Errore generale: {e}")
        return 0

def verify_tenant_names():
    """
    Scopo: Verifica i nomi dei tenant disponibili
    
    Output:
        - list: Lista dei tenant trovati
        
    Ultimo aggiornamento: 2025-08-29
    """
    print("\n🔍 VERIFICA NOMI TENANT DISPONIBILI")
    print("=" * 45)
    
    try:
        import mysql.connector
        import yaml
        
        # Carica config
        with open('config.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        db_config = config['database']
        
        # Connessione MySQL
        conn = mysql.connector.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        
        cursor = conn.cursor()
        
        # Query per trovare tutti i clienti
        cursor.execute("SELECT DISTINCT customer, COUNT(*) as count FROM conversations GROUP BY customer ORDER BY count DESC LIMIT 20")
        result = cursor.fetchall()
        
        print("📋 TENANT DISPONIBILI (top 20):")
        tenant_names = []
        for i, (customer, count) in enumerate(result, 1):
            print(f"   {i:2d}. {customer}: {count} sessioni")
            tenant_names.append(customer)
            
            # Controlla se contiene "alleanza"
            if 'alleanza' in customer.lower():
                print(f"       ⭐ POSSIBILE MATCH per Alleanza!")
        
        cursor.close()
        conn.close()
        
        return tenant_names
        
    except Exception as e:
        print(f"❌ Errore verifica tenant: {e}")
        return []

def main():
    """
    Scopo: Esegue verifica completa per Alleanza
    
    Output:
        - None
        
    Ultimo aggiornamento: 2025-08-29
    """
    print("🚨 VERIFICA COMPLETA TENANT ALLEANZA")
    print("=" * 50)
    
    # Verifica sessioni Alleanza
    count = check_alleanza_sessions()
    
    # Verifica nomi tenant disponibili
    tenants = verify_tenant_names()
    
    print(f"\n🎯 RISULTATI:")
    print(f"📊 Sessioni trovate per Alleanza: {count}")
    print(f"📋 Tenant totali nel database: {len(tenants)}")
    
    # Suggerimenti
    print(f"\n💡 POSSIBILI CAUSE DISCREPANZA 17 vs 15:")
    if count == 0:
        print(f"   • Nome tenant diverso (non 'alleanza')")
        print(f"   • Tenant potrebbe essere 'Alleanza Salute', 'Alleanza Assicurazioni', etc.")
        print(f"   • Controllare la lista dei tenant sopra")
    elif count == 15:
        print(f"   • L'utente ha ragione: sono effettivamente 15 sessioni")
        print(f"   • Sistema potrebbe contare duplicati o sessioni di test")
    elif count == 17:
        print(f"   • Sistema corretto: sono 17 sessioni nel database")
        print(f"   • L'utente potrebbe vedere un subset filtrato")
    else:
        print(f"   • Trovate {count} sessioni - diverso da entrambi i valori")
        print(f"   • Possibile problema di connessione o cache")

if __name__ == "__main__":
    main()
