#!/usr/bin/env python3
"""
============================================================================
Debug Rappresentanti - Tracciamento Completo del Processo
============================================================================

Autore: Valerio Bignardi
Data creazione: 2025-09-07
Ultima modifica: 2025-09-07

Descrizione:
    Script di debug completo per tracciare il processo dei rappresentanti
    dalla selezione durante il clustering fino al salvataggio in MongoDB.
    Include debug dettagliato per identificare dove si perdono i rappresentanti.

Fasi tracciate:
    1. Verifica dati esistenti in MongoDB
    2. Analisi processo clustering HDBSCAN
    3. Tracciamento selezione rappresentanti
    4. Verifica salvataggio MongoDB
    5. Controllo metadati e flag

============================================================================
"""

import sys
import os
import yaml
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from collections import defaultdict
import json
import mysql.connector
from config_loader import load_config

# Aggiungi path progetto
sys.path.append('/home/ubuntu/classificatore')

class RepresentativesDebugger:
    """
    Debugger completo per il processo rappresentanti
    
    Scopo:
        Traccia ogni step del processo rappresentanti per identificare
        dove si perdono i dati tra clustering e salvataggio MongoDB
        
    Metodi:
        debug_mongodb_data: Analizza dati esistenti MongoDB
        debug_clustering_process: Analizza processo clustering
        debug_representatives_selection: Debug selezione rappresentanti
        debug_save_process: Debug salvataggio MongoDB
        
    Data ultima modifica: 2025-09-07
    """
    
    def __init__(self):
        """
        Inizializza debugger con configurazioni
        
        Data ultima modifica: 2025-09-07
        """
        self.config = self._load_config()
        self.tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
        self.tenant_name = "Humanitas"
        self.collection_name = f"humanitas_{self.tenant_id}"
        
        print("🔍 DEBUG RAPPRESENTANTI - TRACCIAMENTO COMPLETO")
        print("="*70)
        print(f"👤 Tenant: {self.tenant_name} ({self.tenant_id})")
        print(f"📊 Collection: {self.collection_name}")
        print(f"🕐 Timestamp: {datetime.now()}")
    
    def _load_config(self) -> dict:
        """
        Carica configurazione completa
        
        Returns:
            dict: Configurazione sistema
            
        Data ultima modifica: 2025-09-07
        """
        try:
            config_path = '/home/ubuntu/classificatore/config.yaml'
    return load_config()
        except Exception as e:
            print(f"❌ Errore caricamento config: {e}")
            return {}
    
    def connect_mongodb(self) -> tuple:
        """
        Connette a MongoDB
        
        Returns:
            tuple: (client, db, collection)
            
        Data ultima modifica: 2025-09-07
        """
        try:
            mongo_url = self.config.get('mongodb', {}).get('url', 'mongodb://localhost:27017')
            client = MongoClient(mongo_url)
            db = client['classificazioni']
            collection = db[self.collection_name]
            
            # Test connessione
            client.admin.command('ping')
            print(f"✅ MongoDB connesso: {mongo_url}")
            
            return client, db, collection
            
        except Exception as e:
            print(f"❌ Errore connessione MongoDB: {e}")
            return None, None, None
    
    def connect_mysql(self):
        """
        Connette al database MySQL TAG
        
        Returns:
            mysql.connector connection
            
        Data ultima modifica: 2025-09-07
        """
        try:
            tag_db_config = self.config['tag_database']
            conn = mysql.connector.connect(**tag_db_config)
            print(f"✅ MySQL TAG connesso: {tag_db_config['host']}")
            return conn
        except Exception as e:
            print(f"❌ Errore connessione MySQL: {e}")
            return None
    
    def debug_mongodb_existing_data(self, collection):
        """
        Debug: Analizza dati esistenti in MongoDB
        
        Args:
            collection: Collezione MongoDB
            
        Data ultima modifica: 2025-09-07
        """
        print("\\n🔍 STEP 1: ANALISI DATI ESISTENTI MONGODB")
        print("-"*50)
        
        try:
            # Conta documenti totali
            total_docs = collection.count_documents({'tenant_id': self.tenant_id})
            print(f"📊 Documenti totali: {total_docs}")
            
            # Cerca documenti con flag rappresentanti (varie forme)
            queries = [
                {'tenant_id': self.tenant_id, 'metadata.is_representative': True},
                {'tenant_id': self.tenant_id, 'metadata.representative': True},
                {'tenant_id': self.tenant_id, 'is_representative': True},
                {'tenant_id': self.tenant_id, 'representative': True},
                {'tenant_id': self.tenant_id, 'cluster_representative': True},
                {'tenant_id': self.tenant_id, 'metadata.cluster_representative': True}
            ]
            
            for i, query in enumerate(queries, 1):
                count = collection.count_documents(query)
                field_path = list(query.keys())[1]  # Escludi tenant_id
                print(f"   🎯 Query {i} [{field_path}]: {count} documenti")
                
                if count > 0:
                    sample = collection.find_one(query)
                    print(f"      📄 Esempio: session_id={sample.get('session_id')}")
                    print(f"      📋 Metadata: {sample.get('metadata', {})}")
            
            # Cerca pattern di cluster_id
            pipeline = [
                {'$match': {'tenant_id': self.tenant_id}},
                {'$group': {
                    '_id': '$metadata.cluster_id',
                    'count': {'$sum': 1}
                }},
                {'$sort': {'count': -1}},
                {'$limit': 10}
            ]
            
            cluster_stats = list(collection.aggregate(pipeline))
            print(f"\\n📊 TOP 10 CLUSTER IDs:")
            for stat in cluster_stats:
                cluster_id = stat['_id']
                count = stat['count']
                print(f"   🔢 Cluster {cluster_id}: {count} documenti")
            
            # Verifica classification_method
            pipeline = [
                {'$match': {'tenant_id': self.tenant_id}},
                {'$group': {
                    '_id': '$classification_method',
                    'count': {'$sum': 1}
                }}
            ]
            
            method_stats = list(collection.aggregate(pipeline))
            print(f"\\n🔧 METODI DI CLASSIFICAZIONE:")
            for stat in method_stats:
                method = stat['_id']
                count = stat['count']
                print(f"   ⚙️ {method}: {count} documenti")
            
        except Exception as e:
            print(f"❌ Errore debug MongoDB: {e}")
    
    def debug_mysql_clustering_history(self, mysql_conn):
        """
        Debug: Analizza cronologia clustering in MySQL
        
        Args:
            mysql_conn: Connessione MySQL
            
        Data ultima modifica: 2025-09-07
        """
        print("\\n🔍 STEP 2: ANALISI CRONOLOGIA CLUSTERING MYSQL")
        print("-"*50)
        
        try:
            cursor = mysql_conn.cursor(dictionary=True)
            
            # Verifica tabella clustering_test_results
            cursor.execute("SHOW TABLES LIKE 'clustering_test_results'")
            if cursor.fetchone():
                print("✅ Tabella clustering_test_results trovata")
                
                # Ultimi test di clustering
                cursor.execute("""
                    SELECT tenant_name, test_date, total_sessions, 
                           num_clusters, num_outliers, num_representatives,
                           algoritmo_clustering, config_clustering
                    FROM clustering_test_results 
                    WHERE tenant_name LIKE '%Humanitas%' OR tenant_name LIKE '%humanitas%'
                    ORDER BY test_date DESC 
                    LIMIT 5
                """)
                
                results = cursor.fetchall()
                if results:
                    print(f"📊 Ultimi {len(results)} test clustering Humanitas:")
                    for i, result in enumerate(results, 1):
                        print(f"   🧪 Test {i}:")
                        print(f"      📅 Data: {result['test_date']}")
                        print(f"      📊 Sessioni: {result['total_sessions']}")
                        print(f"      🔢 Cluster: {result['num_clusters']}")
                        print(f"      🚨 Outlier: {result['num_outliers']}")
                        print(f"      🎯 Rappresentanti: {result['num_representatives']}")
                        print(f"      ⚙️ Algoritmo: {result['algoritmo_clustering']}")
                        print()
                else:
                    print("❌ Nessun test clustering trovato per Humanitas")
            else:
                print("❌ Tabella clustering_test_results non trovata")
            
            # Verifica configurazioni tenant
            cursor.execute("""
                SELECT tenant_id, tenant_name, llm_config 
                FROM engines 
                WHERE tenant_id = %s
            """, (self.tenant_id,))
            
            tenant_config = cursor.fetchone()
            if tenant_config:
                print(f"\\n🏢 CONFIGURAZIONE TENANT:")
                print(f"   👤 Nome: {tenant_config['tenant_name']}")
                print(f"   🆔 ID: {tenant_config['tenant_id']}")
                
                llm_config = tenant_config['llm_config']
                if llm_config:
                    try:
                        config_data = json.loads(llm_config)
                        print(f"   ⚙️ LLM Config: {json.dumps(config_data, indent=2)}")
                    except:
                        print(f"   ⚙️ LLM Config (raw): {llm_config}")
                else:
                    print("   ❌ Nessuna configurazione LLM")
            else:
                print(f"❌ Tenant {self.tenant_id} non trovato in engines")
            
            cursor.close()
            
        except Exception as e:
            print(f"❌ Errore debug MySQL: {e}")
    
    def debug_clustering_code_paths(self):
        """
        Debug: Analizza i path del codice clustering
        
        Data ultima modifica: 2025-09-07
        """
        print("\\n🔍 STEP 3: ANALISI CODICE CLUSTERING")
        print("-"*50)
        
        # Cerca file di clustering
        clustering_files = [
            '/home/ubuntu/classificatore/ClusteringEngine/hdbscan_clustering.py',
            '/home/ubuntu/classificatore/ClusteringEngine/intelligent_clustering.py',
            '/home/ubuntu/classificatore/ClusteringEngine/cluster_representatives.py',
            '/home/ubuntu/classificatore/ClusteringEngine/clustering_engine.py'
        ]
        
        for file_path in clustering_files:
            if os.path.exists(file_path):
                print(f"✅ Trovato: {file_path}")
                
                # Cerca pattern rappresentanti nel codice
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Pattern da cercare
                    patterns = [
                        'is_representative',
                        'representative',
                        'cluster_representative',
                        'representative_selection',
                        'select_representatives',
                        'find_representatives'
                    ]
                    
                    found_patterns = []
                    for pattern in patterns:
                        if pattern in content:
                            # Conta occorrenze
                            count = content.count(pattern)
                            found_patterns.append(f"{pattern}({count}x)")
                    
                    if found_patterns:
                        print(f"   🎯 Pattern trovati: {', '.join(found_patterns)}")
                    else:
                        print(f"   ❌ Nessun pattern rappresentanti trovato")
                        
                except Exception as e:
                    print(f"   ❌ Errore lettura file: {e}")
            else:
                print(f"❌ File non trovato: {file_path}")
    
    def debug_specific_sessions(self, collection):
        """
        Debug: Analizza sessioni specifiche per tracciare metadati
        
        Args:
            collection: Collezione MongoDB
            
        Data ultima modifica: 2025-09-07
        """
        print("\\n🔍 STEP 4: ANALISI SESSIONI SPECIFICHE")
        print("-"*50)
        
        try:
            # Prendi campione di sessioni
            samples = list(collection.find({
                'tenant_id': self.tenant_id
            }).limit(5))
            
            print(f"📊 Analisi {len(samples)} sessioni campione:")
            
            for i, session in enumerate(samples, 1):
                print(f"\\n   📄 Sessione {i}: {session.get('session_id')}")
                print(f"      🏷️ Classification: {session.get('classification')}")
                print(f"      🔧 Method: {session.get('classification_method')}")
                print(f"      📊 Type: {session.get('classification_type')}")
                
                # Analizza metadata completo
                metadata = session.get('metadata', {})
                print(f"      📋 Metadata keys: {list(metadata.keys())}")
                
                # Cerca tutti i flag possibili
                representative_flags = [
                    'is_representative', 'representative', 'cluster_representative'
                ]
                
                for flag in representative_flags:
                    value = metadata.get(flag)
                    if value is not None:
                        print(f"         🎯 {flag}: {value}")
                
                # Cluster info
                cluster_id = metadata.get('cluster_id')
                if cluster_id is not None:
                    print(f"      🔢 Cluster ID: {cluster_id}")
                
                # Altri flag importanti
                other_flags = ['outlier', 'propagated']
                for flag in other_flags:
                    value = metadata.get(flag)
                    if value is not None:
                        print(f"         🏷️ {flag}: {value}")
                
                # Timestamp info
                classified_at = session.get('classified_at')
                if classified_at:
                    print(f"      🕐 Classified at: {classified_at}")
            
        except Exception as e:
            print(f"❌ Errore debug sessioni: {e}")
    
    def search_representatives_in_logs(self):
        """
        Debug: Cerca log di rappresentanti nei file di log
        
        Data ultima modifica: 2025-09-07
        """
        print("\\n🔍 STEP 5: RICERCA LOG RAPPRESENTANTI")
        print("-"*50)
        
        log_files = [
            '/home/ubuntu/classificatore/logging.log',
            '/home/ubuntu/classificatore/server.log',
            '/home/ubuntu/classificatore/esempi_api.log',
            '/home/ubuntu/classificatore/tracing.log'
        ]
        
        search_terms = [
            'representative', 'rappresentant', 'cluster_representative',
            'is_representative', 'select_representatives'
        ]
        
        for log_file in log_files:
            if os.path.exists(log_file):
                print(f"✅ Analizzando: {log_file}")
                
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Cerca nelle ultime 1000 righe
                    recent_lines = lines[-1000:] if len(lines) > 1000 else lines
                    
                    found_lines = []
                    for i, line in enumerate(recent_lines):
                        for term in search_terms:
                            if term.lower() in line.lower():
                                found_lines.append((i, line.strip()))
                                break
                    
                    if found_lines:
                        print(f"   🎯 Trovate {len(found_lines)} righe con pattern rappresentanti:")
                        for i, line in found_lines[-5:]:  # Ultime 5
                            print(f"      📝 [{i}]: {line[:100]}...")
                    else:
                        print(f"   ❌ Nessun pattern rappresentanti trovato")
                        
                except Exception as e:
                    print(f"   ❌ Errore lettura log: {e}")
            else:
                print(f"❌ Log non trovato: {log_file}")
    
    def run_complete_debug(self):
        """
        Esegue debug completo del processo rappresentanti
        
        Returns:
            dict: Risultati debug completo
            
        Data ultima modifica: 2025-09-07
        """
        print("🚀 AVVIO DEBUG COMPLETO RAPPRESENTANTI")
        print("="*70)
        
        results = {
            'debug_timestamp': datetime.now().isoformat(),
            'tenant_id': self.tenant_id,
            'tenant_name': self.tenant_name,
            'steps_completed': []
        }
        
        try:
            # Connessioni
            mongo_client, mongo_db, mongo_collection = self.connect_mongodb()
            mysql_conn = self.connect_mysql()
            
            if mongo_collection is not None:
                # Step 1: Dati MongoDB
                self.debug_mongodb_existing_data(mongo_collection)
                results['steps_completed'].append('mongodb_analysis')
                
                # Step 4: Sessioni specifiche
                self.debug_specific_sessions(mongo_collection)
                results['steps_completed'].append('sessions_analysis')
                
                mongo_client.close()
            
            if mysql_conn is not None:
                # Step 2: Cronologia MySQL
                self.debug_mysql_clustering_history(mysql_conn)
                results['steps_completed'].append('mysql_analysis')
                mysql_conn.close()
            
            # Step 3: Codice clustering
            self.debug_clustering_code_paths()
            results['steps_completed'].append('code_analysis')
            
            # Step 5: Log search
            self.search_representatives_in_logs()
            results['steps_completed'].append('logs_analysis')
            
            print("\\n" + "🏁" + "="*70)
            print("🏁 DEBUG COMPLETO TERMINATO")
            print("🏁" + "="*70)
            
            print(f"\\n📊 RIEPILOGO:")
            print(f"   ✅ Step completati: {len(results['steps_completed'])}")
            print(f"   📋 Step: {', '.join(results['steps_completed'])}")
            
            # Conclusioni
            print(f"\\n💡 POSSIBILI CAUSE MANCANZA RAPPRESENTANTI:")
            print(f"   1. ❌ Clustering non eseguito di recente")
            print(f"   2. ❌ Bug nel codice di selezione rappresentanti")
            print(f"   3. ❌ Problema nel salvataggio metadati MongoDB")
            print(f"   4. ❌ Configurazione clustering errata")
            print(f"   5. ❌ Flag rappresentanti salvato con nome diverso")
            
            results['success'] = True
            return results
            
        except Exception as e:
            print(f"❌ Errore debug completo: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

def main():
    """
    Funzione principale per eseguire il debug
    
    Data ultima modifica: 2025-09-07
    """
    debugger = RepresentativesDebugger()
    results = debugger.run_complete_debug()
    
    # Salva risultati
    output_file = f"/home/ubuntu/classificatore/representatives_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\\n💾 Risultati debug salvati in: {output_file}")

if __name__ == "__main__":
    main()
