#!/usr/bin/env python3
"""
============================================================================
Debug Clustering HDBSCAN - Analisi Outlier
============================================================================

Autore: Valerio Bignardi
Data creazione: 2025-09-07
Ultima modifica: 2025-09-07

Descrizione:
    Script per analizzare perché TUTTI i documenti Humanitas sono
    stati classificati come outlier (cluster_id = -1) invece di
    formare cluster con rappresentanti.

Analisi:
    1. Parametri clustering utilizzati
    2. Distribuzione embeddings
    3. Verifica processo HDBSCAN
    4. Controllo soglie e configurazioni
    5. Suggerimenti per correzione

============================================================================
"""

import sys
import os
import yaml
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import mysql.connector
import json

# Aggiungi path progetto
sys.path.append('/home/ubuntu/classificatore')

class HDBSCANOutlierDebugger:
    """
    Debugger specifico per problema outlier HDBSCAN
    
    Scopo:
        Analizza perché il clustering HDBSCAN ha classificato
        tutti i documenti come outlier invece di creare cluster
        
    Data ultima modifica: 2025-09-07
    """
    
    def __init__(self):
        """
        Inizializza debugger outlier
        
        Data ultima modifica: 2025-09-07
        """
        self.config = self._load_config()
        self.tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
        self.tenant_name = "Humanitas"
        self.collection_name = f"humanitas_{self.tenant_id}"
        
        print("🔍 DEBUG CLUSTERING OUTLIER - ANALISI HDBSCAN")
        print("="*60)
        print(f"👤 Tenant: {self.tenant_name}")
        print(f"📊 Collection: {self.collection_name}")
    
    def _load_config(self) -> dict:
        """Carica configurazione"""
        try:
            with open('/home/ubuntu/classificatore/config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Errore config: {e}")
            return {}
    
    def analyze_clustering_config(self):
        """
        Analizza configurazione clustering HDBSCAN
        
        Data ultima modifica: 2025-09-07
        """
        print("\\n🔧 STEP 1: ANALISI CONFIGURAZIONE CLUSTERING")
        print("-"*50)
        
        # Configurazione clustering dal config.yaml
        clustering_config = self.config.get('clustering', {})
        bertopic_config = self.config.get('bertopic', {})
        
        print(f"📊 Configurazione clustering generale:")
        print(f"   ⚙️ Abilitato: {clustering_config.get('enabled', False)}")
        print(f"   🔢 Algoritmo: {clustering_config.get('algorithm', 'N/A')}")
        print(f"   📏 Min cluster size: {clustering_config.get('min_cluster_size', 'N/A')}")
        print(f"   📐 Min samples: {clustering_config.get('min_samples', 'N/A')}")
        print(f"   🎯 GPU enabled: {clustering_config.get('gpu_enabled', False)}")
        
        print(f"\\n📊 Configurazione BERTopic:")
        print(f"   ⚙️ Abilitato: {bertopic_config.get('enabled', False)}")
        
        hdbscan_params = bertopic_config.get('hdbscan_params', {})
        print(f"   📏 HDBSCAN min_cluster_size: {hdbscan_params.get('min_cluster_size', 'N/A')}")
        print(f"   📐 HDBSCAN min_samples: {hdbscan_params.get('min_samples', 'N/A')}")
        print(f"   📊 HDBSCAN metric: {hdbscan_params.get('metric', 'N/A')}")
        
        # Analisi problema potenziale
        min_cluster_size = hdbscan_params.get('min_cluster_size', 0)
        total_docs = 1360  # Dal debug precedente
        
        print(f"\\n🚨 ANALISI PROBLEMA:")
        print(f"   📊 Documenti totali: {total_docs}")
        print(f"   📏 Min cluster size richiesto: {min_cluster_size}")
        print(f"   📊 Percentuale min cluster: {(min_cluster_size/total_docs)*100:.1f}%")
        
        if min_cluster_size > total_docs * 0.1:
            print(f"   ⚠️ PROBLEMA: min_cluster_size troppo alto!")
            print(f"   💡 Suggerimento: Ridurre min_cluster_size a {max(3, total_docs//100)}")
        
        return {
            'clustering_config': clustering_config,
            'bertopic_config': bertopic_config,
            'hdbscan_params': hdbscan_params
        }
    
    def analyze_embeddings_distribution(self):
        """
        Analizza distribuzione embeddings
        
        Data ultima modifica: 2025-09-07
        """
        print("\\n📊 STEP 2: ANALISI EMBEDDINGS")
        print("-"*50)
        
        try:
            # Connessione MongoDB
            mongo_client = MongoClient(self.config.get('mongodb', {}).get('url', 'mongodb://localhost:27017'))
            db = mongo_client['classificazioni']
            collection = db[self.collection_name]
            
            # Campione documenti per analisi embeddings
            sample_docs = list(collection.find({
                'tenant_id': self.tenant_id
            }).limit(10))
            
            print(f"📊 Analisi embeddings su {len(sample_docs)} documenti:")
            
            embedding_stats = {
                'has_embedding': 0,
                'empty_embedding': 0,
                'no_embedding_field': 0,
                'embedding_models': set()
            }
            
            for doc in sample_docs:
                embedding = doc.get('embedding', None)
                embedding_model = doc.get('embedding_model', 'unknown')
                
                embedding_stats['embedding_models'].add(embedding_model)
                
                if embedding is None:
                    embedding_stats['no_embedding_field'] += 1
                elif len(embedding) == 0:
                    embedding_stats['empty_embedding'] += 1
                else:
                    embedding_stats['has_embedding'] += 1
                    
                    # Analizza primo documento con embedding
                    if embedding_stats['has_embedding'] == 1 and len(embedding) > 0:
                        print(f"   📏 Dimensione embedding: {len(embedding)}")
                        print(f"   📊 Range valori: [{min(embedding):.3f}, {max(embedding):.3f}]")
                        print(f"   📈 Media: {np.mean(embedding):.3f}")
            
            print(f"\\n📊 Statistiche embeddings:")
            print(f"   ✅ Con embedding: {embedding_stats['has_embedding']}")
            print(f"   ❌ Embedding vuoto: {embedding_stats['empty_embedding']}")
            print(f"   ❓ Senza campo embedding: {embedding_stats['no_embedding_field']}")
            print(f"   🏷️ Modelli embedding: {list(embedding_stats['embedding_models'])}")
            
            # Verifica se il problema è negli embeddings
            if embedding_stats['has_embedding'] == 0:
                print(f"\\n🚨 PROBLEMA CRITICO: Nessun documento ha embeddings validi!")
                print(f"   💡 HDBSCAN richiede embeddings per il clustering")
                print(f"   🔧 Soluzione: Rigenerare embeddings per tutti i documenti")
            
            mongo_client.close()
            return embedding_stats
            
        except Exception as e:
            print(f"❌ Errore analisi embeddings: {e}")
            return {}
    
    def analyze_clustering_history(self):
        """
        Analizza cronologia clustering recente
        
        Data ultima modifica: 2025-09-07
        """
        print("\\n📈 STEP 3: ANALISI CRONOLOGIA CLUSTERING")
        print("-"*50)
        
        try:
            # Connessione MySQL
            mysql_conn = mysql.connector.connect(**self.config['tag_database'])
            cursor = mysql_conn.cursor(dictionary=True)
            
            # Verifica struttura tabella clustering_test_results
            cursor.execute("DESCRIBE clustering_test_results")
            columns = cursor.fetchall()
            
            print(f"📋 Colonne tabella clustering_test_results:")
            for col in columns:
                print(f"   📌 {col['Field']}: {col['Type']}")
            
            # Cerca risultati recenti (adatta query alle colonne esistenti)
            cursor.execute("""
                SELECT * FROM clustering_test_results 
                ORDER BY test_date DESC 
                LIMIT 5
            """)
            
            results = cursor.fetchall()
            if results:
                print(f"\\n📊 Ultimi {len(results)} test clustering:")
                for i, result in enumerate(results, 1):
                    print(f"\\n   🧪 Test {i}:")
                    for key, value in result.items():
                        if key in ['test_date', 'total_sessions', 'num_clusters', 'num_outliers', 'num_representatives']:
                            print(f"      📊 {key}: {value}")
            else:
                print("❌ Nessun test clustering trovato")
            
            cursor.close()
            mysql_conn.close()
            
        except Exception as e:
            print(f"❌ Errore analisi cronologia: {e}")
    
    def analyze_recent_logs(self):
        """
        Analizza log recenti per errori clustering
        
        Data ultima modifica: 2025-09-07
        """
        print("\\n📝 STEP 4: ANALISI LOG CLUSTERING")
        print("-"*50)
        
        log_file = '/home/ubuntu/classificatore/logging.log'
        
        if not os.path.exists(log_file):
            print(f"❌ File log non trovato: {log_file}")
            return
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Cerca log clustering recenti
            clustering_lines = []
            error_lines = []
            
            for line in lines[-2000:]:  # Ultime 2000 righe
                if any(term in line.lower() for term in ['clustering', 'hdbscan', 'outlier', 'cluster']):
                    clustering_lines.append(line.strip())
                
                if any(term in line.lower() for term in ['error', 'exception', 'failed']):
                    error_lines.append(line.strip())
            
            print(f"📊 Log clustering trovati: {len(clustering_lines)}")
            if clustering_lines:
                print("   📝 Ultimi log clustering:")
                for line in clustering_lines[-5:]:
                    print(f"      {line}")
            
            print(f"\\n🚨 Log errori trovati: {len(error_lines)}")
            if error_lines:
                print("   ❌ Ultimi errori:")
                for line in error_lines[-3:]:
                    print(f"      {line}")
                    
        except Exception as e:
            print(f"❌ Errore analisi log: {e}")
    
    def suggest_fixes(self, config_analysis, embedding_stats):
        """
        Suggerisce correzioni per il problema outlier
        
        Args:
            config_analysis: Risultati analisi configurazione
            embedding_stats: Statistiche embeddings
            
        Data ultima modifica: 2025-09-07
        """
        print("\\n💡 STEP 5: SUGGERIMENTI CORREZIONE")
        print("-"*50)
        
        suggestions = []
        
        # Controllo embeddings
        if embedding_stats.get('has_embedding', 0) == 0:
            suggestions.append({
                'priority': 'CRITICO',
                'issue': 'Nessun embedding valido',
                'solution': 'Rigenerare embeddings per tutti i documenti',
                'command': 'python regenerate_embeddings.py --tenant humanitas'
            })
        
        # Controllo parametri HDBSCAN
        hdbscan_params = config_analysis.get('hdbscan_params', {})
        min_cluster_size = hdbscan_params.get('min_cluster_size', 8)
        
        if min_cluster_size > 10:
            suggestions.append({
                'priority': 'ALTO',
                'issue': f'min_cluster_size troppo alto ({min_cluster_size})',
                'solution': 'Ridurre min_cluster_size a 3-5',
                'config_change': {'bertopic.hdbscan_params.min_cluster_size': 3}
            })
        
        min_samples = hdbscan_params.get('min_samples', 4)
        if min_samples > 5:
            suggestions.append({
                'priority': 'MEDIO',
                'issue': f'min_samples troppo alto ({min_samples})',
                'solution': 'Ridurre min_samples a 2-3',
                'config_change': {'bertopic.hdbscan_params.min_samples': 2}
            })
        
        # Controllo algoritmo
        if not config_analysis.get('clustering_config', {}).get('enabled', False):
            suggestions.append({
                'priority': 'CRITICO',
                'issue': 'Clustering disabilitato',
                'solution': 'Abilitare clustering in config.yaml',
                'config_change': {'clustering.enabled': True}
            })
        
        print(f"🔧 Trovati {len(suggestions)} suggerimenti:")
        
        for i, suggestion in enumerate(suggestions, 1):
            priority = suggestion['priority']
            emoji = '🚨' if priority == 'CRITICO' else '⚠️' if priority == 'ALTO' else '💡'
            
            print(f"\\n   {emoji} Suggerimento {i} [{priority}]:")
            print(f"      🔍 Problema: {suggestion['issue']}")
            print(f"      💡 Soluzione: {suggestion['solution']}")
            
            if 'config_change' in suggestion:
                print(f"      ⚙️ Modifica config: {suggestion['config_change']}")
            
            if 'command' in suggestion:
                print(f"      💻 Comando: {suggestion['command']}")
        
        return suggestions
    
    def run_analysis(self):
        """
        Esegue analisi completa problema outlier
        
        Returns:
            dict: Risultati analisi e suggerimenti
            
        Data ultima modifica: 2025-09-07
        """
        print("🚀 AVVIO ANALISI PROBLEMA OUTLIER")
        print("="*60)
        
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'tenant_id': self.tenant_id,
            'tenant_name': self.tenant_name
        }
        
        try:
            # Step 1: Configurazione
            config_analysis = self.analyze_clustering_config()
            results['config_analysis'] = config_analysis
            
            # Step 2: Embeddings
            embedding_stats = self.analyze_embeddings_distribution()
            results['embedding_stats'] = embedding_stats
            
            # Step 3: Cronologia
            self.analyze_clustering_history()
            
            # Step 4: Log
            self.analyze_recent_logs()
            
            # Step 5: Suggerimenti
            suggestions = self.suggest_fixes(config_analysis, embedding_stats)
            results['suggestions'] = suggestions
            
            print("\\n" + "🏁" + "="*60)
            print("🏁 ANALISI PROBLEMA OUTLIER COMPLETATA")
            print("🏁" + "="*60)
            
            # Riepilogo critico
            critical_issues = [s for s in suggestions if s['priority'] == 'CRITICO']
            high_issues = [s for s in suggestions if s['priority'] == 'ALTO']
            
            print(f"\\n📊 RIEPILOGO CRITICO:")
            print(f"   🚨 Problemi critici: {len(critical_issues)}")
            print(f"   ⚠️ Problemi ad alta priorità: {len(high_issues)}")
            
            if critical_issues:
                print(f"\\n🚨 AZIONI IMMEDIATE RICHIESTE:")
                for issue in critical_issues:
                    print(f"   ❗ {issue['solution']}")
            
            results['success'] = True
            return results
            
        except Exception as e:
            print(f"❌ Errore analisi: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

def main():
    """
    Funzione principale
    
    Data ultima modifica: 2025-09-07
    """
    debugger = HDBSCANOutlierDebugger()
    results = debugger.run_analysis()
    
    # Salva risultati
    output_file = f"/home/ubuntu/classificatore/outlier_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\\n💾 Risultati salvati in: {output_file}")

if __name__ == "__main__":
    main()
