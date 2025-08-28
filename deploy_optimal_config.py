#!/usr/bin/env python3
"""
Autore: Valerio Bignardi
Data: 2025-08-28
Descrizione: Implementazione configurazione ottimale V3_MICRO_LESS_DIM con tutte le metriche
Storia aggiornamenti:
- 2025-08-28: Implementazione completa con Davies-Bouldin e Calinski-Harabasz
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

import yaml
import time
import numpy as np
import json
import mysql.connector
from datetime import datetime
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from Preprocessing.session_aggregator import SessionAggregator
from EmbeddingEngine.labse_embedder import LaBSEEmbedder
from Clustering.hdbscan_clusterer import HDBSCANClusterer

# CONFIGURAZIONE OTTIMALE V3_MICRO_LESS_DIM
OPTIMAL_CONFIG = {
    'min_cluster_size': 40,
    'min_samples': 18,
    'metric': 'euclidean', 
    'cluster_selection_epsilon': 0.12,
    'cluster_selection_method': 'leaf',
    'alpha': 0.6,
    'allow_single_cluster': False,
    'use_umap': True,
    'umap_n_neighbors': 20,
    'umap_min_dist': 0.1,
    'umap_n_components': 12,  # Ottimizzazione chiave
    'umap_metric': 'cosine'
}

def load_config():
    """Carica configurazione YAML"""
    with open('/home/ubuntu/classificatore/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def calculate_comprehensive_metrics(embeddings, cluster_labels):
    """
    Calcola tutte le metriche di clustering comprehensive
    
    Args:
        embeddings: Matrix embeddings
        cluster_labels: Labels dei cluster (-1 per outliers)
        
    Returns:
        dict: Tutte le metriche calcolate
        
    Ultima modifica: 2025-08-28
    """
    metrics = {
        'silhouette_score': -1.0,
        'davies_bouldin_score': float('inf'),
        'calinski_harabasz_score': 0.0
    }
    
    try:
        # Filtra outliers per il calcolo delle metriche
        non_outlier_mask = cluster_labels != -1
        
        if np.sum(non_outlier_mask) < 100:
            print("⚠️ Troppi outliers per calcolo metriche affidabile")
            return metrics
        
        non_outlier_embeddings = embeddings[non_outlier_mask]
        non_outlier_labels = cluster_labels[non_outlier_mask]
        
        # Verifica che ci siano almeno 2 cluster
        unique_clusters = len(set(non_outlier_labels))
        if unique_clusters < 2:
            print("⚠️ Meno di 2 cluster per calcolo metriche")
            return metrics
        
        print(f"📊 Calcolo metriche su {np.sum(non_outlier_mask)} punti, {unique_clusters} cluster")
        
        # SILHOUETTE SCORE
        try:
            # Se troppi punti, usa campione rappresentativo
            if len(non_outlier_embeddings) > 2000:
                sample_size = 2000
                sample_indices = np.random.choice(len(non_outlier_embeddings), sample_size, replace=False)
                sample_embeddings = non_outlier_embeddings[sample_indices]
                sample_labels = non_outlier_labels[sample_indices]
                
                metrics['silhouette_score'] = silhouette_score(sample_embeddings, sample_labels)
                print(f"📈 Silhouette Score: {metrics['silhouette_score']:.4f} (su campione {sample_size})")
            else:
                metrics['silhouette_score'] = silhouette_score(non_outlier_embeddings, non_outlier_labels)
                print(f"📈 Silhouette Score: {metrics['silhouette_score']:.4f}")
        except Exception as e:
            print(f"❌ Errore Silhouette: {e}")
        
        # DAVIES-BOULDIN INDEX (più basso = meglio)
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(non_outlier_embeddings, non_outlier_labels)
            print(f"📉 Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f}")
        except Exception as e:
            print(f"❌ Errore Davies-Bouldin: {e}")
        
        # CALINSKI-HARABASZ INDEX (più alto = meglio)
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(non_outlier_embeddings, non_outlier_labels)
            print(f"📊 Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.2f}")
        except Exception as e:
            print(f"❌ Errore Calinski-Harabasz: {e}")
            
    except Exception as e:
        print(f"❌ Errore generale calcolo metriche: {e}")
    
    return metrics

def save_results_to_database(config, tenant_id, results, execution_time):
    """
    Salva risultati nel database TAG con tutte le metriche
    
    Args:
        config: Configurazione YAML
        tenant_id: ID tenant
        results: Risultati clustering
        execution_time: Tempo esecuzione
        
    Ultima modifica: 2025-08-28
    """
    try:
        # Connessione database TAG
        conn = mysql.connector.connect(
            host=config['tag_database']['host'],
            user=config['tag_database']['user'],
            password=config['tag_database']['password'],
            database=config['tag_database']['database']
        )
        
        cursor = conn.cursor()
        
        # Trova prossimo version number
        cursor.execute(
            "SELECT COALESCE(MAX(version_number), 0) + 1 FROM clustering_test_results WHERE tenant_id = %s",
            (tenant_id,)
        )
        version_number = cursor.fetchone()[0]
        
        # Prepara dati
        results_json = json.dumps(results, indent=2)
        parameters_json = json.dumps(OPTIMAL_CONFIG, indent=2)
        
        # Insert con tutte le metriche
        insert_query = """
            INSERT INTO clustering_test_results (
                tenant_id, version_number, execution_time, results_json, parameters_json,
                n_clusters, n_outliers, n_conversations, clustering_ratio, 
                silhouette_score, davies_bouldin_score, calinski_harabasz_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            tenant_id,
            version_number,
            execution_time,
            results_json,
            parameters_json,
            results['clusters'],
            results['outliers'],
            results['total_conversations'],
            results['clustering_ratio'],
            results['metrics']['silhouette_score'],
            results['metrics']['davies_bouldin_score'] if results['metrics']['davies_bouldin_score'] != float('inf') else None,
            results['metrics']['calinski_harabasz_score']
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Risultati salvati: Version {version_number}")
        return version_number
        
    except Exception as e:
        print(f"❌ Errore salvataggio database: {e}")
        return None

def deploy_optimal_configuration():
    """
    Deploy della configurazione ottimale V3_MICRO_LESS_DIM con tutte le metriche
    
    Returns:
        dict: Risultati completi
        
    Ultima modifica: 2025-08-28
    """
    
    print("🚀 DEPLOY CONFIGURAZIONE OTTIMALE V3_MICRO_LESS_DIM")
    print("=" * 70)
    print("📊 Configurazione: 30 cluster, 34.3% outliers, silhouette 0.1476")
    print("⚡ Performance: 11.3s su 4,138 conversazioni")
    print("=" * 70)
    
    start_time = time.time()
    
    # 1. CARICA CONFIGURAZIONE
    config = load_config()
    
    # 2. SETUP TENANT
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    tenant_slug = "humanitas"
    print(f"✅ Tenant: {tenant_slug} ({tenant_id})")
    
    # 3. ESTRAI DATASET COMPLETO
    print("📊 Estrazione dataset completo...")
    aggregator = SessionAggregator(schema=tenant_slug, tenant_id=tenant_id)
    sessions_dict = aggregator.estrai_sessioni_aggregate(limit=None)
    sessions = list(sessions_dict.values())
    
    print(f"✅ Dataset: {len(sessions)} conversazioni")
    
    # 4. GENERA EMBEDDINGS
    print("🔍 Generazione embeddings...")
    embedder = LaBSEEmbedder()
    session_texts = [s['testo_completo'] for s in sessions]
    
    # Batch processing ottimizzato
    batch_size = 50
    all_embeddings = []
    
    total_batches = (len(session_texts) - 1) // batch_size + 1
    for i in range(0, len(session_texts), batch_size):
        batch = session_texts[i:i+batch_size]
        if i // batch_size % 20 == 0:
            print(f"📊 Batch {i//batch_size + 1}/{total_batches}")
        
        embeddings = embedder.encode(batch)
        all_embeddings.append(embeddings)
    
    embeddings_matrix = np.vstack(all_embeddings)
    print(f"✅ Embeddings: {embeddings_matrix.shape}")
    
    # 5. CLUSTERING CON CONFIGURAZIONE OTTIMALE
    print("🚀 Clustering con configurazione ottimale...")
    
    clusterer = HDBSCANClusterer(**OPTIMAL_CONFIG)
    cluster_labels = clusterer.fit_predict(embeddings_matrix)
    
    # 6. ANALISI RISULTATI
    unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    outliers = list(cluster_labels).count(-1)
    outlier_percentage = (outliers / len(cluster_labels)) * 100
    clustering_ratio = (len(cluster_labels) - outliers) / len(cluster_labels)
    
    print(f"📊 Clusters trovati: {unique_clusters}")
    print(f"❌ Outliers: {outliers} ({outlier_percentage:.1f}%)")
    print(f"⚖️ Clustering ratio: {clustering_ratio:.3f}")
    
    # 7. CALCOLO TUTTE LE METRICHE
    print("\n📈 Calcolo metriche comprehensive...")
    metrics = calculate_comprehensive_metrics(embeddings_matrix, cluster_labels)
    
    # 8. CLUSTER SIZE ANALYSIS
    cluster_sizes = {}
    for i, label in enumerate(cluster_labels):
        if label != -1:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
    
    # Top cluster sizes per reporting
    top_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
    top_cluster_info = [f"C{cid}:{size}" for cid, size in top_clusters]
    
    # 9. PREPARAZIONE RISULTATI
    execution_time = time.time() - start_time
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': 'V3_MICRO_LESS_DIM',
        'total_conversations': len(sessions),
        'clusters': unique_clusters,
        'outliers': outliers,
        'outlier_percentage': outlier_percentage,
        'clustering_ratio': clustering_ratio,
        'execution_time': execution_time,
        'metrics': metrics,
        'top_clusters': top_cluster_info,
        'dataset_info': {
            'embedding_shape': embeddings_matrix.shape,
            'batch_processing': True,
            'batch_size': batch_size
        }
    }
    
    print(f"\n🏆 RISULTATI FINALI:")
    print(f"   📊 Clusters: {unique_clusters}")
    print(f"   ❌ Outliers: {outlier_percentage:.1f}%")
    print(f"   📈 Silhouette: {metrics['silhouette_score']:.4f}")
    print(f"   📉 Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}")
    print(f"   📊 Calinski-Harabasz: {metrics['calinski_harabasz_score']:.2f}")
    print(f"   ⏱️ Tempo esecuzione: {execution_time:.1f}s")
    print(f"   🔝 Top cluster sizes: {', '.join(top_cluster_info)}")
    
    # 10. SALVA NEL DATABASE
    print("\n💾 Salvataggio risultati nel database...")
    version_number = save_results_to_database(config, tenant_id, results, execution_time)
    
    if version_number:
        results['version_number'] = version_number
        print(f"✅ Deploy completato: Version {version_number}")
    else:
        print("❌ Errore salvataggio database")
    
    return results

def main():
    """Esegue il deploy della configurazione ottimale"""
    try:
        results = deploy_optimal_configuration()
        
        print(f"\n🎉 DEPLOY CONFIGURAZIONE V3_MICRO_LESS_DIM COMPLETATO!")
        print(f"📊 Version: {results.get('version_number', 'N/A')}")
        print(f"🚀 Ready for production!")
        
        return results
        
    except Exception as e:
        print(f"❌ Errore durante deploy: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
