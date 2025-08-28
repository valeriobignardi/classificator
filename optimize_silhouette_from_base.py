#!/usr/bin/env python3
"""
Autore: Valerio Bignardi
Data: 2025-08-28
Descrizione: Ottimizzazione silhouette partendo da OUTLIER_MINIMIZER
Storia aggiornamenti:
- 2025-08-28: Versione iniziale per miglioramento silhouette specifica
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

import yaml
import time
import traceback
import numpy as np
from sklearn.metrics import silhouette_score
from Preprocessing.session_aggregator import SessionAggregator
from EmbeddingEngine.labse_embedder import LaBSEEmbedder
from Clustering.hdbscan_clusterer import HDBSCANClusterer

def load_dataset():
    """
    Carica dataset completo Humanitas per test
    
    Returns:
        dict: Dataset con sessioni e embeddings
    """
    print("🔍 Caricamento dataset completo Humanitas...")
    
    # Setup tenant
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    tenant_slug = "humanitas"
    
    # Estrai TUTTE le sessioni
    aggregator = SessionAggregator(schema=tenant_slug, tenant_id=tenant_id)
    sessions_dict = aggregator.estrai_sessioni_aggregate(limit=None)
    sessions = list(sessions_dict.values())
    
    print(f"✅ Dataset: {len(sessions)} conversazioni")
    
    # Genera embeddings
    embedder = LaBSEEmbedder()
    session_texts = [s['testo_completo'] for s in sessions]
    
    # Batch processing per evitare OOM
    batch_size = 40
    all_embeddings = []
    
    for i in range(0, len(session_texts), batch_size):
        batch = session_texts[i:i+batch_size]
        print(f"📊 Batch {i//batch_size + 1}/{(len(session_texts)-1)//batch_size + 1}")
        
        embeddings = embedder.encode(batch)
        all_embeddings.append(embeddings)
        time.sleep(0.1)
    
    embeddings_matrix = np.vstack(all_embeddings)
    print(f"✅ Embeddings: {embeddings_matrix.shape}")
    
    return {
        'sessions': sessions,
        'embeddings': embeddings_matrix,
        'session_texts': session_texts
    }

def test_silhouette_optimization(config_name, params, dataset_info):
    """
    Test configurazione specifica per ottimizzazione silhouette
    
    Args:
        config_name: Nome configurazione
        params: Parametri clustering  
        dataset_info: Dataset precaricato
        
    Returns:
        dict: Risultati test
    """
    try:
        print(f"\n🚀 Test configurazione: {config_name}")
        print(f"📊 Dataset: {len(dataset_info['sessions'])} conversazioni")
        
        # 1. CLUSTERING
        start_time = time.time()
        
        clusterer = HDBSCANClusterer(**params)
        cluster_labels = clusterer.fit_predict(dataset_info['embeddings'])
        
        elapsed_time = time.time() - start_time
        
        # 2. METRICHE BASE
        unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        outliers = list(cluster_labels).count(-1)
        outlier_percentage = (outliers / len(cluster_labels)) * 100
        
        print(f"📊 Clusters: {unique_clusters}")
        print(f"❌ Outliers: {outliers} ({outlier_percentage:.1f}%)")
        
        # 3. SILHOUETTE SCORE COMPLETO (su sample per velocità)
        silhouette = -1.0
        try:
            if unique_clusters >= 2 and outlier_percentage < 90:
                # Sample per velocità ma rappresentativo
                sample_size = min(2000, len(dataset_info['sessions']))
                sample_indices = np.random.choice(len(dataset_info['sessions']), sample_size, replace=False)
                
                sample_embeddings = dataset_info['embeddings'][sample_indices]
                sample_labels = cluster_labels[sample_indices]
                
                # Solo elementi non-outlier
                non_outlier_mask = sample_labels != -1
                if np.sum(non_outlier_mask) >= 100:
                    silhouette = silhouette_score(
                        sample_embeddings[non_outlier_mask], 
                        sample_labels[non_outlier_mask]
                    )
                    print(f"📈 Silhouette: {silhouette:.4f}")
                else:
                    print("⚠️ Troppi outliers per silhouette")
            else:
                print("⚠️ Cluster insufficienti per silhouette")
        except Exception as e:
            print(f"❌ Errore silhouette: {e}")
        
        # 4. CLUSTER BALANCE
        cluster_balance = 0.0
        if unique_clusters > 0:
            cluster_sizes = []
            for cluster_id in set(cluster_labels):
                if cluster_id != -1:
                    cluster_sizes.append(list(cluster_labels).count(cluster_id))
            
            if cluster_sizes:
                cluster_balance = min(cluster_sizes) / max(cluster_sizes)
        
        # 5. QUALITY SCORE PERSONALIZZATO
        # Pesatura: silhouette (40%), outlier (40%), balance (20%)
        outlier_score = max(0, 1 - outlier_percentage/100)
        silhouette_score_norm = max(0, silhouette) if silhouette > 0 else 0
        
        quality_score = (
            silhouette_score_norm * 0.4 +
            outlier_score * 0.4 + 
            cluster_balance * 0.2
        )
        
        # 6. TARGET CHECK
        target_clusters = 20 <= unique_clusters <= 45
        target_outliers = outlier_percentage <= 35.0
        target_silhouette = silhouette >= 0.5
        
        print(f"⚖️ Balance: {cluster_balance:.4f}")
        print(f"🏆 Quality: {quality_score:.4f}")
        print(f"⏱️ Tempo: {elapsed_time:.1f}s")
        print(f"✅ Target clusters (20-45): {target_clusters}")
        print(f"✅ Target outliers (≤35%): {target_outliers}")
        print(f"✅ Target silhouette (≥0.5): {target_silhouette}")
        
        return {
            'config_name': config_name,
            'clusters': unique_clusters,
            'outliers': outliers,
            'outlier_percentage': outlier_percentage,
            'silhouette': silhouette,
            'cluster_balance': cluster_balance,
            'quality_score': quality_score,
            'execution_time': elapsed_time,
            'target_clusters': target_clusters,
            'target_outliers': target_outliers, 
            'target_silhouette': target_silhouette,
            'params': params
        }
        
    except Exception as e:
        print(f"❌ Errore test {config_name}: {e}")
        traceback.print_exc()
        return {'error': str(e), 'config_name': config_name}

def generate_silhouette_variations():
    """
    Genera variazioni dei parametri di OUTLIER_MINIMIZER per migliorare silhouette
    
    Returns:
        list: Configurazioni ottimizzate per silhouette
    """
    
    # PARAMETRI BASE OUTLIER_MINIMIZER (che ha già 36 cluster nel range)
    base_params = {
        'min_cluster_size': 40,
        'min_samples': 15,
        'metric': 'euclidean', 
        'cluster_selection_epsilon': 0.12,
        'cluster_selection_method': 'leaf',
        'alpha': 0.5,
        'allow_single_cluster': False,
        'use_umap': True,
        'umap_n_neighbors': 15,
        'umap_min_dist': 0.05,
        'umap_n_components': 20,
        'umap_metric': 'cosine'
    }
    
    configs = []
    
    # VARIANTE 1: Aumenta separazione cluster (silhouette focus)
    configs.append({
        'name': 'OUTLIER_BASE_V1_SEPARATION',
        'description': 'Migliora separazione cluster per silhouette',
        'params': {
            **base_params,
            'min_cluster_size': 35,        # Leggermente più piccoli
            'min_samples': 12,             # Meno restrittivo
            'cluster_selection_epsilon': 0.08,  # Meno merge aggressivo  
            'alpha': 0.4,                 # Più conservativo per separazione
            'umap_min_dist': 0.01,        # Separazione più netta in UMAP
            'umap_n_neighbors': 12,       # Meno vicini = più separazione
        }
    })
    
    # VARIANTE 2: Metrica cosine per migliore similarity matching
    configs.append({
        'name': 'OUTLIER_BASE_V2_COSINE',  
        'description': 'Usa metrica cosine per clustering',
        'params': {
            **base_params,
            'metric': 'cosine',            # Cambia metrica principale
            'min_cluster_size': 38,        # Mantenimento cluster count
            'cluster_selection_epsilon': 0.10,  # Adatta per cosine
            'umap_metric': 'cosine',       # Coerenza metrica
            'umap_n_components': 25,       # Più dimensioni per cosine
        }
    })
    
    # VARIANTE 3: UMAP più conservativo per preservare structure 
    configs.append({
        'name': 'OUTLIER_BASE_V3_UMAP_CONSERVATIVE',
        'description': 'UMAP più conservativo per struttura migliore',
        'params': {
            **base_params,
            'min_cluster_size': 42,        # Leggermente più grandi 
            'min_samples': 18,             # Più samples per stabilità
            'alpha': 0.6,                 # Più aggressivo
            'umap_n_neighbors': 20,       # Più contesto locale
            'umap_min_dist': 0.1,         # Più distanza in UMAP
            'umap_n_components': 15,      # Meno dimensioni = meno rumore
        }
    })
    
    # VARIANTE 4: Epsilon più fine per separazione ottimale
    configs.append({
        'name': 'OUTLIER_BASE_V4_FINE_EPSILON',
        'description': 'Epsilon fine-tuned per separazione ottimale',
        'params': {
            **base_params,
            'cluster_selection_epsilon': 0.06,  # Molto più conservativo  
            'min_cluster_size': 45,        # Cluster più grandi
            'min_samples': 20,             # Più conservativo
            'alpha': 0.3,                 # Molto conservativo
            'umap_n_neighbors': 10,       # Molto locale
            'umap_min_dist': 0.001,       # Separazione massima
        }
    })
    
    # VARIANTE 5: Hybrid approach - Balance tra tutti i fattori
    configs.append({
        'name': 'OUTLIER_BASE_V5_HYBRID',
        'description': 'Approccio bilanciato per tutti gli obiettivi',
        'params': {
            **base_params,
            'min_cluster_size': 44,        # Nel range ma vicino al limite
            'min_samples': 16,             # Bilanciato
            'metric': 'cosine',            # Migliore per testi
            'cluster_selection_epsilon': 0.09,  # Compromesso
            'alpha': 0.45,                # Compromesso
            'umap_n_neighbors': 18,       # Bilanciato
            'umap_min_dist': 0.03,        # Bilanciato
            'umap_n_components': 22,      # Bilanciato
            'umap_metric': 'cosine',      # Coerenza
        }
    })
    
    return configs

def main():
    """Esegue ottimizzazione silhouette partendo da OUTLIER_MINIMIZER"""
    
    print("🚀 OTTIMIZZAZIONE SILHOUETTE DA OUTLIER_MINIMIZER BASE")
    print("=" * 80)
    
    # 1. CARICA DATASET UNA VOLTA SOLA
    dataset_info = load_dataset()
    
    # 2. GENERA CONFIGURAZIONI MIRATE
    configs = generate_silhouette_variations()
    print(f"📊 Configurazioni da testare: {len(configs)}")
    
    # 3. ESEGUI TEST
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*50}")
        print(f"🧪 TEST {i}/{len(configs)}: {config['name']}")
        print(f"📝 {config['description']}")
        print(f"{'='*50}")
        
        result = test_silhouette_optimization(
            config['name'], 
            config['params'], 
            dataset_info
        )
        
        if 'error' not in result:
            results.append(result)
    
    # 4. ANALISI FINALE
    if results:
        print(f"\n🏆 RISULTATI FINALI OTTIMIZZAZIONE SILHOUETTE")
        print("=" * 80)
        print("📊 TARGET: Cluster 20-45, Outlier ≤35%, Silhouette ≥0.5")
        print("=" * 80)
        
        # Ordina per quality score
        results.sort(key=lambda x: x['quality_score'], reverse=True)
        
        for i, result in enumerate(results, 1):
            print(f"\n🎯 #{i}: {result['config_name']}")
            print(f"   📊 Clusters: {result['clusters']} {'✅' if result['target_clusters'] else '❌'}")
            print(f"   ❌ Outliers: {result['outlier_percentage']:.1f}% {'✅' if result['target_outliers'] else '❌'}")
            print(f"   📈 Silhouette: {result['silhouette']:.4f} {'✅' if result['target_silhouette'] else '❌'}")
            print(f"   🏆 Quality: {result['quality_score']:.4f}")
            print(f"   ⏱️ Tempo: {result['execution_time']:.1f}s")
            
            # Conta target raggiunti
            targets_met = sum([result['target_clusters'], result['target_outliers'], result['target_silhouette']])
            print(f"   🎯 Target raggiunti: {targets_met}/3")
        
        # Miglior configurazione per silhouette
        best_silhouette = max(results, key=lambda x: x['silhouette'] if x['silhouette'] > 0 else -1)
        print(f"\n🏆 MIGLIORE SILHOUETTE: {best_silhouette['config_name']}")
        print(f"   📈 Silhouette: {best_silhouette['silhouette']:.4f}")
        print(f"   📊 Clusters: {best_silhouette['clusters']}")
        print(f"   ❌ Outliers: {best_silhouette['outlier_percentage']:.1f}%")
        
        # Miglior bilanciamento target
        valid_configs = [r for r in results if r['target_clusters'] and r['target_outliers']]
        if valid_configs:
            best_balanced = max(valid_configs, key=lambda x: x['silhouette'] if x['silhouette'] > 0 else -1)
            print(f"\n🎖️ MIGLIORE BILANCIAMENTO TARGET: {best_balanced['config_name']}")
            print(f"   📈 Silhouette: {best_balanced['silhouette']:.4f}")
            print(f"   📊 Clusters: {best_balanced['clusters']} ✅")
            print(f"   ❌ Outliers: {best_balanced['outlier_percentage']:.1f}% ✅")
        else:
            print("\n⚠️ Nessuna configurazione raggiunge target cluster E outlier simultaneamente")
    
    print(f"\n✅ Ottimizzazione completata!")

if __name__ == "__main__":
    main()
