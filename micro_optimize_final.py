#!/usr/bin/env python3
"""
Autore: Valerio Bignardi
Data: 2025-08-28
Descrizione: Micro-ottimizzazione finale partendo da V3_UMAP_CONSERVATIVE
Storia aggiornamenti:
- 2025-08-28: Versione ultra-mirata per silhouette da migliore configurazione
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

import yaml
import time
import numpy as np
from sklearn.metrics import silhouette_score
from Preprocessing.session_aggregator import SessionAggregator
from EmbeddingEngine.labse_embedder import LaBSEEmbedder
from Clustering.hdbscan_clusterer import HDBSCANClusterer

# CONFIGURAZIONE BASE VINCENTE (V3_UMAP_CONSERVATIVE)
BASE_CONFIG = {
    'min_cluster_size': 42,
    'min_samples': 18,
    'metric': 'euclidean', 
    'cluster_selection_epsilon': 0.12,
    'cluster_selection_method': 'leaf',
    'alpha': 0.6,
    'allow_single_cluster': False,
    'use_umap': True,
    'umap_n_neighbors': 20,
    'umap_min_dist': 0.1,
    'umap_n_components': 15,
    'umap_metric': 'cosine'
}

def load_dataset_fast():
    """Carica dataset velocemente (usa cache se possibile)"""
    print("🔍 Caricamento rapido dataset...")
    
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    tenant_slug = "humanitas"
    
    aggregator = SessionAggregator(schema=tenant_slug, tenant_id=tenant_id)
    sessions_dict = aggregator.estrai_sessioni_aggregate(limit=None)
    sessions = list(sessions_dict.values())
    
    embedder = LaBSEEmbedder()
    session_texts = [s['testo_completo'] for s in sessions]
    
    # Batch processing più veloce
    batch_size = 50
    all_embeddings = []
    
    total_batches = (len(session_texts) - 1) // batch_size + 1
    for i in range(0, len(session_texts), batch_size):
        batch = session_texts[i:i+batch_size]
        if i // batch_size % 20 == 0:  # Log ogni 20 batch
            print(f"📊 Batch {i//batch_size + 1}/{total_batches}")
        
        embeddings = embedder.encode(batch)
        all_embeddings.append(embeddings)
    
    embeddings_matrix = np.vstack(all_embeddings)
    print(f"✅ Dataset: {len(sessions)} conversazioni, {embeddings_matrix.shape} embeddings")
    
    return {'sessions': sessions, 'embeddings': embeddings_matrix}

def test_micro_config(name, params, dataset_info):
    """Test ultra-veloce configurazione"""
    try:
        start_time = time.time()
        
        clusterer = HDBSCANClusterer(**params)
        cluster_labels = clusterer.fit_predict(dataset_info['embeddings'])
        
        clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        outliers = list(cluster_labels).count(-1)
        outlier_pct = (outliers / len(cluster_labels)) * 100
        
        # Silhouette veloce su sample
        silhouette = -1.0
        try:
            if clusters >= 2 and outlier_pct < 90:
                sample_size = min(1500, len(dataset_info['sessions']))
                sample_idx = np.random.choice(len(dataset_info['sessions']), sample_size, replace=False)
                
                sample_emb = dataset_info['embeddings'][sample_idx]
                sample_labels = cluster_labels[sample_idx]
                
                non_outlier = sample_labels != -1
                if np.sum(non_outlier) >= 100:
                    silhouette = silhouette_score(sample_emb[non_outlier], sample_labels[non_outlier])
        except:
            pass
        
        elapsed = time.time() - start_time
        
        # Target check
        target_clusters = 20 <= clusters <= 45
        target_outliers = outlier_pct <= 35.0
        target_silhouette = silhouette >= 0.5
        
        targets_met = sum([target_clusters, target_outliers, target_silhouette])
        
        print(f"🔬 {name}:")
        print(f"   📊 Clusters: {clusters} {'✅' if target_clusters else '❌'}")
        print(f"   ❌ Outliers: {outlier_pct:.1f}% {'✅' if target_outliers else '❌'}")
        print(f"   📈 Silhouette: {silhouette:.4f} {'✅' if target_silhouette else '❌'}")
        print(f"   🎯 Targets: {targets_met}/3 | ⏱️ {elapsed:.1f}s")
        
        return {
            'name': name, 'clusters': clusters, 'outlier_pct': outlier_pct, 
            'silhouette': silhouette, 'targets_met': targets_met, 'time': elapsed,
            'target_clusters': target_clusters, 'target_outliers': target_outliers, 
            'target_silhouette': target_silhouette
        }
    except Exception as e:
        print(f"❌ Errore {name}: {e}")
        return None

def generate_micro_variants():
    """Genera micro-variazioni della configurazione base vincente"""
    
    configs = []
    
    # MICRO-VARIANTE 1: Meno dimensioni UMAP per meno rumore
    configs.append({
        'name': 'V3_MICRO_LESS_DIM',
        'params': {
            **BASE_CONFIG,
            'umap_n_components': 12,  # Ancora meno dimensioni
            'min_cluster_size': 40,   # Leggermente più piccolo
        }
    })
    
    # MICRO-VARIANTE 2: Alpha più basso per separazione migliore  
    configs.append({
        'name': 'V3_MICRO_LOW_ALPHA',
        'params': {
            **BASE_CONFIG,
            'alpha': 0.4,             # Meno aggressivo
            'cluster_selection_epsilon': 0.10,  # Leggermente meno merge
        }
    })
    
    # MICRO-VARIANTE 3: UMAP più conservativo
    configs.append({
        'name': 'V3_MICRO_CONSERVATIVE',
        'params': {
            **BASE_CONFIG,
            'umap_n_neighbors': 25,   # Più contesto
            'umap_min_dist': 0.15,    # Più separazione
            'min_samples': 16,        # Leggermente meno restrittivo
        }
    })
    
    # MICRO-VARIANTE 4: Epsilon più fine
    configs.append({
        'name': 'V3_MICRO_FINE_EPSILON',
        'params': {
            **BASE_CONFIG,
            'cluster_selection_epsilon': 0.08,  # Più fine
            'alpha': 0.5,             # Compromesso
            'umap_n_components': 18,  # Più dimensioni
        }
    })
    
    # MICRO-VARIANTE 5: Metrica cosine finale
    configs.append({
        'name': 'V3_MICRO_COSINE',
        'params': {
            **BASE_CONFIG,
            'metric': 'cosine',       # Cambio metrica
            'umap_metric': 'cosine',  # Coerenza
            'cluster_selection_epsilon': 0.09,  # Adatta per cosine
            'umap_n_components': 20,  # Più dimensioni per cosine
        }
    })
    
    return configs

def main():
    """Esegue micro-ottimizzazione finale"""
    
    print("🔬 MICRO-OTTIMIZZAZIONE FINALE DA V3_UMAP_CONSERVATIVE")
    print("=" * 70)
    print("🎯 TARGET: Cluster 20-45, Outlier ≤35%, Silhouette ≥0.5")
    print("📊 BASE: 25 clusters, 28.5% outliers, 0.1679 silhouette")
    print("=" * 70)
    
    # Carica dataset una volta
    dataset_info = load_dataset_fast()
    
    # Genera micro-varianti
    configs = generate_micro_variants()
    
    # Test rapido
    results = []
    
    # Test configurazione base per confronto
    print("\n🏆 BASELINE V3_UMAP_CONSERVATIVE:")
    baseline = test_micro_config("V3_BASELINE", BASE_CONFIG, dataset_info)
    if baseline:
        results.append(baseline)
    
    # Test micro-varianti
    print("\n🔬 MICRO-VARIANTI:")
    for config in configs:
        result = test_micro_config(config['name'], config['params'], dataset_info)
        if result:
            results.append(result)
    
    # Analisi finale
    if results:
        print(f"\n🏆 CLASSIFICA FINALE MICRO-OTTIMIZZAZIONE")
        print("=" * 60)
        
        # Ordina per target raggiunti, poi per silhouette
        results.sort(key=lambda x: (x['targets_met'], x['silhouette']), reverse=True)
        
        for i, r in enumerate(results, 1):
            print(f"\n#{i}: {r['name']}")
            print(f"    🎯 Target: {r['targets_met']}/3")
            print(f"    📊 Clusters: {r['clusters']} | Outliers: {r['outlier_pct']:.1f}%")
            print(f"    📈 Silhouette: {r['silhouette']:.4f} | ⏱️ {r['time']:.1f}s")
        
        # Miglior configurazione
        best = results[0]
        print(f"\n🏅 CONFIGURAZIONE OTTIMALE: {best['name']}")
        print(f"   🎯 Target raggiunti: {best['targets_met']}/3")
        print(f"   📊 Clusters: {best['clusters']} ({'✅' if best['target_clusters'] else '❌'})")
        print(f"   ❌ Outliers: {best['outlier_pct']:.1f}% ({'✅' if best['target_outliers'] else '❌'})")
        print(f"   📈 Silhouette: {best['silhouette']:.4f} ({'✅' if best['target_silhouette'] else '❌'})")
        
        # Check se abbiamo raggiunto tutti i target
        if best['targets_met'] == 3:
            print("\n🎉 🎉 🎉 TUTTI I TARGET RAGGIUNTI! 🎉 🎉 🎉")
        elif best['targets_met'] >= 2:
            print(f"\n💪 Ottimo progresso: {best['targets_met']}/3 target raggiunti!")
        else:
            print(f"\n🔄 Necessaria ulteriore ottimizzazione: solo {best['targets_met']}/3 target")
    
    print("\n✅ Micro-ottimizzazione completata!")

if __name__ == "__main__":
    main()
