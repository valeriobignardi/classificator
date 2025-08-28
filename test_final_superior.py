#!/usr/bin/env python3
"""
Autore: Valerio Bignardi
Data: 2025-08-28
Descrizione: Test FINALE configurazioni clustering superiori - approccio diretto
Storia aggiornamenti:
- 2025-08-28: Test diretto con componenti individuali per massimo controllo
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

def load_config():
    """Carica configurazione YAML"""
    with open('/home/ubuntu/classificatore/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_superior_clustering(config_name, params, expected_performance):
    """
    Test diretto configurazione clustering con controllo totale
    
    Args:
        config_name (str): Nome configurazione
        params (dict): Parametri clustering
        expected_performance (dict): Performance attese
    
    Returns:
        dict: Risultati del test
    """
    print(f"🧪 TEST DIRETTO {config_name}")
    print("=" * 60)
    print(f"🎯 Performance attese:")
    for key, value in expected_performance.items():
        print(f"   • {key}: {value}")
    print()
    
    try:
        start_time = time.time()
        
        # 1. SETUP TENANT
        print("🔍 Setup tenant Humanitas...")
        tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
        tenant_slug = "humanitas"
        print(f"✅ Tenant: Humanitas ({tenant_slug})")
        
        # 2. ESTRAI SESSIONI
        print("📊 Estrazione sessioni con limite...")
        config = load_config()
        aggregator = SessionAggregator(
            schema=tenant_slug, 
            tenant_id=tenant_id
        )
        
        sessions_dict = aggregator.estrai_sessioni_aggregate(
            limit=250  # Limite ridotto per stabilità
        )
        
        sessions = list(sessions_dict.values())
        
        # DEBUG: controlliamo la struttura della prima sessione
        if sessions:
            print(f"🔍 Struttura sessione: {list(sessions[0].keys())}")
        
        print(f"✅ Dataset estratto: {len(sessions)} conversazioni")
        
        if len(sessions) < 50:
            return {"error": "Dataset insufficiente", "config_name": config_name, "sessions": len(sessions)}
        
        # 3. INIZIALIZZA EMBEDDER
        print("🚀 Inizializzazione LaBSE embedder...")
        embedder = LaBSEEmbedder(device='cuda')
        
        # 4. GENERA EMBEDDINGS
        print("🔍 Generazione embeddings...")
        session_texts = [s['testo_completo'] for s in sessions]
        
        # Processo in batch per stabilità  
        batch_size = 30
        all_embeddings = []
        
        for i in range(0, len(session_texts), batch_size):
            batch = session_texts[i:i+batch_size]
            print(f"📊 Batch {i//batch_size + 1}/{(len(session_texts)-1)//batch_size + 1}: {len(batch)} items")
            
            embeddings = embedder.encode(batch)
            all_embeddings.append(embeddings)
            time.sleep(0.1)  # Piccola pausa
        
        embeddings_matrix = np.vstack(all_embeddings)
        print(f"✅ Embeddings matrix: {embeddings_matrix.shape}")
        
        # 5. CLUSTERING CON PARAMETRI SUPERIORI
        print(f"🚀 Clustering con parametri {config_name}...")
        
        clusterer = HDBSCANClusterer(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            alpha=params['alpha'],
            cluster_selection_method=params['cluster_selection_method'],
            cluster_selection_epsilon=params['cluster_selection_epsilon'],
            metric=params['metric'],
            use_umap=params['use_umap'],
            umap_n_neighbors=params['umap_n_neighbors'],
            umap_min_dist=params['umap_min_dist'],
            umap_n_components=params['umap_n_components'],
            umap_metric=params['umap_metric']
        )
        
        cluster_results = clusterer.fit_predict(embeddings_matrix)
        
        execution_time = time.time() - start_time
        
        # 6. CALCOLA METRICHE
        unique_clusters = set(cluster_results)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_outliers = sum(1 for x in cluster_results if x == -1)
        n_conversations = len(sessions)
        outlier_ratio = n_outliers / n_conversations
        
        # Silhouette score
        try:
            if n_clusters > 1 and len(unique_clusters) > 1:
                silhouette = silhouette_score(embeddings_matrix, cluster_results)
            else:
                silhouette = 0.0
        except Exception as e:
            print(f"⚠️ Errore calcolo silhouette: {str(e)}")
            silhouette = 0.0
        
        # Quality score personalizzato  
        quality_score = (1 - outlier_ratio) * 0.6 + silhouette * 0.4
        
        results = {
            "success": True,
            "config_name": config_name,
            "n_clusters": n_clusters,
            "n_outliers": n_outliers,
            "n_conversations": n_conversations,
            "outlier_ratio": outlier_ratio,
            "silhouette_score": silhouette,
            "quality_score": quality_score,
            "execution_time": execution_time,
            "parameters": params,
            "cluster_distribution": dict(zip(*np.unique(cluster_results, return_counts=True)))
        }
        
        print(f"✅ {config_name} COMPLETATO!")
        print(f"   📊 Conversazioni: {n_conversations}")
        print(f"   🎯 Clusters: {n_clusters}")
        print(f"   ❌ Outliers: {n_outliers} ({outlier_ratio:.1%})")
        print(f"   📈 Silhouette: {silhouette:.4f}")
        print(f"   🏆 Quality Score: {quality_score:.4f}")
        print(f"   ⏱️ Tempo: {execution_time:.1f}s")
        
        return results
        
    except Exception as e:
        error_msg = f"Test {config_name} fallito: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return {"error": error_msg, "config_name": config_name}

def main():
    """Test finale configurazioni clustering superiori progettate"""
    
    print("🚀 TEST FINALE - CONFIGURAZIONI CLUSTERING SUPERIORI PROGETTATE")
    print("=" * 80)
    print("🎯 Obiettivo: BATTERE V22 (12.9% outliers, 0.0836 silhouette)")
    print("💡 Approccio: Test diretto con controllo totale parametri")
    print("=" * 80)
    
    # CONFIGURAZIONI SUPERIORI PROGETTATE - refined basate sui pattern
    superior_configs = {
        "PRECISION_ULTRA": {
            "params": {
                'use_umap': True,
                'metric': 'euclidean',
                'alpha': 0.40,  # Balanced per precision
                'min_cluster_size': 6,  # Più piccolo per sensibilità
                'min_samples': 4,  # Ridotto per catturare pattern sottili
                'cluster_selection_method': 'eom',  # Migliore per density
                'cluster_selection_epsilon': 0.22,  # Fine-tuned
                'umap_metric': 'cosine',  # Ideale per testo
                'umap_min_dist': 0.005,  # Molto basso per precision
                'umap_n_neighbors': 18,  # Ottimizzato dai dati  
                'umap_n_components': 50,  # Balanced
                'umap_random_state': 42
            },
            "expected": {
                "Outliers": "5-9%",
                "Clusters": "65-85", 
                "Silhouette": "0.10-0.15",
                "Quality": ">0.85"
            }
        },
        
        "OPTIMIZER_SUPREME": {
            "params": {
                'use_umap': True,
                'metric': 'euclidean', 
                'alpha': 0.35,  # Più aggressivo
                'min_cluster_size': 5,  # Piccolo per coverage
                'min_samples': 3,  # Minimo per sensibilità max
                'cluster_selection_method': 'leaf',  # Aggressivo
                'cluster_selection_epsilon': 0.25,
                'umap_metric': 'cosine',
                'umap_min_dist': 0.001,  # Minimo assoluto
                'umap_n_neighbors': 15,  # Ridotto per local patterns
                'umap_n_components': 55,  # Più informazione
                'umap_random_state': 42
            },
            "expected": {
                "Outliers": "3-7%",
                "Clusters": "75-95",
                "Silhouette": "0.09-0.14",
                "Quality": ">0.87"
            }
        }
    }
    
    print(f"🧪 TESTING {len(superior_configs)} configurazioni progettate...")
    print()
    
    results = []
    
    for config_name, config_data in superior_configs.items():
        print(f"\n{'='*70}")
        result = test_superior_clustering(
            config_name, 
            config_data["params"],
            config_data["expected"]
        )
        results.append(result)
        
        # Pausa tra test
        time.sleep(2)
    
    print("\n\n🏆 RISULTATI FINALI CONFIGURAZIONI SUPERIORI")
    print("=" * 80)
    print("📊 BASELINE V22: 12.9% outliers, 0.0836 silhouette, 51 clusters")
    print("=" * 80)
    
    best_config = None
    best_score = 0
    success_count = 0
    
    for result in results:
        if "error" in result:
            print(f"❌ {result['config_name']}: ERRORE - {result['error']}")
        else:
            success_count += 1
            config_name = result['config_name']
            outliers = result['outlier_ratio']
            silhouette = result['silhouette_score']
            clusters = result['n_clusters']
            quality = result['quality_score']
            
            # Calcola miglioramenti vs V22
            outlier_improvement = (0.129 - outliers) / 0.129 * 100
            silhouette_improvement = (silhouette - 0.0836) / 0.0836 * 100
            
            # Punteggio combinato miglioramento
            combined_score = outlier_improvement + silhouette_improvement
            
            print(f"\n🎯 {config_name}:")
            print(f"   📊 Dataset: {result['n_conversations']} conv → {clusters} clusters")
            print(f"   ❌ Outliers: {outliers:.1%} ({outlier_improvement:+.1f}% vs V22)")
            print(f"   📈 Silhouette: {silhouette:.4f} ({silhouette_improvement:+.1f}% vs V22)")
            print(f"   🏆 Quality Score: {quality:.4f}")
            print(f"   📊 Combined Improvement: {combined_score:+.1f}%")
            print(f"   ⏱️ Execution Time: {result['execution_time']:.1f}s")
            
            # Valutazione vs V22
            improvements = []
            if outliers < 0.129:
                improvements.append("OUTLIERS")
            if silhouette > 0.0836:
                improvements.append("SILHOUETTE")
            
            if len(improvements) == 2:
                print(f"   🌟🌟 SUPERIORE a V22 su TUTTE le metriche!")
                if combined_score > best_score:
                    best_config = config_name
                    best_score = combined_score
            elif len(improvements) == 1:
                print(f"   ⭐ Superiore su: {', '.join(improvements)}")
            else:
                print(f"   📊 Prestazioni comparabili a V22")
    
    print(f"\n🎖️ VERDICT FINALE:")
    print(f"   📈 Configurazioni testate: {success_count}/{len(superior_configs)}")
    if best_config:
        print(f"   👑 MIGLIORE: {best_config} (Score: +{best_score:.1f}%)")
        print(f"   🚀 PRONTA per deployment su dataset completo!")
    elif success_count > 0:
        print(f"   💡 Configurazioni progettate mostrano miglioramenti parziali")
        print(f"   🔧 Possibile ulteriore fine-tuning necessario")
    else:
        print(f"   ⚠️ Nessuna configurazione testata con successo")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
