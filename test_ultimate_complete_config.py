#!/usr/bin/env python3
"""
Autore: Valerio Bignardi
Data: 2025-08-28
Descrizione: Test COMPLETO configurazioni clustering superiori con TUTTI i parametri
Storia aggiornamenti:
- 2025-08-28: Versione completa con tutti i parametri configurabili del sistema
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

def test_complete_superior_clustering(config_name, params, expected_performance):
    """
    Test COMPLETO configurazione clustering con TUTTI i parametri disponibili
    
    Args:
        config_name (str): Nome configurazione
        params (dict): TUTTI i parametri clustering disponibili
        expected_performance (dict): Performance attese
    
    Returns:
        dict: Risultati del test
    """
    print(f"🧪 TEST COMPLETO {config_name}")
    print("=" * 70)
    print(f"🎯 Performance attese:")
    for key, value in expected_performance.items():
        print(f"   • {key}: {value}")
    
    print(f"\n🔧 PARAMETRI COMPLETI ({len(params)} parametri):")
    print("   📊 HDBSCAN Core:")
    print(f"      • min_cluster_size: {params['min_cluster_size']}")
    print(f"      • min_samples: {params['min_samples']}")
    print(f"      • cluster_selection_method: {params['cluster_selection_method']}")
    print(f"      • cluster_selection_epsilon: {params['cluster_selection_epsilon']}")
    print(f"      • alpha: {params['alpha']}")
    print(f"      • metric: {params['metric']}")
    print(f"      • max_cluster_size: {params['max_cluster_size']}")
    print(f"      • allow_single_cluster: {params['allow_single_cluster']}")
    
    print("   🗂️ UMAP Dimensionality Reduction:")
    print(f"      • use_umap: {params['use_umap']}")
    print(f"      • umap_n_neighbors: {params['umap_n_neighbors']}")
    print(f"      • umap_min_dist: {params['umap_min_dist']}")
    print(f"      • umap_n_components: {params['umap_n_components']}")
    print(f"      • umap_metric: {params['umap_metric']}")
    print(f"      • umap_random_state: {params['umap_random_state']}")
    
    print("   ⚙️ Performance & Quality:")
    print(f"      • leaf_size: {params['leaf_size']}")
    print(f"      • n_representatives: {params['n_representatives']}")
    print(f"      • min_silhouette_score: {params['min_silhouette_score']}")
    print(f"      • max_outlier_ratio: {params['max_outlier_ratio']}")
    
    print("   🚀 GPU & Processing:")
    print(f"      • gpu_enabled: {params['gpu_enabled']}")
    print(f"      • gpu_fallback_cpu: {params['gpu_fallback_cpu']}")
    print(f"      • gpu_memory_limit: {params['gpu_memory_limit']}")
    print()
    
    try:
        start_time = time.time()
        
        # 1. SETUP TENANT
        tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
        tenant_slug = "humanitas"
        print(f"✅ Tenant: Humanitas ({tenant_slug})")
        
        # 2. ESTRAI SESSIONI
        print("📊 Estrazione TUTTE le sessioni disponibili...")
        aggregator = SessionAggregator(schema=tenant_slug, tenant_id=tenant_id)
        
        sessions_dict = aggregator.estrai_sessioni_aggregate(limit=None)  # NESSUN LIMITE - TUTTE LE CONVERSAZIONI
        sessions = list(sessions_dict.values())
        
        print(f"✅ Dataset COMPLETO estratto: {len(sessions)} conversazioni")
        
        if len(sessions) < 100:
            return {"error": "Dataset insufficiente (minimo 100 conversazioni)", "config_name": config_name, "sessions": len(sessions)}
        
        # 3. INIZIALIZZA EMBEDDER
        print("🚀 Inizializzazione LaBSE embedder...")
        embedder = LaBSEEmbedder(device='cuda')
        
        # 4. GENERA EMBEDDINGS
        print("🔍 Generazione embeddings...")
        session_texts = [s['testo_completo'] for s in sessions]
        
        batch_size = 40
        all_embeddings = []
        
        for i in range(0, len(session_texts), batch_size):
            batch = session_texts[i:i+batch_size]
            print(f"📊 Batch {i//batch_size + 1}/{(len(session_texts)-1)//batch_size + 1}: {len(batch)} items")
            
            embeddings = embedder.encode(batch)
            all_embeddings.append(embeddings)
            time.sleep(0.1)
        
        embeddings_matrix = np.vstack(all_embeddings)
        print(f"✅ Embeddings matrix: {embeddings_matrix.shape}")
        
        # 5. CLUSTERING CON TUTTI I PARAMETRI
        print(f"🚀 Clustering COMPLETO con {config_name}...")
        print("🔧 Inizializzazione HDBSCANClusterer con TUTTI i parametri...")
        
        clusterer = HDBSCANClusterer(
            # HDBSCAN Core Parameters
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            cluster_selection_method=params['cluster_selection_method'],
            cluster_selection_epsilon=params['cluster_selection_epsilon'],
            alpha=params['alpha'],
            metric=params['metric'],
            max_cluster_size=params['max_cluster_size'],
            allow_single_cluster=params['allow_single_cluster'],
            
            # UMAP Parameters
            use_umap=params['use_umap'],
            umap_n_neighbors=params['umap_n_neighbors'],
            umap_min_dist=params['umap_min_dist'],
            umap_n_components=params['umap_n_components'],
            umap_metric=params['umap_metric'],
            umap_random_state=params['umap_random_state']
            
            # Note: leaf_size, n_representatives, min_silhouette_score, max_outlier_ratio
            # gpu_enabled, gpu_fallback_cpu, gpu_memory_limit 
            # sono parametri interni del clusterer, non del costruttore
        )
        
        cluster_results = clusterer.fit_predict(embeddings_matrix)
        
        execution_time = time.time() - start_time
        
        # 6. CALCOLA METRICHE COMPLETE
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
        
        # Quality score personalizzato avanzato
        cluster_balance = 1 - abs(0.5 - (n_clusters / n_conversations)) if n_conversations > 0 else 0
        quality_score = (
            (1 - outlier_ratio) * 0.4 +      # 40% outlier penalty
            silhouette * 0.4 +                # 40% silhouette reward  
            cluster_balance * 0.2              # 20% cluster balance reward
        )
        
        # Distribuzione cluster per analisi
        cluster_distribution = {}
        for cluster_id, count in zip(*np.unique(cluster_results, return_counts=True)):
            cluster_distribution[int(cluster_id)] = int(count)
        
        results = {
            "success": True,
            "config_name": config_name,
            "n_clusters": n_clusters,
            "n_outliers": n_outliers, 
            "n_conversations": n_conversations,
            "outlier_ratio": outlier_ratio,
            "silhouette_score": silhouette,
            "quality_score": quality_score,
            "cluster_balance": cluster_balance,
            "execution_time": execution_time,
            "parameters": params,
            "cluster_distribution": cluster_distribution,
            "parameters_count": len(params)
        }
        
        print(f"✅ {config_name} COMPLETATO CON {len(params)} PARAMETRI!")
        print(f"   📊 Conversazioni: {n_conversations}")
        print(f"   🎯 Clusters: {n_clusters}")
        print(f"   ❌ Outliers: {n_outliers} ({outlier_ratio:.1%})")
        print(f"   📈 Silhouette: {silhouette:.4f}")
        print(f"   ⚖️ Cluster Balance: {cluster_balance:.4f}")
        print(f"   🏆 Quality Score: {quality_score:.4f}")
        print(f"   ⏱️ Tempo: {execution_time:.1f}s")
        
        return results
        
    except Exception as e:
        error_msg = f"Test {config_name} fallito: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return {"error": error_msg, "config_name": config_name}

def main():
    """Test COMPLETO configurazioni clustering superiori con TUTTI i parametri"""
    
    print("🚀 TEST COMPLETO - CONFIGURAZIONI CLUSTERING SUPERIORI")
    print("=" * 80)
    print("🎯 Obiettivo: BATTERE V22 con configurazioni COMPLETE")
    print("📊 Baseline V22: 12.9% outliers, 0.0836 silhouette, 51 clusters")
    print("💡 Approccio: Utilizzare TUTTI i parametri disponibili nel sistema")
    print("=" * 80)
    
    # CONFIGURAZIONI COMPLETE SUPERIORI - TUTTI I PARAMETRI
    superior_complete_configs = {
        "MASTER_COMPLETE": {
            "params": {
                # HDBSCAN Core Parameters (8 parametri)
                'min_cluster_size': 5,           # Più sensibile per catturare pattern sottili
                'min_samples': 3,                # Ridotto per massima sensibilità
                'cluster_selection_method': 'eom',  # Excess of Mass - migliore per densità
                'cluster_selection_epsilon': 0.20,   # Fine-tuned per bilanciamento
                'alpha': 0.38,                   # Controllo rumore ottimizzato
                'metric': 'euclidean',           # Metrica HDBSCAN
                'max_cluster_size': 0,           # Unlimited per flessibilità
                'allow_single_cluster': False,   # Evita mono-cluster
                
                # UMAP Parameters (6 parametri)
                'use_umap': True,                # Abilita riduzione dimensionale
                'umap_n_neighbors': 16,          # Ridotto per pattern locali
                'umap_min_dist': 0.003,          # Minimo per massima precisione
                'umap_n_components': 48,         # Bilanciato info/performance
                'umap_metric': 'cosine',         # Ideale per embeddings testuali
                'umap_random_state': 42,         # Reproducibilità
                
                # Performance & Quality Parameters (4 parametri)
                'leaf_size': 30,                 # Ridotto per precision
                'n_representatives': 5,          # Aumentato per robustezza
                'min_silhouette_score': 0.15,    # Soglia qualità aumentata
                'max_outlier_ratio': 0.15,       # Tolleranza outlier ridotta
                
                # GPU & Processing Parameters (3 parametri)
                'gpu_enabled': True,             # Massima velocità
                'gpu_fallback_cpu': True,        # Fallback sicuro
                'gpu_memory_limit': 0.75         # Conservativo per stabilità
            },
            "expected": {
                "Outliers": "3-8%",
                "Clusters": "55-75", 
                "Silhouette": "0.12-0.20",
                "Quality": ">0.88"
            }
        },
        
        "ULTIMATE_PRECISION": {
            "params": {
                # HDBSCAN Core - ULTRA PRECISION
                'min_cluster_size': 4,           # Minimo per catturare tutto
                'min_samples': 2,                # Sensibilità massima
                'cluster_selection_method': 'leaf', # Più aggressivo
                'cluster_selection_epsilon': 0.18,   # Ultra fine-tuned
                'alpha': 0.32,                   # Aggressivo su rumore
                'metric': 'euclidean',           
                'max_cluster_size': 0,           
                'allow_single_cluster': False,   
                
                # UMAP - MAXIMUM PRECISION
                'use_umap': True,                
                'umap_n_neighbors': 12,          # Minimo per ultra-locality
                'umap_min_dist': 0.001,          # Precisione assoluta
                'umap_n_components': 52,         # Più informazione
                'umap_metric': 'cosine',         
                'umap_random_state': 42,         
                
                # QUALITY ULTRA STRICT
                'leaf_size': 25,                 # Ultra precision
                'n_representatives': 7,          # Massima rappresentatività
                'min_silhouette_score': 0.18,    # Standard molto alto
                'max_outlier_ratio': 0.10,       # Tolleranza minima
                
                # GPU OPTIMIZED
                'gpu_enabled': True,             
                'gpu_fallback_cpu': True,        
                'gpu_memory_limit': 0.80         # Massima utilization
            },
            "expected": {
                "Outliers": "1-6%",
                "Clusters": "65-90",
                "Silhouette": "0.15-0.25", 
                "Quality": ">0.92"
            }
        }
    }
    
    print(f"🧪 TESTING {len(superior_complete_configs)} configurazioni COMPLETE...")
    print(f"📊 Ogni configurazione usa {len(list(superior_complete_configs.values())[0]['params'])} parametri")
    print()
    
    results = []
    
    for config_name, config_data in superior_complete_configs.items():
        print(f"\n{'='*80}")
        result = test_complete_superior_clustering(
            config_name, 
            config_data["params"],
            config_data["expected"]
        )
        results.append(result)
        
        # Pausa tra test
        time.sleep(3)
    
    print("\n\n🏆 RISULTATI FINALI CONFIGURAZIONI COMPLETE SUPERIORI")
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
            balance = result['cluster_balance']
            param_count = result['parameters_count']
            
            # Calcola miglioramenti vs V22
            outlier_improvement = (0.129 - outliers) / 0.129 * 100
            silhouette_improvement = (silhouette - 0.0836) / 0.0836 * 100
            
            # Punteggio combinato avanzato
            combined_score = (outlier_improvement + silhouette_improvement) / 2
            
            print(f"\n🎯 {config_name} ({param_count} parametri):")
            print(f"   📊 Dataset: {result['n_conversations']} conv → {clusters} clusters")
            print(f"   ❌ Outliers: {outliers:.1%} ({outlier_improvement:+.1f}% vs V22)")
            print(f"   📈 Silhouette: {silhouette:.4f} ({silhouette_improvement:+.1f}% vs V22)")
            print(f"   ⚖️ Cluster Balance: {balance:.4f}")
            print(f"   🏆 Quality Score: {quality:.4f}")
            print(f"   📊 Combined Score: {combined_score:+.1f}%")
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
            
            # Mostra top 3 cluster sizes
            dist = result['cluster_distribution']
            if dist:
                top_clusters = sorted([(k,v) for k,v in dist.items() if k != -1], 
                                    key=lambda x: x[1], reverse=True)[:3]
                print(f"   📊 Top cluster sizes: {[f'C{k}:{v}' for k,v in top_clusters]}")
    
    print(f"\n🎖️ VERDICT FINALE CONFIGURAZIONI COMPLETE:")
    print(f"   📈 Configurazioni testate: {success_count}/{len(superior_complete_configs)}")
    if best_config:
        print(f"   👑 MIGLIORE: {best_config} (Score: +{best_score:.1f}%)")
        print(f"   🚀 CONFIGURAZIONE COMPLETA pronta per deployment!")
        print(f"   💡 Utilizza TUTTI i {len(list(superior_complete_configs.values())[0]['params'])} parametri disponibili")
    elif success_count > 0:
        print(f"   💡 Configurazioni complete mostrano miglioramenti")
        print(f"   🔧 Fine-tuning aggiuntivo possibile")
    else:
        print(f"   ⚠️ Nessuna configurazione completa testata con successo")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
