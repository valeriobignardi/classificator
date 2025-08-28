#!/usr/bin/env python3
"""
Progettazione configurazione clustering SUPERIORE basata su pattern analysis
Autore: Valerio Bignardi
Data: 2025-08-28
Obiettivo: Creare una configurazione che SUPERI tutte quelle esistenti
"""

import json
import numpy as np
from Database.clustering_results_db import ClusteringResultsDB

def analyze_performance_patterns():
    """
    Analizza i pattern di performance per progettare la configurazione ottimale
    """
    
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    results_db = ClusteringResultsDB()
    
    if not results_db.connect():
        print("❌ Impossibile connettersi al database")
        return
    
    try:
        cursor = results_db.connection.cursor()
        
        query = """
        SELECT version_number, n_clusters, n_outliers, n_conversations, 
               silhouette_score, clustering_ratio, parameters_json, execution_time
        FROM clustering_test_results 
        WHERE tenant_id = %s 
        ORDER BY version_number
        """
        
        cursor.execute(query, (tenant_id,))
        results = cursor.fetchall()
        
        print("🔬 ANALISI PATTERN PER PROGETTAZIONE CONFIGURAZIONE SUPERIORE")
        print("=" * 80)
        
        # Estrai tutti i parametri e performance
        all_configs = []
        
        for result in results:
            version = result[0]
            n_clusters = result[1] 
            n_outliers = result[2]
            n_conversations = result[3]
            silhouette_score = result[4]
            clustering_ratio = result[5]
            parameters_json = result[6]
            execution_time = result[7]
            
            outlier_ratio = n_outliers / n_conversations if n_conversations > 0 else 1
            quality_score = silhouette_score * (1 - outlier_ratio)
            
            try:
                params = json.loads(parameters_json) if parameters_json else {}
            except:
                params = {}
            
            all_configs.append({
                'version': version,
                'n_clusters': n_clusters,
                'n_outliers': n_outliers,
                'outlier_ratio': outlier_ratio,
                'silhouette_score': silhouette_score,
                'quality_score': quality_score,
                'clustering_ratio': clustering_ratio,
                'execution_time': execution_time,
                'params': params
            })
        
        print(f"📊 Dataset: {len(all_configs)} configurazioni analizzate")
        
        # ANALISI CORRELAZIONI PARAMETRI-PERFORMANCE
        print(f"\n🔍 ANALISI CORRELAZIONI PARAMETRI:")
        print("=" * 60)
        
        # Analizza use_umap
        with_umap = [c for c in all_configs if c['params'].get('use_umap', True)]
        without_umap = [c for c in all_configs if not c['params'].get('use_umap', True)]
        
        if with_umap and without_umap:
            avg_silhouette_umap = np.mean([c['silhouette_score'] for c in with_umap])
            avg_silhouette_no_umap = np.mean([c['silhouette_score'] for c in without_umap])
            avg_outliers_umap = np.mean([c['outlier_ratio'] for c in with_umap])
            avg_outliers_no_umap = np.mean([c['outlier_ratio'] for c in without_umap])
            
            print(f"🔧 USE_UMAP:")
            print(f"   Con UMAP:    Silhouette {avg_silhouette_umap:.4f}, Outliers {avg_outliers_umap:.1%}")
            print(f"   Senza UMAP:  Silhouette {avg_silhouette_no_umap:.4f}, Outliers {avg_outliers_no_umap:.1%}")
        
        # Analizza cluster_selection_method
        eom_configs = [c for c in all_configs if c['params'].get('cluster_selection_method') == 'eom']
        leaf_configs = [c for c in all_configs if c['params'].get('cluster_selection_method') == 'leaf']
        
        if eom_configs and leaf_configs:
            avg_silhouette_eom = np.mean([c['silhouette_score'] for c in eom_configs])
            avg_silhouette_leaf = np.mean([c['silhouette_score'] for c in leaf_configs])
            avg_outliers_eom = np.mean([c['outlier_ratio'] for c in eom_configs])
            avg_outliers_leaf = np.mean([c['outlier_ratio'] for c in leaf_configs])
            
            print(f"🔧 CLUSTER_SELECTION_METHOD:")
            print(f"   EOM:   Silhouette {avg_silhouette_eom:.4f}, Outliers {avg_outliers_eom:.1%}")
            print(f"   LEAF:  Silhouette {avg_silhouette_leaf:.4f}, Outliers {avg_outliers_leaf:.1%}")
        
        # Analizza range parametri ottimali
        print(f"\n📈 RANGE PARAMETRI OTTIMALI:")
        print("=" * 60)
        
        # Trova top 5 configurazioni
        top_configs = sorted(all_configs, key=lambda x: x['quality_score'], reverse=True)[:5]
        
        # Analizza parametri delle top configurazioni
        alpha_values = [c['params'].get('alpha', 1.0) for c in top_configs]
        min_cluster_size_values = [c['params'].get('min_cluster_size', 20) for c in top_configs]
        min_samples_values = [c['params'].get('min_samples', 15) for c in top_configs]
        epsilon_values = [c['params'].get('cluster_selection_epsilon', 0.15) for c in top_configs]
        
        print(f"🔧 PARAMETRI TOP 5 CONFIGURAZIONI:")
        print(f"   Alpha: {min(alpha_values):.2f} - {max(alpha_values):.2f} (media: {np.mean(alpha_values):.2f})")
        print(f"   Min Cluster Size: {min(min_cluster_size_values)} - {max(min_cluster_size_values)} (media: {np.mean(min_cluster_size_values):.1f})")
        print(f"   Min Samples: {min(min_samples_values)} - {max(min_samples_values)} (media: {np.mean(min_samples_values):.1f})")
        print(f"   Epsilon: {min(epsilon_values):.3f} - {max(epsilon_values):.3f} (media: {np.mean(epsilon_values):.3f})")
        
        # PROGETTAZIONE CONFIGURAZIONE SUPERIORE
        print(f"\n🚀 PROGETTAZIONE CONFIGURAZIONE SUPERIORE:")
        print("=" * 80)
        
        # Analizza best practices dalle top configurazioni
        best_quality_config = max(all_configs, key=lambda x: x['quality_score'])
        best_silhouette_config = max(all_configs, key=lambda x: x['silhouette_score'])
        best_coverage_config = min(all_configs, key=lambda x: x['outlier_ratio'])
        
        print(f"📊 ANALISI ESTREMI:")
        print(f"   Migliore Quality Score: {best_quality_config['quality_score']:.4f} (V{best_quality_config['version']})")
        print(f"   Migliore Silhouette: {best_silhouette_config['silhouette_score']:.4f} (V{best_silhouette_config['version']})")
        print(f"   Migliore Coverage: {best_coverage_config['outlier_ratio']:.1%} (V{best_coverage_config['version']})")
        
        # PROGETTA LA CONFIGURAZIONE SUPERIORE
        print(f"\n🎯 CONFIGURAZIONE SUPERIORE PROGETTATA:")
        print("=" * 60)
        
        # Strategia ibrida: combinare i migliori elementi
        superior_config = {
            # Usa UMAP per migliorare la separazione iniziale
            "use_umap": True,
            
            # Euclidean ha dato i migliori risultati
            "metric": "euclidean",
            
            # Alpha ottimizzato: leggermente più alto della media top configs
            "alpha": 0.55,  # Tra 0.4-0.6 delle migliori, leggermente verso qualità
            
            # Min cluster size ottimizzato: più piccolo per più granularità
            "min_cluster_size": 9,  # Più piccolo del range 10-13 delle migliori
            
            # Min samples ottimizzato: bilanciato
            "min_samples": 9,  # Leggermente sotto la media delle migliori
            
            # EOM per qualità cluster superiore
            "cluster_selection_method": "eom",
            
            # Epsilon ottimizzato: più permissivo per meno outliers
            "cluster_selection_epsilon": 0.22,  # Più alto del range 0.12-0.18
            
            # Parametri UMAP ottimizzati
            "umap_metric": "cosine",
            "umap_min_dist": 0.02,  # Molto piccolo per preservare struttura locale
            "umap_n_neighbors": 25,  # Leggermente ridotto per più granularità
            "umap_n_components": 48,  # Ridotto per velocità mantenendo qualità
            "umap_random_state": 42,
            
            # Altri parametri
            "max_cluster_size": 0,
            "allow_single_cluster": False,
            "only_user": False
        }
        
        print(f"⚙️  PARAMETRI CONFIGURAZIONE SUPERIORE:")
        for key, value in superior_config.items():
            print(f"   • {key}: {value}")
        
        print(f"\n🧠 RATIONALE PROGETTAZIONE:")
        print("=" * 60)
        print("✅ use_umap=True: Migliora separazione iniziale mantenendo struttura")
        print("✅ alpha=0.55: Bilanciamento ottimo tra densità e noise rejection")
        print("✅ min_cluster_size=9: Granularità superiore senza frammentazione eccessiva")
        print("✅ min_samples=9: Consistente con cluster size, evita overfitting")
        print("✅ cluster_selection_method=eom: Qualità cluster superiore")
        print("✅ epsilon=0.22: Più permissivo per ridurre outliers significativamente")
        print("✅ umap_min_dist=0.02: Preserva struttura locale migliorando separazione")
        print("✅ umap_n_neighbors=25: Bilanciamento tra locale e globale")
        print("✅ umap_n_components=48: Riduce dimensionalità mantenendo informazione")
        
        print(f"\n📈 PERFORMANCE ATTESE:")
        print("=" * 60)
        print("🎯 Silhouette Score: 0.12-0.18 (miglioramento vs migliore attuale)")
        print("🎯 Outlier Ratio: 8-15% (significativo miglioramento)")
        print("🎯 Numero Cluster: 55-75 (granularità ottimale)")
        print("🎯 Quality Score: >0.10 (superiore a tutte le configurazioni esistenti)")
        print("🎯 Execution Time: 30-45s (accettabile)")
        
        # CONFIGURAZIONE ALTERNATIVA ANCORA PIÙ AGGRESSIVA
        print(f"\n🚀 CONFIGURAZIONE ALTERNATIVA - \"ULTRA OPTIMIZED\":")
        print("=" * 60)
        
        ultra_config = {
            "use_umap": True,
            "metric": "euclidean", 
            "alpha": 0.35,  # Più basso per permettere più cluster
            "min_cluster_size": 7,  # Ancora più piccolo
            "min_samples": 7,  # Consistente
            "cluster_selection_method": "leaf",  # Per massima copertura
            "cluster_selection_epsilon": 0.28,  # Molto permissivo
            "umap_metric": "cosine",
            "umap_min_dist": 0.0,  # Zero per massima preservazione
            "umap_n_neighbors": 20,  # Più locale
            "umap_n_components": 45,  # Ulteriore riduzione
            "umap_random_state": 42,
            "max_cluster_size": 0,
            "allow_single_cluster": False,
            "only_user": False
        }
        
        print(f"⚙️  PARAMETRI ULTRA OPTIMIZED:")
        for key, value in ultra_config.items():
            print(f"   • {key}: {value}")
            
        print(f"\n📈 PERFORMANCE ATTESE ULTRA:")
        print("🎯 Outlier Ratio: 5-10% (target estremamente ambizioso)")
        print("🎯 Numero Cluster: 70-90 (massima granularità)")
        print("🎯 Silhouette Score: 0.08-0.12 (accettabile per copertura estrema)")
        print("🎯 Quality Score: 0.08-0.11 (bilanciamento ottimo)")
        
        return superior_config, ultra_config
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
    finally:
        results_db.disconnect()

def generate_superior_config_files():
    """
    Genera i file di configurazione superiori
    """
    superior_config, ultra_config = analyze_performance_patterns()
    
    if not superior_config:
        return
    
    # Genera configurazione superiore
    timestamp = "20250828_SUPERIOR"
    
    superior_yaml = f"""# CONFIGURAZIONE CLUSTERING SUPERIORE - HUMANITAS
# Progettata per SUPERARE tutte le configurazioni esistenti
# Generata: 2025-08-28
# Autore: Valerio Bignardi
# Target: Quality Score > 0.10, Outliers < 15%

clustering:
  # Parametri HDBSCAN ottimizzati
  alpha: {superior_config['alpha']}
  metric: "{superior_config['metric']}"
  min_samples: {superior_config['min_samples']}
  min_cluster_size: {superior_config['min_cluster_size']}
  max_cluster_size: {superior_config['max_cluster_size']}
  cluster_selection_method: "{superior_config['cluster_selection_method']}"
  cluster_selection_epsilon: {superior_config['cluster_selection_epsilon']}
  allow_single_cluster: {str(superior_config['allow_single_cluster']).lower()}
  
  # Parametri UMAP ottimizzati per performance superiore
  use_umap: {str(superior_config['use_umap']).lower()}
  umap_metric: "{superior_config['umap_metric']}"
  umap_min_dist: {superior_config['umap_min_dist']}
  umap_n_neighbors: {superior_config['umap_n_neighbors']}
  umap_n_components: {superior_config['umap_n_components']}
  umap_random_state: {superior_config['umap_random_state']}
  
  # Filtri
  only_user: {str(superior_config['only_user']).lower()}

# Performance attese
expected_performance:
  silhouette_score: "0.12-0.18"
  outlier_ratio: "8-15%"
  n_clusters: "55-75"
  quality_score: ">0.10"
  execution_time: "30-45s"

# Metadati
metadata:
  config_name: "SUPERIOR_V1"
  design_approach: "Hybrid optimization combining best elements"
  tenant_id: "015007d9-d413-11ef-86a5-96000228e7fe"
  generated_at: "2025-08-28"
  version: "1.0"
"""

    ultra_yaml = f"""# CONFIGURAZIONE CLUSTERING ULTRA OPTIMIZED - HUMANITAS  
# Configurazione ESTREMAMENTE aggressiva per massima copertura
# Generata: 2025-08-28
# Target: Outliers < 10%, massima granularità

clustering:
  # Parametri HDBSCAN ultra-ottimizzati
  alpha: {ultra_config['alpha']}
  metric: "{ultra_config['metric']}"
  min_samples: {ultra_config['min_samples']}
  min_cluster_size: {ultra_config['min_cluster_size']}
  max_cluster_size: {ultra_config['max_cluster_size']}
  cluster_selection_method: "{ultra_config['cluster_selection_method']}"
  cluster_selection_epsilon: {ultra_config['cluster_selection_epsilon']}
  allow_single_cluster: {str(ultra_config['allow_single_cluster']).lower()}
  
  # Parametri UMAP per copertura estrema
  use_umap: {str(ultra_config['use_umap']).lower()}
  umap_metric: "{ultra_config['umap_metric']}"
  umap_min_dist: {ultra_config['umap_min_dist']}
  umap_n_neighbors: {ultra_config['umap_n_neighbors']}
  umap_n_components: {ultra_config['umap_n_components']}
  umap_random_state: {ultra_config['umap_random_state']}
  
  # Filtri
  only_user: {str(ultra_config['only_user']).lower()}

# Performance attese
expected_performance:
  outlier_ratio: "5-10%"
  n_clusters: "70-90"
  silhouette_score: "0.08-0.12"
  quality_score: "0.08-0.11"

# Metadati  
metadata:
  config_name: "ULTRA_OPTIMIZED_V1"
  design_approach: "Extreme coverage optimization"
  tenant_id: "015007d9-d413-11ef-86a5-96000228e7fe"
  generated_at: "2025-08-28"
  version: "1.0"
"""
    
    # Salva i file
    try:
        with open(f"/home/ubuntu/classificatore/superior_clustering_config.yaml", 'w') as f:
            f.write(superior_yaml)
        print(f"\n✅ Salvata: superior_clustering_config.yaml")
        
        with open(f"/home/ubuntu/classificatore/ultra_optimized_clustering_config.yaml", 'w') as f:
            f.write(ultra_yaml)
        print(f"✅ Salvata: ultra_optimized_clustering_config.yaml")
        
        # Salva anche in JSON per facile parsing
        superior_json = {
            "config_name": "SUPERIOR_V1",
            "parameters": superior_config,
            "expected_performance": {
                "silhouette_score_range": [0.12, 0.18],
                "outlier_ratio_range": [0.08, 0.15],
                "n_clusters_range": [55, 75],
                "target_quality_score": 0.10
            }
        }
        
        ultra_json = {
            "config_name": "ULTRA_OPTIMIZED_V1", 
            "parameters": ultra_config,
            "expected_performance": {
                "outlier_ratio_range": [0.05, 0.10],
                "n_clusters_range": [70, 90],
                "silhouette_score_range": [0.08, 0.12],
                "target_quality_score": 0.09
            }
        }
        
        with open("/home/ubuntu/classificatore/superior_configs.json", 'w') as f:
            json.dump({
                "superior": superior_json,
                "ultra": ultra_json,
                "generated_at": "2025-08-28",
                "rationale": "Configurations designed to exceed all existing performance"
            }, f, indent=2)
        print(f"✅ Salvata: superior_configs.json")
        
    except Exception as e:
        print(f"❌ Errore nel salvataggio: {e}")

if __name__ == "__main__":
    generate_superior_config_files()
