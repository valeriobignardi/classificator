#!/usr/bin/env python3
"""
Raccomandazioni configurazioni clustering ottimali per Humanitas
Autore: Valerio Bignardi  
Data: 2025-08-28
Obiettivo: Suggerire configurazioni ottimizzate basate sui test eseguiti
"""

import json
import numpy as np
from Database.clustering_results_db import ClusteringResultsDB

def generate_optimal_configurations():
    """
    Analizza i risultati e propone configurazioni ottimizzate
    """
    
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    results_db = ClusteringResultsDB()
    
    if not results_db.connect():
        print("❌ Impossibile connettersi al database")
        return
    
    try:
        cursor = results_db.connection.cursor()
        
        # Query risultati
        query = """
        SELECT version_number, n_clusters, n_outliers, n_conversations, 
               silhouette_score, clustering_ratio, parameters_json, execution_time
        FROM clustering_test_results 
        WHERE tenant_id = %s 
        ORDER BY version_number
        """
        
        cursor.execute(query, (tenant_id,))
        results = cursor.fetchall()
        
        print("🎯 ANALISI CONFIGURAZIONI CLUSTERING - HUMANITAS")
        print("=" * 80)
        
        # Analizza le migliori configurazioni
        best_configs = []
        
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
            
            best_configs.append({
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
        
        # Ordina per quality score
        best_configs.sort(key=lambda x: x['quality_score'], reverse=True)
        
        print("📊 CONFIGURAZIONI ANALIZZATE:")
        print(f"• Totale configurazioni testate: {len(best_configs)}")
        print(f"• Range Silhouette Score: {min(c['silhouette_score'] for c in best_configs):.4f} - {max(c['silhouette_score'] for c in best_configs):.4f}")
        print(f"• Range Outlier Ratio: {min(c['outlier_ratio'] for c in best_configs):.1%} - {max(c['outlier_ratio'] for c in best_configs):.1%}")
        print(f"• Range Cluster Count: {min(c['n_clusters'] for c in best_configs)} - {max(c['n_clusters'] for c in best_configs)}")
        
        print("\n🏆 TOP 3 CONFIGURAZIONI CONSIGLIATE:")
        print("=" * 80)
        
        for i, config in enumerate(best_configs[:3], 1):
            print(f"\n🥇 CONFIGURAZIONE #{i} - Versione {config['version']}")
            print(f"   📊 METRICHE:")
            print(f"      • Quality Score: {config['quality_score']:.4f}")
            print(f"      • Silhouette Score: {config['silhouette_score']:.4f}")
            print(f"      • Cluster creati: {config['n_clusters']}")
            print(f"      • Outliers: {config['n_outliers']} ({config['outlier_ratio']:.1%})")
            print(f"      • Clustering Ratio: {config['clustering_ratio']:.3f}")
            print(f"      • Execution Time: {config['execution_time']:.2f}s")
            
            print(f"   ⚙️  PARAMETRI:")
            params = config['params']
            key_params = ['use_umap', 'metric', 'min_cluster_size', 'min_samples', 
                         'cluster_selection_method', 'cluster_selection_epsilon', 'alpha']
            
            for key in key_params:
                if key in params:
                    print(f"      • {key}: {params[key]}")
            
            # Valutazione pro/contro
            print(f"   ✅ PRO:")
            if config['silhouette_score'] > 0.1:
                print(f"      • Buona qualità cluster (Silhouette: {config['silhouette_score']:.3f})")
            if config['outlier_ratio'] < 0.3:
                print(f"      • Bassa percentuale outliers ({config['outlier_ratio']:.1%})")
            if config['execution_time'] < 300:
                print(f"      • Tempo di esecuzione accettabile ({config['execution_time']:.1f}s)")
            if 20 <= config['n_clusters'] <= 80:
                print(f"      • Numero clusters bilanciato ({config['n_clusters']})")
                
            print(f"   ⚠️  CONTRO:")
            if config['silhouette_score'] < 0.15:
                print(f"      • Qualità cluster migliorabile (Silhouette < 0.15)")
            if config['outlier_ratio'] > 0.4:
                print(f"      • Alta percentuale outliers ({config['outlier_ratio']:.1%})")
            if config['n_clusters'] < 15:
                print(f"      • Pochi clusters potrebbero perdere granularità")
            if config['n_clusters'] > 100:
                print(f"      • Troppi clusters potrebbero creare confusione")
        
        print("\n🔬 CONFIGURAZIONI OTTIMIZZATE PROPOSTE:")
        print("=" * 80)
        
        # Proponi configurazioni ottimizzate basate sui migliori pattern
        best_config = best_configs[0]
        
        print("\n🎯 CONFIGURAZIONE OTTIMIZZATA #1 - Per QUALITÀ MASSIMA:")
        optimized_config_1 = {
            "use_umap": False,  # Migliori risultati senza UMAP
            "metric": "euclidean",
            "min_cluster_size": 12,  # Slightly smaller for more clusters
            "min_samples": 10,  # Reduce for better coverage
            "cluster_selection_method": "eom",
            "cluster_selection_epsilon": 0.12,  # Slightly tighter
            "alpha": 0.6,  # Balance between noise and clusters
            "umap_metric": "cosine",
            "umap_min_dist": 0.1,
            "max_cluster_size": 0,
            "umap_n_neighbors": 30,
            "umap_n_components": 52,
            "umap_random_state": 42,
            "allow_single_cluster": False,
            "only_user": False
        }
        
        print("   ⚙️  Parametri:")
        for key, value in optimized_config_1.items():
            print(f"      • {key}: {value}")
        
        print(f"\n   🎯 Obiettivo: Massimizzare Silhouette Score (target > 0.20)")
        print(f"   📈 Risultato atteso: ~25-30 clusters, ~35-45% outliers")
        
        print("\n🎯 CONFIGURAZIONE OTTIMIZZATA #2 - Per COPERTURA MASSIMA:")
        optimized_config_2 = {
            "use_umap": True,
            "metric": "euclidean", 
            "min_cluster_size": 8,  # Smaller clusters
            "min_samples": 6,  # Lower threshold
            "cluster_selection_method": "leaf",
            "cluster_selection_epsilon": 0.25,  # More permissive
            "alpha": 0.3,  # Lower alpha for more clusters
            "umap_metric": "cosine",
            "umap_min_dist": 0.0,
            "max_cluster_size": 0,
            "umap_n_neighbors": 30,
            "umap_n_components": 52,
            "umap_random_state": 42,
            "allow_single_cluster": False,
            "only_user": False
        }
        
        print("   ⚙️  Parametri:")
        for key, value in optimized_config_2.items():
            print(f"      • {key}: {value}")
            
        print(f"\n   🎯 Obiettivo: Minimizzare outliers (target < 20%)")
        print(f"   📈 Risultato atteso: ~60-80 clusters, ~15-25% outliers")
        
        print("\n🎯 CONFIGURAZIONE OTTIMIZZATA #3 - BILANCIATA:")
        optimized_config_3 = {
            "use_umap": True,
            "metric": "euclidean",
            "min_cluster_size": 10,  # Balanced
            "min_samples": 8,  # Balanced  
            "cluster_selection_method": "eom",
            "cluster_selection_epsilon": 0.18,  # Balanced
            "alpha": 0.45,  # Balanced
            "umap_metric": "cosine",
            "umap_min_dist": 0.05,
            "max_cluster_size": 0,
            "umap_n_neighbors": 30,
            "umap_n_components": 52,
            "umap_random_state": 42,
            "allow_single_cluster": False,
            "only_user": False
        }
        
        print("   ⚙️  Parametri:")
        for key, value in optimized_config_3.items():
            print(f"      • {key}: {value}")
            
        print(f"\n   🎯 Obiettivo: Bilanciare qualità e copertura")
        print(f"   📈 Risultato atteso: ~40-50 clusters, ~25-30% outliers, Silhouette ~0.10-0.15")
        
        print(f"\n📋 PIANO DI TEST CONSIGLIATO:")
        print("=" * 50)
        print("1. ✅ Testare configurazione BILANCIATA (#3) per baseline")
        print("2. 🔬 Se silhouette troppo basso, provare configurazione QUALITÀ (#1)")
        print("3. 📈 Se troppi outliers, provare configurazione COPERTURA (#2)")
        print("4. 🎯 Fine-tuning incrementale sui parametri migliori")
        
        print(f"\n🎯 RACCOMANDAZIONE FINALE:")
        print("=" * 50)
        
        if best_config['quality_score'] > 0.08:
            print("✅ La configurazione attuale (Versione {}) è già BUONA".format(best_config['version']))
            print("   Suggerisco piccoli aggiustamenti per ottimizzare ulteriormente")
        else:
            print("⚠️  La configurazione attuale ha margini di miglioramento")
            print("   Raccomando di testare le configurazioni ottimizzate proposte")
        
        print(f"\n📊 TARGET OBIETTIVI:")
        print(f"• Silhouette Score target: > 0.15 (ottimo > 0.20)")
        print(f"• Outlier Ratio target: < 25% (ottimo < 15%)")  
        print(f"• Numero cluster target: 30-60 (bilanciato)")
        print(f"• Execution time target: < 5 minuti")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        results_db.disconnect()

if __name__ == "__main__":
    generate_optimal_configurations()
