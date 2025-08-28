#!/usr/bin/env python3
"""
Test configurazioni clustering SUPERIORI
Autore: Valerio Bignardi  
Data: 2025-08-28
Obiettivo: Testare le configurazioni progettate per superare le esistenti
"""

import yaml
import json
import sys
import os
from datetime import datetime

# Aggiungi il path del progetto
sys.path.append('/home/ubuntu/classificatore')

from Clustering.clustering_test_service_new import ClusteringTestService
from Database.clustering_results_db import ClusteringResultsDB

def test_superior_configurations():
    """
    Testa entrambe le configurazioni superiori progettate
    """
    
    print("🚀 TEST CONFIGURAZIONI CLUSTERING SUPERIORI")
    print("=" * 80)
    print("🎯 Obiettivo: SUPERARE tutte le configurazioni esistenti")
    print("📊 Migliore attuale: Quality Score 0.0854 (V16)")
    print("=" * 80)
    
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    
    # Carica le configurazioni progettate
    try:
        with open("/home/ubuntu/classificatore/superior_configs.json", 'r') as f:
            configs_data = json.load(f)
    except Exception as e:
        print(f"❌ Errore caricamento configurazioni: {e}")
        return
    
    results_db = ClusteringResultsDB()
    if not results_db.connect():
        print("❌ Impossibile connettersi al database")
        return
    
    try:
        # Ottieni prossimo version number
        cursor = results_db.connection.cursor()
        cursor.execute("SELECT MAX(version_number) FROM clustering_test_results WHERE tenant_id = %s", (tenant_id,))
        max_version = cursor.fetchone()[0] or 0
        next_version = max_version + 1
        
        print(f"📍 Prossimo version number: {next_version}")
        
        # TEST 1: CONFIGURAZIONE SUPERIOR
        print(f"\n🧪 TEST 1 - CONFIGURAZIONE SUPERIOR V1")
        print("=" * 60)
        
        superior_config = configs_data["superior"]["parameters"]
        expected_perf = configs_data["superior"]["expected_performance"]
        
        print(f"🎯 Performance attese:")
        print(f"   • Quality Score: >{expected_perf['target_quality_score']}")
        print(f"   • Outliers: {expected_perf['outlier_ratio_range'][0]*100:.0f}-{expected_perf['outlier_ratio_range'][1]*100:.0f}%")
        print(f"   • Silhouette: {expected_perf['silhouette_score_range'][0]:.3f}-{expected_perf['silhouette_score_range'][1]:.3f}")
        print(f"   • Clusters: {expected_perf['n_clusters_range'][0]}-{expected_perf['n_clusters_range'][1]}")
        
        # Inizializza clustering test service
        clustering_service = ClusteringTestService()
        
        print(f"\n▶️  Avvio clustering SUPERIOR V1...")
        start_time = datetime.now()
        
        try:
            results = clustering_service.run_clustering_test(
                tenant_id=tenant_id,
                custom_parameters=superior_config,
                sample_size=None  # Usa tutto il dataset
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            if results and 'cluster_labels' in results:
                # Calcola metriche
                n_conversations = len(results['cluster_labels'])
                n_outliers = sum(1 for label in results['cluster_labels'] if label == -1)
                n_clusters = len(set(results['cluster_labels'])) - (1 if -1 in results['cluster_labels'] else 0)
                outlier_ratio = n_outliers / n_conversations if n_conversations > 0 else 1
                silhouette_score = results.get('silhouette_score', 0)
                clustering_ratio = (n_conversations - n_outliers) / n_conversations if n_conversations > 0 else 0
                quality_score = silhouette_score * (1 - outlier_ratio)
                
                print(f"\n✅ RISULTATI SUPERIOR V1:")
                print(f"   🎯 Quality Score: {quality_score:.4f}")
                print(f"   📊 Silhouette Score: {silhouette_score:.4f}")
                print(f"   🔍 Outliers: {n_outliers}/{n_conversations} ({outlier_ratio:.1%})")
                print(f"   🧩 Cluster: {n_clusters}")
                print(f"   ⏱️  Tempo: {execution_time:.1f}s")
                
                # Confronta con target
                if quality_score > expected_perf['target_quality_score']:
                    print(f"   🎉 SUPERIOR: Quality Score SUPERA il target!")
                else:
                    print(f"   ⚠️  Quality Score sotto target ({expected_perf['target_quality_score']})")
                
                if expected_perf['outlier_ratio_range'][0] <= outlier_ratio <= expected_perf['outlier_ratio_range'][1]:
                    print(f"   ✅ Outliers nel range atteso")
                else:
                    print(f"   ⚠️  Outliers fuori range atteso")
                
                # Salva nel database
                cursor.execute("""
                    INSERT INTO clustering_test_results 
                    (tenant_id, version_number, n_clusters, n_outliers, n_conversations, 
                     silhouette_score, clustering_ratio, parameters_json, execution_time, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """, (tenant_id, next_version, n_clusters, n_outliers, n_conversations,
                      silhouette_score, clustering_ratio, json.dumps(superior_config), execution_time))
                
                results_db.connection.commit()
                print(f"   💾 Salvato come Version {next_version}")
                
                next_version += 1
                
            else:
                print(f"   ❌ Clustering fallito per SUPERIOR V1")
                
        except Exception as e:
            print(f"   ❌ Errore durante clustering SUPERIOR: {e}")
            import traceback
            traceback.print_exc()
        
        # TEST 2: CONFIGURAZIONE ULTRA OPTIMIZED  
        print(f"\n🧪 TEST 2 - CONFIGURAZIONE ULTRA OPTIMIZED V1")
        print("=" * 60)
        
        ultra_config = configs_data["ultra"]["parameters"]
        ultra_expected = configs_data["ultra"]["expected_performance"]
        
        print(f"🎯 Performance attese (ULTRA):")
        print(f"   • Outliers: {ultra_expected['outlier_ratio_range'][0]*100:.0f}-{ultra_expected['outlier_ratio_range'][1]*100:.0f}%")
        print(f"   • Clusters: {ultra_expected['n_clusters_range'][0]}-{ultra_expected['n_clusters_range'][1]}")
        print(f"   • Quality Score: {ultra_expected['target_quality_score']}")
        
        print(f"\n▶️  Avvio clustering ULTRA OPTIMIZED V1...")
        start_time = datetime.now()
        
        try:
            results = clustering_service.run_clustering_test(
                tenant_id=tenant_id,
                custom_parameters=ultra_config,
                sample_size=None  # Usa tutto il dataset
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            if results and 'cluster_labels' in results:
                # Calcola metriche
                n_conversations = len(results['cluster_labels'])
                n_outliers = sum(1 for label in results['cluster_labels'] if label == -1)
                n_clusters = len(set(results['cluster_labels'])) - (1 if -1 in results['cluster_labels'] else 0)
                outlier_ratio = n_outliers / n_conversations if n_conversations > 0 else 1
                silhouette_score = results.get('silhouette_score', 0)
                clustering_ratio = (n_conversations - n_outliers) / n_conversations if n_conversations > 0 else 0
                quality_score = silhouette_score * (1 - outlier_ratio)
                
                print(f"\n✅ RISULTATI ULTRA OPTIMIZED V1:")
                print(f"   🎯 Quality Score: {quality_score:.4f}")
                print(f"   📊 Silhouette Score: {silhouette_score:.4f}")
                print(f"   🔍 Outliers: {n_outliers}/{n_conversations} ({outlier_ratio:.1%})")
                print(f"   🧩 Cluster: {n_clusters}")
                print(f"   ⏱️  Tempo: {execution_time:.1f}s")
                
                # Confronta con target
                if outlier_ratio <= ultra_expected['outlier_ratio_range'][1]:
                    print(f"   🎉 ULTRA SUCCESS: Outliers sotto target!")
                else:
                    print(f"   ⚠️  Outliers sopra target")
                
                if ultra_expected['n_clusters_range'][0] <= n_clusters <= ultra_expected['n_clusters_range'][1]:
                    print(f"   ✅ Clusters nel range atteso")
                else:
                    print(f"   ⚠️  Clusters fuori range atteso")
                
                # Salva nel database
                cursor.execute("""
                    INSERT INTO clustering_test_results 
                    (tenant_id, version_number, n_clusters, n_outliers, n_conversations, 
                     silhouette_score, clustering_ratio, parameters_json, execution_time, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """, (tenant_id, next_version, n_clusters, n_outliers, n_conversations,
                      silhouette_score, clustering_ratio, json.dumps(ultra_config), execution_time))
                
                results_db.connection.commit()
                print(f"   💾 Salvato come Version {next_version}")
                
            else:
                print(f"   ❌ Clustering fallito per ULTRA OPTIMIZED V1")
                
        except Exception as e:
            print(f"   ❌ Errore durante clustering ULTRA: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n🏆 RIEPILOGO TEST COMPLETATO")
        print("=" * 80)
        print("✅ Entrambe le configurazioni sono state testate")
        print("💾 Risultati salvati nel database")
        print("📊 Puoi ora confrontare le performance con lo storico")
        
    except Exception as e:
        print(f"❌ Errore generale: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        results_db.disconnect()

if __name__ == "__main__":
    test_superior_configurations()
