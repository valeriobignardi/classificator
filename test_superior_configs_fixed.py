#!/usr/bin/env python3
"""
Test configurazioni clustering SUPERIORI - versione corretta
Autore: Valerio Bignardi  
Data: 2025-08-28
"""

import json
import sys
import os
from datetime import datetime

# Aggiungi il path del progetto
sys.path.append('/home/ubuntu/classificatore')

from Clustering.clustering_test_service_new import ClusteringTestService
from Database.clustering_results_db import ClusteringResultsDB

def test_and_compare_superior_configs():
    """
    Testa le configurazioni superiori e le confronta con le esistenti
    """
    
    print("🚀 TEST CONFIGURAZIONI CLUSTERING SUPERIORI")
    print("=" * 80)
    print("🎯 Obiettivo: SUPERARE tutte le 22 configurazioni esistenti")
    print("📊 Migliore attuale: Quality Score 0.0854")
    print("=" * 80)
    
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    
    # Carica le configurazioni progettate
    with open("/home/ubuntu/classificatore/superior_configs.json", 'r') as f:
        configs_data = json.load(f)
    
    # Inizializza i servizi
    clustering_service = ClusteringTestService()
    results_db = ClusteringResultsDB()
    
    if not results_db.connect():
        print("❌ Impossibile connettersi al database")
        return
    
    # Ottieni prossimo version number
    cursor = results_db.connection.cursor()
    cursor.execute("SELECT MAX(version_number) FROM clustering_test_results WHERE tenant_id = %s", (tenant_id,))
    max_version = cursor.fetchone()[0] or 0
    next_version = max_version + 1
    
    results_summary = []
    
    # TEST 1: CONFIGURAZIONE SUPERIOR
    print(f"\n🧪 TEST 1 - CONFIGURAZIONE SUPERIOR V1")
    print("=" * 60)
    
    superior_config = configs_data["superior"]["parameters"]
    expected_perf = configs_data["superior"]["expected_performance"]
    
    print(f"🎯 Target: Quality Score >{expected_perf['target_quality_score']}")
    print(f"▶️  Avvio clustering...")
    
    try:
        start_time = datetime.now()
        results = clustering_service.run_clustering_test(
            tenant_id=tenant_id,
            custom_parameters=superior_config,
            sample_size=None
        )
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        if results and results.get('success', False):
            stats = results.get('statistics', {})
            quality_metrics = results.get('quality_metrics', {})
            
            n_conversations = stats.get('total_conversations', 0)
            n_outliers = stats.get('n_outliers', 0)
            n_clusters = stats.get('n_clusters', 0)
            outlier_ratio = n_outliers / n_conversations if n_conversations > 0 else 1
            silhouette_score = quality_metrics.get('silhouette_score', 0)
            quality_score = silhouette_score * (1 - outlier_ratio)
            
            print(f"\n✅ RISULTATI SUPERIOR V1:")
            print(f"   🎯 Quality Score: {quality_score:.4f}")
            print(f"   📊 Silhouette: {silhouette_score:.4f}")
            print(f"   🔍 Outliers: {n_outliers}/{n_conversations} ({outlier_ratio:.1%})")
            print(f"   🧩 Clusters: {n_clusters}")
            print(f"   ⏱️  Tempo: {execution_time:.1f}s")
            
            # Valutazione vs target
            success_indicators = []
            if quality_score > expected_perf['target_quality_score']:
                print(f"   🎉 SUPERIOR SUCCESS: Quality Score SUPERA il target!")
                success_indicators.append("✅ Quality Score")
            else:
                print(f"   ⚠️  Quality Score sotto target")
                success_indicators.append("❌ Quality Score")
            
            if expected_perf['outlier_ratio_range'][0] <= outlier_ratio <= expected_perf['outlier_ratio_range'][1]:
                print(f"   ✅ Outliers nel range atteso")
                success_indicators.append("✅ Outliers")
            else:
                print(f"   ⚠️  Outliers fuori range")
                success_indicators.append("❌ Outliers")
            
            if quality_score > 0.0854:  # Migliore esistente
                print(f"   🏆 RECORD: SUPERA LA MIGLIORE CONFIGURAZIONE ESISTENTE!")
                success_indicators.append("🏆 NEW RECORD")
            
            # Salva nel database
            cursor.execute("""
                INSERT INTO clustering_test_results 
                (tenant_id, version_number, n_clusters, n_outliers, n_conversations, 
                 silhouette_score, clustering_ratio, parameters_json, execution_time, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (tenant_id, next_version, n_clusters, n_outliers, n_conversations,
                  silhouette_score, stats.get('clustering_ratio', 0), 
                  json.dumps(superior_config), execution_time))
            
            results_db.connection.commit()
            print(f"   💾 Salvato come Version {next_version}")
            
            results_summary.append({
                'config': 'SUPERIOR V1',
                'version': next_version,
                'quality_score': quality_score,
                'silhouette': silhouette_score,
                'outlier_ratio': outlier_ratio,
                'n_clusters': n_clusters,
                'execution_time': execution_time,
                'success_indicators': success_indicators,
                'new_record': quality_score > 0.0854
            })
            
            next_version += 1
            
        else:
            print(f"   ❌ Clustering fallito per SUPERIOR V1")
            results_summary.append({
                'config': 'SUPERIOR V1',
                'version': None,
                'quality_score': 0,
                'success': False
            })
            
    except Exception as e:
        print(f"   ❌ Errore SUPERIOR: {e}")
        import traceback
        traceback.print_exc()
    
    # TEST 2: CONFIGURAZIONE ULTRA OPTIMIZED
    print(f"\n🧪 TEST 2 - CONFIGURAZIONE ULTRA OPTIMIZED V1")
    print("=" * 60)
    
    ultra_config = configs_data["ultra"]["parameters"]
    ultra_expected = configs_data["ultra"]["expected_performance"]
    
    print(f"🎯 Target: Outliers <{ultra_expected['outlier_ratio_range'][1]*100:.0f}%")
    print(f"▶️  Avvio clustering ultra-aggressivo...")
    
    try:
        start_time = datetime.now()
        results = clustering_service.run_clustering_test(
            tenant_id=tenant_id,
            custom_parameters=ultra_config,
            sample_size=None
        )
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        if results and results.get('success', False):
            stats = results.get('statistics', {})
            quality_metrics = results.get('quality_metrics', {})
            
            n_conversations = stats.get('total_conversations', 0)
            n_outliers = stats.get('n_outliers', 0)
            n_clusters = stats.get('n_clusters', 0)
            outlier_ratio = n_outliers / n_conversations if n_conversations > 0 else 1
            silhouette_score = quality_metrics.get('silhouette_score', 0)
            quality_score = silhouette_score * (1 - outlier_ratio)
            
            print(f"\n✅ RISULTATI ULTRA OPTIMIZED V1:")
            print(f"   🎯 Quality Score: {quality_score:.4f}")
            print(f"   📊 Silhouette: {silhouette_score:.4f}")
            print(f"   🔍 Outliers: {n_outliers}/{n_conversations} ({outlier_ratio:.1%})")
            print(f"   🧩 Clusters: {n_clusters}")
            print(f"   ⏱️  Tempo: {execution_time:.1f}s")
            
            success_indicators_ultra = []
            if outlier_ratio <= ultra_expected['outlier_ratio_range'][1]:
                print(f"   🎉 ULTRA SUCCESS: Outliers sotto target!")
                success_indicators_ultra.append("✅ Outliers")
            else:
                print(f"   ⚠️  Outliers sopra target")
                success_indicators_ultra.append("❌ Outliers")
            
            if ultra_expected['n_clusters_range'][0] <= n_clusters <= ultra_expected['n_clusters_range'][1]:
                print(f"   ✅ Clusters nel range atteso")
                success_indicators_ultra.append("✅ Clusters")
            else:
                print(f"   ⚠️  Clusters fuori range")
                success_indicators_ultra.append("❌ Clusters")
            
            if quality_score > 0.0854:
                print(f"   🏆 RECORD: ANCHE ULTRA SUPERA IL RECORD!")
                success_indicators_ultra.append("🏆 NEW RECORD")
            
            # Salva nel database
            cursor.execute("""
                INSERT INTO clustering_test_results 
                (tenant_id, version_number, n_clusters, n_outliers, n_conversations, 
                 silhouette_score, clustering_ratio, parameters_json, execution_time, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (tenant_id, next_version, n_clusters, n_outliers, n_conversations,
                  silhouette_score, stats.get('clustering_ratio', 0), 
                  json.dumps(ultra_config), execution_time))
            
            results_db.connection.commit()
            print(f"   💾 Salvato come Version {next_version}")
            
            results_summary.append({
                'config': 'ULTRA OPTIMIZED V1',
                'version': next_version,
                'quality_score': quality_score,
                'silhouette': silhouette_score,
                'outlier_ratio': outlier_ratio,
                'n_clusters': n_clusters,
                'execution_time': execution_time,
                'success_indicators': success_indicators_ultra,
                'new_record': quality_score > 0.0854
            })
            
        else:
            print(f"   ❌ Clustering fallito per ULTRA OPTIMIZED V1")
            results_summary.append({
                'config': 'ULTRA OPTIMIZED V1',
                'version': None,
                'quality_score': 0,
                'success': False
            })
            
    except Exception as e:
        print(f"   ❌ Errore ULTRA: {e}")
        import traceback
        traceback.print_exc()
    
    # RIEPILOGO FINALE
    print(f"\n🏆 RIEPILOGO FINALE - CONFIGURAZIONI SUPERIORI")
    print("=" * 80)
    
    successful_configs = [r for r in results_summary if r.get('success', True) and r['quality_score'] > 0]
    new_records = [r for r in successful_configs if r.get('new_record', False)]
    
    if new_records:
        print(f"🎉 SUCCESSO! {len(new_records)} configurazione/i hanno stabilito NUOVI RECORD!")
        
        best_new = max(new_records, key=lambda x: x['quality_score'])
        print(f"\n🏅 MIGLIORE NUOVA CONFIGURAZIONE:")
        print(f"   • Nome: {best_new['config']}")
        print(f"   • Version: {best_new['version']}")
        print(f"   • Quality Score: {best_new['quality_score']:.4f} (vs 0.0854 precedente)")
        print(f"   • Miglioramento: +{((best_new['quality_score']/0.0854)-1)*100:.1f}%")
        print(f"   • Outliers: {best_new['outlier_ratio']:.1%}")
        print(f"   • Clusters: {best_new['n_clusters']}")
    else:
        print(f"⚠️  Nessuna configurazione ha superato il record esistente")
        print(f"📊 Migliore risultato: {max(successful_configs, key=lambda x: x['quality_score'])['quality_score']:.4f}" if successful_configs else "N/A")
    
    print(f"\n📈 DETTAGLI COMPLETI:")
    for result in results_summary:
        if result.get('success', True) and result['quality_score'] > 0:
            print(f"   • {result['config']}: Quality {result['quality_score']:.4f}, " +
                  f"Outliers {result['outlier_ratio']:.1%}, {' '.join(result['success_indicators'])}")
    
    # Salva riepilogo
    summary_data = {
        'test_date': datetime.now().isoformat(),
        'tenant_id': tenant_id,
        'results_summary': results_summary,
        'new_records_count': len(new_records),
        'best_previous_quality_score': 0.0854,
        'improvement_achieved': len(new_records) > 0
    }
    
    with open("/home/ubuntu/classificatore/superior_test_results.json", 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n💾 Riepilogo salvato in: superior_test_results.json")
    results_db.disconnect()

if __name__ == "__main__":
    test_and_compare_superior_configs()
