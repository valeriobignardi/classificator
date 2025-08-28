#!/usr/bin/env python3
"""
Debug test per verificare perché le metriche Davies-Bouldin e Calinski-Harabasz
restituiscono sempre 0.00 nel popup dei risultati
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Clustering.clustering_test_service import ClusteringTestService

def test_metrics_debug():
    """
    Test diretto per debuggare il calcolo delle metriche
    """
    print("🔍 DEBUG: Test diretto calcolo metriche clustering")
    print("=" * 60)
    
    # Tenant Humanitas
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    
    # Parametri semplici per test veloce
    test_parameters = {
        'min_cluster_size': 5,
        'min_samples': 3,
        'cluster_selection_epsilon': 0.0,
        'metric': 'euclidean'
    }
    
    print(f"🚀 Avvio test clustering per tenant: {tenant_id}")
    print(f"📋 Parametri: {test_parameters}")
    print()
    
    # Inizializza servizio
    service = ClusteringTestService()
    
    # Esegui test con sample limitato per debugging
    result = service.run_clustering_test(
        tenant_id=tenant_id,
        custom_parameters=test_parameters,
        sample_size=20  # Solo 20 conversazioni per debug veloce
    )
    
    print("\n" + "=" * 60)
    print("📊 RISULTATI TEST:")
    
    if result.get('success'):
        quality_metrics = result.get('quality_metrics', {})
        statistics = result.get('statistics', {})
        
        print(f"✅ Test completato con successo!")
        print(f"📈 Statistiche:")
        print(f"   - Conversazioni totali: {statistics.get('total_conversations', 'N/A')}")
        print(f"   - Cluster: {statistics.get('n_clusters', 'N/A')}")
        print(f"   - Outliers: {statistics.get('n_outliers', 'N/A')}")
        print(f"   - Ratio clustering: {statistics.get('clustering_ratio', 'N/A')}")
        
        print(f"\n🎯 Metriche di Qualità:")
        print(f"   - Silhouette Score: {quality_metrics.get('silhouette_score', 'N/A')}")
        print(f"   - Davies-Bouldin: {quality_metrics.get('davies_bouldin_score', 'N/A')}")
        print(f"   - Calinski-Harabasz: {quality_metrics.get('calinski_harabasz_score', 'N/A')}")
        
        # Controlla se ci sono note di debug
        if 'note' in quality_metrics:
            print(f"   - Note: {quality_metrics['note']}")
            
    else:
        print(f"❌ Test fallito: {result.get('error', 'Errore sconosciuto')}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_metrics_debug()
