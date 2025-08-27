#!/usr/bin/env python3
"""
Test completo per verificare il flusso dei parametri di clustering
dalla configurazione tenant fino all'algoritmo HDBSCAN
"""

import sys
import os
import yaml
import numpy as np
from datetime import datetime

# Aggiunge i percorsi necessari
sys.path.append('/home/ubuntu/classificatore/Utils')
sys.path.append('/home/ubuntu/classificatore/Clustering')
sys.path.append('/home/ubuntu/classificatore/Preprocessing')

def test_parameter_flow():
    """
    Test completo del flusso dei parametri
    1. API endpoint lettura
    2. Helper configurazione tenant
    3. HDBSCANClusterer inizializzazione
    4. Algoritmo HDBSCAN reale
    """
    print("🧪 [TEST] Test completo flusso parametri clustering")
    print("=" * 60)
    
    tenant_id = "humanitas"
    
    # Test 1: Verifica endpoint API
    print(f"\n🔍 [TEST 1] Verifica endpoint API per tenant {tenant_id}")
    try:
        import requests
        response = requests.get(f"http://localhost:5000/api/clustering/{tenant_id}/parameters")
        if response.status_code == 200:
            data = response.json()
            params = data.get('parameters', {})
            
            print(f"   ✅ API Response OK")
            print(f"   📋 only_user: {params.get('only_user', {}).get('value', 'N/A')}")
            print(f"   📋 cluster_selection_method: {params.get('cluster_selection_method', {}).get('value', 'N/A')}")
            print(f"   📋 alpha: {params.get('alpha', {}).get('value', 'N/A')}")
            print(f"   📋 cluster_selection_epsilon: {params.get('cluster_selection_epsilon', {}).get('value', 'N/A')}")
            print(f"   📋 min_cluster_size: {params.get('min_cluster_size', {}).get('value', 'N/A')}")
        else:
            print(f"   ❌ API Error: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️ API Test fallito: {e}")
    
    # Test 2: Helper configurazione tenant
    print(f"\n🔍 [TEST 2] Helper configurazione tenant")
    try:
        from tenant_config_helper import get_tenant_config_helper
        
        helper = get_tenant_config_helper()
        
        only_user = helper.get_only_user_setting(tenant_id)
        method = helper.get_clustering_parameter(tenant_id, 'cluster_selection_method', 'eom')
        alpha = helper.get_clustering_parameter(tenant_id, 'alpha', 1.0)
        epsilon = helper.get_clustering_parameter(tenant_id, 'cluster_selection_epsilon', 0.08)
        min_size = helper.get_clustering_parameter(tenant_id, 'min_cluster_size', 5)
        
        print(f"   ✅ Helper inizializzato")
        print(f"   📋 only_user: {only_user}")
        print(f"   📋 cluster_selection_method: {method}")
        print(f"   📋 alpha: {alpha}")
        print(f"   📋 cluster_selection_epsilon: {epsilon}")
        print(f"   📋 min_cluster_size: {min_size}")
        
    except Exception as e:
        print(f"   ❌ Helper test fallito: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: HDBSCANClusterer inizializzazione
    print(f"\n🔍 [TEST 3] HDBSCANClusterer con parametri tenant")
    try:
        from hdbscan_clusterer import HDBSCANClusterer
        
        # Simula il caricamento della configurazione come fa il pipeline
        config_path = '/home/ubuntu/classificatore/config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        clustering_config = config.get('clustering', {})
        
        # Sovrascrivi con parametri tenant (simula il comportamento del pipeline)
        helper = get_tenant_config_helper() 
        tenant_params = helper._load_tenant_config(tenant_id)
        
        print(f"   📋 Parametri tenant caricati: {list(tenant_params.keys())}")
        
        # Merge parametri come farebbe il sistema reale
        final_config = clustering_config.copy()
        final_config.update(tenant_params)
        
        print(f"   🔄 Parametri finali per HDBSCAN:")
        for key in ['min_cluster_size', 'cluster_selection_epsilon', 'cluster_selection_method', 'alpha']:
            value = final_config.get(key, 'N/A')
            print(f"      {key}: {value}")
        
        # Inizializza clusterer con config finale
        clusterer = HDBSCANClusterer(
            min_cluster_size=final_config.get('min_cluster_size'),
            cluster_selection_epsilon=final_config.get('cluster_selection_epsilon'),
            cluster_selection_method=final_config.get('cluster_selection_method'),
            alpha=final_config.get('alpha'),
            config_path=config_path
        )
        
        print(f"   ✅ HDBSCANClusterer inizializzato")
        print(f"   📋 clusterer.min_cluster_size: {clusterer.min_cluster_size}")
        print(f"   📋 clusterer.cluster_selection_epsilon: {clusterer.cluster_selection_epsilon}")
        print(f"   📋 clusterer.cluster_selection_method: {clusterer.cluster_selection_method}")
        print(f"   📋 clusterer.alpha: {clusterer.alpha}")
        
    except Exception as e:
        print(f"   ❌ Clusterer test fallito: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Test clustering reale con embeddings fake
    print(f"\n🔍 [TEST 4] Test algoritmo HDBSCAN con parametri reali")
    try:
        # Crea embeddings finti per test
        np.random.seed(42)
        fake_embeddings = np.random.rand(50, 384).astype(np.float32)  # 50 documenti, 384 dimensioni
        
        print(f"   📊 Embeddings di test: {fake_embeddings.shape}")
        print(f"   🔄 Avvio clustering...")
        
        # Esegui clustering con debug 
        labels = clusterer.fit_predict(fake_embeddings)
        
        # Analisi risultati
        unique_labels = np.unique(labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        n_outliers = len([l for l in labels if l == -1])
        
        print(f"   ✅ Clustering completato!")
        print(f"   📊 Risultati:")
        print(f"      Clusters trovati: {n_clusters}")
        print(f"      Outliers: {n_outliers}")
        print(f"      Labels unique: {unique_labels}")
        
        # Verifica che i parametri siano stati effettivamente usati
        if hasattr(clusterer, 'clusterer') and clusterer.clusterer:
            actual_params = {
                'min_cluster_size': clusterer.clusterer.min_cluster_size,
                'cluster_selection_epsilon': clusterer.clusterer.cluster_selection_epsilon,
                'cluster_selection_method': clusterer.clusterer.cluster_selection_method,
                'alpha': clusterer.clusterer.alpha
            }
            print(f"   🎯 Parametri effettivamente usati da HDBSCAN:")
            for key, value in actual_params.items():
                print(f"      {key}: {value}")
                
        else:
            print(f"   ⚠️ Impossibile verificare parametri HDBSCAN interni")
        
    except Exception as e:
        print(f"   ❌ Test algoritmo fallito: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🧪 [TEST] Test parametri completato")
    print("=" * 60)

if __name__ == "__main__":
    test_parameter_flow()
