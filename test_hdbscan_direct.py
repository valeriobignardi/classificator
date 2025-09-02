#!/usr/bin/env python3
"""
Author: Valerio Bignardi
Date: 2025-09-02
Description: Test semplificato per verificare SOLO il comportamento di max_cluster_size

Test diretto per verificare se max_cluster_size=0 vs None causa errori HDBSCAN

Ultimo aggiornamento: 2025-09-02
"""

import numpy as np
import hdbscan
from sklearn.datasets import make_blobs

def test_hdbscan_direct_max_cluster_size():
    """
    Scopo: Test diretto HDBSCAN con parametri reali del nostro sistema
    
    Replica esattamente i parametri del pipeline per vedere se
    max_cluster_size=None vs 0 causa l'errore specifico
    
    Ultimo aggiornamento: 2025-09-02
    """
    print("🧪 Test DIRETTO HDBSCAN: max_cluster_size=None vs 0")
    print("=" * 60)
    
    # Genera dati simili a quelli reali
    X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.60, random_state=42)
    print(f"📊 Dati generati: {X.shape[0]} campioni, {X.shape[1]} features")
    
    # Parametri reali dal nostro sistema
    base_params = {
        'min_cluster_size': 13,  # Dal log: min_cluster_size: 13
        'min_samples': 16,       # Dal log: min_samples: 16 
        'alpha': 0.4,           # Dal log: alpha: 0.4
        'cluster_selection_method': 'eom',
        'cluster_selection_epsilon': 0.28,
        'metric': 'euclidean',
        'allow_single_cluster': False,
        'prediction_data': True,
        'match_reference_implementation': True
    }
    
    test_cases = [
        ("❌ max_cluster_size=None (PROBLEMATICO)", None),
        ("✅ max_cluster_size=0 (DOVREBBE FUNZIONARE)", 0)
    ]
    
    for test_name, max_cluster_value in test_cases:
        print(f"\n🔍 Test: {test_name}")
        print("-" * 50)
        
        try:
            params = base_params.copy()
            params['max_cluster_size'] = max_cluster_value
            
            print(f"   📋 Parametri HDBSCAN:")
            for key, value in params.items():
                print(f"      {key}: {value}")
            
            # Test diretto HDBSCAN (questo dovrebbe replicare l'errore)
            clusterer = hdbscan.HDBSCAN(**params)
            labels = clusterer.fit_predict(X)
            
            # Analizza risultati
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            print(f"   ✅ SUCCESSO!")
            print(f"   📊 Cluster trovati: {n_clusters}")
            print(f"   🔇 Punti noise: {n_noise}")
            
            if n_clusters > 0:
                for cluster_id in set(labels):
                    if cluster_id != -1:
                        size = list(labels).count(cluster_id)
                        print(f"   📏 Cluster {cluster_id}: {size} punti")
                        
        except Exception as e:
            print(f"   ❌ ERRORE: {type(e).__name__}: {e}")
            if "'<=' not supported between instances of 'NoneType' and 'int'" in str(e):
                print("   🎯 QUESTO È L'ERRORE CHE VEDIAMO NEL PIPELINE!")
                print("   🔧 La fix dovrebbe cambiare None -> 0")

if __name__ == "__main__":
    test_hdbscan_direct_max_cluster_size()
