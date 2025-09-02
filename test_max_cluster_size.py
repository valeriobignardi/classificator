#!/usr/bin/env python3
"""
Author: Valerio Bignardi
Date: 2025-09-02
Description: Test per verificare il comportamento di max_cluster_size in HDBSCAN

Test per verificare il comportamento del parametro max_cluster_size con:
1. None (valore problematico)
2. 0 (valore di default - unlimited)
3. Un numero intero positivo

Ultimo aggiornamento: 2025-09-02
"""

import numpy as np
import hdbscan
from sklearn.datasets import make_blobs

def test_max_cluster_size_behavior():
    """
    Scopo: Testare il comportamento di HDBSCAN con diversi valori di max_cluster_size
    
    Test cases:
    1. max_cluster_size=None (dovrebbe causare errore)
    2. max_cluster_size=0 (default, unlimited)
    3. max_cluster_size=50 (limite esplicito)
    
    Ultimo aggiornamento: 2025-09-02
    """
    print("🧪 Test comportamento max_cluster_size in HDBSCAN")
    print("=" * 60)
    
    # Genera dati di test
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.60, random_state=42)
    print(f"📊 Dati generati: {X.shape[0]} campioni, {X.shape[1]} features")
    
    # Test parametri base
    base_params = {
        'min_cluster_size': 10,
        'min_samples': 5,
        'metric': 'euclidean'
    }
    
    test_cases = [
        ("max_cluster_size=None", None),
        ("max_cluster_size=0 (default)", 0), 
        ("max_cluster_size=50", 50)
    ]
    
    for test_name, max_cluster_value in test_cases:
        print(f"\n🔍 Test: {test_name}")
        print("-" * 40)
        
        try:
            # Crea e testa il clusterer
            params = base_params.copy()
            if max_cluster_value is not None:
                params['max_cluster_size'] = max_cluster_value
            else:
                # Test esplicito con None
                params['max_cluster_size'] = None
            
            print(f"   📋 Parametri: {params}")
            
            clusterer = hdbscan.HDBSCAN(**params)
            labels = clusterer.fit_predict(X)
            
            # Analisi risultati
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            print(f"   ✅ SUCCESSO!")
            print(f"   📊 Cluster trovati: {n_clusters}")
            print(f"   🔇 Punti noise: {n_noise}")
            
            if n_clusters > 0:
                cluster_sizes = []
                for cluster_id in set(labels):
                    if cluster_id != -1:  # Ignora noise
                        size = list(labels).count(cluster_id)
                        cluster_sizes.append(size)
                        print(f"   📏 Cluster {cluster_id}: {size} punti")
                
                max_size = max(cluster_sizes) if cluster_sizes else 0
                print(f"   🎯 Dimensione massima cluster: {max_size}")
                
                if max_cluster_value and max_cluster_value > 0 and max_size > max_cluster_value:
                    print(f"   ⚠️  ATTENZIONE: Cluster supera max_cluster_size={max_cluster_value}")
            
        except Exception as e:
            print(f"   ❌ ERRORE: {type(e).__name__}: {e}")
            print(f"   🔍 Questo conferma che {test_name} causa problemi")

if __name__ == "__main__":
    test_max_cluster_size_behavior()
