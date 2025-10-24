#!/usr/bin/env python3
"""
Test dello fix della conversione del parametro alpha per HDBSCAN

Questo script testa direttamente la classe HDBSCANClusterer per verificare
che il parametro alpha venga convertito correttamente da intero a float.

Autore: GitHub Copilot
Data: 24 Settembre 2025
"""

import sys
import os
import numpy as np

# Aggiungi il path del modulo
sys.path.append('/home/ubuntu/classificatore')

from Clustering.hdbscan_clusterer import HDBSCANClusterer

def test_alpha_conversion():
    """
    Testa la conversione automatica del parametro alpha da int a float.
    
    Returns:
        bool: True se il test è superato, False altrimenti
    """
    print("🧪 Test conversione parametro alpha...")
    
    # Crea dati di test semplici
    X = np.random.random((50, 10))  # 50 campioni, 10 features
    
    # Test 1: Alpha come intero (caso che causava l'errore)
    print("\n📋 Test 1: Alpha come intero (1)")
    try:
        clusterer = HDBSCANClusterer(
            min_cluster_size=5,
            min_samples=3,
            metric='cosine',
            alpha=1  # INTERO - dovrebbe essere convertito automaticamente
        )
        
        print(f"✅ Alpha inizializzato: {clusterer.alpha} (tipo: {type(clusterer.alpha)})")
        
        # Verifica che sia un float
        if isinstance(clusterer.alpha, float):
            print("✅ Alpha convertito correttamente in float")
        else:
            print("❌ Alpha NON convertito in float")
            return False
            
        # Test clustering
        labels = clusterer.fit_predict(X)
        print(f"✅ Clustering completato: {len(np.unique(labels))} cluster trovati")
        
    except Exception as e:
        print(f"❌ Errore test 1: {e}")
        return False
    
    # Test 2: Alpha come float (dovrebbe funzionare normalmente)  
    print("\n📋 Test 2: Alpha come float (1.0)")
    try:
        clusterer2 = HDBSCANClusterer(
            min_cluster_size=5,
            min_samples=3,
            metric='cosine',
            alpha=1.0  # FLOAT - dovrebbe funzionare normalmente
        )
        
        print(f"✅ Alpha inizializzato: {clusterer2.alpha} (tipo: {type(clusterer2.alpha)})")
        
        # Test clustering
        labels2 = clusterer2.fit_predict(X)
        print(f"✅ Clustering completato: {len(np.unique(labels2))} cluster trovati")
        
    except Exception as e:
        print(f"❌ Errore test 2: {e}")
        return False
    
    # Test 3: Alpha None (dovrebbe usare default 1.0)
    print("\n📋 Test 3: Alpha None (default)")
    try:
        clusterer3 = HDBSCANClusterer(
            min_cluster_size=5,
            min_samples=3,
            metric='cosine',
            alpha=None  # None - dovrebbe usare default 1.0
        )
        
        print(f"✅ Alpha inizializzato: {clusterer3.alpha} (tipo: {type(clusterer3.alpha)})")
        
        if clusterer3.alpha == 1.0:
            print("✅ Default 1.0 applicato correttamente")
        else:
            print("❌ Default NON applicato correttamente")
            return False
            
        # Test clustering
        labels3 = clusterer3.fit_predict(X)
        print(f"✅ Clustering completato: {len(np.unique(labels3))} cluster trovati")
        
    except Exception as e:
        print(f"❌ Errore test 3: {e}")
        return False
    
    print("\n🎉 Tutti i test superati! La conversione alpha funziona correttamente.")
    return True

if __name__ == "__main__":
    success = test_alpha_conversion()
    sys.exit(0 if success else 1)