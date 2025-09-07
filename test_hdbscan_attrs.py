#!/usr/bin/env python3
"""
Test rapido per verificare attributi HDBSCANClusterer
"""
import sys
import os
sys.path.append('/home/ubuntu/classificatore')

try:
    from Clustering.hdbscan_clusterer import HDBSCANClusterer
    
    print("✅ Import HDBSCANClusterer riuscito")
    
    # Test inizializzazione
    clusterer = HDBSCANClusterer(
        min_cluster_size=5,
        min_samples=3,
        use_umap=False
    )
    
    print("✅ Inizializzazione HDBSCANClusterer riuscita")
    
    # Verifica attributi
    print(f"🔍 Attributo use_umap: {hasattr(clusterer, 'use_umap')} = {getattr(clusterer, 'use_umap', 'N/A')}")
    print(f"🔍 Attributo umap_enabled: {hasattr(clusterer, 'umap_enabled')}")
    
    # Lista attributi UMAP-related
    umap_attrs = [attr for attr in dir(clusterer) if 'umap' in attr.lower()]
    print(f"🔍 Attributi UMAP disponibili: {umap_attrs}")
    
    print("✅ Test completato con successo")
    
except Exception as e:
    print(f"❌ Errore durante test: {e}")
    import traceback
    traceback.print_exc()
