#!/usr/bin/env python3
"""
Author: Valerio Bignardi
Date: 2025-09-02
Description: Test temporaneo per verificare se max_cluster_size=0 risolve il problema

Test per verificare se cambiando max_cluster_size da None a 0
il clustering funziona nel nostro pipeline senza errori.

Ultimo aggiornamento: 2025-09-02
"""

import os
import sys
import yaml

# Aggiunge il path per importare i moduli del pipeline
sys.path.append('/home/ubuntu/classificatore')
sys.path.append('/home/ubuntu/classificatore/Pipeline')
sys.path.append('/home/ubuntu/classificatore/Utils')
sys.path.append('/home/ubuntu/classificatore/Preprocessing')

from tenant import Tenant
from session_aggregator import SessionAggregator

def test_clustering_with_zero_max_cluster_size():
    """
    Scopo: Testare se max_cluster_size=0 risolve il problema di clustering
    
    Test:
    1. Carica 20 sessioni (piccolo set per test veloce)
    2. Testa clustering con max_cluster_size=0 invece di None
    3. Verifica se il clustering va a buon fine
    
    Ultimo aggiornamento: 2025-09-02
    """
    print("🧪 Test max_cluster_size=0 vs None nel pipeline")
    print("=" * 60)
    
    try:
        # Carica configurazione
        print("📋 Caricamento configurazione...")
        with open('/home/ubuntu/classificatore/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Risolve tenant
        print("🏥 Risoluzione tenant humanitas...")
        tenant = Tenant.from_slug("humanitas")
        if not tenant:
            print("❌ Impossibile risolvere tenant humanitas")
            return
        print(f"✅ Tenant risolto: {tenant.tenant_name} ({tenant.tenant_id})")
        
        # Carica sessioni di test
        print("📊 Caricamento sessioni di test...")
        aggregator = SessionAggregator(tenant=tenant)
        sessioni_dict = aggregator.estrai_sessioni_aggregate(limit=20)  # Solo 20 per test veloce
        
        if not sessioni_dict:
            print("❌ Nessuna sessione caricata")
            return
        
        print(f"✅ Caricate {len(sessioni_dict)} sessioni per il test")
        
        # Prepara parametri clustering
        clustering_config = config.get('clustering', {})
        
        # Test 1: con None (dovrebbe fallire)
        print("\n🔍 Test 1: max_cluster_size=None")
        print("-" * 40)
        test_clustering_params(sessioni_dict, clustering_config, max_cluster_size=None)
        
        # Test 2: con 0 (dovrebbe funzionare)
        print("\n🔍 Test 2: max_cluster_size=0")
        print("-" * 40)
        test_clustering_params(sessioni_dict, clustering_config, max_cluster_size=0)
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()

def test_clustering_params(sessioni_dict, clustering_config, max_cluster_size):
    """
    Scopo: Testa clustering con parametri specifici
    
    Parametri:
    - sessioni_dict: Dizionario delle sessioni
    - clustering_config: Configurazione clustering
    - max_cluster_size: Valore da testare per max_cluster_size
    
    Ultimo aggiornamento: 2025-09-02
    """
    try:
        from Clustering.hdbscan_clusterer import HDBSCANClusterer
        
        # Prepara parametri
        params = {
            'min_cluster_size': clustering_config.get('min_cluster_size', 5),
            'min_samples': clustering_config.get('min_samples', 3),
            'alpha': 1.0,
            'cluster_selection_method': 'eom',
            'cluster_selection_epsilon': 0.05,
            'metric': 'euclidean',
            'allow_single_cluster': False,
            'max_cluster_size': max_cluster_size,  # Questo è il parametro sotto test
            'use_umap': True,
            'umap_n_neighbors': 15,
            'umap_min_dist': 0.0,
            'umap_n_components': 10
        }
        
        print(f"   📋 Parametri clustering: max_cluster_size={max_cluster_size}")
        print(f"   📋 Altri parametri: min_cluster_size={params['min_cluster_size']}, min_samples={params['min_samples']}")
        
        # Inizializza clusterer
        clusterer = HDBSCANClusterer(**params)
        print("   ✅ HDBSCANClusterer inizializzato")
        
        # Prepara testi per clustering
        testi = [sessioni_dict[session_id].get('testo_completo', '') for session_id in list(sessioni_dict.keys())]
        testi = [testo for testo in testi if testo.strip()]  # Rimuove testi vuoti
        
        if len(testi) < 5:
            print(f"   ⚠️  Troppo pochi testi validi ({len(testi)}), salto il test")
            return
        
        print(f"   📊 Testi da clusterizzare: {len(testi)}")
        
        # Testa il clustering (la parte che causa l'errore)
        print("   🔄 Avvio clustering con cluster_intelligently...")
        
        # Usa il metodo del sistema reale per testare il comportamento effettivo
        from Clustering.intelligent_intent_clusterer import IntelligentIntentClusterer
        
        intelligent_clusterer = IntelligentIntentClusterer(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            alpha=params['alpha'],
            cluster_selection_epsilon=params['cluster_selection_epsilon'],
            metric=params['metric'],
            allow_single_cluster=params['allow_single_cluster'],
            max_cluster_size=max_cluster_size  # Questo è il parametro sotto test
        )
        
        # Genera embeddings prima del clustering (come fa il sistema reale)
        from EmbeddingEngine.simple_embedding_manager import SimpleEmbeddingManager
        embedder = SimpleEmbeddingManager()
        embeddings = embedder.generate_embeddings_bulk(testi)
        
        # Testa il clustering con gli embeddings
        cluster_labels, cluster_info = intelligent_clusterer.cluster_intelligently(testi, embeddings)
        
        # Analizza risultati
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"   ✅ SUCCESSO! Clustering completato")
        print(f"   📊 Cluster trovati: {n_clusters}")
        print(f"   🔇 Punti noise: {n_noise}")
        
        if n_clusters > 0:
            for cluster_id in set(cluster_labels):
                if cluster_id != -1:
                    size = list(cluster_labels).count(cluster_id)
                    print(f"   📏 Cluster {cluster_id}: {size} punti")
        
    except Exception as e:
        print(f"   ❌ ERRORE: {type(e).__name__}: {e}")
        if "not supported between instances of 'NoneType' and 'int'" in str(e):
            print("   🎯 Questo è esattamente l'errore che vediamo nel pipeline!")

if __name__ == "__main__":
    test_clustering_with_zero_max_cluster_size()
