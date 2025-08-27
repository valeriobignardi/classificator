#!/usr/bin/env python3
"""
Test debug parametri HDBSCAN dal flusso React
"""

import sys
import os
import numpy as np

# Aggiunge i percorsi necessari
sys.path.append('/home/ubuntu/classificatore/Pipeline')
sys.path.append('/home/ubuntu/classificatore/Utils')

def test_hdbscan_debug():
    """
    Test per verificare i debug dei parametri HDBSCAN
    """
    print("🧪 [TEST DEBUG] Test parametri HDBSCAN dall'interfaccia React")
    print("=" * 70)
    
    try:
        from end_to_end_pipeline import EndToEndPipeline
        
        print(f"🔍 [TEST] Inizializzo pipeline con tenant 'humanitas'...")
        print(f"   (I parametri dovrebbero venire dalla config tenant appena aggiornata)")
        
        # Inizializza pipeline che dovrebbe leggere i parametri tenant
        pipeline = EndToEndPipeline(
            tenant_slug="humanitas",
            min_cluster_size=None,  # Usa da config tenant
            confidence_threshold=0.7
        )
        
        print(f"\n🔍 [TEST] Pipeline inizializzata")
        print(f"   Tenant ID: {pipeline.tenant_id}")
        print(f"   Tenant Slug: {pipeline.tenant_slug}")
        
        # Crea embeddings fittizi per test
        print(f"\n🔍 [TEST] Creo embeddings fittizi per test clustering...")
        np.random.seed(42)
        fake_embeddings = np.random.rand(20, 384).astype(np.float32)
        print(f"   Embeddings shape: {fake_embeddings.shape}")
        
        # Esegui clustering - qui dovremmo vedere i debug
        print(f"\n🔍 [TEST] Avvio clustering per vedere debug parametri...")
        print(f"🎯 [ATTESO] Dovremmo vedere i parametri:")
        print(f"   cluster_selection_method: leaf")
        print(f"   alpha: 0.5") 
        print(f"   min_cluster_size: 10")
        print(f"   cluster_selection_epsilon: 0.15")
        print(f"   allow_single_cluster: False")
        print(f"")
        print(f"🔥 [DEBUG INIZIO] ===================================")
        
        # Questo dovrebbe mostrare i debug dell'HDBSCANClusterer
        # Uso esegui_clustering che internamente usa l'HDBSCANClusterer
        # Creo sessioni fittizie in formato corretto
        fake_sessions = {}
        for i in range(20):
            fake_sessions[f"session_{i:03d}"] = {
                'session_id': f"session_{i:03d}",
                'testo_completo': f"Testo di test numero {i} per clustering",
                'messaggi_totali': 1
            }
        
        print(f"   Sessioni create: {len(fake_sessions)}")
        print(f"")
        
        # Questo dovrebbe attivare l'HDBSCANClusterer con i debug
        result = pipeline.esegui_clustering(fake_sessions)
        
        # esegui_clustering restituisce: (embeddings, cluster_labels, representatives, suggested_labels)
        if len(result) == 4:
            embeddings, cluster_labels, representatives, suggested_labels = result
            print(f"   Embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else type(embeddings)}")
            print(f"   Cluster labels type: {type(cluster_labels)}")
            print(f"   Representatives type: {type(representatives)}")
            print(f"   Suggested labels type: {type(suggested_labels)}")
            
            if cluster_labels and len(cluster_labels) > 0:
                print(f"   N sessioni clusterizzate: {len(cluster_labels)}")
                unique_clusters = set(cluster_labels.values()) if isinstance(cluster_labels, dict) else set(cluster_labels)
                print(f"   N clusters unici: {len(unique_clusters)}")
                print(f"   Cluster IDs: {sorted(unique_clusters)}")
            else:
                print(f"   ⚠️ Nessun risultato di clustering")
        else:
            print(f"   ⚠️ Return inaspettato: {len(result)} elementi invece di 4")
            print(f"   Tipi: {[type(r) for r in result]}")
        
        
        print(f"✅ [TEST] Clustering completato!")
        print(f"   Vedi i debug sopra per verificare i parametri HDBSCAN")
        print(f"   che arrivano dalla configurazione tenant via React")
        
        return True
        
    except Exception as e:
        print(f"❌ [TEST] Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🧪 [TEST DEBUG] Completato")
    print("=" * 70)

if __name__ == "__main__":
    test_hdbscan_debug()
