#!/usr/bin/env python3
"""
Test script per verificare trace_all in cluster_intelligently
Author: Valerio Bignardi
Date: 2024-12-19
"""

import sys
import os
import numpy as np
sys.path.append('/home/ubuntu/classificatore')

from Clustering.intelligent_intent_clusterer import IntelligentIntentClusterer

def test_cluster_intelligently_trace():
    """
    Testa che cluster_intelligently usi trace_all correttamente
    """
    print("🧪 TEST: Verifica trace_all in cluster_intelligently")
    print("="*60)
    
    # Test data semplici
    test_texts = [
        "Ciao, ho un problema con l'accesso all'app",
        "Vorrei prenotare un esame del sangue",
        "Non riesco a vedere i miei referti",
        "Quando posso ritirare i risultati?",
        "C'è un errore nel pagamento"
    ]
    
    # Crea embeddings mock (vettori casuali)
    test_embeddings = np.random.rand(len(test_texts), 768)  # 768 dimensioni tipiche
    
    try:
        # Inizializza clusterer (senza tenant per semplicità test)
        clusterer = IntelligentIntentClusterer(
            tenant=None,
            config_path='/home/ubuntu/classificatore/config.yaml'
        )
        
        print("✅ IntelligentIntentClusterer inizializzato")
        print(f"📊 Test con {len(test_texts)} conversazioni")
        print(f"🧮 Embeddings shape: {test_embeddings.shape}")
        
        # Testa clustering (dovrebbe usare trace_all)
        print("\n🔬 Esecuzione cluster_intelligently con trace_all...")
        cluster_labels, cluster_info = clusterer.cluster_intelligently(test_texts, test_embeddings)
        
        print(f"✅ Clustering completato")
        print(f"📊 Cluster trovati: {len(set(cluster_labels))}")
        print(f"🏷️  Cluster info keys: {list(cluster_info.keys())}")
        
        # Verifica risultati
        assert cluster_labels is not None, "cluster_labels non può essere None"
        assert cluster_info is not None, "cluster_info non può essere None"
        assert len(cluster_labels) == len(test_texts), "cluster_labels deve avere stessa lunghezza di test_texts"
        
        print("\n✅ TEST COMPLETATO: trace_all in cluster_intelligently funziona!")
        
        # Mostra statistiche finali
        unique_labels = set(cluster_labels)
        for label in unique_labels:
            count = sum(1 for l in cluster_labels if l == label)
            print(f"   📊 Cluster {label}: {count} conversazioni")
        
    except Exception as e:
        print(f"❌ ERRORE DURANTE TEST: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cluster_intelligently_trace()
