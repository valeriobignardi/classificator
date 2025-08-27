#!/usr/bin/env python3
"""
Test rapido per verificare il nuovo endpoint di statistiche clustering
"""

import sys
import os

# Aggiungi percorsi
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_clustering_service_visualization():
    """Test del nuovo metodo di visualizzazione"""
    print("🧪 TEST: ClusteringTestService con visualizzazioni")
    print("=" * 60)
    
    try:
        from Clustering.clustering_test_service import ClusteringTestService
        
        service = ClusteringTestService()
        
        # Test con dati simulati
        import numpy as np
        embeddings = np.random.randn(50, 384)
        cluster_labels = np.array([0, 0, 1, 1, 2, -1, 0, 2] * 6 + [1, 2])
        texts = [f"Testo esempio {i}" for i in range(50)]
        session_ids = [f"session_{i}" for i in range(50)]
        
        print("🎨 Test generazione dati visualizzazione...")
        viz_data = service._generate_visualization_data(embeddings, cluster_labels, texts, session_ids)
        
        print(f"✅ Dati generati:")
        print(f"   📊 Punti totali: {viz_data.get('total_points', 0)}")
        print(f"   🔍 Cluster: {viz_data.get('n_clusters', 0)}")
        print(f"   📈 Outliers: {viz_data.get('n_outliers', 0)}")
        print(f"   🎯 Dimensioni originali: {viz_data.get('dimensions', {}).get('original', 0)}")
        
        if viz_data.get('points'):
            sample_point = viz_data['points'][0]
            print(f"   📍 Punto esempio: cluster={sample_point.get('cluster_label')}, "
                  f"t-SNE=({sample_point.get('tsne_x', 0):.3f}, {sample_point.get('tsne_y', 0):.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mongo_reader():
    """Test del MongoClassificationReader"""
    print("\n🧪 TEST: MongoClassificationReader")
    print("=" * 60)
    
    try:
        from mongo_classification_reader import MongoClassificationReader
        
        reader = MongoClassificationReader()
        print("✅ MongoClassificationReader inizializzato")
        
        # Verifica che il metodo esista
        method = getattr(reader, 'get_tenant_classifications_with_clustering', None)
        if method:
            print("✅ Metodo get_tenant_classifications_with_clustering disponibile")
        else:
            print("❌ Metodo get_tenant_classifications_with_clustering NON trovato")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test MongoReader: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TEST NUOVO SISTEMA VISUALIZZAZIONI BACKEND")
    print("=" * 60)
    
    test1_ok = test_clustering_service_visualization()
    test2_ok = test_mongo_reader()
    
    print("\n" + "=" * 60)
    print("📊 RISULTATI FINALI:")
    print(f"   🎨 ClusteringService visualizzazioni: {'✅ OK' if test1_ok else '❌ FAIL'}")
    print(f"   📊 MongoReader nuovo metodo: {'✅ OK' if test2_ok else '❌ FAIL'}")
    
    if test1_ok and test2_ok:
        print("\n🎉 TUTTI I TEST BACKEND COMPLETATI CON SUCCESSO!")
        print("📋 Prossimo passo: Implementazione componenti React")
    else:
        print("\n❌ Alcuni test falliti. Controllare errori sopra.")
