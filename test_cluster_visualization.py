"""
Script di test per il sistema di visualizzazione cluster
"""

import numpy as np
import sys
import os

# Aggiungi il path dinamico per importare i moduli
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_cluster_visualization():
    """Test del sistema di visualizzazione cluster"""
    
    try:
        from Utils.cluster_visualization import ClusterVisualizationManager
        
        print("🧪 TEST SISTEMA VISUALIZZAZIONE CLUSTER")
        print("=" * 60)
        
        # Crea dati di test simulati
        np.random.seed(42)
        n_samples = 100
        n_clusters = 4
        
        # 1. Genera embedding simulati (2D per semplicità)
        embeddings = np.random.randn(n_samples, 10)
        
        # 2. Genera cluster labels simulati  
        cluster_labels = np.random.choice([-1, 0, 1, 2, 3], size=n_samples, 
                                        p=[0.1, 0.2, 0.3, 0.25, 0.15])
        
        # 3. Crea cluster_info simulato
        cluster_info = {}
        for cluster_id in [0, 1, 2, 3]:
            indices = np.where(cluster_labels == cluster_id)[0]
            cluster_info[cluster_id] = {
                'intent': f'test_cluster_{cluster_id}',
                'size': len(indices),
                'indices': indices.tolist(),
                'intent_string': f'Cluster Test {cluster_id}',
                'classification_method': 'test_hdbscan',
                'average_confidence': np.random.uniform(0.6, 0.9)
            }
        
        # 4. Genera testi simulati
        session_texts = [f"Testo di esempio per sessione {i}" for i in range(n_samples)]
        
        # 5. Test visualizzazione PARAMETRI CLUSTERING
        print("\n🔬 Test 1: Visualizzazione PARAMETRI CLUSTERING")
        print("-" * 50)
        
        visualizer = ClusterVisualizationManager(output_dir="./test_visualizations")
        
        results = visualizer.visualize_clustering_parameters(
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            cluster_info=cluster_info,
            session_texts=session_texts,
            save_html=True,
            show_console=True
        )
        
        print(f"\n✅ Test 1 completato!")
        print(f"   📊 Metriche qualità: {len(results['quality_metrics'])} parametri")
        print(f"   📈 Statistiche cluster: {len(results['cluster_statistics'])} categorie")
        print(f"   📄 File generati: {len(results['generated_files'])}")
        
        # 6. Test visualizzazione STATISTICHE COMPLETE
        print("\n🔬 Test 2: Visualizzazione STATISTICHE COMPLETE")
        print("-" * 50)
        
        # Genera predizioni finali simulate
        final_predictions = []
        for i in range(n_samples):
            final_predictions.append({
                'predicted_label': np.random.choice(['prenotazione', 'info', 'reclamo', 'altro']),
                'confidence': np.random.uniform(0.5, 1.0),
                'method': np.random.choice(['LLM', 'ML', 'ENSEMBLE', 'FALLBACK'])
            })
        
        results2 = visualizer.visualize_classification_statistics(
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            final_predictions=final_predictions,
            cluster_info=cluster_info,
            session_texts=session_texts,
            save_html=True,
            show_console=True
        )
        
        print(f"\n✅ Test 2 completato!")
        print(f"   📊 Metriche qualità: {len(results2['quality_metrics'])} parametri")
        print(f"   📈 Statistiche classificazione: {len(results2['classification_statistics'])} categorie")
        print(f"   📄 File generati: {len(results2['generated_files'])}")
        
        # 7. Riepilogo
        print(f"\n🎉 TUTTI I TEST COMPLETATI CON SUCCESSO!")
        print(f"   🎯 Test eseguiti: 2/2")
        print(f"   📁 File HTML generati: {len(results['generated_files']) + len(results2['generated_files'])}")
        print(f"   📂 Directory output: ./test_visualizations")
        
        # Lista file generati
        print(f"\n📄 FILE GENERATI:")
        all_files = results['generated_files'] + results2['generated_files']
        for i, file_path in enumerate(all_files, 1):
            file_name = os.path.basename(file_path)
            print(f"   {i:2d}. {file_name}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Errore importazione: {e}")
        print("💡 Assicurarsi che plotly sia installato: pip install plotly")
        return False
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cluster_visualization()
    if success:
        print("\n✅ Test completato con successo!")
        exit(0)
    else:
        print("\n❌ Test fallito!")
        exit(1)
