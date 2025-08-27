"""
Autore: GitHub Copilot  
Data di creazione: 26 Agosto 2025
Ultima modifica: 26 Agosto 2025

Esempi di utilizzo avanzato del sistema di visualizzazione cluster
"""

def esempio_visualizzazione_parametri_clustering():
    """
    Esempio di visualizzazione durante la fase PARAMETRI CLUSTERING
    """
    print("🎯 ESEMPIO 1: Visualizzazione PARAMETRI CLUSTERING")
    print("=" * 60)
    
    from Utils.cluster_visualization import ClusterVisualizationManager
    import numpy as np
    
    # Dati di esempio (vengono dai risultati clustering)
    embeddings = np.random.randn(150, 384)  # Embeddings realistici
    cluster_labels = np.array([0, 0, 1, 1, 2, -1, 0, 2] * 18 + [1, 2])  # Mix cluster + outliers
    
    cluster_info = {
        0: {
            'intent': 'prenotazione_esami',
            'size': 42,
            'indices': [i for i, l in enumerate(cluster_labels) if l == 0],
            'intent_string': 'Prenotazione Esami',
            'classification_method': 'hierarchical_adaptive',
            'average_confidence': 0.85
        },
        1: {
            'intent': 'info_contatti',
            'size': 36,
            'indices': [i for i, l in enumerate(cluster_labels) if l == 1],
            'intent_string': 'Informazioni Contatti',
            'classification_method': 'hdbscan',
            'average_confidence': 0.78
        },
        2: {
            'intent': 'reclami',
            'size': 54,
            'indices': [i for i, l in enumerate(cluster_labels) if l == 2],
            'intent_string': 'Reclami e Problemi',
            'classification_method': 'hdbscan',
            'average_confidence': 0.72
        }
    }
    
    session_texts = [f"Conversazione di esempio {i} per il cluster" for i in range(len(embeddings))]
    
    # Crea visualizzatore
    visualizer = ClusterVisualizationManager(output_dir="./examples_output")
    
    # Esegui visualizzazione PARAMETRI
    results = visualizer.visualize_clustering_parameters(
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        cluster_info=cluster_info,
        session_texts=session_texts,
        save_html=True,
        show_console=True
    )
    
    print("\n✅ Risultati:")
    print(f"   📊 Metriche: {list(results['quality_metrics'].keys())}")
    print(f"   📈 Statistiche: {list(results['cluster_statistics'].keys())}")
    print(f"   📄 File HTML: {len(results['generated_files'])}")
    
    return results

def esempio_visualizzazione_statistiche_complete():
    """
    Esempio di visualizzazione durante la fase STATISTICHE COMPLETE
    """
    print("\n🎯 ESEMPIO 2: Visualizzazione STATISTICHE COMPLETE")
    print("=" * 60)
    
    from Utils.cluster_visualization import ClusterVisualizationManager
    import numpy as np
    
    # Simula risultato post-classificazione
    embeddings = np.random.randn(200, 768)  # BGE-M3 embeddings
    cluster_labels = np.random.choice([-1, 0, 1, 2, 3], size=200, p=[0.15, 0.25, 0.20, 0.25, 0.15])
    
    # Predizioni finali realistiche 
    etichette_possibili = ['prenotazione_esami', 'info_contatti', 'reclami', 'fatturazione', 'altro']
    metodi_possibili = ['LLM', 'ML', 'ENSEMBLE', 'FALLBACK', 'CLUSTER_PROPAGATED']
    
    final_predictions = []
    for i in range(200):
        final_predictions.append({
            'predicted_label': np.random.choice(etichette_possibili, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
            'confidence': np.random.beta(2, 1.5),  # Distribuzione realistica confidenze
            'method': np.random.choice(metodi_possibili, p=[0.40, 0.25, 0.20, 0.10, 0.05]),
            'ensemble_confidence': np.random.beta(2.5, 1.2),
            'llm_prediction': {'predicted_label': np.random.choice(etichette_possibili), 'confidence': np.random.uniform(0.6, 1.0)},
            'ml_prediction': {'predicted_label': np.random.choice(etichette_possibili), 'confidence': np.random.uniform(0.5, 0.9)}
        })
    
    session_texts = [f"Conversazione realistica numero {i} con contenuto vario" for i in range(len(embeddings))]
    
    # Crea visualizzatore
    visualizer = ClusterVisualizationManager(output_dir="./examples_output")
    
    # Esegui visualizzazione STATISTICHE
    results = visualizer.visualize_classification_statistics(
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        final_predictions=final_predictions,
        session_texts=session_texts,
        save_html=True,
        show_console=True
    )
    
    print("\n✅ Risultati:")
    print(f"   📊 Metriche qualità: {len(results['quality_metrics'])}")
    print(f"   📈 Statistiche classificazione: {len(results['classification_statistics'])}")
    print(f"   📄 File HTML generati: {len(results['generated_files'])}")
    
    return results

def esempio_analisi_comparativa():
    """
    Esempio di analisi comparativa tra due configurazioni
    """
    print("\n🔬 ESEMPIO 3: Analisi Comparativa Configurazioni")
    print("=" * 60)
    
    from Utils.cluster_visualization import ClusterVisualizationManager
    import numpy as np
    
    # Simula due configurazioni diverse
    configurations = [
        {"name": "BGE-M3_HDBSCAN", "embedding_dim": 1024, "min_cluster_size": 5},
        {"name": "OpenAI_Hierarchical", "embedding_dim": 1536, "min_cluster_size": 3}
    ]
    
    results_comparison = {}
    
    for config in configurations:
        print(f"\n🔧 Testando configurazione: {config['name']}")
        
        # Simula embeddings diversi
        embeddings = np.random.randn(100, config['embedding_dim'])
        
        # Simula clustering con parametri diversi
        if config['min_cluster_size'] == 5:
            cluster_labels = np.random.choice([-1, 0, 1, 2], size=100, p=[0.20, 0.30, 0.30, 0.20])
        else:
            cluster_labels = np.random.choice([-1, 0, 1, 2, 3, 4], size=100, p=[0.10, 0.18, 0.18, 0.18, 0.18, 0.18])
        
        # Predizioni realistiche per configurazione
        final_predictions = []
        for i in range(100):
            confidence_base = 0.9 if 'OpenAI' in config['name'] else 0.75
            final_predictions.append({
                'predicted_label': np.random.choice(['prenotazione', 'info', 'reclamo', 'altro']),
                'confidence': np.random.normal(confidence_base, 0.15),
                'method': 'LLM' if 'OpenAI' in config['name'] else 'ENSEMBLE'
            })
        
        # Visualizza
        visualizer = ClusterVisualizationManager(output_dir=f"./comparison_{config['name']}")
        
        result = visualizer.visualize_classification_statistics(
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            final_predictions=final_predictions,
            save_html=True,
            show_console=False  # Silenzioso per comparazione
        )
        
        # Salva metriche chiave per comparazione
        results_comparison[config['name']] = {
            'silhouette_score': result['quality_metrics']['silhouette_score'],
            'n_clusters': result['quality_metrics']['n_clusters'],
            'outlier_percentage': result['quality_metrics']['outlier_percentage'],
            'avg_confidence': result['classification_statistics']['confidence_stats']['mean'],
            'files_generated': len(result['generated_files'])
        }
    
    # Stampa comparazione
    print(f"\n📊 COMPARAZIONE RISULTATI:")
    print(f"{'Metrica':<20} {'BGE-M3':<15} {'OpenAI':<15} {'Migliore':<10}")
    print("-" * 65)
    
    metrics = ['silhouette_score', 'n_clusters', 'avg_confidence']
    for metric in metrics:
        bgem3_val = results_comparison['BGE-M3_HDBSCAN'][metric]
        openai_val = results_comparison['OpenAI_Hierarchical'][metric]
        
        if metric == 'outlier_percentage':
            better = 'BGE-M3' if bgem3_val < openai_val else 'OpenAI'
        else:
            better = 'BGE-M3' if bgem3_val > openai_val else 'OpenAI'
        
        print(f"{metric:<20} {bgem3_val:<15.3f} {openai_val:<15.3f} {better:<10}")
    
    return results_comparison

def esempio_integrazione_pipeline():
    """
    Esempio di come integrare nella pipeline esistente
    """
    print("\n🔗 ESEMPIO 4: Integrazione Pipeline Esistente")
    print("=" * 60)
    
    # Simula integrazione nella pipeline
    class MockPipeline:
        def __init__(self):
            self.visualizer = None
        
        def setup_visualization(self):
            """Setup del sistema visualizzazione"""
            try:
                from Utils.cluster_visualization import ClusterVisualizationManager
                self.visualizer = ClusterVisualizationManager(output_dir="./pipeline_output")
                print("✅ Sistema visualizzazione inizializzato")
                return True
            except ImportError:
                print("⚠️ plotly non disponibile - visualizzazioni disabilitate")
                return False
        
        def esegui_clustering_con_visualizzazione(self, sessioni):
            """Clustering con visualizzazione automatica"""
            import numpy as np
            
            print("🧩 Eseguendo clustering...")
            
            # Simula clustering (in realtà viene da self.esegui_clustering())
            embeddings = np.random.randn(len(sessioni), 768)
            cluster_labels = np.random.choice([-1, 0, 1, 2], size=len(sessioni))
            cluster_info = {0: {'size': 25, 'intent_string': 'Test Cluster'}}
            
            # Visualizzazione PARAMETRI CLUSTERING
            if self.visualizer:
                print("📊 Generando visualizzazioni clustering...")
                self.visualizer.visualize_clustering_parameters(
                    embeddings=embeddings,
                    cluster_labels=cluster_labels,
                    cluster_info=cluster_info,
                    save_html=True,
                    show_console=False
                )
            
            return embeddings, cluster_labels, cluster_info
        
        def classifica_con_visualizzazione(self, sessioni, embeddings, cluster_labels):
            """Classificazione con visualizzazioni statistiche"""
            import numpy as np
            
            print("🏷️ Eseguendo classificazione...")
            
            # Simula classificazione
            final_predictions = []
            for i in range(len(sessioni)):
                final_predictions.append({
                    'predicted_label': np.random.choice(['prenotazione', 'info', 'altro']),
                    'confidence': np.random.uniform(0.6, 1.0),
                    'method': 'ENSEMBLE'
                })
            
            # Visualizzazione STATISTICHE COMPLETE
            if self.visualizer:
                print("📈 Generando visualizzazioni statistiche finali...")
                self.visualizer.visualize_classification_statistics(
                    embeddings=embeddings,
                    cluster_labels=cluster_labels,
                    final_predictions=final_predictions,
                    save_html=True,
                    show_console=False
                )
            
            return final_predictions
    
    # Test integrazione
    pipeline = MockPipeline()
    
    if pipeline.setup_visualization():
        # Simula sessioni
        sessioni_mock = [{'id': f'session_{i}'} for i in range(50)]
        
        # Esegui pipeline completa
        embeddings, cluster_labels, cluster_info = pipeline.esegui_clustering_con_visualizzazione(sessioni_mock)
        predictions = pipeline.classifica_con_visualizzazione(sessioni_mock, embeddings, cluster_labels)
        
        print(f"✅ Pipeline completata:")
        print(f"   📊 {len(sessioni_mock)} sessioni processate")
        print(f"   🧩 {len(set(cluster_labels))} cluster trovati")
        print(f"   🏷️ {len(predictions)} classificazioni generate")
        print(f"   📄 Grafici salvati in ./pipeline_output/")

if __name__ == "__main__":
    print("🚀 ESEMPI DI UTILIZZO SISTEMA VISUALIZZAZIONE CLUSTER")
    print("=" * 80)
    
    # Esegui tutti gli esempi
    esempio_visualizzazione_parametri_clustering()
    esempio_visualizzazione_statistiche_complete()  
    esempio_analisi_comparativa()
    esempio_integrazione_pipeline()
    
    print(f"\n🎉 TUTTI GLI ESEMPI COMPLETATI!")
    print(f"   📂 Controlla le directory per i file HTML generati")
    print(f"   🌐 Apri i file .html nel browser per visualizzazione interattiva")
