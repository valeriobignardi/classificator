#!/usr/bin/env python3
"""
Autore: Valerio Bignardi
Data: 2025-08-28
Descrizione: Test robusto configurazioni clustering superiori progettate
Storia aggiornamenti:
- 2025-08-28: Prima versione con gestione robusta embeddings
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

from Clustering.clustering_test_service import ClusteringTestService
import yaml
import time
import traceback

def load_config():
    """
    Carica la configurazione dal file YAML
    
    Returns:
        dict: Configurazione completa
    """
    with open('/home/ubuntu/classificatore/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_clustering_with_limited_data(tenant_id, config_name, params, expected_performance):
    """
    Testa configurazione clustering con dataset limitato per evitare problemi memoria
    
    Args:
        tenant_id (str): ID del tenant
        config_name (str): Nome configurazione
        params (dict): Parametri clustering
        expected_performance (dict): Performance attese
    
    Returns:
        dict: Risultati del test
    """
    print(f"🧪 TEST {config_name}")
    print("=" * 60)
    print(f"🎯 Performance attese:")
    for key, value in expected_performance.items():
        print(f"   • {key}: {value}")
    print()
    
    try:
        print(f"▶️  Avvio clustering {config_name}...")
        
        # Inizializza servizio con dataset limitato
        service = ClusteringService()
        
        # Modifica temporaneamente i parametri per limitare il dataset
        params_limited = params.copy()
        
        print(f"🚀 Avvio test clustering limitato (500 conv) per tenant {tenant_id}...")
        
        # Esegui clustering con limit ridotto
        start_time = time.time()
        
        # APPROCCIO DIRETTO - usa il metodo interno più semplice
        from Pipeline.pipeline import Pipeline
        
        pipeline = Pipeline(
            tenant_slug=tenant_id,
            confidence_threshold=0.7,
            auto_mode=True,
            auto_retrain=False,
            custom_params=params_limited
        )
        
        # Estrai dataset limitato manualmente
        print("📊 Estrazione dataset limitato (500 conversazioni)...")
        sessions = pipeline.lettore_conversazioni.aggregator.estrai_sessioni_aggregate(
            giorni_indietro=30,
            limit=500  # LIMITE RIDOTTO
        )
        
        print(f"✅ Dataset estratto: {len(sessions)} conversazioni")
        
        if len(sessions) < 50:
            print(f"⚠️ Dataset troppo piccolo: {len(sessions)} conversazioni")
            return {"error": "Dataset insufficiente", "sessions": len(sessions)}
        
        # Genera embeddings con gestione errori
        print("🔍 Generazione embeddings con gestione robusta...")
        
        try:
            # Usa il clusterer della pipeline per evitare conflitti
            clusterer = pipeline.clusterer
            
            # Processa in batch piccoli
            batch_size = 100
            all_embeddings = []
            session_texts = [s['full_conversation'] for s in sessions]
            
            for i in range(0, len(session_texts), batch_size):
                batch = session_texts[i:i+batch_size]
                print(f"📊 Batch {i//batch_size + 1}/{(len(session_texts)-1)//batch_size + 1}: {len(batch)} conversazioni")
                
                embeddings = pipeline.embedder.encode(batch)
                all_embeddings.append(embeddings)
                
                # Piccola pausa per evitare overload GPU
                time.sleep(0.5)
            
            import numpy as np
            embeddings_matrix = np.vstack(all_embeddings)
            
            print(f"✅ Embeddings generati: {embeddings_matrix.shape}")
            
            # Esegui clustering
            print("🚀 Esecuzione clustering...")
            cluster_results = clusterer.fit_predict(embeddings_matrix)
            
            execution_time = time.time() - start_time
            
            # Calcola metriche
            n_clusters = len(set(cluster_results)) - (1 if -1 in cluster_results else 0)
            n_outliers = sum(1 for x in cluster_results if x == -1)
            n_conversations = len(sessions)
            outlier_ratio = n_outliers / n_conversations
            
            # Calcola silhouette score
            try:
                from sklearn.metrics import silhouette_score
                if n_clusters > 1:
                    silhouette = silhouette_score(embeddings_matrix, cluster_results)
                else:
                    silhouette = 0.0
            except:
                silhouette = 0.0
            
            results = {
                "success": True,
                "config_name": config_name,
                "n_clusters": n_clusters,
                "n_outliers": n_outliers,
                "n_conversations": n_conversations,
                "outlier_ratio": outlier_ratio,
                "silhouette_score": silhouette,
                "execution_time": execution_time,
                "parameters": params_limited
            }
            
            print(f"✅ CLUSTERING COMPLETATO!")
            print(f"   📊 Conversazioni: {n_conversations}")
            print(f"   🎯 Clusters: {n_clusters}")
            print(f"   ❌ Outliers: {n_outliers} ({outlier_ratio:.1%})")
            print(f"   📈 Silhouette: {silhouette:.4f}")
            print(f"   ⏱️ Tempo: {execution_time:.1f}s")
            
            return results
            
        except Exception as e:
            print(f"❌ Errore durante clustering: {str(e)}")
            traceback.print_exc()
            return {"error": f"Clustering failed: {str(e)}", "config_name": config_name}
            
    except Exception as e:
        print(f"❌ Errore generale: {str(e)}")
        traceback.print_exc()
        return {"error": f"Test failed: {str(e)}", "config_name": config_name}

def main():
    """
    Funzione principale per testare configurazioni clustering superiori
    """
    print("🚀 TEST CONFIGURAZIONI CLUSTERING SUPERIORI (ROBUSTO)")
    print("=" * 80)
    print("🎯 Obiettivo: PROGETTARE configurazioni superiori all'esistente")
    print("📊 Approach: Dataset limitato per stabilità")
    print("=" * 80)
    
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    
    # Configurazioni progettate basate sui pattern identificati
    superior_configs = {
        "PRECISION MASTER": {
            "params": {
                'use_umap': True,
                'metric': 'euclidean',
                'alpha': 0.45,  # Bilanciato per precision
                'min_cluster_size': 8,  # Piccolo per precision
                'min_samples': 6,  # Ridotto per catturare più pattern
                'cluster_selection_method': 'eom',  # Migliore per densità
                'cluster_selection_epsilon': 0.25,
                'umap_metric': 'cosine',  # Ottimo per testo
                'umap_min_dist': 0.01,  # Molto basso per precision
                'umap_n_neighbors': 22,  # Ottimizzato
                'umap_n_components': 50,  # Equilibrato
                'umap_random_state': 42,
                'max_cluster_size': 0,
                'allow_single_cluster': False,
                'only_user': False
            },
            "expected": {
                "Outliers": "6-12%",
                "Clusters": "60-85", 
                "Silhouette": "0.12-0.18",
                "Quality Score": ">0.10"
            }
        },
        
        "ULTRA BALANCE": {
            "params": {
                'use_umap': True,
                'metric': 'euclidean', 
                'alpha': 0.40,  # Bilanciato
                'min_cluster_size': 6,  # Molto piccolo per copertura
                'min_samples': 5,  # Ridotto
                'cluster_selection_method': 'leaf',  # Più aggressivo
                'cluster_selection_epsilon': 0.30,  # Leggermente più alto
                'umap_metric': 'cosine',
                'umap_min_dist': 0.005,  # Minimo per massima precision
                'umap_n_neighbors': 18,  # Ridotto per pattern locali
                'umap_n_components': 55,  # Aumentato per più info
                'umap_random_state': 42,
                'max_cluster_size': 0,
                'allow_single_cluster': False, 
                'only_user': False
            },
            "expected": {
                "Outliers": "5-9%",
                "Clusters": "75-95",
                "Silhouette": "0.10-0.16", 
                "Quality Score": ">0.09"
            }
        }
    }
    
    results = []
    
    for config_name, config_data in superior_configs.items():
        try:
            result = test_clustering_with_limited_data(
                tenant_id, 
                config_name, 
                config_data["params"],
                config_data["expected"]
            )
            results.append(result)
            
            print()
            time.sleep(2)  # Pausa tra test
            
        except Exception as e:
            print(f"❌ Errore test {config_name}: {str(e)}")
            results.append({"error": str(e), "config_name": config_name})
    
    print("\n🏆 RIEPILOGO TEST SUPERIORI")
    print("=" * 80)
    
    for result in results:
        if "error" in result:
            print(f"❌ {result['config_name']}: ERRORE - {result['error']}")
        else:
            print(f"✅ {result['config_name']}:")
            print(f"   📊 {result['n_conversations']} conv → {result['n_clusters']} clusters")
            print(f"   ❌ Outliers: {result['outlier_ratio']:.1%}")
            print(f"   📈 Silhouette: {result['silhouette_score']:.4f}")
            print(f"   ⏱️ Tempo: {result['execution_time']:.1f}s")
    
    print("\n🎯 Test configurazioni superiori completato!")
    print("💡 Confronta i risultati con V22 (Outliers: 12.9%, Sil: 0.0836)")

if __name__ == "__main__":
    main()
