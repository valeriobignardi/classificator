#!/usr/bin/env python3
"""
Autore: Valerio Bignardi
Data: 2025-08-28
Descrizione: Test DIRETTO configurazioni clustering superiori progettate
Storia aggiornamenti:
- 2025-08-28: Prima versione con approccio diretto pipeline
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

from Pipeline.end_to_end_pipeline import EndToEndPipeline
import yaml
import time
import traceback
import numpy as np
from sklearn.metrics import silhouette_score

def test_superior_config_direct(tenant_id, config_name, params, expected_performance):
    """
    Test DIRETTO configurazione clustering con pipeline
    
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
        start_time = time.time()
        
        print(f"🚀 Inizializzazione pipeline con parametri personalizzati...")
        
        # Crea pipeline con parametri custom
        pipeline = EndToEndPipeline(
            tenant_slug=tenant_id,
            confidence_threshold=0.7,
            auto_mode=True,
            auto_retrain=False,
            custom_params=params
        )
        
        print("📊 Estrazione dataset limitato (300 conversazioni)...")
        
        # Estrai sessioni limitate per stabilità
        sessions = pipeline.lettore_conversazioni.aggregator.estrai_sessioni_aggregate(
            giorni_indietro=30,
            limit=300  # Dataset ridotto per stabilità
        )
        
        print(f"✅ Dataset estratto: {len(sessions)} conversazioni")
        
        if len(sessions) < 50:
            return {"error": "Dataset insufficiente", "config_name": config_name, "sessions": len(sessions)}
        
        # Prepara testi per embeddings  
        session_texts = [s['full_conversation'] for s in sessions]
        print(f"📝 Preparazione {len(session_texts)} testi per clustering...")
        
        # Genera embeddings con batch processing
        print("🔍 Generazione embeddings con batch processing...")
        batch_size = 50
        all_embeddings = []
        
        for i in range(0, len(session_texts), batch_size):
            batch = session_texts[i:i+batch_size]
            print(f"📊 Processing batch {i//batch_size + 1}/{(len(session_texts)-1)//batch_size + 1}: {len(batch)} items")
            
            try:
                embeddings = pipeline.embedder.encode(batch)
                all_embeddings.append(embeddings)
                time.sleep(0.2)  # Piccola pausa
            except Exception as e:
                print(f"⚠️ Errore batch {i//batch_size + 1}: {str(e)}")
                continue
        
        if not all_embeddings:
            return {"error": "Nessun embedding generato", "config_name": config_name}
        
        # Combina tutti gli embeddings
        embeddings_matrix = np.vstack(all_embeddings)
        print(f"✅ Embeddings matrix: {embeddings_matrix.shape}")
        
        # Esegui clustering con i parametri custom
        print("🚀 Esecuzione clustering con parametri superiori...")
        cluster_results = pipeline.clusterer.fit_predict(embeddings_matrix)
        
        execution_time = time.time() - start_time
        
        # Calcola metriche performance
        unique_clusters = set(cluster_results)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_outliers = sum(1 for x in cluster_results if x == -1)
        n_conversations = len(sessions)
        outlier_ratio = n_outliers / n_conversations
        
        # Calcola silhouette score
        try:
            if n_clusters > 1 and len(set(cluster_results)) > 1:
                silhouette = silhouette_score(embeddings_matrix, cluster_results)
            else:
                silhouette = 0.0
        except Exception as e:
            print(f"⚠️ Errore calcolo silhouette: {str(e)}")
            silhouette = 0.0
        
        # Calcola quality score (inverso outlier ratio + silhouette weight)  
        quality_score = (1 - outlier_ratio) * 0.7 + silhouette * 0.3
        
        results = {
            "success": True,
            "config_name": config_name,
            "n_clusters": n_clusters,
            "n_outliers": n_outliers,
            "n_conversations": n_conversations,
            "outlier_ratio": outlier_ratio,
            "silhouette_score": silhouette,
            "quality_score": quality_score,
            "execution_time": execution_time,
            "parameters": params
        }
        
        print(f"✅ CLUSTERING COMPLETATO - {config_name}!")
        print(f"   📊 Conversazioni: {n_conversations}")
        print(f"   🎯 Clusters: {n_clusters}")
        print(f"   ❌ Outliers: {n_outliers} ({outlier_ratio:.1%})")
        print(f"   📈 Silhouette: {silhouette:.4f}")
        print(f"   🏆 Quality Score: {quality_score:.4f}")
        print(f"   ⏱️ Tempo: {execution_time:.1f}s")
        
        return results
        
    except Exception as e:
        error_msg = f"Test {config_name} fallito: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return {"error": error_msg, "config_name": config_name}

def main():
    """
    Funzione principale per testare configurazioni clustering superiori progettate
    """
    print("🚀 TEST CONFIGURAZIONI CLUSTERING SUPERIORI PROGETTATE")
    print("=" * 80)
    print("🎯 Obiettivo: PROGETTARE E TESTARE configurazioni superiori")
    print("📊 Current Best V22: 12.9% outliers, 0.0836 silhouette")
    print("🎯 Target: <10% outliers, >0.10 silhouette")
    print("=" * 80)
    
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    
    # CONFIGURAZIONI SUPERIORI PROGETTATE basate sui pattern
    superior_configs = {
        "PRECISION_MASTER": {
            "params": {
                'use_umap': True,
                'metric': 'euclidean',
                'alpha': 0.42,  # Ottimizzato per precision
                'min_cluster_size': 7,  # Piccolo per catturare più pattern
                'min_samples': 5,  # Ridotto per sensitivity
                'cluster_selection_method': 'eom',  # Migliore per density-based
                'cluster_selection_epsilon': 0.24,  # Fine-tuned
                'umap_metric': 'cosine',  # Perfetto per similarity testuale
                'umap_min_dist': 0.008,  # Molto basso per precision
                'umap_n_neighbors': 20,  # Ottimizzato
                'umap_n_components': 52,  # Bilanciato info/complessità  
                'umap_random_state': 42,
                'max_cluster_size': 0,
                'allow_single_cluster': False,
                'only_user': False
            },
            "expected": {
                "Outliers": "6-10%",
                "Clusters": "60-80", 
                "Silhouette": "0.11-0.16",
                "Quality Score": ">0.85"
            }
        },
        
        "ULTRA_OPTIMIZER": {
            "params": {
                'use_umap': True,
                'metric': 'euclidean', 
                'alpha': 0.38,  # Aggressivo per ottimizzazione
                'min_cluster_size': 6,  # Molto piccolo
                'min_samples': 4,  # Minimo per catturare tutto
                'cluster_selection_method': 'leaf',  # Più aggressivo
                'cluster_selection_epsilon': 0.26,  
                'umap_metric': 'cosine',
                'umap_min_dist': 0.005,  # Minimo assoluto
                'umap_n_neighbors': 18,  # Ridotto per local patterns
                'umap_n_components': 55,  # Aumentato per più informazione
                'umap_random_state': 42,
                'max_cluster_size': 0,
                'allow_single_cluster': False, 
                'only_user': False
            },
            "expected": {
                "Outliers": "4-8%",
                "Clusters": "70-90",
                "Silhouette": "0.10-0.15", 
                "Quality Score": ">0.87"
            }
        }
    }
    
    results = []
    
    for config_name, config_data in superior_configs.items():
        print(f"\n{'='*60}")
        result = test_superior_config_direct(
            tenant_id, 
            config_name, 
            config_data["params"],
            config_data["expected"]
        )
        results.append(result)
        
        # Pausa tra test per evitare conflitti
        time.sleep(3)
    
    print("\n\n🏆 RIEPILOGO FINALE TEST CONFIGURAZIONI SUPERIORI")
    print("=" * 80)
    print("📊 RIFERIMENTO V22: 12.9% outliers, 0.0836 silhouette, 51 clusters")
    print("=" * 80)
    
    success_count = 0
    for result in results:
        if "error" in result:
            print(f"❌ {result['config_name']}: ERRORE - {result['error']}")
        else:
            success_count += 1
            config_name = result['config_name']
            outliers = result['outlier_ratio']
            silhouette = result['silhouette_score']
            clusters = result['n_clusters']
            quality = result['quality_score']
            
            # Confronto con V22
            outlier_improvement = (0.129 - outliers) / 0.129 * 100
            silhouette_improvement = (silhouette - 0.0836) / 0.0836 * 100
            
            print(f"\n✅ {config_name}:")
            print(f"   📊 {result['n_conversations']} conversazioni → {clusters} clusters")
            print(f"   ❌ Outliers: {outliers:.1%} ({outlier_improvement:+.1f}% vs V22)")
            print(f"   📈 Silhouette: {silhouette:.4f} ({silhouette_improvement:+.1f}% vs V22)")
            print(f"   🏆 Quality Score: {quality:.4f}")
            print(f"   ⏱️ Tempo: {result['execution_time']:.1f}s")
            
            # Valutazione miglioramento
            if outliers < 0.129 and silhouette > 0.0836:
                print(f"   🌟 SUPERIORE a V22 su ENTRAMBE le metriche!")
            elif outliers < 0.129:
                print(f"   ⭐ Migliore su outliers!")
            elif silhouette > 0.0836:
                print(f"   ⭐ Migliore su silhouette!")
            else:
                print(f"   📊 Prestazioni comparabili")
    
    print(f"\n🎯 RISULTATO FINALE: {success_count}/{len(superior_configs)} configurazioni testate con successo")
    if success_count > 0:
        print("💡 Le configurazioni progettate sono pronte per test su dataset completo!")
    print("=" * 80)

if __name__ == "__main__":
    main()
