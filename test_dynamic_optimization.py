#!/usr/bin/env python3
"""
Autore: Valerio Bignardi
Data: 2025-08-28
Descrizione: Ottimizzazione DINAMICA clustering per target specifici
             - Minimizzare outliers  
             - 20-45 cluster
             - Silhouette > 0.5
Storia aggiornamenti:
- 2025-08-28: Sistema ottimizzazione dinamica con test iterativi
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

import yaml
import time
import traceback
import numpy as np
from sklearn.metrics import silhouette_score
from Preprocessing.session_aggregator import SessionAggregator
from EmbeddingEngine.labse_embedder import LaBSEEmbedder
from Clustering.hdbscan_clusterer import HDBSCANClusterer

def load_config():
    """Carica configurazione YAML"""
    with open('/home/ubuntu/classificatore/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_configuration(config_name, params, dataset_info):
    """
    Test SINGOLA configurazione clustering con analisi performance
    
    Args:
        config_name: Nome configurazione
        params: Dizionario parametri clustering  
        dataset_info: Info dataset pre-caricato
        
    Returns:
        dict: Risultati test con metriche
        
    Ultima modifica: 2025-08-28
    """
    try:
        print(f"\n🚀 TESTING: {config_name}")
        print("="*80)
        
        # 1. SETUP DATASET
        sessions = dataset_info['sessions']
        embeddings_matrix = dataset_info['embeddings']
        
        print(f"📊 Dataset: {len(sessions)} conversazioni")
        print(f"📐 Embeddings: {embeddings_matrix.shape}")
        
        # 2. CLUSTERING CON PARAMETRI SPECIFICI
        start_time = time.time()
        
        clusterer = HDBSCANClusterer(**params)
        
        cluster_labels = clusterer.fit_predict(embeddings_matrix)
        
        elapsed_time = time.time() - start_time
        
        # 3. ANALISI RISULTATI
        unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        outliers = list(cluster_labels).count(-1)
        outlier_percentage = (outliers / len(cluster_labels)) * 100
        
        # 4. SILHOUETTE SCORE (solo se abbastanza cluster)
        try:
            if unique_clusters >= 2 and outliers < len(sessions) * 0.9:
                # Calcola silhouette su subset per velocità
                sample_size = min(1000, len(sessions))
                sample_indices = np.random.choice(len(sessions), sample_size, replace=False)
                
                sample_embeddings = embeddings_matrix[sample_indices]
                sample_labels = cluster_labels[sample_indices]
                
                # Filtra outliers per silhouette
                non_outlier_mask = sample_labels != -1
                if np.sum(non_outlier_mask) >= 50:  # Minimo campioni per silhouette
                    silhouette = silhouette_score(
                        sample_embeddings[non_outlier_mask], 
                        sample_labels[non_outlier_mask]
                    )
                else:
                    silhouette = -1.0
            else:
                silhouette = -1.0
        except Exception as e:
            print(f"⚠️ Errore silhouette: {e}")
            silhouette = -1.0
        
        # 5. CLUSTER BALANCE
        if unique_clusters > 0:
            cluster_sizes = []
            for cluster_id in set(cluster_labels):
                if cluster_id != -1:  # Escludi outliers
                    cluster_sizes.append(list(cluster_labels).count(cluster_id))
            
            if cluster_sizes:
                cluster_balance = 1.0 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
                cluster_balance = max(0, min(1, cluster_balance))
            else:
                cluster_balance = 0.0
        else:
            cluster_balance = 0.0
        
        # 6. QUALITY SCORE COMBINATO
        outlier_score = 1.0 - (outlier_percentage / 100.0)  # Meno outliers = meglio
        silhouette_normalized = max(0, (silhouette + 1) / 2)  # Da [-1,1] a [0,1]
        cluster_count_score = 1.0 if 20 <= unique_clusters <= 45 else 0.5  # Target range
        
        quality_score = (outlier_score * 0.4 + 
                        silhouette_normalized * 0.4 + 
                        cluster_balance * 0.1 +
                        cluster_count_score * 0.1)
        
        # 7. TARGET COMPLIANCE CHECK
        target_compliance = {
            'outliers_low': outlier_percentage < 10.0,  # < 10% outliers
            'cluster_range': 20 <= unique_clusters <= 45,  # 20-45 cluster  
            'silhouette_high': silhouette > 0.5,  # Silhouette > 0.5
            'all_targets': False
        }
        
        target_compliance['all_targets'] = all([
            target_compliance['outliers_low'],
            target_compliance['cluster_range'], 
            target_compliance['silhouette_high']
        ])
        
        results = {
            'config_name': config_name,
            'success': True,
            'clusters': unique_clusters,
            'outliers': outliers,
            'outlier_percentage': outlier_percentage,
            'silhouette': silhouette,
            'cluster_balance': cluster_balance,
            'quality_score': quality_score,
            'execution_time': elapsed_time,
            'target_compliance': target_compliance,
            'sessions_count': len(sessions)
        }
        
        # 8. STAMPA RISULTATI
        print(f"✅ {config_name} COMPLETATO!")
        print(f"   📊 Dataset: {len(sessions)} conversazioni")
        print(f"   🎯 Clusters: {unique_clusters}")
        print(f"   ❌ Outliers: {outliers} ({outlier_percentage:.1f}%)")
        print(f"   📈 Silhouette: {silhouette:.4f}")
        print(f"   ⚖️ Balance: {cluster_balance:.4f}")
        print(f"   🏆 Quality: {quality_score:.4f}")
        print(f"   ⏱️ Time: {elapsed_time:.1f}s")
        
        # TARGET STATUS
        print(f"   🎯 TARGET STATUS:")
        print(f"      {'✅' if target_compliance['outliers_low'] else '❌'} Outliers < 10%: {outlier_percentage:.1f}%")
        print(f"      {'✅' if target_compliance['cluster_range'] else '❌'} Clusters 20-45: {unique_clusters}")
        print(f"      {'✅' if target_compliance['silhouette_high'] else '❌'} Silhouette > 0.5: {silhouette:.4f}")
        print(f"      {'🏆' if target_compliance['all_targets'] else '🔧'} ALL TARGETS: {'ACHIEVED' if target_compliance['all_targets'] else 'NOT MET'}")
        
        return results
        
    except Exception as e:
        print(f"❌ ERRORE in {config_name}: {e}")
        traceback.print_exc()
        return {
            'config_name': config_name,
            'success': False,
            'error': str(e),
            'sessions_count': len(dataset_info['sessions']) if dataset_info else 0
        }

def load_dataset():
    """
    Carica TUTTO il dataset Humanitas una sola volta per riuso
    
    Returns:
        dict: Dataset info con sessioni ed embeddings
        
    Ultima modifica: 2025-08-28
    """
    print("📊 CARICAMENTO DATASET COMPLETO...")
    print("="*80)
    
    # 1. SETUP TENANT
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    tenant_slug = "humanitas"
    
    # 2. ESTRAI TUTTE LE SESSIONI
    aggregator = SessionAggregator(schema=tenant_slug, tenant_id=tenant_id)
    sessions_dict = aggregator.estrai_sessioni_aggregate(limit=None)  # TUTTE
    sessions = list(sessions_dict.values())
    
    print(f"✅ Dataset estratto: {len(sessions)} conversazioni TOTALI")
    
    # 3. GENERA EMBEDDINGS UNA VOLTA SOLA
    print("🚀 Generazione embeddings LaBSE...")
    embedder = LaBSEEmbedder(device='cuda')
    
    session_texts = [s['testo_completo'] for s in sessions]
    batch_size = 50
    all_embeddings = []
    
    total_batches = (len(session_texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(session_texts), batch_size):
        batch = session_texts[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        print(f"📊 Batch {batch_num}/{total_batches}: {len(batch)} items")
        
        embeddings = embedder.encode(batch)
        all_embeddings.append(embeddings)
        
        if batch_num % 10 == 0:
            print(f"   💾 Completati {batch_num}/{total_batches} batch...")
    
    embeddings_matrix = np.vstack(all_embeddings)
    print(f"✅ Embeddings generati: {embeddings_matrix.shape}")
    
    return {
        'sessions': sessions,
        'embeddings': embeddings_matrix,
        'tenant_id': tenant_id,
        'tenant_slug': tenant_slug
    }

def generate_optimization_configs():
    """
    Genera configurazioni di test per ottimizzazione dinamica
    
    Returns:
        list: Lista configurazioni da testare
        
    Ultima modifica: 2025-08-28
    """
    
    print("🔧 GENERAZIONE CONFIGURAZIONI OTTIMIZZAZIONE...")
    print("="*80)
    
    configs = []
    
    # STRATEGIA 1: CLUSTER COMPATTI - Pochi cluster molto coesi
    configs.append({
        'name': 'COMPACT_CLUSTERS',
        'description': 'Cluster compatti con alta coesione',
        'params': {
            'min_cluster_size': 80,   # Cluster MOLTO grandi per coesione
            'min_samples': 40,        # Molto conservativo
            'metric': 'cosine',       # Migliore per testi
            'cluster_selection_epsilon': 0.0,  # Nessun merge forzato
            'cluster_selection_method': 'eom',  # Excess of mass
            'alpha': 0.9,            # Massima conservatività
            'allow_single_cluster': False,
            # UMAP per riduzione dimensionale ottimale
            'use_umap': True,
            'umap_n_neighbors': 30,   # Neighborhood ampio
            'umap_min_dist': 0.2,     # Distanza minima maggiore
            'umap_n_components': 10,  # Poche dimensioni per compattezza
            'umap_metric': 'cosine'
        }
    })
    
    # STRATEGIA 2: BALANCED_PRECISION - Bilanciamento outliers/cluster
    configs.append({
        'name': 'BALANCED_PRECISION',
        'description': 'Bilanciamento tra precisione e recall',
        'params': {
            'min_cluster_size': 60,   # Cluster medio-grandi
            'min_samples': 25, 
            'metric': 'euclidean',
            'cluster_selection_epsilon': 0.05,
            'cluster_selection_method': 'leaf',
            'alpha': 0.7,
            'allow_single_cluster': False,
            # UMAP bilanciato
            'use_umap': True,
            'umap_n_neighbors': 20,
            'umap_min_dist': 0.1,
            'umap_n_components': 15,
            'umap_metric': 'euclidean'
        }
    })
    
    # STRATEGIA 3: SILHOUETTE_FOCUS - Ottimizzato per silhouette alta
    configs.append({
        'name': 'SILHOUETTE_FOCUS',
        'description': 'Ottimizzato per massima silhouette',
        'params': {
            'min_cluster_size': 100,  # Cluster ancora più grandi
            'min_samples': 50,        # Estremamente conservativo
            'metric': 'cosine',       # Ottimale per semantica
            'cluster_selection_epsilon': 0.0,
            'cluster_selection_method': 'eom',
            'alpha': 0.95,           # Massima conservatività
            'allow_single_cluster': False,
            # UMAP ultra-conservativo
            'use_umap': True,
            'umap_n_neighbors': 40,
            'umap_min_dist': 0.3,     # Distanza molto maggiore
            'umap_n_components': 8,   # Ultra ridotto
            'umap_metric': 'cosine'
        }
    })
    
    # STRATEGIA 4: OUTLIER_MINIMIZER - Minimizza outliers aggressivamente  
    configs.append({
        'name': 'OUTLIER_MINIMIZER',
        'description': 'Minimizza outliers aggressivamente',
        'params': {
            'min_cluster_size': 40,   # Cluster medi
            'min_samples': 15,        # Meno restrittivo ma non troppo
            'metric': 'euclidean',
            'cluster_selection_epsilon': 0.12,  # Merge aggressivo
            'cluster_selection_method': 'leaf',
            'alpha': 0.5,            # Meno conservativo
            'allow_single_cluster': False,
            # UMAP meno riduttivo
            'use_umap': True,
            'umap_n_neighbors': 15,
            'umap_min_dist': 0.05,    # Distanza piccola
            'umap_n_components': 20,  # Più dimensioni
            'umap_metric': 'cosine'
        }
    })
    
    # STRATEGIA 5: TARGET_FOCUSED - Specifico per 20-45 cluster
    configs.append({
        'name': 'TARGET_FOCUSED',
        'description': 'Ottimizzato per 20-45 cluster con bassa outlier rate',
        'params': {
            'min_cluster_size': 70,   # Dimensione per ~30 cluster
            'min_samples': 30,
            'metric': 'cosine',
            'cluster_selection_epsilon': 0.08,
            'cluster_selection_method': 'eom',
            'alpha': 0.8,
            'allow_single_cluster': False,
            # UMAP per target specifico
            'use_umap': True,
            'umap_n_neighbors': 25,
            'umap_min_dist': 0.15,
            'umap_n_components': 12,
            'umap_metric': 'cosine'
        }
    })
    
    print(f"✅ Configurazioni generate: {len(configs)}")
    for i, config in enumerate(configs, 1):
        print(f"   {i}. {config['name']}: {config['description']}")
    
    return configs

def run_dynamic_optimization():
    """
    Esegue ottimizzazione dinamica per trovare configurazione ottimale
    
    Ultima modifica: 2025-08-28
    """
    print("\n🎯 AVVIO OTTIMIZZAZIONE DINAMICA CLUSTERING")
    print("="*80)
    print("OBIETTIVI:")
    print("  🎯 Outliers < 10%")
    print("  📊 Cluster: 20-45")  
    print("  📈 Silhouette > 0.5")
    print("="*80)
    
    # 1. CARICA DATASET UNA VOLTA SOLA
    dataset_info = load_dataset()
    
    # 2. GENERA CONFIGURAZIONI TEST
    configs = generate_optimization_configs()
    
    # 3. TEST ITERATIVO CONFIGURAZIONI
    results = []
    best_result = None
    
    print(f"\n🔄 INIZIANDO TEST SU {len(configs)} CONFIGURAZIONI...")
    print("="*80)
    
    for i, config in enumerate(configs, 1):
        print(f"\n📋 CONFIGURAZIONE {i}/{len(configs)}: {config['name']}")
        print(f"📝 Descrizione: {config['description']}")
        
        result = test_configuration(
            config['name'],
            config['params'],
            dataset_info
        )
        
        results.append(result)
        
        # Aggiorna miglior risultato se tutti i target sono raggiunti
        if result.get('target_compliance', {}).get('all_targets', False):
            if best_result is None or result['quality_score'] > best_result['quality_score']:
                best_result = result
                print(f"🏆 NUOVO BEST RESULT: {config['name']}")
        
        time.sleep(1)  # Pausa tra test
    
    # 4. ANALISI FINALE RISULTATI
    print("\n📊 ANALISI FINALE RISULTATI")
    print("="*80)
    
    # Ordina per quality score
    results.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
    
    print(f"📈 CLASSIFICA PERFORMANCE:")
    for i, result in enumerate(results[:5], 1):  # Top 5
        if result.get('success'):
            compliance = result.get('target_compliance', {})
            print(f"  {i}. {result['config_name']}:")
            print(f"     🏆 Quality: {result['quality_score']:.4f}")
            print(f"     ❌ Outliers: {result['outlier_percentage']:.1f}%")
            print(f"     🎯 Clusters: {result['clusters']}")
            print(f"     📈 Silhouette: {result['silhouette']:.4f}")
            print(f"     🎯 Targets: {sum(compliance.values())-1}/3 ({'ALL' if compliance.get('all_targets') else 'PARTIAL'})")
    
    # 5. BEST CONFIGURATION REPORT
    if best_result:
        print(f"\n🏆 CONFIGURAZIONE OTTIMALE TROVATA!")
        print("="*80)
        print(f"🎯 Nome: {best_result['config_name']}")
        print(f"📊 Risultati:")
        print(f"   ❌ Outliers: {best_result['outlier_percentage']:.1f}% (target: <10%)")
        print(f"   🎯 Clusters: {best_result['clusters']} (target: 20-45)")  
        print(f"   📈 Silhouette: {best_result['silhouette']:.4f} (target: >0.5)")
        print(f"   🏆 Quality Score: {best_result['quality_score']:.4f}")
        print(f"   ⏱️ Execution Time: {best_result['execution_time']:.1f}s")
        print("✅ TUTTI GLI OBIETTIVI RAGGIUNTI!")
    else:
        print(f"\n⚠️ NESSUNA CONFIGURAZIONE HA RAGGIUNTO TUTTI I TARGET")
        print("="*80)
        print("🔧 SUGGERIMENTI PER ITERAZIONE SUCCESSIVA:")
        
        # Trova miglior risultato parziale
        if results and results[0].get('success'):
            partial_best = results[0]
            compliance = partial_best.get('target_compliance', {})
            
            print(f"📊 Miglior risultato parziale: {partial_best['config_name']}")
            print(f"   🏆 Quality: {partial_best['quality_score']:.4f}")
            
            if not compliance.get('outliers_low'):
                print(f"   🔧 Ridurre outliers: {partial_best['outlier_percentage']:.1f}% → <10%")
            if not compliance.get('cluster_range'):
                print(f"   🔧 Adjust clusters: {partial_best['clusters']} → 20-45")
            if not compliance.get('silhouette_high'):
                print(f"   🔧 Migliorare silhouette: {partial_best['silhouette']:.4f} → >0.5")
    
    return {
        'results': results,
        'best_result': best_result,
        'dataset_info': dataset_info
    }

if __name__ == "__main__":
    try:
        optimization_results = run_dynamic_optimization()
        print(f"\n✅ OTTIMIZZAZIONE DINAMICA COMPLETATA!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Ottimizzazione interrotta dall'utente")
    except Exception as e:
        print(f"\n❌ ERRORE DURANTE OTTIMIZZAZIONE: {e}")
        traceback.print_exc()
