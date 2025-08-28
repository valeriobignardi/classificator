#!/usr/bin/env python3
"""
Analisi configurazioni clustering per ottimizzazione parametri
Autore: Valerio Bignardi
Data: 2025-08-28
Obiettivo: Trovare la configurazione ottimale per minimizzare outliers e massimizzare silhouette
"""

import yaml
import json
from Database.clustering_results_db import ClusteringResultsDB
import pandas as pd
import numpy as np

def analyze_clustering_configurations():
    """
    Analizza tutte le configurazioni clustering per il tenant Humanitas
    e propone la configurazione ottimale
    """
    
    # Tenant Humanitas
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    
    # Inizializza database
    results_db = ClusteringResultsDB()
    
    if not results_db.connect():
        print("❌ Impossibile connettersi al database")
        return
    
    try:
        # Query per ottenere tutti i risultati con parametri disponibili
        query = """
        SELECT 
            version_number,
            created_at,
            n_clusters,
            n_outliers, 
            n_conversations,
            silhouette_score,
            clustering_ratio,
            execution_time,
            parameters_json
        FROM clustering_test_results 
        WHERE tenant_id = %s 
        ORDER BY version_number
        """
        
        cursor = results_db.connection.cursor()
        cursor.execute(query, (tenant_id,))
        results = cursor.fetchall()
        
        if not results:
            print(f"❌ Nessun risultato trovato per tenant {tenant_id}")
            return
        
        print(f"📊 Analizzando {len(results)} configurazioni clustering per Humanitas...")
        print("=" * 80)
        
        # Analizza ogni configurazione
        best_configs = []
        
        for result in results:
            version = result[0]
            created_at = result[1]
            n_clusters = result[2]
            n_outliers = result[3]
            n_conversations = result[4]
            silhouette_score = result[5]
            clustering_ratio = result[6]
            execution_time = result[7]
            parameters_json = result[8]
            
            # Parse parametri JSON
            try:
                params = json.loads(parameters_json) if parameters_json else {}
            except:
                params = {}
            
            # Calcola metriche di qualità
            outlier_ratio = n_outliers / n_conversations if n_conversations > 0 else 1
            quality_score = silhouette_score * (1 - outlier_ratio)  # Combina silhouette e outliers
            
            config_analysis = {
                'version': version,
                'created_at': created_at,
                'n_clusters': n_clusters,
                'n_outliers': n_outliers,
                'n_conversations': n_conversations,
                'silhouette_score': silhouette_score,
                'clustering_ratio': clustering_ratio,
                'outlier_ratio': outlier_ratio,
                'quality_score': quality_score,
                'execution_time': execution_time,
                'params': params
            }
            
            best_configs.append(config_analysis)
            
            print(f"🔍 Versione {version} ({created_at}):")
            print(f"   📈 Cluster: {n_clusters}, Outliers: {n_outliers} ({outlier_ratio:.1%})")
            print(f"   🎯 Silhouette: {silhouette_score:.4f}, Quality: {quality_score:.4f}")
            print(f"   ⚙️  Parametri: {json.dumps(params, indent=4) if params else 'Default'}")
            print("-" * 40)
        
        # Trova le migliori configurazioni
        print("\n🏆 TOP CONFIGURAZIONI:")
        print("=" * 80)
        
        # Ordina per quality score (silhouette * (1-outlier_ratio))
        best_configs.sort(key=lambda x: x['quality_score'], reverse=True)
        
        print("📊 TOP 5 per Quality Score (Silhouette × (1-Outlier_Ratio)):")
        for i, config in enumerate(best_configs[:5]):
            print(f"{i+1}. Versione {config['version']} - Quality: {config['quality_score']:.4f}")
            print(f"   📈 Silhouette: {config['silhouette_score']:.4f}")
            print(f"   📉 Outliers: {config['n_outliers']} ({config['outlier_ratio']:.1%})")
            print(f"   🎯 Cluster: {config['n_clusters']}")
        
        print("\n" + "=" * 80)
        
        # Filtra configurazioni con alta silhouette (>0.15) e bassa outlier ratio (<0.5)
        good_configs = [c for c in best_configs if c['silhouette_score'] > 0.15 and c['outlier_ratio'] < 0.5]
        
        if good_configs:
            print(f"🎯 CONFIGURAZIONI OTTIME (Silhouette > 0.15, Outliers < 50%):")
            
            best_config = good_configs[0]
            print(f"\n🥇 CONFIGURAZIONE RACCOMANDATA - Versione {best_config['version']}:")
            print(f"   📊 Risultati:")
            print(f"      • Cluster: {best_config['n_clusters']}")
            print(f"      • Outliers: {best_config['n_outliers']} ({best_config['outlier_ratio']:.1%})")
            print(f"      • Silhouette Score: {best_config['silhouette_score']:.4f}")
            print(f"      • Quality Score: {best_config['quality_score']:.4f}")
            print(f"      • Clustering Ratio: {best_config['clustering_ratio']:.3f}")
            
            print(f"\n   ⚙️  PARAMETRI OTTIMALI:")
            params = best_config['params']
            if params:
                for key, value in params.items():
                    print(f"      • {key}: {value}")
            else:
                print("      • Configurazione DEFAULT")
        
        else:
            # Trova la migliore disponibile
            best_config = best_configs[0]
            print(f"⚠️  Nessuna configurazione ideale trovata. Migliore disponibile:")
            print(f"🥇 VERSIONE {best_config['version']}:")
            print(f"   📊 Silhouette: {best_config['silhouette_score']:.4f}")
            print(f"   📉 Outliers: {best_config['n_outliers']} ({best_config['outlier_ratio']:.1%})")
            print(f"   🎯 Cluster: {best_config['n_clusters']}")
            
            params = best_config['params']
            if params:
                print(f"   ⚙️  Parametri:")
                for key, value in params.items():
                    print(f"      • {key}: {value}")
        
        print("\n📈 ANALISI TREND:")
        print("=" * 40)
        
        # Analizza correlazioni
        silhouette_scores = [c['silhouette_score'] for c in best_configs]
        outlier_ratios = [c['outlier_ratio'] for c in best_configs]
        
        avg_silhouette = np.mean(silhouette_scores)
        avg_outlier_ratio = np.mean(outlier_ratios)
        
        print(f"📊 Media Silhouette: {avg_silhouette:.4f}")
        print(f"📊 Media Outlier Ratio: {avg_outlier_ratio:.1%}")
        
        print(f"\n🎯 RACCOMANDAZIONI GENERALI:")
        print("• Puntare a Silhouette Score > 0.20 per cluster di alta qualità")
        print("• Mantenere Outlier Ratio < 30% per buona copertura")
        print("• Bilanciare numero di cluster vs qualità dei cluster")
        print("• Considerare execution time per deployment in produzione")
        
    except Exception as e:
        print(f"❌ Errore durante l'analisi: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        results_db.disconnect()

if __name__ == "__main__":
    analyze_clustering_configurations()
