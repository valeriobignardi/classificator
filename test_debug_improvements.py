#!/usr/bin/env python3
"""
Test dei miglioramenti debug della pipeline

Autore: Valerio Bignardi
Data: 2025-08-29
"""

import sys
import os
import time
sys.path.append(os.path.dirname(__file__))

from Pipeline.end_to_end_pipeline import EndToEndPipeline

def test_debug_improvements():
    """
    Test per verificare che i nuovi debug siano chiari e informativi
    """
    print("=" * 80)
    print("TEST MIGLIORAMENTI DEBUG PIPELINE")
    print("=" * 80)
    
    try:
        print("\nTest 1: Inizializzazione pipeline con debug migliorato")
        pipeline = EndToEndPipeline(tenant_slug="humanitas")
        
        print("\nTest 2: Estrazione sessioni con debug migliorato")
        sessioni = pipeline.estrai_sessioni(
            giorni_indietro=7,
            limit=20,
            force_full_extraction=False
        )
        
        if len(sessioni) >= 3:
            print("\nTest 3: Clustering con debug migliorato")
            embeddings, cluster_labels, representatives, suggested_labels = pipeline.esegui_clustering(
                sessioni, force_reprocess=False
            )
            
            print(f"\nTest 4: Verifica risultati")
            print(f"   📊 Sessioni processate: {len(sessioni)}")
            print(f"   🧠 Embeddings generati: {embeddings.shape}")
            print(f"   🎯 Cluster trovati: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
            print(f"   👥 Rappresentanti: {sum(len(reps) for reps in representatives.values())}")
            
            print("\n✅ TUTTI I TEST COMPLETATI")
            print("🎯 I nuovi debug forniscono:")
            print("   • Indicazione chiara delle fasi")
            print("   • Metriche temporali per ogni fase")
            print("   • Statistiche input/output")
            print("   • Progress indicators")
            print("   • Gestione errori migliorata")
            
        else:
            print(f"⚠️ Dataset troppo piccolo per test clustering: {len(sessioni)} sessioni")
            
    except Exception as e:
        print(f"❌ Errore durante test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug_improvements()
