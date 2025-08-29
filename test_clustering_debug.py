#!/usr/bin/env python3
"""
Test script per verificare i nuovi debug del clustering
Autore: Valerio Bignardi  
Data: 29 Agosto 2025

Test delle modifiche:
1. Debug dettagliato risultati clustering
2. Errore esplicativo se clustering fallisce
3. Rimozione fallback keyword
4. Interruzione processo con messaggio chiaro
"""

import os
import sys
from datetime import datetime

# Assicura che la root del progetto sia nel PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Pipeline.end_to_end_pipeline import EndToEndPipeline


def test_clustering_debug():
    """
    Test del nuovo sistema di debug clustering
    """
    print(f"🧪 TEST DEBUG CLUSTERING - {datetime.now()}")
    print("🎯 Obiettivo: Verificare nuovi debug e gestione errori")
    print("-" * 60)
    
    try:
        # Inizializza pipeline
        print("🔧 Inizializzazione pipeline...")
        pipe = EndToEndPipeline(
            tenant_slug='humanitas', 
            auto_mode=False
        )
        print("✅ Pipeline inizializzata")
        
        # Test con dataset molto piccolo per forzare clustering fallito
        print(f"\n🧪 TEST 1: Dataset piccolo (forza clustering fallimento)")
        print("🎯 Aspettato: Clustering fallisce → Errore esplicativo → Processo interrotto")
        
        try:
            risultato = pipe.esegui_training_interattivo(
                max_human_review_sessions=10,  # Piccolo per test
                confidence_threshold=0.9       # Alto per test
            )
            print("❌ ERRORE: Il training non dovrebbe completarsi con dataset così piccolo!")
            
        except ValueError as e:
            print(f"✅ SUCCESSO: Errore atteso catturato:")
            print(f"   📋 Messaggio: {str(e)}")
            print(f"   🎯 Comportamento corretto: processo interrotto")
            
        except Exception as e:
            print(f"⚠️ ERRORE INASPETTATO: {type(e).__name__}: {e}")
    
        print(f"\n🧪 TEST 2: Dataset normale (clustering dovrebbe funzionare)")
        print("🎯 Aspettato: Clustering riesce → Debug dettagliato → Training procede")
        
        # Prova con parametri più permissivi
        try:
            risultato = pipe.esegui_training_interattivo(
                max_human_review_sessions=50,
                confidence_threshold=0.5
            )
            
            if risultato and 'training_metrics' in risultato:
                print("✅ SUCCESSO: Training completato correttamente")
                print(f"   📊 Cluster: {risultato.get('clustering', {}).get('n_clusters', 'N/A')}")
                print(f"   📋 Accuracy: {risultato.get('training_metrics', {}).get('training_accuracy', 'N/A')}")
            else:
                print("⚠️ WARNING: Training completato ma risultato incompleto")
                
        except ValueError as e:
            print(f"⚠️ Clustering fallito anche con parametri permissivi:")
            print(f"   📋 Messaggio: {str(e)}")
            print(f"   💡 Potrebbe indicare problemi di dataset o configurazione")
            
        except Exception as e:
            print(f"❌ ERRORE INASPETTATO: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n✅ Test debug clustering completato!")
        
    except Exception as e:
        print(f"❌ ERRORE CRITICO nel test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_clustering_debug()
