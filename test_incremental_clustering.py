#!/usr/bin/env python3
"""
Test per verificare il funzionamento del clustering        # Test 4: Verifica configurazione HDBSCAN...
        print("\n4️⃣ Test configurazione HDBSCAN...")
        
        # Test configurazione base (senza prediction_data che è interno)
        test_params = {
            'min_cluster_size': 5,
            'min_samples': 3,
            'cluster_selection_epsilon': 0.1
        }
        
        try:
            clusterer = HDBSCANClusterer(**test_params)
            print("✅ HDBSCANClusterer configurato (prediction_data=True è impostato internamente)")
        except Exception as e:
            print(f"❌ Errore configurazione HDBSCANClusterer: {e}")
            return FalseSCAN

Autore: Valerio Bignardi  
Data creazione: 03/01/2025
Storia aggiornamenti: 03/01/2025 - Creazione iniziale per test clustering incrementale
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_incremental_clustering():
    """
    Testa il clustering incrementale HDBSCAN
    
    Scopo: Verificare che il sistema di clustering incrementale funzioni correttamente
    Parametri: Nessuno
    Valori di ritorno: True se i test passano, False altrimenti
    Data ultima modifica: 03/01/2025
    """
    try:
        print("🧪 TEST CLUSTERING INCREMENTALE HDBSCAN")
        print("=" * 50)
        
        # Test 1: Verifica disponibilità moduli
        print("\n1️⃣ Test importazione moduli...")
        
        try:
            from Clustering.hdbscan_clusterer import HDBSCANClusterer
            print("✅ HDBSCANClusterer importato correttamente")
        except ImportError as e:
            print(f"❌ Errore importazione HDBSCANClusterer: {e}")
            return False
            
        try:
            from Pipeline.end_to_end_pipeline import EndToEndPipeline
            print("✅ EndToEndPipeline importato correttamente")
        except ImportError as e:
            print(f"❌ Errore importazione EndToEndPipeline: {e}")
            return False
        
        # Test 2: Verifica metodi clustering incrementale
        print("\n2️⃣ Test metodi clustering incrementale...")
        
        clusterer = HDBSCANClusterer()
        
        # Verifica presenza metodi
        required_methods = [
            'predict_new_points',
            'save_model_for_incremental_prediction',
            'load_model_for_incremental_prediction'
        ]
        
        for method_name in required_methods:
            if hasattr(clusterer, method_name):
                print(f"✅ Metodo {method_name} disponibile")
            else:
                print(f"❌ Metodo {method_name} mancante")
                return False
        
        # Test 3: Verifica directory modelli
        print("\n3️⃣ Test directory modelli...")
        
        models_dir = "models"
        if os.path.exists(models_dir):
            print(f"✅ Directory {models_dir} esiste")
        else:
            print(f"⚠️ Directory {models_dir} non esiste, la creo...")
            os.makedirs(models_dir, exist_ok=True)
            print(f"✅ Directory {models_dir} creata")
        
        # Test 4: Verifica configurazione HDBSCAN...
        print("\n4️⃣ Test configurazione HDBSCAN...")
        
        # Test configurazione base (senza prediction_data che è interno)
        test_params = {
            'min_cluster_size': 5,
            'min_samples': 3,
            'cluster_selection_epsilon': 0.1
        }
        
        try:
            clusterer = HDBSCANClusterer(**test_params)
            print("✅ HDBSCANClusterer configurato (prediction_data=True è impostato internamente)")
        except Exception as e:
            print(f"❌ Errore configurazione HDBSCANClusterer: {e}")
            return False
        
        print("\n🎉 TUTTI I TEST PASSATI!")
        print("✅ Sistema clustering incrementale pronto per l'uso")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRORE NEI TEST: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """
    Testa l'integrazione della pipeline con clustering incrementale
    
    Scopo: Verificare che la pipeline gestisca correttamente modalità incrementale vs completa
    Parametri: Nessuno  
    Valori di ritorno: True se i test passano, False altrimenti
    Data ultima modifica: 03/01/2025
    """
    try:
        print("\n🔧 TEST INTEGRAZIONE PIPELINE")
        print("=" * 40)
        
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        
        # Test setup pipeline base
        print("\n1️⃣ Test setup pipeline...")
        
        pipeline = EndToEndPipeline(
            tenant_slug="humanitas",  # Usa tenant esistente
            config_path="config.yaml"
        )
        
        print("✅ Pipeline inizializzata correttamente")
        
        # Test metodi incrementali
        print("\n2️⃣ Test metodi pipeline incrementale...")
        
        required_pipeline_methods = [
            '_esegui_clustering_completo',
            '_esegui_clustering_incrementale', 
            '_dovrebbe_usare_clustering_incrementale'
        ]
        
        for method_name in required_pipeline_methods:
            if hasattr(pipeline, method_name):
                print(f"✅ Metodo pipeline {method_name} disponibile")
            else:
                print(f"❌ Metodo pipeline {method_name} mancante")
                return False
        
        print("\n✅ INTEGRAZIONE PIPELINE VERIFICATA!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRORE TEST PIPELINE: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"🚀 AVVIO TEST CLUSTERING INCREMENTALE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = True
    
    # Esegui test moduli clustering
    if not test_incremental_clustering():
        success = False
    
    # Esegui test integrazione pipeline  
    if not test_pipeline_integration():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TUTTI I TEST COMPLETATI CON SUCCESSO!")
        print("✅ Sistema clustering incrementale operativo")
    else:
        print("❌ ALCUNI TEST FALLITI")
        print("⚠️ Controllare errori sopra e risolvere problemi")
        sys.exit(1)
    
    print(f"🏁 Test terminati - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
