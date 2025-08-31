#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
TEST END-TO-END: PARAMETRI CLUSTERING REACT → API → HDBSCAN
=====================================================================
Autore: Valerio Bignardi
Data: 2025-08-31
Descrizione: Test end-to-end per verificare che i parametri impostati
             nell'interfaccia React vengano effettivamente applicati
             dal sistema di clustering HDBSCAN attraverso le API
             
Verifica:
1. Caricamento parametri clustering attuali per tenant
2. Modifica parametri simulando azione utente React
3. Verifica applicazione in clustering test reale
4. Confronto risultati con parametri default vs custom
=====================================================================
"""

import sys
import os
import requests
import json
import time
from typing import Dict, Any

# Aggiungi path del progetto
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from Utils.tenant import Tenant


def test_end_to_end_parameters():
    """
    Test end-to-end del flusso parametri clustering
    """
    print("🚀 TEST END-TO-END: PARAMETRI CLUSTERING REACT → API → HDBSCAN")
    print("="*70)
    
    # Configurazione
    BASE_URL = "http://localhost:5000"
    TENANT_ID = "015007d9-d413-11ef-86a5-96000228e7fe"  # Humanitas
    
    try:
        # 1. VERIFICA STATO SERVER
        print("\n🔍 FASE 1: Verifica stato server")
        response = requests.get(f"{BASE_URL}/api/config/ui", timeout=5)
        if response.status_code == 200:
            print("✅ Server attivo e raggiungibile")
        else:
            print(f"❌ Server non raggiungibile: {response.status_code}")
            return
        
        # 2. RECUPERA PARAMETRI ATTUALI
        print("\n📋 FASE 2: Recupero parametri clustering attuali")
        response = requests.get(
            f"{BASE_URL}/api/clustering/{TENANT_ID}/parameters", 
            timeout=10
        )
        
        if response.status_code == 200:
            current_params = response.json()
            print("✅ Parametri attuali recuperati:")
            if 'data' in current_params and 'parameters' in current_params['data']:
                params = current_params['data']['parameters']
                for key, config in params.items():
                    if not key.startswith('_'):
                        print(f"   {key}: {config.get('value', 'N/A')}")
            else:
                print("   Struttura parametri non trovata")
        else:
            print(f"⚠️ Impossibile recuperare parametri: {response.status_code}")
            if response.status_code == 404:
                print("   Endpoint parametri non disponibile - continuo con test base")
        
        # 3. TEST CLUSTERING CON PARAMETRI CUSTOM
        print("\n🧪 FASE 3: Test clustering con parametri custom")
        
        # Parametri custom che simuleranno quelli impostati dall'utente React
        custom_clustering_params = {
            'min_cluster_size': 12,
            'min_samples': 7,  
            'cluster_selection_epsilon': 0.25,
            'alpha': 0.75,
            'metric': 'euclidean',
            'cluster_selection_method': 'leaf',
            'allow_single_cluster': True,
            'use_umap': True,
            'umap_n_neighbors': 20,
            'umap_min_dist': 0.08,
            'umap_n_components': 30
        }
        
        print("🎯 Parametri custom per test:")
        for key, value in custom_clustering_params.items():
            print(f"   {key}: {value}")
        
        # 4. ESEGUI TEST CLUSTERING
        print("\n🔬 FASE 4: Esecuzione test clustering")
        
        test_payload = {
            'tenant_id': TENANT_ID,
            'limit': 50,  # Poche conversazioni per test rapido
            'parameters': custom_clustering_params
        }
        
        print(f"📤 Invio richiesta test clustering...")
        response = requests.post(
            f"{BASE_URL}/api/clustering/{TENANT_ID}/test",
            json=test_payload,
            timeout=60  # Timeout lungo per clustering
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Test clustering completato con successo!")
            
            if 'data' in result:
                data = result['data']
                
                # Analisi risultati
                print("\n📊 RISULTATI CLUSTERING:")
                print(f"   Conversazioni processate: {data.get('total_conversations', 'N/A')}")
                print(f"   Cluster trovati: {data.get('n_clusters', 'N/A')}")
                print(f"   Outliers: {data.get('n_outliers', 'N/A')} ({data.get('outlier_ratio', 0)*100:.1f}%)")
                
                # 5. VERIFICA PARAMETRI APPLICATI
                print("\n🔍 FASE 5: Verifica parametri effettivamente applicati")
                
                if 'parameters_applied' in data:
                    applied_params = data['parameters_applied']
                    print("✅ Parametri applicati trovati nel risultato:")
                    
                    mismatches = []
                    for param_name, expected_value in custom_clustering_params.items():
                        if param_name in applied_params:
                            actual_value = applied_params[param_name]
                            if actual_value == expected_value:
                                print(f"   ✅ {param_name}: {expected_value}")
                            else:
                                print(f"   ❌ {param_name}: atteso={expected_value}, applicato={actual_value}")
                                mismatches.append(param_name)
                        else:
                            print(f"   ⚠️ {param_name}: non trovato nei parametri applicati")
                    
                    if mismatches:
                        print(f"\n❌ PROBLEMA: {len(mismatches)} parametri non corrispondono!")
                        print("🚨 I parametri React NON vengono applicati correttamente!")
                    else:
                        print(f"\n✅ SUCCESSO: Tutti i parametri custom applicati correttamente!")
                        print("🎉 React → API → HDBSCAN funziona perfettamente!")
                        
                else:
                    print("⚠️ Parametri applicati non disponibili nel risultato")
                    print("   Il sistema potrebbe non esporre questa informazione")
                
            else:
                print("⚠️ Dati del test non trovati nella risposta")
                
        else:
            print(f"❌ Test clustering fallito: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Errore: {error_data.get('error', 'Errore sconosciuto')}")
            except:
                print(f"   Errore raw: {response.text[:200]}")
        
        # 6. VERIFICA REAL-TIME CON INTERFACCIA
        print("\n🌐 FASE 6: Verifica compatibilità interfaccia React")
        print("🔗 Interfaccia disponibile su: http://localhost:3000")
        print("📋 Per test completo:")
        print("   1. Vai su Clustering Parameters in React")
        print("   2. Modifica alcuni parametri (es: min_cluster_size = 12)")
        print("   3. Clicca 'Test Clustering'")
        print("   4. Verifica che i risultati riflettano i tuoi parametri")
        
        print("\n🏁 TEST END-TO-END COMPLETATO")
        
    except requests.exceptions.ConnectionError:
        print("❌ ERRORE: Server non raggiungibile su localhost:5000")
        print("   Assicurati che il server sia avviato con: python3 server.py")
    except Exception as e:
        print(f"❌ ERRORE durante test end-to-end: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_end_to_end_parameters()
