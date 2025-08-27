#!/usr/bin/env python3

"""
Test Production HDBSCAN Debug - Verifica parametri tenant in produzione
Autore: GitHub Copilot
Data: 27 Agosto 2025
"""

import requests
import json
import sys

def test_production_hdbscan():
    """
    Testa l'API di produzione per verificare i parametri HDBSCAN
    che arrivano dalla configurazione tenant tramite React UI
    """
    
    print("🧪 [PRODUCTION TEST] Test parametri HDBSCAN da React a produzione")
    print("=" * 70)
    
    # URL API server
    base_url = "http://localhost:5000"
    
    # Test con tenant wopta che hai mostrato nel log
    tenant_id = "16c222a9-f293-11ef-9315-96000228e7fe"  # wopta
    
    print(f"🎯 [TEST] Tenant ID: {tenant_id}")
    print(f"📡 [TEST] Server: {base_url}")
    print()
    
    try:
        # Endpoint di test clustering che dovrebbe mostrare tutti i debug
        url = f"{base_url}/api/clustering/{tenant_id}/test"
        
        print(f"🚀 [REQUEST] POST {url}")
        print("   Questo dovrebbe attivare la pipeline completa con:")
        print("   - Caricamento parametri tenant")
        print("   - Inizializzazione HDBSCANClusterer con parametri React")
        print("   - Debug logging completo")
        print()
        
        # Fai la richiesta con headers corretti
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json={}, headers=headers, timeout=300)
        
        print(f"📡 [RESPONSE] Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ [SUCCESS] Clustering test completato")
            print(f"   📊 Cluster trovati: {result.get('clusters', 'N/A')}")
            print(f"   🔍 Outliers: {result.get('outliers', 'N/A')}")
            print(f"   ⏱️ Tempo: {result.get('execution_time', 'N/A')}")
            print()
            print("🔍 [DEBUG] Controlla i log del server per vedere:")
            print("   🔧 [FIX DEBUG] Parametri tenant passati a HDBSCANClusterer")
            print("   🎯 [DEBUG REACT] Parametri configurati")
            print("   Dovrebbero mostrare i valori dal file tenant config:")
            print("   - cluster_selection_method: leaf")  
            print("   - alpha: 0.1")
            print("   - cluster_selection_epsilon: 0.01")
            print("   - min_cluster_size: 2")
            print("   - allow_single_cluster: false")
            
            return True
        else:
            print(f"❌ [ERROR] Server response: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ [ERROR] Server non raggiungibile su {base_url}")
        print("   Assicurati che il server sia avviato con: python server.py")
        return False
        
    except Exception as e:
        print(f"❌ [ERROR] Errore durante il test: {e}")
        return False

def test_parameters_endpoint():
    """
    Testa l'endpoint specifico per i parametri di clustering  
    """
    
    print("\n🎛️ [PARAMETERS TEST] Test endpoint parametri clustering")
    print("=" * 70)
    
    base_url = "http://localhost:5000"
    tenant_id = "16c222a9-f293-11ef-9315-96000228e7fe"  # wopta
    
    try:
        # Test GET parametri - uso endpoint server standard
        url = f"{base_url}/api/clustering/parameters/{tenant_id}" 
        print(f"🔍 [GET] {url}")
        
        response = requests.get(url)
        
        if response.status_code == 200:
            params = response.json()
            print(f"✅ [SUCCESS] Parametri tenant recuperati:")
            for key, value in params.items():
                print(f"   {key}: {value}")
            return params
        else:
            print(f"⚠️  [INFO] Endpoint parameters non trovato: {response.status_code}")
            print(f"   Provo endpoint alternativo...")
            
            # Prova endpoint diverso se presente
            return None
            
    except Exception as e:
        print(f"❌ [ERROR] Parameters test failed: {e}")
        return None

if __name__ == "__main__":
    print("🚀 AVVIO TEST PRODUZIONE HDBSCAN DEBUG")
    print("=" * 70)
    
    # Test parametri
    params = test_parameters_endpoint()
    
    # Test clustering
    success = test_production_hdbscan()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ [RESULT] Test completato - Controlla i log del server!")
        print("   I parametri tenant dovrebbero essere visibili nei debug logs")
        print("   con la fix applicata per passare tutti i parametri a HDBSCAN")
    else:
        print("❌ [RESULT] Test fallito - Controlla server e connessione")
        
    print("=" * 70)
