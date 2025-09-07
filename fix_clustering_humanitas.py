#!/usr/bin/env python3
"""
Script per correggere i parametri di clustering di Humanitas via API
Autore: Valerio Bignardi
Data: 2025-01-17
"""

import requests
import json
import sys

def fix_clustering_parameters():
    """
    Corregge i parametri di clustering troppo restrittivi di Humanitas
    """
    print("🔧 CORREZIONE PARAMETRI CLUSTERING HUMANITAS")
    print("="*60)
    
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    
    # Parametri ottimali (meno restrittivi)
    optimal_params = {
        'min_cluster_size': 5,        # Era 13 (troppo alto)
        'min_samples': 3,             # Era 16 (troppo alto)
        'cluster_selection_epsilon': 0.1,  # Era 0.28 (troppo alto)
        'alpha': 1.0,                 # Era 0.4 
        'cluster_selection_method': 'eom'  # Mantiene eom
    }
    
    print("📊 PARAMETRI DA APPLICARE:")
    for param, value in optimal_params.items():
        print(f"   - {param}: {value}")
    
    # Dati per l'API del server
    api_data = {
        'parameters': optimal_params
    }
    
    try:
        # Chiamata API per aggiornare i parametri
        url = f"http://localhost:8888/api/update_clustering_parameters/{tenant_id}"
        
        print(f"\\n🚀 Invio richiesta a: {url}")
        print(f"📊 Payload: {json.dumps(api_data, indent=2)}")
        
        response = requests.post(url, json=api_data)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ PARAMETRI AGGIORNATI CON SUCCESSO!")
                print("📊 Nuovi parametri salvati nel database MySQL")
                
                print("\\n🎯 PROSSIMI PASSI:")
                print("1. 🔄 Riavvia la pipeline di clustering")
                print("2. 🧠 Esegui un nuovo clustering sui dati Humanitas")
                print("3. ✅ Verifica che si formino cluster validi (non tutti outlier)")
                print("4. 👥 Testa la selezione rappresentanti")
                
                return True
            else:
                print(f"❌ Errore API: {result.get('error', 'Errore sconosciuto')}")
                return False
        else:
            print(f"❌ Errore HTTP {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ ERRORE: Server non raggiungibile!")
        print("💡 Assicurati che il server sia in esecuzione su localhost:8888")
        print("💡 Comando: python server.py")
        return False
        
    except Exception as e:
        print(f"❌ Errore imprevisto: {e}")
        return False

def verify_server_running():
    """
    Verifica che il server sia in esecuzione
    """
    try:
        response = requests.get("http://localhost:8888/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    print("🔍 Verifico che il server sia in esecuzione...")
    
    if not verify_server_running():
        print("❌ Server non in esecuzione!")
        print("🚀 Avvia il server con: python server.py")
        print("⏱️  Attendi che sia completamente caricato, poi rilancia questo script")
        sys.exit(1)
    
    print("✅ Server in esecuzione")
    
    success = fix_clustering_parameters()
    
    if success:
        print("\\n🎉 CORREZIONE COMPLETATA!")
        print("📈 I parametri di clustering sono ora ottimizzati per 1360 documenti")
    else:
        print("\\n❌ CORREZIONE FALLITA!")
        sys.exit(1)
