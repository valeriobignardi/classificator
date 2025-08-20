#!/usr/bin/env python3
"""
Test del nuovo endpoint API per training supervisionato avanzato
"""

import requests
import json
import time

def test_advanced_training_endpoint():
    """
    Testa il nuovo endpoint /train/supervised/advanced/<client_name>
    """
    print("🧪 TEST ENDPOINT TRAINING SUPERVISIONATO AVANZATO")
    print("=" * 60)
    
    # Configurazione test
    base_url = "http://localhost:5000"
    client_name = "humanitas"
    endpoint = f"{base_url}/train/supervised/advanced/{client_name}"
    
    # Payload di test
    payload = {
        "max_human_review_sessions": 10,  # Limitato per test veloce
        "representatives_per_cluster": 2,
        "force_retrain": False
    }
    
    print(f"🎯 Endpoint: {endpoint}")
    print(f"📋 Payload: {json.dumps(payload, indent=2)}")
    print()
    
    try:
        print("🚀 Invio richiesta...")
        start_time = time.time()
        
        response = requests.post(
            endpoint,
            json=payload,
            timeout=300  # 5 minuti timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️  Durata richiesta: {duration:.1f} secondi")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ RICHIESTA SUCCESSFUL!")
            
            try:
                result = response.json()
                print("📋 RISULTATO:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
                # Valida struttura risposta
                expected_keys = ['success', 'message', 'client', 'extraction_stats', 
                               'clustering_stats', 'human_review_stats', 'training_metrics']
                
                missing_keys = [key for key in expected_keys if key not in result]
                if missing_keys:
                    print(f"⚠️ Chiavi mancanti nella risposta: {missing_keys}")
                else:
                    print("✅ Struttura risposta valida")
                
                # Verifica logica avanzata
                extraction_stats = result.get('extraction_stats', {})
                if extraction_stats.get('extraction_mode') == 'FULL_DATASET':
                    print("✅ Estrazione completa confermata")
                else:
                    print("⚠️ Estrazione completa non confermata")
                
                human_review_stats = result.get('human_review_stats', {})
                max_sessions = human_review_stats.get('max_sessions_for_review', 0)
                actual_sessions = human_review_stats.get('actual_sessions_for_review', 0)
                
                if actual_sessions <= max_sessions:
                    print(f"✅ Limite review rispettato: {actual_sessions}/{max_sessions}")
                else:
                    print(f"⚠️ Limite review superato: {actual_sessions}/{max_sessions}")
                
            except json.JSONDecodeError as e:
                print(f"❌ Errore parsing JSON: {e}")
                print(f"Raw response: {response.text[:500]}...")
                
        else:
            print("❌ RICHIESTA FAILED!")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ TIMEOUT: La richiesta ha superato i 5 minuti")
        
    except requests.exceptions.ConnectionError:
        print("🔌 ERRORE CONNESSIONE: Server non raggiungibile")
        print("💡 Assicurati che il server Flask sia avviato su porta 5000")
        
    except Exception as e:
        print(f"❌ ERRORE INASPETTATO: {e}")

def test_server_health():
    """
    Verifica che il server sia attivo
    """
    print("🏥 TEST HEALTH SERVER")
    print("-" * 30)
    
    try:
        response = requests.get("http://localhost:5000/health", timeout=10)
        if response.status_code == 200:
            print("✅ Server attivo e raggiungibile")
            return True
        else:
            print(f"⚠️ Server risponde ma con status {response.status_code}")
            return False
    except:
        print("❌ Server non raggiungibile")
        return False

if __name__ == "__main__":
    print("🧪 TEST COMPLETO API TRAINING SUPERVISIONATO AVANZATO")
    print("=" * 80)
    
    # Test 1: Health check
    if test_server_health():
        print()
        # Test 2: Nuovo endpoint
        test_advanced_training_endpoint()
    else:
        print()
        print("💡 SUGGERIMENTI:")
        print("1. Avvia il server Flask: python server.py")
        print("2. Verifica che sia in ascolto su porta 5000")
        print("3. Controlla che non ci siano errori nei log")
    
    print()
    print("🏁 TEST COMPLETATO")
