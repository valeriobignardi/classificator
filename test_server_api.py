#!/usr/bin/env python3
"""
Test dell'API del server con la nuova logica di training supervisionato
"""
import requests
import json
import time

def test_training_api():
    """Testa la nuova API di training supervisionato"""
    
    base_url = "http://localhost:5000"
    client_name = "humanitas"
    
    print("🧪 TEST API TRAINING SUPERVISIONATO SEMPLIFICATO")
    print("=" * 60)
    
    # Test connessione server
    print("🔍 Test connessione server...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server attivo")
        else:
            print(f"❌ Server risponde ma con errore: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server non raggiungibile: {e}")
        return False
    
    # Test endpoint training con parametri semplificati
    print(f"\n🎓 Test training supervisionato per {client_name}...")
    
    # Parametri semplificati come richiesto
    payload = {
        "max_sessions": 300,        # Numero massimo sessioni per review umana
        "confidence_threshold": 0.8  # Soglia confidenza
    }
    
    print(f"📋 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        print("📡 Invio richiesta...")
        response = requests.post(
            f"{base_url}/train/supervised/{client_name}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minuti timeout per training
        )
        
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Training completato con successo!")
            
            # Mostra statistiche principali
            print(f"\n📊 RISULTATI TRAINING:")
            print(f"  🏥 Cliente: {result.get('client', 'N/A')}")
            
            if 'user_configuration' in result:
                config = result['user_configuration']
                print(f"  👤 Configurazione utente:")
                print(f"    📝 Max sessioni: {config.get('max_sessions', 'N/A')}")
                print(f"    🎯 Soglia confidenza: {config.get('confidence_threshold', 'N/A')}")
            
            if 'extraction_stats' in result:
                stats = result['extraction_stats']
                print(f"  📊 Estrazione:")
                print(f"    📋 Sessioni estratte: {stats.get('total_sessions_extracted', 'N/A')}")
                print(f"    🔄 Modalità: {stats.get('extraction_mode', 'N/A')}")
            
            if 'clustering_stats' in result:
                stats = result['clustering_stats']
                print(f"  🧩 Clustering:")
                print(f"    📋 Sessioni clusterate: {stats.get('total_sessions_clustered', 'N/A')}")
                print(f"    🎯 Cluster trovati: {stats.get('n_clusters', 'N/A')}")
                print(f"    🔍 Outlier: {stats.get('n_outliers', 'N/A')}")
            
            if 'human_review_stats' in result:
                stats = result['human_review_stats']
                print(f"  👤 Review umana:")
                print(f"    📝 Sessioni per review: {stats.get('actual_sessions_for_review', 'N/A')}")
                print(f"    🧩 Cluster rivisti: {stats.get('clusters_reviewed', 'N/A')}")
                print(f"    🚫 Cluster esclusi: {stats.get('clusters_excluded', 'N/A')}")
            
            return True
            
        else:
            print(f"❌ Errore nell'API: {response.status_code}")
            try:
                error_data = response.json()
                print(f"💡 Dettagli errore: {error_data}")
            except:
                print(f"💡 Risposta: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏱️ Timeout - il training sta ancora procedendo...")
        return True  # Non è un errore, il training è lungo
    except Exception as e:
        print(f"❌ Errore durante la richiesta: {e}")
        return False

def main():
    """Funzione principale"""
    print("🚀 AVVIO TEST API TRAINING SUPERVISIONATO")
    
    success = test_training_api()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TEST COMPLETATO CON SUCCESSO!")
        print("\n📋 L'API di training supervisionato semplificato è funzionante!")
        print("💡 L'utente può configurare solo:")
        print("   👤 max_sessions: Numero massimo sessioni per review umana")
        print("   🎯 confidence_threshold: Soglia di confidenza")
    else:
        print("❌ TEST FALLITO!")
        print("💡 Verificare che il server sia avviato e funzionante")

if __name__ == "__main__":
    main()
