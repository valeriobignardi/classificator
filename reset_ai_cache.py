#!/usr/bin/env python3
"""
Script per resettare il cache del servizio AI Configuration via API
"""

import requests
import json

def reset_ai_config_cache():
    """Reset del cache del servizio AI Configuration"""
    
    # Endpoint per resettare cache (da implementare se necessario)
    url = "http://localhost:5000/api/ai-config/reset-cache"
    
    try:
        response = requests.post(url, headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            print("✅ Cache del servizio AI Configuration resettato")
            return True
        else:
            print(f"❌ Errore reset cache: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Errore connessione: {e}")
        return False

def test_openai_availability():
    """Testa la disponibilità OpenAI dopo reset"""
    
    url = "http://localhost:5000/api/ai-config/015007d9-d413-11ef-86a5-96000228e7fe/embedding-engines"
    
    try:
        response = requests.get(url, headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            data = response.json()
            openai_large = data['engines']['openai_large']['available']
            openai_small = data['engines']['openai_small']['available']
            
            print(f"🔍 OpenAI Large available: {openai_large}")
            print(f"🔍 OpenAI Small available: {openai_small}")
            
            return openai_large and openai_small
        else:
            print(f"❌ Errore test: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Errore test: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Tentativo reset cache servizio AI Configuration...")
    
    if reset_ai_config_cache():
        print("\n🧪 Test disponibilità OpenAI dopo reset...")
        if test_openai_availability():
            print("✅ OpenAI ora disponibile!")
        else:
            print("❌ OpenAI ancora non disponibile - riavvio server necessario")
    else:
        print("❌ Reset cache fallito - riavvio server necessario")
