#!/usr/bin/env python3
"""
Test rapido per verificare se le predizioni ML/LLM vengono salvate durante training supervisionato
"""

import sys
sys.path.append('/home/ubuntu/classificazione_discussioni')

import requests
import json

print("🧪 TEST: Predizioni salvate durante training supervisionato")

# Test con un mini training
payload = {
    "max_sessions": 10,  # Solo 10 sessioni per test rapido
    "confidence_threshold": 0.85,
    "force_retrain": False,
    "disagreement_threshold": 0.3
}

print(f"📋 Lancio mini training supervisionato...")
print(f"   • Max sessioni: {payload['max_sessions']}")
print(f"   • Soglia confidenza: {payload['confidence_threshold']}")

try:
    response = requests.post("http://localhost:5000/train/humanitas", 
                           json=payload, 
                           timeout=60)
    
    if response.status_code == 200:
        print(f"✅ Training completato!")
        
        # Ora verifica se le predizioni sono nel database
        print(f"\n📋 Controllo predizioni salvate...")
        
        review_response = requests.get("http://localhost:5000/review-queue/humanitas?limit=3")
        
        if review_response.status_code == 200:
            sessions = review_response.json()
            
            if sessions:
                print(f"✅ Trovate {len(sessions)} sessioni nella review queue")
                
                for i, session in enumerate(sessions[:2]):
                    print(f"\n📝 Sessione {i+1}:")
                    print(f"   session_id: {session.get('session_id', 'N/A')[:8]}...")
                    print(f"   classification: {session.get('classification', 'N/A')}")
                    print(f"   ml_prediction: {session.get('ml_prediction', 'MANCANTE')}")
                    print(f"   ml_confidence: {session.get('ml_confidence', 'MANCANTE')}")
                    print(f"   llm_prediction: {session.get('llm_prediction', 'MANCANTE')}")
                    print(f"   llm_confidence: {session.get('llm_confidence', 'MANCANTE')}")
                    
                    if session.get('llm_prediction') and session.get('llm_prediction') != 'N/A':
                        print(f"   🎉 SUCCESSO: Predizione LLM salvata!")
                    else:
                        print(f"   ❌ ERRORE: Predizione LLM non salvata")
            else:
                print(f"❌ Nessuna sessione nella review queue")
        else:
            print(f"❌ Errore nella review queue API: {review_response.status_code}")
            
    else:
        print(f"❌ Errore nel training: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")

except Exception as e:
    print(f"❌ Errore durante test: {e}")

print(f"\n🏁 Test completato!")
