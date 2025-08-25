#!/usr/bin/env python3
"""
Test API server per verificare che le predizioni separate appaiano nella review queue
"""

import sys
import requests
import json
sys.path.append('/home/ubuntu/classificazione_discussioni')

print("🧪 TEST: API Review Queue con predizioni separate")

# Prima avvia il server in background se non è già avviato
import subprocess
import time

print("\n📋 Test 1: Avvio server API in background")
try:
    # Check se server già attivo
    response = requests.get("http://localhost:5000/health", timeout=2)
    print("   ✅ Server già attivo")
except:
    print("   🚀 Avvio server in background...")
    server_process = subprocess.Popen([
        "python", "server.py"
    ], cwd="/home/ubuntu/classificazione_discussioni")
    time.sleep(10)  # Attendi che il server si avvii

print("\n📋 Test 2: Recupera review queue per Alleanza")
try:
    response = requests.get("http://localhost:5000/api/review/Alleanza/cases")
    
    if response.status_code == 200:
        cases = response.json().get("cases", [])
        print(f"   ✅ Trovati {len(cases)} casi in review")
        
        if cases:
            print(f"\n📋 Test 3: Analizza predizioni nei primi 3 casi")
            for i, case in enumerate(cases[:3]):
                print(f"\n   📝 Caso {i+1}:")
                print(f"     session_id: {case.get('session_id', 'N/A')[:12]}...")
                print(f"     ml_prediction: {case.get('ml_prediction', 'MANCANTE')}")
                print(f"     ml_confidence: {case.get('ml_confidence', 'MANCANTE')}")
                print(f"     llm_prediction: {case.get('llm_prediction', 'MANCANTE')}")
                print(f"     llm_confidence: {case.get('llm_confidence', 'MANCANTE')}")
                
                # Verifica se abbiamo predizioni separate
                has_ml = case.get('ml_prediction') not in [None, '', 'N/A', 'MANCANTE']
                has_llm = case.get('llm_prediction') not in [None, '', 'N/A', 'MANCANTE']
                
                if has_ml or has_llm:
                    print(f"     ✅ SUCCESSO: Ha predizioni separate!")
                else:
                    print(f"     ❌ Problema: Nessuna predizione separata")
                    
            # Cerca specificamente la nostra sessione di test
            test_session_found = False
            for case in cases:
                if case.get('llm_prediction') == 'prenotazione_esami' and case.get('llm_confidence') == 1.0:
                    test_session_found = True
                    print(f"\n   🎯 FOUND TEST SESSION:")
                    print(f"     ml_prediction: {case.get('ml_prediction')}")
                    print(f"     ml_confidence: {case.get('ml_confidence')}")
                    print(f"     llm_prediction: {case.get('llm_prediction')}")
                    print(f"     llm_confidence: {case.get('llm_confidence')}")
                    break
            
            if not test_session_found:
                print(f"   ⚠️ Sessione di test non trovata in review queue (normale se confidenza alta)")
        else:
            print(f"   ⚠️ Nessun caso in review queue")
    else:
        print(f"   ❌ Errore API: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        
except Exception as e:
    print(f"   ❌ Errore nel test API: {e}")

print(f"\n🏁 Test API completato!")
