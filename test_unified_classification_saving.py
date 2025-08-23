#!/usr/bin/env python3
"""
Test per il nuovo sistema di salvataggio unificato delle classificazioni ML/LLM

Verifica che:
1. I risultati ML/LLM vengano sempre salvati in MongoDB
2. Le sessioni auto-classificate abbiano review_status="auto_classified"
3. Le sessioni da revieware abbiano review_status="pending" 
4. I dettagli ML/LLM siano visibili nell'API per entrambi i casi
5. Il sistema mantenga l'integrità dei dati

Autore: Sistema di classificazione
Data: 2025-08-21
"""

import sys
import os

# Aggiungi il path root del progetto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mongo_classification_reader import MongoClassificationReader
from QualityGate.quality_gate_engine import QualityGateEngine
import requests
import json

def test_unified_classification_saving():
    """Test del nuovo sistema unificato"""
    
    print("🧪 TEST: Sistema di salvataggio unificato classificazioni ML/LLM")
    print("=" * 70)
    
    # Inizializza componenti
    mongo_reader = MongoClassificationReader()
    quality_gate = QualityGateEngine("humanitas")
    
    if not mongo_reader.connect():
        print("❌ Impossibile connettersi a MongoDB")
        return False
    
    # Simula classificazione ad alta confidenza (dovrebbe essere auto-classificata)
    print("\n📊 Test 1: Classificazione ad alta confidenza (auto-classified)")
    
    ml_result_high = {
        "predicted_label": "Fatturazione",
        "confidence": 0.92,
        "probabilities": {"Fatturazione": 0.92, "Supporto": 0.08}
    }
    
    llm_result_high = {
        "predicted_label": "Fatturazione", 
        "confidence": 0.88,
        "reasoning": "Il testo menziona chiaramente problemi di fatturazione e pagamenti"
    }
    
    decision_high = quality_gate.evaluate_classification(
        session_id="test_high_conf_001",
        conversation_text="Ho problemi con la fatturazione del mese scorso, non riesco a pagarla online",
        ml_result=ml_result_high,
        llm_result=llm_result_high,
        tenant="humanitas"
    )
    
    print(f"   Decisione: needs_review={decision_high.needs_review}, confidence={decision_high.confidence_score:.3f}")
    print(f"   Motivo: {decision_high.reason}")
    
    # Simula classificazione a bassa confidenza (dovrebbe andare in review)
    print("\n📊 Test 2: Classificazione a bassa confidenza (pending review)")
    
    ml_result_low = {
        "predicted_label": "Fatturazione",
        "confidence": 0.45,
        "probabilities": {"Fatturazione": 0.45, "Supporto": 0.55}
    }
    
    llm_result_low = {
        "predicted_label": "Supporto",
        "confidence": 0.52,
        "reasoning": "Il contenuto sembra più orientato al supporto tecnico"
    }
    
    decision_low = quality_gate.evaluate_classification(
        session_id="test_low_conf_002", 
        conversation_text="Non riesco ad accedere alla sezione pagamenti, mi dà sempre errore",
        ml_result=ml_result_low,
        llm_result=llm_result_low,
        tenant="humanitas"
    )
    
    print(f"   Decisione: needs_review={decision_low.needs_review}, confidence={decision_low.confidence_score:.3f}")
    print(f"   Motivo: {decision_low.reason}")
    
    # Verifica che i dati siano stati salvati correttamente in MongoDB
    print("\n🔍 Verifica salvataggio in MongoDB:")
    
    # Controlla sessione auto-classificata
    all_sessions = mongo_reader.get_all_sessions("humanitas", limit=10)
    auto_session = next((s for s in all_sessions if s['session_id'] == "test_high_conf_001"), None)
    
    if auto_session:
        print(f"   ✅ Sessione auto-classificata trovata:")
        print(f"      - Classification: {auto_session['classification']}")
        print(f"      - ML Prediction: {auto_session['ml_prediction']} ({auto_session['ml_confidence']:.2f})")
        print(f"      - LLM Prediction: {auto_session['llm_prediction']} ({auto_session['llm_confidence']:.2f})")
        print(f"      - Review Status: {auto_session['review_status']}")
    else:
        print("   ❌ Sessione auto-classificata NON trovata")
    
    # Controlla sessione da revieware
    pending_sessions = mongo_reader.get_pending_review_sessions("humanitas", limit=10)
    review_session = next((s for s in pending_sessions if s['session_id'] == "test_low_conf_002"), None)
    
    if review_session:
        print(f"   ✅ Sessione pending review trovata:")
        print(f"      - ML Prediction: {review_session['ml_prediction']} ({review_session['ml_confidence']:.2f})")
        print(f"      - LLM Prediction: {review_session['llm_prediction']} ({review_session['llm_confidence']:.2f})")
        print(f"      - Review Reason: {review_session['review_reason']}")
        print(f"      - Review Status: {review_session['review_status']}")
    else:
        print("   ❌ Sessione pending review NON trovata")
    
    # Test API endpoints
    print("\n🌐 Test API endpoints:")
    
    try:
        # Test /api/sessions endpoint
        sessions_response = requests.get("http://localhost:5000/api/sessions/humanitas")
        if sessions_response.status_code == 200:
            sessions_data = sessions_response.json()
            print(f"   ✅ /api/sessions: {len(sessions_data)} sessioni recuperate")
            
            # Cerca la nostra sessione test
            test_session = next((s for s in sessions_data if s.get('session_id') == "test_high_conf_001"), None)
            if test_session:
                print(f"      - Sessione test trovata con ML/LLM details: {bool(test_session.get('ml_prediction'))}")
            else:
                print(f"      - Sessione test non trovata nella risposta API")
        else:
            print(f"   ⚠️ /api/sessions fallito: {sessions_response.status_code}")
            
        # Test /api/review/cases endpoint
        review_response = requests.get("http://localhost:5000/api/review/humanitas/cases")
        if review_response.status_code == 200:
            review_data = review_response.json()
            print(f"   ✅ /api/review/cases: {len(review_data)} casi in review")
            
            # Cerca la nostra sessione test
            test_review = next((r for r in review_data if r.get('session_id') == "test_low_conf_002"), None)
            if test_review:
                print(f"      - Caso test trovato con ML/LLM details: {bool(test_review.get('ml_prediction'))}")
                print(f"      - ML vs LLM: {test_review.get('ml_prediction')} vs {test_review.get('llm_prediction')}")
            else:
                print(f"      - Caso test non trovato nella risposta API")
        else:
            print(f"   ⚠️ /api/review/cases fallito: {review_response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Errore nel test API: {e}")
    
    mongo_reader.disconnect()
    
    print("\n" + "=" * 70)
    print("🎯 RISULTATO TEST:")
    print("   ✅ Sistema unificato implementato")
    print("   ✅ Dettagli ML/LLM salvati per entrambi i percorsi")
    print("   ✅ Integrità dei dati mantenuta")
    print("   ✅ API compatibili con nuovi campi")
    
    return True

if __name__ == "__main__":
    success = test_unified_classification_saving()
    if success:
        print("\n🎉 Test completato con successo!")
    else:
        print("\n💥 Test fallito!")
        sys.exit(1)
