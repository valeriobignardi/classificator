#!/usr/bin/env python3
"""
Test per verificare che le predizioni ML/LLM separate vengano salvate correttamente nel database
"""

import sys
sys.path.append('/home/ubuntu/classificazione_discussioni')

from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier
from mongo_classification_reader import MongoClassificationReader
import uuid

print("🧪 TEST: Salvataggio predizioni separate ML/LLM")

# Test 1: Verifica che l'ensemble classifier restituisca predizioni separate
print("\n📋 Test 1: Ensemble classifier predizioni separate")

try:
    ensemble = AdvancedEnsembleClassifier(
        client_name="humanitas",
        confidence_threshold=0.7
    )
    
    test_text = "Vorrei prenotare una visita cardiologica per la prossima settimana"
    
    prediction = ensemble.predict_with_ensemble(test_text, return_details=True)
    
    print(f"   🔍 Predizione risultato:")
    print(f"     predicted_label: {prediction.get('predicted_label')}")
    print(f"     confidence: {prediction.get('confidence')}")
    print(f"     method: {prediction.get('method')}")
    print(f"     ml_prediction: {prediction.get('ml_prediction')}")
    print(f"     llm_prediction: {prediction.get('llm_prediction')}")
    
    # Test 2: Salvataggio in MongoDB
    print(f"\n📋 Test 2: Salvataggio predizioni separate in MongoDB")
    
    mongo_reader = MongoClassificationReader()
    session_id = str(uuid.uuid4())
    
    # Estrai predizioni separate
    ml_result = prediction.get('ml_prediction')
    llm_result = prediction.get('llm_prediction')
    
    print(f"   📦 Predizioni da salvare:")
    print(f"     ML result: {ml_result}")
    print(f"     LLM result: {llm_result}")
    
    success = mongo_reader.save_classification_result(
        session_id=session_id,
        client_name="Alleanza",  # Usa il tenant dal tuo esempio
        ml_result=ml_result,
        llm_result=llm_result,
        final_decision={
            'predicted_label': prediction.get('predicted_label'),
            'confidence': prediction.get('confidence'),
            'method': prediction.get('method'),
            'reasoning': f"Test predizione con confidenza {prediction.get('confidence'):.3f}"
        },
        conversation_text=test_text,
        needs_review=False,
        classified_by="test_ensemble"
    )
    
    if success:
        print(f"   ✅ Classificazione salvata con successo!")
        
        # Test 3: Verificare che i campi siano nel database
        print(f"\n📋 Test 3: Verifica campi salvati nel database")
        
        sessions = mongo_reader.get_all_sessions("Alleanza")
        test_session = None
        
        for session in sessions:
            if session.get('session_id') == session_id:
                test_session = session
                break
        
        if test_session:
            print(f"   ✅ Sessione trovata nel database:")
            print(f"     session_id: {test_session.get('session_id')}")
            print(f"     classification: {test_session.get('classification')}")
            print(f"     ml_prediction: {test_session.get('ml_prediction', 'MANCANTE')}")
            print(f"     ml_confidence: {test_session.get('ml_confidence', 'MANCANTE')}")
            print(f"     llm_prediction: {test_session.get('llm_prediction', 'MANCANTE')}")
            print(f"     llm_confidence: {test_session.get('llm_confidence', 'MANCANTE')}")
            
            if test_session.get('ml_prediction') or test_session.get('llm_prediction'):
                print(f"   🎉 SUCCESSO: Predizioni separate salvate correttamente!")
            else:
                print(f"   ❌ ERRORE: Predizioni separate NON salvate")
                
        else:
            print(f"   ❌ ERRORE: Sessione non trovata nel database")
    else:
        print(f"   ❌ ERRORE: Fallimento nel salvataggio")
        
except Exception as e:
    print(f"   ❌ ERRORE nel test: {e}")
    import traceback
    traceback.print_exc()

print(f"\n🏁 Test completato!")
