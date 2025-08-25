#!/usr/bin/env python3
"""
Test per verificare che ML/LLM predictions siano separate nella review queue
"""

import sys
import os
sys.path.append('/home/ubuntu/classificazione_discussioni')

from mongo_classification_reader import MongoClassificationReader

print("🧪 TEST: Verifico se le predizioni ML/LLM separate sono nel database")

# Connetti a MongoDB e prendi alcuni esempi
mongo_reader = MongoClassificationReader()
mongo_reader.connect()

# Prendi alcune classificazioni dal database per vedere la struttura
tenant_slug = "humanitas"
sessions = mongo_reader.get_all_sessions(client_name=tenant_slug, limit=10)

if sessions:
    print(f"✅ Trovate {len(sessions)} classificazioni di esempio")
    
    for i, session in enumerate(sessions[:3]):
        print(f"\n📋 Esempio {i+1}:")
        print(f"   session_id: {session.get('session_id', 'N/A')}")
        print(f"   classification: {session.get('classification', 'N/A')}")
        print(f"   ml_prediction: {session.get('ml_prediction', 'N/A')}")
        print(f"   llm_prediction: {session.get('llm_prediction', 'N/A')}")
        print(f"   ml_confidence: {session.get('ml_confidence', 'N/A')}")
        print(f"   llm_confidence: {session.get('llm_confidence', 'N/A')}")
        
        # Verifica se ha predizioni separate
        has_separate_ml = 'ml_prediction' in session and session['ml_prediction']
        has_separate_llm = 'llm_prediction' in session and session['llm_prediction']
        
        if has_separate_ml and has_separate_llm:
            ml_pred = session['ml_prediction']
            llm_pred = session['llm_prediction']
            if ml_pred != llm_pred:
                print(f"   🎉 TROVATO DISAGREEMENT: ML='{ml_pred}' vs LLM='{llm_pred}'")
            else:
                print(f"   ⚖️ Agreement: ML=LLM='{ml_pred}'")
        else:
            print(f"   ⚠️ Predizioni separate non disponibili")
    
    # Statistiche generali
    total_separate = 0
    total_disagreement = 0
    
    for session in sessions:
        has_ml = 'ml_prediction' in session and session['ml_prediction']
        has_llm = 'llm_prediction' in session and session['llm_prediction']
        
        if has_ml and has_llm:
            total_separate += 1
            if session['ml_prediction'] != session['llm_prediction']:
                total_disagreement += 1
    
    print(f"\n📊 STATISTICHE:")
    print(f"   Total sessions: {len(sessions)}")
    print(f"   Con predizioni separate: {total_separate}")
    print(f"   Con disagreement: {total_disagreement}")
    
    if total_separate > 0:
        print(f"   % Disagreement: {(total_disagreement/total_separate)*100:.1f}%")
        print("✅ Sistema salva correttamente predizioni separate!")
    else:
        print("⚠️ Nessuna predizione separata trovata nel database")
        
else:
    print("❌ Errore nel recuperare dati dal database")
    print(f"Sessions: {sessions}")

mongo_reader.disconnect()
print("\n🔍 Test completato!")
