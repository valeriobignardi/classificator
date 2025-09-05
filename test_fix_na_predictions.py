#!/usr/bin/env python3
"""
File: test_fix_na_predictions.py
Autore: Valerio Bignardi
Data: 2025-09-05

Descrizione: Test per verificare la correzione del problema N/A nelle predizioni ML/LLM

Storia aggiornamenti:
- 2025-09-05: Creazione test per verificare fix predizioni N/A
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Classification.intelligent_classifier import IntelligentClassifier
from mongo_classification_reader import MongoClassificationReader
import pymongo
from datetime import datetime

def test_na_predictions_fix():
    """
    Test per verificare che le predizioni ML/LLM non siano più N/A
    """
    print("🧪 TEST: Verifica correzione predizioni N/A")
    print("=" * 60)
    
    # Inizializza classifier
    try:
        config_path = "/home/ubuntu/classificatore/config.yaml"
        classifier = IntelligentClassifier(
            config_path=config_path,
            client_name="humanitas"
        )
        print("✅ IntelligentClassifier inizializzato")
    except Exception as e:
        print(f"❌ Errore inizializzazione classifier: {e}")
        return False
    
    # Testo di test
    test_text = """[UTENTE] Buongiorno, vorrei prenotare una visita cardiologica per la prossima settimana. [ASSISTENTE] Buongiorno! Sarò felice di aiutarla con la prenotazione. Per procedere con la prenotazione di una visita cardiologica, ho bisogno di alcune informazioni."""
    
    print(f"\n📝 Testo di test (primi 100 caratteri): {test_text[:100]}...")
    
    # Esegui classificazione
    try:
        result = classifier.classify_conversation(test_text)
        print(f"\n🎯 Risultato classificazione:")
        print(f"   Label: {result.get('predicted_label', 'MISSING')}")
        print(f"   Confidence: {result.get('confidence', 0.0):.3f}")
        print(f"   Method: {result.get('method', 'unknown')}")
        print(f"   Motivation: {str(result.get('motivation', ''))[:100]}...")
        
        predicted_label = result.get('predicted_label', '')
    except Exception as e:
        print(f"❌ Errore durante classificazione: {e}")
        return False
    
    # Controlla MongoDB per vedere se i campi sono popolati
    try:
        client = pymongo.MongoClient('mongodb://localhost:27017')
        db = client['classificazioni']
        collection = db['humanitas_015007d9-d413-11ef-86a5-96000228e7fe']
        
        # Cerca il documento più recente (dovrebbe essere quello appena creato)
        latest_doc = collection.find().sort('_id', -1).limit(1)[0]
        
        print(f"\n📋 Documento MongoDB più recente:")
        print(f"   ID: {latest_doc['_id']}")
        print(f"   predicted_label: {latest_doc.get('predicted_label', 'MISSING')}")
        print(f"   ml_prediction: {latest_doc.get('ml_prediction', 'MISSING')}")
        print(f"   llm_prediction: {latest_doc.get('llm_prediction', 'MISSING')}")
        print(f"   classification: {latest_doc.get('classification', 'MISSING')}")
        print(f"   classification_method: {latest_doc.get('classification_method', 'MISSING')}")
        print(f"   embedding presente: {'embedding' in latest_doc}")
        print(f"   embedding_model: {latest_doc.get('embedding_model', 'MISSING')}")
        
        # Verifica se la correzione ha funzionato
        has_predicted_label = 'predicted_label' in latest_doc and latest_doc['predicted_label'] != ''
        has_ml_or_llm = ('ml_prediction' in latest_doc and latest_doc['ml_prediction'] != '') or \
                       ('llm_prediction' in latest_doc and latest_doc['llm_prediction'] != '')
        
        if has_predicted_label and has_ml_or_llm:
            print(f"\n✅ SUCCESSO: Fix applicato correttamente!")
            print(f"   ✓ predicted_label popolato: {latest_doc.get('predicted_label')}")
            print(f"   ✓ ml/llm_prediction popolato")
            return True
        else:
            print(f"\n❌ FALLIMENTO: Fix non applicato correttamente")
            print(f"   predicted_label OK: {has_predicted_label}")
            print(f"   ml/llm_prediction OK: {has_ml_or_llm}")
            return False
            
    except Exception as e:
        print(f"❌ Errore controllo MongoDB: {e}")
        return False

if __name__ == "__main__":
    success = test_na_predictions_fix()
    print(f"\n🏁 Test completato: {'✅ SUCCESSO' if success else '❌ FALLIMENTO'}")
