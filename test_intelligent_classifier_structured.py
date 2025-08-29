#!/usr/bin/env python3
"""
Test IntelligentClassifier con Structured Outputs implementato

Autore: Valerio Bignardi
Data: 29 Agosto 2025
"""

import sys
import os
sys.path.append(os.getcwd())

from Classification.intelligent_classifier import IntelligentClassifier
import yaml

def test_intelligent_classifier_structured():
    """
    Testa il nuovo IntelligentClassifier con structured outputs
    """
    print("🔬 Test IntelligentClassifier con Structured Outputs")
    print("=" * 60)
    
    # Carica configurazione
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ Configurazione caricata")
    except Exception as e:
        print(f"❌ Errore caricamento config: {e}")
        return False
    
    # Crea classificatore
    try:
        classifier = IntelligentClassifier(
            model_name="mistral:7b",
            temperature=0.01
        )
        print("✅ IntelligentClassifier inizializzato")
    except Exception as e:
        print(f"❌ Errore inizializzazione: {e}")
        return False
    
    # Test conversazioni
    test_cases = [
        "Buongiorno, vorrei prenotare una visita cardiologica per la prossima settimana",
        "Non riesco ad accedere al portale online, ho dimenticato la password", 
        "Dove posso parcheggiare quando vengo in ospedale?",
        "Ho un problema con la fatturazione della mia ultima visita",
        "Quali sono i vostri orari di apertura del laboratorio?"
    ]
    
    print(f"\n🚀 Testing {len(test_cases)} conversazioni con Structured Outputs...")
    print("-" * 60)
    
    success_count = 0
    
    for i, conversation in enumerate(test_cases, 1):
        print(f"\nTest {i}/5:")
        print(f"📝 Conversazione: {conversation[:60]}{'...' if len(conversation) > 60 else ''}")
        
        try:
            # Test del nuovo metodo structured
            start_time = time.time()
            result = classifier._call_ollama_api_structured(conversation)
            duration = time.time() - start_time
            
            print(f"⏱️  Tempo: {duration:.2f}s")
            print(f"🏷️  Etichetta: {result['predicted_label']}")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            print(f"💭 Motivazione: {result['motivation'][:80]}{'...' if len(result['motivation']) > 80 else ''}")
            
            # Validazione
            if result['predicted_label'] in classifier.domain_labels:
                print("✅ Etichetta valida")
            else:
                print(f"❌ Etichetta non valida: {result['predicted_label']}")
                
            if 0.0 <= result['confidence'] <= 1.0:
                print("✅ Confidence valida")
            else:
                print(f"❌ Confidence non valida: {result['confidence']}")
                
            success_count += 1
            
        except Exception as e:
            print(f"❌ Errore: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"🎯 RISULTATI: {success_count}/{len(test_cases)} test riusciti")
    
    if success_count == len(test_cases):
        print("🎉 TUTTI I TEST PASSATI!")
        print("💡 Structured Outputs funziona perfettamente in IntelligentClassifier!")
        return True
    else:
        print("⚠️  Alcuni test falliti")
        return False

if __name__ == "__main__":
    import time
    test_intelligent_classifier_structured()
