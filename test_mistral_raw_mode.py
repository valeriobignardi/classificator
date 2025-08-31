#!/usr/bin/env python3
"""
Test per verificare il supporto raw mode di Mistral 7B v0.3 con function calling

Autore: Valerio Bignardi  
Data: 2025-08-31
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from Classification.intelligent_classifier import IntelligentClassifier
import yaml

def test_mistral_raw_mode():
    """
    Testa il supporto raw mode per Mistral 7B v0.3 function calling
    """
    print("🔧 Test Mistral 7B v0.3 Raw Mode Function Calling")
    print("=" * 60)
    
    # Carica config per verificare raw mode
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    raw_mode_config = config.get('llm', {}).get('ollama', {}).get('raw_mode', {})
    print(f"📋 Configurazione raw mode: {raw_mode_config}")
    
    # Inizializza classifier
    classifier = IntelligentClassifier(
        model_name="mistral:7b",
        client_name="humanitas",
        enable_logging=True
    )
    
    # Test conversation
    test_conversation = """
    Paziente: Buongiorno, vorrei prenotare una visita cardiologica
    Operatore: Certo, che tipo di visita le serve?
    Paziente: Una visita di controllo per ipertensione
    """
    
    print(f"\n📝 Test conversation: {test_conversation}")
    
    try:
        result = classifier.classify_with_motivation(test_conversation)
        
        print(f"\n✅ RISULTATO:")
        print(f"   Etichetta: {result.predicted_label}")
        print(f"   Confidence: {result.confidence}")
        print(f"   Motivazione: {result.motivation}")
        print(f"   Metodo: {result.method}")
        print(f"   Raw mode utilizzato: controllare nei log sopra")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRORE nel test: {e}")
        return False

if __name__ == "__main__":
    success = test_mistral_raw_mode()
    sys.exit(0 if success else 1)
