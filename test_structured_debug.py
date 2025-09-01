#!/usr/bin/env python3
"""
Test diretto del metodo _call_ollama_api_structured per debuggare il problema
Autore: Valerio Bignardi  
Data: 2025-09-01
"""

import sys
import os
import traceback

# Aggiungi path del progetto
sys.path.append('/home/ubuntu/classificatore')

from Classification.intelligent_classifier import IntelligentClassifier

def test_structured_method_directly():
    """
    Testa direttamente il metodo _call_ollama_api_structured per identificare l'errore
    """
    print("🧪 TEST DIRETTO _call_ollama_api_structured")
    print("="*80)
    
    try:
        # Inizializza i componenti necessari
        print("🔧 Inizializzazione IntelligentClassifier...")
        
        # Intelligent classifier con parametri minimali
        classifier = IntelligentClassifier(
            client_name="humanitas",  # Uso client_name per semplicità
            enable_logging=True  # Debug abilitato
        )
        
        print("✅ Componenti inizializzati")
        print("="*80)
        
        # Test text (stesso dei log)
        conversation_text = "[UTENTE] ho verificato tutto ma continua a non funzionare [UTENTE] quali sono le procedure in caso di allerta meteo? [UTENTE] è prevista acqua alta cosa devo fare?"
        
        print(f"📝 Testo da classificare: {conversation_text}")
        print("="*80)
        
        # Chiama direttamente il metodo structured
        print("🚀 Chiamata diretta _call_ollama_api_structured...")
        result = classifier._call_ollama_api_structured(conversation_text)
        
        print("✅ RISULTATO RICEVUTO:")
        print("="*80)
        print(f"🏷️  predicted_label: {result.get('predicted_label', 'N/A')}")
        print(f"📊 confidence: {result.get('confidence', 'N/A')}")
        print(f"💭 motivation: {result.get('motivation', 'N/A')}")
        print("="*80)
        
    except Exception as e:
        print("❌ ERRORE CATTURATO!")
        print("="*80)
        print(f"Tipo errore: {type(e).__name__}")
        print(f"Messaggio: {str(e)}")
        print("="*80)
        print("📋 TRACEBACK COMPLETO:")
        print("-"*80)
        traceback.print_exc()
        print("="*80)
        
        # Dettagli aggiuntivi se è un errore di database
        if "database" in str(e).lower() or "connection" in str(e).lower():
            print("🔍 ANALISI ERRORE DATABASE:")
            print("❌ Probabilmente problema di connessione o configurazione database")

if __name__ == "__main__":
    test_structured_method_directly()
