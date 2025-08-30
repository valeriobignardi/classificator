#!/usr/bin/env python3
"""
File: test_database_tools_integration.py
Autore: Valerio Bignardi
Data creazione: 2025-01-27
Scopo: Test integrazione ToolManager con IntelligentClassifier
"""

import sys
import os
import traceback
from typing import Dict, Any

# Aggiungi il percorso della directory classificatore
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from Utils.tool_manager import ToolManager
    from Classification.intelligent_classifier import IntelligentClassifier
except ImportError as e:
    print(f"❌ Errore import: {e}")
    sys.exit(1)


def test_tool_manager_availability():
    """
    Test disponibilità ToolManager e tools nel database
    
    Returns:
        bool: True se il ToolManager è disponibile e contiene tools
    """
    print("🔧 Test ToolManager...")
    
    try:
        tool_manager = ToolManager()
        print(f"✅ ToolManager inizializzato correttamente")
        
        # Controlla se esiste il tool di classificazione
        classify_tool = tool_manager.get_tool_by_name("classify_conversation")
        if classify_tool:
            print(f"✅ Tool 'classify_conversation' trovato nel database:")
            print(f"   - Nome: {classify_tool['nome']}")
            print(f"   - Descrizione: {classify_tool['descrizione'][:80]}...")
            print(f"   - Schema: {len(classify_tool['schema_json'])} caratteri")
            return True
        else:
            print("⚠️ Tool 'classify_conversation' non trovato nel database")
            # Mostra tutti i tools disponibili
            all_tools = tool_manager.get_all_tools()
            print(f"Tools disponibili: {[tool['nome'] for tool in all_tools]}")
            return False
            
    except Exception as e:
        print(f"❌ Errore ToolManager: {e}")
        traceback.print_exc()
        return False


def test_intelligent_classifier_with_tools():
    """
    Test IntelligentClassifier con integrazione database tools
    
    Returns:
        bool: True se il test passa
    """
    print("\n🤖 Test IntelligentClassifier con database tools...")
    
    try:
        # Crea classificatore con logging abilitato
        classifier = IntelligentClassifier(enable_logging=True)
        print(f"✅ IntelligentClassifier inizializzato")
        
        # Verifica che il ToolManager sia stato inizializzato
        if not classifier.tool_manager:
            print("⚠️ ToolManager non inizializzato nel classificatore")
            return False
        
        print(f"✅ ToolManager integrato nel classificatore")
        
        # Test classificazione semplice per verificare che funzioni
        test_text = "Buongiorno, vorrei prenotare una visita cardiologica"
        print(f"\n🧪 Test classificazione: '{test_text}'")
        
        result = classifier.classify_with_motivation(test_text)
        
        print(f"✅ Classificazione completata:")
        print(f"   - Etichetta: {result.predicted_label}")
        print(f"   - Confidenza: {result.confidence:.3f}")
        print(f"   - Motivazione: {result.motivation}")
        print(f"   - Metodo: {result.method}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore IntelligentClassifier: {e}")
        traceback.print_exc()
        return False


def test_tool_fallback():
    """
    Test funzionamento fallback quando database tools non sono disponibili
    
    Returns:
        bool: True se il fallback funziona
    """
    print("\n🔄 Test fallback tools...")
    
    try:
        # Simula un classificatore senza ToolManager (forzando None)
        classifier = IntelligentClassifier(enable_logging=True)
        
        # Forza il tool_manager a None per testare il fallback
        original_tool_manager = classifier.tool_manager
        classifier.tool_manager = None
        
        print("⚠️ ToolManager disabilitato per test fallback")
        
        # Test classificazione con fallback
        test_text = "Ciao, non riesco ad accedere al mio account"
        result = classifier.classify_with_motivation(test_text)
        
        print(f"✅ Fallback funziona:")
        print(f"   - Etichetta: {result.predicted_label}")
        print(f"   - Confidenza: {result.confidence:.3f}")
        print(f"   - Metodo: {result.method}")
        
        # Ripristina il tool_manager
        classifier.tool_manager = original_tool_manager
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test fallback: {e}")
        traceback.print_exc()
        return False


def main():
    """
    Esegue tutti i test di integrazione
    """
    print("🚀 Test Integrazione Database Tools con IntelligentClassifier")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: ToolManager
    if test_tool_manager_availability():
        success_count += 1
    
    # Test 2: IntelligentClassifier con tools
    if test_intelligent_classifier_with_tools():
        success_count += 1
    
    # Test 3: Fallback
    if test_tool_fallback():
        success_count += 1
    
    print(f"\n📊 Risultati: {success_count}/{total_tests} test superati")
    
    if success_count == total_tests:
        print("✅ Tutti i test sono passati! Integrazione database tools OK")
    else:
        print(f"⚠️ {total_tests - success_count} test falliti")
        
    return success_count == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
