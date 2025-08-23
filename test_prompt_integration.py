#!/usr/bin/env python3
"""
Test Script per l'integrazione del PromptManager con IntelligentClassifier
Verifica che il sistema carichi correttamente i prompt dal database
"""
import sys
import os
import traceback

# Aggiungi il path del progetto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_prompt_integration():
    """Test dell'integrazione PromptManager con IntelligentClassifier"""
    
    print("🧪 === TEST INTEGRAZIONE PROMPT MANAGER ===")
    print()
    
    try:
        # Import del classificatore
        from Classification.intelligent_classifier import IntelligentClassifier
        print("✅ Import IntelligentClassifier completato")
        
        # Inizializzazione con client_name per testing
        classifier = IntelligentClassifier(
            client_name="humanitas",
            enable_logging=True,
            enable_cache=False  # Disabilito cache per test
        )
        print("✅ IntelligentClassifier inizializzato")
        
        # Verifica stato PromptManager
        if classifier.prompt_manager:
            print(f"✅ PromptManager attivo per tenant: {classifier.tenant_id}")
            
            # Test caricamento prompt sistema
            system_prompt = classifier._build_system_message("Test context per classificazione urgente")
            print(f"✅ System prompt caricato: {len(system_prompt)} caratteri")
            
            # Test caricamento prompt utente
            user_prompt = classifier._build_user_message(
                "Buongiorno, vorrei prenotare una visita cardiologica",
                "Paziente con richiesta urgente"
            )
            print(f"✅ User prompt caricato: {len(user_prompt)} caratteri")
            
            # Verifica che i prompt contengano variabili risolte
            if "available_labels" in system_prompt or "ETICHETTE DISPONIBILI" in system_prompt:
                print("✅ Variabili dinamiche risolte nel system prompt")
            else:
                print("⚠️ Variabili dinamiche potrebbero non essere risolte correttamente")
            
            if "ESEMPIO" in user_prompt or "examples_text" in user_prompt:
                print("✅ Esempi dinamici inclusi nel user prompt")
            else:
                print("⚠️ Esempi dinamici potrebbero non essere inclusi")
                
            print()
            print("📋 === ANTEPRIMA PROMPT SISTEMA ===")
            print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
            print()
            print("📋 === ANTEPRIMA PROMPT UTENTE ===") 
            print(user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt)
            
        else:
            print("❌ PromptManager non inizializzato - Usando fallback hardcoded")
            
            # Test fallback
            system_prompt = classifier._build_system_message("Test fallback")
            user_prompt = classifier._build_user_message("Test fallback message")
            
            print(f"🔄 Fallback system prompt: {len(system_prompt)} caratteri")
            print(f"🔄 Fallback user prompt: {len(user_prompt)} caratteri")
        
        print()
        print("✅ === TEST COMPLETATO CON SUCCESSO ===")
        return True
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        print("🔍 Traceback completo:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prompt_integration()
    sys.exit(0 if success else 1)
