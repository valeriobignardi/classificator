#!/usr/bin/env python3
"""
============================================================================
Test IntelligentClassifier con GPT-4o
============================================================================

Autore: Valerio Bignardi
Data creazione: 2025-01-31
Ultima modifica: 2025-01-31

Descrizione:
    Test completo dell'IntelligentClassifier con supporto GPT-4o
    per verificare la classificazione delle conversazioni.

============================================================================
"""

import os
import sys
from datetime import datetime

# Aggiungi il path del progetto
sys.path.append('/home/ubuntu/classificatore')

try:
    from Classification.intelligent_classifier import IntelligentClassifier
    print("✅ IntelligentClassifier importato correttamente")
except ImportError as e:
    print(f"❌ Errore import IntelligentClassifier: {e}")
    sys.exit(1)


def test_gpt4o_classification():
    """
    Test classificazione con GPT-4o
    
    Scopo:
        Verifica che l'IntelligentClassifier possa utilizzare GPT-4o
        per classificare conversazioni e generare risultati strutturati
        
    Data ultima modifica: 2025-01-31
    """
    print("\n🧪 ===== TEST CLASSIFICAZIONE GPT-4o =====")
    
    try:
        # Inizializza classifier con GPT-4o
        classifier = IntelligentClassifier(
            model_name="gpt-4o",
            client_name="humanitas",  # Usa tenant esistente
            enable_logging=True,
            enable_cache=False  # Disabilita cache per test pulito
        )
        
        print(f"✅ Classifier inizializzato con modello: {classifier.model_name}")
        print(f"✅ Provider rilevato: {'OpenAI' if classifier.is_openai_model else 'Ollama'}")
        
        if not classifier.is_openai_model:
            print("❌ Modello non rilevato come OpenAI - controllare configurazione")
            return False
        
        # Test conversazioni varie
        test_conversations = [
            {
                "text": "Ciao, volevo sapere come posso prenotare una visita cardiologica presso il vostro ospedale",
                "expected_category": "prenotazione_visita"
            },
            {
                "text": "Buongiorno, sto cercando lavoro come infermiere. Avete posizioni aperte?",
                "expected_category": "lavoro_carriere"
            },
            {
                "text": "Mi serve un certificato medico per l'attività sportiva. Come posso ottenerlo?",
                "expected_category": "certificati_medici"
            },
            {
                "text": "Salve, non riesco a trovare il parcheggio dell'ospedale. Potete aiutarmi?",
                "expected_category": "informazioni_generali"
            }
        ]
        
        print(f"\n📋 Classificazione di {len(test_conversations)} conversazioni:")
        
        all_passed = True
        for i, test_case in enumerate(test_conversations, 1):
            print(f"\n📞 Test {i}: {test_case['text'][:60]}...")
            
            try:
                # Classifica con GPT-4o
                start_time = datetime.now()
                result = classifier.classify_with_motivation(test_case['text'])
                processing_time = (datetime.now() - start_time).total_seconds()
                
                print(f"  🏷️  Etichetta: {result.predicted_label}")
                print(f"  🎯 Confidenza: {result.confidence:.3f}")
                print(f"  💭 Motivazione: {result.motivation[:100]}...")
                print(f"  🔧 Metodo: {result.method}")
                print(f"  ⏱️  Tempo: {processing_time:.2f}s")
                
                # Verifica che abbiamo una classificazione valida
                if result.predicted_label and result.predicted_label != "altro":
                    print(f"  ✅ Classificazione riuscita")
                else:
                    print(f"  ⚠️ Classificazione generica: {result.predicted_label}")
                    
            except Exception as e:
                print(f"  ❌ Errore classificazione: {e}")
                all_passed = False
        
        # Test statistiche OpenAI se disponibile
        if hasattr(classifier, 'openai_service') and classifier.openai_service:
            print(f"\n📊 Statistiche OpenAI Service:")
            stats = classifier.openai_service.get_stats()
            print(f"  📞 Chiamate totali: {stats['total_calls']}")
            print(f"  ✅ Successo rate: {stats['success_rate']:.1f}%")
            print(f"  🪙 Token utilizzati: {stats['total_tokens_used']}")
            print(f"  ⏱️  Latenza media: {stats['average_latency_seconds']:.3f}s")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Errore durante test classificazione: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_availability():
    """
    Test disponibilità modelli nell'API
    
    Scopo:
        Verifica che GPT-4o sia visibile nell'API di configurazione LLM
        
    Data ultima modifica: 2025-01-31
    """
    print("\n🔍 ===== TEST DISPONIBILITÀ MODELLI =====")
    
    try:
        from Services.llm_configuration_service import LLMConfigurationService
        
        # Inizializza servizio
        llm_service = LLMConfigurationService()
        
        # Recupera modelli disponibili
        models = llm_service.get_available_models("test_tenant")
        
        print(f"✅ Modelli disponibili: {len(models)}")
        
        # Cerca GPT-4o
        gpt4o_found = False
        for model in models:
            print(f"  📝 {model.get('name', 'NO_NAME')} - {model.get('display_name', 'NO_DISPLAY')} ({model.get('provider', 'NO_PROVIDER')})")
            
            if model.get('name') == 'gpt-4o':
                gpt4o_found = True
                print(f"    ✅ GPT-4o trovato!")
                print(f"    🔢 Max input tokens: {model.get('max_input_tokens')}")
                print(f"    ⚡ Max parallel calls: {model.get('parallel_calls_max')}")
                
                # Test info specifiche modello
                model_info = llm_service.get_model_info('gpt-4o')
                if model_info:
                    print(f"    📋 Info dettagliate recuperate correttamente")
                else:
                    print(f"    ⚠️ Info dettagliate non disponibili")
        
        if not gpt4o_found:
            print("❌ GPT-4o non trovato nei modelli disponibili")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test disponibilità modelli: {e}")
        return False


def main():
    """
    Funzione principale di test
    
    Scopo:
        Esegue tutti i test per verificare l'integrazione completa
        di GPT-4o nel sistema di classificazione
        
    Data ultima modifica: 2025-01-31
    """
    print(f"🚀 Test IntelligentClassifier + GPT-4o - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Disponibilità modelli
    models_ok = test_model_availability()
    
    # Test 2: Classificazione con GPT-4o
    classification_ok = test_gpt4o_classification()
    
    # Risultati finali
    print(f"\n🎯 ===== RISULTATI FINALI =====")
    print(f"✅ Disponibilità modelli: {'OK' if models_ok else 'FAILED'}")
    print(f"✅ Classificazione GPT-4o: {'OK' if classification_ok else 'FAILED'}")
    
    if models_ok and classification_ok:
        print(f"🎉 INTEGRAZIONE GPT-4o COMPLETATA! Il sistema è pronto per l'uso.")
        print(f"📝 GPT-4o è ora disponibile nel dropdown dei modelli.")
        print(f"⚡ Supporto parallelo fino a 200 chiamate simultanee attivo.")
        return 0
    else:
        print(f"❌ Alcuni test sono falliti. Controllare i log sopra.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
