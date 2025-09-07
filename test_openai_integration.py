#!/usr/bin/env python3
"""
============================================================================
Test OpenAI Service - Verifica funzionamento GPT-4o
============================================================================

Autore: Valerio Bignardi
Data creazione: 2025-01-31
Ultima modifica: 2025-01-31

Descrizione:
    Script di test per verificare il corretto funzionamento del servizio
    OpenAI con modello GPT-4o e supporto parallelismo.

============================================================================
"""

import os
import sys
import asyncio
import json
from datetime import datetime

# Aggiungi il path del progetto
sys.path.append('/home/ubuntu/classificatore')

# Import del servizio OpenAI
try:
    from Services.openai_service import OpenAIService, sync_chat_completion
    print("✅ OpenAI Service importato correttamente")
except ImportError as e:
    print(f"❌ Errore import OpenAI Service: {e}")
    sys.exit(1)

# Verifica che la chiave API sia disponibile
if not os.getenv('OPENAI_API_KEY'):
    print("❌ OPENAI_API_KEY non trovata in environment")
    sys.exit(1)
else:
    api_key_preview = os.getenv('OPENAI_API_KEY')[:20] + "..."
    print(f"✅ OPENAI_API_KEY trovata: {api_key_preview}")


def test_openai_service():
    """
    Test completo del servizio OpenAI
    
    Scopo:
        Testa le funzionalità principali del servizio OpenAI incluso
        il parallelismo e la gestione degli errori
        
    Data ultima modifica: 2025-01-31
    """
    print("\n🧪 ===== TEST OPENAI SERVICE =====")
    
    try:
        # Inizializza servizio
        service = OpenAIService(
            max_parallel_calls=5,  # Limitato per test
            rate_limit_per_minute=100
        )
        print("✅ Servizio OpenAI inizializzato")
        
        # Test 1: Chiamata singola
        print("\n📞 Test 1: Chiamata singola GPT-4o")
        messages = [
            {"role": "system", "content": "Sei un assistente che risponde brevemente in italiano."},
            {"role": "user", "content": "Dimmi solo 'OK' se funzioni correttamente."}
        ]
        
        response = sync_chat_completion(
            service,
            model="gpt-4o",
            messages=messages,
            max_tokens=10,
            temperature=0.1
        )
        
        if response and 'choices' in response:
            reply = response['choices'][0]['message']['content'].strip()
            print(f"✅ Risposta GPT-4o: '{reply}'")
            print(f"📊 Token utilizzati: {response.get('usage', {}).get('total_tokens', 'N/A')}")
        else:
            print(f"❌ Risposta malformata: {response}")
            return False
        
        # Test 2: Statistiche
        print("\n📊 Test 2: Statistiche servizio")
        stats = service.get_stats()
        print(f"✅ Chiamate totali: {stats['total_calls']}")
        print(f"✅ Chiamate riuscite: {stats['successful_calls']}")
        print(f"✅ Latenza media: {stats['average_latency_seconds']}s")
        print(f"✅ Cache entries: {stats['cache_size']}")
        
        # Test 3: Chiamate multiple (parallelismo limitato)
        print("\n🔄 Test 3: Chiamate parallele (3 chiamate)")
        
        requests = [
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "Rispondi sempre con un solo numero."},
                    {"role": "user", "content": f"Quanto fa 2+{i}?"}
                ],
                "max_tokens": 5,
                "temperature": 0.0
            }
            for i in range(1, 4)  # 3 richieste: 2+1, 2+2, 2+3
        ]
        
        # Esegui chiamate parallele
        async def test_parallel():
            return await service.batch_chat_completions(requests, max_concurrent=3)
        
        parallel_results = asyncio.run(test_parallel())
        
        print(f"✅ Risultati paralleli ricevuti: {len(parallel_results)}")
        for i, result in enumerate(parallel_results):
            if result.get('success'):
                reply = result['choices'][0]['message']['content'].strip()
                print(f"  📞 Chiamata {i+1}: '{reply}'")
            else:
                print(f"  ❌ Chiamata {i+1} fallita: {result.get('error', 'Errore sconosciuto')}")
        
        # Statistiche finali
        print("\n📊 Statistiche finali:")
        final_stats = service.get_stats()
        print(f"✅ Chiamate totali: {final_stats['total_calls']}")
        print(f"✅ Successo rate: {final_stats['success_rate']:.1f}%")
        print(f"✅ Token totali utilizzati: {final_stats['total_tokens_used']}")
        print(f"✅ Max parallelismo raggiunto: {final_stats['max_parallel_reached']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore durante test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_integration():
    """
    Test integrazione con configurazione YAML
    
    Scopo:
        Verifica che il servizio OpenAI possa essere inizializzato
        correttamente dalla configurazione del progetto
        
    Data ultima modifica: 2025-01-31
    """
    print("\n🔧 ===== TEST INTEGRAZIONE CONFIG =====")
    
    try:
        import yaml
        
        # Carica configurazione
        with open('/home/ubuntu/classificatore/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Verifica presenza configurazione OpenAI
        llm_config = config.get('llm', {})
        openai_config = llm_config.get('openai', {})
        models = llm_config.get('models', {}).get('available', [])
        
        print(f"✅ Config LLM caricata")
        print(f"✅ OpenAI config: {openai_config}")
        
        # Cerca modello GPT-4o
        gpt4o_model = None
        for model in models:
            if isinstance(model, dict) and model.get('name') == 'gpt-4o':
                gpt4o_model = model
                break
        
        if gpt4o_model:
            print(f"✅ Modello GPT-4o trovato in config:")
            print(f"  📝 Display name: {gpt4o_model.get('display_name')}")
            print(f"  🏭 Provider: {gpt4o_model.get('provider')}")
            print(f"  🔢 Max input tokens: {gpt4o_model.get('max_input_tokens')}")
            print(f"  ⚡ Max parallel calls: {gpt4o_model.get('parallel_calls_max')}")
            return True
        else:
            print("❌ Modello GPT-4o non trovato in configurazione")
            return False
            
    except Exception as e:
        print(f"❌ Errore test configurazione: {e}")
        return False


def main():
    """
    Funzione principale di test
    
    Scopo:
        Esegue tutti i test del servizio OpenAI e verifica l'integrazione
        con la configurazione del progetto
        
    Data ultima modifica: 2025-01-31
    """
    print(f"🚀 Avvio test OpenAI Service - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Configurazione
    config_ok = test_config_integration()
    
    # Test 2: Servizio OpenAI
    service_ok = test_openai_service()
    
    # Risultati finali
    print(f"\n🎯 ===== RISULTATI FINALI =====")
    print(f"✅ Config integration: {'OK' if config_ok else 'FAILED'}")
    print(f"✅ OpenAI service: {'OK' if service_ok else 'FAILED'}")
    
    if config_ok and service_ok:
        print(f"🎉 TUTTI I TEST SUPERATI! GPT-4o è pronto per l'uso.")
        return 0
    else:
        print(f"❌ Alcuni test sono falliti. Controllare i log sopra.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
