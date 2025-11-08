#!/usr/bin/env python3
"""
============================================================================
Test Configurazione Azure OpenAI
============================================================================

Autore: Valerio Bignardi
Data creazione: 2025-11-08
Ultima modifica: 2025-11-08

Descrizione:
    Test per verificare che la configurazione Azure OpenAI funzioni
    correttamente con le nuove modifiche implementate.

============================================================================
"""

import sys
import os
import asyncio
import json

# Aggiungi path per importi
sys.path.append('/home/ubuntu/classificatore')
sys.path.append('/home/ubuntu/classificatore/Services')

from Services.openai_service import OpenAIService


async def test_azure_openai_configuration():
    """
    Test della configurazione Azure OpenAI
    
    Data ultima modifica: 2025-11-08
    """
    print("🧪 TEST CONFIGURAZIONE AZURE OPENAI")
    print("=" * 60)
    
    try:
        # Inizializza OpenAIService (dovrebbe auto-detect Azure)
        openai_service = OpenAIService()
        
        print(f"✅ OpenAIService inizializzato")
        print(f"   🔧 Usa Azure: {openai_service.use_azure}")
        if openai_service.use_azure:
            print(f"   🌩️ Azure Endpoint: {openai_service.azure_endpoint}")
            print(f"   📅 API Version: {openai_service.azure_api_version}")
            print(f"   🤖 GPT-4o Deployment: {openai_service.gpt4o_deployment}")
            print(f"   🚀 GPT-5 Deployment: {openai_service.gpt5_deployment}")
        else:
            print(f"   🤖 Base URL: {openai_service.base_url}")
        
        # Test chiamata GPT-4o
        print(f"\n🧪 Test chiamata GPT-4o...")
        
        test_messages = [
            {"role": "user", "content": "Rispondi con una sola parola: 'test'"}
        ]
        
        response = await openai_service.chat_completion(
            model="gpt-4o",
            messages=test_messages,
            max_tokens=10,
            temperature=0.1
        )
        
        print(f"✅ Chiamata GPT-4o riuscita!")
        print(f"   📝 Risposta: {response.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')}")
        print(f"   🔢 Token utilizzati: {response.get('usage', {}).get('total_tokens', 'N/A')}")
        
        # Statistiche finali
        print(f"\n📊 STATISTICHE CHIAMATE:")
        print(f"   ✅ Successi: {openai_service.stats.successful_calls}")
        print(f"   ❌ Errori: {openai_service.stats.failed_calls}")
        print(f"   🔢 Token totali: {openai_service.stats.total_tokens_used}")
        print(f"   ⏱️  Latenza media: {openai_service.stats.average_latency:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRORE: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_variables():
    """
    Verifica le variabili d'ambiente Azure
    
    Data ultima modifica: 2025-11-08
    """
    print("🔍 VERIFICA VARIABILI D'AMBIENTE")
    print("=" * 60)
    
    env_vars = {
        'AZURE_OPENAI_API_KEY': os.getenv('AZURE_OPENAI_API_KEY'),
        'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'AZURE_OPENAI_API_VERSION': os.getenv('AZURE_OPENAI_API_VERSION'),
        'AZURE_OPENAI_GPT4O_DEPLOYMENT': os.getenv('AZURE_OPENAI_GPT4O_DEPLOYMENT'),
        'AZURE_OPENAI_GPT5_DEPLOYMENT': os.getenv('AZURE_OPENAI_GPT5_DEPLOYMENT'),
    }
    
    all_configured = True
    for var_name, var_value in env_vars.items():
        status = "✅" if var_value else "❌"
        masked_value = "***" + str(var_value)[-4:] if var_value and len(str(var_value)) > 4 else str(var_value)
        print(f"   {status} {var_name}: {masked_value}")
        
        if not var_value:
            all_configured = False
    
    print(f"\n🎯 Configurazione completa: {'✅ SÌ' if all_configured else '❌ NO'}")
    return all_configured


def main():
    """
    Esegue tutti i test per Azure OpenAI
    
    Data ultima modifica: 2025-11-08
    """
    print("🔬 TEST COMPLETO AZURE OPENAI CONFIGURATION")
    print("=" * 80)
    
    # Test 1: Variabili d'ambiente
    env_ok = test_environment_variables()
    
    if not env_ok:
        print("⚠️ Alcune variabili d'ambiente mancano. Il test potrebbe fallire.")
    
    # Test 2: Configurazione e chiamata
    print("\n")
    test_result = asyncio.run(test_azure_openai_configuration())
    
    # Risultato finale
    print("\n" + "=" * 80)
    if test_result:
        print("🎉 TUTTI I TEST AZURE OPENAI RIUSCITI!")
        print("✅ Il sistema è configurato correttamente per Azure OpenAI")
    else:
        print("🚨 TEST AZURE OPENAI FALLITI")
        print("❌ Verificare configurazione e variabili d'ambiente")
        
    return test_result


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)