#!/usr/bin/env python3
"""
============================================================================
Test Versioni Aggiornate Modelli Azure OpenAI
============================================================================

Autore: Valerio Bignardi
Data creazione: 2025-11-08

Descrizione:
    Test delle versioni più recenti dei modelli GPT-4o e GPT-5 su Azure
    
Versioni testate:
    - GPT-4o: 2024-11-20 (ultima versione disponibile)
    - GPT-5: 2025-08-07 (versione principale famiglia GPT-5)

============================================================================
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Carica variabili ambiente
load_dotenv()

# Aggiungi path per importare OpenAIService
sys.path.append(os.path.join(os.path.dirname(__file__), 'Services'))
from openai_service import OpenAIService


async def test_gpt4o_latest():
    """
    Test GPT-4o versione 2024-11-20
    
    Versione: 2024-11-20
    Context: 128K input, 16K output
    Features: Multimodal (text + images), structured outputs, function calling
    """
    print("\n" + "="*80)
    print("🤖 TEST GPT-4o (2024-11-20) - Ultima versione disponibile")
    print("="*80)
    
    service = OpenAIService()
    
    # Verifica configurazione
    print(f"\n📋 Configurazione:")
    print(f"   Deployment: {service.gpt4o_deployment}")
    print(f"   Versione: {service.gpt4o_version}")
    print(f"   Endpoint: {service.azure_endpoint}")
    print(f"   API Version: {service.azure_api_version}")
    
    # Test semplice
    messages = [
        {
            "role": "system",
            "content": "Sei un assistente utile e conciso."
        },
        {
            "role": "user",
            "content": "Salutami in italiano e dimmi quale versione di GPT-4o sei."
        }
    ]
    
    try:
        print("\n🚀 Invio richiesta a GPT-4o...")
        response = await service.chat_completion(
            model='gpt-4o',
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        
        # Estrai risposta
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            usage = response.get('usage', {})
            
            print(f"\n✅ RISPOSTA GPT-4o:")
            print(f"   {content}")
            print(f"\n📊 Token utilizzati:")
            print(f"   Prompt: {usage.get('prompt_tokens', 'N/A')}")
            print(f"   Completion: {usage.get('completion_tokens', 'N/A')}")
            print(f"   Totale: {usage.get('total_tokens', 'N/A')}")
            
            return True
        else:
            print(f"\n❌ ERRORE: Risposta non valida")
            print(f"   Response: {response}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERRORE durante chiamata GPT-4o:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        print(f"\n📋 Traceback:")
        traceback.print_exc()
        return False


async def test_gpt5_latest():
    """
    Test GPT-5 versione 2025-08-07
    
    Versione: 2025-08-07
    Context: 400K input, 128K output
    Features: Reasoning, multimodal, structured outputs, function calling
    API: Usa 'responses' endpoint (non 'chat/completions')
    """
    print("\n" + "="*80)
    print("🚀 TEST GPT-5 (2025-08-07) - Versione principale famiglia GPT-5")
    print("="*80)
    
    service = OpenAIService()
    
    # Verifica configurazione
    print(f"\n📋 Configurazione:")
    print(f"   Deployment: {service.gpt5_deployment}")
    print(f"   Versione: {service.gpt5_version}")
    print(f"   Endpoint: {service.azure_endpoint}")
    print(f"   API Version: {service.azure_api_version}")
    
    # Test semplice con input stringa
    input_text = "Ciao! Sono GPT-5. Dimmi in 2 frasi perché sei migliore di GPT-4o."
    
    try:
        print("\n🚀 Invio richiesta a GPT-5...")
        print(f"   Input: {input_text}")
        
        response = await service.responses_completion(
            model='gpt-5',
            input_data=input_text,
            max_tokens=200
        )
        
        # Estrai risposta (GPT-5 usa formato diverso)
        output_text = response.get('output_text', '')
        
        if output_text:
            print(f"\n✅ RISPOSTA GPT-5:")
            print(f"   {output_text}")
            
            # Mostra statistiche se disponibili
            if 'usage' in response:
                usage = response['usage']
                print(f"\n📊 Token utilizzati:")
                print(f"   Input: {usage.get('input_tokens', 'N/A')}")
                print(f"   Output: {usage.get('output_tokens', 'N/A')}")
                print(f"   Totale: {usage.get('total_tokens', 'N/A')}")
            
            return True
        else:
            print(f"\n❌ ERRORE: output_text vuoto")
            print(f"   Response keys: {list(response.keys())}")
            print(f"   Response completa: {response}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERRORE durante chiamata GPT-5:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        print(f"\n📋 Traceback:")
        traceback.print_exc()
        return False


async def test_function_calling_gpt4o():
    """
    Test function calling con GPT-4o
    
    Verifica che la versione 2024-11-20 supporti correttamente
    structured outputs e function calling.
    """
    print("\n" + "="*80)
    print("🔧 TEST Function Calling GPT-4o (2024-11-20)")
    print("="*80)
    
    service = OpenAIService()
    
    # Definisci una funzione tool
    get_weather_tool = service.create_function_tool(
        name="get_weather",
        description="Ottieni informazioni meteo per una città specifica",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Nome della città (es. 'Milano', 'Roma')"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Unità di misura temperatura"
                }
            },
            "required": ["city"]
        }
    )
    
    messages = [
        {
            "role": "user",
            "content": "Che tempo fa a Milano oggi?"
        }
    ]
    
    try:
        print("\n🚀 Invio richiesta con function calling...")
        response = await service.chat_completion(
            model='gpt-4o',
            messages=messages,
            tools=[get_weather_tool],
            tool_choice="auto",
            temperature=0.1,
            max_tokens=150
        )
        
        # Verifica se ha chiamato la funzione
        tool_calls = service.extract_tool_calls(response)
        
        if tool_calls:
            print(f"\n✅ GPT-4o ha chiamato la funzione:")
            for tc in tool_calls:
                print(f"   Funzione: {tc['function']['name']}")
                print(f"   Argomenti: {tc['function']['arguments']}")
            return True
        else:
            message = response['choices'][0]['message']
            print(f"\n⚠️ GPT-4o non ha chiamato funzioni:")
            print(f"   Risposta: {message.get('content', 'N/A')}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERRORE durante function calling:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """
    Esegue tutti i test
    """
    print("\n" + "="*80)
    print("🧪 TEST SUITE MODELLI AZURE OPENAI AGGIORNATI")
    print("="*80)
    print(f"\nData test: 2025-11-08")
    print(f"Modelli testati:")
    print(f"  • GPT-4o: 2024-11-20 (ultima versione)")
    print(f"  • GPT-5: 2025-08-07 (famiglia GPT-5)")
    
    results = {}
    
    # Test GPT-4o
    results['gpt4o'] = await test_gpt4o_latest()
    
    # Test GPT-5
    results['gpt5'] = await test_gpt5_latest()
    
    # Test function calling GPT-4o
    results['function_calling'] = await test_function_calling_gpt4o()
    
    # Riepilogo
    print("\n" + "="*80)
    print("📊 RIEPILOGO TEST")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name.upper()}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 TUTTI I TEST SUPERATI!")
    else:
        print("⚠️ ALCUNI TEST SONO FALLITI")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
