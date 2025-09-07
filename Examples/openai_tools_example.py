#!/usr/bin/env python3
"""
============================================================================
Esempio utilizzo OpenAI Tools/Function Calling
============================================================================

Autore: Valerio Bignardi
Data creazione: 2025-09-07
Ultima modifica: 2025-09-07

Descrizione:
    Esempio dimostrativo di come utilizzare il nuovo supporto per OpenAI
    tools e function calling nel servizio OpenAI aggiornato.

============================================================================
"""

import sys
import os
import asyncio
import json
from typing import Dict, Any, List

# Aggiungi la directory padre al path per importare i moduli
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Services.openai_service import OpenAIService


def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """
    Funzione di esempio per ottenere il meteo corrente
    
    Args:
        location: La città di cui ottenere il meteo
        unit: Unità di misura (celsius o fahrenheit)
        
    Returns:
        Dizionario con informazioni meteo simulate
    """
    # Simula dati meteo (in un caso reale chiameresti una API meteo)
    weather_data = {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "soleggiato",
        "humidity": 65,
        "wind_speed": 10
    }
    
    print(f"🌤️ Meteo richiesto per {location}: {weather_data}")
    return weather_data


def calculate_sum(a: float, b: float) -> Dict[str, Any]:
    """
    Funzione di esempio per calcolare la somma
    
    Args:
        a: Primo numero
        b: Secondo numero
        
    Returns:
        Dizionario con il risultato
    """
    result = a + b
    print(f"🧮 Calcolo somma: {a} + {b} = {result}")
    return {"result": result}


async def demo_openai_tools():
    """
    Dimostra l'utilizzo di OpenAI tools/function calling
    """
    print("🚀 Demo OpenAI Tools/Function Calling")
    print("=" * 50)
    
    # 1. Inizializza il servizio OpenAI
    service = OpenAIService(
        max_parallel_calls=10,
        rate_limit_per_minute=1000
    )
    
    # 2. Definisci le funzioni disponibili
    available_functions = {
        "get_current_weather": get_current_weather,
        "calculate_sum": calculate_sum
    }
    
    # 3. Crea i tools usando il metodo helper
    weather_tool = service.create_function_tool(
        name="get_current_weather",
        description="Ottieni le informazioni meteo correnti per una località",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "La città di cui ottenere il meteo, es. 'Roma, Italia'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Unità di misura per la temperatura"
                }
            },
            "required": ["location"]
        }
    )
    
    calculator_tool = service.create_function_tool(
        name="calculate_sum",
        description="Calcola la somma di due numeri",
        parameters={
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "Primo numero da sommare"
                },
                "b": {
                    "type": "number",
                    "description": "Secondo numero da sommare"
                }
            },
            "required": ["a", "b"]
        }
    )
    
    tools = [weather_tool, calculator_tool]
    
    # 4. Valida i tools
    for tool in tools:
        if service.validate_tool_schema(tool):
            print(f"✅ Tool '{tool['function']['name']}' validato correttamente")
        else:
            print(f"❌ Tool '{tool['function']['name']}' non valido!")
            return
    
    # 5. Esempio di conversazione con function calling
    messages = [
        {
            "role": "user",
            "content": "Qual è il meteo a Milano oggi? E poi calcola la somma di 15 e 27."
        }
    ]
    
    print("\n🗣️ Invio messaggio utente...")
    print(f"Utente: {messages[0]['content']}")
    
    try:
        # Prima chiamata - l'AI decide quali tools usare
        response = await service.chat_completion(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1
        )
        
        print("\n🤖 Risposta AI ricevuta")
        
        # Estrai tool calls se presenti
        tool_calls = service.extract_tool_calls(response)
        
        if tool_calls:
            print(f"🔧 AI ha richiesto {len(tool_calls)} tool calls:")
            
            # Aggiungi il messaggio dell'assistant alla conversazione
            assistant_message = response["choices"][0]["message"]
            messages.append(assistant_message)
            
            # Esegui i tool calls
            tool_messages = service.execute_tool_calls(tool_calls, available_functions)
            
            # Aggiungi le risposte dei tools alla conversazione
            messages.extend(tool_messages)
            
            print("\n🔄 Invio risposte tools all'AI per risposta finale...")
            
            # Seconda chiamata per ottenere la risposta finale
            final_response = await service.chat_completion(
                model="gpt-4o",
                messages=messages,
                tools=tools,  # Mantieni tools disponibili
                temperature=0.1
            )
            
            final_message = final_response["choices"][0]["message"]["content"]
            print(f"\n🎯 Risposta finale AI:")
            print(f"Assistant: {final_message}")
            
        else:
            # Nessun tool call, risposta diretta
            content = response["choices"][0]["message"]["content"]
            print(f"Assistant: {content}")
    
    except Exception as e:
        print(f"❌ Errore durante la demo: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n📊 Statistiche servizio:")
    stats = service.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")


def demo_tool_creation():
    """
    Dimostra la creazione e validazione di tools
    """
    print("\n🛠️ Demo creazione e validazione tools")
    print("=" * 40)
    
    service = OpenAIService()
    
    # Esempio di tool valido
    valid_tool = service.create_function_tool(
        name="search_database",
        description="Cerca informazioni nel database",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query di ricerca"
                },
                "limit": {
                    "type": "integer",
                    "description": "Numero massimo di risultati"
                }
            },
            "required": ["query"]
        }
    )
    
    print("✅ Tool valido creato:")
    print(json.dumps(valid_tool, indent=2, ensure_ascii=False))
    print(f"Validazione: {service.validate_tool_schema(valid_tool)}")
    
    # Esempio di tool non valido
    invalid_tool = {
        "type": "function",
        "function": {
            "name": "test_function"
            # Manca description e parameters
        }
    }
    
    print("\n❌ Tool non valido:")
    print(json.dumps(invalid_tool, indent=2))
    print(f"Validazione: {service.validate_tool_schema(invalid_tool)}")


if __name__ == "__main__":
    print("🎮 Avvio demo OpenAI Tools")
    
    # Demo validazione tools
    demo_tool_creation()
    
    # Demo function calling (richiede API key OpenAI valida)
    if os.getenv('OPENAI_API_KEY'):
        print("\n🔑 API Key trovata, avvio demo function calling...")
        asyncio.run(demo_openai_tools())
    else:
        print("\n⚠️ OPENAI_API_KEY non trovata - saltando demo function calling")
        print("   Per testare function calling, imposta la variabile d'ambiente OPENAI_API_KEY")
    
    print("\n✅ Demo completata!")
