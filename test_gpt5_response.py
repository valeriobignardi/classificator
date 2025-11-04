#!/usr/bin/env python3
"""
Test GPT-5 Responses API Structure

Autore: Valerio Bignardi
Data creazione: 2025-11-03
Scopo: Verificare struttura risposta reale di GPT-5 Responses API
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Aggiungi directory progetto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Services.openai_service import OpenAIService


async def test_gpt5_response():
    """Test chiamata GPT-5 per verificare struttura risposta"""
    
    print("=" * 80)
    print("🧪 TEST GPT-5 RESPONSES API - STRUTTURA RISPOSTA")
    print("=" * 80)
    
    # Inizializza servizio
    service = OpenAIService(
        api_key=os.getenv("OPENAI_API_KEY"),
        max_parallel_calls=1
    )
    
    # Messaggio test semplice
    messages = [
        {"role": "system", "content": "Sei un assistente che risponde brevemente."},
        {"role": "user", "content": "Dimmi ciao"}
    ]
    
    print("\n📨 Invio richiesta GPT-5...")
    print(f"   Model: gpt-5")
    print(f"   Messages: {len(messages)}")
    
    try:
        # Chiamata senza tools (JSON-only mode)
        print("\n🔹 TEST 1: Modalità JSON-only (senza tools)")
        response1 = await service.responses_completion(
            model="gpt-5",
            input_data=messages
        )
        
        print("\n✅ Risposta GPT-5 ricevuta (JSON-only):")
        print(json.dumps(response1, indent=2, ensure_ascii=False))
        print("\n📊 Chiavi disponibili:", list(response1.keys()))
        print(f"   'output_text' presente: {'output_text' in response1}")
        if 'output_text' in response1:
            print(f"   'output_text' valore: {response1['output_text']}")
        
        # Chiamata con tools
        print("\n\n🔹 TEST 2: Modalità con tools")
        classification_tools = [{
            "type": "function",
            "function": {
                "name": "classify_test",
                "description": "Classifica un messaggio",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Etichetta classificazione"},
                        "confidence": {"type": "number", "description": "Confidenza 0-1"}
                    },
                    "required": ["label", "confidence"]
                }
            }
        }]
        
        response2 = await service.responses_completion(
            model="gpt-5",
            input_data=messages,
            text={'format': {'type': 'text'}},
            tools=classification_tools,
            tool_choice={"type": "function", "function": {"name": "classify_test"}}
        )
        
        print("\n✅ Risposta GPT-5 ricevuta (con tools):")
        print(json.dumps(response2, indent=2, ensure_ascii=False))
        print("\n📊 Chiavi disponibili:", list(response2.keys()))
        
        # Estrai tool calls
        try:
            tool_calls = service.extract_responses_tool_calls(response2)
            print(f"\n🔧 Tool calls estratti: {len(tool_calls)}")
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    print(f"\n   Tool #{i+1}:")
                    print(f"   - Function: {tc.get('function', {}).get('name')}")
                    print(f"   - Arguments: {tc.get('function', {}).get('arguments')}")
        except Exception as e:
            print(f"\n⚠️ Errore estrazione tool calls: {e}")
        
        print("\n" + "=" * 80)
        print("✅ TEST COMPLETATO CON SUCCESSO")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("❌ TEST FALLITO")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_gpt5_response())
