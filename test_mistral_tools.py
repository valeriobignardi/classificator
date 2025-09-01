#!/usr/bin/env python3
"""
Test Mistral con function tools per replicare il comportamento del sistema
Autore: Valerio Bignardi  
Data: 2025-09-01
"""

import requests
import json

def test_mistral_with_function_tools():
    """
    Testa Mistral con function tools come fa il sistema reale
    """
    
    system_prompt = """Sei un classificatore esperto di Alleanza con profonda conoscenza del dominio sanitario.

MISSIONE: Classifica conversazioni con pazienti/utenti in base al loro intento principale.

OUTPUT FORMAT (SOLO JSON):
{"predicted_label": "etichetta_precisa", "confidence": 0.X, "motivation": "ragionamento_breve"}

CRITICAL: Genera ESCLUSIVAMENTE JSON valido. Zero testo aggiuntivo."""

    user_prompt = """Analizza questo testo:

TESTO DA CLASSIFICARE:
"[UTENTE] ho verificato tutto ma continua a non funzionare [UTENTE] quali sono le procedure in caso di allerta meteo? [UTENTE] è prevista acqua alta cosa devo fare?"

OUTPUT (SOLO JSON):"""

    # Function tool per classificazione
    tools = [
        {
            "type": "function",
            "function": {
                "name": "classify_conversation",
                "description": "Classifica una conversazione con un'etichetta appropriata",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "predicted_label": {
                            "type": "string",
                            "description": "L'etichetta di classificazione più appropriata"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Livello di confidenza da 0.0 a 1.0"
                        },
                        "motivation": {
                            "type": "string", 
                            "description": "Breve ragionamento per la scelta dell'etichetta"
                        }
                    },
                    "required": ["predicted_label", "confidence", "motivation"]
                }
            }
        }
    ]
    
    payload = {
        "model": "mistral-nemo:latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "tools": tools,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "max_tokens": 150
        }
    }
    
    print("🧪 TEST MISTRAL CON FUNCTION TOOLS")
    print("="*80)
    print(f"📋 Model: {payload['model']}")
    print(f"🛠️  Tools: {len(tools)} function tools")
    print(f"🌡️  Temperature: {payload['options']['temperature']}")
    print("="*80)
    
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ RISPOSTA RICEVUTA!")
            print("="*80)
            print("📋 RESPONSE STRUCTURE:")
            print(f"   Keys: {list(result.keys())}")
            
            message = result.get('message', {})
            print(f"   Message keys: {list(message.keys())}")
            
            # Controlla content
            content = message.get('content', '')
            print(f"📄 Content: '{content}'")
            print(f"🔥 Contiene fire: {'🔥' in content}")
            if content:
                print(f"📏 Lunghezza content: {len(content)}")
                print(f"🔤 Primi 50 char: {repr(content[:50])}")
            
            # Controlla tool_calls
            tool_calls = message.get('tool_calls', [])
            print(f"🛠️  Tool calls: {len(tool_calls)}")
            
            if tool_calls:
                print("="*80)
                print("🔧 TOOL CALLS ANALYSIS:")
                for i, tool_call in enumerate(tool_calls):
                    print(f"   Tool Call {i+1}:")
                    print(f"     Function: {tool_call.get('function', {}).get('name', 'N/A')}")
                    arguments = tool_call.get('function', {}).get('arguments', {})
                    print(f"     Arguments: {arguments}")
                    
                    if isinstance(arguments, dict):
                        predicted_label = arguments.get('predicted_label', 'N/A')
                        confidence = arguments.get('confidence', 'N/A')
                        motivation = arguments.get('motivation', 'N/A')[:50]
                        print(f"       🏷️  Label: {predicted_label}")
                        print(f"       📊 Confidence: {confidence}")
                        print(f"       💭 Motivation: {motivation}...")
            else:
                print("⚠️  Nessun tool call trovato")
                
                # Se non ci sono tool calls, analizza il content
                if content:
                    print("="*80)
                    print("🔍 ANALISI CONTENT (nessun tool call):")
                    try:
                        parsed = json.loads(content)
                        print("✅ Content è JSON valido!")
                        print(f"   🏷️  Label: {parsed.get('predicted_label', 'N/A')}")
                        print(f"   📊 Confidence: {parsed.get('confidence', 'N/A')}")
                    except json.JSONDecodeError:
                        print("❌ Content non è JSON valido")
                        print(f"Raw content: {repr(content)}")
        else:
            print(f"❌ ERRORE HTTP: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ ERRORE CONNESSIONE: {e}")
    
    print("="*80)
    print("🏁 Test completato")

if __name__ == "__main__":
    test_mistral_with_function_tools()
