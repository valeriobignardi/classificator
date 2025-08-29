#!/usr/bin/env python3
"""
Test script per verificare il supporto delle function tools in Ollama

Documentazione: https://ollama.ai/blog/tool-support
API Chat: https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-tools

Autore: Valerio Bignardi
Data: 29 Agosto 2025
"""

import json
import requests
import sys

def test_ollama_function_tools():
    """
    Testa le function tools di Ollama per classificazione strutturata
    """
    
    # Configurazione
    ollama_url = "http://localhost:11434"
    model_name = "mistral:7b"
    
    # Definizione della function tool per classificazione
    classification_tool = {
        "type": "function",
        "function": {
            "name": "classify_conversation",
            "description": "Classifica una conversazione in una categoria specifica con confidence e motivazione",
            "parameters": {
                "type": "object",
                "properties": {
                    "predicted_label": {
                        "type": "string",
                        "description": "Etichetta di classificazione precisa tra: altro, info_contatti, info_esami, info_orari, info_parcheggio, prenotazione_esami, problema_accesso_portale, problema_amministrativo, problema_prenotazione_portale, ritiro_cartella_clinica_referti"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Livello di confidenza della classificazione (0.0-1.0)"
                    },
                    "motivation": {
                        "type": "string",
                        "description": "Breve spiegazione del ragionamento per questa classificazione"
                    }
                },
                "required": ["predicted_label", "confidence", "motivation"]
            }
        }
    }
    
    # Test conversation
    conversation_text = "Ciao, ho bisogno di prenotare una risonanza magnetica per domani mattina. Come posso fare?"
    
    # Payload per API chat con function tools
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": """Sei un classificatore esperto per conversazioni di un ospedale. 
Classifica la conversazione dell'utente utilizzando la funzione classify_conversation.
Le categorie disponibili sono:
- altro: conversazioni generiche o non classificabili
- info_contatti: richieste di informazioni su contatti, telefoni, email
- info_esami: informazioni su tipi di esami, preparazione, risultati
- info_orari: orari di apertura, orari visite, disponibilità
- info_parcheggio: informazioni su parcheggi, accesso veicoli
- prenotazione_esami: richieste di prenotazione esami diagnostici
- problema_accesso_portale: problemi di login o accesso al portale online
- problema_amministrativo: problemi burocratici, documenti, pagamenti
- problema_prenotazione_portale: problemi tecnici durante prenotazioni online
- ritiro_cartella_clinica_referti: ritiro documenti medici, referti, cartelle

Analizza attentamente il contenuto e fornisci una classificazione precisa."""
            },
            {
                "role": "user", 
                "content": f"Classifica questa conversazione: {conversation_text}"
            }
        ],
        "tools": [classification_tool],
        "tool_choice": "required"  # Forza l'uso della function tool
    }
    
    try:
        print(f"🔍 Testing Ollama function tools con {model_name}...")
        print(f"📝 Conversazione test: {conversation_text}")
        print(f"🔗 URL Ollama: {ollama_url}/api/chat")
        
        response = requests.post(
            f"{ollama_url}/api/chat",
            json=payload,
            timeout=30
        )
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📄 Raw Response:\n{response.text[:500]}...")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"✅ Risposta ricevuta!")
                print(f"📋 Risultato completo:\n{json.dumps(result, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"❌ Errore parsing JSON response: {e}")
                print(f"📄 Contenuto raw:\n{response.text}")
                return False, None
            
            # Estrai la function call
            message = result.get('message', {})
            tool_calls = message.get('tool_calls', [])
            
            if tool_calls:
                tool_call = tool_calls[0]
                function_result = tool_call.get('function', {})
                arguments_str = function_result.get('arguments', '{}')
                
                try:
                    classification = json.loads(arguments_str)
                    print(f"\n🎯 CLASSIFICAZIONE ESTRATTA:")
                    print(f"   Etichetta: {classification.get('predicted_label')}")
                    print(f"   Confidence: {classification.get('confidence')}")
                    print(f"   Motivazione: {classification.get('motivation')}")
                    
                    return True, classification
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Errore parsing arguments: {e}")
                    return False, None
            else:
                print("❌ Nessuna tool call nella risposta")
                return False, None
                
        else:
            print(f"❌ Errore API: {response.status_code}")
            print(f"📄 Dettagli: {response.text}")
            return False, None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore connessione: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Errore generico: {e}")
        return False, None

if __name__ == "__main__":
    success, result = test_ollama_function_tools()
    
    if success:
        print(f"\n🎉 TEST SUPERATO!")
        print(f"   Le function tools funzionano correttamente")
        print(f"   Possiamo implementarle nell'IntelligentClassifier")
    else:
        print(f"\n💡 FALLBACK NECESSARIO")
        print(f"   Dobbiamo continuare con il parsing manuale JSON")
    
    sys.exit(0 if success else 1)
