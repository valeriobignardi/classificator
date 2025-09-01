#!/usr/bin/env python3
"""
Test diretto del modello Mistral per vedere cosa restituisce
Autore: Valerio Bignardi
Data: 2025-09-01
"""

import requests
import json

def test_mistral_direct():
    """
    Testa direttamente il modello Mistral con un prompt reale dal log
    """
    
    # Prompt system dal log
    system_prompt = """Sei un classificatore esperto di Alleanza con profonda conoscenza del dominio sanitario.

MISSIONE: Classifica conversazioni con pazienti/utenti in base al loro intento principale.

APPROCCIO ANALITICO:
1. Identifica l'intento principale (non dettagli secondari)
2. Considera il contesto ospedaliero (prenotazioni, referti, informazioni)
3. Distingui richieste operative da richieste informative
4. Se incerto tra 2 etichette, scegli la più specifica

CONFIDENCE GUIDELINES:
- 0.9-1.0: Intento chiarissimo e inequivocabile
- 0.7-0.8: Intento probabile con piccole ambiguità
- 0.5-0.6: Intento possibile ma con dubbi ragionevoli  
- 0.3-0.4: Molto incerto, probabilmente "altro"
- 0.0-0.2: Impossibile classificare

LABELING GUIDELINES:
la logica con cui devi creare le etichette è: <tipologia di richiesta>_<ambito a cui si riferisce> 

#LABELING EXAMPLE 
- info_prenotazione
- info_esame
- info_ricovero
- info_prericovero
- parere_clinico
- info_fertility_center 
- problema_accesso_portale
- problema_registrazione_portale
- info_ritiro_referto_cartella_clinica

CONTESTO SPECIFICO: Richiesta operativa - preferisci etichette di azione

OUTPUT FORMAT (SOLO JSON):
{"predicted_label": "etichetta_precisa", "confidence": 0.X, "motivation": "ragionamento_breve"}

CRITICAL: Genera ESCLUSIVAMENTE JSON valido. Zero testo aggiuntivo."""

    # Prompt user dal log
    user_prompt = """Analizza questo testo seguendo l'approccio degli esempi:

ESEMPIO 1 (ALTA CERTEZZA):
Input: "Devo cambiare la mail con cui mi sono registrato al portale"
Output: cambio_anagrafica
Ragionamento: Cambio delle informazioni anagrafiche richiesto

ESEMPIO 2 (ALTA CERTEZZA):
Input: "In caso di morte, come funziona l'indennizzo?"
Output: info_caso_morte
Ragionamento: Richiesta informazioni su come funziona la copertura in caso di morte

TESTO DA CLASSIFICARE:
"[UTENTE] ho verificato tutto ma continua a non funzionare [UTENTE] quali sono le procedure in caso di allerta meteo? [UTENTE] è prevista acqua alta cosa devo fare? [UTENTE] e se il lasco non fosse sufficiente? "

RAGIONA STEP-BY-STEP:
1. Identifica l'intento principale
2. Confronta con gli esempi
3. Scegli l'etichetta più appropriata
4. Valuta la tua certezza

OUTPUT (SOLO JSON):"""

    # Prepara la richiesta Ollama
    prompt_completo = f"<|system|>\n{system_prompt}\n\n<|user|>\n{user_prompt}\n\n<|assistant|>"
    
    payload = {
        "model": "mistral-nemo:latest",
        "prompt": prompt_completo,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 150
        }
    }
    
    print("🧪 TEST DIRETTO MODELLO MISTRAL")
    print("="*80)
    print(f"📋 Model: {payload['model']}")
    print(f"🌡️  Temperature: {payload['options']['temperature']}")
    print(f"🎯 Max Tokens: {payload['options']['max_tokens']}")
    print("="*80)
    print("🚀 Invio richiesta a Ollama...")
    
    try:
        # Test 1: /api/generate (quello che abbiamo appena testato)
        print("🧪 TEST 1: /api/generate (come nel test precedente)")
        print("-"*40)
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            raw_response = result.get('response', '')
            print(f"✅ Generate response: {raw_response[:100]}...")
            print(f"🔥 Contiene fire: {'🔥' in raw_response}")
        
        print("\n" + "="*80)
        
        # Test 2: /api/chat (quello che usa il sistema reale)  
        print("🧪 TEST 2: /api/chat (come nel sistema reale)")
        print("-"*40)
        
        chat_payload = {
            "model": "mistral-nemo:latest",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 150
            }
        }
        
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=chat_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ RISPOSTA RICEVUTA!")
            print("="*80)
            print("📤 RAW RESPONSE:")
            print("-"*40)
            raw_response = result.get('message', {}).get('content', '')
            print(repr(raw_response))  # Mostra caratteri nascosti
            print("-"*40)
            print("📄 PRETTY RESPONSE:")
            print(raw_response)
            print("="*80)
            
            # Analisi caratteri
            print("🔍 ANALISI CARATTERI:")
            print(f"   📏 Lunghezza: {len(raw_response)} caratteri")
            if raw_response:
                print(f"   🔤 Primi 10 char: {raw_response[:10]}")
                print(f"   🔤 Ultimi 10 char: {raw_response[-10:]}")
                print(f"   🔥 Contiene emoji fire: {'🔥' in raw_response}")
                print(f"   📊 Count emoji fire: {raw_response.count('🔥')}")
                print(f"   📄 Contiene JSON: {'{' in raw_response and '}' in raw_response}")
            
            # Test parsing JSON
            print("="*80)
            print("🧪 TEST PARSING JSON:")
            try:
                parsed = json.loads(raw_response.strip())
                print("✅ JSON VALIDO!")
                print(f"   🏷️  Predicted Label: {parsed.get('predicted_label', 'N/A')}")
                print(f"   📊 Confidence: {parsed.get('confidence', 'N/A')}")
                print(f"   💭 Motivation: {parsed.get('motivation', 'N/A')[:50]}...")
            except json.JSONDecodeError as e:
                print(f"❌ JSON INVALIDO: {e}")
                print("🔧 Tentativo parsing YAML-like...")
                
                # Test del nostro parser YAML-like
                lines = raw_response.strip().split('\n')
                parsed_yaml = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        if key == 'predicted_label':
                            parsed_yaml['predicted_label'] = value
                        elif key == 'confidence':
                            try:
                                parsed_yaml['confidence'] = float(value)
                            except:
                                parsed_yaml['confidence'] = value
                        elif key == 'motivation':
                            parsed_yaml['motivation'] = value
                
                if parsed_yaml:
                    print("✅ YAML-LIKE PARSING RIUSCITO!")
                    print(f"   🏷️  Predicted Label: {parsed_yaml.get('predicted_label', 'N/A')}")
                    print(f"   📊 Confidence: {parsed_yaml.get('confidence', 'N/A')}")
                    print(f"   💭 Motivation: {parsed_yaml.get('motivation', 'N/A')[:50]}...")
                else:
                    print("❌ Anche parsing YAML-like fallito")
            
        else:
            print(f"❌ ERRORE HTTP: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ ERRORE CONNESSIONE: {e}")
    
    print("="*80)
    print("🏁 Test completato")

if __name__ == "__main__":
    test_mistral_direct()
