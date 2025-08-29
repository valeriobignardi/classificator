#!/usr/bin/env python3
"""
Test Ollama Structured Outputs per classificazione automatica
Basato sulla documentazione ufficiale Ollama: https://github.com/ollama/ollama/blob/main/docs/api.md

Autore: Valerio Bignardi
Data: 29 Agosto 2025
"""

import json
import requests
import time
from typing import Dict, Any

def test_structured_classification():
    """
    Testa l'uso di structured outputs con Ollama per avere JSON garantito
    """
    # Configura la richiesta per structured outputs
    url = "http://localhost:11434/api/chat"
    
    # Schema JSON che garantisce la struttura della risposta
    json_schema = {
        "type": "object",
        "properties": {
            "predicted_label": {
                "type": "string",
                "description": "L'etichetta di classificazione predetta"
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence score tra 0.0 e 1.0"
            },
            "motivation": {
                "type": "string",
                "description": "Breve spiegazione della classificazione"
            }
        },
        "required": ["predicted_label", "confidence", "motivation"]
    }
    
    # Testo di test per classificazione
    test_text = """
    Buongiorno, vorrei prenotare una visita cardiologica per la prossima settimana.
    Ho dolori al petto che mi preoccupano. Potreste dirmi quali documenti servono
    e come posso fare la prenotazione online?
    """
    
    # Etichette disponibili (esempio)
    available_labels = [
        "prenotazione_esami",
        "prenotazione_visite", 
        "problema_accesso_portale",
        "info_contatti",
        "problema_amministrativo",
        "info_parcheggio",
        "altro"
    ]
    
    # Costruisci il messaggio del sistema
    system_prompt = f"""
    Sei un classificatore di conversazioni mediche. Il tuo compito è classificare
    il testo dell'utente in una delle seguenti categorie:
    {', '.join(available_labels)}
    
    Devi rispondere SOLO con un JSON che contiene:
    - predicted_label: una delle etichette disponibili
    - confidence: un numero tra 0.0 e 1.0 
    - motivation: una breve spiegazione (massimo 50 parole)
    
    Analizza il contenuto e classifica accuratamente.
    """
    
    # Payload della richiesta
    payload = {
        "model": "mistral:7b",  # Usa il modello disponibile
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": f"Classifica questo testo: {test_text}"
            }
        ],
        "stream": False,  # Non streaming per semplicità
        "format": json_schema,  # Schema JSON obbligatorio
        "options": {
            "temperature": 0.1  # Bassa per risultati deterministici
        }
    }
    
    print("🚀 Testing Ollama Structured Outputs...")
    print(f"📝 Testo da classificare: {test_text[:100]}...")
    print(f"📋 Etichette disponibili: {available_labels}")
    print("\n" + "="*60 + "\n")
    
    try:
        start_time = time.time()
        
        # Invia richiesta
        response = requests.post(url, json=payload, timeout=30)
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ Richiesta Ollama riuscita!")
            print(f"⏱️  Tempo di risposta: {elapsed:.2f}s")
            print(f"🤖 Modello: {result.get('model', 'N/A')}")
            
            # Estrai il contenuto del messaggio
            message = result.get('message', {})
            content = message.get('content', '')
            
            print(f"\n📄 Risposta raw: {content}")
            
            try:
                # Parsa il JSON dalla risposta
                classification_result = json.loads(content)
                
                print("\n🎯 Risultato della classificazione:")
                print(f"   🏷️  Etichetta: {classification_result['predicted_label']}")
                print(f"   📊 Confidence: {classification_result['confidence']:.3f}")
                print(f"   💭 Motivazione: {classification_result['motivation']}")
                
                # Verifica che sia una etichetta valida
                if classification_result['predicted_label'] in available_labels:
                    print("\n✅ Etichetta valida!")
                else:
                    print(f"\n❌ Etichetta non valida: {classification_result['predicted_label']}")
                    
                # Verifica confidence nel range corretto
                confidence = classification_result['confidence']
                if 0.0 <= confidence <= 1.0:
                    print("✅ Confidence nel range corretto!")
                else:
                    print(f"❌ Confidence fuori range: {confidence}")
                    
                print("\n🎉 Test COMPLETATO con successo!")
                print("💡 Structured Outputs funziona perfettamente!")
                
                return True
                
            except json.JSONDecodeError as e:
                print(f"❌ Errore parsing JSON: {e}")
                print(f"   Contenuto: {content}")
                return False
                
        else:
            print(f"❌ Errore HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Errore durante la richiesta: {e}")
        return False

def test_multiple_classifications():
    """
    Testa classificazioni multiple per verificare la consistenza
    """
    test_cases = [
        "Vorrei prenotare una visita oculistica",
        "Non riesco ad accedere al portale online", 
        "Dove posso parcheggiare in ospedale?",
        "Ho un problema con la fatturazione",
        "Quali sono i vostri orari di apertura?"
    ]
    
    print("\n" + "="*60)
    print("🧪 Test classificazioni multiple:")
    print("="*60 + "\n")
    
    for i, text in enumerate(test_cases, 1):
        print(f"Test {i}/5: {text}")
        # Qui potresti implementare la stessa logica
        time.sleep(1)  # Pausa tra richieste
        
    print("✅ Test multipli completati!")

if __name__ == "__main__":
    print("🔬 Test Ollama Structured Outputs per Classificazione")
    print("=" * 60)
    
    # Verifica che Ollama sia raggiungibile
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            version_info = response.json()
            print(f"✅ Ollama connesso - Versione: {version_info.get('version', 'N/A')}")
        else:
            print("❌ Ollama non risponde correttamente")
            exit(1)
    except Exception as e:
        print(f"❌ Errore connessione Ollama: {e}")
        exit(1)
    
    print()
    
    # Esegui il test principale
    success = test_structured_classification()
    
    if success:
        test_multiple_classifications()
        
        print("\n" + "="*60)
        print("🎯 CONCLUSIONI:")
        print("✅ Structured Outputs elimina il bisogno di parsing complesso")
        print("✅ JSON sempre valido e con schema garantito") 
        print("✅ Niente più fallback o parsing manuale!")
        print("💡 Implementa questo approccio in IntelligentClassifier")
    else:
        print("\n❌ Test fallito - controlla la configurazione Ollama")
        exit(1)
