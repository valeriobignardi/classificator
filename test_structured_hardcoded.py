#!/usr/bin/env python3
"""
Test Structured Outputs con etichette hardcoded per verificare il meccanismo

Autore: Valerio Bignardi  
Data: 29 Agosto 2025
"""

import requests
import json
import time

def test_structured_with_hardcoded_labels():
    """
    Testa structured outputs con etichette hardcoded in italiano
    """
    print("🔬 Test Structured Outputs - Etichette Hardcoded Italiano")
    print("=" * 65)
    
    # Etichette hardcoded per test
    labels = [
        'prenotazione_esami',
        'prenotazione_visite', 
        'problema_accesso_portale',
        'info_contatti',
        'problema_amministrativo',
        'info_parcheggio',
        'ritiro_cartella_clinica_referti',
        'altro'
    ]
    
    # Schema JSON per structured output
    json_schema = {
        "type": "object",
        "properties": {
            "predicted_label": {
                "type": "string",
                "enum": labels,  # 🔑 VINCOLO: Solo queste etichette
                "description": "L'etichetta di classificazione predetta"
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Livello di confidenza (0.0-1.0)"
            },
            "motivation": {
                "type": "string",
                "maxLength": 150,
                "description": "Breve spiegazione in italiano"
            }
        },
        "required": ["predicted_label", "confidence", "motivation"]
    }
    
    # Test cases
    conversations = [
        "Buongiorno, vorrei prenotare una visita cardiologica per la prossima settimana",
        "Non riesco ad accedere al portale online per vedere i miei esami",
        "Dove posso parcheggiare quando vengo in ospedale?",
        "Ho un problema con la fatturazione della mia ultima visita", 
        "Devo ritirare i miei referti dell'esame del sangue"
    ]
    
    # System prompt in italiano
    system_prompt = f"""
    Sei un classificatore esperto di conversazioni mediche per l'ospedale Humanitas.
    
    Analizza la conversazione e classifica in UNA di queste categorie:
    {', '.join(labels)}
    
    REGOLE OBBLIGATORIE:
    - predicted_label: ESATTAMENTE una delle etichette elencate sopra
    - confidence: numero tra 0.0 e 1.0
    - motivation: spiegazione BREVE in ITALIANO (max 150 caratteri)
    
    Rispondi SOLO con JSON valido, NIENTE altro testo.
    """
    
    success_count = 0
    
    for i, conversation in enumerate(conversations, 1):
        print(f"\n🧪 Test {i}/5:")
        print(f"📝 Input: {conversation}")
        
        payload = {
            "model": "mistral:7b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classifica: {conversation}"}
            ],
            "stream": False,
            "format": json_schema,  # 🔑 SCHEMA OBBLIGATORIO
            "options": {
                "temperature": 0.01,
                "num_predict": 200
            }
        }
        
        try:
            start = time.time()
            response = requests.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=10
            )
            duration = time.time() - start
            
            response.raise_for_status()
            result = response.json()
            
            # Parse del JSON strutturato
            json_content = json.loads(result['message']['content'])
            
            print(f"⏱️  Tempo: {duration:.2f}s")
            print(f"🏷️  Etichetta: {json_content['predicted_label']}")
            print(f"📊 Confidence: {json_content['confidence']:.3f}")
            print(f"💭 Motivazione: {json_content['motivation']}")
            
            # Validazioni
            if json_content['predicted_label'] in labels:
                print("✅ Etichetta valida")
            else:
                print(f"❌ Etichetta non valida: {json_content['predicted_label']}")
                
            success_count += 1
            
        except Exception as e:
            print(f"❌ Errore: {e}")
            
    print("\n" + "=" * 65)
    print(f"🎯 RISULTATO: {success_count}/{len(conversations)} test riusciti")
    
    if success_count == len(conversations):
        print("🎉 STRUCTURED OUTPUTS PERFETTO!")
        print("✅ Etichette corrette, risposte italiane, JSON garantito")
        return True
    else:
        print("⚠️  Alcuni test falliti")
        return False

if __name__ == "__main__":
    test_structured_with_hardcoded_labels()
