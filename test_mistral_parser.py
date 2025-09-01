#!/usr/bin/env python3
"""
Test per il nuovo parser formato Mistral-nemo pseudo function call
Autore: Valerio Bignardi  
Data: 2025-09-01
"""

import sys
import os
import json

# Aggiungi path del progetto
sys.path.append('/home/ubuntu/classificatore')

def test_mistral_nemo_parser():
    """
    Simula il contenuto restituito da mistral-nemo e testa il parsing
    """
    print("🧪 TEST PARSER FORMATO MISTRAL-NEMO")
    print("="*80)
    
    # Simula il contenuto problematico visto nei log
    mistral_content = '**{"name": "classify_conversation", "arguments": {"predicted_label": "Visita Specialistica", "confidence": 0.8, "motivation": "La conversazione riguarda richiesta di visita specialistica"}}'
    
    print(f"📝 Content da parsare: {mistral_content}")
    print("-"*80)
    
    # Simula il parsing
    try:
        if '**{"name": "classify_conversation"' in mistral_content:
            print("✅ Formato Mistral-nemo riconosciuto")
            
            # Estrai solo la parte JSON dopo "arguments":
            start_idx = mistral_content.find('"arguments": ')
            if start_idx != -1:
                start_idx += len('"arguments": ')
                print(f"🔍 Inizio arguments a posizione: {start_idx}")
                
                # Trova la fine del JSON degli arguments
                json_content = mistral_content[start_idx:]
                print(f"📄 JSON content grezzo: {json_content}")
                
                # Rimuovi eventuali caratteri trailing
                if json_content.endswith('}}'):
                    json_content = json_content[:-1]  # Rimuovi la } finale dell'oggetto esterno
                    print(f"📄 JSON content pulito: {json_content}")
                
                try:
                    arguments = json.loads(json_content)
                    print(f"✅ JSON parsato correttamente: {arguments}")
                    
                    if all(key in arguments for key in ['predicted_label', 'confidence', 'motivation']):
                        print("✅ Tutti i campi richiesti presenti")
                        print(f"   🏷️  Label: {arguments['predicted_label']}")
                        print(f"   📊 Confidence: {arguments['confidence']}")
                        print(f"   💭 Motivation: {arguments['motivation']}")
                        return arguments
                    else:
                        print("❌ Campi mancanti")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Errore parsing JSON: {e}")
            else:
                print("❌ 'arguments': non trovato")
        else:
            print("❌ Formato Mistral-nemo non riconosciuto")
            
    except Exception as e:
        print(f"❌ Errore generale: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*80)

if __name__ == "__main__":
    test_mistral_nemo_parser()
