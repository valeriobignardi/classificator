#!/usr/bin/env python3
"""
Test rapido per verificare la connessione OpenAI
"""

import os
import sys
from dotenv import load_dotenv

# Carica variabili d'ambiente
load_dotenv()

# Aggiungi path per l'embedder
sys.path.append('EmbeddingEngine')

try:
    from openai_embedder import OpenAIEmbedder
    
    # Test con la chiave dal file .env
    api_key = os.getenv('OPENAI_API_KEY')
    
    print(f"🔑 API Key trovata: {'✅ Sì' if api_key else '❌ No'}")
    if api_key:
        print(f"   Chiave inizia con: {api_key[:12]}...")
    
    # Test modello large
    print("\n🧪 Test OpenAI text-embedding-3-large:")
    try:
        embedder_large = OpenAIEmbedder(
            api_key=api_key,
            model_name="text-embedding-3-large",
            test_on_init=True
        )
        print("✅ text-embedding-3-large funziona!")
        
    except Exception as e:
        print(f"❌ text-embedding-3-large fallito: {e}")
    
    # Test modello small
    print("\n🧪 Test OpenAI text-embedding-3-small:")
    try:
        embedder_small = OpenAIEmbedder(
            api_key=api_key,
            model_name="text-embedding-3-small",
            test_on_init=True
        )
        print("✅ text-embedding-3-small funziona!")
        
    except Exception as e:
        print(f"❌ text-embedding-3-small fallito: {e}")

except Exception as e:
    print(f"❌ Errore generale: {e}")
