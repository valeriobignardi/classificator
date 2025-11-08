#!/usr/bin/env python3
"""
Verifica deployments embeddings disponibili su Azure OpenAI
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
api_version = os.getenv('AZURE_OPENAI_API_VERSION')

# Test deployment text-embedding-ada-002 (modello legacy ma comunemente disponibile)
test_deployments = [
    'text-embedding-ada-002',
    'text-embedding-3-large',
    'text-embedding-3-small',
    'embeddings',
    'embedding'
]

print("=" * 80)
print("🔍 VERIFICA DEPLOYMENTS EMBEDDINGS SU AZURE OPENAI")
print("=" * 80)
print(f"📍 Endpoint: {azure_endpoint}")
print(f"📅 API Version: {api_version}")
print()

for deployment in test_deployments:
    print(f"🧪 Test deployment: {deployment}")
    
    url = f"{azure_endpoint}/openai/deployments/{deployment}/embeddings"
    url += f"?api-version={api_version}"
    
    headers = {
        "api-key": azure_api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": "Test embedding"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            embedding_dim = len(result['data'][0]['embedding'])
            print(f"   ✅ FUNZIONA! Dimensione: {embedding_dim}")
        elif response.status_code == 404:
            print(f"   ❌ Deployment non trovato (404)")
        else:
            print(f"   ⚠️ HTTP {response.status_code}: {response.text[:100]}")
    
    except Exception as e:
        print(f"   ❌ Errore: {e}")
    
    print()

print("=" * 80)
print("💡 RACCOMANDAZIONE:")
print("Usa il deployment che ha risposto con ✅ FUNZIONA!")
print("Aggiorna AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT nel file .env")
print("=" * 80)
