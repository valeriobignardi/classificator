#!/usr/bin/env python3
"""
============================================================================
Verifica Deployments Azure OpenAI
============================================================================

Autore: Valerio Bignardi
Data: 2025-11-08

Descrizione:
    Verifica quali deployments sono effettivamente disponibili su Azure

============================================================================
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def check_azure_deployments():
    """
    Verifica deployments disponibili su Azure OpenAI
    """
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-10-21')
    
    print(f"\n🔍 Verifica Deployments Azure OpenAI")
    print(f"="*80)
    print(f"Endpoint: {endpoint}")
    print(f"API Version: {api_version}")
    
    # URL per ottenere lista deployments
    # https://{endpoint}/openai/deployments?api-version={api-version}
    url = f"{endpoint.rstrip('/')}/openai/deployments?api-version={api_version}"
    
    headers = {
        'api-key': api_key
    }
    
    try:
        print(f"\n🚀 Richiesta a: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            deployments = data.get('data', [])
            
            print(f"\n✅ Trovati {len(deployments)} deployments:\n")
            
            for dep in deployments:
                dep_id = dep.get('id', 'N/A')
                model = dep.get('model', 'N/A')
                status = dep.get('status', 'N/A')
                scale_type = dep.get('scale_settings', {}).get('scale_type', 'N/A')
                
                print(f"📦 Deployment: {dep_id}")
                print(f"   Modello: {model}")
                print(f"   Status: {status}")
                print(f"   Scale Type: {scale_type}")
                print()
                
        else:
            print(f"\n❌ Errore HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()


def test_specific_deployment(deployment_name, prompt="Ciao!"):
    """
    Test specifico deployment
    """
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-10-21')
    
    # URL per chat completions
    url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    
    headers = {
        'api-key': api_key,
        'Content-Type': 'application/json'
    }
    
    payload = {
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'max_tokens': 50
    }
    
    try:
        print(f"\n🧪 Test deployment: {deployment_name}")
        print(f"   URL: {url}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"   ✅ SUCCESSO: {content[:100]}...")
            return True
        else:
            print(f"   ❌ ERRORE HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔍 VERIFICA DEPLOYMENTS AZURE OPENAI")
    print("="*80)
    
    # Verifica deployments disponibili
    check_azure_deployments()
    
    # Test deployments configurati
    print("\n" + "="*80)
    print("🧪 TEST DEPLOYMENTS CONFIGURATI")
    print("="*80)
    
    gpt4o_deployment = os.getenv('AZURE_OPENAI_GPT4O_DEPLOYMENT', 'gpt-4o')
    gpt5_deployment = os.getenv('AZURE_OPENAI_GPT5_DEPLOYMENT', 'gpt-5')
    
    print(f"\nDeployment configurati nel .env:")
    print(f"  GPT-4o: {gpt4o_deployment}")
    print(f"  GPT-5: {gpt5_deployment}")
    
    # Test GPT-4o
    test_specific_deployment(gpt4o_deployment, "Rispondi solo: OK")
    
    # Test GPT-5 (probabilmente fallirà se non è deployato)
    test_specific_deployment(gpt5_deployment, "Rispondi solo: OK")
    
    print("\n" + "="*80)
    print("📝 NOTA:")
    print("Se GPT-5 non è disponibile, devi creare il deployment su Azure Portal.")
    print("="*80 + "\n")
