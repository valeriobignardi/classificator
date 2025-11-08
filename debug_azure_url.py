#!/usr/bin/env python3
"""
Debug Azure OpenAI URL Construction
"""

import os

def debug_azure_url_construction():
    """
    Debug dell'URL costruito per Azure OpenAI
    """
    print("🔍 DEBUG URL CONSTRUCTION AZURE OPENAI")
    print("=" * 60)
    
    # Variabili d'ambiente
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-11-20')
    gpt4o_deployment = os.getenv('AZURE_OPENAI_GPT4O_DEPLOYMENT', 'gpt-4o')
    
    print(f"🌩️ Azure Endpoint: {azure_endpoint}")
    print(f"📅 API Version: {api_version}")
    print(f"🤖 GPT-4o Deployment: {gpt4o_deployment}")
    
    # Costruzione URL come nel codice
    endpoint = 'chat/completions'
    
    # URL che stiamo costruendo
    constructed_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{gpt4o_deployment}/{endpoint}"
    constructed_url += f"?api-version={api_version}"
    
    print(f"\n📡 URL Costruito:")
    print(f"   {constructed_url}")
    
    # URL corretto secondo documentazione Microsoft
    correct_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{gpt4o_deployment}/chat/completions"
    correct_url += f"?api-version={api_version}"
    
    print(f"\n✅ URL Corretto (secondo docs Microsoft):")
    print(f"   {correct_url}")
    
    if constructed_url == correct_url:
        print(f"\n✅ URL COSTRUITO CORRETTAMENTE!")
    else:
        print(f"\n❌ URL NON CORRETTO!")
        print(f"   Differenze trovate.")

if __name__ == "__main__":
    debug_azure_url_construction()