#!/usr/bin/env python3
"""
Script per testare chiamata diretta a deployment Azure OpenAI.

Autore: Valerio Bignardi  
Data: 2025-11-08
"""
import asyncio
import os
import aiohttp
import json
from dotenv import load_dotenv


async def test_direct_deployment_call():
    """Testa chiamata diretta a un deployment."""
    load_dotenv()
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment_name = "gpt-4o"
    api_version = "2024-02-15-preview"
    
    print("🧪 TEST CHIAMATA DIRETTA DEPLOYMENT AZURE OPENAI")
    print("=" * 80)
    print(f"📍 Endpoint: {endpoint}")
    print(f"🤖 Deployment: {deployment_name}")
    print(f"📅 API Version: {api_version}")
    print()
    
    # URL secondo documentazione Microsoft
    url = f"{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [
            {"role": "user", "content": "Dimmi ciao in italiano"}
        ],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    print(f"🌐 URL: {url}")
    print(f"📦 Payload: {json.dumps(payload, indent=2)}")
    print()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                status = response.status
                text = await response.text()
                
                print(f"📊 Status Code: {status}")
                print(f"📄 Response: {text}")
                print()
                
                if status == 200:
                    print("✅ SUCCESSO! Azure OpenAI funziona correttamente")
                    data = json.loads(text)
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0].get("message", {}).get("content", "")
                        print(f"💬 Risposta LLM: {content}")
                elif status == 404:
                    print(f"❌ Deployment '{deployment_name}' non trovato")
                    print("💡 Verifica il nome del deployment nel portale Azure")
                elif status == 401:
                    print("❌ API Key non valida")
                elif status == 429:
                    print("❌ Rate limit superato")
                else:
                    print(f"❌ Errore HTTP {status}")
                    
    except Exception as e:
        print(f"❌ Errore: {e}")


if __name__ == "__main__":
    asyncio.run(test_direct_deployment_call())
