#!/usr/bin/env python3
"""
Script per testare le versioni API più recenti di Azure OpenAI.

Autore: Valerio Bignardi
Data: 2025-11-08
"""
import asyncio
import os
import aiohttp
import json
from dotenv import load_dotenv


async def test_api_version(endpoint: str, api_key: str, api_version: str, deployment: str):
    """Testa una specifica versione API con chiamata chat completions."""
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 5,
        "temperature": 0.1
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as response:
                status = response.status
                if status == 200:
                    data = await response.json()
                    return api_version, status, "✅ FUNZIONA", data.get("model", "")
                else:
                    text = await response.text()
                    return api_version, status, text[:100], ""
    except Exception as e:
        return api_version, "ERROR", str(e)[:100], ""


async def main():
    """Testa le versioni API più recenti di Azure OpenAI."""
    load_dotenv()
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = "gpt-4o"
    
    print("🧪 TEST VERSIONI API RECENTI AZURE OPENAI")
    print("=" * 100)
    print(f"📍 Endpoint: {endpoint}")
    print(f"🤖 Deployment: {deployment}")
    print()
    
    # Versioni API dalla più recente alla più vecchia (GA = General Availability, non preview)
    api_versions = [
        "2024-12-01-preview",
        "2024-11-01-preview", 
        "2024-10-21",
        "2024-10-01-preview",
        "2024-09-01-preview",
        "2024-08-01-preview",
        "2024-07-01-preview",
        "2024-06-01",
        "2024-05-01-preview",
        "2024-04-01-preview",
        "2024-03-01-preview",
        "2024-02-15-preview",
        "2024-02-01",
        "2023-12-01-preview",
    ]
    
    print(f"🔍 Testando {len(api_versions)} versioni API (dalla più recente)...")
    print()
    
    tasks = [test_api_version(endpoint, api_key, version, deployment) for version in api_versions]
    results = await asyncio.gather(*tasks)
    
    print("📊 RISULTATI:")
    print("=" * 100)
    print(f"{'Versione API':<30} | {'Status':<10} | {'Risultato':<40} | {'Modello'}")
    print("-" * 100)
    
    working_versions = []
    for version, status, result, model in results:
        status_emoji = "✅" if status == 200 else "❌" if isinstance(status, int) else "⚠️"
        print(f"{status_emoji} {version:<28} | {str(status):<10} | {result:<40} | {model}")
        if status == 200:
            working_versions.append(version)
    
    print()
    if working_versions:
        latest = working_versions[0]
        print(f"🎯 VERSIONE PIÙ RECENTE FUNZIONANTE: {latest}")
        print()
        print(f"💡 Aggiorna .env con: AZURE_OPENAI_API_VERSION={latest}")
    else:
        print("❌ Nessuna versione API funzionante trovata!")


if __name__ == "__main__":
    asyncio.run(main())
