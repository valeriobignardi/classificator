#!/usr/bin/env python3
"""
Script per testare diverse versioni API di Azure OpenAI.

Autore: Valerio Bignardi
Data: 2025-11-08
"""
import asyncio
import os
import aiohttp
from dotenv import load_dotenv


async def test_api_version(endpoint: str, api_key: str, api_version: str):
    """Testa una specifica versione API."""
    url = f"{endpoint}/openai/deployments?api-version={api_version}"
    
    headers = {
        "api-key": api_key
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                status = response.status
                text = await response.text()
                return api_version, status, text[:200]
    except Exception as e:
        return api_version, "ERROR", str(e)[:200]


async def main():
    """Testa diverse versioni API di Azure OpenAI."""
    load_dotenv()
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    
    print("🧪 TEST VERSIONI API AZURE OPENAI")
    print("=" * 80)
    print(f"📍 Endpoint: {endpoint}")
    print()
    
    # Versioni API stabili e documentate per Azure OpenAI
    api_versions = [
        "2024-10-21",
        "2024-08-01-preview",
        "2024-06-01",
        "2024-05-01-preview",
        "2024-04-01-preview",
        "2024-03-01-preview",
        "2024-02-15-preview",
        "2023-12-01-preview",
        "2023-05-15",
    ]
    
    print(f"🔍 Testando {len(api_versions)} versioni API...")
    print()
    
    tasks = [test_api_version(endpoint, api_key, version) for version in api_versions]
    results = await asyncio.gather(*tasks)
    
    print("📊 RISULTATI:")
    print("=" * 80)
    
    for version, status, response in sorted(results, key=lambda x: x[1] == 200, reverse=True):
        status_emoji = "✅" if status == 200 else "❌" if isinstance(status, int) else "⚠️"
        print(f"{status_emoji} {version:25} | Status: {status:6} | {response}")
    
    print()
    print("💡 Usa la versione con status 200 per configurare AZURE_OPENAI_API_VERSION")


if __name__ == "__main__":
    asyncio.run(main())
