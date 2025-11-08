#!/usr/bin/env python3
"""
Script per elencare i deployment disponibili su Azure OpenAI.

Autore: Valerio Bignardi
Data: 2025-11-08
"""
import asyncio
import os
import aiohttp
from dotenv import load_dotenv


async def list_azure_deployments():
    """Elenca tutti i deployment disponibili su Azure OpenAI."""
    load_dotenv()
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-11-20")
    
    print("🔍 LISTA DEPLOYMENT AZURE OPENAI")
    print("=" * 80)
    print(f"📍 Endpoint: {endpoint}")
    print(f"📅 API Version: {api_version}")
    print()
    
    # URL per listare i deployment
    url = f"{endpoint}/openai/deployments?api-version={api_version}"
    
    headers = {
        "api-key": api_key
    }
    
    print(f"🌐 URL chiamata: {url}")
    print()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                status = response.status
                text = await response.text()
                
                print(f"📊 Status Code: {status}")
                print(f"📄 Response: {text}")
                print()
                
                if status == 200:
                    import json
                    data = json.loads(text)
                    
                    if "data" in data:
                        deployments = data["data"]
                        print(f"✅ Trovati {len(deployments)} deployment:")
                        print()
                        
                        for dep in deployments:
                            print(f"   🤖 ID: {dep.get('id')}")
                            print(f"      📦 Model: {dep.get('model')}")
                            print(f"      📅 Created: {dep.get('created_at')}")
                            print(f"      🔧 Status: {dep.get('status')}")
                            print()
                    else:
                        print("⚠️ Formato risposta inatteso")
                else:
                    print(f"❌ Errore HTTP {status}")
                    
    except Exception as e:
        print(f"❌ Errore: {e}")


if __name__ == "__main__":
    asyncio.run(list_azure_deployments())
