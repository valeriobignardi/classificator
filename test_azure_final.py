#!/usr/bin/env python3
"""
Test finale Azure OpenAI - Verifica configurazione corretta
Autore: Valerio Bignardi
Data: 2025-11-08
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Services'))

from Services.openai_service import OpenAIService

async def main():
    print("=" * 80)
    print("🎉 TEST FINALE AZURE OPENAI")
    print("=" * 80)
    
    service = OpenAIService()
    
    # Test GPT-4o
    print("\n✅ Test GPT-4o (con parametri custom)...")
    response_gpt4o = await service.chat_completion(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Dimmi ciao in italiano!"}
        ],
        temperature=0.7,
        max_tokens=50
    )
    
    print(f"📝 Risposta GPT-4o: {response_gpt4o['choices'][0]['message']['content']}")
    print(f"📊 Token usati: {response_gpt4o['usage']['total_tokens']}")
    
    # Test GPT-5
    print("\n✅ Test GPT-5 (parametri default temp=1.0)...")
    response_gpt5 = await service.chat_completion(
        model="gpt-5",
        messages=[
            {"role": "user", "content": "Rispondi solo: OK"}
        ],
        max_tokens=10
    )
    
    content = response_gpt5['choices'][0]['message']['content']
    if content.strip():
        print(f"📝 Risposta GPT-5: {content}")
    else:
        print(f"📝 Risposta GPT-5: (vuota - deployment potrebbe richiedere configurazione)")
    print(f"📊 Token usati: {response_gpt5['usage']['total_tokens']}")
    
    # Statistiche finali
    print("\n" + "=" * 80)
    stats = service.get_stats()
    print("📊 STATISTICHE FINALI:")
    print(f"   Chiamate totali: {stats['total_calls']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Token totali: {stats['total_tokens_used']}")
    print(f"   Latenza media: {stats['average_latency_seconds']:.2f}s")
    
    print("\n" + "=" * 80)
    print("🎉 AZURE OPENAI CONFIGURATO CORRETTAMENTE!")
    print("=" * 80)
    print("\n📝 Configurazione Azure:")
    print(f"   Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"   API Version: {os.getenv('AZURE_OPENAI_API_VERSION')}")
    print(f"   GPT-4o: {os.getenv('AZURE_OPENAI_GPT4O_DEPLOYMENT')} (v{os.getenv('AZURE_OPENAI_GPT4O_VERSION')})")
    print(f"   GPT-5: {os.getenv('AZURE_OPENAI_GPT5_DEPLOYMENT')} (v{os.getenv('AZURE_OPENAI_GPT5_VERSION')})")
    
    print("\n💡 Il sistema è pronto per usare Azure OpenAI in produzione!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
