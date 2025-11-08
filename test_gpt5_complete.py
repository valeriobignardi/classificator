#!/usr/bin/env python3
"""
Test GPT-5 completo con Azure OpenAI - Verifica funzionalità complete
Autore: Valerio Bignardi
Data: 2025-11-08
"""

import asyncio
from Services.openai_service import OpenAIService

async def main():
    print("=" * 80)
    print("🧪 TEST COMPLETO GPT-5 vs GPT-4o")
    print("=" * 80)
    
    # Inizializza servizio
    openai_service = OpenAIService()
    
    # Test 1: GPT-4o (per confronto)
    print("\n" + "=" * 80)
    print("📋 TEST 1: GPT-4o con parametri custom")
    print("=" * 80)
    
    gpt4o_response = await openai_service.chat_completion(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Sei un assistente esperto di intelligenza artificiale."},
            {"role": "user", "content": "Spiega brevemente cos'è GPT-4o in 2-3 frasi."}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    print(f"\n✅ RISPOSTA GPT-4o:")
    print(f"   {gpt4o_response['choices'][0]['message']['content']}")
    print(f"\n📊 Token utilizzati:")
    print(f"   Prompt: {gpt4o_response['usage']['prompt_tokens']}")
    print(f"   Completion: {gpt4o_response['usage']['completion_tokens']}")
    print(f"   Totale: {gpt4o_response['usage']['total_tokens']}")
    
    # Test 2: GPT-5 (con parametri default)
    print("\n" + "=" * 80)
    print("📋 TEST 2: GPT-5 con parametri default (temp=1.0)")
    print("=" * 80)
    
    gpt5_response = await openai_service.chat_completion(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "Sei un assistente esperto di intelligenza artificiale."},
            {"role": "user", "content": "Spiega brevemente cos'è GPT-5 in 2-3 frasi."}
        ],
        temperature=0.7,  # Questo valore verrà IGNORATO per GPT-5
        max_tokens=200
    )
    
    print(f"\n✅ RISPOSTA GPT-5:")
    print(f"   {gpt5_response['choices'][0]['message']['content']}")
    print(f"\n📊 Token utilizzati:")
    print(f"   Prompt: {gpt5_response['usage']['prompt_tokens']}")
    print(f"   Completion: {gpt5_response['usage']['completion_tokens']}")
    print(f"   Totale: {gpt5_response['usage']['total_tokens']}")
    
    # Test 3: GPT-5 con prompt complesso
    print("\n" + "=" * 80)
    print("📋 TEST 3: GPT-5 con prompt complesso")
    print("=" * 80)
    
    gpt5_complex_response = await openai_service.chat_completion(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "Sei un esperto di machine learning e NLP."},
            {"role": "user", "content": "Elenca 3 differenze principali tra GPT-4o e GPT-5. Rispondi in modo conciso."}
        ],
        max_tokens=300
    )
    
    print(f"\n✅ RISPOSTA GPT-5:")
    print(f"   {gpt5_complex_response['choices'][0]['message']['content']}")
    print(f"\n📊 Token utilizzati:")
    print(f"   Prompt: {gpt5_complex_response['usage']['prompt_tokens']}")
    print(f"   Completion: {gpt5_complex_response['usage']['completion_tokens']}")
    print(f"   Totale: {gpt5_complex_response['usage']['total_tokens']}")
    
    print("\n" + "=" * 80)
    print("✅ TUTTI I TEST COMPLETATI CON SUCCESSO!")
    print("=" * 80)
    
    # Mostra statistiche
    stats = openai_service.get_stats()
    print(f"\n📊 STATISTICHE FINALI:")
    print(f"   Chiamate totali: {stats['total_calls']}")
    print(f"   Chiamate riuscite: {stats['successful_calls']}")
    print(f"   Chiamate fallite: {stats['failed_calls']}")
    print(f"   Token totali usati: {stats['total_tokens_used']}")
    print(f"   Latenza media: {stats['average_latency_seconds']:.2f}s")
    print(f"   Success rate: {stats['success_rate']:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())
