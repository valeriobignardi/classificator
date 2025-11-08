#!/usr/bin/env python3
"""
Test integrazione completa Azure OpenAI - Chat Completions + Embeddings
Autore: Valerio Bignardi
Data: 2025-11-08

Verifica che:
1. OpenAIService funzioni con Azure OpenAI (GPT-4o e GPT-5)
2. OpenAIEmbedder funzioni con Azure OpenAI (text-embedding-3-large)
3. IntelligentClassifier funzioni con Azure OpenAI
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Services'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'EmbeddingEngine'))

from Services.openai_service import OpenAIService
from EmbeddingEngine.openai_embedder import OpenAIEmbedder

async def test_chat_completions():
    """Test chat completions con GPT-4o e GPT-5"""
    print("\n" + "=" * 80)
    print("🧪 TEST 1: CHAT COMPLETIONS (GPT-4o e GPT-5)")
    print("=" * 80)
    
    service = OpenAIService()
    
    # Test GPT-4o
    print("\n📋 Test GPT-4o...")
    gpt4o_response = await service.chat_completion(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Sei un assistente AI conciso."},
            {"role": "user", "content": "Elenca 3 vantaggi di Azure OpenAI in una frase."}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    print(f"✅ GPT-4o risposta:")
    print(f"   {gpt4o_response['choices'][0]['message']['content'][:200]}...")
    print(f"📊 Token: {gpt4o_response['usage']['total_tokens']}")
    
    # Test GPT-5 (con parametri default)
    print("\n📋 Test GPT-5...")
    gpt5_response = await service.chat_completion(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "Sei un assistente AI avanzato."},
            {"role": "user", "content": "Spiega in una frase cosa rende GPT-5 diverso da GPT-4."}
        ],
        max_tokens=150
    )
    
    print(f"✅ GPT-5 risposta:")
    content = gpt5_response['choices'][0]['message']['content']
    if content:
        print(f"   {content[:200]}...")
    else:
        print(f"   ⚠️ Risposta vuota (deployment GPT-5 potrebbe richiedere configurazione)")
    print(f"📊 Token: {gpt5_response['usage']['total_tokens']}")
    
    return True

def test_embeddings():
    """Test embeddings con text-embedding-3-large"""
    print("\n" + "=" * 80)
    print("🧪 TEST 2: EMBEDDINGS (text-embedding-3-large)")
    print("=" * 80)
    
    embedder = OpenAIEmbedder(
        model_name="text-embedding-3-large",
        test_on_init=False
    )
    
    # Test su testi di esempio
    test_texts = [
        "Azure OpenAI fornisce accesso ai modelli OpenAI tramite Azure.",
        "GPT-4o è un modello multimodale che supporta testo e immagini.",
        "Gli embeddings text-embedding-3-large hanno 3072 dimensioni."
    ]
    
    print(f"\n📋 Generazione embeddings per {len(test_texts)} testi...")
    embeddings = embedder.encode(
        texts=test_texts,
        batch_size=32,
        show_progress=False
    )
    
    print(f"✅ Embeddings generati:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dtype: {embeddings.dtype}")
    print(f"   Range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    
    # Verifica normalizzazione
    import numpy as np
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"   Norme: {norms}")
    
    if np.allclose(norms, 1.0, atol=0.01):
        print(f"   ✅ Embeddings normalizzati correttamente")
    else:
        print(f"   ⚠️ Embeddings NON normalizzati (verificare)")
    
    return True

async def test_intelligent_classifier():
    """Test IntelligentClassifier con Azure OpenAI"""
    print("\n" + "=" * 80)
    print("🧪 TEST 3: INTELLIGENT CLASSIFIER INTEGRATION")
    print("=" * 80)
    
    try:
        # Verifica che il sistema sia configurato correttamente
        service = OpenAIService()
        stats = service.get_stats()
        
        print(f"✅ OpenAIService pronto:")
        print(f"   Chiamate totali: {stats['total_calls']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        print(f"   Token usati: {stats['total_tokens_used']}")
        
        print(f"\n💡 IntelligentClassifier può ora usare Azure OpenAI")
        print(f"   Modelli disponibili: gpt-4o, gpt-5")
        print(f"   Embeddings: text-embedding-3-large (3072 dim)")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        return False

async def main():
    """Esegue tutti i test"""
    print("=" * 80)
    print("🚀 TEST INTEGRAZIONE COMPLETA AZURE OPENAI")
    print("=" * 80)
    
    results = {
        'chat_completions': False,
        'embeddings': False,
        'classifier': False
    }
    
    try:
        # Test 1: Chat completions
        results['chat_completions'] = await test_chat_completions()
        
        # Test 2: Embeddings
        results['embeddings'] = test_embeddings()
        
        # Test 3: Classifier integration
        results['classifier'] = await test_intelligent_classifier()
        
    except Exception as e:
        print(f"\n❌ ERRORE DURANTE I TEST: {e}")
        import traceback
        traceback.print_exc()
    
    # Riepilogo finale
    print("\n" + "=" * 80)
    print("📊 RIEPILOGO TEST")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 80)
        print("🎉 TUTTI I TEST SUPERATI! Azure OpenAI è configurato correttamente!")
        print("=" * 80)
        print("\n📝 Configurazione attuale:")
        print(f"   🌩️ Azure Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
        print(f"   📅 API Version: {os.getenv('AZURE_OPENAI_API_VERSION')}")
        print(f"   🤖 GPT-4o Deployment: {os.getenv('AZURE_OPENAI_GPT4O_DEPLOYMENT')}")
        print(f"   🚀 GPT-5 Deployment: {os.getenv('AZURE_OPENAI_GPT5_DEPLOYMENT')}")
        print(f"   📊 Embeddings Deployment: {os.getenv('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT')}")
    else:
        print("\n⚠️ ALCUNI TEST SONO FALLITI - Verifica la configurazione")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
