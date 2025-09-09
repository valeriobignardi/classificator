#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Autore: Valerio Bignardi
Data creazione: 2025-01-31
Descrizione: Test specifico per il tracing delle chiamate batch OpenAI
Data ultima modifica: 2025-01-31
"""

import asyncio
import sys
import os

# Aggiungi il percorso del progetto al Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Services.openai_service import OpenAIService


async def test_openai_batch_tracing():
    """
    Test diretto del tracing per il metodo batch_chat_completions
    
    Scopo: Verificare che il tracing mostri ENTER/EXIT e statistiche
           per le operazioni batch parallele
           
    Data ultima modifica: 2025-01-31
    """
    print("🔍 Test Tracing OpenAI Batch - Inizio")
    
    try:
        # Inizializza il servizio OpenAI
        service = OpenAIService()
        
        # Prepara richieste di test
        test_requests = [
            {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Sei un assistente medico."},
                    {"role": "user", "content": "Classifica: dolore toracico acuto"}
                ],
                "max_tokens": 50,
                "temperature": 0.1
            },
            {
                "model": "gpt-4o-mini", 
                "messages": [
                    {"role": "system", "content": "Sei un assistente medico."},
                    {"role": "user", "content": "Classifica: richiesta visita cardiologica"}
                ],
                "max_tokens": 50,
                "temperature": 0.1
            },
            {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Sei un assistente medico."},
                    {"role": "user", "content": "Classifica: controllo nutrizionale diabete"}
                ],
                "max_tokens": 50,
                "temperature": 0.1
            }
        ]
        
        print(f"📊 Esecuzione batch con {len(test_requests)} richieste")
        print("🔍 Osserva i log di tracing:")
        print("   ➡️  ENTER - parametri batch")
        print("   ➡️  INFO  - concorrenza effettiva") 
        print("   ➡️  INFO  - avvio esecuzione parallela")
        print("   ➡️  EXIT  - statistiche complete")
        print()
        
        # Esegui il batch con tracing (concorrenza 3)
        start_time = asyncio.get_event_loop().time()
        
        results = await service.batch_chat_completions(
            requests=test_requests,
            max_concurrent=3
        )
        
        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time
        
        print(f"✅ Batch completato in {execution_time:.2f}s")
        print(f"📈 Risultati processati: {len(results)}")
        
        # Analizza risultati
        success_count = sum(1 for r in results if r.get('success', False))
        error_count = len(results) - success_count
        
        print(f"   ✅ Successi: {success_count}")
        print(f"   ❌ Errori: {error_count}")
        
        # Mostra statistiche servizio
        stats = service.get_stats()
        print(f"\n📊 Statistiche servizio OpenAI:")
        print(f"   🔢 Chiamate totali: {stats['total_calls']}")
        print(f"   ✅ Chiamate riuscite: {stats['successful_calls']}")
        print(f"   ❌ Chiamate fallite: {stats['failed_calls']}")
        print(f"   📈 Success rate: {stats['success_rate']:.1f}%")
        print(f"   ⏱️  Latenza media: {stats['average_latency_seconds']}s")
        print(f"   🔄 Max parallele raggiunto: {stats['max_parallel_reached']}")
        print(f"   📊 Token totali usati: {stats['total_tokens_used']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        return []


async def test_large_batch():
    """
    Test con batch più grande per verificare concorrenza
    
    Scopo: Testare il comportamento del tracing con molte richieste parallele
    
    Data ultima modifica: 2025-01-31
    """
    print("\n🚀 Test Large Batch - 10 richieste")
    
    try:
        service = OpenAIService()
        
        # Genera 10 richieste
        large_batch = []
        for i in range(10):
            large_batch.append({
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": f"Dimmi il numero {i+1} in parole"}
                ],
                "max_tokens": 20
            })
        
        print(f"📊 Batch di {len(large_batch)} richieste, concorrenza=5")
        
        # Esegui con concorrenza limitata
        results = await service.batch_chat_completions(
            requests=large_batch,
            max_concurrent=5
        )
        
        success_count = sum(1 for r in results if r.get('success', False))
        print(f"✅ Completate {success_count}/{len(results)} richieste")
        
        # Statistiche finali
        final_stats = service.get_stats()
        print(f"📈 Statistiche finali: {final_stats['total_calls']} chiamate totali")
        
    except Exception as e:
        print(f"❌ Errore nel test large batch: {e}")


if __name__ == "__main__":
    print("🚀 Test Tracing Batch OpenAI Service")
    
    # Test base
    asyncio.run(test_openai_batch_tracing())
    
    # Test con batch più grande
    asyncio.run(test_large_batch())
    
    print("\n✅ Tutti i test completati!")
