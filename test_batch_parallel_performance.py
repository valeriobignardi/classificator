#!/usr/bin/env python3
"""
Test Batch Processing Parallelo - Performance Comparison
Autore: Valerio Bignardi
Data: 2025-09-09

Test per verificare il miglioramento di performance del batch processing parallelo
"""

import asyncio
import time
from typing import List

def simulate_openai_request(text: str) -> dict:
    """Simula una richiesta OpenAI con delay"""
    # Simula latenza OpenAI
    time.sleep(0.1)  # 100ms per richiesta
    return {
        'predicted_label': 'test_label',
        'confidence': 0.8,
        'motivation': f'Test classification for: {text[:50]}'
    }

async def simulate_openai_request_async(text: str) -> dict:
    """Versione asincrona della simulazione"""
    # Simula latenza asincrona
    await asyncio.sleep(0.1)  # 100ms per richiesta
    return {
        'predicted_label': 'test_label',
        'confidence': 0.8,
        'motivation': f'Test classification for: {text[:50]}'
    }

async def batch_process_sequential(conversations: List[str], batch_size: int = 5):
    """Processamento sequenziale (metodo attuale)"""
    print(f"🔄 SEQUENTIAL: Processando {len(conversations)} conversazioni, batch_size={batch_size}")
    
    start_time = time.time()
    
    # Dividi in chunks
    chunks = [conversations[i:i + batch_size] for i in range(0, len(conversations), batch_size)]
    print(f"   📦 Diviso in {len(chunks)} chunk")
    
    results = []
    for chunk_idx, chunk in enumerate(chunks):
        print(f"   🔄 Processando chunk {chunk_idx + 1}/{len(chunks)}")
        
        # Simula processamento chunk sequenziale
        chunk_tasks = [simulate_openai_request_async(conv) for conv in chunk]
        chunk_results = await asyncio.gather(*chunk_tasks)
        results.extend(chunk_results)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"   ✅ SEQUENTIAL completato in {total_time:.2f}s")
    return results, total_time

async def batch_process_parallel(conversations: List[str], batch_size: int = 5):
    """Processamento parallelo (metodo ottimizzato)"""
    print(f"🚀 PARALLEL: Processando {len(conversations)} conversazioni, batch_size={batch_size}")
    
    start_time = time.time()
    
    # Dividi in chunks
    chunks = [conversations[i:i + batch_size] for i in range(0, len(conversations), batch_size)]
    print(f"   📦 Diviso in {len(chunks)} chunk")
    
    # Crea task per tutti i chunk simultaneamente
    chunk_tasks = []
    for chunk_idx, chunk in enumerate(chunks):
        print(f"   🎯 Creando task per chunk {chunk_idx + 1}")
        
        # Crea task per l'intero chunk
        async def process_chunk(chunk_data):
            chunk_task_list = [simulate_openai_request_async(conv) for conv in chunk_data]
            return await asyncio.gather(*chunk_task_list)
        
        chunk_tasks.append(process_chunk(chunk))
    
    print(f"   🚀 Avvio processamento parallelo di {len(chunk_tasks)} chunk...")
    
    # Esegue tutti i chunk in parallelo
    all_chunk_results = await asyncio.gather(*chunk_tasks)
    
    # Combina risultati
    results = []
    for chunk_results in all_chunk_results:
        results.extend(chunk_results)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"   ✅ PARALLEL completato in {total_time:.2f}s")
    return results, total_time

async def run_performance_test():
    """Esegue test di performance comparativo"""
    
    print("🧪 PERFORMANCE TEST: Sequential vs Parallel Batch Processing")
    print("=" * 70)
    
    # Crea conversazioni di test
    conversations = [f"Test conversation {i}: Example text for classification" for i in range(20)]
    batch_size = 5  # Crea 4 chunk da 5 conversazioni
    
    print(f"📊 Configurazione test:")
    print(f"   Conversazioni totali: {len(conversations)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Chunk attesi: {len(conversations) // batch_size}")
    print(f"   Simulazione latenza: 100ms per conversazione\n")
    
    # Test sequenziale
    seq_results, seq_time = await batch_process_sequential(conversations, batch_size)
    
    print()
    
    # Test parallelo  
    par_results, par_time = await batch_process_parallel(conversations, batch_size)
    
    # Calcola miglioramento
    improvement = ((seq_time - par_time) / seq_time) * 100
    speedup = seq_time / par_time
    
    print(f"\n🎯 RISULTATI PERFORMANCE:")
    print(f"   Sequential: {seq_time:.2f}s ({len(seq_results)} risultati)")
    print(f"   Parallel:   {par_time:.2f}s ({len(par_results)} risultati)")
    print(f"   Miglioramento: {improvement:.1f}% più veloce")
    print(f"   Speedup: {speedup:.1f}x")
    
    if improvement > 0:
        print(f"   🚀 SUCCESSO: Il processamento parallelo è più veloce!")
    else:
        print(f"   ⚠️ ATTENZIONE: Nessun miglioramento rilevato")

if __name__ == "__main__":
    asyncio.run(run_performance_test())
