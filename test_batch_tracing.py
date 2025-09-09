#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Autore: Valerio Bignardi
Data creazione: 2025-01-31
Descrizione: Test del tracing per le operazioni batch di OpenAI
Data ultima modifica: 2025-01-31
"""

import asyncio
import sys
import os

# Aggiungi il percorso del progetto al Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Services.openai_service import OpenAIService
from Classification.intelligent_classifier import IntelligentClassifier


def test_batch_tracing():
    """
    Test del tracing per le operazioni batch di OpenAI
    
    Scopo: Verificare che il tracing funzioni correttamente durante 
           l'esecuzione delle operazioni batch parallele
    
    Data ultima modifica: 2025-01-31
    """
    print("🔍 Test Batch Tracing - Inizio")
    
    try:
        # Inizializza il classificatore con integrazione database
        classifier = IntelligentClassifier()
        
        # Crea alcuni testi di test per verificare il batch processing
        test_texts = [
            "Il paziente presenta dolore toracico acuto",
            "Richiesta appuntamento per visita cardiologica", 
            "Consulenza nutrizionale per diabete tipo 2",
            "Prescrizione farmaci antipertensivi",
            "Controllo post-operatorio chirurgia generale"
        ]
        
        print(f"📊 Testing batch processing con {len(test_texts)} testi")
        print("🔍 Verifica che il tracing mostri:")
        print("   - ENTER con parametri batch")
        print("   - INFO con concorrenza effettiva")
        print("   - EXIT con statistiche complete")
        print()
        
        # Esegui la classificazione che userà il batch processing
        results = classifier.classify_batch(test_texts)
        
        print("✅ Batch processing completato con tracing")
        print(f"📈 Risultati: {len(results)} classificazioni")
        
        # Verifica risultati
        for i, result in enumerate(results):
            status = "✅" if result.get('success', False) else "❌"
            print(f"   {status} Testo {i+1}: {result.get('classification', 'N/A')}")
    
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()


async def test_openai_service_direct():
    """
    Test diretto del servizio OpenAI per verificare il tracing
    
    Scopo: Test del metodo batch_chat_completions con tracing
    
    Data ultima modifica: 2025-01-31
    """
    print("\n🔧 Test Diretto OpenAI Service")
    
    try:
        service = OpenAIService()
        
        # Crea richieste di test
        requests = [
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Classifica: dolore toracico"}],
                "max_tokens": 50
            },
            {
                "model": "gpt-4o-mini", 
                "messages": [{"role": "user", "content": "Classifica: visita cardiologica"}],
                "max_tokens": 50
            }
        ]
        
        print(f"📊 Testing batch_chat_completions con {len(requests)} richieste")
        
        # Esegui batch con tracing
        results = await service.batch_chat_completions(requests, max_concurrent=2)
        
        print(f"✅ Batch completato: {len(results)} risposte")
        
        # Mostra statistiche
        stats = service.get_stats()
        print(f"📈 Statistiche servizio:")
        print(f"   - Chiamate totali: {stats['total_calls']}")
        print(f"   - Successi: {stats['successful_calls']}")
        print(f"   - Errori: {stats['failed_calls']}")
        print(f"   - Latenza media: {stats['average_latency_seconds']}s")
    
    except Exception as e:
        print(f"❌ Errore nel test diretto: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Avvio Test Batch Tracing")
    
    # Test del tracing batch
    test_batch_tracing()
    
    # Test diretto servizio
    asyncio.run(test_openai_service_direct())
    
    print("\n✅ Test completati")
