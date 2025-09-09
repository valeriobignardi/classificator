#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test del tracing nel servizio OpenAI - debug import
"""

import asyncio
import sys
import os

# Aggiungi il percorso del progetto  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test dell'import specifico
print("🔍 Test Import OpenAI Service")
try:
    from Services.openai_service import OpenAIService
    print("✅ Import riuscito")
    
    # Crea il servizio
    service = OpenAIService()
    print("✅ Servizio inizializzato")
    
    # Testa la funzione trace_all del servizio
    print("\n🧪 Test trace_all del servizio")
    
    # Modifica temporanea per test
    original_trace = service.__class__.__dict__.get('trace_all', None)
    
    if hasattr(service, 'trace_all'):
        print("✅ trace_all trovata nel servizio")
    else:
        print("❌ trace_all non trovata nel servizio")
        
    # Test diretto della funzione (se esiste in scope)
    import Services.openai_service as oas_module
    if hasattr(oas_module, 'trace_all'):
        print("✅ trace_all trovata nel modulo")
        # Testa direttamente
        oas_module.trace_all("test_function", "ENTER", test_param="value")
    else:
        print("❌ trace_all non trovata nel modulo")
        
except Exception as e:
    print(f"❌ Errore durante import: {e}")
    import traceback
    traceback.print_exc()


async def test_with_print_debug():
    """Test con debug diretto nel batch_chat_completions"""
    print("\n🧪 Test con debug diretto")
    
    try:
        service = OpenAIService()
        
        # Test con una sola richiesta semplice
        requests = [{
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Ciao"}],
            "max_tokens": 10
        }]
        
        print("🔍 Chiamando batch_chat_completions...")
        
        # Qui dovrebbero apparire i log di tracing
        results = await service.batch_chat_completions(requests, max_concurrent=1)
        
        print(f"✅ Risultati: {len(results)}")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        

if __name__ == "__main__":
    print("🚀 Debug Tracing OpenAI Service")
    
    # Test import
    print("=" * 50)
    
    # Test esecuzione
    print("\n" + "=" * 50)
    asyncio.run(test_with_print_debug())
    
    print("\n✅ Debug completato")
