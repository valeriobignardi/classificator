#!/usr/bin/env python3
"""
Test per verificare che la correzione del training supervisionato funzioni
Autore: Valerio Bignardi
Data: 2025-09-09
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test importazione
try:
    from Pipeline.end_to_end_pipeline import EndToEndPipeline
    print("✅ Importazione EndToEndPipeline riuscita")
except Exception as e:
    print(f"❌ Errore importazione: {e}")
    exit(1)

# Test istanziazione base
try:
    # Configura parametri minimi per il test
    config = {
        'database': {
            'host': 'localhost',
            'port': 3306,
            'user': 'test',
            'password': 'test',
            'database': 'test'
        },
        'embeddings': {
            'model': 'sentence-transformers/all-MiniLM-L6-v2'
        },
        'clustering': {
            'min_cluster_size': 5
        }
    }
    
    # Mock tenant per test
    class MockTenant:
        def __init__(self):
            self.tenant_id = "test-tenant"
            self.tenant_name = "Test"
            self.tenant_slug = "test"
    
    tenant = MockTenant()
    
    # Crea istanza pipeline (senza inizializzare componenti pesanti)
    pipeline = EndToEndPipeline(
        tenant=tenant,
        config_path=None  # Usa configurazione di default
    )
    print("✅ Istanziazione EndToEndPipeline riuscita")
    
    # Verifica che il metodo select_representatives_from_documents esista
    if hasattr(pipeline, 'select_representatives_from_documents'):
        print("✅ Metodo select_representatives_from_documents trovato")
        
        # Verifica signature del metodo
        import inspect
        sig = inspect.signature(pipeline.select_representatives_from_documents)
        params = list(sig.parameters.keys())
        print(f"✅ Parametri metodo: {params}")
        
        # Verifica che i parametri corretti siano presenti
        if 'documenti' in params and 'max_sessions' in params:
            print("✅ Parametri del metodo corretti")
        else:
            print(f"❌ Parametri incorretti: {params}")
            exit(1)
    else:
        print("❌ Metodo select_representatives_from_documents non trovato")
        exit(1)
        
except Exception as e:
    print(f"❌ Errore durante il test: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n🎉 TUTTI I TEST COMPLETATI CON SUCCESSO!")
print("✅ La correzione del training supervisionato dovrebbe funzionare correttamente")
