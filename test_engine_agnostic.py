#!/usr/bin/env python3
"""
Test validazione correzione engine-agnostic per esempi

Autore: Valerio Bignardi
Data: 2025-09-02
Scopo: Verificare che gli esempi siano ora indipendenti dall'engine
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Utils.prompt_manager import PromptManager

# Mock tenant class per il test
class MockTenant:
    def __init__(self, tenant_id, tenant_name):
        self.tenant_id = tenant_id
        self.tenant_name = tenant_name
        self.tenant_slug = tenant_name.lower()

def test_engine_agnostic_examples():
    """
    Test che gli esempi siano indipendenti dall'engine
    """
    print("🔍 Test esempi engine-agnostic")
    print("=" * 40)
    
    try:
        pm = PromptManager()
        pm.connect()
        
        tenant = MockTenant('015007d9-d413-11ef-86a5-96000228e7fe', 'Humanitas')
        
        # Test con engines diversi
        engines_to_test = [
            'LLM',
            'mistral:7b', 
            'gpt-4',
            'llama3',
            'random_engine',
            'whatever'
        ]
        
        results = {}
        for engine in engines_to_test:
            esempi = pm.get_examples_list(tenant=tenant, engine=engine)
            results[engine] = len(esempi)
            print(f"Engine '{engine}': {len(esempi)} esempi")
        
        # Verifica che tutti gli engine restituiscano lo stesso numero di esempi
        unique_counts = set(results.values())
        if len(unique_counts) == 1:
            count = list(unique_counts)[0]
            print(f"\n✅ SUCCESSO: Tutti gli engine restituiscono {count} esempi")
            print("✅ Gli esempi sono correttamente engine-agnostic!")
        else:
            print(f"\n❌ ERRORE: Conteggi diversi per engines: {results}")
            
        pm.disconnect()
        
    except Exception as e:
        print(f"❌ ERRORE durante il test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_engine_agnostic_examples()
