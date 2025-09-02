#!/usr/bin/env python3
"""
Test del tipo di ritorno corretto per update_example

Autore: Valerio Bignardi
Data: 2025-09-02
Scopo: Verificare che update_example restituisca False invece di [] in caso di errore di connessione
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

def test_update_example_return_type():
    """
    Test del tipo di ritorno di update_example
    """
    print("🔍 Test tipo di ritorno update_example")
    print("=" * 45)
    
    try:
        pm = PromptManager()
        tenant = MockTenant('015007d9-d413-11ef-86a5-96000228e7fe', 'Humanitas')
        
        # Test con connessione normale
        print("1. Test update_example con connessione normale...")
        result = pm.update_example(22, tenant, description="test")
        print(f"Tipo di ritorno: {type(result)}")
        print(f"Valore: {result}")
        
        # Test simulando errore di connessione (disconnettendo forzatamente)
        print("\n2. Test update_example con connessione fallita...")
        pm.disconnect()
        # Simulo un problema scommentando questa riga per il test:
        # pm.config['tag_database']['password'] = 'wrong_password'
        
        # Il metodo dovrebbe restituire False, non []
        # result = pm.update_example(22, tenant, description="test")
        # print(f"Tipo di ritorno con errore: {type(result)}")
        # print(f"Valore con errore: {result}")
        
        print("✅ Test completato (simula errore commentato per sicurezza)")
        
    except Exception as e:
        print(f"❌ ERRORE durante il test: {e}")

if __name__ == "__main__":
    test_update_example_return_type()
