#!/usr/bin/env python3
"""
Test debug configurabile MySQL per PromptManager

Autore: Valerio Bignardi
Data: 2025-09-02
Scopo: Testare i debug configurabili per le credenziali MySQL
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

def test_mysql_debug():
    """
    Test dei debug configurabili per MySQL
    """
    print("🔍 Test debug configurabile MySQL")
    print("=" * 50)
    
    try:
        # Inizializza PromptManager
        print("1. Inizializzazione PromptManager con debug attivato...")
        pm = PromptManager()
        
        # Forza riconnessione per vedere i debug
        if pm.connection:
            pm.disconnect()
        
        print("\n2. Test connessione con debug credenziali...")
        connection_ok = pm.connect()
        print(f"Risultato connessione: {connection_ok}")
        
        if connection_ok:
            print("\n3. Test recupero esempi con debug query...")
            tenant = MockTenant('015007d9-d413-11ef-86a5-96000228e7fe', 'Humanitas')
            esempi = pm.get_examples_list(tenant=tenant, engine='LLM')
            print(f"✅ Recuperati {len(esempi)} esempi")
            
        pm.disconnect()
        print("\n4. Test completato")
        
    except Exception as e:
        print(f"❌ ERRORE durante il test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mysql_debug()
