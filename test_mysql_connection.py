#!/usr/bin/env python3
"""
Test connessione MySQL per PromptManager

Autore: Valerio Bignardi
Data: 2025-09-02
Scopo: Verificare la connessione MySQL dal PromptManager
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

def test_mysql_connection():
    """
    Test la connessione MySQL attraverso il PromptManager
    """
    print("🔍 Test connessione MySQL per PromptManager")
    print("=" * 50)
    
    try:
        # Inizializza PromptManager
        print("1. Inizializzazione PromptManager...")
        pm = PromptManager()
        print("✅ PromptManager inizializzato")
        
        # Test connessione esplicita
        print("\n2. Test connessione MySQL...")
        connection_ok = pm.connect()
        print(f"Risultato connessione: {connection_ok}")
        
        if connection_ok:
            print("✅ Connessione MySQL riuscita!")
            
            # Test recupero tenant (mock per test)
            print("\n3. Test con tenant mock...")
            tenant = MockTenant('015007d9-d413-11ef-86a5-96000228e7fe', 'Humanitas')
            
            if tenant:
                print(f"✅ Tenant mock: {tenant.tenant_name}")
                
                # Test recupero esempi con engine mistral:7b
                print("\n4a. Test recupero esempi con engine mistral:7b...")
                esempi_mistral = pm.get_examples_list(tenant=tenant, engine='mistral:7b')
                print(f"✅ Recuperati {len(esempi_mistral)} esempi per mistral:7b")
                
                # Test recupero esempi con engine LLM
                print("\n4b. Test recupero esempi con engine LLM...")
                esempi_llm = pm.get_examples_list(tenant=tenant, engine='LLM')
                print(f"✅ Recuperati {len(esempi_llm)} esempi per LLM")
                
                if esempi_llm:
                    print("Primo esempio:")
                    print(f"  - ID: {esempi_llm[0]['id']}")
                    print(f"  - Nome: {esempi_llm[0]['esempio_name']}")
                    print(f"  - Tipo: {esempi_llm[0]['esempio_type']}")
                    print(f"  - Categoria: {esempi_llm[0]['categoria']}")
                else:
                    print("⚠️ Nessun esempio trovato per tenant humanitas")
                
            else:
                print("❌ Tenant mock fallito")
                
        else:
            print("❌ Connessione MySQL fallita!")
            
        # Disconnect
        pm.disconnect()
        print("\n5. Disconnesso da MySQL")
        
    except Exception as e:
        print(f"❌ ERRORE durante il test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mysql_connection()
