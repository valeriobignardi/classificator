#!/usr/bin/env python3
"""
Test script per verificare il funzionamento del tenant_config_helper
"""

import sys
import os

# Aggiungi il percorso del progetto
sys.path.append('/home/ubuntu/classificatore/Utils')

try:
    from tenant_config_helper import get_only_user_for_tenant, TenantConfigHelper
    
    print("✅ Import dell'helper riuscito")
    
    # Test con tenant humanitas
    tenant_id = "humanitas"  
    result = get_only_user_for_tenant(tenant_id)
    print(f"🎯 Test only_user per tenant '{tenant_id}': {result}")
    
    # Test con tenant inesistente
    tenant_id = "test_tenant"
    result = get_only_user_for_tenant(tenant_id)  
    print(f"🎯 Test only_user per tenant inesistente '{tenant_id}': {result}")
    
    # Test helper diretto
    helper = TenantConfigHelper()
    print("✅ Helper inizializzato correttamente")
    
    print("✅ Tutti i test dell'helper completati")
    
except Exception as e:
    print(f"❌ Errore durante il test: {e}")
    import traceback
    traceback.print_exc()
