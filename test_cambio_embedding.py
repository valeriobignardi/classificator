#!/usr/bin/env python3
"""
Test per simulare il cambio di embedding engine via API

Valerio Bignardi
2025-08-25
"""

import sys
import os
import traceback

# Aggiunta path per import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from EmbeddingEngine.embedding_manager import EmbeddingManager
from Database.database_ai_config_service import DatabaseAIConfigService

def test_cambio_embedding_engine():
    """
    Test per simulare il cambio di embedding engine come nell'API
    """
    print("🎯 === TEST CAMBIO EMBEDDING ENGINE ===")
    
    tenant_id = "16c222a9-f293-11ef-9315-96000228e7fe"
    tenant_slug = "wopta"
    
    # 1. Verifica configurazione corrente nel database
    print(f"\n📊 Step 1: Verifica configurazione database per tenant {tenant_slug}")
    try:
        db_service = DatabaseAIConfigService()
        config = db_service.get_tenant_configuration(tenant_id, force_no_cache=True)
        current_engine = config.get('embedding_engine', 'NONE')
        print(f"✅ Configurazione DB corrente: {current_engine}")
    except Exception as e:
        print(f"❌ Errore lettura DB: {e}")
        return
    
    # 2. Test EmbeddingManager normale
    print(f"\n📊 Step 2: Test EmbeddingManager get_shared_embedder normale")
    try:
        embedding_manager = EmbeddingManager()
        embedder1 = embedding_manager.get_shared_embedder(tenant_slug)
        embedder1_type = type(embedder1).__name__
        print(f"✅ Embedder normale: {embedder1_type}")
    except Exception as e:
        print(f"❌ Errore get_shared_embedder normale: {e}")
        return
    
    # 3. Simula cambio configurazione nel database
    print(f"\n📊 Step 3: Simulo cambio da '{current_engine}' a 'labse'")
    try:
        # Simula il salvataggio della nuova configurazione
        result = db_service.set_embedding_engine(
            tenant_id=tenant_id,
            engine_type='labse'
        )
        if result and result.get('success'):
            print(f"✅ Configurazione 'labse' salvata nel database")
        else:
            print(f"❌ Errore salvataggio configurazione: {result}")
            return
    except Exception as e:
        print(f"❌ Errore salvataggio: {e}")
        return
    
    # 4. Verifica che il database sia stato aggiornato
    print(f"\n📊 Step 4: Verifica aggiornamento database")
    try:
        config_after = db_service.get_tenant_configuration(tenant_id, force_no_cache=True)
        new_engine = config_after.get('embedding_engine', 'NONE')
        print(f"✅ Configurazione DB dopo salvataggio: {new_engine}")
        
        if new_engine != 'labse':
            print(f"❌ ERRORE: Database non aggiornato! Dovrebbe essere 'labse' ma è '{new_engine}'")
            return
    except Exception as e:
        print(f"❌ Errore verifica DB: {e}")
        return
    
    # 5. Test switch_tenant_embedder con force_reload=True (come nell'API)
    print(f"\n📊 Step 5: Test switch_tenant_embedder(force_reload=True)")
    try:
        print(f"🔧 Chiamata switch_tenant_embedder({tenant_id}, force_reload=True)...")
        embedder2 = embedding_manager.switch_tenant_embedder(tenant_id, force_reload=True)
        embedder2_type = type(embedder2).__name__
        print(f"✅ Embedder dopo switch forzato: {embedder2_type}")
        
        # Verifica che sia effettivamente cambiato
        if 'LaBSE' in embedder2_type:
            print(f"🎉 SUCCESS! Embedder cambiato correttamente da {embedder1_type} a {embedder2_type}")
        else:
            print(f"❌ FALLIMENTO! Embedder dovrebbe essere LaBSE ma è {embedder2_type}")
            
    except Exception as e:
        print(f"❌ Errore switch_tenant_embedder: {e}")
        traceback.print_exc()
        return
    
    # 6. Verifica get_shared_embedder dopo il cambio
    print(f"\n📊 Step 6: Verifica get_shared_embedder dopo il cambio")
    try:
        embedder3 = embedding_manager.get_shared_embedder(tenant_slug)
        embedder3_type = type(embedder3).__name__
        print(f"✅ Embedder condiviso dopo switch: {embedder3_type}")
        
        if embedder3_type == embedder2_type:
            print(f"✅ COERENZA OK: Embedder condiviso corrisponde a quello forzato")
        else:
            print(f"❌ INCOERENZA: Embedder condiviso ({embedder3_type}) != switch forzato ({embedder2_type})")
            
    except Exception as e:
        print(f"❌ Errore get_shared_embedder finale: {e}")
    
    print(f"\n🏁 Test completato!")

if __name__ == "__main__":
    test_cambio_embedding_engine()
