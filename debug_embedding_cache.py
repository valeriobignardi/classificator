#!/usr/bin/env python3
"""
Debug script per tracciare il problema della cache degli embedder

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
from AIConfiguration.ai_configuration_service import AIConfigurationService

def test_debug_cache_issue():
    """
    Test per tracciare il problema della cache embedder
    """
    print("🐛 === DEBUG CACHE EMBEDDER ISSUE ===")
    
    tenant_id = "16c222a9-f293-11ef-9315-96000228e7fe"
    tenant_slug = "wopta"
    
    print(f"🎯 Tenant: {tenant_slug} (UUID: {tenant_id})")
    
    # 1. Test DatabaseAIConfigService diretto
    print("\n📊 Step 1: Test DatabaseAIConfigService diretto")
    try:
        db_service = DatabaseAIConfigService()
        
        # Test senza force_no_cache
        print("🔍 Test get_tenant_configuration() normale...")
        config1 = db_service.get_tenant_configuration(tenant_id)
        print(f"✅ Config normale: embedding_engine = {config1.get('embedding_engine', 'NON TROVATO')}")
        
        # Test con force_no_cache
        print("🔍 Test get_tenant_configuration(force_no_cache=True)...")
        config2 = db_service.get_tenant_configuration(tenant_id, force_no_cache=True)
        print(f"✅ Config force_no_cache: embedding_engine = {config2.get('embedding_engine', 'NON TROVATO')}")
        
    except Exception as e:
        print(f"❌ Errore DatabaseAIConfigService: {e}")
        traceback.print_exc()
    
    # 2. Test AIConfigurationService
    print("\n📊 Step 2: Test AIConfigurationService")
    try:
        ai_service = AIConfigurationService()
        
        # Test signature normale
        print("🔍 Test AIConfigurationService.get_tenant_configuration()...")
        ai_config = ai_service.get_tenant_configuration(tenant_id)
        embedding_current = ai_config.get('embedding_engine', {}).get('current', 'NON TROVATO')
        print(f"✅ AIConfig normale: embedding_engine = {embedding_current}")
        
        # Test con force_no_cache (dovrebbe fallire)
        print("🔍 Test AIConfigurationService.get_tenant_configuration(force_no_cache=True)...")
        try:
            ai_config_force = ai_service.get_tenant_configuration(tenant_id, force_no_cache=True)
            print("⚠️  UNEXPECTED: Parametro force_no_cache accettato!")
        except TypeError as te:
            print(f"❌ CONFERMATO: AIConfigurationService NON accetta force_no_cache: {te}")
        
    except Exception as e:
        print(f"❌ Errore AIConfigurationService: {e}")
        traceback.print_exc()
    
    # 3. Test EmbeddingManager
    print("\n📊 Step 3: Test EmbeddingManager")
    try:
        embedding_manager = EmbeddingManager()
        
        print("🔍 Test get_shared_embedder()...")
        current_embedder = embedding_manager.get_shared_embedder(tenant_slug)
        if current_embedder:
            embedder_type = type(current_embedder).__name__
            print(f"✅ Embedder attuale: {embedder_type}")
        else:
            print("❌ Nessun embedder condiviso trovato")
        
        print("🔍 Test force_reload_embedder()...")
        try:
            embedding_manager.force_reload_embedder(tenant_id)
            print("✅ Force reload completato")
        except Exception as fe:
            print(f"❌ ERRORE force_reload: {fe}")
        
        # Controlla embedder dopo reload
        new_embedder = embedding_manager.get_shared_embedder(tenant_slug)
        if new_embedder:
            new_embedder_type = type(new_embedder).__name__
            print(f"✅ Embedder dopo reload: {new_embedder_type}")
        else:
            print("❌ Nessun embedder dopo reload")
            
    except Exception as e:
        print(f"❌ Errore EmbeddingManager: {e}")
        traceback.print_exc()
    
    print("\n🏁 Debug completato!")

if __name__ == "__main__":
    test_debug_cache_issue()
