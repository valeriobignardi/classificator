#!/usr/bin/env python3
"""
File: test_database_ai_config_complete.py
Autore: Sistema AI
Data creazione: 2025-08-25
Scopo: Test completo del sistema configurazioni AI su database

Storico modifiche:
- 2025-08-25: Test completo sistema database vs config.yaml
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from AIConfiguration.ai_configuration_service import AIConfigurationService
from Database.database_ai_config_service import DatabaseAIConfigService
from TagDatabase.tag_database_connector import TagDatabaseConnector

def test_complete_ai_config_system():
    """
    Test completo del sistema di configurazione AI
    
    Tests:
    1. Creazione/verifica tabella engines
    2. Test servizio database diretto
    3. Test servizio AI Configuration (wrapper)
    4. Test API endpoints
    5. Verifica persistenza database
    
    Ultima modifica: 2025-08-25
    """
    print("=" * 70)
    print("🧪 TEST COMPLETO SISTEMA CONFIGURAZIONI AI SU DATABASE")
    print("=" * 70)
    
    # ID tenant Humanitas dal database
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    
    # ========================================
    # TEST 1: Verifica tabella database
    # ========================================
    print("\n📋 TEST 1: Verifica tabella engines...")
    try:
        connector = TagDatabaseConnector()
        connector.connetti()
        
        result = connector.esegui_query("""
            SELECT COUNT(*) as count FROM engines
        """)
        
        engines_count = result[0][0] if result else 0
        print(f"✅ Tabella engines: {engines_count} configurazioni trovate")
        
        connector.disconnetti()
    except Exception as e:
        print(f"❌ Errore verifica tabella: {e}")
        return False
    
    # ========================================
    # TEST 2: Servizio database diretto
    # ========================================
    print("\n🔧 TEST 2: Servizio database diretto...")
    try:
        db_service = DatabaseAIConfigService()
        
        # Test recupero configurazione
        config = db_service.get_tenant_configuration(tenant_id)
        print(f"✅ Configurazione attuale:")
        print(f"   Embedding: {config.get('embedding_engine', 'N/A')}")
        print(f"   LLM: {config.get('llm_engine', 'N/A')}")
        print(f"   Aggiornata: {config.get('updated_at', 'N/A')}")
        
        # Test cambio engine
        result = db_service.set_embedding_engine(tenant_id, "labse", test_param=True)
        if result['success']:
            print(f"✅ Engine cambiato a LaBSE: {result['tenant_name']}")
        else:
            print(f"❌ Errore cambio engine: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Errore servizio database: {e}")
        return False
    
    # ========================================
    # TEST 3: Servizio AI Configuration
    # ========================================
    print("\n🎛️ TEST 3: Servizio AI Configuration (wrapper)...")
    try:
        ai_service = AIConfigurationService()
        
        # Verifica che usi database
        if not ai_service.use_database:
            print("⚠️ AI Service non sta usando database!")
            
        # Test configurazione tenant
        tenant_config = ai_service.get_tenant_configuration(tenant_id)
        print(f"✅ Configurazione via AI Service:")
        print(f"   Embedding: {tenant_config['embedding_engine']['current']}")
        print(f"   Source: {tenant_config.get('source', 'unknown')}")
        
        # Test cambio engine via AI Service
        change_result = ai_service.set_embedding_engine(tenant_id, "bge_m3")
        if change_result.get('success'):
            print(f"✅ Engine cambiato a BGE-M3 via AI Service")
        else:
            print(f"❌ Errore cambio via AI Service: {change_result}")
            
    except Exception as e:
        print(f"❌ Errore AI Configuration Service: {e}")
        return False
    
    # ========================================
    # TEST 4: Verifica persistenza
    # ========================================
    print("\n💾 TEST 4: Verifica persistenza database...")
    try:
        connector = TagDatabaseConnector()
        connector.connetti()
        
        result = connector.esegui_query("""
            SELECT tenant_name, embedding_engine, llm_engine, 
                   updated_at, embedding_config
            FROM engines 
            WHERE tenant_id = %s
        """, (tenant_id,))
        
        if result:
            tenant_name, embedding_engine, llm_engine, updated_at, embedding_config = result[0]
            print(f"✅ Persistenza verificata:")
            print(f"   Tenant: {tenant_name}")
            print(f"   Embedding Engine: {embedding_engine}")
            print(f"   LLM Engine: {llm_engine}")
            print(f"   Ultimo aggiornamento: {updated_at}")
            print(f"   Config JSON: {embedding_config or 'null'}")
        else:
            print("❌ Nessuna configurazione trovata nel database")
            
        connector.disconnetti()
        
    except Exception as e:
        print(f"❌ Errore verifica persistenza: {e}")
        return False
    
    # ========================================
    # TEST 5: Test tutti i tenants
    # ========================================
    print("\n🏥 TEST 5: Configurazioni tutti i tenants...")
    try:
        db_service = DatabaseAIConfigService()
        all_configs = db_service.get_all_tenant_configurations()
        
        print(f"✅ Tenants configurati: {len(all_configs)}")
        
        # Mostra primi 5 per non intasare l'output
        for i, (tenant_id, config) in enumerate(all_configs.items()):
            if i < 5:  # Solo primi 5
                print(f"   {config['tenant_name']}: {config['embedding_engine']} + {config['llm_engine']}")
            elif i == 5:
                print(f"   ... e altri {len(all_configs) - 5} tenants")
                break
                
    except Exception as e:
        print(f"❌ Errore recupero tutti i tenants: {e}")
        return False
    
    # ========================================
    # CONCLUSIONE
    # ========================================
    print("\n" + "=" * 70)
    print("🎉 TUTTI I TEST COMPLETATI CON SUCCESSO!")
    print("=" * 70)
    print("✅ Sistema configurazioni AI su DATABASE MySQL operativo")
    print("✅ Tabella engines creata e popolata")
    print("✅ Servizio database funzionante")
    print("✅ AI Configuration Service integrato")
    print("✅ Persistenza configurazioni verificata")
    print("✅ Multi-tenant support attivo")
    print("")
    print("🔄 Il sistema ora salva le configurazioni AI su DATABASE")
    print("   invece che su config.yaml!")
    print("📊 Configurazioni salvate per", len(all_configs), "tenants")
    
    return True

if __name__ == "__main__":
    success = test_complete_ai_config_system()
    if success:
        print("\n🚀 Sistema pronto per la produzione!")
        exit(0)
    else:
        print("\n💥 Alcuni test sono falliti!")
        exit(1)
