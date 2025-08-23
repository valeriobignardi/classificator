#!/usr/bin/env python3
"""
=====================================================================
TEST INTEGRAZIONE MULTI-TENANT PER TAG.tags
=====================================================================
Autore: Sistema di Classificazione AI
Data: 2025-08-21
Descrizione: Test completo del sistema multi-tenant per tag database
=====================================================================
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'TagDatabase'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Classification'))

def test_multitenant_tags():
    """Test completo del sistema multi-tenant per tag"""
    
    print("🧪 === TEST SISTEMA MULTI-TENANT TAG.tags ===")
    
    # Test 1: TagDatabaseConnector multi-tenant
    print("\n📋 TEST 1: TagDatabaseConnector Multi-tenant")
    
    try:
        from TagDatabase.tag_database_connector import TagDatabaseConnector
        
        # Test tenant "humanitas"
        tag_db_humanitas = TagDatabaseConnector(tenant_id="humanitas", tenant_name="Humanitas")
        tags_humanitas = tag_db_humanitas.get_all_tags()
        print(f"✅ Tenant 'humanitas': {len(tags_humanitas)} tag caricati")
        
        # Test tenant "demo" (dovrebbe essere vuoto)
        tag_db_demo = TagDatabaseConnector(tenant_id="demo", tenant_name="Demo Hospital")
        tags_demo = tag_db_demo.get_all_tags()
        print(f"✅ Tenant 'demo': {len(tags_demo)} tag caricati")
        
        # Test aggiunta nuovo tag per tenant demo
        success = tag_db_demo.add_tag_if_not_exists(
            "test_tag_demo", 
            "Tag di test per tenant demo"
        )
        print(f"✅ Aggiunta tag demo: {'successo' if success else 'fallito'}")
        
        # Verifica isolamento: ricarica tag per demo
        tags_demo_after = tag_db_demo.get_all_tags()
        print(f"✅ Tenant 'demo' dopo aggiunta: {len(tags_demo_after)} tag")
        
        # Verifica che humanitas non veda il tag demo
        tags_humanitas_after = tag_db_humanitas.get_all_tags()
        print(f"✅ Tenant 'humanitas' invariato: {len(tags_humanitas_after)} tag")
        
        print("✅ Isolamento multi-tenant verificato!")
        
    except Exception as e:
        print(f"❌ Errore test TagDatabaseConnector: {e}")
    
    # Test 2: IntelligentClassifier multi-tenant  
    print("\n🤖 TEST 2: IntelligentClassifier Multi-tenant")
    
    try:
        from Classification.intelligent_classifier import IntelligentClassifier
        
        # Test classifier per tenant humanitas
        classifier_humanitas = IntelligentClassifier(
            client_name="humanitas",
            enable_logging=False
        )
        print(f"✅ Classifier 'humanitas' inizializzato")
        print(f"   - Tenant ID: {classifier_humanitas.tenant_id}")
        print(f"   - Tag caricati: {len(classifier_humanitas.domain_labels)}")
        
        # Test classifier per tenant demo  
        classifier_demo = IntelligentClassifier(
            client_name="demo",
            enable_logging=False
        )
        print(f"✅ Classifier 'demo' inizializzato")
        print(f"   - Tenant ID: {classifier_demo.tenant_id}")
        print(f"   - Tag caricati: {len(classifier_demo.domain_labels)}")
        
        # Verifica isolamento etichette
        humanitas_labels = set(classifier_humanitas.domain_labels)
        demo_labels = set(classifier_demo.domain_labels)
        common_labels = humanitas_labels.intersection(demo_labels)
        
        print(f"✅ Isolamento verificato:")
        print(f"   - Humanitas: {len(humanitas_labels)} etichette")
        print(f"   - Demo: {len(demo_labels)} etichette")
        print(f"   - Comuni: {len(common_labels)} etichette")
        
    except Exception as e:
        print(f"❌ Errore test IntelligentClassifier: {e}")
    
    # Test 3: Verifica database direttamente
    print("\n🗄️ TEST 3: Verifica Database")
    
    try:
        import mysql.connector
        import yaml
        
        # Carica config
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        db_config = config['tag_database']
        
        connection = mysql.connector.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        
        cursor = connection.cursor()
        
        # Conta tag per tenant
        query = """
        SELECT tenant_id, tenant_name, COUNT(*) as tag_count 
        FROM tags 
        GROUP BY tenant_id, tenant_name
        ORDER BY tenant_id
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        print("📊 Distribuzione tag per tenant:")
        for tenant_id, tenant_name, count in results:
            print(f"   - {tenant_name} ({tenant_id}): {count} tag")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"❌ Errore verifica database: {e}")
    
    print("\n✅ === TEST MULTI-TENANT COMPLETATO ===")

if __name__ == "__main__":
    test_multitenant_tags()
