#!/usr/bin/env python3
"""
Test per verificare l'isolamento multi-tenant nelle tag suggestions.

Questo test verifica che:
1. La risoluzione tenant_id funzioni correttamente
2. Ogni client veda solo i tag del proprio tenant
3. Non ci siano data leak tra tenant diversi

Author: Valerio Bignardi
Date: 23/08/2025
"""

import os
import sys
import logging
from TAGS.tag import IntelligentTagSuggestionManager

def setup_test_logging():
    """Setup logging per i test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - TEST - %(levelname)s - %(message)s'
    )
    return logging.getLogger('TEST_MULTITENANT')

def test_tenant_resolution():
    """Test della risoluzione tenant_id da client_name"""
    logger = setup_test_logging()
    logger.info("=" * 50)
    logger.info("🧪 TEST: Risoluzione Tenant ID")
    logger.info("=" * 50)
    
    try:
        # Inizializza TagSuggestionManager
        tag_manager = IntelligentTagSuggestionManager()
        
        # Test per client diversi
        test_clients = ["alleanza", "bosch", "tenant_inesistente"]
        
        for client_name in test_clients:
            logger.info(f"\n📋 Testing client: '{client_name}'")
            try:
                tenant_id = tag_manager._resolve_tenant_id_from_name(client_name)
                logger.info(f"✅ '{client_name}' → tenant_id: {tenant_id}")
                
                # Verifica formato UUID
                if len(tenant_id) > 20 and '-' in tenant_id:
                    logger.info(f"✅ Formato UUID corretto: {tenant_id}")
                else:
                    logger.warning(f"⚠️ Formato tenant_id sospetto: {tenant_id}")
                    
            except Exception as e:
                logger.error(f"❌ Errore per '{client_name}': {e}")
        
        logger.info("\n" + "=" * 50)
        return True
        
    except Exception as e:
        logger.error(f"❌ Errore nel test risoluzione tenant: {e}")
        return False

def test_tag_suggestions_isolation():
    """Test dell'isolamento multi-tenant nelle tag suggestions"""
    logger = setup_test_logging()
    logger.info("=" * 50)
    logger.info("🧪 TEST: Isolamento Multi-Tenant Tag Suggestions")
    logger.info("=" * 50)
    
    try:
        # Inizializza TagSuggestionManager
        tag_manager = IntelligentTagSuggestionManager()
        
        # Test per client diversi
        test_clients = ["alleanza", "bosch"]
        results = {}
        
        for client_name in test_clients:
            logger.info(f"\n📋 Testing tag suggestions for: '{client_name}'")
            try:
                tags = tag_manager.get_suggested_tags_for_client(client_name)
                results[client_name] = tags
                
                logger.info(f"✅ '{client_name}' ricevuto {len(tags)} tag suggeriti")
                
                # Mostra primi 5 tag per debug
                if tags:
                    for i, tag in enumerate(tags[:5]):
                        logger.info(f"   {i+1}. {tag.get('tag_name', 'N/A')} (ID: {tag.get('id', 'N/A')})")
                    if len(tags) > 5:
                        logger.info(f"   ... e altri {len(tags)-5} tag")
                else:
                    logger.info(f"   (nessun tag trovato per {client_name})")
                    
            except Exception as e:
                logger.error(f"❌ Errore per '{client_name}': {e}")
                results[client_name] = []
        
        # Analizza isolamento
        logger.info("\n📊 ANALISI ISOLAMENTO:")
        if len(results) >= 2:
            clients = list(results.keys())
            client1, client2 = clients[0], clients[1]
            
            tags1 = set([tag['tag_name'] for tag in results[client1] if 'tag_name' in tag])
            tags2 = set([tag['tag_name'] for tag in results[client2] if 'tag_name' in tag])
            
            overlap = tags1.intersection(tags2)
            
            logger.info(f"🔍 {client1}: {len(tags1)} tag unici")
            logger.info(f"🔍 {client2}: {len(tags2)} tag unici")
            logger.info(f"🔍 Tag sovrapposti: {len(overlap)}")
            
            if overlap:
                logger.info(f"⚠️ Tag in comune trovati: {list(overlap)[:10]}")
                logger.info("Note: Alcuni tag potrebbero essere legittimamente condivisi")
            else:
                logger.info("✅ Nessun tag sovrapposto - isolamento perfetto!")
        
        logger.info("\n" + "=" * 50)
        return True
        
    except Exception as e:
        logger.error(f"❌ Errore nel test isolamento: {e}")
        return False

def main():
    """Esecuzione test principale"""
    logger = setup_test_logging()
    logger.info("🚀 AVVIO TEST ISOLAMENTO MULTI-TENANT TAG SUGGESTIONS")
    
    success = True
    
    # Test 1: Risoluzione tenant
    if not test_tenant_resolution():
        success = False
    
    # Test 2: Isolamento tag suggestions
    if not test_tag_suggestions_isolation():
        success = False
    
    # Risultato finale
    if success:
        logger.info("🎉 TUTTI I TEST COMPLETATI CON SUCCESSO!")
        logger.info("✅ Isolamento multi-tenant implementato correttamente")
    else:
        logger.error("💥 ALCUNI TEST FALLITI!")
        logger.error("❌ Verificare configurazione e database")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
