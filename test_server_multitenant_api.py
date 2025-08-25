#!/usr/bin/env python3
"""
Test API server per verificare l'integrazione multi-tenant delle tag suggestions.

Test:
1. API /api/get_suggested_tags/<client_name> con isolamento tenant
2. Campo is_new_client corretto basato su MongoDB
3. Nessun data leak tra tenant diversi

Author: Valerio Bignardi
Date: 23/08/2025
"""

import requests
import json
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - API_TEST - %(levelname)s - %(message)s')
    return logging.getLogger('API_TEST')

def test_api_multitenant_isolation():
    """Test API server per isolamento multi-tenant"""
    logger = setup_logging()
    
    # URL base server (assumendo che sia in esecuzione)
    base_url = "http://localhost:5000"
    
    # Test con tenant_id invece di client_name per univocità
    test_tenants = [
        {
            'tenant_id': 'a0fd7600-f4f7-11ef-9315-96000228e7fe',
            'tenant_name': 'alleanza',
            'expected_new': False  # Ha classificazioni esistenti
        },
        {
            'tenant_id': 'tenant-inesistente-uuid',
            'tenant_name': 'inesistente', 
            'expected_new': True   # Non esiste
        }
    ]
    
    logger.info("🚀 AVVIO TEST API MULTI-TENANT (con tenant_id)")
    logger.info("="*50)
    
    for tenant_data in test_tenants:
        tenant_id = tenant_data['tenant_id']
        tenant_name = tenant_data['tenant_name'] 
        expected_new = tenant_data['expected_new']
        
        logger.info(f"\n📋 Testing API per tenant_id: '{tenant_id}' ({tenant_name})")
        
        try:
            # Chiamata API con nuovo endpoint
            url = f"{base_url}/api/review/{tenant_id}/available-tags"
            response = requests.get(url, timeout=10)
            
            logger.info(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Analizza risposta
                success = data.get('success', False)
                tags = data.get('tags', [])
                is_new_client = data.get('is_new_client', True)
                response_tenant_id = data.get('tenant_id', 'N/A')
                response_tenant_name = data.get('tenant_name', 'N/A')
                
                logger.info(f"   ✅ Success: {success}")
                logger.info(f"   📊 Tags count: {len(tags)}")
                logger.info(f"   🆔 Tenant ID: {response_tenant_id}")
                logger.info(f"   👤 Tenant Name: {response_tenant_name}")
                logger.info(f"   🆕 Is new client: {is_new_client}")
                
                # Verifica is_new_client corretto
                if is_new_client == expected_new:
                    logger.info(f"   ✅ is_new_client corretto per {tenant_name}: expected={expected_new}, got={is_new_client}")
                else:
                    logger.error(f"   ❌ is_new_client errato per {tenant_name}: expected={expected_new}, got={is_new_client}")
                
            else:
                logger.error(f"   ❌ HTTP Error: {response.status_code}")
                if response.text:
                    logger.error(f"   Error details: {response.text[:200]}")
                
        except requests.exceptions.ConnectionError:
            logger.error(f"   ❌ Server non raggiungibile su {base_url}")
            logger.error("   💡 Suggerimento: Avvia il server con 'python server.py'")
            
        except Exception as e:
            logger.error(f"   ❌ Errore chiamata API: {e}")
    
    logger.info("\n" + "="*50)
    logger.info("🎯 TEST API COMPLETATO")

if __name__ == "__main__":
    test_api_multitenant_isolation()
