#!/usr/bin/env python3
"""
Script per sincronizzare i tenants dal database remoto alla cache locale
"""

import sys
import os

# Aggiunge i percorsi necessari
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MySql'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from connettore import MySqlConnettore
from tag_manager import TagDatabaseManager

def sincronizza_tenants():
    """Sincronizza i tenants dal database remoto alla cache locale"""
    print("=== SINCRONIZZAZIONE TENANTS ===")
    
    # Connettore per il database remoto (tenants)
    connettore_remoto = MySqlConnettore()
    
    # Manager per il database locale (tag)
    tag_manager = TagDatabaseManager()
    
    try:
        print("🔍 Lettura tenants dal database remoto...")
        
        # Legge tutti i tenants dal database remoto
        query_tenants = """
        SELECT tenant_id, tenant_name, tenant_slug 
        FROM common.tenants 
        WHERE is_active = 1
        ORDER BY tenant_name
        """
        
        tenants = connettore_remoto.esegui_query(query_tenants)
        
        if tenants:
            print(f"✅ Trovati {len(tenants)} tenants attivi")
            
            # Mostra i tenants trovati
            print("\n📋 Tenants trovati:")
            for tenant in tenants:
                tenant_id, tenant_name, tenant_slug = tenant
                print(f"  - {tenant_name} ({tenant_slug}) [ID: {tenant_id}]")
            
            # Aggiorna la cache locale
            print(f"\n🔄 Aggiornamento cache locale...")
            if tag_manager.crea_database_e_tabelle():
                if tag_manager.aggiorna_cache_tenants(tenants):
                    print("✅ Cache tenants sincronizzata con successo!")
                    
                    # Verifica che Humanitas sia presente
                    humanitas_id = tag_manager.ottieni_tenant_id('humanitas')
                    if humanitas_id:
                        print(f"✅ Humanitas trovato con ID: {humanitas_id}")
                    else:
                        print("⚠️  Humanitas non trovato nella cache")
                else:
                    print("❌ Errore durante l'aggiornamento della cache")
            else:
                print("❌ Errore durante la creazione delle tabelle")
        else:
            print("❌ Nessun tenant trovato nel database remoto")
    
    except Exception as e:
        print(f"❌ Errore durante la sincronizzazione: {e}")
    
    finally:
        connettore_remoto.disconnetti()
        tag_manager.disconnetti()

if __name__ == "__main__":
    sincronizza_tenants()
