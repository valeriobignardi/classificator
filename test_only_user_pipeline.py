#!/usr/bin/env python3
"""
Test script per verificare che il parametro only_user funzioni correttamente
nell'intera pipeline di estrazione e clustering
"""

import sys
import os
import yaml

# Aggiungi i percorsi necessari
sys.path.append('/home/ubuntu/classificatore/Utils')
sys.path.append('/home/ubuntu/classificatore/Preprocessing')
sys.path.append('/home/ubuntu/classificatore/LettoreConversazioni')

from tenant_config_helper import get_only_user_for_tenant
from session_aggregator import SessionAggregator
from lettore import LettoreConversazioni

def test_only_user_pipeline():
    """
    Testa l'intera pipeline con only_user=True e only_user=False
    """
    print("🧪 [TEST] Avvio test completo parametro only_user")
    print("="*60)
    
    tenant_id = "humanitas"
    schema = "humanitas"
    limit = 5  # Limitiamo a 5 sessioni per il test
    
    # Test 1: Verifica configurazione tenant
    print("\n🔍 [TEST 1] Verifica configurazione tenant")
    only_user_setting = get_only_user_for_tenant(tenant_id)
    print(f"   Tenant {tenant_id} only_user: {only_user_setting}")
    
    # Test 2: Test LettoreConversazioni diretto
    print("\n🔍 [TEST 2] Test LettoreConversazioni diretto")
    lettore = LettoreConversazioni(schema=schema, tenant_id=tenant_id)
    print(f"   LettoreConversazioni only_user: {lettore.only_user}")
    
    # Legge alcune conversazioni direttamente
    conversazioni = lettore.leggi_conversazioni()
    if conversazioni:
        user_msgs = sum(1 for conv in conversazioni if conv[4] == 'USER')
        agent_msgs = sum(1 for conv in conversazioni if conv[4] == 'AGENT')
        print(f"   Conversazioni totali: {len(conversazioni)}")
        print(f"   Messaggi USER: {user_msgs}")
        print(f"   Messaggi AGENT: {agent_msgs}")
        
        # Verifica coerenza
        if lettore.only_user and agent_msgs > 0:
            print(f"   ❌ ERRORE: only_user=True ma trovati {agent_msgs} messaggi AGENT")
        elif lettore.only_user and agent_msgs == 0:
            print(f"   ✅ CORRETTO: only_user=True e 0 messaggi AGENT")
        else:
            print(f"   ✅ CORRETTO: Configurazione coerente")
    
    # Test 3: Test SessionAggregator completo
    print("\n🔍 [TEST 3] Test SessionAggregator completo")
    aggregator = SessionAggregator(schema=schema, tenant_id=tenant_id)
    print(f"   SessionAggregator only_user: {aggregator.only_user}")
    
    # Estrazione con limite
    sessioni_aggregate = aggregator.estrai_sessioni_aggregate(limit=limit)
    
    if sessioni_aggregate:
        print(f"   Sessioni aggregate: {len(sessioni_aggregate)}")
        
        # Analisi dettagliata delle prime 3 sessioni
        for i, (session_id, dati) in enumerate(list(sessioni_aggregate.items())[:3]):
            print(f"\n   📋 Sessione {i+1} ({session_id}):")
            print(f"      Messaggi USER: {dati['num_messaggi_user']}")
            print(f"      Messaggi AGENT: {dati['num_messaggi_agent']}")
            print(f"      Testo completo (primi 100 char): {dati['testo_completo'][:100]}...")
            
            # Verifica contenuto testo
            utente_tags = dati['testo_completo'].count('[UTENTE]')
            assistente_tags = dati['testo_completo'].count('[ASSISTENTE]')
            print(f"      Tags nel testo: [UTENTE]={utente_tags}, [ASSISTENTE]={assistente_tags}")
            
            if aggregator.only_user and assistente_tags > 0:
                print(f"      ❌ ERRORE: only_user=True ma testo contiene {assistente_tags} tag [ASSISTENTE]")
            elif aggregator.only_user and assistente_tags == 0:
                print(f"      ✅ CORRETTO: only_user=True e 0 tag [ASSISTENTE]")
            else:
                print(f"      ✅ CORRETTO: Configurazione coerente")
    
    print("\n🧪 [TEST] Test completato")
    print("="*60)

def test_modifica_configurazione():
    """
    Modifica la configurazione e testa che il sistema la rilevi
    """
    print("\n🔧 [TEST MODIFICA] Test cambio configurazione")
    
    # Leggi configurazione attuale
    config_file = "/home/ubuntu/classificatore/tenant_configs/humanitas_clustering.yaml"
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    current_only_user = config['clustering_parameters']['only_user']
    print(f"   Configurazione attuale only_user: {current_only_user}")
    
    # Inverti il valore
    new_only_user = not current_only_user
    config['clustering_parameters']['only_user'] = new_only_user
    
    # Salva temporaneamente
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"   Nuova configurazione only_user: {new_only_user}")
    
    # Invalida cache e rileggi
    from tenant_config_helper import get_tenant_config_helper
    helper = get_tenant_config_helper()
    helper.invalidate_cache("humanitas")
    
    # Verifica che il cambiamento sia rilevato
    new_setting = get_only_user_for_tenant("humanitas")
    print(f"   Configurazione riletta: {new_setting}")
    
    if new_setting == new_only_user:
        print("   ✅ CORRETTO: Cambio configurazione rilevato")
    else:
        print("   ❌ ERRORE: Cambio configurazione NON rilevato")
    
    # Ripristina configurazione originale
    config['clustering_parameters']['only_user'] = current_only_user
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    helper.invalidate_cache("humanitas")
    print(f"   Configurazione ripristinata a: {current_only_user}")

if __name__ == "__main__":
    test_only_user_pipeline()
    test_modifica_configurazione()
