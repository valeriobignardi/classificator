#!/usr/bin/env python3
"""
Test per verificare il funzionamento del parametro only_user
Autore: Sistema di IA
Data: 25/08/2025
"""

import sys
import os

# Configura il percorso per importare i moduli
sys.path.append('/home/ubuntu/classificazione_discussioni_bck_23_08_2025')

from Preprocessing.session_aggregator import SessionAggregator
from LettoreConversazioni.lettore import LettoreConversazioni
import yaml

def test_only_user_configuration():
    """
    Test per verificare che il parametro only_user sia caricato correttamente
    """
    print("=== TEST CONFIGURAZIONE ONLY_USER ===\n")
    
    # Verifica il contenuto di config.yaml
    config_path = '/home/ubuntu/classificazione_discussioni_bck_23_08_2025/config.yaml'
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"📋 Config caricato da: {config_path}")
        
        # Verifica la presenza del parametro
        conversation_reading = config.get('conversation_reading', {})
        only_user = conversation_reading.get('only_user', None)
        
        print(f"📝 Configurazione conversation_reading: {conversation_reading}")
        print(f"🎯 Parametro only_user: {only_user}")
        
        if only_user is None:
            print("❌ ERROR: Parametro only_user non trovato nella configurazione!")
            return False
        
        print("✅ Configurazione caricata correttamente\n")
        return True
        
    except Exception as e:
        print(f"❌ Errore nel caricamento configurazione: {e}")
        return False

def test_lettore_conversazioni():
    """
    Test del LettoreConversazioni con il parametro only_user
    """
    print("=== TEST LETTORE CONVERSAZIONI ===\n")
    
    try:
        lettore = LettoreConversazioni(schema='humanitas')
        print(f"🔧 LettoreConversazioni inizializzato")
        print(f"🎯 Parametro only_user: {lettore.only_user}")
        
        # Test di lettura con limite ridotto
        print("📖 Test lettura conversazioni (limite 5)...")
        risultati = lettore.leggi_conversazioni(limit=5)
        
        if risultati:
            print(f"✅ Trovati {len(risultati)} risultati")
            
            # Analizza i tipi di said_by
            said_by_types = set()
            for riga in risultati:
                said_by = riga[4]  # Posizione del said_by
                said_by_types.add(said_by)
            
            print(f"🔍 Tipi di said_by presenti: {said_by_types}")
            
            # Se only_user è True, dovremmo avere solo 'USER'
            if lettore.only_user and said_by_types != {'USER'}:
                print(f"❌ ERROR: Con only_user=True dovremmo avere solo 'USER', ma abbiamo: {said_by_types}")
                return False
            
            print("✅ Filtraggio said_by corretto")
        else:
            print("⚠️ Nessun risultato trovato")
        
        lettore.chiudi_connessione()
        return True
        
    except Exception as e:
        print(f"❌ Errore nel test LettoreConversazioni: {e}")
        return False

def test_session_aggregator():
    """
    Test del SessionAggregator con il parametro only_user
    """
    print("\n=== TEST SESSION AGGREGATOR ===\n")
    
    try:
        aggregator = SessionAggregator(schema='humanitas')
        print(f"🔧 SessionAggregator inizializzato")
        print(f"🎯 Parametro only_user: {aggregator.only_user}")
        
        # Test aggregazione con limite piccolo
        print("📊 Test aggregazione sessioni (limite 3)...")
        sessioni = aggregator.estrai_sessioni_aggregate(limit=3)
        
        if sessioni:
            print(f"✅ Aggregate {len(sessioni)} sessioni")
            
            # Analizza i tipi di messaggi nelle sessioni
            all_said_by = set()
            for session_id, dati in sessioni.items():
                for msg in dati['messaggi']:
                    all_said_by.add(msg['said_by'])
            
            print(f"🔍 Tipi di said_by nelle sessioni: {all_said_by}")
            
            # Se only_user è True, dovremmo avere solo 'USER'
            if aggregator.only_user and all_said_by != {'USER'}:
                print(f"❌ ERROR: Con only_user=True dovremmo avere solo 'USER', ma abbiamo: {all_said_by}")
                return False
            
            # Mostra esempio di una sessione
            first_session = list(sessioni.values())[0]
            print(f"\n📱 Esempio sessione:")
            print(f"   ID: {first_session['session_id']}")
            print(f"   Messaggi totali: {first_session['num_messaggi_totali']}")
            print(f"   Messaggi USER: {first_session['num_messaggi_user']}")
            print(f"   Messaggi AGENT: {first_session['num_messaggi_agent']}")
            print(f"   Testo completo (primi 150 char): {first_session['testo_completo'][:150]}...")
            
            print("✅ SessionAggregator funziona correttamente")
        else:
            print("⚠️ Nessuna sessione aggregata trovata")
        
        aggregator.chiudi_connessione()
        return True
        
    except Exception as e:
        print(f"❌ Errore nel test SessionAggregator: {e}")
        return False

def main():
    """
    Esegue tutti i test
    """
    print("🚀 AVVIO TEST FILTRAGGIO ONLY_USER\n")
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Configurazione
    if test_only_user_configuration():
        success_count += 1
    
    # Test 2: LettoreConversazioni
    if test_lettore_conversazioni():
        success_count += 1
    
    # Test 3: SessionAggregator
    if test_session_aggregator():
        success_count += 1
    
    print(f"\n🎯 RISULTATI FINALI:")
    print(f"   ✅ Test superati: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🏆 TUTTI I TEST SUPERATI! Il filtraggio only_user funziona correttamente.")
        return True
    else:
        print("❌ Alcuni test hanno fallito. Controllare i messaggi di errore.")
        return False

if __name__ == "__main__":
    main()
