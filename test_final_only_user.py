#!/usr/bin/env python3
"""
Test semplificato per verificare il funzionamento del parametro only_user
Autore: Sistema di IA
Data: 25/08/2025
"""

import sys
import os

# Configura il percorso per importare i moduli
sys.path.append('/home/ubuntu/classificazione_discussioni_bck_23_08_2025')

from Preprocessing.session_aggregator import SessionAggregator
import yaml

def test_final_only_user():
    """
    Test finale per verificare che only_user=True funzioni correttamente
    """
    print("=== TEST FINALE ONLY_USER ===\n")
    
    config_path = '/home/ubuntu/classificazione_discussioni_bck_23_08_2025/config.yaml'
    
    # Verifica la configurazione attuale
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    current_only_user = config.get('conversation_reading', {}).get('only_user', False)
    print(f"📋 Configurazione attuale only_user: {current_only_user}")
    
    if not current_only_user:
        print("❌ ERROR: La configurazione deve avere only_user: true per questo test!")
        return False
    
    # Test SessionAggregator con only_user=True
    print("\n🔧 Test SessionAggregator con only_user=True...")
    
    try:
        aggregator = SessionAggregator(schema='humanitas')
        print(f"🎯 Aggregator configurato: only_user = {aggregator.only_user}")
        
        if not aggregator.only_user:
            print("❌ ERROR: SessionAggregator dovrebbe avere only_user=True!")
            return False
        
        # Test con un limite piccolo per velocità
        print("📊 Test aggregazione con limite 2...")
        sessioni = aggregator.estrai_sessioni_aggregate(limit=2)
        
        if not sessioni:
            print("⚠️ Nessuna sessione trovata")
            return False
        
        print(f"✅ Aggregate {len(sessioni)} sessioni")
        
        # Verifica che tutte le sessioni abbiano solo messaggi USER
        total_user = 0
        total_agent = 0
        all_said_by = set()
        
        for session_id, dati in sessioni.items():
            total_user += dati['num_messaggi_user']
            total_agent += dati['num_messaggi_agent']
            
            for msg in dati['messaggi']:
                all_said_by.add(msg['said_by'])
        
        print(f"📊 Totale messaggi USER: {total_user}")
        print(f"📊 Totale messaggi AGENT: {total_agent}")
        print(f"🔍 Tipi said_by presenti: {all_said_by}")
        
        # Verifiche critiche per only_user=True
        success = True
        
        if total_agent > 0:
            print("❌ ERROR: Con only_user=True non dovremmo avere messaggi AGENT!")
            success = False
        
        if all_said_by != {'USER'}:
            print(f"❌ ERROR: Con only_user=True dovremmo avere solo 'USER', ma abbiamo: {all_said_by}")
            success = False
        
        if total_user == 0:
            print("❌ ERROR: Dovremmo avere almeno alcuni messaggi USER!")
            success = False
        
        # Verifica il contenuto del testo completo
        for session_id, dati in list(sessioni.items())[:1]:  # Solo la prima sessione per test
            testo = dati['testo_completo']
            print(f"\n📝 Esempio testo sessione {session_id}:")
            print(f"   {testo[:150]}{'...' if len(testo) > 150 else ''}")
            
            if "[ASSISTENTE]" in testo:
                print("❌ ERROR: Il testo non dovrebbe contenere [ASSISTENTE] con only_user=True!")
                success = False
            
            if "[UTENTE]" not in testo and testo.strip():
                print("❌ ERROR: Il testo dovrebbe contenere solo [UTENTE]!")
                success = False
        
        aggregator.chiudi_connessione()
        
        if success:
            print("\n🏆 SUCCESS: Il parametro only_user=True funziona perfettamente!")
            print("✅ Solo messaggi USER sono stati inclusi")
            print("✅ Testo completo contiene solo [UTENTE]")
            print("✅ Nessun messaggio AGENT presente")
            return True
        else:
            print("\n❌ FAIL: Il parametro only_user=True non funziona correttamente!")
            return False
        
    except Exception as e:
        print(f"❌ Errore nel test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Esegue il test finale
    """
    print("🚀 AVVIO TEST FINALE PARAMETRO ONLY_USER\n")
    
    if test_final_only_user():
        print("\n🎯 RISULTATO FINALE: ✅ SUCCESS")
        print("📋 Il parametro only_user è implementato correttamente e funziona come richiesto.")
        print("🔧 Il sistema ora può filtrare le conversazioni per includere solo messaggi USER.")
        return True
    else:
        print("\n🎯 RISULTATO FINALE: ❌ FAIL")
        print("📋 Il parametro only_user necessita di ulteriori correzioni.")
        return False

if __name__ == "__main__":
    main()
