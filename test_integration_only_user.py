#!/usr/bin/env python3
"""
Test end-to-end per verificare l'integrazione del parametro only_user
Autore: Sistema di IA  
Data: 25/08/2025
"""

import sys
import os

# Configura il percorso per importare i moduli
sys.path.append('/home/ubuntu/classificazione_discussioni_bck_23_08_2025')

from Preprocessing.session_aggregator import SessionAggregator
from Pipeline.end_to_end_pipeline import EndToEndPipeline
import yaml

def test_pipeline_integration():
    """
    Test di integrazione della pipeline con il parametro only_user
    """
    print("=== TEST INTEGRAZIONE PIPELINE ===\n")
    
    try:
        # Crea una pipeline per test con il parametro corretto
        pipeline = EndToEndPipeline(tenant_slug='humanitas')
        
        print("🚀 Pipeline inizializzata")
        print(f"🎯 Aggregator only_user: {pipeline.aggregator.only_user}")
        
        # Test estrazione sessioni con limite ridotto per velocità
        print("📊 Test estrazione sessioni dalla pipeline...")
        sessioni = pipeline.estrai_sessioni(limit=5, giorni_indietro=30)
        
        if sessioni:
            print(f"✅ Pipeline ha estratto {len(sessioni)} sessioni")
            
            # Analizza i contenuti per verificare il filtraggio
            total_user_msgs = 0
            total_agent_msgs = 0
            
            for session_id, dati in sessioni.items():
                total_user_msgs += dati['num_messaggi_user']
                total_agent_msgs += dati['num_messaggi_agent']
            
            print(f"📊 Messaggi USER totali: {total_user_msgs}")
            print(f"📊 Messaggi AGENT totali: {total_agent_msgs}")
            
            # Con only_user=True dovremmo avere solo messaggi USER
            if pipeline.aggregator.only_user and total_agent_msgs > 0:
                print(f"❌ ERROR: Con only_user=True non dovremmo avere messaggi AGENT!")
                return False
            
            # Mostra un esempio di testo completo
            first_session = list(sessioni.values())[0]
            print(f"\n📝 Esempio testo completo sessione:")
            print(f"   {first_session['testo_completo'][:200]}...")
            
            # Verifica che il testo non contenga [ASSISTENTE]
            if pipeline.aggregator.only_user and "[ASSISTENTE]" in first_session['testo_completo']:
                print(f"❌ ERROR: Con only_user=True il testo non dovrebbe contenere [ASSISTENTE]!")
                return False
            
            print("✅ Pipeline integrata correttamente con only_user")
            
        else:
            print("⚠️ Nessuna sessione estratta dalla pipeline")
        
        # Cleanup
        pipeline.chiudi_connessioni()
        return True
        
    except Exception as e:
        print(f"❌ Errore nel test pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_modes():
    """
    Test per verificare entrambe le modalità: only_user=True e only_user=False
    """
    print("\n=== TEST MODALITÀ CONFIGURAZIONE ===\n")
    
    # Prima salva la configurazione attuale
    config_path = '/home/ubuntu/classificazione_discussioni_bck_23_08_2025/config.yaml'
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            original_config = yaml.safe_load(f)
        
        original_only_user = original_config.get('conversation_reading', {}).get('only_user', False)
        print(f"📋 Configurazione originale only_user: {original_only_user}")
        
        # Test modalità only_user=False
        print("\n🔄 Testing only_user=False...")
        test_config = original_config.copy()
        test_config['conversation_reading']['only_user'] = False
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f, default_flow_style=False, allow_unicode=True)
        
        # IMPORTANTE: Aspetta un momento per assicurare che il file sia scritto
        import time
        time.sleep(0.1)
        
        # Test aggregator con only_user=False
        aggregator_false = SessionAggregator(schema='humanitas')
        print(f"🎯 Aggregator only_user: {aggregator_false.only_user}")
        
        if aggregator_false.only_user != False:
            print("❌ ERROR: only_user dovrebbe essere False!")
            return False
        
        # Test una sessione
        sessioni_false = aggregator_false.estrai_sessioni_aggregate(limit=1)
        user_msgs_false = sum(s['num_messaggi_user'] for s in sessioni_false.values())
        agent_msgs_false = sum(s['num_messaggi_agent'] for s in sessioni_false.values())
        
        print(f"📊 Con only_user=False: USER={user_msgs_false}, AGENT={agent_msgs_false}")
        
        aggregator_false.chiudi_connessione()
        
        # Ripristina configurazione originale
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(original_config, f, default_flow_style=False, allow_unicode=True)
        
        # IMPORTANTE: Aspetta un momento per assicurare che il file sia scritto
        import time
        time.sleep(0.1)
        
        # Test modalità only_user=True
        print("\n🔄 Testing only_user=True...")
        aggregator_true = SessionAggregator(schema='humanitas')
        print(f"🎯 Aggregator only_user: {aggregator_true.only_user}")
        
        if aggregator_true.only_user != True:
            print("❌ ERROR: only_user dovrebbe essere True!")
            return False
        
        # Test una sessione
        sessioni_true = aggregator_true.estrai_sessioni_aggregate(limit=1)
        user_msgs_true = sum(s['num_messaggi_user'] for s in sessioni_true.values())
        agent_msgs_true = sum(s['num_messaggi_agent'] for s in sessioni_true.values())
        
        print(f"📊 Con only_user=True: USER={user_msgs_true}, AGENT={agent_msgs_true}")
        
        # Verifica che con only_user=True abbiamo 0 messaggi AGENT
        if agent_msgs_true > 0:
            print("❌ ERROR: Con only_user=True non dovremmo avere messaggi AGENT!")
            return False
        
        aggregator_true.chiudi_connessione()
        
        print("✅ Entrambe le modalità funzionano correttamente")
        return True
        
    except Exception as e:
        print(f"❌ Errore nel test modalità: {e}")
        # Ripristina configurazione in caso di errore
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(original_config, f, default_flow_style=False, allow_unicode=True)
        except:
            pass
        return False

def main():
    """
    Esegue tutti i test di integrazione
    """
    print("🚀 AVVIO TEST INTEGRAZIONE ONLY_USER\n")
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Integrazione Pipeline
    if test_pipeline_integration():
        success_count += 1
    
    # Test 2: Test modalità configurazione
    if test_configuration_modes():
        success_count += 1
    
    print(f"\n🎯 RISULTATI FINALI INTEGRAZIONE:")
    print(f"   ✅ Test superati: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🏆 INTEGRAZIONE ONLY_USER COMPLETATA CON SUCCESSO!")
        print("📋 Il parametro only_user funziona correttamente in tutte le configurazioni.")
        return True
    else:
        print("❌ Alcuni test di integrazione hanno fallito.")
        return False

if __name__ == "__main__":
    main()
