#!/usr/bin/env python3
"""
Test per la nuova funzionalità di aggiunta casi come esempi LLM

Autore: Valerio Bignardi
Data: 2025-09-07
Descrizione: Script di test per verificare il corretto funzionamento
             del nuovo metodo aggiungi_caso_come_esempio_llm
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Pipeline.end_to_end_pipeline import EndToEndPipeline
try:
    from Utils.tenant_manager import TenantManager
except ImportError:
    # Fallback al mock se il vero TenantManager non è disponibile
    from Utils.mock_tenant_manager import TenantManager
from typing import Dict, Any
import json


def test_aggiungi_caso_come_esempio_llm():
    """
    Testa la funzionalità di aggiunta casi come esempi LLM
    
    Scopo: Verificare che il nuovo metodo funzioni correttamente
    Input: Dati di test simulati
    Output: Risultato dell'operazione di test
    Data ultima modifica: 2025-09-07
    """
    print("🧪 [TEST ESEMPI LLM] Avvio test...")
    
    try:
        # 1. Carica configurazione
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            print(f"❌ File configurazione non trovato: {config_path}")
            return False
        
        # 2. Carica tenant di test
        tenant_manager = TenantManager(config_path)
        tenants = tenant_manager.get_all_tenants()
        
        if not tenants:
            print("❌ Nessun tenant trovato per i test")
            return False
        
        # Usa il primo tenant disponibile
        test_tenant = tenants[0]
        print(f"   🏢 Tenant test: {test_tenant.tenant_name}")
        
        # 3. Inizializza pipeline
        print(f"   🔧 Inizializzazione pipeline...")
        pipeline = EndToEndPipeline(
            config_path=config_path,
            tenant=test_tenant
        )
        
        # 4. Dati di test
        test_cases = [
            {
                "session_id": "test_session_001",
                "conversation_text": "Ciao, ho bisogno di prenotare una visita cardiologica urgente",
                "etichetta_corretta": "PRENOTAZIONE_VISITA",
                "categoria": "CARDIOLOGIA",
                "note_utente": "L'utente ha specificato che è urgente, quindi è chiaramente una prenotazione"
            },
            {
                "session_id": "test_session_002", 
                "conversation_text": "Buongiorno, volevo informazioni sui prezzi delle analisi del sangue",
                "etichetta_corretta": "RICHIESTA_INFORMAZIONI",
                "categoria": "ANALISI",
                "note_utente": "Richiesta esplicita di informazioni sui prezzi, non vuole prenotare"
            },
            {
                "session_id": "test_session_003",
                "conversation_text": "Il mio appuntamento di oggi è stato cancellato, quando posso riprogrammarlo?",
                "etichetta_corretta": "GESTIONE_APPUNTAMENTI",
                "categoria": None,  # Test categoria automatica
                "note_utente": None  # Test senza note
            }
        ]
        
        # 5. Esegui test per ogni caso
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\n   📝 Test caso {i+1}/{len(test_cases)}")
            print(f"      Session: {test_case['session_id']}")
            print(f"      Etichetta: {test_case['etichetta_corretta']}")
            
            # Chiama il nuovo metodo
            result = pipeline.aggiungi_caso_come_esempio_llm(
                session_id=test_case["session_id"],
                conversation_text=test_case["conversation_text"],
                etichetta_corretta=test_case["etichetta_corretta"],
                categoria=test_case["categoria"],
                note_utente=test_case["note_utente"]
            )
            
            results.append({
                "test_case": test_case,
                "result": result
            })
            
            if result['success']:
                print(f"      ✅ Successo - ID esempio: {result['esempio_id']}")
            else:
                print(f"      ❌ Errore: {result['message']}")
        
        # 6. Riepilogo risultati
        print(f"\n📊 [RIEPILOGO TEST]")
        successi = sum(1 for r in results if r['result']['success'])
        fallimenti = len(results) - successi
        
        print(f"   ✅ Successi: {successi}/{len(results)}")
        print(f"   ❌ Fallimenti: {fallimenti}/{len(results)}")
        
        # Dettagli fallimenti
        if fallimenti > 0:
            print(f"\n❌ [DETTAGLI FALLIMENTI]")
            for i, r in enumerate(results):
                if not r['result']['success']:
                    print(f"   Caso {i+1}: {r['result']['message']}")
        
        # 7. Salva risultati dettagliati
        results_file = "test_esempi_llm_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n   💾 Risultati salvati in: {results_file}")
        
        # Test considerato passato se almeno il 50% dei casi funziona
        success_rate = successi / len(results)
        test_passed = success_rate >= 0.5
        
        if test_passed:
            print(f"✅ [TEST COMPLETATO] Test superato con {success_rate:.1%} successi")
        else:
            print(f"❌ [TEST FALLITO] Solo {success_rate:.1%} successi")
        
        return test_passed
        
    except Exception as e:
        print(f"❌ [TEST ERROR] Errore durante test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_manual_retrain_model():
    """
    Testa la funzionalità di riaddestramento manuale del modello
    
    Scopo: Verificare che il riaddestramento funzioni correttamente
    Input: Nessuno (usa dati esistenti in MongoDB)
    Output: Risultato del riaddestramento
    Data ultima modifica: 2025-09-07
    """
    print("\n🔄 [TEST RIADDESTRAMENTO] Avvio test riaddestramento...")
    
    try:
        # 1. Carica configurazione
        config_path = "config.yaml"
        tenant_manager = TenantManager(config_path)
        tenants = tenant_manager.get_all_tenants()
        
        if not tenants:
            print("❌ Nessun tenant trovato per i test")
            return False
        
        test_tenant = tenants[0]
        print(f"   🏢 Tenant test: {test_tenant.tenant_name}")
        
        # 2. Inizializza pipeline
        pipeline = EndToEndPipeline(
            config_path=config_path,
            tenant=test_tenant
        )
        
        # 3. Test riaddestramento (solo verifica API, non force)
        print(f"   🎓 Avvio riaddestramento di test...")
        result = pipeline.manual_retrain_model(force=False)
        
        # 4. Analizza risultato
        if result['success']:
            accuracy = result['accuracy']
            training_stats = result['training_stats']
            
            print(f"   ✅ Riaddestramento completato")
            print(f"      📊 Accuracy: {accuracy:.3f}")
            print(f"      📚 Samples: {training_stats.get('training_samples', 'N/A')}")
            print(f"      ⏱️ Tempo: {training_stats.get('processing_time', 'N/A'):.2f}s")
            
            return True
        else:
            print(f"   ❌ Riaddestramento fallito: {result['message']}")
            return False
        
    except Exception as e:
        print(f"❌ [TEST RIADDESTRAMENTO ERROR] Errore: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """
    Main entry point per i test
    """
    print("🚀 [AVVIO TEST SUITE] Test nuove funzionalità esempi LLM")
    print("=" * 60)
    
    # Test 1: Aggiunta esempi LLM
    test1_passed = test_aggiungi_caso_come_esempio_llm()
    
    # Test 2: Riaddestramento manuale (solo se test 1 è passato)
    test2_passed = False
    if test1_passed:
        test2_passed = test_manual_retrain_model()
    else:
        print("\n⏭️ [SKIP] Test riaddestramento saltato (test esempi fallito)")
    
    # Riepilogo finale
    print("\n" + "=" * 60)
    print("📋 [RIEPILOGO FINALE]")
    print(f"   Test Esempi LLM: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Test Riaddestramento: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    overall_success = test1_passed and test2_passed
    
    if overall_success:
        print(f"🎉 [SUCCESSO] Tutti i test sono passati!")
        sys.exit(0)
    else:
        print(f"💥 [FALLIMENTO] Alcuni test sono falliti")
        sys.exit(1)
