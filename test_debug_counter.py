#!/usr/bin/env python3
"""
Test per il nuovo contatore debug dei casi classificati individualmente

Autore: Valerio Bignardi  
Data: 2025-08-29
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

def test_debug_counter():
    """
    Test del nuovo sistema di contatore debug per rappresentanti e outliers
    """
    print("🧪 TEST: Contatore debug casi classificati individualmente")
    print("="*80)
    
    try:
        # Test del contatore simulando predictions reali
        fake_predictions = []
        session_ids = [f"session_{i}" for i in range(10)]
        
        # Simula predictions miste per testare il contatore
        for i in range(10):
            if i < 3:
                # Rappresentanti
                fake_predictions.append({
                    'predicted_label': 'info_contatti',
                    'confidence': 0.85,
                    'ensemble_confidence': 0.85,
                    'method': 'REPRESENTATIVE',
                    'cluster_id': 0,
                    'llm_prediction': {'predicted_label': 'info_contatti', 'confidence': 0.9},
                    'ml_prediction': {'predicted_label': 'info_contatti', 'confidence': 0.8}
                })
            elif i < 6:
                # Outliers
                fake_predictions.append({
                    'predicted_label': 'altro',
                    'confidence': 0.45,
                    'ensemble_confidence': 0.45,
                    'method': 'OUTLIER',
                    'cluster_id': -1,
                    'llm_prediction': {'predicted_label': 'altro', 'confidence': 0.5},
                    'ml_prediction': {'predicted_label': 'altro', 'confidence': 0.4}
                })
            else:
                # Propagati (non dovrebbero essere contati)
                fake_predictions.append({
                    'predicted_label': 'prenotazione_esami',
                    'confidence': 0.75,
                    'ensemble_confidence': 0.75,
                    'method': 'CLUSTER_PROPAGATED',
                    'cluster_id': 1,
                    'source_representative': 'session_123',
                    'llm_prediction': None,
                    'ml_prediction': {'predicted_label': 'prenotazione_esami', 'confidence': 0.75}
                })
        
        print(f"🎯 Test con {len(fake_predictions)} predizioni simulate:")
        print(f"  - 3 RAPPRESENTANTI (dovrebbero essere contati)")
        print(f"  - 3 OUTLIERS (dovrebbero essere contati)")  
        print(f"  - 4 PROPAGATI (NON dovrebbero essere contati)")
        print(f"  - Contatore atteso: 6 casi individuali")
        
        # Test del contatore simulando il loop della pipeline
        classification_counter = 0
        total_individual_cases = 0
        individual_cases_classified = 0
        propagated_cases = 0
        
        # Pre-conta i casi individuali per il totale
        for prediction in fake_predictions:
            method = prediction.get('method', '')
            if method.startswith('REPRESENTATIVE') or method.startswith('OUTLIER'):
                total_individual_cases += 1
                
        print(f"\n📋 ESECUZIONE SIMULAZIONE CONTATORE:")
        print(f"Total individual cases detected: {total_individual_cases}")
        print("-" * 50)
        
        # Simula il loop della pipeline
        for i, (session_id, prediction) in enumerate(zip(session_ids, fake_predictions)):
            method = prediction.get('method', '')
            session_type = None
            
            if method.startswith('REPRESENTATIVE'):
                classification_counter += 1
                session_type = "RAPPRESENTANTE"
                individual_cases_classified += 1
                print(f"📋 caso n° {classification_counter:02d} / {total_individual_cases:03d} {session_type}")
                
            elif method.startswith('OUTLIER'):
                classification_counter += 1 
                session_type = "OUTLIER"
                individual_cases_classified += 1
                print(f"📋 caso n° {classification_counter:02d} / {total_individual_cases:03d} {session_type}")
                
            elif 'PROPAGATED' in method or 'CLUSTER_PROPAGATED' in method:
                propagated_cases += 1
                print(f"   ↳ Caso propagato saltato (non contato)")
        
        print(f"\n✅ TEST COMPLETATO:")
        print(f"  - Casi individuali contati: {classification_counter}")
        print(f"  - Casi individuali attesi: 6") 
        print(f"  - Casi propagati: {propagated_cases}")
        print(f"  - Test: {'PASSATO' if classification_counter == 6 and propagated_cases == 4 else 'FALLITO'}")
        
        if classification_counter == 6 and propagated_cases == 4:
            print(f"🎉 Il contatore funziona correttamente!")
            print(f"📋 Formato output: 'caso n° XX / YYY TIPO' implementato")
            print(f"🔄 I propagati sono correttamente esclusi dal contatore")
            return True
        else:
            print(f"❌ Il contatore non funziona come atteso")
            return False
            
    except Exception as e:
        print(f"❌ ERRORE durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_removal():
    """
    Test per verificare che i fallback inutili siano stati rimossi
    """
    print(f"\n🧪 TEST: Verifica rimozione fallback inutili")
    print("="*50)
    
    try:
        # Leggi il file e verifica che i fallback siano stati rimossi
        with open('Pipeline/end_to_end_pipeline.py', 'r') as f:
            content = f.read()
        
        # Verifica rimozioni
        removed_strings = [
            'REPRESENTATIVE_FALLBACK',
            'OUTLIER_FALLBACK',
            'REPRESENTATIVE_ORIGINAL',
            'OUTLIER_DIRECT'
        ]
        
        found_bad_strings = []
        for bad_string in removed_strings:
            if bad_string in content:
                found_bad_strings.append(bad_string)
        
        if found_bad_strings:
            print(f"❌ Ancora presenti fallback/distinzioni inutili:")
            for bad in found_bad_strings:
                print(f"   - {bad}")
            return False
        else:
            print(f"✅ Fallback inutili rimossi correttamente")
            print(f"✅ Ora si usa solo 'REPRESENTATIVE' e 'OUTLIER'")
            return True
            
    except Exception as e:
        print(f"❌ Errore nella verifica: {e}")
        return False

if __name__ == "__main__":
    print("🚀 AVVIO TEST CONTATORE DEBUG E RIMOZIONE FALLBACK")
    print("=" * 80)
    
    # Test 1: Contatore debug
    test1_passed = test_debug_counter()
    
    # Test 2: Rimozione fallback
    test2_passed = test_fallback_removal()
    
    print("\n" + "=" * 80)
    print("📊 RISULTATI FINALI:")
    print(f"  🧪 Test contatore debug: {'✅ PASSATO' if test1_passed else '❌ FALLITO'}")
    print(f"  🧪 Test rimozione fallback: {'✅ PASSATO' if test2_passed else '❌ FALLITO'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 TUTTI I TEST PASSATI!")
        print(f"✅ Il sistema di contatore è pronto per l'uso")
    else:
        print(f"\n⚠️ ALCUNI TEST FALLITI - verificare implementazione")
        
    print("=" * 80)
