#!/usr/bin/env python3
"""
Test completo della logica intelligente dei propagated
Autore: Valerio Bignardi
Data: 2025-01-28
Storia modifiche:
    - 2025-01-28: Test logica consenso 0.7 e review obbligatoria 50-50
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

from Pipeline.end_to_end_pipeline import EndToEndPipeline

def test_propagated_logic():
    """
    Testa la logica intelligente per i propagated con vari scenari
    
    Scopo:
        Verifica che _determine_propagated_status funzioni correttamente
        con soglia 0.7 e review obbligatoria per 50-50
        
    Output:
        Risultati test per tutti gli scenari
    
    Data ultima modifica: 2025-01-28
    """
    print("🧪 TEST LOGICA PROPAGATED INTELLIGENTE")
    
    # Inizializza pipeline per accedere al metodo
    pipeline = EndToEndPipeline("humanitas")
    
    # Test Scenario 1: Consenso forte (80%) → Auto-classifica
    print("\n📋 SCENARIO 1: Consenso forte 80%")
    cluster_reps_consensus = [
        {'classification': 'info_polizza', 'human_reviewed': True},
        {'classification': 'info_polizza', 'human_reviewed': True}, 
        {'classification': 'info_polizza', 'human_reviewed': True},
        {'classification': 'info_polizza', 'human_reviewed': True},
        {'classification': 'assistenza', 'human_reviewed': True}  # 4/5 = 80%
    ]
    
    result = pipeline._determine_propagated_status(cluster_reps_consensus)
    expected_needs_review = False  # Consenso >= 70%
    
    print(f"  Risultato: {result}")
    print(f"  ✅ needs_review: {result['needs_review']} (atteso: {expected_needs_review})")
    print(f"  ✅ propagated_label: {result['propagated_label']}")
    print(f"  ✅ reason: {result['reason']}")
    
    assert result['needs_review'] == expected_needs_review, "Consenso 80% dovrebbe auto-classificare"
    assert result['propagated_label'] == 'info_polizza', "Label più votata dovrebbe essere info_polizza"
    
    # Test Scenario 2: Disaccordo 60% → Review obbligatoria  
    print("\n📋 SCENARIO 2: Disaccordo 60%")
    cluster_reps_disagreement = [
        {'classification': 'info_polizza', 'human_reviewed': True},
        {'classification': 'info_polizza', 'human_reviewed': True},
        {'classification': 'info_polizza', 'human_reviewed': True}, 
        {'classification': 'assistenza', 'human_reviewed': True},
        {'classification': 'altro', 'human_reviewed': True}  # 3/5 = 60%
    ]
    
    result = pipeline._determine_propagated_status(cluster_reps_disagreement)
    expected_needs_review = True  # Consenso < 70%
    
    print(f"  Risultato: {result}")
    print(f"  ✅ needs_review: {result['needs_review']} (atteso: {expected_needs_review})")
    print(f"  ✅ reason: {result['reason']}")
    
    assert result['needs_review'] == expected_needs_review, "Disaccordo 60% dovrebbe richiedere review"
    
    # Test Scenario 3: 50-50 → Review obbligatoria (richiesta specifica)
    print("\n📋 SCENARIO 3: Caso 50-50 → Review obbligatoria")
    cluster_reps_5050 = [
        {'classification': 'info_polizza', 'human_reviewed': True},
        {'classification': 'assistenza', 'human_reviewed': True}  # 1/2 = 50%
    ]
    
    result = pipeline._determine_propagated_status(cluster_reps_5050)
    expected_needs_review = True  # 50-50 → review obbligatoria
    
    print(f"  Risultato: {result}")
    print(f"  ✅ needs_review: {result['needs_review']} (atteso: {expected_needs_review})")
    print(f"  ✅ reason: {result['reason']}")
    
    assert result['needs_review'] == expected_needs_review, "50-50 dovrebbe richiedere review obbligatoria"
    assert '50_50' in result['reason'], "Reason dovrebbe indicare 50-50"
    
    # Test Scenario 4: Nessun rappresentante reviewed → Review obbligatoria
    print("\n📋 SCENARIO 4: Nessun rappresentante reviewed")
    cluster_reps_no_review = [
        {'classification': 'info_polizza', 'human_reviewed': False},
        {'classification': 'assistenza', 'human_reviewed': False}
    ]
    
    result = pipeline._determine_propagated_status(cluster_reps_no_review)
    expected_needs_review = True  # Nessun reviewed → review
    
    print(f"  Risultato: {result}")
    print(f"  ✅ needs_review: {result['needs_review']} (atteso: {expected_needs_review})")
    print(f"  ✅ reason: {result['reason']}")
    
    assert result['needs_review'] == expected_needs_review, "Nessun reviewed dovrebbe richiedere review"
    assert 'no_reviewed' in result['reason'], "Reason dovrebbe indicare nessun reviewed"
    
    # Test Scenario 5: Consenso esatto 70% → Auto-classifica (soglia inclusa)
    print("\n📋 SCENARIO 5: Consenso esatto 70% (soglia)")
    cluster_reps_exact_threshold = [
        {'classification': 'info_polizza', 'human_reviewed': True},
        {'classification': 'info_polizza', 'human_reviewed': True},
        {'classification': 'info_polizza', 'human_reviewed': True},
        {'classification': 'assistenza', 'human_reviewed': True},
        {'classification': 'altro', 'human_reviewed': True},
        {'classification': 'cortesia', 'human_reviewed': True},
        {'classification': 'info_polizza', 'human_reviewed': True},  # 4/7 ≈ 57%
        {'classification': 'info_polizza', 'human_reviewed': True},
        {'classification': 'info_polizza', 'human_reviewed': True},
        {'classification': 'info_polizza', 'human_reviewed': True}   # 7/10 = 70%
    ]
    
    result = pipeline._determine_propagated_status(cluster_reps_exact_threshold)
    expected_needs_review = False  # 70% = soglia → auto-classifica
    
    print(f"  Risultato: {result}")
    print(f"  ✅ needs_review: {result['needs_review']} (atteso: {expected_needs_review})")
    print(f"  ✅ reason: {result['reason']}")
    
    assert result['needs_review'] == expected_needs_review, "70% esatto dovrebbe auto-classificare"
    
    print(f"\n✅ TUTTI I TEST COMPLETATI CON SUCCESSO!")
    print(f"🎯 Logica propagated funziona correttamente:")
    print(f"   • Consenso ≥ 70% → Auto-classificazione")
    print(f"   • Consenso < 70% → Review umana") 
    print(f"   • Caso 50-50 → Review obbligatoria")
    print(f"   • Nessun reviewed → Review obbligatoria")
    
    return True

if __name__ == "__main__":
    test_propagated_logic()
