#!/usr/bin/env python3
"""
Test correzione propagati: NON devono mai andare automaticamente in review

Autore: Valerio Bignardi  
Data: 2025-09-05
Aggiornamenti:
- 2025-09-05: Creato per testare correzione propagati
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pipeline'))

def test_propagated_never_review():
    """
    Test che i propagati non vadano mai automaticamente in review
    
    Scopo: Verificare correzione logica propagazione
    """
    print("🧪 TEST: Propagati NON devono mai andare automaticamente in review")
    
    try:
        # Test 1: _determine_propagated_status
        print("\n1️⃣ Test _determine_propagated_status...")
        
        # Simula rappresentanti CON CLASSIFICAZIONI REVIEWED
        mock_representatives = [
            {
                'session_id': 'rep1', 
                'cluster_id': 1,
                'is_representative': True,
                'classification': {
                    'label': 'info_prenotazione',
                    'confidence': 0.9,
                    'status': 'reviewed'  # ✅ IMPORTANTE: Deve essere reviewed
                }
            },
            {
                'session_id': 'rep2',
                'cluster_id': 1, 
                'is_representative': True,
                'classification': {
                    'label': 'info_prestazioni',
                    'confidence': 0.8,
                    'status': 'reviewed'  # ✅ IMPORTANTE: Deve essere reviewed
                }
            },
            {
                'session_id': 'rep3',
                'cluster_id': 1,
                'is_representative': True, 
                'classification': {
                    'label': 'info_prenotazione',
                    'confidence': 0.7,
                    'status': 'reviewed'  # ✅ IMPORTANTE: Deve essere reviewed
                }
            }
        ]
        
        # Importa il metodo da testare  
        from end_to_end_pipeline import EndToEndPipeline
        
        # Crea istanza per accedere al metodo
        pipeline = EndToEndPipeline()
        
        # Test caso consenso 67% (2/3)
        result = pipeline._determine_propagated_status(mock_representatives, 0.7)
        
        print(f"   📊 Risultato consenso 67%:")
        print(f"      needs_review: {result['needs_review']}")
        print(f"      propagated_label: {result['propagated_label']}")
        print(f"      reason: {result['reason']}")
        
        if result['needs_review'] == False:
            print("   ✅ CORRETTO: Propagati NON vanno in review automaticamente")
        else:
            print("   ❌ ERRORE: Propagati vanno ancora in review!")
            return False
        
        # Test 2: Caso 50-50 
        print("\n2️⃣ Test caso disaccordo 50-50...")
        
        mock_representatives_5050 = [
            {
                'session_id': 'rep1',
                'cluster_id': 1,
                'is_representative': True,
                'classification': {
                    'label': 'info_prenotazione',
                    'confidence': 0.9,
                    'status': 'reviewed'  # ✅ IMPORTANTE: Deve essere reviewed
                }
            },
            {
                'session_id': 'rep2', 
                'cluster_id': 1,
                'is_representative': True,
                'classification': {
                    'label': 'info_prestazioni',
                    'confidence': 0.8,
                    'status': 'reviewed'  # ✅ IMPORTANTE: Deve essere reviewed
                }
            }
        ]
        
        result_5050 = pipeline._determine_propagated_status(mock_representatives_5050, 0.7)
        
        print(f"   📊 Risultato 50-50:")
        print(f"      needs_review: {result_5050['needs_review']}")
        print(f"      propagated_label: {result_5050['propagated_label']}")
        print(f"      reason: {result_5050['reason']}")
        
        if result_5050['needs_review'] == False:
            print("   ✅ CORRETTO: Anche 50-50 NON va in review automaticamente")
        else:
            print("   ❌ ERRORE: Caso 50-50 va ancora in review!")
            return False
            
        print("\n🎉 TUTTI I TEST PASSATI!")
        print("✅ I propagati NON vanno mai automaticamente in review")
        print("✅ Correzione implementata correttamente")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRORE DURANTE TEST: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_propagated_never_review()
    exit(0 if success else 1)
