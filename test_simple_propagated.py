#!/usr/bin/env python3
"""
Test semplice correzione propagati - Test diretto del metodo

Autore: Valerio Bignardi  
Data: 2025-09-05
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pipeline'))

def test_simple_propagated():
    """
    Test diretto del metodo _determine_propagated_status
    """
    print("🧪 TEST SEMPLICE: _determine_propagated_status")
    
    # Test senza usare pipeline completa
    print("\n1️⃣ Test metodo diretto...")
    
    # Creo mock data nel formato corretto
    mock_representatives = [
        {
            'session_id': 'rep1',
            'human_reviewed': True,  # ✅ Campo corretto
            'classification': 'info_prenotazione'  # ✅ Label diretta, non oggetto
        },
        {
            'session_id': 'rep2', 
            'human_reviewed': True,  # ✅ Campo corretto
            'classification': 'info_prestazioni'  # ✅ Label diretta
        },
        {
            'session_id': 'rep3',
            'human_reviewed': True,  # ✅ Campo corretto 
            'classification': 'info_prenotazione'  # ✅ Label diretta
        }
    ]
    
    # Importa solo la classe
    from Pipeline.end_to_end_pipeline import EndToEndPipeline
    
    # Crea istanza minimale
    pipeline = EndToEndPipeline()
    
    # Test metodo diretto
    result = pipeline._determine_propagated_status(mock_representatives, 0.7)
    
    print(f"   📊 Risultato:")
    print(f"      needs_review: {result['needs_review']}")
    print(f"      propagated_label: {result['propagated_label']}")
    print(f"      reason: {result['reason']}")
    
    if result['needs_review'] == False:
        print("   ✅ CORRETTO: Propagati NON vanno in review!")
        return True
    else:
        print("   ❌ ERRORE: Propagati vanno ancora in review")
        return False

if __name__ == "__main__":
    success = test_simple_propagated()
    exit(0 if success else 1)
