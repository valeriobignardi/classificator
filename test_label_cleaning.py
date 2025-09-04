#!/usr/bin/env python3
"""
Test script per verificare il funzionamento della pulizia delle etichette

Autore: Valerio Bignardi  
Data: 2025-09-04
"""

import sys
import os

# Aggiungi il path del progetto
sys.path.append(os.path.dirname(__file__))

# Importa la funzione di pulizia
from Classification.intelligent_classifier import clean_label_text

def test_label_cleaning():
    """
    Testa la funzione di pulizia delle etichette con vari casi problematici
    """
    print("🧪 Test della funzione clean_label_text")
    print("=" * 50)
    
    # Casi di test basati sui problemi reali visti nei log
    test_cases = [
        # Caso problematico dal log
        ('"convenzioni_viaggio_strutture_alberghiere"', 'convenzioni_viaggio_strutture_alberghiere'),
        ('"strutture_convenzionate_alberghiere"', 'strutture_convenzionate_alberghiere'),
        ('"info_contatti_medico"', 'info_contatti_medico'),
        ('"prenotazione_esami"', 'prenotazione_esami'),
        
        # Altri casi di virgolette
        ("'prenotazione_esami'", 'prenotazione_esami'),
        ('"altro"', 'altro'),
        ("'altro'", 'altro'),
        
        # Casi con backslash
        ('info\\generali', 'info\\generali'),
        ('prenotazione\\"esami', 'prenotazione"esami'),
        ("prenotazione\\'esami", "prenotazione'esami"),
        
        # Casi normali (non dovrebbero cambiare)
        ('prenotazione_esami', 'prenotazione_esami'),
        ('info_amministrative', 'info_amministrative'),
        ('altro', 'altro'),
        
        # Casi edge
        ('', ''),
        ('   ', ''),
        ('   "test"   ', 'test'),
        (None, None),
    ]
    
    for i, (input_label, expected) in enumerate(test_cases, 1):
        try:
            result = clean_label_text(input_label)
            status = "✅ PASS" if result == expected else "❌ FAIL"
            print(f"Test {i:2d}: {status}")
            print(f"   Input:    {repr(input_label)}")
            print(f"   Expected: {repr(expected)}")
            print(f"   Got:      {repr(result)}")
            
            if result != expected:
                print(f"   🚨 ERRORE: Risultato non corrispondente!")
            print()
            
        except Exception as e:
            print(f"Test {i:2d}: ❌ EXCEPTION")
            print(f"   Input:    {repr(input_label)}")
            print(f"   Error:    {e}")
            print()

    print("🏁 Test completato!")

if __name__ == "__main__":
    test_label_cleaning()
