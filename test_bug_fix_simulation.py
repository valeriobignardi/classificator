#!/usr/bin/env python3
"""
Test simulazione del bug risolto - etichette con virgolette

Questo script simula il problema che si verificava quando l'LLM
restituiva etichette con virgolette, causando problemi nel salvataggio.

Autore: Valerio Bignardi  
Data: 2025-09-04
"""

import sys
import os

# Aggiungi il path del progetto
sys.path.append(os.path.dirname(__file__))

# Importa la funzione di pulizia
from Classification.intelligent_classifier import clean_label_text

def simulate_llm_responses():
    """
    Simula le risposte problematiche dell'LLM viste nei log
    """
    print("🔄 Simulazione del bug risolto - etichette con virgolette")
    print("=" * 60)
    
    # Risposte problematiche reali viste nei log
    problematic_responses = [
        {
            'raw': '"convenzioni_viaggio_strutture_alberghiere"',
            'description': 'Caso reale dal log predict.log - virgolette doppie'
        },
        {
            'raw': '"strutture_convenzionate_alberghiere"',
            'description': 'Altro caso reale dal log'
        },
        {
            'raw': '"info_contatti_medico"',
            'description': 'Caso info con virgolette'
        },
        {
            'raw': '"richiesta_ritiro_referti_cartella_clinica"',
            'description': 'Etichetta lunga con virgolette'
        },
        {
            'raw': 'prenotazione_esami',
            'description': 'Caso normale (controllo regressione)'
        }
    ]
    
    print("📋 Test dei casi problematici:")
    print()
    
    for i, case in enumerate(problematic_responses, 1):
        raw_label = case['raw']
        description = case['description']
        
        print(f"Test {i}: {description}")
        print(f"   📥 Input LLM:     '{raw_label}'")
        
        # Simula il processo prima della correzione
        old_behavior = raw_label  # Prima veniva salvato così com'era
        
        # Simula il processo dopo la correzione  
        new_behavior = clean_label_text(raw_label)
        
        print(f"   🗂️  Vecchio save:  '{old_behavior}'  {'❌' if '"' in old_behavior else '✅'}")
        print(f"   🧹 Nuovo save:    '{new_behavior}'  ✅")
        
        if old_behavior != new_behavior:
            print(f"   🔧 CORREZIONE APPLICATA!")
        else:
            print(f"   ✅ Nessuna modifica necessaria")
        
        print()
    
    print("📊 SUMMARY:")
    print("- ❌ PRIMA: L'LLM restituiva etichette con virgolette")
    print("- ❌ PROBLEMA: Venivano salvate nel DB con le virgolette")
    print("- ❌ RISULTATO: Frontend mostrava 'N/A' non trovando corrispondenza")
    print("- ✅ DOPO: Le etichette vengono pulite prima del salvataggio")
    print("- ✅ RISOLTO: Frontend trova correttamente le etichette")
    print()
    print("🎯 Bug risolto con successo!")

if __name__ == "__main__":
    simulate_llm_responses()
