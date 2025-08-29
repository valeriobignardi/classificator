#!/usr/bin/env python3
"""
Correzione errori di formato e conteggio nel training supervisionato

Autore: Valerio Bignardi  
Data di creazione: 2025-08-29
Storia degli aggiornamenti:
- 2025-08-29: Correzione errore "Unknown format code 'f' for object of type 'str'"
                e verifica conteggio sessioni per tenant Alleanza
"""

import sys
import os
from datetime import datetime

# Aggiungi percorsi
sys.path.append('.')

def analyze_format_error():
    """
    Scopo: Analizza l'errore di formato nei file di classificazione
    
    Output:
        - bool: True se analisi completata
        
    Ultimo aggiornamento: 2025-08-29
    """
    print("🔍 ANALISI ERRORE DI FORMATO")
    print("=" * 50)
    
    print(f"❌ Errore riportato: Unknown format code 'f' for object of type 'str'")
    print(f"📊 Questo indica che una variabile string viene passata a un format .3f")
    
    print(f"\n🎯 POSSIBILI CAUSE:")
    print(f"1. train_accuracy è una stringa invece di float/int")
    print(f"2. Qualche metrica viene convertita in stringa da JSON serialization")
    print(f"3. Errore nel caricamento metriche da database/cache")
    
    print(f"\n🔧 LOCAZIONI DA CONTROLLARE:")
    print(f"• advanced_ensemble_classifier.py riga 267 e 382")
    print(f"• Calcolo train_accuracy = self.ml_ensemble.score()")
    print(f"• Serializzazione JSON nelle metriche")
    
    return True

def test_tenant_data():
    """
    Scopo: Verifica dati tenant Alleanza
    
    Output:
        - int: Numero di sessioni trovate
        
    Ultimo aggiornamento: 2025-08-29
    """
    try:
        print("\n📊 VERIFICA CONTEGGIO SESSIONI ALLEANZA")
        print("=" * 50)
        
        # Connessione DB per verificare conteggio reale
        from LettoreConversazioni.lettore import LettoreConversazioni
        
        lettore = LettoreConversazioni()
        
        # Test conteggio per alleanza (slug o nome esatto)
        print("🔍 Ricerca con slug 'alleanza'...")
        sessioni = lettore.get_conversations('alleanza', limit=None)
        print(f"📊 Sessioni trovate per 'alleanza': {len(sessioni)}")
        
        # Se non trova con slug, prova con nome completo
        if len(sessioni) == 0:
            print("🔍 Ricerca con nome 'Alleanza'...")
            sessioni = lettore.get_conversations('Alleanza', limit=None)
            print(f"📊 Sessioni trovate per 'Alleanza': {len(sessioni)}")
            
        if len(sessioni) == 0:
            print("🔍 Ricerca con nome 'alleanza salute'...")
            sessioni = lettore.get_conversations('alleanza salute', limit=None)
            print(f"📊 Sessioni trovate per 'alleanza salute': {len(sessioni)}")
        
        print(f"\n🎯 RISULTATO: {len(sessioni)} sessioni trovate")
        
        if len(sessioni) != 17 and len(sessioni) != 15:
            print(f"⚠️ DISCREPANZA: Sistema mostra 17, utente dice 15, DB ha {len(sessioni)}")
        
        return len(sessioni)
        
    except Exception as e:
        print(f"❌ Errore verifica tenant: {e}")
        return 0

def test_format_issue():
    """
    Scopo: Simula e testa il problema di formato
    
    Output:
        - bool: True se test completato
        
    Ultimo aggiornamento: 2025-08-29
    """
    print(f"\n🧪 TEST ERRORE FORMATO")
    print("=" * 40)
    
    test_values = [
        0.8235294117647058,  # float normale
        "0.8235294117647058",  # string che causa errore
        None,  # None value
        "N/A"  # stringa non numerica
    ]
    
    for i, val in enumerate(test_values, 1):
        try:
            result = f"Training accuracy: {val:.3f}"
            print(f"✅ Test {i}: {result}")
        except Exception as e:
            print(f"❌ Test {i}: ERRORE - {e}")
            
            # Tentativo correzione
            try:
                if isinstance(val, str) and val not in ["N/A", None, ""]:
                    fixed_val = float(val)
                    result = f"Training accuracy: {fixed_val:.3f}"
                    print(f"🔧 Test {i} CORRETTO: {result}")
                else:
                    result = f"Training accuracy: {val} (non-numeric)"
                    print(f"🔧 Test {i} ALTERNATIVO: {result}")
            except:
                print(f"� Test {i} FALLBACK: Training accuracy: N/A")
    
    return True

def suggest_fixes():
    """
    Scopo: Suggerisce correzioni per i problemi identificati
    
    Output:
        - bool: True se suggerimenti completati
        
    Ultimo aggiornamento: 2025-08-29
    """
    print(f"\n🛠️ SUGGERIMENTI DI CORREZIONE")
    print("=" * 50)
    
    print(f"1. 🔢 CORREZIONE ERRORE FORMATO:")
    print(f"   Prima: print(f'Training accuracy: {{train_accuracy:.3f}}')")
    print(f"   Dopo:  print(f'Training accuracy: {{float(train_accuracy):.3f}}')")
    
    print(f"\n2. 🧹 PROTEZIONE TYPE SAFETY:")
    print(f"   try:")
    print(f"       accuracy_val = float(train_accuracy) if isinstance(train_accuracy, str) else train_accuracy")
    print(f"       print(f'Training accuracy: {{accuracy_val:.3f}}')")
    print(f"   except (ValueError, TypeError):")
    print(f"       print(f'Training accuracy: {{train_accuracy}} (non-numeric)')")
    
    print(f"\n3. 📊 VERIFICA CONTEGGIO SESSIONI:")
    print(f"   Il sistema mostra 17 sessioni ma l'utente dice 15")
    print(f"   Possibili cause:")
    print(f"   • Conteggio include sessioni duplicate/test")
    print(f"   • Differenza tra sessioni caricate vs processate")
    print(f"   • Cache non aggiornato")
    
    return True

def main():
    """
    Scopo: Esegue analisi completa dei problemi di training
    
    Output:
        - None
        
    Ultimo aggiornamento: 2025-08-29
    """
    print("🚨 ANALISI PROBLEMI TRAINING SUPERVISIONATO")
    print("=" * 60)
    
    # Analisi errori
    analyze_format_error()
    
    # Test dati tenant
    count = test_tenant_data()
    
    # Test formato
    test_format_issue()
    
    # Suggerimenti
    suggest_fixes()
    
    print(f"\n🎯 AZIONI IMMEDIATE RACCOMANDATE:")
    print(f"1. Verifica conteggio reale sessioni Alleanza: {count if count > 0 else 'NON TROVATE'}")
    print(f"2. Correggi advanced_ensemble_classifier.py con float() casting")
    print(f"3. Controlla serializzazione JSON nelle metriche")
    print(f"4. Implementa type safety per tutte le metriche numeriche")
    
    print(f"\n💡 L'errore più probabile è train_accuracy come stringa")
    print(f"   invece di numero in advanced_ensemble_classifier.py linee 267 e 382")

if __name__ == "__main__":
    main()
