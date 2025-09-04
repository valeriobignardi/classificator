#!/usr/bin/env python3
"""
Test per la pulizia dei tag nell'AltroTagValidator

Autore: Valerio Bignardi
Data: 2025-09-04
Storia: Creato per testare fix caratteri strani nei tag
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HumanReview.altro_tag_validator import AltroTagValidator
from Utils.tenant import Tenant

def test_tag_cleaning():
    """
    Test della funzione di pulizia tag per caratteri problematici
    
    Scopo della funzione: Verifica che i tag con backslash e caratteri strani vengano puliti correttamente
    Parametri di input: None
    Parametri di output: None (stampa risultati)
    Valori di ritorno: None
    Tracciamento aggiornamenti: 2025-09-04 - Creato per test pulizia tag
    
    Autore: Valerio Bignardi
    Data: 2025-09-04
    """
    
    print("🧪 Test pulizia tag con caratteri problematici\n")
    
    # Simula un tenant per il validator
    test_tenant = Tenant(
        tenant_id="test-uuid-123",
        tenant_name="Test Tenant", 
        tenant_slug="test",
        tenant_database="test_database",
        tenant_status=1
    )
    
    # Crea validator
    validator = AltroTagValidator(tenant=test_tenant)
    
    # Test cases con tag problematici
    test_cases = [
        # Caso originale del problema
        ("INFO\\_GENERALI", "INFO_GENERALI"),
        ("INFO\\GENERALI", "INFO_GENERALI"),
        
        # Virgolette
        ('"PRENOTAZIONI"', "PRENOTAZIONI"),
        ("'APPUNTAMENTI'", "APPUNTAMENTI"),
        ('\"INFORMAZIONI\"', "INFORMAZIONI"),
        
        # Caratteri speciali misti
        ("INFO-GENERALE", "INFO_GENERALE"),
        ("INFO.GENERALE", "INFO_GENERALE"),
        ("INFO/GENERALE", "INFO_GENERALE"),
        ("INFO@GENERALE", "INFO_GENERALE"),
        
        # Spazi multipli
        ("INFO   GENERALE", "INFO_GENERALE"),
        ("INFO\t\tGENERALE", "INFO_GENERALE"),
        
        # Underscore multipli
        ("INFO___GENERALE", "INFO_GENERALE"),
        ("___INFO_GENERALE___", "INFO_GENERALE"),
        
        # Casi misti complessi
        ("\"INFO\\_GENERALE-TEST\"", "INFO_GENERALE_TEST"),
        ("'DATA\\APPUNTAMENTO'", "DATA_APPUNTAMENTO"),
        
        # Casi limite
        ("", ""),
        ("   ", ""),
        ("A", "A"),
        ("AB", "AB"),
    ]
    
    successi = 0
    fallimenti = 0
    
    for input_tag, expected_output in test_cases:
        try:
            result = validator._clean_tag_text(input_tag)
            
            if result == expected_output:
                print(f"✅ PASS: '{input_tag}' → '{result}'")
                successi += 1
            else:
                print(f"❌ FAIL: '{input_tag}' → '{result}' (expected: '{expected_output}')")
                fallimenti += 1
                
        except Exception as e:
            print(f"💥 ERROR: '{input_tag}' → Exception: {e}")
            fallimenti += 1
    
    print(f"\n📊 Risultati test:")
    print(f"   ✅ Successi: {successi}")
    print(f"   ❌ Fallimenti: {fallimenti}")
    print(f"   📈 Success rate: {successi/(successi+fallimenti)*100:.1f}%")
    
    return fallimenti == 0

def test_llm_response_extraction():
    """
    Test dell'estrazione tag da response LLM reali
    
    Scopo della funzione: Simula response LLM con tag problematici per test end-to-end
    Parametri di input: None
    Parametri di output: None (stampa risultati)
    Valori di ritorno: None
    Tracciamento aggiornamenti: 2025-09-04 - Creato per test estrazione LLM
    
    Autore: Valerio Bignardi
    Data: 2025-09-04
    """
    
    print("\n🤖 Test estrazione tag da response LLM\n")
    
    # Simula un tenant per il validator
    test_tenant = Tenant(
        tenant_id="test-uuid-123",
        tenant_name="Test Tenant", 
        tenant_slug="test",
        tenant_database="test_database",
        tenant_status=1
    )
    
    # Crea validator
    validator = AltroTagValidator(tenant=test_tenant)
    
    # Simula response LLM problematiche
    test_responses = [
        # Caso problematico originale
        'Basandomi sul contenuto, suggerisco il tag "INFO\\_GENERALI"',
        
        # Altri casi problematici
        'Il tag appropriato è "PRENOTAZIONI\\_VISITE"',
        'Classificherei come "DATI\\PAZIENTE"',
        'Propongo il tag: \'INFORMAZIONI/MEDICHE\'',
        
        # Response più complesse
        '''Analizzando la conversazione, il tema principale riguarda richieste di informazioni generali.
        Suggerisco il tag "INFO\\_SANITARIE" come classificazione appropriata.''',
        
        # Fallback cases
        'Non posso identificare un tag specifico INFO_GENERALI',
        'La conversazione riguarda argomenti vari PRENOTAZIONI',
    ]
    
    expected_results = [
        "INFO_GENERALI",
        "PRENOTAZIONI_VISITE", 
        "DATI_PAZIENTE",
        "INFORMAZIONI_MEDICHE",
        "INFO_SANITARIE",
        "INFO_GENERALI",
        "PRENOTAZIONI"
    ]
    
    successi = 0
    fallimenti = 0
    
    for i, (response, expected) in enumerate(zip(test_responses, expected_results)):
        try:
            result = validator._extract_tag_from_llm_response(response)
            
            if result == expected:
                print(f"✅ PASS #{i+1}: Estratto '{result}' come atteso")
                successi += 1
            else:
                print(f"❌ FAIL #{i+1}: Estratto '{result}' (expected: '{expected}')")
                print(f"   📝 Response: {response[:50]}...")
                fallimenti += 1
                
        except Exception as e:
            print(f"💥 ERROR #{i+1}: Exception: {e}")
            print(f"   📝 Response: {response[:50]}...")
            fallimenti += 1
    
    print(f"\n📊 Risultati test estrazione:")
    print(f"   ✅ Successi: {successi}")
    print(f"   ❌ Fallimenti: {fallimenti}")
    print(f"   📈 Success rate: {successi/(successi+fallimenti)*100:.1f}%")
    
    return fallimenti == 0

if __name__ == "__main__":
    print("🧪 AVVIO TEST PULIZIA TAG\n" + "="*50)
    
    try:
        # Test pulizia diretta
        success1 = test_tag_cleaning()
        
        # Test estrazione da LLM response
        success2 = test_llm_response_extraction()
        
        print("\n" + "="*50)
        if success1 and success2:
            print("🎉 TUTTI I TEST SUPERATI!")
            exit(0)
        else:
            print("💥 ALCUNI TEST FALLITI!")
            exit(1)
            
    except Exception as e:
        print(f"💥 ERRORE CRITICO NEI TEST: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
