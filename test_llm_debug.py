#!/usr/bin/env python3
"""
Test per debug avanzato della risposta LLM
Autore: Valerio Bignardi
Data: 2025-09-01
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

from Classification.intelligent_classifier import IntelligentClassifier
from Utils.tenant import Tenant
import logging

# Configura logging per vedere i debug
logging.basicConfig(level=logging.INFO)

def test_llm_raw_response():
    """Test della risposta RAW del modello LLM con debug completo"""
    
    print("🔧 Inizializzazione test LLM debug...")
    
    # Carica tenant
    tenant = Tenant.from_uuid('015007d9-d413-11ef-86a5-96000228e7fe')
    print(f"✅ Tenant: {tenant.tenant_name}")
    
    # Inizializza classificatore con logging abilitato
    classifier = IntelligentClassifier(
        tenant=tenant,  # Usa 'tenant' non 'tenant_id'
        enable_logging=True  # IMPORTANTE: abilita debug
    )
    
    # Testo di prova semplice
    test_conversation = """
    [UTENTE] Ciao, vorrei prenotare una visita cardiologica
    [ASSISTENTE] Certo, la aiuto con la prenotazione. Per quando preferirebbe?
    [UTENTE] La prossima settimana se possibile
    """
    
    print("🚀 Avvio classificazione con debug completo...")
    print("="*80)
    
    try:
        result = classifier.classify_with_motivation(test_conversation)
        print("="*80)
        print(f"🎯 RISULTATO FINALE:")
        print(f"  - Label: {result.predicted_label}")
        print(f"  - Confidence: {result.confidence}")  
        print(f"  - Motivation: {result.motivation}")
        print(f"  - Success: {result.success}")
        
    except Exception as e:
        print("="*80)
        print(f"❌ ERRORE durante test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llm_raw_response()
