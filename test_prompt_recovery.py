#!/usr/bin/env python3
"""
Test rapido per verificare che il prompt_altro_validator sia accessibile
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'Utils'))

try:
    from Utils.prompt_manager import PromptManager
    
    print("🔧 Test recupero prompt_altro_validator...")
    
    pm = PromptManager()
    if pm.connect():
        print("✅ Connesso al database")
        
        # Test recupero prompt per Humanitas
        tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"  # Humanitas
        
        # Test con sostituzione variabile
        variables = {"LISTA_TAG": "test_tag1, test_tag2, test_tag3"}
        
        processed_prompt = pm.get_prompt(
            tenant_or_id=tenant_id,
            engine='LLM',
            prompt_type='SYSTEM',
            prompt_name='prompt_altro_validator',
            variables=variables
        )
        
        if processed_prompt:
            print("✅ Prompt recuperato e processato con successo!")
            print(f"📏 Lunghezza prompt: {len(processed_prompt)} caratteri")
            print(f"📋 Inizio prompt: {processed_prompt[:300]}...")
            print(f"🔍 Contiene tag test: {'test_tag1' in processed_prompt}")
        else:
            print("❌ Prompt non trovato o non processabile!")
            print(f"📋 Risultato: {processed_prompt}")
            
        pm.disconnect()
    else:
        print("❌ Impossibile connettersi al database")
        
except Exception as e:
    print(f"❌ Errore: {e}")
