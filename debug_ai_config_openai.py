#!/usr/bin/env python3
"""
Debug del servizio AI Configuration per OpenAI
"""

import os
import sys
from dotenv import load_dotenv

# Carica variabili d'ambiente
load_dotenv()

# Aggiungi path per il servizio
sys.path.append('AIConfiguration')

try:
    from ai_configuration_service import AIConfigurationService
    
    print("🔍 Test del servizio AI Configuration...")
    
    # Crea servizio
    service = AIConfigurationService()
    
    # Test metodo _check_openai_available direttamente
    print(f"\n🧪 Test _check_openai_available:")
    available = service._check_openai_available()
    print(f"   Risultato: {available}")
    
    # Mostra configurazione
    print(f"\n🔑 Configurazione OpenAI nel servizio:")
    print(f"   openai_api_key presente: {bool(service.openai_api_key)}")
    if service.openai_api_key:
        print(f"   Chiave inizia con: {service.openai_api_key[:12]}...")
    
    # Test diretto import OpenAI nel servizio
    print(f"\n📦 Test import OpenAI nel servizio:")
    try:
        import openai
        client = openai.OpenAI(api_key=service.openai_api_key)
        models = client.models.list()
        print(f"   ✅ Import OpenAI e connessione OK")
        print(f"   📊 Numero modelli disponibili: {len(models.data)}")
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        
except Exception as e:
    print(f"❌ Errore generale: {e}")
    import traceback
    traceback.print_exc()
