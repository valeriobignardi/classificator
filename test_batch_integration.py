#!/usr/bin/env python3
"""
Test script per verificare l'integrazione batch processing database

Autore: Valerio Bignardi
Data: 2025-09-09
Scopo: Testare l'integrazione tra database e classificatore per batch processing
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from Classification.intelligent_classifier import IntelligentClassifier
from Utils.tenant import Tenant

def test_batch_integration():
    """
    Test dell'integrazione batch processing database
    """
    print("=" * 80)
    print("🧪 TEST INTEGRAZIONE BATCH PROCESSING DATABASE")
    print("=" * 80)
    
    try:
        # Crea oggetto tenant Humanitas
        tenant = Tenant.from_slug("humanitas")
        print(f"✅ Tenant creato: {tenant.tenant_name} ({tenant.tenant_id})")
        
        # Inizializza classificatore con tenant
        classifier = IntelligentClassifier(
            tenant=tenant,
            enable_logging=True
        )
        
        print(f"✅ Classificatore inizializzato")
        print(f"📊 AI Config Service disponibile: {hasattr(classifier, 'ai_config_service') and classifier.ai_config_service is not None}")
        
        # Test 1: Verifica configurazione batch processing
        print("\n🔧 TEST 1: Configurazione Batch Processing")
        print("-" * 50)
        
        if hasattr(classifier, 'ai_config_service') and classifier.ai_config_service:
            batch_config = classifier.ai_config_service.get_batch_processing_config(tenant.tenant_id)
            print(f"📈 Configurazione batch database: {batch_config}")
        else:
            print("⚠️ AI Config Service non disponibile")
        
        # Test 2: Verifica caricamento config integrato
        print("\n🔧 TEST 2: Caricamento Config Integrato")
        print("-" * 50)
        
        config = classifier._load_config()
        if config:
            print(f"📊 Sorgente config: {config.get('source', 'unknown')}")
            print(f"📈 Batch size: {config.get('pipeline', {}).get('classification_batch_size', 'non trovato')}")
            print(f"🔄 Max parallel calls: {config.get('pipeline', {}).get('max_parallel_calls', 'non trovato')}")
        else:
            print("❌ Config non caricato")
        
        # Test 3: Test metodo batch processing semplice
        print("\n🔧 TEST 3: Metodo Batch Processing Semplice")
        print("-" * 50)
        
        simple_config = classifier._load_config()  # Il metodo semplice senza parametri
        if simple_config:
            print(f"📊 Sorgente config semplice: {simple_config.get('source', 'unknown')}")
            print(f"📈 Batch size semplice: {simple_config.get('pipeline', {}).get('classification_batch_size', 'non trovato')}")
        else:
            print("❌ Config semplice non caricato")
            
        print("\n" + "=" * 80)
        print("✅ TEST COMPLETATO CON SUCCESSO!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERRORE NEL TEST: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_batch_integration()
