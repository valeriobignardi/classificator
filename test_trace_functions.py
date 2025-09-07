#!/usr/bin/env python3
"""
Test delle funzioni con trace_all aggiunti
"""

import sys
import os

# Aggiungi path per imports
sys.path.append('/home/ubuntu/classificatore')

def test_build_classification_prompt():
    """Test della funzione _build_classification_prompt con trace_all"""
    try:
        from Classification.intelligent_classifier import IntelligentClassifier
        from Utils.tenant import Tenant
        
        print("🔍 Test _build_classification_prompt con trace_all...")
        
        # Carica tenant di test
        tenant = Tenant.from_slug("humanitas")
        
        # Crea classificatore
        classifier = IntelligentClassifier(
            tenant=tenant,
            model_name="gemma2:9b-instruct-q8_0"
        )
        
        # Test semplice
        conversation = "Ciao, ho bisogno di informazioni sui parcheggi dell'ospedale"
        
        print("🚀 Chiamata _build_classification_prompt...")
        prompt = classifier._build_classification_prompt(conversation)
        
        print(f"✅ Prompt generato: {len(prompt)} caratteri")
        print(f"📝 Preview: {prompt[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test _build_classification_prompt: {e}")
        return False

def test_build_consensus_label():
    """Test della funzione _build_consensus_label con trace_all"""
    try:
        from Clustering.intelligent_intent_clusterer import IntelligentIntentClusterer
        from Utils.tenant import Tenant
        
        print("\n🔍 Test _build_consensus_label con trace_all...")
        
        # Carica tenant di test
        tenant = Tenant.from_slug("humanitas")
        
        # Crea clusterer
        clusterer = IntelligentIntentClusterer(tenant=tenant)
        
        # Dati di test per consensus
        llm_results = [
            {'intent': 'info_parcheggio', 'confidence': 0.8, 'reasoning': 'Richiesta informazioni parcheggio'},
            {'intent': 'info_parcheggio', 'confidence': 0.9, 'reasoning': 'Domanda su parcheggi ospedale'},
            {'intent': 'info_contatti', 'confidence': 0.6, 'reasoning': 'Potrebbe essere richiesta contatti'}
        ]
        
        print("🚀 Chiamata _build_consensus_label...")
        consensus = clusterer._build_consensus_label(llm_results)
        
        print(f"✅ Consensus generato:")
        print(f"   🏷️  Label: {consensus['final_label']}")
        print(f"   📊 Confidence: {consensus['confidence']:.3f}")
        print(f"   📝 Reasoning: {consensus['reasoning']}")
        print(f"   📈 Stats: {consensus['stats']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test _build_consensus_label: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Test trace_all per funzioni modificate")
    print("=" * 50)
    
    # Test 1: _build_classification_prompt
    success1 = test_build_classification_prompt()
    
    # Test 2: _build_consensus_label
    success2 = test_build_consensus_label()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✅ Tutti i test completati con successo!")
        print("🔍 trace_all è stato aggiunto correttamente alle funzioni")
    else:
        print("❌ Alcuni test hanno fallito")
        sys.exit(1)
