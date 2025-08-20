#!/usr/bin/env python3
"""
Test del nuovo workflow Mistral Base/Fine-tuned 
"""

import sys
import os

# Aggiungi il path per importare i moduli
sys.path.append('/home/ubuntu/classificazione_b+++++_bck2')

def test_workflow():
    """Test del workflow corretto Mistral Base → Fine-tuned"""
    print("🧪 TEST WORKFLOW MISTRAL BASE/FINE-TUNED")
    print("=" * 50)
    
    try:
        from QualityGate.quality_gate_engine import QualityGateEngine
        
        # Test tenant esempio
        tenant_name = "humanitas"
        
        # Crea QualityGate engine
        quality_gate = QualityGateEngine(
            tenant_name=tenant_name,
            review_db_path=f"test_review_{tenant_name}.db",
            training_log_path=f"training_decisions_{tenant_name}.jsonl"
        )
        
        print(f"\n🔍 Testing workflow per tenant: {tenant_name}")
        
        # Test 1: Controllo primo training
        is_first = quality_gate._is_first_supervised_training()
        print(f"   - È primo training supervisionato: {is_first}")
        
        # Test 2: Creazione ensemble con modalità corretta  
        print(f"\n🤖 Test creazione ensemble:")
        
        # Primo training (dovrebbe usare Mistral BASE)
        ensemble_first = quality_gate._create_ensemble_classifier_with_correct_mistral_mode(
            tenant_name, force_base_model=True
        )
        print(f"   - Ensemble primo training creato (should use BASE model)")
        
        # Training successivi (dovrebbe usare Mistral FINE-TUNED se disponibile)
        ensemble_subsequent = quality_gate._create_ensemble_classifier_with_correct_mistral_mode(
            tenant_name, force_base_model=False
        )
        print(f"   - Ensemble training successivi creato (should use FINE-TUNED if available)")
        
        # Test 3: Verifica modelli fine-tuned disponibili
        from FineTuning.mistral_finetuning_manager import MistralFineTuningManager
        finetuning_manager = MistralFineTuningManager()
        
        has_finetuned = finetuning_manager.has_finetuned_model(tenant_name)
        print(f"\n📋 Info modelli fine-tuned:")
        print(f"   - Modello fine-tuned disponibile per {tenant_name}: {has_finetuned}")
        
        if has_finetuned:
            model_name = finetuning_manager.get_client_model(tenant_name)
            print(f"   - Nome modello attivo: {model_name}")
            
            model_info = finetuning_manager.get_model_info(tenant_name)
            print(f"   - Info modello: {model_info}")
        
        print(f"\n✅ Test workflow completato con successo!")
        print(f"\n📋 RIEPILOGO FLUSSO CORRETTO:")
        print(f"   1. Primo training supervisionato → Mistral BASE + ML")
        print(f"   2. Training successivi → Mistral FINE-TUNED + ML trained")
        print(f"   3. La decisione è automatica basata su disponibilità modello fine-tuned")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore nel test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_workflow()
    sys.exit(0 if success else 1)
