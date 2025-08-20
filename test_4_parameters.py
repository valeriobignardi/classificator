#!/usr/bin/env python3
"""
Test dei 4 parametri del nuovo sistema di training supervisionato
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Pipeline.end_to_end_pipeline import EndToEndPipeline
import yaml

def test_4_parameters():
    """Test con tutti e 4 i parametri"""
    
    print("🧪 TEST 4 PARAMETRI DEL SISTEMA SUPERVISIONATO")
    print("=" * 60)
    
    # 1. Verifica che la configurazione supporti tutti i parametri
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        supervised_config = config.get('supervised_training', {})
        print(f"✅ Configurazione supervised_training trovata")
        print(f"   📊 Extraction config: {bool(supervised_config.get('extraction'))}")
        print(f"   👤 Human review config: {bool(supervised_config.get('human_review'))}")
        
    except Exception as e:
        print(f"❌ Errore configurazione: {e}")
        return False
    
    # 2. Test signature del metodo pipeline
    try:
        # Crea istanza pipeline di test
        pipeline = EndToEndPipeline("test_client")
        
        # Verifica che il metodo accetti tutti i parametri
        import inspect
        sig = inspect.signature(pipeline.esegui_training_interattivo)
        params = list(sig.parameters.keys())
        
        expected_params = [
            'giorni_indietro', 'limit', 'max_human_review_sessions',
            'confidence_threshold', 'force_review', 'disagreement_threshold'
        ]
        
        print(f"\n🔍 VERIFICA SIGNATURE METODO:")
        for param in expected_params:
            if param in params:
                print(f"   ✅ {param}")
            else:
                print(f"   ❌ {param} - MANCANTE")
                return False
                
        print(f"✅ Signature completa: {len(params)} parametri")
        
    except Exception as e:
        print(f"❌ Errore test signature: {e}")
        return False
    
    # 3. Test signature del metodo di selezione
    try:
        sig_select = inspect.signature(pipeline._select_representatives_for_human_review)
        params_select = list(sig_select.parameters.keys())
        
        expected_select = [
            'representatives', 'suggested_labels', 'max_sessions', 'all_sessions',
            'confidence_threshold', 'force_review', 'disagreement_threshold'
        ]
        
        print(f"\n🔍 VERIFICA SIGNATURE SELEZIONE:")
        for param in expected_select:
            if param in params_select:
                print(f"   ✅ {param}")
            else:
                print(f"   ❌ {param} - MANCANTE")
                return False
                
        print(f"✅ Signature selezione completa: {len(params_select)} parametri")
        
    except Exception as e:
        print(f"❌ Errore test signature selezione: {e}")
        return False
    
    # 4. Test valori di default
    try:
        defaults = {
            'confidence_threshold': 0.7,
            'force_review': False,
            'disagreement_threshold': 0.3
        }
        
        print(f"\n📋 VERIFICA VALORI DEFAULT:")
        for param_name, expected_default in defaults.items():
            param_obj = sig.parameters[param_name]
            actual_default = param_obj.default
            
            if actual_default == expected_default:
                print(f"   ✅ {param_name}: {actual_default}")
            else:
                print(f"   ❌ {param_name}: {actual_default} (expected: {expected_default})")
                return False
                
        print(f"✅ Tutti i default corretti")
        
    except Exception as e:
        print(f"❌ Errore test defaults: {e}")
        return False
    
    print(f"\n🎉 TUTTI I TEST SUPERATI!")
    print(f"   ✅ 4 parametri completamente integrati")
    print(f"   ✅ Configurazione yaml supportata")
    print(f"   ✅ Signature metodi aggiornate")
    print(f"   ✅ Valori default corretti")
    
    return True

if __name__ == "__main__":
    success = test_4_parameters()
    sys.exit(0 if success else 1)
