#!/usr/bin/env python3
"""
Verifica quale modello è configurato per humanitas nel database
Autore: Valerio Bignardi  
Data: 2025-09-01
"""

import sys
import os

# Aggiungi path del progetto
sys.path.append('/home/ubuntu/classificatore')
sys.path.append('/home/ubuntu/classificatore/AIConfiguration')

def check_humanitas_model():
    """
    Controlla quale modello è configurato per humanitas nel database
    """
    print("🔍 VERIFICA MODELLO CONFIGURATO PER HUMANITAS")
    print("="*80)
    
    try:
        from ai_configuration_service import AIConfigurationService
        ai_service = AIConfigurationService()  # Senza parametri, usa database automaticamente
        
        print("✅ AIConfigurationService inizializzato")
        print("="*80)
        
        # Ottieni configurazione per humanitas
        config = ai_service.get_tenant_configuration("humanitas", force_no_cache=True)
        
        if config:
            print("📋 CONFIGURAZIONE TROVATA per humanitas:")
            print("-"*80)
            
            # Controlla modello LLM
            if 'llm_model' in config:
                llm_config = config['llm_model']
                print(f"🤖 LLM_MODEL:")
                if isinstance(llm_config, dict):
                    for key, value in llm_config.items():
                        print(f"   {key}: {value}")
                else:
                    print(f"   Valore: {llm_config}")
            else:
                print("❌ 'llm_model' non trovato nella configurazione")
            
            print("-"*80)
            print("🗂️  CONFIGURAZIONE COMPLETA:")
            for key, value in config.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for subkey, subvalue in value.items():
                        print(f"     {subkey}: {subvalue}")
                else:
                    print(f"   {key}: {value}")
        else:
            print("❌ CONFIGURAZIONE NON TROVATA per humanitas")
            
        print("="*80)
        
        # Verifica anche altri tenant per confronto
        print("🔍 VERIFICA ALTRI TENANT:")
        print("-"*80)
        
        # Lista tenant da controllare
        test_tenants = ["a0fd7600-f4f7-11ef-9315-96000228e7fe", "016-010", "test"]
        
        for tenant_name in test_tenants:
            try:
                tenant_config = ai_service.get_tenant_configuration(tenant_name, force_no_cache=True)
                if tenant_config and 'llm_model' in tenant_config:
                    current_model = tenant_config['llm_model'].get('current', 'N/A')
                    print(f"   {tenant_name}: {current_model}")
                else:
                    print(f"   {tenant_name}: No config or no llm_model")
            except Exception as e:
                print(f"   {tenant_name}: Errore - {e}")
                
    except ImportError as e:
        print(f"❌ Errore import AIConfigurationService: {e}")
    except Exception as e:
        print(f"❌ Errore generico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_humanitas_model()
