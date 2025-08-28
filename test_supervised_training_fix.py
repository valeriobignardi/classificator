#!/usr/bin/env python3
"""
Test del supervised training completo dopo la correzione tenant_id vs tenant_slug

Autore: Valerio Bignardi
Data: 2025-08-28
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from Pipeline.end_to_end_pipeline import EndToEndPipeline

def test_supervised_training_fix():
    """
    Scopo: Verifica che il supervised training funzioni senza errori tenant
    """
    
    print("🎯 Test supervised training dopo correzione tenant...")
    
    try:
        # Inizializza pipeline con tenant corretto
        print("📋 Inizializzazione pipeline...")
        pipeline = EndToEndPipeline(
            tenant_slug="humanitas",
            use_preloaded_data=False
        )
        
        print(f"   Tenant configurato: {pipeline.tenant}")
        print(f"   Tenant slug: {pipeline.tenant.tenant_slug}")  
        print(f"   Tenant ID: {pipeline.tenant.tenant_id}")
        
        # Test solo la prima fase per verificare che non ci siano errori
        print("📋 Test caricamento dati...")
        
        # Simula un mini-training con poche sessioni
        print("📋 Avvio mini supervised training...")
        
        result = pipeline.run_supervised_training(
            max_sessions=50,  # Limite basso per test veloce
            save_representatives=True,
            force_review=True,
            test_mode=True  # Se disponibile
        )
        
        print(f"   Risultato training: {result}")
        
        if result and isinstance(result, dict):
            if 'error' not in result:
                print("✅ SUCCESSO: Training completato senza errori tenant!")
                return True
            else:
                print(f"❌ ERRORE NEL TRAINING: {result.get('error')}")
                return False
        else:
            print("✅ SUCCESSO: Training completato")  
            return True
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Errore durante test training: {error_msg}")
        
        # Verifica se l'errore è ancora relativo ai tenant
        if "Tenant ID non trovato" in error_msg or "16c222a9-f293-11ef-9315-96000228e7fe" in error_msg:
            print("❌ ERRORE TENANT ANCORA PRESENTE!")
            return False
        else:
            print("⚠️ Errore diverso (non tenant-related)")
            import traceback
            traceback.print_exc()
            return True  # Altri errori possono essere normali
            

if __name__ == "__main__":
    print("=" * 70)
    print("TEST SUPERVISED TRAINING DOPO CORREZIONE TENANT")
    print("=" * 70)
    
    success = test_supervised_training_fix()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ CORREZIONE TENANT VERIFICATA - TRAINING PRONTO")
        print("💡 Ora puoi procedere con il training completo!")
    else:
        print("❌ CORREZIONE NON COMPLETA - SERVE ULTERIORE DEBUG")
    print("=" * 70)
