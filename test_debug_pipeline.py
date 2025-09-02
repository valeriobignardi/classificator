#!/usr/bin/env python3
"""
Test Debug Pipeline

Autore: Valerio Bignardi  
Data creazione: 2025-09-02
Storia aggiornamenti:
- 2025-09-02: Creazione script per testare il sistema di debug

Scopo: Esegue una pipeline ridotta per testare il sistema di debug
       e identificare la causa dei casi LLM_STRUCTURED
"""

import os
import sys

# Aggiungi path per import
sys.path.append('/home/ubuntu/classificatore')
sys.path.append('/home/ubuntu/classificatore/Pipeline')

def test_debug_pipeline():
    """
    Scopo: Testa il sistema di debug con una pipeline ridotta
    
    Output: Log dettagliato del flusso di esecuzione
    
    Ultimo aggiornamento: 2025-09-02
    """
    print("🔍 [TEST DEBUG PIPELINE] Avvio test sistema debug...")
    
    try:
        # Test import debug utility
        from Pipeline.debug_pipeline import debug_pipeline, debug_flow, debug_exception
        
        debug_pipeline("test_debug_pipeline", "ENTRY - Test avviato", {
            "test_type": "debug_system_validation"
        }, "ENTRY")
        
        # Test import EndToEndPipeline 
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        from tenant import Tenant
        
        # Crea tenant Humanitas
        tenant = Tenant.from_slug("humanitas")
        debug_pipeline("test_debug_pipeline", "Tenant creato", {
            "tenant_name": tenant.tenant_name,
            "tenant_id": tenant.tenant_id
        }, "SUCCESS")
        
        # Inizializza pipeline
        pipeline = EndToEndPipeline(tenant=tenant)
        debug_pipeline("test_debug_pipeline", "Pipeline inizializzata", {
            "tenant_slug": pipeline.tenant_slug
        }, "SUCCESS")
        
        # Test con 100 sessioni per vedere clustering ottimizzato
        print("🔍 [TEST] Esecuzione pipeline completa con debug attivato...")
        print("🔍 [TEST] Limite: 100 sessioni per testare clustering ottimizzato")
        
        risultati = pipeline.esegui_pipeline_completa(
            giorni_indietro=7,  # 7 giorni per avere più sessioni
            limit=100,          # 100 sessioni per superare il threshold di 10
            interactive_mode=False,  # No review interattiva per test
            use_ensemble=True,
            force_full_extraction=False
        )
        
        debug_pipeline("test_debug_pipeline", "Pipeline completata", {
            "risultati_keys": list(risultati.keys()) if isinstance(risultati, dict) else "Non dict"
        }, "SUCCESS")
        
        print("🔍 [TEST DEBUG PIPELINE] Test completato con successo!")
        return True
        
    except Exception as e:
        print(f"❌ [TEST DEBUG PIPELINE] Errore: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Avvio test sistema debug pipeline...")
    success = test_debug_pipeline()
    
    if success:
        print("✅ Test debug completato con successo!")
        print("🔍 Controlla l'output sopra per identificare dove si verifica cluster_metadata=None")
    else:
        print("❌ Test debug fallito!")
        
    print("🏁 Fine test debug pipeline")
