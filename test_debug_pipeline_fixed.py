#!/usr/bin/env python3
"""
Test Debug Pipeline - FIXED VERSION

Autore: Valerio Bignardi  
Data creazione: 2025-09-02
Storia aggiornamenti:
- 2025-09-02: Creazione script corretto per testare il sistema di debug

Scopo: Esegue una pipeline ridotta per testare il sistema di debug
       e identificare la causa dei casi LLM_STRUCTURED usando SessionAggregator
"""

import os
import sys

# Aggiungi path per import
sys.path.append('/home/ubuntu/classificatore')
sys.path.append('/home/ubuntu/classificatore/Pipeline')

def test_debug_pipeline():
    """
    Scopo: Testa il sistema di debug con una pipeline ridotta usando SessionAggregator
    
    Output: Log dettagliato del flusso di esecuzione con sessioni formattate correttamente
    
    Ultimo aggiornamento: 2025-09-02
    """
    print("🔍 [TEST DEBUG PIPELINE FIXED] Avvio test sistema debug...")
    
    try:
        # Test import debug utility
        from Pipeline.debug_pipeline import debug_pipeline, debug_flow, debug_exception
        
        debug_pipeline("test_debug_pipeline_fixed", "ENTRY - Test avviato", {
            "test_type": "debug_system_validation"
        }, "ENTRY")
        
        # Test import EndToEndPipeline 
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        from tenant import Tenant
        
        # Crea tenant Humanitas
        tenant = Tenant.from_slug("humanitas")
        debug_pipeline("test_debug_pipeline_fixed", "Tenant creato", {
            "tenant_name": tenant.tenant_name,
            "tenant_id": tenant.tenant_id
        }, "SUCCESS")
        
        # Inizializza pipeline
        pipeline = EndToEndPipeline(tenant=tenant)
        debug_pipeline("test_debug_pipeline_fixed", "Pipeline inizializzata", {
            "tenant_slug": pipeline.tenant_slug
        }, "SUCCESS")
        
        # ✅ USA SessionAggregator per ottenere sessioni formattate correttamente
        from Preprocessing.session_aggregator import SessionAggregator
        
        print("🔍 [TEST FIXED] Caricamento sessioni tramite SessionAggregator...")
        print("🎯 [TEST FIXED] Usando schema 'humanitas' per le conversazioni")
        
        # Usa SessionAggregator per ottenere le sessioni nel formato corretto
        aggregator = SessionAggregator(tenant=tenant, schema='humanitas')
        
        print("📖 [TEST FIXED] Estrazione sessioni aggregate...")
        sessioni_dict = aggregator.estrai_sessioni_aggregate(
            limit=100           # Limita a 100 sessioni
        )
        
        print(f"📊 [TEST FIXED] Sessioni aggregate caricate: {len(sessioni_dict)}")
        
        if len(sessioni_dict) == 0:
            print("❌ ERRORE: Nessuna sessione trovata nel database")
            return False
        
        # Debug del formato delle prime 2 sessioni
        print("🔍 [TEST FIXED] Debug formato sessioni:")
        for i, (session_id, session_data) in enumerate(list(sessioni_dict.items())[:2]):
            print(f"📋 [TEST DEBUG] Sessione {i+1}: {session_id}")
            print(f"   📝 Keys: {list(session_data.keys())}")
            if 'testo_completo' in session_data:
                print(f"   ✅ testo_completo presente, length: {len(session_data['testo_completo'])}")
                print(f"   📄 Preview: {session_data['testo_completo'][:200]}...")
            else:
                print(f"   ❌ MANCA testo_completo!")
                print(f"   📄 Session data keys: {list(session_data.keys())}")
                break
        
        # Test con 100 sessioni per vedere clustering ottimizzato
        print("🔍 [TEST FIXED] Esecuzione classificazione con debug attivato...")
        print(f"🔍 [TEST FIXED] Sessioni: {len(sessioni_dict)} (dovrebbe usare clustering ottimizzato)")
        
        # Usa il metodo diretto di classificazione
        risultati = pipeline.classifica_e_salva_sessioni(
            sessioni=sessioni_dict,
            batch_size=32,
            use_ensemble=True,
            optimize_clusters=True,
            force_review=False
        )
        
        debug_pipeline("test_debug_pipeline_fixed", "Pipeline completata", {
            "risultati_keys": list(risultati.keys()) if isinstance(risultati, dict) else "Non dict"
        }, "SUCCESS")
        
        print("🔍 [TEST DEBUG PIPELINE FIXED] Test completato con successo!")
        return True
        
    except Exception as e:
        print(f"❌ [TEST DEBUG PIPELINE FIXED] Errore: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Avvio test sistema debug pipeline FIXED...")
    success = test_debug_pipeline()
    
    if success:
        print("✅ Test debug FIXED completato con successo!")
        print("🔍 Controlla l'output sopra per identificare dove si verifica cluster_metadata=None")
    else:
        print("❌ Test debug FIXED fallito!")
        
    print("🏁 Fine test debug pipeline FIXED")
