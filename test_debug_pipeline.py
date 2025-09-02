#!/usr/bin/env python3
"""
Test Debug Pipeline

Autore: Valeri        # Legge le conversazioni e raggruppa per sessioni
        print("� [TEST] Lettura conversazioni dal database...")
        conversazioni = lettore.leggi_conversazioni()
        
        # Usa SessionAggregator per convertire nel formato corretto
        print("🔄 [TEST] Aggregazione sessioni...")
        sessioni_aggregate = pipeline.aggregator._raggruppa_per_sessioni(conversazioni)
        sessioni_filtrate = pipeline.aggregator.filtra_sessioni_vuote(sessioni_aggregate)
        
        # Prendi solo le prime 100 sessioni
        session_keys = list(sessioni_filtrate.keys())[:100]
        sessioni_dict = {k: sessioni_filtrate[k] for k in session_keys}
        
        print(f"📊 [TEST] Sessioni caricate: {len(sessioni_dict)}")
        
        if len(sessioni_dict) == 0:
            print("❌ ERRORE: Nessuna sessione trovata nel database")
            return Falseeazione: 2025-09-02
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
        
        # Carica ESATTAMENTE 100 sessioni dal database
        from LettoreConversazioni.lettore import LettoreConversazioni
        
        print("🔍 [TEST] Caricamento sessioni da database...")
        print("🎯 [TEST] Usando schema 'humanitas' per le conversazioni")
        lettore = LettoreConversazioni(tenant=tenant, schema='humanitas')
        
        # Legge le conversazioni e raggruppa per sessioni
        print("� [TEST] Lettura conversazioni dal database...")
        conversazioni = lettore.leggi_conversazioni()
        
        # Raggruppa per sessioni e prendi le prime 100
        sessioni_dict = {}
        for conv in conversazioni:
            session_id = conv[0]  # session_id è il primo elemento
            if session_id not in sessioni_dict:
                sessioni_dict[session_id] = []
                # Limita a 100 sessioni
                if len(sessioni_dict) > 100:
                    break
            sessioni_dict[session_id].append(conv)
        
        print(f"📊 [TEST] Sessioni caricate: {len(sessioni_dict)}")
        
        if len(sessioni_dict) == 0:
            print("❌ ERRORE: Nessuna sessione trovata nel database")
            return False
        
        # Test con 100 sessioni per vedere clustering ottimizzato
        print("🔍 [TEST] Esecuzione classificazione con debug attivato...")
        print(f"🔍 [TEST] Sessioni: {len(sessioni_dict)} (dovrebbe usare clustering ottimizzato)")
        
        # Usa il metodo diretto di classificazione
        risultati = pipeline.classifica_e_salva_sessioni(
            sessioni=sessioni_dict,
            batch_size=32,
            use_ensemble=True,
            optimize_clusters=True,
            force_review=False
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
