#!/usr/bin/env python3
"""
Script di test per debuggare la funzione _select_representatives_for_human_review
Autore: Valerio Bignardi
Data: 2025-01-17
"""

import sys
import os
import yaml
from datetime import datetime

# Aggiungi il path del progetto
sys.path.append('/home/ubuntu/classificatore')

from Pipeline.end_to_end_pipeline import EndToEndPipeline
from Utils.tenant import Tenant

def test_select_representatives():
    """
    Test della funzione _select_representatives_for_human_review con debug dettagliato
    """
    print("🔍 Avvio test funzione _select_representatives_for_human_review")
    
    # Crea il file di log vuoto
    debug_log_path = "/home/ubuntu/classificatore/rappresentanti.log"
    with open(debug_log_path, "w", encoding="utf-8") as f:
        f.write(f"DEBUG LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
    
    try:
        # Inizializza la pipeline
        print("⚙️ Inizializzazione pipeline...")
        pipeline = EndToEndPipeline()
        
        # Crea un tenant di test
        tenant = Tenant(
            tenant_id="015007d9-d413-11ef-86a5-96000228e7fe",
            tenant_name="Humanitas Test",
            tenant_slug="humanitas",
            tenant_database="humanitas_015007d9-d413-11ef-86a5-96000228e7fe",
            tenant_status=1
        )
        pipeline.tenant = tenant
        
        # Simula dati representatives con alcuni cluster di test
        representatives = {
            "0": [{"session_id": "sess_001"}, {"session_id": "sess_002"}],  # 2 reps
            "1": [{"session_id": "sess_003"}],  # 1 rep (troppo piccolo)
            "2": [{"session_id": "sess_004"}, {"session_id": "sess_005"}, {"session_id": "sess_006"}],  # 3 reps
            "3": [{"session_id": "sess_007"}, {"session_id": "sess_008"}, {"session_id": "sess_009"}, {"session_id": "sess_010"}],  # 4 reps
        }
        
        suggested_labels = {
            "0": "categoria_A",
            "1": "categoria_B", 
            "2": "categoria_C",
            "3": "categoria_D"
        }
        
        print(f"📊 Dati di test:")
        print(f"  - Representatives: {len(representatives)} cluster")
        print(f"  - Total sessions: {sum(len(reps) for reps in representatives.values())}")
        print(f"  - Suggested labels: {len(suggested_labels)}")
        
        # Chiama la funzione con debug
        print("🚀 Chiamata _select_representatives_for_human_review...")
        
        result_reps, result_stats = pipeline._select_representatives_for_human_review(
            representatives=representatives,
            suggested_labels=suggested_labels,
            max_sessions=50,
            confidence_threshold=0.7,
            force_review=False,
            disagreement_threshold=0.3,
            all_sessions=None
        )
        
        print("✅ Funzione completata!")
        print(f"📋 Risultati:")
        print(f"  - Cluster selezionati: {len(result_reps)}")
        print(f"  - Stats: {result_stats}")
        
        # Mostra contenuto log
        print(f"\n📄 Debug log scritto in: {debug_log_path}")
        with open(debug_log_path, "r", encoding="utf-8") as f:
            log_content = f.read()
            print("📄 Contenuto log (primi 2000 caratteri):")
            print("-" * 60)
            print(log_content[:2000])
            if len(log_content) > 2000:
                print("... (troncato)")
            print("-" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_select_representatives()
    if success:
        print("✅ Test completato con successo!")
    else:
        print("❌ Test fallito!")
        sys.exit(1)
