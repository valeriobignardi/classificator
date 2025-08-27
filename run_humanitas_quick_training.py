#!/usr/bin/env python3
"""
Script per training rapido Humanitas senza BERTopic
Autore: AI Assistant
Data: 23-08-2025
Descrizione: Training veloce solo LLM per popolare review queue rapidamente
"""

import os
import sys
import json
from datetime import datetime

# Assicura che la root del progetto sia nel PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Pipeline.end_to_end_pipeline import EndToEndPipeline


def main():
    """
    Training rapido solo LLM per popolare review queue
    """
    print(f"⚡ Training veloce Humanitas (solo LLM) - {datetime.now()}")
    print("🎯 Obiettivo: Popolare rapidamente review queue")
    
    try:
        # Inizializzo pipeline con BERTopic disabilitato
        print("🔧 Inizializzazione pipeline (LLM only)...")
        pipe = EndToEndPipeline(
            tenant_slug='humanitas', 
            auto_mode=False
        )
        print("✅ Pipeline inizializzata")
        
        # Disabilito BERTopic temporaneamente per velocità
        if hasattr(pipe, '_bertopic_provider'):
            pipe._bertopic_provider = None
        
        # Eseguo classificazione diretta su un subset per test
        print("🚀 Avvio classificazione diretta su subset (100 documenti)...")
        
        risultato = pipe.esegui_pipeline_completa(
            giorni_indietro=None,
            limit=100,                  # Limit per test rapido
            batch_size=16,
            interactive_mode=False,
            use_ensemble=True
            # skip_clustering rimosso - parametro non esistente
        )
        
        print("\n⚡ Training veloce completato!")
        print("📋 Risultato:")
        print(json.dumps(risultato, ensure_ascii=False, indent=2))
        
        return risultato
        
    except Exception as e:
        print(f"❌ Errore durante il training veloce: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
