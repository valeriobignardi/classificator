#!/usr/bin/env python3
"""
Script per training completo dei dati Humanitas
Autore: AI Assistant
Data: 23-08-2025
Descrizione: Esegue training completo su tutti i 5144 documenti Humanitas per popolare review queue
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
    Funzione principale per training completo Humanitas
    """
    print(f"🚀 Avvio training completo Humanitas - {datetime.now()}")
    print("📊 Target: 5144 documenti in attesa di classificazione")
    
    try:
        # Inizializzo pipeline con configurazione per elaborazione massiva
        print("🔧 Inizializzazione pipeline...")
        pipe = EndToEndPipeline(
            tenant_slug='humanitas', 
            auto_mode=False  # Modalità supervisione umana
        )
        print("✅ Pipeline inizializzata")
        
        # Eseguo pipeline completa con parametri per gestire tutti i documenti
        print("🔄 Avvio elaborazione completa...")
        print("⚙️ Parametri:")
        print("   - giorni_indietro: None (tutti i documenti)")
        print("   - limit: None (nessun limite)")
        print("   - batch_size: 32 (ottimizzato per performance)")
        print("   - use_ensemble: True (ML + LLM)")
        print("   - interactive_mode: False (automatico)")
        
        risultato = pipe.esegui_pipeline_completa(
            giorni_indietro=None,       # Processa TUTTI i documenti (non solo ultimi 7 giorni)
            limit=None,                 # Nessun limite sul numero di documenti
            batch_size=32,              # Batch più grande per performance
            interactive_mode=False,     # Non richiede input utente
            use_ensemble=True,          # Usa sia ML che LLM per classificazioni
        )
        
        print("\n🎉 Training completato con successo!")
        print("📋 Risultato:")
        print(json.dumps(risultato, ensure_ascii=False, indent=2))
        
        # Statistiche finali
        print(f"\n📈 Statistiche finali:")
        if 'total_sessions_processed' in risultato:
            print(f"   - Sessioni processate: {risultato['total_sessions_processed']}")
        if 'classification_stats' in risultato:
            stats = risultato['classification_stats']
            print(f"   - Classificazioni ML: {stats.get('ml_classifications', 0)}")
            print(f"   - Classificazioni LLM: {stats.get('llm_classifications', 0)}")
        
        print("\n✅ I casi sono ora disponibili nella review queue!")
        
        return risultato
        
    except Exception as e:
        print(f"❌ Errore durante il training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
