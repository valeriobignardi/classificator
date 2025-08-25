#!/usr/bin/env python3
"""
Script per training ottimizzato Humanitas con BERTopic
Autore: AI Assistant
Data: 24-08-2025
Descrizione: Training BERTopic ottimizzato per grandi dataset
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
    Training ottimizzato con BERTopic per Humanitas
    """
    print(f"🚀 Training ottimizzato Humanitas con BERTopic - {datetime.now()}")
    print("⚡ Configurazione ottimizzata per grandi dataset")
    
    try:
        # Inizializzo pipeline ottimizzata
        print("🔧 Inizializzazione pipeline ottimizzata...")
        pipe = EndToEndPipeline(
            tenant_slug='humanitas', 
            auto_mode=False
        )
        print("✅ Pipeline inizializzata")
        
        # Eseguo training con subset iniziale per BERTopic efficiente
        print("🔄 Avvio training con subset ottimizzato...")
        print("⚙️ Parametri ottimizzati:")
        print("   - giorni_indietro: 30 (subset recente per training iniziale)")
        print("   - limit: 1000 (subset gestibile per BERTopic)")
        print("   - batch_size: 16 (bilanciato per GPU)")
        print("   - use_ensemble: True (ML + LLM)")
        print("   - interactive_mode: False")
        
        risultato = pipe.esegui_pipeline_completa(
            giorni_indietro=30,         # Subset recente per training iniziale
            limit=1000,                 # Limit gestibile per BERTopic
            batch_size=16,              # Batch ottimizzato
            interactive_mode=False,
            use_ensemble=True,
        )
        
        print("\n🎉 Training ottimizzato completato!")
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
        print("💡 Una volta confermato il funzionamento, potrà essere eseguito su tutto il dataset")
        
        return risultato
        
    except Exception as e:
        print(f"❌ Errore durante il training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
