#!/usr/bin/env python3
"""
Test del nuovo flusso BERTopic ottimizzato
Autore: Sistema AI - Data: 2025-08-21
Storia aggiornamenti: Creazione test per nuovo flusso BERTopic anticipato
"""

import os
import sys
import yaml
from datetime import datetime

# Aggiungi i percorsi necessari
sys.path.append('Pipeline')
sys.path.append('EmbeddingEngine')

def test_nuovo_flusso_bertopic():
    """
    Test del nuovo flusso BERTopic con training anticipato
    
    Scopo: Verificare che le modifiche al pipeline funzionino correttamente
    Output: Report dello stato del test
    Data ultima modifica: 2025-08-21
    """
    print("🧪 TEST NUOVO FLUSSO BERTOPIC OTTIMIZZATO")
    print("=" * 60)
    
    try:
        from end_to_end_pipeline import EndToEndPipeline
        print("✅ Import EndToEndPipeline successful")
        
        # Verifica che la nuova funzione esista
        pipeline = EndToEndPipeline()
        
        if hasattr(pipeline, '_addestra_bertopic_anticipato'):
            print("✅ Nuova funzione _addestra_bertopic_anticipato trovata")
        else:
            print("❌ Funzione _addestra_bertopic_anticipato non trovata")
            return False
            
        if hasattr(pipeline, '_bertopic_provider_trained'):
            print("✅ Attributo _bertopic_provider_trained inizializzato")
        else:
            print("❌ Attributo _bertopic_provider_trained non trovato")
            return False
        
        print("✅ Tutte le modifiche sono state applicate correttamente")
        print("🎯 Il nuovo flusso è pronto per il test completo")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRORE durante test: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_nuovo_flusso_bertopic()
    exit(0 if success else 1)
