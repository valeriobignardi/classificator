#!/usr/bin/env python3
"""
Script per il download del modello LaBSE durante la build dell'immagine Docker
Autore: Valerio Bignardi
Data: 2025-08-29

Scopo:
- Scarica il modello sentence-transformers/LaBSE durante la build dell'immagine
- Evita problemi di permessi e ritardi durante l'avvio del container
- Permette avvio istantaneo del servizio

Parametri di input: 
- MODEL_NAME (da variabile ambiente, default: sentence-transformers/LaBSE)

Parametri di output:
- Modello scaricato nella cache di Hugging Face

Valore di ritorno:
- 0: Successo
- 1: Errore nel download

Data ultima modifica: 2025-08-29
"""

import os
import sys
from sentence_transformers import SentenceTransformer

def download_labse_model():
    """
    Scarica il modello LaBSE nella cache locale
    
    Returns:
        bool: True se download completato con successo
    """
    try:
        model_name = os.getenv("MODEL_NAME", "sentence-transformers/LaBSE")
        cache_dir = os.getenv("TRANSFORMERS_CACHE", "/app/cache")
        
        print(f"🚀 Avvio download modello: {model_name}")
        print(f"📂 Directory cache: {cache_dir}")
        
        # Scarica il modello
        model = SentenceTransformer(model_name)
        
        print(f"✅ Modello {model_name} scaricato con successo!")
        print(f"📊 Dimensione embedding: {model.get_sentence_embedding_dimension()}")
        print(f"🔧 Device disponibili: {model.device}")
        
        # Test rapido per verificare che funzioni
        test_embedding = model.encode(["Test download modello"])
        print(f"🧪 Test embedding completato: shape {test_embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore durante il download del modello: {e}")
        return False

if __name__ == "__main__":
    success = download_labse_model()
    sys.exit(0 if success else 1)
