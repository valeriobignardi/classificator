#!/bin/bash

#
# Script di inizializzazione volume HuggingFace
# Autore: Valerio Bignardi
# Data: 2025-08-29
#
# Scopo:
# - Scarica il modello LaBSE nel volume Docker dedicato
# - Inizializza correttamente la cache HuggingFace
# - Evita problemi di permessi durante l'esecuzione
#
# Funzionamento:
# 1. Crea un container temporaneo per scaricare il modello
# 2. Usa il volume huggingface-cache per salvare il modello
# 3. Il container principale utilizzerà questo volume pre-popolato
#
# Data ultima modifica: 2025-08-29
#

echo "🚀 Inizializzazione cache HuggingFace per LaBSE..."
echo "========================================================"
echo "📋 Parametri configurazione:"
echo "   Volume: docker-embedding-service_huggingface-cache"
echo "   Modello: sentence-transformers/LaBSE"
echo ""

echo "🔍 Verifica esistenza volume..."
# Crea il volume se non esiste già
echo "💾 Creazione volume docker-embedding-service_huggingface-cache..."
docker volume create docker-embedding-service_huggingface-cache

if [ $? -eq 0 ]; then
    echo "✅ Volume creato con successo"
else
    echo "❌ Errore nella creazione del volume"
    exit 1
fi

echo ""
echo "📥 Download modello nel volume..."
echo "⚠️  Questo processo potrebbe richiedere alcuni minuti..."
echo ""

# Esegue il download del modello in un container temporaneo
docker run --rm \
    -e HF_HOME=/cache \
    -e TRANSFORMERS_CACHE=/cache \
    -e HF_HUB_CACHE=/cache/hub \
    -v docker-embedding-service_huggingface-cache:/cache \
    python:3.11-slim \
    bash -c "
        pip install sentence-transformers torch && 
        python3 -c \"
import os
import logging
from sentence_transformers import SentenceTransformer

# Configura logging
logging.basicConfig(level=logging.INFO)

# Verifica configurazione cache
cache_dir = os.getenv('HF_HOME', '/cache')
print(f'Directory cache: {cache_dir}')

try:
    # Carica e inizializza il modello
    print('Caricamento modello sentence-transformers/LaBSE...')
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    # Test funzionamento
    print('Test funzionamento modello...')
    test_sentence = 'This is a test sentence'
    embedding = model.encode(test_sentence)
    print(f'Test completato. Dimensione embedding: {len(embedding)}')
    
    print('Modello LaBSE scaricato e testato con successo!')
    print(f'Cache salvata in: {cache_dir}')
    
except Exception as e:
    print(f'Errore durante il download: {str(e)}')
    exit(1)
        \"
    "

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Inizializzazione completata con successo!"
    echo "📊 Verifica contenuto volume:"
    docker run --rm \
        -v docker-embedding-service_huggingface-cache:/cache \
        python:3.11-slim \
        find /cache -type f -name "*.json" -o -name "*.bin" -o -name "*.safetensors" | head -10
    echo ""
    echo "🚀 Il volume è pronto per l'uso nel servizio di embedding!"
else
    echo "❌ Errore durante l'inizializzazione. Verifica i log sopra."
    exit 1
fi
