#!/bin/bash
"""
Script di inizializzazione volume HuggingFace
Autore: Valerio Bignardi
Data: 2025-08-29

Scopo:
- Scarica il modello LaBSE nel volume Docker dedicato
- Inizializza correttamente la cache HuggingFace
- Evita problemi di permessi durante l'esecuzione

Funzionamento:
1. Crea un container temporaneo per scaricare il modello
2. Usa il volume huggingface-cache per salvare il modello
3. Il container principale utilizzerà questo volume pre-popolato

Data ultima modifica: 2025-08-29
"""

set -e

echo "🚀 Inizializzazione cache HuggingFace per LaBSE..."
echo "========================================================"

# Nome del volume
VOLUME_NAME="docker-embedding-service_huggingface-cache"
MODEL_NAME="sentence-transformers/LaBSE"

echo "📋 Parametri configurazione:"
echo "   Volume: $VOLUME_NAME"
echo "   Modello: $MODEL_NAME"
echo ""

# Verifica se il volume esiste
echo "🔍 Verifica esistenza volume..."
if ! docker volume inspect $VOLUME_NAME >/dev/null 2>&1; then
    echo "💾 Creazione volume $VOLUME_NAME..."
    docker volume create $VOLUME_NAME
    echo "✅ Volume creato con successo"
else
    echo "✅ Volume già esistente"
fi

echo ""
echo "📥 Download modello nel volume..."
echo "⚠️  Questo processo potrebbe richiedere alcuni minuti..."
echo ""

# Scarica il modello nel volume usando un container temporaneo
docker run --rm \
    -v $VOLUME_NAME:/cache \
    -e HF_HOME=/cache \
    -e TRANSFORMERS_CACHE=/cache \
    -e HF_HUB_CACHE=/cache \
    python:3.11-slim \
    bash -c "
        pip install --no-cache-dir sentence-transformers torch &&
        python -c '
from sentence_transformers import SentenceTransformer
import os

print(\"🚀 Avvio download modello: $MODEL_NAME\")
print(f\"📂 Directory cache: {os.getenv(\"HF_HOME\", \"/cache\")}\")

try:
    model = SentenceTransformer(\"$MODEL_NAME\")
    print(\"✅ Modello scaricato con successo!\")
    print(f\"📊 Dimensione embedding: {model.get_sentence_embedding_dimension()}\")
    
    # Test rapido
    test_embedding = model.encode([\"Test inizializzazione cache\"])
    print(f\"🧪 Test embedding completato: shape {test_embedding.shape}\")
    print(\"🎉 Cache inizializzata correttamente!\")
    
except Exception as e:
    print(f\"❌ Errore durante inizializzazione: {e}\")
    exit(1)
        '
    "

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Inizializzazione completata con successo!"
    echo "📋 Verifica contenuto volume:"
    docker run --rm -v $VOLUME_NAME:/cache alpine ls -la /cache/
    echo ""
    echo "🚀 Ora puoi avviare il servizio con:"
    echo "   ./scripts/manage_service.sh start -p 8081"
else
    echo ""
    echo "❌ Inizializzazione fallita!"
    exit 1
fi
