#!/bin/bash
# Script di entrypoint per il container backend
# Autore: GitHub Copilot
# Data: 2025-09-20
# Scopo: Gestione avvio servizi e controlli pre-esecuzione

set -e

echo "🚀 Avvio container backend del sistema di classificazione Humanitas"
echo "📅 $(date)"
echo "🏠 Directory di lavoro: $(pwd)"
echo "👤 Utente: $(whoami)"

# Funzione per logging strutturato
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Controllo connessioni database
check_database_connection() {
    log "🔍 Controllo connessioni database..."
    
    # Controllo MySQL
    if [ -n "${MYSQL_HOST:-}" ]; then
        log "🗄️ Verifica connessione MySQL su ${MYSQL_HOST}:${MYSQL_PORT:-3306}"
        timeout 10s python3 -c "
import mysql.connector
try:
    conn = mysql.connector.connect(
        host='${MYSQL_HOST}',
        port='${MYSQL_PORT:-3306}',
        user='${MYSQL_USER:-root}',
        password='${MYSQL_PASSWORD}',
        database='${MYSQL_DATABASE:-TAG}'
    )
    conn.close()
    print('✅ MySQL: Connessione riuscita')
except Exception as e:
    print(f'❌ MySQL: Errore connessione - {e}')
    exit(1)
" || log "⚠️ MySQL non disponibile, continuando..."
    fi
    
    # Controllo MongoDB
    if [ -n "${MONGODB_URL:-}" ]; then
        log "🍃 Verifica connessione MongoDB su ${MONGODB_URL}"
        timeout 10s python3 -c "
import pymongo
try:
    client = pymongo.MongoClient('${MONGODB_URL}', serverSelectionTimeoutMS=5000)
    client.server_info()
    print('✅ MongoDB: Connessione riuscita')
except Exception as e:
    print(f'❌ MongoDB: Errore connessione - {e}')
    exit(1)
" || log "⚠️ MongoDB non disponibile, continuando..."
    fi
}

# Controllo servizi LLM
check_llm_services() {
    log "🤖 Controllo servizi LLM..."
    
    # Controllo Ollama
    if [ -n "${OLLAMA_URL:-}" ]; then
        log "🦙 Verifica Ollama su ${OLLAMA_URL}"
        curl -f -s "${OLLAMA_URL}/api/tags" > /dev/null || \
            log "⚠️ Ollama non raggiungibile, ma continuando..."
    fi
    
    # Controllo OpenAI API key
    if [ -n "${OPENAI_API_KEY:-}" ]; then
        log "🔑 OpenAI API Key configurata"
    fi
}

# Controllo GPU availability
check_gpu_availability() {
    log "🎮 Controllo disponibilità GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        if [ "$GPU_COUNT" -gt 0 ]; then
            log "✅ GPU disponibili: $GPU_COUNT"
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        else
            log "⚠️ CUDA installato ma nessuna GPU rilevata"
        fi
    else
        log "ℹ️ CUDA/GPU non disponibile, utilizzerò CPU"
        export CUDA_VISIBLE_DEVICES=""
    fi
}

# Inizializzazione directory e permessi
setup_directories() {
    log "📁 Configurazione directory di lavoro..."
    
    # Crea directory se non esistono
    mkdir -p debug_logs training_logs semantic_cache bertopic backup
    
    # Verifica permessi
    if [ ! -w "." ]; then
        log "❌ Errore: Directory /app non scrivibile"
        exit 1
    fi
    
    log "✅ Directory configurate correttamente"
}

# Pre-caricamento modelli critici (opzionale)
preload_models() {
    if [ "${PRELOAD_MODELS:-false}" = "true" ]; then
        log "🧠 Pre-caricamento modelli di ML..."
        python3 -c "
try:
    import sentence_transformers
    import transformers
    print('✅ Librerie ML caricate correttamente')
except Exception as e:
    print(f'⚠️ Errore caricamento librerie ML: {e}')
" || true
    fi
}

# Controllo configurazione
check_configuration() {
    log "⚙️ Controllo file di configurazione..."
    
    if [ ! -f "config.yaml" ]; then
        log "❌ File config.yaml mancante"
        exit 1
    fi
    
    if [ ! -f "server.py" ]; then
        log "❌ File server.py mancante"
        exit 1
    fi
    
    log "✅ File di configurazione presenti"
}

# Gestione segnali per shutdown graceful
trap 'log "🛑 Ricevuto segnale di terminazione, arresto graceful..."; exit 0' SIGTERM SIGINT

# Main execution
main() {
    log "🔧 Avvio controlli pre-esecuzione..."
    
    setup_directories
    check_configuration
    check_gpu_availability
    check_database_connection
    check_llm_services
    preload_models
    
    log "✅ Tutti i controlli completati con successo"
    log "🚀 Avvio applicazione: $@"
    
    # Esegui il comando passato
    exec "$@"
}

# Esegui main se script chiamato direttamente
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi