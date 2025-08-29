#!/bin/bash
# Script per build e deploy servizio embedding
# Autore: Valerio Bignardi
# Data: 2025-08-29

set -e  # Exit su errore

# Configurazioni
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DOCKER_SERVICE_DIR")"
IMAGE_NAME="labse-embedding-service"
CONTAINER_NAME="labse-embedding"
DEFAULT_PORT=8080

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funzioni helper
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Funzione help
show_help() {
    echo "🐳 Script gestione servizio LaBSE embedding"
    echo ""
    echo "Uso: $0 [COMANDO] [OPZIONI]"
    echo ""
    echo "COMANDI:"
    echo "  build         Builda immagine Docker"
    echo "  start         Avvia servizio"
    echo "  stop          Ferma servizio"  
    echo "  restart       Riavvia servizio"
    echo "  status        Mostra stato servizio"
    echo "  logs          Mostra logs servizio"
    echo "  health        Verifica salute servizio"
    echo "  clean         Rimuove container e immagini"
    echo "  deploy        Build + start completo"
    echo "  test          Testa funzionamento servizio"
    echo ""
    echo "OPZIONI:"
    echo "  -p, --port PORT     Porta servizio (default: $DEFAULT_PORT)"
    echo "  -f, --force         Forza operazione (per clean)"
    echo "  -d, --detach        Avvia in background"
    echo "  -h, --help          Mostra questo help"
    echo ""
    echo "ESEMPI:"
    echo "  $0 deploy              # Build e avvia servizio"
    echo "  $0 start -p 8090       # Avvia su porta 8090"  
    echo "  $0 logs                # Mostra logs in tempo reale"
    echo "  $0 clean -f            # Rimozione forzata"
}

# Verifica prerequisiti
check_prerequisites() {
    log_info "Verifica prerequisiti..."
    
    # Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker non installato o non nel PATH"
        exit 1
    fi
    
    # Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose non installato o non nel PATH"
        exit 1
    fi
    
    # Directory servizio
    if [ ! -d "$DOCKER_SERVICE_DIR" ]; then
        log_error "Directory servizio non trovata: $DOCKER_SERVICE_DIR"
        exit 1
    fi
    
    # File Docker necessari
    if [ ! -f "$DOCKER_SERVICE_DIR/Dockerfile" ]; then
        log_error "Dockerfile non trovato in: $DOCKER_SERVICE_DIR"
        exit 1
    fi
    
    if [ ! -f "$DOCKER_SERVICE_DIR/docker-compose.yml" ]; then
        log_error "docker-compose.yml non trovato in: $DOCKER_SERVICE_DIR"
        exit 1
    fi
    
    log_success "Prerequisiti verificati"
}

# Build immagine
build_image() {
    log_info "Build immagine Docker..."
    
    cd "$DOCKER_SERVICE_DIR"
    
    # Build con cache busting se necessario
    if [ "$FORCE_BUILD" = true ]; then
        docker build --no-cache -t "$IMAGE_NAME:latest" .
    else
        docker build -t "$IMAGE_NAME:latest" .
    fi
    
    if [ $? -eq 0 ]; then
        log_success "Immagine buildata: $IMAGE_NAME:latest"
    else
        log_error "Errore build immagine"
        exit 1
    fi
}

# Avvia servizio
start_service() {
    log_info "Avvio servizio embedding..."
    
    cd "$DOCKER_SERVICE_DIR"
    
    # Configura porta se specificata
    if [ ! -z "$PORT" ]; then
        export EMBEDDING_SERVICE_PORT="$PORT"
        log_info "Porta configurata: $PORT"
    fi
    
    # Avvia con docker-compose
    if [ "$DETACH" = true ]; then
        docker-compose up -d
    else
        docker-compose up
    fi
    
    if [ $? -eq 0 ]; then
        log_success "Servizio avviato"
        
        # Attendi che il servizio sia pronto
        if [ "$DETACH" = true ]; then
            log_info "Attesa avvio servizio..."
            sleep 10
            check_health
        fi
    else
        log_error "Errore avvio servizio"
        exit 1
    fi
}

# Ferma servizio
stop_service() {
    log_info "Arresto servizio embedding..."
    
    cd "$DOCKER_SERVICE_DIR"
    docker-compose down
    
    if [ $? -eq 0 ]; then
        log_success "Servizio fermato"
    else
        log_warning "Possibili errori durante arresto"
    fi
}

# Stato servizio
show_status() {
    log_info "Stato servizio embedding:"
    echo ""
    
    cd "$DOCKER_SERVICE_DIR"
    docker-compose ps
    
    echo ""
    
    # Verifica container in esecuzione
    if docker ps | grep -q "$CONTAINER_NAME"; then
        log_success "Container in esecuzione"
        
        # Statistiche risorse
        echo ""
        log_info "Utilizzo risorse:"
        docker stats "$CONTAINER_NAME" --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
        
    else
        log_warning "Container non in esecuzione"
    fi
}

# Mostra logs
show_logs() {
    log_info "Logs servizio embedding:"
    echo ""
    
    cd "$DOCKER_SERVICE_DIR"
    
    if [ "$FOLLOW_LOGS" = true ]; then
        docker-compose logs -f
    else
        docker-compose logs --tail=100
    fi
}

# Verifica salute
check_health() {
    local port="${PORT:-$DEFAULT_PORT}"
    local health_url="http://localhost:$port/health"
    
    log_info "Verifica salute servizio su porta $port..."
    
    # Attendi che il servizio risponda
    local attempts=0
    local max_attempts=30
    
    while [ $attempts -lt $max_attempts ]; do
        if curl -s -f "$health_url" > /dev/null 2>&1; then
            break
        fi
        
        attempts=$((attempts + 1))
        log_info "Tentativo $attempts/$max_attempts - attesa risposta servizio..."
        sleep 2
    done
    
    # Test salute dettagliato
    if response=$(curl -s "$health_url" 2>/dev/null); then
        echo ""
        log_success "Servizio raggiungibile:"
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
        
        # Test funzionale
        echo ""
        log_info "Test funzionale embedding..."
        test_response=$(curl -s -X POST "$health_url/../embed" \
            -H "Content-Type: application/json" \
            -d '{"texts": ["test funzionamento"]}' 2>/dev/null)
        
        if echo "$test_response" | grep -q "embeddings"; then
            log_success "Test funzionale superato"
        else
            log_warning "Test funzionale fallito o incompleto"
        fi
        
    else
        log_error "Servizio non raggiungibile su $health_url"
        exit 1
    fi
}

# Test completo
run_tests() {
    local port="${PORT:-$DEFAULT_PORT}"
    
    log_info "Esecuzione test completo servizio..."
    
    # Test health
    check_health
    
    # Test endpoint info
    echo ""
    log_info "Test endpoint /info:"
    if info_response=$(curl -s "http://localhost:$port/info" 2>/dev/null); then
        echo "$info_response" | python3 -m json.tool 2>/dev/null || echo "$info_response"
    else
        log_warning "Endpoint /info non raggiungibile"
    fi
    
    # Test embedding multipli
    echo ""
    log_info "Test embedding multipli:"
    test_data='{"texts": ["Primo test", "Secondo test", "Terzo test"], "normalize_embeddings": true}'
    
    if embed_response=$(curl -s -X POST "http://localhost:$port/embed" \
        -H "Content-Type: application/json" \
        -d "$test_data" 2>/dev/null); then
        
        # Verifica struttura risposta
        if echo "$embed_response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    embeddings = data.get('embeddings', [])
    print(f'✅ Embeddings generati: {len(embeddings)} x {len(embeddings[0]) if embeddings else 0}')
    print(f'⏱️ Tempo processamento: {data.get(\"processing_time\", \"N/A\")}s')
except Exception as e:
    print(f'❌ Errore parsing risposta: {e}')
    sys.exit(1)
" 2>/dev/null; then
            log_success "Test embedding multipli superato"
        else
            log_error "Test embedding multipli fallito"
        fi
    else
        log_error "Impossibile testare embedding multipli"
    fi
}

# Pulizia
clean_all() {
    log_info "Pulizia container e immagini..."
    
    cd "$DOCKER_SERVICE_DIR"
    
    # Ferma servizio
    docker-compose down
    
    # Rimuovi container
    if docker ps -a | grep -q "$CONTAINER_NAME"; then
        if [ "$FORCE_CLEAN" = true ]; then
            docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
        else
            docker rm "$CONTAINER_NAME" 2>/dev/null || true
        fi
    fi
    
    # Rimuovi immagine
    if docker images | grep -q "$IMAGE_NAME"; then
        if [ "$FORCE_CLEAN" = true ]; then
            docker rmi -f "$IMAGE_NAME:latest" 2>/dev/null || true
        else
            echo ""
            read -p "Rimuovere anche l'immagine $IMAGE_NAME:latest? (y/N): " confirm
            if [[ $confirm =~ ^[Yy]$ ]]; then
                docker rmi "$IMAGE_NAME:latest" 2>/dev/null || true
            fi
        fi
    fi
    
    # Pulizia volumi orfani
    docker volume prune -f > /dev/null 2>&1 || true
    
    log_success "Pulizia completata"
}

# Deploy completo
deploy_service() {
    log_info "Deploy completo servizio embedding..."
    
    # Build
    build_image
    
    # Avvia
    DETACH=true
    start_service
    
    # Test
    echo ""
    run_tests
    
    log_success "Deploy completato con successo!"
}

# Parse parametri
PORT=""
FORCE_BUILD=false
FORCE_CLEAN=false
DETACH=false
FOLLOW_LOGS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_BUILD=true
            FORCE_CLEAN=true
            shift
            ;;
        -d|--detach)
            DETACH=true
            shift
            ;;
        -F|--follow)
            FOLLOW_LOGS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        build|start|stop|restart|status|logs|health|clean|deploy|test)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Parametro sconosciuto: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
done

# Comando di default
if [ -z "$COMMAND" ]; then
    show_help
    exit 0
fi

# Esegui comando
case $COMMAND in
    build)
        check_prerequisites
        build_image
        ;;
    start)
        check_prerequisites
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        log_info "Riavvio servizio..."
        stop_service
        sleep 2
        check_prerequisites
        start_service
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    health)
        check_health
        ;;
    test)
        run_tests
        ;;
    clean)
        clean_all
        ;;
    deploy)
        check_prerequisites
        deploy_service
        ;;
    *)
        log_error "Comando non riconosciuto: $COMMAND"
        show_help
        exit 1
        ;;
esac
