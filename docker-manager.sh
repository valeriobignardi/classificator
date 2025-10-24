#!/bin/bash
# Script di gestione Docker per il sistema di classificazione
# Autore: GitHub Copilot
# Data: 2025-09-22
# Scopo: Automatizzare build, deploy e gestione dei container

set -e

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funzioni di utilità
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

# Verifica prerequisiti
check_prerequisites() {
    log_info "Controllo prerequisiti..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker non trovato. Installa Docker prima di continuare."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose non trovato. Installa Docker Compose prima di continuare."
        exit 1
    fi
    
    # Verifica che Docker daemon sia in esecuzione
    if ! docker info &> /dev/null; then
        log_error "Docker daemon non in esecuzione. Avvia Docker prima di continuare."
        exit 1
    fi
    
    log_success "Prerequisiti verificati"
}

# Crea file .env se non esiste
setup_environment() {
    if [ ! -f .env ]; then
        log_info "Creazione file .env da template..."
        cp .env.example .env
        log_warning "Modifica il file .env con le tue configurazioni prima di continuare"
        log_warning "Particolare attenzione a: OPENAI_API_KEY, password database"
        read -p "Premi ENTER per continuare dopo aver configurato .env..."
    fi
}

# Build delle immagini
build_images() {
    log_info "Build delle immagini Docker..."
    
    log_info "Building backend image..."
    docker build -f Dockerfile.backend -t classificatore-backend:latest .
    
    log_info "Building frontend image..."
    docker build -f Dockerfile.frontend -t classificatore-frontend:latest .
    
    log_success "Build completato"
}

# Avvio servizi
start_services() {
    log_info "Avvio servizi Docker..."
    docker-compose up -d
    log_success "Servizi avviati"
    
    # Attendi che i servizi siano pronti
    log_info "Attendo che i servizi siano pronti..."
    sleep 10
    
    # Verifica health dei servizi
    check_health
}

# Arresto servizi
stop_services() {
    log_info "Arresto servizi Docker..."
    docker-compose down
    log_success "Servizi arrestati"
}

# Restart servizi
restart_services() {
    log_info "Riavvio servizi Docker..."
    stop_services
    start_services
    log_success "Servizi riavviati"
}

# Verifica health dei servizi
check_health() {
    log_info "Controllo stato servizi..."
    
    # Verifica backend
    if curl -f -s http://localhost:5000/health > /dev/null; then
        log_success "Backend: OK"
    else
        log_warning "Backend: Non raggiungibile"
    fi
    
    # Verifica frontend
    if curl -f -s http://localhost:3005/health > /dev/null; then
        log_success "Frontend: OK"
    else
        log_warning "Frontend: Non raggiungibile"
    fi
    
    # Mostra status container
    log_info "Status container:"
    docker-compose ps
}

# Visualizza logs
show_logs() {
    local service="${1:-}"
    
    if [ -n "$service" ]; then
        log_info "Logs per servizio: $service"
        docker-compose logs -f "$service"
    else
        log_info "Logs di tutti i servizi:"
        docker-compose logs -f
    fi
}

# Pulizia completa
cleanup() {
    log_warning "ATTENZIONE: Questa operazione rimuoverà tutti i container e volumi!"
    read -p "Sei sicuro? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Pulizia in corso..."
        
        # Arresta e rimuovi container
        docker-compose down -v --remove-orphans
        
        # Rimuovi immagini
        docker rmi classificatore-backend:latest classificatore-frontend:latest 2>/dev/null || true
        
        # Rimuovi volumi
        docker volume rm classificatore-semantic-cache classificatore-training-logs 2>/dev/null || true
        
        log_success "Pulizia completata"
    else
        log_info "Pulizia annullata"
    fi
}

# Backup database
backup_databases() {
    local backup_dir="./backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    log_info "Backup database in corso..."
    log_warning "NOTA: I database sono servizi locali, non containerizzati."
    log_info "Per backup database locali usa comandi diretti del sistema."
    
    log_success "Backup directory creata: $backup_dir"
}

# Monitoraggio risorse
monitor_resources() {
    log_info "Monitoraggio risorse container:"
    docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
}

# Setup iniziale completo
initial_setup() {
    log_info "Setup iniziale del sistema di classificazione..."
    
    check_prerequisites
    setup_environment
    build_images
    start_services
    
    log_success "Setup iniziale completato!"
    log_info "Accedi a:"
    log_info "  - Frontend: http://localhost:3005"
    log_info "  - Backend API: http://localhost:5000"
    log_info "Servizi esterni utilizzati:"
    log_info "  - MongoDB: localhost:27017"
    log_info "  - MySQL: localhost:3306"
    log_info "  - Ollama: localhost:11434"
}

# Menu principale
show_menu() {
    echo
    echo "=== Sistema di Classificazione - Gestione Docker ==="
    echo "1) Setup iniziale completo"
    echo "2) Build immagini"
    echo "3) Avvia servizi"
    echo "4) Arresta servizi"
    echo "5) Riavvia servizi"
    echo "6) Verifica stato"
    echo "7) Mostra logs (tutti)"
    echo "8) Mostra logs (specifico servizio)"
    echo "9) Monitoraggio risorse"
    echo "10) Backup database"
    echo "11) Pulizia completa"
    echo "12) Esci"
    echo
}

# Main
main() {
    if [ $# -eq 0 ]; then
        # Modalità interattiva
        while true; do
            show_menu
            read -p "Seleziona un'opzione (1-12): " choice
            
            case $choice in
                1) initial_setup ;;
                2) build_images ;;
                3) start_services ;;
                4) stop_services ;;
                5) restart_services ;;
                6) check_health ;;
                7) show_logs ;;
                8) 
                    read -p "Nome servizio (backend/frontend): " service
                    show_logs "$service"
                    ;;
                9) monitor_resources ;;
                10) backup_databases ;;
                11) cleanup ;;
                12) 
                    log_info "Uscita..."
                    exit 0
                    ;;
                *) 
                    log_error "Opzione non valida"
                    ;;
            esac
            
            echo
            read -p "Premi ENTER per continuare..."
        done
    else
        # Modalità comando
        case "$1" in
            setup) initial_setup ;;
            build) build_images ;;
            start) start_services ;;
            stop) stop_services ;;
            restart) restart_services ;;
            status) check_health ;;
            logs) show_logs "$2" ;;
            monitor) monitor_resources ;;
            backup) backup_databases ;;
            cleanup) cleanup ;;
            *)
                echo "Uso: $0 [setup|build|start|stop|restart|status|logs|monitor|backup|cleanup]"
                echo "Oppure esegui senza parametri per il menu interattivo"
                exit 1
                ;;
        esac
    fi
}

# Esegui main se script chiamato direttamente
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi