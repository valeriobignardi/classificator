#!/bin/bash

# Script per commit e push automatico del progetto classificatore
# Autore: Sistema AI Assistant
# Data creazione: 27 agosto 2025
# 
# Descrizione: Script che replica esattamente il processo di commit e push
# che abbiamo eseguito con successo, includendo controlli di sicurezza

# Colori per output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funzione per stampare messaggi colorati
print_msg() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️  [WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}❌ [ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️  [INFO]${NC} $1"
}

# Funzione di controllo prerequisiti
check_prerequisites() {
    print_msg "🔍 Controllo prerequisiti..."
    
    # Controlla se siamo nella directory corretta
    if [[ ! -f "config.yaml" || ! -d "human-review-ui" ]]; then
        print_error "Errore: Non siamo nella directory root del progetto classificatore"
        print_info "Esegui: cd /home/ubuntu/classificatore"
        exit 1
    fi
    
    # Controlla se git è disponibile
    if ! command -v git &> /dev/null; then
        print_error "Git non è installato"
        exit 1
    fi
    
    # Controlla se siamo in un repository git
    if [[ ! -d ".git" ]]; then
        print_error "Non siamo in un repository Git"
        exit 1
    fi
    
    print_msg "✅ Prerequisiti verificati"
}

# Funzione per mostrare lo status del repository
show_git_status() {
    print_msg "📊 Status attuale del repository:"
    echo "=================================="
    git status --short | head -20
    if [[ $(git status --short | wc -l) -gt 20 ]]; then
        print_info "... e altri $(( $(git status --short | wc -l) - 20 )) file"
    fi
    echo "=================================="
}

# Funzione per il commit
perform_commit() {
    local commit_message="$1"
    
    print_msg "📝 Inizio processo di commit..."
    
    # Mostra status prima del commit
    show_git_status
    
    # Chiedi conferma all'utente se non in modalità automatica
    if [[ "$AUTO_MODE" != "true" ]]; then
        echo
        print_warning "Vuoi procedere con il commit di tutti questi file? [y/N]"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            print_info "Operazione annullata dall'utente"
            exit 0
        fi
    fi
    
    # Aggiungi tutti i file
    print_msg "📦 Aggiunta di tutti i file al staging area..."
    if ! git add .; then
        print_error "Errore durante git add"
        exit 1
    fi
    
    # Esegui il commit
    print_msg "💾 Esecuzione commit..."
    if ! git commit -m "$commit_message"; then
        print_error "Errore durante git commit"
        exit 1
    fi
    
    # Mostra informazioni sul commit appena creato
    local commit_hash=$(git rev-parse --short HEAD)
    print_msg "✅ Commit eseguito con successo! Hash: $commit_hash"
    
    # Mostra statistiche del commit
    print_info "📈 Statistiche commit:"
    git show --stat --pretty=format:"" HEAD | head -10
}

# Funzione per il push
perform_push() {
    print_msg "🚀 Inizio processo di push..."
    
    # Controlla la connessione al remote
    print_info "🔍 Controllo connessione al repository remoto..."
    if ! git remote -v | grep -q origin; then
        print_error "Remote 'origin' non configurato"
        exit 1
    fi
    
    # Esegui il push
    print_msg "📤 Push verso repository remoto..."
    if ! git push; then
        print_error "Errore durante git push"
        print_warning "Possibili cause:"
        print_warning "- Problemi di connessione di rete"
        print_warning "- Credenziali non valide"
        print_warning "- Branch locale indietro rispetto al remote"
        exit 1
    fi
    
    print_msg "✅ Push completato con successo!"
}

# Funzione principale
main() {
    print_msg "🚀 Inizio Git Commit & Push Script"
    print_msg "=================================="
    
    # Controlla se è stata passata un messaggio di commit
    local commit_message="$1"
    if [[ -z "$commit_message" ]]; then
        commit_message="Aggiornamenti automatici sistema classificatore

- Fix API clustering e componenti frontend
- Miglioramenti UX e correzioni bug
- Ottimizzazioni performance e sicurezza
- Aggiornamento documentazione e configurazioni

Commit automatico eseguito il $(date '+%Y-%m-%d alle %H:%M:%S')"
    fi
    
    # Controlla modalità automatica
    if [[ "$2" == "--auto" ]] || [[ "$AUTO_MODE" == "true" ]]; then
        export AUTO_MODE="true"
        print_info "🤖 Modalità automatica attiva (nessuna conferma richiesta)"
    fi
    
    # Esegui controlli preliminari
    check_prerequisites
    
    # Esegui commit
    perform_commit "$commit_message"
    
    # Chiedi conferma per il push se non in modalità automatica
    if [[ "$AUTO_MODE" != "true" ]]; then
        echo
        print_warning "Vuoi procedere con il push verso il repository remoto? [y/N]"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            print_info "Push annullato. Il commit locale è stato salvato."
            print_info "Puoi eseguire 'git push' manualmente quando sei pronto."
            exit 0
        fi
    fi
    
    # Esegui push
    perform_push
    
    print_msg "🎉 Operazione completata con successo!"
    print_info "📊 Repository locale e remoto sono ora sincronizzati"
}

# Funzione per mostrare l'help
show_help() {
    echo "Git Commit & Push Script per il progetto Classificatore"
    echo ""
    echo "Uso:"
    echo "  $0 [MESSAGGIO_COMMIT] [--auto]"
    echo ""
    echo "Parametri:"
    echo "  MESSAGGIO_COMMIT  Messaggio personalizzato per il commit (opzionale)"
    echo "  --auto           Modalità automatica (nessuna conferma richiesta)"
    echo ""
    echo "Esempi:"
    echo "  $0                                    # Commit con messaggio default"
    echo "  $0 \"Fix bug clustering API\"          # Commit con messaggio personalizzato"
    echo "  $0 \"Update frontend\" --auto          # Commit automatico senza conferme"
    echo ""
    echo "Variabili d'ambiente:"
    echo "  AUTO_MODE=true                       # Abilita modalità automatica"
    echo ""
    echo "Note:"
    echo "- Lo script deve essere eseguito dalla directory root del progetto"
    echo "- Richiede git installato e configurato"
    echo "- In modalità non automatica chiede conferma prima di ogni operazione"
}

# Gestione parametri da riga di comando
case "$1" in
    -h|--help)
        show_help
        exit 0
        ;;
    *)
        main "$1" "$2"
        ;;
esac
