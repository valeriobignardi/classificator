#!/bin/bash
# =============================================================================
# File: git_commit_push.sh
# Autore: GitHub Copilot
# Data creazione: 2025-08-27
# Descrizione: Script automatico per commit e push Git del progetto classificatore
# 
# Questo script replica esattamente il processo che ha risolto il problema
# del commit tramite interfaccia VS Code, eseguendo:
# 1. Verifica dello stato del repository
# 2. Staging di tutti i file modificati
# 3. Commit con messaggio descrittivo
# 4. Push verso il repository remoto
# =============================================================================

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funzione per stampare con colori
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Funzione per generare messaggio di commit automatico basato sui file modificati
generate_commit_message() {
    local message="Aggiornamento automatico progetto classificatore"
    local details=""
    
    # Controlla se ci sono nuovi componenti React
    if git diff --cached --name-only | grep -q "human-review-ui/src/components/"; then
        details="${details}\n- Aggiornamento componenti React"
    fi
    
    # Controlla se ci sono modifiche al sistema embedding
    if git diff --cached --name-only | grep -q "EmbeddingEngine/"; then
        details="${details}\n- Modifiche sistema embedding"
    fi
    
    # Controlla se ci sono modifiche al clustering
    if git diff --cached --name-only | grep -q "Clustering/"; then
        details="${details}\n- Aggiornamento sistema clustering"
    fi
    
    # Controlla se ci sono nuovi test
    if git diff --cached --name-only | grep -q "test_"; then
        details="${details}\n- Aggiunta/modifica test"
    fi
    
    # Controlla se ci sono modifiche al server
    if git diff --cached --name-only | grep -q "server.py"; then
        details="${details}\n- Aggiornamento server API"
    fi
    
    # Controlla se ci sono nuovi file di configurazione
    if git diff --cached --name-only | grep -q "config\|yaml"; then
        details="${details}\n- Aggiornamento configurazioni"
    fi
    
    # Controlla se ci sono modifiche al database
    if git diff --cached --name-only | grep -q "Database/"; then
        details="${details}\n- Modifiche sistema database"
    fi
    
    if [ -n "$details" ]; then
        message="${message}${details}"
    fi
    
    # Aggiunge timestamp
    message="${message}\n\n[Auto-commit: $(date '+%Y-%m-%d %H:%M:%S')]"
    
    echo -e "$message"
}

# Verifica che siamo nella directory corretta
if [ ! -d ".git" ]; then
    print_error "Errore: Non sono nella root di un repository Git!"
    print_error "Assicurati di essere in /home/ubuntu/classificatore"
    exit 1
fi

print_status "🚀 Avvio script automatico Git commit + push"
print_status "Directory corrente: $(pwd)"

# Verifica connessione repository remoto
print_status "🔍 Verifica connessione repository remoto..."
if ! git remote -v | grep -q "github.com"; then
    print_error "Repository remoto non configurato correttamente!"
    exit 1
fi
print_success "Repository remoto OK"

# Mostra stato attuale
print_status "📊 Stato attuale del repository:"
git status --short | head -20
if [ $(git status --short | wc -l) -gt 20 ]; then
    print_warning "... e altri $(( $(git status --short | wc -l) - 20 )) file"
fi

# Chiedi conferma se ci sono molti file
file_count=$(git status --short | wc -l)
if [ $file_count -gt 100 ]; then
    print_warning "⚠️  Attenzione: $file_count file da committare (molti!)"
    echo -n "Vuoi continuare? (y/N): "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_status "Operazione annullata dall'utente"
        exit 0
    fi
fi

# Aggiungi tutti i file al staging (come fatto precedentemente)
print_status "📦 Staging di tutti i file modificati..."
if git add .; then
    print_success "Staging completato"
else
    print_error "Errore durante lo staging"
    exit 1
fi

# Verifica se ci sono file in staging
if git diff --cached --quiet; then
    print_warning "Nessun file da committare"
    exit 0
fi

# Conta i file in staging
staged_files=$(git diff --cached --name-only | wc -l)
print_status "📁 File in staging: $staged_files"

# Genera messaggio di commit automatico
commit_message=$(generate_commit_message)
print_status "📝 Messaggio di commit generato:"
echo -e "$commit_message" | sed 's/^/   /'

# Possibilità di personalizzare il messaggio
echo -n "Vuoi usare questo messaggio o inserirne uno personalizzato? (usa/personalizza) [usa]: "
read -r message_choice
if [[ "$message_choice" =~ ^[Pp] ]]; then
    echo -n "Inserisci il messaggio di commit personalizzato: "
    read -r custom_message
    if [ -n "$custom_message" ]; then
        commit_message="$custom_message"
    fi
fi

# Esegui commit
print_status "💾 Esecuzione commit..."
if git commit -m "$commit_message"; then
    commit_hash=$(git log -1 --format="%h")
    print_success "✅ Commit completato con hash: $commit_hash"
else
    print_error "❌ Errore durante il commit"
    exit 1
fi

# Esegui push
print_status "🌐 Push verso repository remoto..."
if git push; then
    print_success "✅ Push completato con successo!"
    print_success "🎉 Tutti i file sono stati sincronizzati su GitHub!"
else
    print_error "❌ Errore durante il push"
    print_error "Potrebbe essere necessario un pull prima del push"
    echo -n "Vuoi provare un pull + push? (y/N): "
    read -r pull_response
    if [[ "$pull_response" =~ ^[Yy]$ ]]; then
        print_status "📥 Esecuzione pull..."
        if git pull --rebase; then
            print_status "🌐 Nuovo tentativo di push..."
            if git push; then
                print_success "✅ Push completato dopo pull!"
            else
                print_error "❌ Push fallito anche dopo pull"
                exit 1
            fi
        else
            print_error "❌ Errore durante pull"
            exit 1
        fi
    else
        exit 1
    fi
fi

# Riepilogo finale
print_success "📊 RIEPILOGO OPERAZIONI:"
print_success "   • File committati: $staged_files"
print_success "   • Commit hash: $(git log -1 --format='%h')"
print_success "   • Branch: $(git branch --show-current)"
print_success "   • Repository: $(git remote get-url origin)"
print_success "   • Data: $(date '+%Y-%m-%d %H:%M:%S')"

# Mostra ultime 3 righe del log per conferma
print_status "📋 Ultimi commit:"
git log --oneline -3 | sed 's/^/   /'

print_success "🎊 Script completato con successo!"