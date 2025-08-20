# 🚀 Sistema di Deploy Universale

Un sistema plug-and-play per deployare qualsiasi progetto su EC2 o altri server, riutilizzabile su tutti i tuoi progetti.

## 🎯 Cosa Abbiamo Creato

### 1. **Universal Deploy Script** (`universal-deploy.sh`)
- Script bash universale che funziona con qualsiasi tipo di progetto
- Auto-detection del tipo di progetto (Node.js, Python, Docker, Generic)
- Configurazione interattiva
- Backup automatico prima del deploy
- Health check post-deployment
- Setup SSL automatico

### 2. **Installer** (`install-deploy-tools.sh`)
- Installa gli strumenti di deploy in qualsiasi progetto
- Crea wrapper script (`./deploy`)
- Genera template e configurazioni
- Setup automatico per quick-start

### 3. **NPM Package** (opzionale)
- Package npm per progetti Node.js
- Installazione globale: `npm install -g @yourname/universal-deploy`
- Comando: `npx @yourname/universal-deploy init`

## 🚀 Come Utilizzarlo

### Installazione in Qualsiasi Progetto

```bash
# Opzione 1: Download diretto
curl -fsSL https://raw.githubusercontent.com/your-repo/universal-deploy/main/install.sh | bash

# Opzione 2: Copia manuale
cp -r /path/to/universal-deploy ./
./universal-deploy/install-deploy-tools.sh install

# Opzione 3: NPM (per progetti Node.js)
npx @yourname/universal-deploy init
```

### Utilizzo

```bash
# Configurazione iniziale
./deploy config

# Deploy completo
./deploy deploy

# Altri comandi
./deploy test      # Test connessione
./deploy sync      # Solo sync codice
./deploy health    # Health check
./deploy status    # Status applicazione
```

## 📁 Struttura Generata

```
your-project/
├── deploy                    # ← Comando principale
├── deploy-config.env        # ← Configurazione server
├── .deployrc               # ← Configurazione progetto
└── deploy-tools/           # ← Tools di deployment
    ├── universal-deploy.sh
    ├── quick-setup.sh
    └── templates/
```

## 🔧 File di Configurazione

### `deploy-config.env` (auto-generato)
```bash
SERVER_HOST="your-ec2-ip"
SSH_USER="ubuntu"
SSH_KEY="~/.ssh/webscraping.pem"
REMOTE_PROJECT_DIR="/home/ubuntu/your-project"
PROJECT_TYPE="docker"  # auto-detected
MAIN_PORT="3000"
BUILD_COMMAND="docker-compose build"
START_COMMAND="docker-compose up -d"
```

### `.deployrc` (configurazione progetto)
```bash
PROJECT_NAME="your-project"
# Override defaults here
# MAIN_PORT=8080
# HEALTH_ENDPOINT="/api/health"
```

## 🎯 Esempi di Utilizzo

### Progetto Node.js + React (come questo)
```bash
# Installazione
./universal-deploy/install-deploy-tools.sh install

# Configurazione
./deploy config
# Auto-detected: Docker (tramite docker-compose.yml)
# Server: 10.8.0.1 (VPN)
# SSH Key: ~/.ssh/webscraping.pem
# Build: docker-compose build
# Start: docker-compose up -d

# Deploy
./deploy deploy
```

### Progetto Python Flask
```bash
./deploy config
# Auto-detected: Python
# Port: 8000
# Build: pip install -r requirements.txt
# Start: python app.py

./deploy deploy
```

### Progetto Generic
```bash
./deploy config
# Type: Generic
# Custom build/start commands
# Manual port configuration

./deploy deploy
```

## 🌟 Caratteristiche Avanzate

### Multi-Server Support
```bash
# Configurazioni multiple
./deploy config --server production
./deploy config --server staging

# Deploy specifico
./deploy deploy --server production
```

### Template Personalizzabili
- `deploy-tools/templates/docker-compose.template.yml`
- `deploy-tools/templates/Dockerfile.template`
- `deploy-tools/templates/nginx.template.conf`

### SSL Automatico
```bash
./deploy ssl  # Setup Let's Encrypt + Nginx
```

### Backup e Rollback
```bash
# Backup automatico prima del deploy
BACKUP_BEFORE_DEPLOY=true

# Rollback manuale
ssh user@server 'cd project_backup_20241217_143022 && docker-compose up -d'
```

## 🔄 Workflow Completo

1. **Setup iniziale** (una volta per progetto):
   ```bash
   ./universal-deploy/install-deploy-tools.sh install
   ./deploy config
   ```

2. **Primo deploy**:
   ```bash
   ./deploy deploy
   ```

3. **Deploy successivi**:
   ```bash
   ./deploy sync     # Solo codice
   ./deploy restart  # Restart servizi
   ```

4. **Maintenance**:
   ```bash
   ./deploy health   # Check status
   ./deploy status   # Info dettagliate
   ```

## 📦 Packaging per Altri Progetti

### Metodo 1: Copy-Paste Directory
```bash
# Nel nuovo progetto
cp -r /path/to/webScrapingMultiTenant/universal-deploy ./
./universal-deploy/install-deploy-tools.sh install
```

### Metodo 2: Git Submodule
```bash
git submodule add https://github.com/your-username/universal-deploy.git deploy-tools
./deploy-tools/install-deploy-tools.sh install
```

### Metodo 3: Download Script
```bash
# Crea install.sh che scarica tutto
curl -fsSL https://your-domain.com/universal-deploy/install.sh | bash
```

## 🎁 Vantaggi del Sistema

✅ **Plug & Play**: Funziona su qualsiasi progetto
✅ **Auto-Detection**: Riconosce automaticamente il tipo di progetto
✅ **Riutilizzabile**: Un sistema per tutti i progetti
✅ **Configurabile**: Template e configurazioni personalizzabili
✅ **Sicuro**: Backup automatici, SSL, best practices
✅ **Multi-Platform**: EC2, VPS, server dedicati
✅ **Zero Dependencies**: Solo bash e standard Unix tools

## 🚀 Deploy del Progetto Attuale

Per deployare questo progetto specifico:

```bash
# È già installato, vai diretto alla configurazione
./deploy config

# Usa le configurazioni VPN esistenti:
# Server: 10.8.0.1
# SSH Key: ~/.ssh/webscraping.pem  
# Type: Docker
# Porta: 3000 (React), 3001 (API), 8080 (WS)

# Deploy
./deploy deploy
```

Il sistema utilizzerà automaticamente le configurazioni esistenti in `.env.vpn` e `docker-compose.yml`.

---

🔥 **Sistema completamente plug-and-play pronto per essere riutilizzato su qualsiasi progetto!**
