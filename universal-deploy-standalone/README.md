# 🚀 Universal Deploy Tools

Sistema di deployment universale e riutilizzabile per qualsiasi progetto su EC2 o altri server.

## 🎯 Caratteristiche

- ✅ **Auto-detection** del tipo di progetto (Node.js, Python, Docker, etc.)
- ✅ **Configurazione interattiva** guidata
- ✅ **Backup automatico** prima del deploy
- ✅ **Health check** post-deployment
- ✅ **SSL automatico** con Let's Encrypt
- ✅ **Multi-server** support
- ✅ **Template** personalizzabili

## 🚀 Installazione Rapida

### Opzione 1: Script Standalone
```bash
# In qualsiasi progetto
curl -fsSL https://raw.githubusercontent.com/your-repo/universal-deploy/main/install.sh | bash
```

### Opzione 2: Installazione Locale
```bash
# Copia i file di deploy nel tuo progetto
./install-deploy-tools.sh install
```

### Opzione 3: NPM Package (per progetti Node.js)
```bash
npm install -g @yourname/universal-deploy
# oppure
npx @yourname/universal-deploy init
```

## 🛠️ Utilizzo

### 1. Prima configurazione
```bash
./deploy config
```

### 2. Deploy completo
```bash
./deploy deploy
```

### 3. Comandi rapidi
```bash
./deploy test      # Test connessione SSH
./deploy sync      # Solo sync codice
./deploy health    # Health check
./deploy ssl       # Setup SSL
./deploy status    # Status applicazione
./deploy restart   # Restart servizi
```

## 📁 Struttura File Generati

```
your-project/
├── deploy                    # Script wrapper principale
├── deploy-config.env        # Configurazione deployment
├── .deployrc               # Configurazione progetto
└── deploy-tools/           # Tools di deployment
    ├── universal-deploy.sh
    ├── quick-setup.sh
    ├── templates/
    │   ├── docker-compose.template.yml
    │   ├── Dockerfile.template
    │   └── nginx.template.conf
    └── ...
```

## 🔧 Configurazione

Il file `deploy-config.env` contiene tutte le configurazioni:

```bash
# Server
SERVER_HOST="your-server-ip"
SSH_USER="ubuntu"
SSH_KEY="~/.ssh/your-key.pem"
REMOTE_PROJECT_DIR="/home/ubuntu/your-project"

# Progetto
PROJECT_TYPE="nodejs"
MAIN_PORT="3000"
HEALTH_ENDPOINT="/health"
BUILD_COMMAND="npm install && npm run build"
START_COMMAND="npm start"
```

## 🐳 Supporto Tipi di Progetto

### Node.js
- Auto-detect tramite `package.json`
- Build: `npm install && npm run build`
- Start: `npm start`
- Port: 3000

### Python
- Auto-detect tramite `requirements.txt`
- Build: `pip install -r requirements.txt`
- Start: `python main.py`
- Port: 8000

### Docker
- Auto-detect tramite `docker-compose.yml`
- Build: `docker-compose build`
- Start: `docker-compose up -d`
- Port: 80

### Generic
- Configurazione manuale
- Comandi personalizzabili

## 🌐 Multi-Server Deployment

```bash
# Configura più server
./deploy config --server production
./deploy config --server staging

# Deploy su server specifico
./deploy deploy --server production
./deploy deploy --server staging
```

## 🔒 SSL Automatico

```bash
# Durante la configurazione
Domain: yourdomain.com

# Oppure successivamente
./deploy ssl
```

## 📊 Monitoring e Logs

```bash
# Status completo
./deploy status

# Logs in tempo reale
ssh -i ~/.ssh/your-key.pem user@server 'cd /path/to/project && tail -f *.log'

# Health check
./deploy health
```

## 🚀 Esempi di Utilizzo

### Progetto Node.js + React
```bash
# Installazione
curl -fsSL https://install-url | bash

# Configurazione
./deploy config
# Server IP: 1.2.3.4
# SSH Key: ~/.ssh/mykey.pem
# Port: 3000
# Build: npm install && npm run build
# Start: npm start

# Deploy
./deploy deploy
```

### Progetto Python Flask
```bash
./deploy config
# Auto-detected: Python
# Port: 5000
# Build: pip install -r requirements.txt
# Start: python app.py

./deploy deploy
```

### Progetto Docker Multi-Container
```bash
./deploy config
# Auto-detected: Docker
# Port: 80
# Build: docker-compose build
# Start: docker-compose up -d

./deploy deploy
```

## 🔄 Workflow Tipico

1. **Setup iniziale**:
   ```bash
   ./install-deploy-tools.sh install
   ./deploy config
   ```

2. **Primo deploy**:
   ```bash
   ./deploy deploy
   ```

3. **Deploy successivi**:
   ```bash
   ./deploy sync    # Solo codice
   ./deploy restart # Restart servizi
   ```

4. **Maintenance**:
   ```bash
   ./deploy health  # Check status
   ./deploy status  # Detailed info
   ```

## 🛡️ Security Best Practices

- ✅ SSH key-based authentication
- ✅ Non-root user deployment
- ✅ Firewall configuration
- ✅ SSL/TLS encryption
- ✅ Backup before deployment
- ✅ Health checks

## 🤝 Contribuire

1. Fork il repository
2. Crea feature branch
3. Commit cambiamenti
4. Push e crea PR

## 📝 License

MIT License - Libero di usare in progetti commerciali e open source.

---

## 🆘 Troubleshooting

### Errore SSH
```bash
# Verifica chiave
chmod 400 ~/.ssh/your-key.pem

# Test manuale
ssh -i ~/.ssh/your-key.pem user@server
```

### Port già in uso
```bash
# Cambia porta in deploy-config.env
MAIN_PORT="3001"

# Oppure termina processo
./deploy restart
```

### Problemi Docker
```bash
# Verifica Docker
ssh user@server 'docker --version'

# Restart Docker
ssh user@server 'sudo systemctl restart docker'
```
