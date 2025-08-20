# 🚀 Universal Deploy System - Standalone Package

Sistema completo di dockerizzazione e deploy automatico per qualsiasi progetto.

## 📦 **Contenuto del Package**

```
universal-deploy-standalone/
├── universal-deploy.sh           # ← Script principale di deploy
├── install-deploy-tools.sh       # ← Installer per nuovi progetti
├── quick-setup.sh               # ← Setup rapido progetti comuni
├── Dockerfile.template          # ← Template Dockerfile universale
├── docker-compose.template.yml  # ← Template Docker Compose
├── nginx.template.conf          # ← Template configurazione Nginx
├── ec2-config.env.template      # ← Template configurazione server
├── UNIVERSAL_DEPLOY_GUIDE.md    # ← Documentazione completa
├── package.json                 # ← Metadata NPM (opzionale)
└── README.md                    # ← Questo file
```

## 🎯 **Funzionalità**

### **Auto-Detection Progetti:**
- ✅ **Node.js** (rileva `package.json`)
- ✅ **Python** (rileva `requirements.txt`, `pyproject.toml`)
- ✅ **Go** (rileva `go.mod`)
- ✅ **Java** (rileva `pom.xml`, `build.gradle`)
- ✅ **Docker** (rileva `docker-compose.yml`)
- ✅ **Generic** (fallback universale)

### **Deploy Automatico:**
- ✅ **Configurazione interattiva**
- ✅ **Test connessione SSH**
- ✅ **Backup deployment precedente**
- ✅ **Sync codice con esclusioni intelligenti**
- ✅ **Setup ambiente remoto** (Docker, Node.js, Python)
- ✅ **Deploy applicazione**
- ✅ **Health check post-deployment**
- ✅ **Setup SSL automatico** (Let's Encrypt)
- ✅ **Gestione servizi** (start/stop/restart/logs)

## 🚀 **Installazione in Qualsiasi Progetto**

### **Metodo 1: Copia Manuale**
```bash
# Vai nel tuo progetto
cd /path/to/your-project

# Copia il sistema universal-deploy
cp -r /path/to/universal-deploy-standalone ./universal-deploy

# Installa nel progetto
./universal-deploy/install-deploy-tools.sh install
```

### **Metodo 2: Download GitHub** (se pubblicato)
```bash
# Clone o download
git clone https://github.com/your-username/universal-deploy.git
cd your-project
cp -r /path/to/universal-deploy ./

# Installa
./universal-deploy/install-deploy-tools.sh install
```

### **Metodo 3: Script Automatico**
```bash
# In futuro, uno script come:
curl -fsSL https://raw.githubusercontent.com/your-repo/universal-deploy/main/install.sh | bash
```

## 🔧 **Utilizzo**

### **1. Installazione nel Progetto**
```bash
./universal-deploy/install-deploy-tools.sh install
```

Questo crea:
- `./deploy` - comando principale
- `deploy-config.env` - configurazione server
- `.deployrc` - configurazione progetto
- Template Docker se necessari

### **2. Configurazione Server**
```bash
./deploy config
```

Ti chiederà:
- IP del server (EC2, VPS, etc.)
- Credenziali SSH
- Directory progetto remota
- Porta applicazione
- Comandi build/start

### **3. Deploy**
```bash
# Test connessione
./deploy test

# Deploy completo
./deploy deploy
```

### **4. Gestione**
```bash
./deploy status     # Status applicazione
./deploy restart    # Riavvia servizi
./deploy health     # Health check
./deploy ssl        # Setup SSL
./deploy sync       # Solo sync codice
```

## 📋 **File Generati**

Dopo l'installazione nel progetto:

```
your-project/
├── deploy                      # ← Comando principale
├── deploy-config.env          # ← Config server (come ec2-config.env)
├── .deployrc                  # ← Config progetto
├── deploy-tools/              # ← Tools di deploy
│   ├── universal-deploy.sh
│   ├── templates/
│   └── ...
├── Dockerfile                 # ← Generato se necessario
├── docker-compose.yml         # ← Generato se necessario
└── .deployignore             # ← Esclusioni sync
```

## 🎯 **Esempi di Utilizzo**

### **Progetto Node.js + React**
```bash
# Auto-rileva Node.js da package.json
./deploy config
# Genera Dockerfile Node.js
# Configura porta 3000
# Setup comando "npm start"
```

### **Progetto Python Django**
```bash
# Auto-rileva Python da requirements.txt
./deploy config  
# Genera Dockerfile Python
# Configura porta 8000
# Setup comando "python manage.py runserver"
```

### **Progetto Docker Esistente**
```bash
# Rileva docker-compose.yml esistente
./deploy config
# Usa configurazione Docker esistente
# Setup deploy con docker-compose
```

## 🌐 **Deploy su Diversi Providers**

### **EC2 AWS**
```bash
SERVER_HOST="3.64.188.81"
SSH_USER="ubuntu"  
SSH_KEY="~/.ssh/your-key.pem"
```

### **DigitalOcean Droplet**
```bash
SERVER_HOST="your-droplet-ip"
SSH_USER="root"
SSH_KEY="~/.ssh/id_rsa"
```

### **VPS Generico**
```bash
SERVER_HOST="your-vps-ip"
SSH_USER="your-user"
SSH_KEY="~/.ssh/your-key"
```

## 🔒 **Sicurezza**

- ✅ **Connessioni SSH sicure** con chiavi private
- ✅ **Backup automatico** prima di ogni deploy
- ✅ **SSL/TLS automatico** con Let's Encrypt
- ✅ **Firewall configuration** suggestions
- ✅ **File permissions** corretti
- ✅ **Secrets management** per variabili sensibili

## 🆘 **Troubleshooting**

### **Errore connessione SSH**
```bash
./deploy test  # Testa connessione
# Verifica: IP, chiave SSH, security group
```

### **Deploy fallito**
```bash
./deploy status  # Verifica stato servizi
./deploy sync    # Solo sync codice senza deploy
```

### **Applicazione non risponde**
```bash
./deploy health  # Health check dettagliato
./deploy restart # Riavvia servizi
```

## 📚 **Documentazione Completa**

Vedi `UNIVERSAL_DEPLOY_GUIDE.md` per:
- Configurazione avanzata
- Customizzazione template
- Integrazione CI/CD
- Best practices
- Esempi progetti specifici

## 🤝 **Contribuire**

Questo sistema è nato dal progetto WebScrapingMultiTenant e può essere migliorato per supportare più casi d'uso.

## 📄 **Licenza**

MIT License - Utilizzabile in progetti commerciali e open source.

---

**Creato da: Valerio Bignardi**  
**Versione: 1.0.0**  
**Data: Luglio 2025**

🚀 **Deploy Any Project, Anywhere, Anytime!**
