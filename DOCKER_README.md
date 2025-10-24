# 🐳 Sistema Humanitas - Guida Docker

**Autore:** GitHub Copilot  
**Data:** 2025-09-20  
**Versione:** 1.0.0

Sistema di classificazione automatica delle conversazioni completamente dockerizzato con supporto per frontend React, backend Flask, database MongoDB/MySQL e servizi LLM.

## 📋 Indice

- [🏗️ Architettura](#architettura)
- [⚡ Quick Start](#quick-start)
- [🔧 Configurazione](#configurazione)
- [🚀 Deployment](#deployment)
- [📊 Monitoraggio](#monitoraggio)
- [🛠️ Sviluppo](#sviluppo)
- [🔍 Troubleshooting](#troubleshooting)

## 🏗️ Architettura

### Servizi Inclusi

| Servizio | Porta | Descrizione | Stato |
|----------|-------|-------------|-------|
| **Frontend** | 3000 | React SPA con Nginx | ✅ Essenziale |
| **Backend** | 5000 | Flask API + ML/AI | ✅ Essenziale |
| **MongoDB** | 27017 | Database classificazioni | ✅ Essenziale |
| **MySQL** | 3306 | Database tag/config | ✅ Essenziale |
| **Redis** | 6379 | Cache e sessioni | ✅ Essenziale |
| **Ollama** | 11434 | Servizio LLM locale | ✅ Essenziale |
| **Mongo Express** | 8081 | UI MongoDB | 🔧 Monitoring |
| **phpMyAdmin** | 8080 | UI MySQL | 🔧 Monitoring |

### Volumi Persistenti

- `mongodb-data`: Dati MongoDB
- `mysql-data`: Dati MySQL
- `ollama-data`: Modelli LLM
- `redis-data`: Cache Redis
- `semantic-cache`: Cache semantica ML
- `training-logs`: Log training modelli

## ⚡ Quick Start

### 1. Prerequisiti

```bash
# Verifica Docker
docker --version
docker-compose --version

# Clona il repository (se necessario)
git clone <repository-url>
cd classificatore
```

### 2. Setup Automatico

```bash
# Metodo 1: Script interattivo
./docker-manager.sh

# Metodo 2: Setup diretto
./docker-manager.sh setup
```

### 3. Verifica Funzionamento

```bash
# Accedi alle interfacce
# Frontend: http://localhost:3000
# Backend API: http://localhost:5000/health
# MongoDB Express: http://localhost:8081 (se monitoring attivo)
# phpMyAdmin: http://localhost:8080 (se monitoring attivo)
```

## 🔧 Configurazione

### File di Configurazione

#### 1. `.env` - Variabili Ambiente

```bash
# Copia il template
cp .env.example .env

# Modifica le configurazioni
vim .env
```

**Configurazioni Essenziali:**

```env
# API Keys
OPENAI_API_KEY=sk-your-openai-key-here

# Database Passwords (CAMBIA IN PRODUZIONE!)
MONGODB_PASSWORD=your-secure-password
MYSQL_ROOT_PASSWORD=your-secure-password

# GPU (0 per prima GPU, vuoto per disabilitare)
CUDA_VISIBLE_DEVICES=0

# Performance
PRELOAD_MODELS=false
```

#### 2. `config.yaml` - Configurazione Applicazione

Il file esistente viene utilizzato direttamente dal container. Principale configurazioni che potresti voler modificare:

```yaml
# Database connections (aggiornate automaticamente per Docker)
mongodb:
  url: mongodb://admin:password@mongodb:27017/classificazioni

# LLM Services
llm:
  ollama:
    url: http://ollama:11434
```

### Profili Docker Compose

#### Profilo Default (Produzione)
```bash
docker-compose up -d
```

#### Profilo Monitoring (Sviluppo)
```bash
docker-compose --profile monitoring up -d
```

## 🚀 Deployment

### Deployment Locale

```bash
# Setup completo
./docker-manager.sh setup

# Solo build
./docker-manager.sh build

# Solo avvio
./docker-manager.sh start
```

### Deployment Produzione

```bash
# 1. Configura ambiente produzione
cp .env.example .env.production
# Modifica password e configurazioni sicure

# 2. Build ottimizzato
docker-compose -f docker-compose.yml build --no-cache

# 3. Deploy
docker-compose up -d

# 4. Verifica
curl http://localhost:5000/health
curl http://localhost:3000/health
```

### GPU Support

Il sistema supporta automaticamente GPU NVIDIA se disponibili:

```bash
# Verifica GPU
nvidia-smi

# Abilita GPU nei container
CUDA_VISIBLE_DEVICES=0 docker-compose up -d
```

## 📊 Monitoraggio

### Stato Servizi

```bash
# Verifica health
./docker-manager.sh status

# Monitoraggio risorse
./docker-manager.sh monitor

# Logs in tempo reale
./docker-manager.sh logs
./docker-manager.sh logs backend
```

### Interfacce Web

| Interfaccia | URL | Credenziali |
|-------------|-----|-------------|
| **App Principale** | http://localhost:3000 | - |
| **API Backend** | http://localhost:5000 | - |
| **MongoDB Express** | http://localhost:8081 | admin/humanitas2025 |
| **phpMyAdmin** | http://localhost:8080 | taggenerator/zsRxiYmcVG9XX7Q3TvAT |

### Health Checks

Tutti i servizi hanno health checks automatici:

```bash
# Verifica health manualmente
curl http://localhost:5000/health    # Backend
curl http://localhost:3000/health    # Frontend
curl http://localhost:11434/api/tags # Ollama
```

## 🛠️ Sviluppo

### Sviluppo Frontend

```bash
# Sviluppo locale (hot reload)
cd human-review-ui
npm start  # Porta 3001 (evita conflitto con container)

# Build e test nel container
docker-compose build frontend
docker-compose up frontend
```

### Sviluppo Backend

```bash
# Debug mode
FLASK_DEBUG=1 docker-compose up backend

# Accesso shell container
docker exec -it humanitas-backend bash

# Logs in tempo reale
docker-compose logs -f backend
```

### Aggiornamento Codice

```bash
# Rebuild dopo modifiche
docker-compose build backend frontend
docker-compose up -d

# Solo riavvio (se nessuna modifica dipendenze)
docker-compose restart backend frontend
```

## 🔍 Troubleshooting

### Problemi Comuni

#### 1. Servizi Non Avviano

```bash
# Verifica logs
docker-compose logs

# Verifica prerequisiti
./docker-manager.sh

# Cleanup e riavvio
docker-compose down -v
docker-compose up -d
```

#### 2. Problemi Database

```bash
# Reset database
docker-compose down -v
docker volume rm humanitas-mongodb-data humanitas-mysql-data
docker-compose up -d

# Backup prima del reset
./docker-manager.sh backup
```

#### 3. Problemi GPU

```bash
# Verifica disponibilità
nvidia-smi

# Disabilita GPU
export CUDA_VISIBLE_DEVICES=""
docker-compose up -d

# Reinstalla driver NVIDIA
# Vedi: https://docs.nvidia.com/datacenter/cloud-native/
```

#### 4. Out of Memory

```bash
# Monitoraggio memoria
docker stats

# Riduzione memoria Redis
# Modifica docker-compose.yml: --maxmemory 128mb

# Disabilita preload modelli
PRELOAD_MODELS=false docker-compose up -d
```

### Debug Avanzato

#### Accesso Shell Container

```bash
# Backend
docker exec -it humanitas-backend bash

# Frontend
docker exec -it humanitas-frontend sh

# Database
docker exec -it humanitas-mongodb mongosh
docker exec -it humanitas-mysql mysql -u root -p
```

#### Logs Dettagliati

```bash
# Tutti i servizi
docker-compose logs -f

# Servizio specifico
docker-compose logs -f backend

# Con timestamp
docker-compose logs -f -t backend
```

#### Network Debugging

```bash
# Test connettività inter-container
docker exec humanitas-backend ping mongodb
docker exec humanitas-backend curl http://ollama:11434/api/tags

# Inspect network
docker network inspect humanitas-classification-network
```

## 📝 Comandi Utili

### Gestione Rapida

```bash
# Setup completo
./docker-manager.sh setup

# Avvio
./docker-manager.sh start

# Arresto
./docker-manager.sh stop

# Riavvio
./docker-manager.sh restart

# Pulizia completa
./docker-manager.sh cleanup
```

### Docker Compose

```bash
# Avvio servizi base
docker-compose up -d

# Avvio con monitoring
docker-compose --profile monitoring up -d

# Build senza cache
docker-compose build --no-cache

# Logs specifici
docker-compose logs -f backend frontend

# Restart servizio singolo
docker-compose restart backend

# Scaling (se supportato)
docker-compose up -d --scale backend=2
```

### Backup e Restore

```bash
# Backup automatico
./docker-manager.sh backup

# Backup manuale MongoDB
docker exec humanitas-mongodb mongodump --out /tmp/backup
docker cp humanitas-mongodb:/tmp/backup ./backup-mongo

# Backup manuale MySQL
docker exec humanitas-mysql mysqldump -u root -p --all-databases > backup-mysql.sql
```

## 🛡️ Sicurezza

### Checklist Produzione

- [ ] Cambiate tutte le password di default
- [ ] Configurate HTTPS per il frontend
- [ ] Limitate l'accesso di rete ai servizi
- [ ] Configurate firewall appropriato
- [ ] Monitorate i log per attività sospette
- [ ] Backup regolari dei database
- [ ] Aggiornamenti di sicurezza regolari

### Hardening

```bash
# Rimuovi servizi monitoring in produzione
docker-compose up -d --remove-orphans

# Usa Docker secrets per password
# Vedi: https://docs.docker.com/engine/swarm/secrets/

# Abilita Docker content trust
export DOCKER_CONTENT_TRUST=1
```

---

## 📞 Supporto

Per problemi o domande:

1. Controlla i logs: `./docker-manager.sh logs`
2. Verifica la documentazione troubleshooting
3. Apri un issue nel repository
4. Contatta il team di sviluppo

**Made with ❤️ by GitHub Copilot**