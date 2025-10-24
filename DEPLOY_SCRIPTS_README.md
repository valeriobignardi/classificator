# 🚀 Scripts di Deploy e Gestione

Questo documento descrive gli script disponibili per gestire il sistema di classificazione dockerizzato.

## 📁 Script Disponibili

### 🎯 `quick-deploy.sh`
**Deploy rapido e completo**
```bash
./quick-deploy.sh
```
- Ferma container esistenti
- Ricostruisce le immagini
- Avvia tutti i servizi
- Verifica che tutto funzioni

### 🔧 `docker-manager.sh` 
**Gestione completa del sistema**

**Modalità interattiva:**
```bash
./docker-manager.sh
```

**Modalità comando:**
```bash
./docker-manager.sh [comando]
```

Comandi disponibili:
- `setup` - Setup iniziale completo
- `build` - Build delle immagini
- `start` - Avvia servizi
- `stop` - Ferma servizi
- `restart` - Riavvia servizi
- `status` - Verifica stato servizi
- `logs [servizio]` - Visualizza logs
- `monitor` - Monitoraggio risorse
- `backup` - Backup (info servizi locali)
- `cleanup` - Pulizia completa

### 🧪 `test-docker.sh`
**Test delle immagini Docker**
```bash
./test-docker.sh
```
- Testa che le immagini siano costruite correttamente
- Verifica l'avvio dei container
- Test di base delle funzionalità

## 🏗️ Architettura Sistema

```
┌─────────────────────────────────────────────┐
│           DOCKER CONTAINERS                │
├─────────────────────────────────────────────┤
│ classificatore-frontend:3005                │
│ classificatore-backend (network_mode: host) │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│              LOCAL SERVICES                 │
├─────────────────────────────────────────────┤
│ MongoDB:27017  │ MySQL:3306  │ Ollama:11434 │
└─────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prima volta:
```bash
# 1. Setup completo
./docker-manager.sh setup

# O in alternativa
./quick-deploy.sh
```

### Deploy successivi:
```bash
# Deploy rapido con rebuild
./quick-deploy.sh

# O gestione manuale
./docker-manager.sh build
./docker-manager.sh start
```

### Monitoraggio:
```bash
# Stato servizi
./docker-manager.sh status

# Logs in tempo reale
./docker-manager.sh logs

# Logs di un servizio specifico
./docker-manager.sh logs backend
./docker-manager.sh logs frontend
```

## 🔗 URL Servizi

- **Frontend**: http://localhost:3005
- **Backend API**: http://localhost:5000
- **Health Check**: http://localhost:5000/health

## 📋 Servizi Esterni

I seguenti servizi devono essere disponibili localmente:

- **MongoDB**: localhost:27017
- **MySQL**: localhost:3306  
- **Ollama**: localhost:11434

## ⚠️ Note Importanti

1. **Network Mode**: Il backend usa `network_mode: host` per accedere ai servizi locali
2. **GPU**: Configurato per usare GPU NVIDIA se disponibile
3. **Volumi**: I dati critici sono persistenti nei volumi Docker
4. **Ports**: Frontend su 3005, backend accessibile direttamente su host

## 🛠️ Troubleshooting

### Container non si avvia:
```bash
# Verifica logs
./docker-manager.sh logs

# Rebuild completo
./docker-manager.sh stop
./docker-manager.sh build
./docker-manager.sh start
```

### Reset completo:
```bash
# ATTENZIONE: Rimuove tutto
./docker-manager.sh cleanup
./docker-manager.sh setup
```

### Problemi di rete:
- Verifica che i servizi locali (MongoDB, MySQL, Ollama) siano in esecuzione
- Il backend usa network_mode: host quindi deve poter accedere a localhost

## 📝 Configurazione

File di configurazione principali:
- `docker-compose.yml` - Orchestrazione container
- `.env` - Variabili d'ambiente
- `config.yaml` - Configurazione applicazione