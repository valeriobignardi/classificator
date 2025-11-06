# Sistema di Classificazione Conversazioni Multi‑Tenant

Piattaforma completa per classificare conversazioni testuali in contesti multi‑tenant, con Review Umana, riaddestramento ML, configurazione LLM per tenant, clustering e statistiche.

Indice rapido
- Panoramica
- Funzionalità Chiave
- Guida Operativa (UI)
- API Principali (Quick‑ref)
- Persistenza e Path
- Architettura (sintesi)
- Setup e Avvio
- Note Operative

## Panoramica

Obiettivi principali:
- Classificare conversazioni assegnando etichette da un set tenant‑specific.
- Selezionare automaticamente i casi “difficili” per Review Umana (disagreement, bassa confidenza, novità).
- Riaddestrare periodicamente il modello ML usando le decisioni umane + classificazioni LLM filtrate.
- Analizzare i dati con clustering/visualizzazioni e, se necessario, fine‑tuning del modello LLM.

## Funzionalità Chiave

- Review Umana e Training Supervisionato
  - Selezione casi attraverso Quality Gate (ensemble disagreement, soglie di confidenza, novelty)
  - Risoluzione caso con propagazione ai membri del cluster (se rappresentante)
  - Tracciamento decisioni in file JSONL per riaddestramento ML
- Riaddestramento Modello ML
  - Automatico (threshold decisioni) o manuale via API/UI
  - Dataset combinato: review umane + classificazioni LLM (filtrate, senza propagati, con testo completo)
- Classifica Tutto / Batch
  - Pipeline end‑to‑end con opzioni force (reprocess totale, force_review, clean training ML)
- Configurazione Parametri
  - Soglie Review Queue (DB `TAG.soglie`), parametri clustering (UMAP/HDBSCAN), batch
- Configurazione Avanzata LLM per Tenant
  - Scelta modello (Ollama/OpenAI), parametri tokenization/generation
  - Validazione e clamp automatico ai limiti del modello (input/output tokens)
  - Reset parametri (DB‑first, nessuna scrittura su file)
- Statistiche e Visualizzazioni
  - Panoramica review/classificazioni, distribuzione etichette, visualizzazioni 2D/3D clustering
- Fine‑Tuning
  - Gestore Mistral tenant‑aware per generazione dataset + addestramento modelli personalizzati

## Guida Operativa (UI)

### 1) Selezione Tenant
- Barra laterale → elenco tenant da `/api/tenants` (UUID/Slug/Name).
- Tutte le sezioni operano sul tenant attivo.

### 2) Dashboard di Review
- Elenco casi pending (rappresentanti, outlier; propagati opzionali).
- Dettaglio caso con decisione umana (label, confidenza, note) e tracciamento propagazione.

### 3) Training Supervisionato
- Avvio da UI o `POST /api/train/supervised/<tenant>`.
- Parametri tipici:
  - `force_review`: forza la review dei casi; attiva auto‑clear Mongo per evitare somma sessioni.
  - `force_retrain_ml`: pulizia training ML.
- La risoluzione dei casi scrive nel JSONL di training del tenant.

### 4) File Training (nuova sezione UI)
- Lista per tenant dei file riconosciuti in:
  - `TRAINING_DATA_DIR` (se impostata) → `data/training` → fallback `/tmp/classificatore/training`.
- Convenzione nomi: `training_decisions_{tenant_id}.jsonl`.
- Visualizzazione contenuto con caricamento progressivo.

### 5) Riaddestramento ML
- Automatico ogni N decisioni (config qualità) oppure manuale.
- Primo training: review umane + LLM finali.
- Riaddestramenti: review umane (opzionale + LLM, secondo config).
- Validazioni: numero classi >= 2, dataset minimo.

### 6) Configurazione Parametri
- Review Queue Thresholds (DB `TAG.soglie`): soglie rappresentanti/outlier/propagazione.
- Parametri Clustering: UMAP/HDBSCAN con strumenti di test e visualizzazioni.
- Batch Processing (persistenza `engines.llm_config`): `classification_batch_size`, `max_parallel_calls`.

### 7) Configurazione Avanzata LLM
- Modelli disponibili per tenant: `/api/llm/models/<tenant_id>` (Ollama + OpenAI).
- Parametri per tenant: `/api/llm/parameters/<tenant_id>` (get/put).
- Validazione con clamp automatico ai limiti del modello (token input/output).
- Reset default: `POST /api/llm/reset-parameters/<tenant_id>` → cancella `llm_config` in DB (no file write).
- Note: GPT‑5 non supporta alcuni parametri generation (ignorati con warning).

### 8) Statistiche
- Review/Generali: `/api/review/<tenant_id>/stats`.
- Clustering: `/api/statistics/<tenant>/clustering?include_visualizations=true&sample_limit=5000`.
- UI: distribuzione etichette, counts, visualizzazioni 2D/3D (PCA, t‑SNE).

### 9) Fine‑Tuning
- Gestore Mistral tenant‑aware: dataset da `data/training/training_decisions_{tenant_id}.jsonl` + recupero conversazioni.
- Supporto naming/registry per modelli e switch in pipeline.

## API Principali (Quick‑ref)

```bash
# Training supervisionato (forza review; auto‑clear Mongo abilitato se force_review=true)
curl -X POST http://localhost:5000/api/train/supervised/<tenant_slug> \
  -H "Content-Type: application/json" \
  -d '{"force_review": true, "force_retrain_ml": false}'

# Classifica tutto con ri‑process da zero
curl -X POST http://localhost:5000/classify/all/<tenant_slug> \
  -H "Content-Type: application/json" \
  -d '{"force_reprocess_all": true}'

# Parametri LLM (get/put) e reset default
# a) GET  /api/llm/parameters/<tenant_id>
# b) PUT  /api/llm/parameters/<tenant_id>   # body: { parameters, model_name? }
# c) POST /api/llm/reset-parameters/<tenant_id>

# Modelli LLM disponibili per tenant
curl http://localhost:5000/api/llm/models/<tenant_id>

# File di training per tenant
curl http://localhost:5000/api/training-files/<tenant_id>
```

## Persistenza e Path

- JSONL training decisions (per tenant):
  - `TRAINING_DATA_DIR` (env consigliata) → `data/training` → fallback `/tmp/classificatore/training`.
  - Nome: `training_decisions_{tenant_id}.jsonl`.
- Modelli ML + Fine‑tuned: directory tenant‑aware (gestore interno).
- MongoDB: un’unica collection per tenant: `{tenant_slug}_{tenant_id}`.

## Architettura (sintesi)

- API Server (`server.py`): Flask REST, orchestrazione multi‑tenant.
- Pipeline (`Pipeline/end_to_end_pipeline.py`): flusso end‑to‑end (estrazione → embedding → clustering → classificazione → salvataggio).
- Ensemble Classifier (`Classification/advanced_ensemble_classifier.py`): ML+LLM con quality gate.
- Clustering (`Clustering/hdbscan_clusterer.py`): HDBSCAN/UMAP (CPU/GPU), rappresentanti e propagazione.
- Embedding Engine (`EmbeddingEngine/`): manager modulare (LaBSE, BGE, OpenAI...).
- Storage:
  - MySQL (origine + TAG locale per tenants/tags/soglie)
  - MongoDB (risultati e metadata)
  - File JSONL (training decisions per tenant)

## Setup e Avvio

1) Requisiti
- Python 3.11, Docker (consigliato), MySQL, MongoDB, opzionale GPU

2) Configurazione base
- Copia `config_template.yaml` → `config.yaml` e configura: MySQL, MongoDB, LLM/Embedding, parametri default.

3) Docker Compose (consigliato)
- Monta una directory host scrivibile su `/data/training` e imposta `TRAINING_DATA_DIR=/data/training` (compose già allineato).

4) Avvio
```bash
docker compose build && docker compose up -d
# Backend health: http://localhost:5000/health
```

## Note Operative

- Etichette normalizzate in Mongo (MAIUSCOLO + underscore) al salvataggio → elimina duplicati case‑insensitive.
- Se la directory configurata non è scrivibile, salvataggio training su `/tmp/classificatore/training` (l’UI la legge comunque).
- Il clamp automatico dei token evita errori quando si cambia modello LLM con limiti diversi.

