# Sistema di Classificazione Conversazioni Multi-Tenant

## Panoramica Generale

Questo progetto è un sistema avanzato per la classificazione automatica di conversazioni testuali (come chat, email o ticket di supporto). È progettato per essere **multi-tenant**, il che significa che può gestire più clienti contemporaneamente, mantenendo i loro dati, modelli e configurazioni completamente isolati.

L'obiettivo principale è analizzare grandi volumi di conversazioni per:
1.  **Classificare** ogni conversazione assegnandole un'etichetta (o "tag") da un set predefinito (es. "Supporto Tecnico", "Informazioni Commerciali").
2.  **Scoprire** nuove categorie o argomenti emergenti in modo non supervisionato attraverso il clustering.
3.  **Fornire** insight operativi e strategici sui contenuti delle interazioni con i clienti.

Il sistema è costruito per essere robusto, scalabile ed efficiente, sfruttando tecniche di Machine Learning, Deep Learning (LLM) e accelerazione hardware (GPU).

---

## Architettura

Il sistema è composto da diversi moduli specializzati che collaborano per orchestrare i flussi di lavoro di analisi.

![Diagramma Architettura (placeholder)](https://via.placeholder.com/800x400.png?text=Diagramma+Architettura)

### Componenti Chiave

-   **API Server (`server.py`)**:
    -   Basato su **Flask**, espone le funzionalità del sistema tramite un'interfaccia RESTful.
    -   Gestisce le richieste in entrata, orchestra le pipeline per i diversi tenant e restituisce i risultati.

-   **Orchestratore (`Pipeline/end_to_end_pipeline.py`)**:
    -   È il cuore del sistema, la classe `EndToEndPipeline` gestisce l'intero processo, dall'estrazione dei dati al salvataggio dei risultati.
    -   Coordina il lavoro di tutti gli altri componenti (embedding, clustering, classificazione).

-   **Classificatore Ensemble (`Classification/advanced_ensemble_classifier.py`)**:
    -   Un sofisticato classificatore ibrido che combina:
        -   **Modelli ML Tradizionali**: Un `VotingClassifier` che include RandomForest, SVM e Logistic Regression per riconoscere pattern statistici.
        -   **Large Language Model (LLM)**: Un modello linguistico (es. Mistral) per una profonda comprensione semantica del testo.
    -   Utilizza una strategia di "voting" intelligente per combinare le predizioni, risolvendo i disaccordi e sfruttando i punti di forza di entrambi gli approcci.

-   **Motore di Clustering (`Clustering/hdbscan_clusterer.py`)**:
    -   Utilizza l'algoritmo **HDBSCAN** per raggruppare conversazioni simili in cluster, senza bisogno di etichette predefinite.
    -   **Supporto GPU**: Sfrutta `cuML` per eseguire il clustering su GPU, ottenendo un'accelerazione drastica (fino a 10-20x più veloce della CPU).
    -   **Clustering Incrementale**: Supporta la persistenza del modello, permettendo di classificare nuovi dati in cluster esistenti senza dover rieseguire l'analisi completa.

-   **Topic Modeling (`TopicModeling/bertopic_feature_provider.py`)**:
    -   Usa **BERTopic** per analizzare il corpus e identificare i topic principali.
    -   Le informazioni sui topic vengono usate come *feature aggiuntive* per arricchire i dati di training del classificatore ML, migliorandone l'accuratezza.

-   **Embedding Engine (`EmbeddingEngine/`)**:
    -   **Architettura Modulare**: Basato su un'interfaccia astratta (`base_embedder.py`), permette di integrare facilmente diversi modelli di embedding (es. LaBSE, OpenAI, BGE).
    -   **Implementazione Principale**: Utilizza **LaBSE** (`labse_embedder.py`), un modello di sentence-transformer agnostico rispetto alla lingua, ideale per contesti multilingue.
    -   **Gestione Robusta (`simple_embedding_manager.py`)**: Un manager singleton gestisce il ciclo di vita degli embedder. Per garantire l'isolamento tra tenant e prevenire conflitti di memoria (specialmente su GPU), il manager mantiene un solo modello attivo alla volta. Quando la configurazione cambia, esegue un "reset atomico", distruggendo l'istanza corrente e creandone una nuova, garantendo così uno stato sempre pulito.
    -   Responsabile della conversione del testo in vettori numerici (embeddings) di alta qualità.

-   **Storage**:
    -   **MySQL (Database Sorgente)**: Il sistema si connette a un database MySQL remoto per leggere le conversazioni originali da analizzare. Questo è il punto di partenza del flusso di dati.
    -   **MySQL (Database Locale `TAG`)**: Un database MySQL locale (`TagDatabase/`) funge da cache di configurazione. Contiene le informazioni sui tenant e, soprattutto, l'elenco dei tag di classificazione validi per ciascuno di essi. Questo garantisce che il sistema utilizzi sempre un set di etichette coerente.
    -   **MongoDB (Database dei Risultati)**: Utilizzato per salvare i risultati dettagliati delle classificazioni e delle analisi (`MongoDB/`). Per ogni conversazione, viene creato un documento JSON che include l'etichetta predetta, la confidenza, l'ID del cluster e il vettore di embedding. Questo approccio sfrutta la flessibilità di MongoDB per dati semi-strutturati e permette un'analisi approfondita dei risultati.

---

## Flussi di Lavoro (Workflow)

Il sistema supporta diversi flussi di lavoro, attivabili tramite API.

### 1. Training Supervisionato

-   **Endpoint**: `POST /train/<client_name>`
-   **Scopo**: Addestrare i modelli di classificazione su un set di dati etichettato.
-   **Processo**:
    1.  **Estrazione Dati**: Le conversazioni e le relative etichette vengono lette dal database MySQL.
    2.  **Preprocessing**: Il testo viene pulito e preparato per l'analisi.
    3.  **Embedding**: Le conversazioni vengono trasformate in vettori numerici tramite l'Embedding Engine.
    4.  **Feature Enhancement**: Il modello BERTopic viene addestrato sul corpus per estrarre feature relative ai topic.
    5.  **Training del Classificatore**: L'`AdvancedEnsembleClassifier` viene addestrato utilizzando gli embedding e le feature di topic.
    6.  **Salvataggio Modelli**: I modelli addestrati (classificatore, BERTopic, etc.) vengono salvati su disco per un uso futuro.

### 2. Classificazione Completa (con Re-Clustering)

-   **Endpoint**: `POST /classify/all/<client_name>?force_reprocess=true`
-   **Scopo**: Analizzare l'intero set di dati di un cliente, rieseguendo il clustering per scoprire nuove categorie.
-   **Processo**:
    1.  **Caricamento Modelli**: Vengono caricati i modelli di classificazione precedentemente addestrati.
    2.  **Estrazione Dati**: Tutte le conversazioni del cliente vengono estratte.
    3.  **Embedding**: Tutte le conversazioni vengono vettorizzate.
    4.  **Clustering Completo**: `HDBSCANClusterer` viene eseguito sull'intero set di embedding. Questo processo (potenzialmente lungo ma accelerato da GPU) identifica cluster di conversazioni simili. Viene salvato un nuovo modello di clustering.
    5.  **Classificazione**: L'`AdvancedEnsembleClassifier` predice l'etichetta per ogni conversazione.
    6.  **Salvataggio Risultati**: I risultati (etichetta predetta, cluster, confidenza, etc.) vengono salvati in MongoDB.

### 3. Classificazione Incrementale (Veloce)

-   **Endpoint**: `POST /classify/new/<client_name>`
-   **Scopo**: Analizzare solo le nuove conversazioni in modo rapido ed efficiente.
-   **Processo**:
    1.  **Caricamento Modelli**: Vengono caricati sia i modelli di classificazione sia il **modello di clustering** salvato durante l'ultima esecuzione completa.
    2.  **Estrazione Dati**: Vengono estratte solo le conversazioni nuove (non ancora analizzate).
    3.  **Embedding**: Solo le nuove conversazioni vengono vettorizzate.
    4.  **Clustering Incrementale**: Invece di rieseguire il clustering, il modello HDBSCAN caricato viene usato per **predire** a quale cluster esistente appartengono le nuove conversazioni (`predict_new_points`). Questo passo è estremamente veloce.
    5.  **Classificazione**: L'ensemble classifier assegna un'etichetta a ogni nuova conversazione.
    6.  **Salvataggio Risultati**: I risultati vengono aggiunti in MongoDB.

---

## Funzionalità Chiave

-   **Multi-tenancy**: Pieno supporto per gestire più clienti in modo isolato e sicuro.
-   **Classificazione Ibrida**: Combina la velocità e la capacità di generalizzazione dei modelli ML con la profonda comprensione contestuale degli LLM.
-   **Accelerazione GPU**: Utilizzo di `cuML` per il clustering HDBSCAN, che riduce i tempi di elaborazione da ore a minuti su grandi dataset.
-   **Efficienza Incrementale**: La capacità di processare nuovi dati senza rielaborare l'intero storico permette un'analisi quasi in tempo reale.
-   **Configurabilità Estrema**: La maggior parte dei parametri (soglie di confidenza, parametri di clustering, etc.) è definita nel file `config.yaml` e può essere modificata dinamicamente.
-   **Human-in-the-loop**: Il sistema è predisposto per l'integrazione con un'interfaccia di revisione umana (`human-review-ui/`), essenziale per il controllo qualità, la correzione degli errori e il riaddestramento continuo dei modelli.

---

## Guida Rapida

### Prerequisiti

-   Python 3.9+
-   NVIDIA GPU con driver CUDA e cuDF/cuML installati (per il supporto GPU)
-   Accesso a un'istanza MySQL e MongoDB.

### Installazione

1.  Clonare il repository:
    ```bash
    git clone <repository-url>
    cd classificatore
    ```

2.  Creare un ambiente virtuale e installare le dipendenze:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
    *Nota: per il supporto GPU, potrebbero essere necessarie versioni specifiche di alcune librerie. Fare riferimento a `requirements_bertopic.txt` e alla documentazione di RAPIDS.*

### Configurazione

1.  Copiare `config.example.yaml` in `config.yaml`.
2.  Modificare `config.yaml` per inserire:
    -   Le credenziali di connessione per MySQL e MongoDB.
    -   I percorsi per il salvataggio dei modelli.
    -   I parametri per il clustering e la classificazione.
    -   L'URL dell'endpoint per l'LLM (se si usa un servizio come Ollama).

### Avvio del Server

Eseguire il server Flask:
```bash
python server.py
```
Il server sarà in ascolto sulla porta specificata nella configurazione (default: 5000).

### Esempi di Chiamate API

-   **Addestrare i modelli per il cliente "humanitas"**:
    ```bash
    curl -X POST http://localhost:5000/train/humanitas
    ```

-   **Eseguire una classificazione completa per "humanitas"**:
    ```bash
    curl -X POST "http://localhost:5000/classify/all/humanitas?force_reprocess=true"
    ```

-   **Classificare solo le nuove conversazioni per "humanitas"**:
    ```bash
    curl -X POST http://localhost:5000/classify/new/humanitas
    ```
