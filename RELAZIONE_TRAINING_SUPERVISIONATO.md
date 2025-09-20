# Analisi della Pipeline di Training Supervisionato

Questo documento analizza in dettaglio il flusso di training supervisionato implementato nella classe `EndToEndPipeline`. La pipeline è progettata per essere un processo a più fasi, che parte dalla classificazione iniziale dei dati, passa per la revisione umana e culmina nel riaddestramento del modello di Machine Learning.

## Architettura Generale

Il training supervisionato è suddiviso in tre fasi logiche principali:

1.  **Fase 1: Classificazione Iniziale e Preparazione Dati**: Questa fase classifica un nuovo set di sessioni, utilizzando l'LLM quando un modello ML non è ancora disponibile, e prepara i dati per la revisione umana.
2.  **Fase 2: Revisione Umana e Correzione**: Un operatore umano corregge le etichette generate automaticamente attraverso un'interfaccia esterna (es. UI React). Le correzioni vengono salvate.
3.  **Fase 3: Riaddestramento del Modello ML**: Il modello di Machine Learning viene riaddestrato utilizzando i dati puliti e verificati dalla fase di revisione.

Di seguito, un'analisi dettagliata delle funzioni chiave per ogni fase.

---

## Fase 0: Scoperta di Nuovi Intent tramite Clustering

Prima ancora che il training supervisionato possa iniziare, è fondamentale analizzare le conversazioni non etichettate per scoprire nuovi intent o raggruppare quelle simili. Questa fase è gestita principalmente da `IntelligentIntentClusterer`, che utilizza un approccio ibrido ML+LLM.

### 1. `IntelligentIntentClusterer.cluster_intelligently(...)`

-   **Scopo**: È la funzione principale che orchestra il processo di clustering. L'obiettivo è raggruppare le conversazioni in cluster semanticamente coerenti e assegnare a ciascun cluster un'etichetta (intent) significativa.
-   **Architettura Ottimizzata**: Invece di classificare ogni singola conversazione con l'LLM (un processo costoso e lento), viene adottata una strategia ottimizzata in più fasi:
    1.  **Clustering Semantico con HDBSCAN**: Tutte le conversazioni vengono prima raggruppate in base alla vicinanza semantica dei loro embedding. Questo viene fatto da `HDBSCANClusterer`.
    2.  **Selezione dei Rappresentanti**: Per ogni cluster generato, vengono selezionati alcuni esempi "rappresentanti" (di solito da 3 a 5) utilizzando una strategia di selezione diversificata per catturare la variabilità del cluster.
    3.  **Analisi LLM solo sui Rappresentanti**: Solo i testi dei rappresentanti vengono inviati all'LLM per la classificazione. Questo riduce drasticamente il numero di chiamate all'LLM.
    4.  **Consenso e Propagazione**: L'etichetta del cluster viene decisa tramite un meccanismo di consenso basato sulle risposte dell'LLM per i suoi rappresentanti. L'etichetta decisa viene poi propagata a tutte le conversazioni all'interno di quel cluster.
    5.  **Gestione degli Outlier**: Le conversazioni che non rientrano in nessun cluster (outlier) vengono analizzate individualmente con l'LLM per tentare di recuperarle o raggrupparle in nuovi micro-cluster.

### 2. `HDBSCANClusterer.fit_predict(...)`

-   **Scopo**: Questa funzione, contenuta nel modulo `hdbscan_clusterer.py`, è il motore del clustering semantico. Utilizza l'algoritmo HDBSCAN, che è particolarmente efficace perché non richiede di specificare a priori il numero di cluster.
-   **Funzioni Chiave**:
    -   **Supporto UMAP**: Prima di eseguire HDBSCAN, può applicare una riduzione dimensionale tramite UMAP (`_apply_umap_reduction`). Questo aiuta a migliorare la qualità del clustering riducendo il rumore negli embedding ad alta dimensionalità.
    -   **Supporto GPU**: È in grado di sfruttare la GPU tramite `cuML` per accelerare drasticamente il calcolo su grandi moli di dati, con un fallback automatico su CPU in caso di problemi.
    -   **Configurazione Flessibile**: I parametri di HDBSCAN (come `min_cluster_size`, `min_samples`, etc.) e UMAP possono essere configurati dinamicamente per ogni tenant tramite file YAML specifici, permettendo un tuning fine del processo di clustering.
-   **Output**: Restituisce un array di etichette numeriche, dove ogni numero rappresenta un cluster e `-1` indica un outlier.

Questo processo di clustering "intelligente" permette di scoprire e definire nuovi intent in modo data-driven, fornendo la base per la successiva fase di classificazione e revisione umana.

---

## Fase 1: Classificazione e Preparazione

Questa fase è orchestrata principalmente dalla funzione `esegui_training_supervisionato_fase1`.

### 1. `esegui_training_supervisionato_fase1(...)`

-   **Scopo**: È il punto di ingresso della pipeline di training. Riceve le sessioni già processate da un clustering preliminare e gestisce la loro classificazione iniziale.
-   **Funzioni Richiamate**:
    -   `_classify_sessions_batch_llm_only(...)` (se il modello ML non è disponibile).
    -   `_get_mongo_reader()` per ottenere un'istanza per salvare i dati.
-   **Logica**:
    1.  Determina se un modello ML è già addestrato e disponibile.
    2.  **Caso 1: Modello ML non disponibile (primo avvio)**:
        -   Richiama `_classify_sessions_batch_llm_only` per classificare tutte le sessioni usando solo l'LLM. In questo scenario, a tutte le classificazioni viene assegnata una confidenza bassa (es. 0.5) per forzarne la revisione umana.
    3.  **Caso 2: Modello ML disponibile**:
        -   Utilizza l'ensemble classifier (ML+LLM) per ottenere una classificazione più accurata.
    4.  Salva ogni classificazione nel database MongoDB, popolando una coda di revisione (`review_queue`) con i casi che richiedono un controllo umano (es. bassa confidenza, disaccordo tra modelli).
-   **Output**: Restituisce un dizionario con le statistiche della fase, indicando quante sessioni sono state classificate e quante sono state messe in coda per la revisione.

### 2. `_classify_sessions_batch_llm_only(...)`

-   **Scopo**: Eseguire una classificazione di massa utilizzando esclusivamente l'LLM, con ottimizzazioni per le chiamate in batch.
-   **Funzioni Richiamate**:
    -   `self.ensemble_classifier.llm_classifier.classify_multiple_conversations_optimized(...)`
-   **Logica**:
    1.  **Percorso Ottimizzato**: Se il classificatore LLM supporta la classificazione batch (`classify_multiple_conversations_optimized`), invia tutte le conversazioni in un'unica richiesta per massimizzare l'efficienza e ridurre i tempi di attesa.
    2.  **Percorso di Fallback**: Se il metodo ottimizzato fallisce o non è disponibile, la funzione ripiega su un metodo più semplice: assegna a ogni sessione l'etichetta suggerita dal clustering (`suggested_labels`) con una confidenza di base predefinita (es. 0.5). Questo garantisce che il processo continui anche in caso di errori con l'LLM.
-   **Output**: Una lista di dizionari, ognuno rappresentante il risultato della classificazione per una sessione.

---

## Fase 2: Revisione Umana e Correzione

Questa fase avviene esternamente alla pipeline (tramite UI) ma interagisce con essa attraverso funzioni specifiche.

### 1. `update_training_file_with_human_corrections(...)`

-   **Scopo**: Aggiornare il file di training (solitamente un JSON) con le etichette corrette fornite da un operatore umano.
-   **Logica**:
    1.  Carica il file JSON contenente i dati preparati nella Fase 1.
    2.  Cerca il record corrispondente alla `session_id` fornita.
    3.  Aggiorna il campo `final_label` con l'etichetta corretta (`corrected_label`) e imposta un flag `human_reviewed: true`.
    4.  Salva nuovamente il file JSON aggiornato.
-   **Contesto**: Questa funzione viene invocata dall'API del server ogni volta che un utente salva una correzione dall'interfaccia di revisione.

### 2. `aggiungi_caso_come_esempio_llm(...)`

-   **Scopo**: Oltre a correggere i dati per il training ML, questa funzione permette di salvare un caso particolarmente significativo come **esempio positivo** per il fine-tuning futuro dell'LLM.
-   **Logica**:
    1.  Si connette al database MySQL (`TAG`).
    2.  Inserisce la conversazione e la sua etichetta corretta nella tabella degli esempi (`llm_training_examples`).
    3.  Questo dato verrà utilizzato per migliorare i prompt dell'LLM o per creare dataset di fine-tuning.
-   **Contesto**: Viene chiamata quando un revisore identifica un caso esemplare che può aiutare l'LLM a migliorare su casistiche ambigue.

---

## Fase 3: Riaddestramento del Modello ML

Una volta che un numero sufficiente di dati è stato revisionato, si può avviare il riaddestramento del modello ML.

### 1. `load_training_data_from_file(...)`

-   **Scopo**: Caricare i dati di training puliti e corretti dal file JSON, pronti per essere usati per addestrare il modello ML.
-   **Logica**:
    1.  Identifica il file di training più recente (se non ne viene specificato uno).
    2.  Legge il file JSON e lo carica in memoria come una lista di dizionari.
    3.  Filtra i dati per assicurarsi che siano pronti per il training (es. che abbiano un'`final_label`).
-   **Output**: Una lista di dati di training pronti per essere passati al classificatore ML.

### 2. `manual_retrain_model(...)`

-   **Scopo**: È la funzione principale che orchestra il riaddestramento del modello ML. Viene tipicamente invocata manualmente dall'utente tramite un'azione sull'interfaccia.
-   **Funzioni Richiamate**:
    -   `load_training_data_from_file()`
    -   Metodi di training dell'ML ensemble classifier (es. `self.ensemble_classifier.train_ml_ensemble(...)`).
-   **Logica**:
    1.  Chiama `load_training_data_from_file()` per ottenere il dataset di training aggiornato.
    2.  Verifica che ci siano abbastanza dati corretti per procedere con un addestramento significativo.
    3.  Invoca il metodo di training del componente ML dell'ensemble, passandogli i testi e le etichette finali.
    4.  Al termine del training, salva il nuovo modello addestrato su disco.
    5.  Restituisce le metriche di performance del nuovo modello (es. accuracy).
-   **Output**: Un dizionario che riassume l'esito del riaddestramento, incluse le metriche di accuratezza e un messaggio di stato.
