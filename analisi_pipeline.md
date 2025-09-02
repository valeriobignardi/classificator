
# Analisi della Pipeline di Classificazione - Trace.log

Questo documento analizza la sequenza di operazioni registrate nel file `Trace.log`, descrivendo in dettaglio ogni fase della pipeline di classificazione delle conversazioni per il tenant "Humanitas".

## Schema Generale della Pipeline

La pipeline elabora le conversazioni in più fasi, dall'estrazione dei dati grezzi alla classificazione finale tramite un modello linguistico di grandi dimensioni (LLM).

1.  **Inizializzazione e Configurazione**
2.  **Estrazione Dati**
3.  **Creazione degli Embedding**
4.  **Riduzione della Dimensionalità (UMAP)**
5.  **Clustering (HDBSCAN)**
6.  **Selezione dei Rappresentanti di Cluster**
7.  **Classificazione tramite LLM**
8.  **Salvataggio dei Risultati**

---

## 1. Inizializzazione e Configurazione

*   **Timestamp:** `2025-09-02T12:15:21`
*   **Tenant:** Humanitas (`015007d9-d413-11ef-86a5-96000228e7fe`)
*   **Azione:** La pipeline viene avviata per il tenant specificato. Vengono caricate le configurazioni, inclusi i parametri per l'embedding, il clustering e il modello LLM.
*   **Log di riferimento:**
    ```
    INFO:root:Inizio del processo di training per il tenant: 015007d9-d413-11ef-86a5-96000228e7fe
    ```

## 2. Estrazione Dati

*   **Timestamp:** `2025-09-02T12:15:22`
*   **Sorgente:** Database MySQL.
*   **Azione:** Vengono estratte le conversazioni da analizzare.
*   **Risultato:** Sono state estratte **3549 conversazioni**.
*   **Log di riferimento:**
    ```
    INFO:LettoreConversazioni.mysql_reader:Estratte 3549 conversazioni per il tenant 015007d9-d413-11ef-86a5-96000228e7fe dal 2025-08-26 al 2025-09-02.
    ```

## 3. Creazione degli Embedding

*   **Timestamp:** `2025-09-02T12:15:22`
*   **Modello di Embedding:** `sentence-transformers/LaBSE`
*   **Azione:** Ogni conversazione viene trasformata in un vettore numerico (embedding) per rappresentarne il significato semantico. Il processo viene eseguito in batch.
*   **Log di riferimento:**
    ```
    INFO:EmbeddingEngine.embedding_service:Creazione degli embedding per 3549 conversazioni in batch da 32...
    INFO:EmbeddingEngine.embedding_service:Embedding creati con successo per 3549 conversazioni.
    ```

## 4. Riduzione della Dimensionalità (UMAP)

*   **Timestamp:** `2025-09-02T12:15:30`
*   **Algoritmo:** UMAP (Uniform Manifold Approximation and Projection)
*   **Azione:** La dimensionalità degli embedding viene ridotta per rendere più efficace il successivo processo di clustering.
*   **Parametri:**
    *   `n_neighbors`: 15
    *   `n_components`: 17
    *   `min_dist`: 0.0
*   **Log di riferimento:**
    ```
    INFO:Clustering.hdbscan_clustering:Esecuzione di UMAP con n_neighbors=15, n_components=17, min_dist=0.0...
    ```

## 5. Clustering (HDBSCAN)

*   **Timestamp:** `2025-09-02T12:15:32`
*   **Algoritmo:** HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
*   **Implementazione:** `cuml.cluster.HDBSCAN` (accelerato su GPU NVIDIA A10G).
*   **Azione:** Gli embedding a dimensionalità ridotta vengono raggruppati in cluster basati sulla loro densità. Le conversazioni che non appartengono a nessun cluster vengono etichettate come "outlier".
*   **Parametri:**
    *   `min_cluster_size`: 17
    *   `min_samples`: 5
    *   `cluster_selection_epsilon`: 0.5
*   **Risultato:**
    *   **36 cluster** identificati.
    *   **577 conversazioni** classificate come outlier.
*   **Log di riferimento:**
    ```
    INFO:Clustering.hdbscan_clustering:Esecuzione di HDBSCAN con min_cluster_size=17, min_samples=5, cluster_selection_epsilon=0.5...
    INFO:Clustering.hdbscan_clustering:Clustering completato. Trovati 36 cluster e 577 outlier.
    ```

## 6. Selezione dei Rappresentanti di Cluster

*   **Timestamp:** `2025-09-02T12:15:33`
*   **Azione:** Da ogni cluster, vengono selezionati alcuni esempi rappresentativi che verranno poi inviati al modello LLM per la classificazione e l'assegnazione di un'etichetta significativa.
*   **Risultato:** Sono stati selezionati **159 campioni rappresentativi** per la classificazione.
*   **Log di riferimento:**
    ```
    INFO:Clustering.hdbscan_clustering:Campioni rappresentativi per la classificazione LLM: 159
    ```

## 7. Classificazione tramite LLM

*   **Timestamp:** `2025-09-02T12:15:33` in poi.
*   **Modello LLM:** `mistral:7b` (eseguito tramite istanza locale di Ollama).
*   **Azione:** Ciascuno dei 159 campioni rappresentativi viene inviato al modello LLM per ottenere un'etichetta, una confidenza e una motivazione.
*   **Dettagli del Processo:**
    *   **Prompt di Sistema:** Al modello viene fornito un prompt di sistema dettagliato che lo istruisce a comportarsi come un "classificatore esperto per l'ospedale Humanitas". Il prompt include la missione, un approccio analitico, una lista di etichette valide (`<|LABELING|>`), e le linee guida per la confidenza.
    *   **Prompt Utente:** Per ogni campione, viene costruito un prompt che include:
        *   Esempi di classificazione (few-shot examples).
        *   Il testo della conversazione da classificare (`<|TESTO DA CLASSIFICARE|>`).
    *   **Output:** Il modello risponde con un JSON strutturato contenente `predicted_label`, `confidence`, e `motivation`.
*   **Esempio di Classificazione (dal log):**
    *   **Testo da Classificare:** `"[UTENTE] sono già registrato in Humanitas conte: devo scaricare referto..."`
    *   **Risposta del Modello:**
        ```json
        {
          "predicted_label": "Referti",
          "confidence": 0.98,
          "motivation": "L'utente chiede come scaricare un referto ed è già registrato al portale Humanitas con te."
        }
        ```
*   **Log di riferimento:**
    ```
    🤖 LLM DEBUG - CLASSIFICATION | 2025-09-02T12:15:40.841395
    ...
    📥 INPUT SECTION
    ...
    📤 OUTPUT SECTION
    ...
    ```

## 8. Salvataggio dei Risultati

*   **Azione:** I risultati della classificazione LLM vengono salvati.
*   **Database di Destinazione:** MongoDB.
*   **Collection:** `humanitas_015007d9-d413-11ef-86a5-96000228e7fe`.
*   **Log di riferimento:**
    ```
    INFO:root:Collection generata: humanitas_015007d9-d413-11ef-86a5-96000228e7fe per tenant Humanitas
    ```
*   **Verifica Tag:** Il sistema controlla anche se l'etichetta (`predicted_label`) esiste già nel database `TAG.db` per il tenant corrente, garantendo la coerenza dei tag.

---
**Fine dell'analisi.**
