# ANALISI DEL TRAINING SUPERVISIONATO DALL’ESSERE UMANO

Documento tecnico di audit del flusso end‑to‑end per il training supervisionato, convalidato sul codice presente nel repository al 2025-10-24.

## Scopo

- Descrivere il flusso end‑to‑end del training supervisionato (entrypoint → pipeline → review → retraining ML).
- Validare la correttezza logica rispetto a scikit‑learn e alla pipeline di embedding.
- Evidenziare rischi/edge cases e proporre miglioramenti concreti e a basso rischio.

## Entry point e parametri

- Endpoint backend: `server.py` → rotta di training supervisionato (carica soglie da MySQL se presenti, altrimenti fallback config)
  - Chiama: `pipeline.esegui_training_interattivo(max_human_review_sessions, confidence_threshold, force_review, disagreement_threshold)`
  - Allegati ai risultati: `user_configuration` (soglie effettive usate)
- Config globale: `config.yaml` → sezione `supervised_training.human_review` (es. `max_total_sessions`, strategie di selezione) e `pipeline.*` per soglie generali.

## Flusso end‑to‑end (overview)

```
[Client]
   ↓  POST /train/supervised/<tenant>
[server.py]
   ↓ carica soglie (MySQL → fallback config.yaml)
   ↓ chiama pipeline.esegui_training_interattivo(...)
[Pipeline/end_to_end_pipeline.py]
   1) Estrazione completa sessioni (force_full_extraction=True)
   2) Clustering completo (DocumentoProcessing)
   3) Selezione rappresentanti (budget "human_limit")
   4) Classificazione completa (ensemble ML+LLM) e salvataggio
   → Ritorna metriche (extraction/clustering/review/training)
   ↓
[QualityGate/quality_gate_engine.py] (asyncrono/on-demand)
   - evaluate/save → review queue
   - _retrain_ml_model(): primo training vs riaddestramento
       * _load_human_decisions_for_training()
       * _load_llm_classifications_from_mongodb() (solo primo training)
       * _remove_duplicate_training_data()
       * _prepare_training_data() → embeddings per X, labels y
       * _update_ml_model_with_new_data()/train_ml_ensemble()
   ↓
[Classification/advanced_ensemble_classifier.py]
   - train_ml_ensemble(): VotingClassifier (RF + SVM prob + LR)
```

## Contratti essenziali delle funzioni chiave

- `server.py` → supervised training
  - Input: tenant, force_review (opz.), soglie da DB (confidence/disagreement/max sessions)
  - Output: JSON con `extraction_stats`, `clustering_stats`, `human_review_stats`, `training_metrics` e `warnings`
  - Errori: tenant/pipeline non trovata, eccezioni interne (500)

- `EndToEndPipeline.esegui_training_interattivo(...)`
  - Input: `max_human_review_sessions`, `confidence_threshold`, `force_review`, `disagreement_threshold`
  - Output: dict con metriche e riepiloghi fasi 1‑4
  - Errori: dataset troppo piccolo (<5), problemi in estrazione o clustering

- `QualityGateEngine._retrain_ml_model()`
  - Input: training log + (solo first‑train) classificazioni LLM da Mongo
  - Output: boolean `success`; aggiorna/salva modello ML
  - Errori: nessun dato valido; embedding failures; salvataggio modello

- `AdvancedEnsembleClassifier.train_ml_ensemble(X, y)`
  - Input: features (embeddings) e labels
  - Output: metriche di training (train accuracy, counts)
  - Errori: classi insufficienti, shape incoerenti, feature nulle

## Coerenza dei dati lungo la pipeline

- Estrazione completa: la pipeline forza `limit=None` e `force_full_extraction=True` per il dataset totale.
- Clustering: usa l’architettura DocumentoProcessing; calcola statistiche (n_clusters, n_outliers, rappresentanti, propagati).
- Selezione rappresentanti: `select_representatives_from_documents(documenti, max_sessions=human_limit)` con budget dal DB/config; statistiche su cluster inclusi/esclusi.
- Classificazione: `classifica_e_salva_documenti_unified(...)` con ensemble ML+LLM; salva su MongoDB e popola la review queue (rispettando `needs_review`).
- Retraining: QualityGate unisce review umane (training log) + classificazioni LLM (solo al primo training), rimuove duplicati, ricostruisce `conversation_text` (preferendo quello già presente) e genera embeddings multi‑tenant.

Osservazioni:
- La pulizia delle label (es. rimozione caratteri speciali) è gestita a valle nelle componenti LLM; mantenere coerenza anche lato retraining.
- Il fallback a testi sintetici per sessioni non risolvibili garantisce robustezza ma va monitorato per non introdurre bias.

## Correttezza ML vs scikit‑learn

- Ensemble: `VotingClassifier(..., voting='soft')` su RandomForest, SVC(probability=True), LogisticRegression(max_iter=1000) → corretto.
- Mancanze attuali:
  - Solo training accuracy; assente validazione out‑of‑sample (hold‑out o k‑fold).
  - Gestione sbilanciamento classi non esplicita (class_weight/stratify).
  - Nessuna metrica di calibrazione/confusione salvata.

Raccomandazioni immediate:
- Split train/validation stratificato (ad es. 80/20) in `QualityGateEngine` prima del call a `train_ml_ensemble`.
- Metriche aggiuntive: F1‑macro, balanced accuracy, per‑class precision/recall, confusion matrix.
- Class weights: attivare `class_weight='balanced'` su LR/SVM e considerarlo per RF.
- Persistenza artefatti: salvare modello, report metriche e versioning per tenant.

## Efficienza e performance

- Mongo accesso:
  - Evitare `get_all_sessions()` full scan per ogni retraining; introdurre cache o query mirate per i soli `session_id` necessari, con projection dei campi.
  - Indici consigliati su `session_id`, `client_name`, `predicted_label`.
- Embeddings:
  - Riutilizzare l’istanza embedder e processare in batch; conservare cache embeddings per sessioni già usate nel training.
- Job asincroni:
  - Persistenza del registry dei job (oltre l’in‑memory) e retry per robustezza.
- Logging/Osservabilità:
  - Arricchire i log del retraining con tempi per fase e count per etichetta; salvare un JSON di riepilogo per run.

## Rischi ed edge cases

- Dataset piccolo (<5) o classi <2 → blocco training; gestire messaggi chiari e fallback (posticipare retraining).
- Label “altro” o non valide → già filtrate in più punti; continuare a validare contro il db tag.
- Testi mancanti o troppo corti → fallback sintetico; monitorare percentuale di synthetic per label.
- Sbilanciamento forte → abilitare class weights e metriche macro.
- Concorrenza retraining → serializzare o mettere lock per tenant.

## Quick wins (basso rischio)

1) Validation split + metriche out‑of‑sample in `QualityGateEngine` (rapido, valore alto).
2) `class_weight='balanced'` su LR/SVM, revisione RF per class weights.
3) Query mirate a Mongo per `session_id` necessari; riduzione full scans.
4) Persistenza metriche/artefatti modello per tenant con versione e timestamp.
5) Cache embeddings training (fingerprint su session_id + text).

## Next steps proposti

- Implementare le metriche e la validazione nel retraining.
- Introdurre caching e query selettive su Mongo.
- Persistenza dei job asincroni e integrazione frontend del polling.
- Valutare WebSocket per aggiornamenti real‑time della review queue.

## Appendice: riferimenti a file e simboli

- `server.py` → rotta supervised training, chiamata a `pipeline.esegui_training_interattivo(...)`.
- `Pipeline/end_to_end_pipeline.py` → `esegui_training_interattivo`, `select_representatives_from_documents`, `classifica_e_salva_documenti_unified`.
- `QualityGate/quality_gate_engine.py` → `_retrain_ml_model`, `_prepare_training_data`, `_load_llm_classifications_from_mongodb`, `_remove_duplicate_training_data`.
- `Classification/advanced_ensemble_classifier.py` → `train_ml_ensemble` (VotingClassifier RF+SVM+LR).
- `config.yaml` → `supervised_training.*`, `pipeline.*`, `llm.*`.
