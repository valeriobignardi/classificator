"""
Autore: Valerio Bignardi
Data di creazione: 2025-09-02
Scopo: Documentazione completa del flusso di training supervisionato
       con focus su gestione rappresentanti, outlier e propagazione
"""

# FLUSSO COMPLETO DEL TRAINING SUPERVISIONATO

## PANORAMICA GENERALE

Il training supervisionato nel sistema di classificazione multi-tenant è un processo sofisticato che combina:
- **Clustering automatico** (HDBSCAN + UMAP) 
- **Selezione intelligente dei rappresentanti**
- **Classificazione LLM dei rappresentanti**
- **Review umano interattivo** 
- **Propagazione delle etichette**
- **Training dell'ensemble ML+LLM**

## FASI DETTAGLIATE DEL PROCESSO

### FASE 1: PREPARAZIONE DATI E EMBEDDING

**File**: `Pipeline/end_to_end_pipeline.py` → metodo `_esegui_clustering_completo()`

1. **Estrazione sessioni** dal database MongoDB remoto
2. **Creazione embeddings** con LaBSE (embedding engine)
3. **Riduzione dimensionalità** con UMAP (se configurato)

**Output**: 
- `sessioni: Dict[session_id, session_data]`
- `embeddings: numpy.ndarray` (vettori numerici delle conversazioni)
- `session_ids: List[str]` (identificatori sessioni)

### FASE 2: CLUSTERING GERARCHICO/INTELLIGENTE

**File**: `Pipeline/end_to_end_pipeline.py` → metodo `_esegui_clustering_completo()`

Il sistema supporta 3 modalità di clustering:

#### A) Clustering Gerarchico (Default)
```python
if hierarchical_config.get('enabled', True):
    hierarchical_clusterer = HierarchicalIntentClusterer(...)
    session_memberships, cluster_info = hierarchical_clusterer.cluster_hierarchically(...)
```

#### B) Clustering Intelligente LLM+ML
```python
elif intelligent_config.get('enabled', True):
    intelligent_clusterer = IntelligentIntentClusterer(...)
    cluster_labels, cluster_info = intelligent_clusterer.cluster_intelligently(...)
```

#### C) Clustering HDBSCAN Geometrico (Fallback)
```python
else:
    cluster_labels = self.clusterer.fit_predict(embeddings)
```

**Output**:
- `cluster_labels: numpy.ndarray` (etichette cluster per ogni sessione)
- `cluster_info: Dict` (metadati per ogni cluster)

**Parametri Clustering HDBSCAN**:
- `min_cluster_size`: Dimensione minima cluster (default: 17)
- `min_samples`: Campioni minimi per densità (default: 5)  
- `cluster_selection_epsilon`: Soglia di selezione (default: 0.5)

### FASE 3: GESTIONE DEGLI OUTLIER

**Localizzazione**: `Pipeline/end_to_end_pipeline.py` → linee 1134-1180

Gli **outlier** sono sessioni che HDBSCAN non riesce a raggruppare in nessun cluster (cluster_id = -1).

#### Selezione Rappresentanti Outlier

```python
# Trova tutti gli outlier
outlier_indices = [i for i, label in enumerate(cluster_labels) if label == -1]
n_outliers = len(outlier_indices)

# Seleziona massimo 5 rappresentanti outlier più diversi
if n_outliers > max_outlier_reps:
    # Selezione per massima diversità usando distanza coseno
    outlier_embeddings = embeddings[outlier_indices]
    distances = cosine_distances(outlier_embeddings)
    
    selected_outlier_indices = [outlier_indices[0]]
    for _ in range(min(4, len(outlier_indices) - 1)):
        # Trova outlier più distante da quelli già selezionati
        max_min_dist = -1
        best_idx = -1
        for idx in outlier_indices:
            if idx not in selected_outlier_indices:
                min_dist_to_selected = min(distances[...])
                if min_dist_to_selected > max_min_dist:
                    max_min_dist = min_dist_to_selected
                    best_idx = idx
        
        if best_idx != -1:
            selected_outlier_indices.append(best_idx)
```

#### Creazione Rappresentanti Outlier

```python
# Crea dati rappresentanti per outlier
for idx in selected_outlier_indices:
    session_id = session_ids[idx]
    session_data = sessioni[session_id].copy()
    session_data['session_id'] = session_id
    session_data['classification_confidence'] = 0.3  # Bassa confidenza
    session_data['classification_method'] = 'outlier'
    outlier_representatives.append(session_data)

# Aggiungi outlier ai representatives con cluster_id = -1
representatives[-1] = outlier_representatives
suggested_labels[-1] = "Casi Outlier"
```

### FASE 4: SELEZIONE RAPPRESENTANTI CLUSTER

**Localizzazione**: `Pipeline/end_to_end_pipeline.py` → linee 1080-1130

Per ogni **cluster valido** (cluster_id ≥ 0):

#### Strategia di Selezione

```python
for cluster_id, info in cluster_info.items():
    cluster_indices = info['indices']
    
    # Selezione intelligente: massimo 3 rappresentanti più diversi
    if len(cluster_indices) <= 3:
        selected_indices = cluster_indices  # Usa tutti
    else:
        # Seleziona 3 punti più distanti tra loro
        cluster_embeddings = embeddings[cluster_indices]
        distances = cosine_distances(cluster_embeddings)
        
        selected_indices = [cluster_indices[0]]  # Primo punto
        
        for _ in range(2):  # Altri 2 punti
            max_min_dist = -1
            best_idx = -1
            
            for idx in cluster_indices:
                if idx not in selected_indices:
                    min_dist_to_selected = min(...)  # Distanza minima dai selezionati
                    if min_dist_to_selected > max_min_dist:
                        max_min_dist = min_dist_to_selected
                        best_idx = idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
```

#### Creazione Struttura Rappresentanti

```python
for idx in selected_indices:
    session_id = session_ids[idx]
    session_data = sessioni[session_id].copy()
    session_data['session_id'] = session_id
    
    # Aggiungi metadati clustering
    if 'average_confidence' in info:
        session_data['classification_confidence'] = info['average_confidence']
    if 'classification_method' in info:
        session_data['classification_method'] = info['classification_method']
        
    cluster_representatives.append(session_data)

representatives[cluster_id] = cluster_representatives
suggested_labels[cluster_id] = info['intent_string']  # Etichetta suggerita
```

### FASE 5: CLASSIFICAZIONE RAPPRESENTANTI

**Localizzazione**: `Pipeline/end_to_end_pipeline.py` → metodo `classifica_e_salva_sessioni()` → linee 4000-4150

Questa è la **fase più critica** dove solo i rappresentanti vengono classificati con l'ensemble ML+LLM.

#### Processo di Classificazione

```python
representative_predictions = {}
total_representatives = sum(len(reps) for reps in representatives.values())

for cluster_id, reps in representatives.items():
    cluster_predictions = []
    
    for rep in reps:
        rep_text = rep['testo_completo']
        
        try:
            # CLASSIFICAZIONE CON ENSEMBLE ML+LLM
            prediction = self.ensemble_classifier.predict_with_ensemble(
                rep_text,
                return_details=True,
                embedder=self.embedder
            )
            
            # Aggiungi metadati cluster
            prediction['representative_session_id'] = rep['session_id']
            prediction['cluster_id'] = cluster_id
            cluster_predictions.append(prediction)
            
        except Exception as e:
            # Fallback in caso di errore
            cluster_predictions.append({
                'predicted_label': 'altro',
                'confidence': 0.3,
                'ensemble_confidence': 0.3,
                'method': 'REP_FALLBACK',
                'representative_session_id': rep['session_id'],
                'cluster_id': cluster_id
            })
    
    representative_predictions[cluster_id] = cluster_predictions
```

**Dettagli Ensemble Prediction**:
L'`ensemble_classifier.predict_with_ensemble()` combina:
- **ML Prediction**: Modelli tradizionali (Random Forest, SVM, etc.)
- **LLM Prediction**: IntelligentClassifier via Ollama/OpenAI
- **Consenso**: Algoritmo di fusione per etichetta finale

### FASE 6: LOGICA DI CONSENSO E PROPAGAZIONE

**Localizzazione**: `Pipeline/end_to_end_pipeline.py` → linee 4100-4200

#### Calcolo del Consenso per Cluster

```python
cluster_final_labels = {}

for cluster_id, predictions in representative_predictions.items():
    if not predictions:
        continue
        
    # Conta le etichette per trovare consenso
    label_counts = {}
    for pred in predictions:
        label = pred.get('predicted_label', 'altro')
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Trova etichetta più votata
    most_common_label = max(label_counts.keys(), key=lambda k: label_counts[k])
    consensus_votes = label_counts[most_common_label]
    total_votes = len(predictions)
    consensus_ratio = consensus_votes / total_votes
    
    # LOGICA CONSENSO (soglia 70%)
    if consensus_ratio >= 0.7:
        auto_classified = True
        review_needed = False
        reason = f"consenso_{consensus_ratio:.0%}"
    elif consensus_ratio == 0.5 and total_votes == 2:
        auto_classified = False
        review_needed = True
        reason = "pareggio_50_50"
    else:
        auto_classified = False  
        review_needed = True
        reason = f"consenso_basso_{consensus_ratio:.0%}"
    
    # Scegli migliore predizione per confidence
    best_prediction = max(predictions, key=lambda p: p.get('confidence', 0))
    
    cluster_final_labels[cluster_id] = {
        'label': most_common_label,
        'confidence': best_prediction.get('confidence', 0.5),
        'consensus_ratio': consensus_ratio,
        'total_representatives': total_votes,
        'needs_review': review_needed,
        'reason': reason,
        'method': 'CLUSTER_PROPAGATED',
        'source_representative': best_prediction['representative_session_id']
    }
```

### FASE 7: REVIEW UMANO INTERATTIVO

**File**: `HumanReview/interactive_trainer.py` → metodo `review_cluster_representatives()`

#### Ordine di Processamento

```python
# Separa cluster normali da outlier
normal_clusters = [cid for cid in suggested_labels.keys() if cid >= 0]
outlier_clusters = [cid for cid in suggested_labels.keys() if cid == -1]

# Ordina: prima cluster crescenti (0,1,2...), poi outlier (-1)
normal_clusters_sorted = sorted(normal_clusters)
processing_order = normal_clusters_sorted + outlier_clusters
```

#### Review di Ogni Cluster

```python
for cluster_id in processing_order:
    if cluster_id in representatives:
        suggested_label = suggested_labels[cluster_id]
        cluster_reps = representatives[cluster_id]
        
        # Messaggio specifico per outlier
        if cluster_id == -1:
            print(f"🔍 PROCESSAMENTO OUTLIER ({len(cluster_reps)} rappresentanti)")
        
        # Review umano interattivo
        final_label, human_confidence = self.interactive_trainer.review_cluster_representatives(
            cluster_id=cluster_id,
            representatives=cluster_reps,
            suggested_label=suggested_label
        )
        
        # Aggiorna etichetta se modificata dall'umano
        if final_label != suggested_label:
            reviewed_labels[cluster_id] = final_label
```

#### Interfaccia Review

L'`interactive_trainer` presenta:
1. **ID del cluster** e etichetta suggerita
2. **Conversazioni rappresentative** (max 3 esempi)
3. **Proposta LLM** (se disponibile)
4. **Opzioni umano**: 
   - Accettare etichetta suggerita
   - Modificare etichetta 
   - Assegnare nuova etichetta personalizzata

### FASE 8: TRAINING ENSEMBLE ML

**Localizzazione**: `Pipeline/end_to_end_pipeline.py` → metodo `allena_classificatore()` → linee 1750-1850

#### Preparazione Dati Training

```python
# Prepara dati con etichette reviewed
session_ids = list(sessioni.keys())
session_texts = [sessioni[sid]['testo_completo'] for sid in session_ids]
train_embeddings = self._get_embedder().encode(session_texts, session_ids=session_ids)

# Crea training labels con gestione outlier
train_labels = []
for i, session_id in enumerate(session_ids):
    cluster_id = cluster_labels[i] if i < len(cluster_labels) else -1
    
    # Usa etichetta reviewed se disponibile
    if cluster_id in reviewed_labels:
        final_label = reviewed_labels[cluster_id]
        label_source = "reviewed"
    else:
        final_label = 'altro'  # Fallback
        label_source = "fallback"
    
    train_labels.append(final_label)
```

#### Feature Augmentation con BERTopic

```python
# Usa BERTopic pre-addestrato per feature enhancement
ml_features = train_embeddings
bertopic_provider = getattr(self, '_bertopic_provider_trained', None)

if bertopic_provider is not None:
    try:
        tr = bertopic_provider.transform(
            session_texts,
            embeddings=train_embeddings,
            return_one_hot=True,
            top_k=15
        )
        
        topic_probas = tr.get('topic_probas') 
        one_hot = tr.get('one_hot')
        
        # Concatena features: [embeddings + topic_probas + one_hot]
        parts = [train_embeddings]
        if topic_probas is not None and topic_probas.size > 0:
            parts.append(topic_probas)
        if one_hot is not None and one_hot.size > 0:
            parts.append(one_hot)
            
        ml_features = np.concatenate(parts, axis=1)
        
        # Inietta provider nell'ensemble per inferenza
        self.ensemble_classifier.set_bertopic_provider(bertopic_provider, ...)
        
    except Exception as e:
        print(f"⚠️ Errore BERTopic: {e}, uso solo embeddings")
```

#### Training Ensemble

```python
# Allena ensemble ML avanzato  
metrics = self.ensemble_classifier.train_ml_ensemble(ml_features, train_labels)
```

L'ensemble addestra simultaneamente:
- **Random Forest**
- **Support Vector Machine**
- **Gradient Boosting**
- **Logistic Regression**
- **Neural Network** (se abilitato)

### FASE 9: PROPAGAZIONE FINALE ALLE SESSIONI

**Localizzazione**: `Pipeline/end_to_end_pipeline.py` → metodo `_propagate_labels_to_sessions()` → linee 4400-4600

#### Propagazione da Cluster a Sessioni

```python
for i, session_id in enumerate(session_ids):
    session_data = sessioni[session_id]
    cluster_id = cluster_labels[i] if i < len(cluster_labels) else -1
    
    if cluster_id in reviewed_labels:
        # PROPAGAZIONE DA CLUSTER
        final_label = reviewed_labels[cluster_id]
        confidence = 0.85  # Alta confidenza per propagazione
        method = 'CLUSTER_PROPAGATION'
        notes = f"Propagata da cluster {cluster_id}"
        
    else:
        # CLASSIFICAZIONE INDIVIDUALE OUTLIER
        session_text = session_data.get('testo_completo', '')
        
        if session_text and hasattr(self, 'ensemble_classifier'):
            try:
                # Classifica outlier con ensemble completo ML+LLM
                outlier_prediction = self.ensemble_classifier.predict_with_ensemble(
                    session_text,
                    return_details=True,
                    embedder=self.embedder
                )
                
                final_label = outlier_prediction.get('predicted_label', 'altro')
                confidence = outlier_prediction.get('ensemble_confidence', 0.5)
                method = 'OUTLIER_ENSEMBLE_CLASSIFICATION'
                
            except Exception as e:
                # Fallback per errori
                final_label = 'altro'
                confidence = 0.3
                method = 'OUTLIER_FALLBACK'
```

#### Validazione Tag "ALTRO" per Outlier

```python
# Validazione speciale per tag "altro"
if final_label == 'altro' and hasattr(self, 'interactive_trainer') and self.interactive_trainer.altro_validator:
    try:
        conversation_text = session_data.get('testo_completo', '')
        if conversation_text:
            # Esegui validazione automatica del tag "altro"
            validated_label, validated_confidence, validation_info = self.interactive_trainer.handle_altro_classification(
                conversation_text=conversation_text,
                force_human_decision=False
            )
            
            # Usa risultato validazione se diverso da "altro"
            if validated_label != 'altro':
                final_label = validated_label
                confidence = validated_confidence
                method = f"{method}_ALTRO_VAL"
                print(f"✅ OUTLIER RICLASSIFICATO: {session_id} 'altro' -> '{validated_label}'")
    except Exception as e:
        print(f"⚠️ Errore validazione altro per outlier {session_id}: {e}")
```

### FASE 10: SALVATAGGIO MONGODB UNIFICATO

**Localizzazione**: `Pipeline/end_to_end_pipeline.py` → linee 4500-4600

#### Estrazione Embedding per MongoDB

```python
# Estrai embedding finale per MongoDB
try:
    # Ottieni embedding per questa sessione
    session_embedding = None
    if hasattr(self.clusterer, 'final_embeddings') and self.clusterer.final_embeddings is not None:
        # Usa embeddings finali processati dal clusterer (post-UMAP)
        if i < len(self.clusterer.final_embeddings):
            session_embedding = self.clusterer.final_embeddings[i]
    elif i < len(embeddings):
        # Fallback agli embeddings originali
        session_embedding = embeddings[i]
    
    # Ottieni nome embedder con info UMAP
    embedder_name = self._get_embedder_name()
    
    # Salva in MongoDB con embedding
    mongo_reader.save_classification_result(
        session_id=session_id,
        predicted_label=final_label,
        confidence=confidence,
        method=method,
        notes=notes,
        ml_result=ml_result,
        llm_result=llm_result,
        embedding=session_embedding,  # 🆕 Embedding salvato
        embedding_model=embedder_name  # 🆕 Modello embedding salvato
    )
    
except Exception as e:
    print(f"❌ Errore salvataggio MongoDB per {session_id}: {e}")
```

## GESTIONE OUTLIER: APPROFONDIMENTO

Gli **outlier** richiedono gestione speciale perché:

1. **Non appartengono a cluster**: HDBSCAN li marca con cluster_id = -1
2. **Sono eterogenei**: Rappresentano conversazioni diverse tra loro
3. **Richiedono classificazione individuale**: Non possono beneficiare della propagazione

### Strategia Outlier

1. **Selezione Rappresentanti**: Max 5 outlier più diversi per review umano
2. **Review Specifico**: Cluster speciale "Casi Outlier" per etichettatura manuale
3. **Classificazione Individuale**: Ogni outlier classificato con ensemble ML+LLM
4. **Validazione ALTRO**: Outlier classificati come "altro" vengono rivalidati

### Perché gli Outlier sono Importanti

- **Copertura Completa**: Garantiscono che tutte le sessioni abbiano un'etichetta
- **Casi Edge**: Catturano conversazioni atipiche o nuove categorie
- **Quality Assurance**: Evitano perdita di informazioni importanti

## METRICHE E MONITORAGGIO

Il sistema traccia:

- **Clustering Success Rate**: Percentuale sessioni clusterizzate vs outlier
- **Consensus Rate**: Percentuale cluster con consenso ≥ 70%  
- **Human Review Rate**: Percentuale cluster che richiedono review umano
- **Classification Accuracy**: Accuratezza ensemble ML sui dati di test
- **Propagation Statistics**: Distribuzione etichette propagate

## OTTIMIZZAZIONI PRESTAZIONI

1. **Classificazione Solo Rappresentanti**: 150-200 rappresentanti invece di 10k+ sessioni
2. **Embedding Caching**: Embeddings riutilizzati tra fasi
3. **GPU Acceleration**: HDBSCAN e UMAP su GPU quando disponibile
4. **Batch Processing**: Classificazioni raggruppate per efficienza

## CONCLUSIONI

Il training supervisionato è un processo **ibrido umano-AI** che combina:

- **Automazione intelligente** (clustering, selezione rappresentanti)
- **Supervisione umana** (review interattivo, validazione)
- **Machine Learning avanzato** (ensemble, feature augmentation)
- **Gestione robusta degli edge case** (outlier, consenso basso)

Questo approccio garantisce **alta qualità** delle etichette mantenendo **scalabilità** operativa.
