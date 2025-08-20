# Training Supervisionato Semplificato

## 🎯 Obiettivo

Il training supervisionato serve ad **insegnare al modello a etichettare correttamente**, così che quando sarà eseguita la modalità "Classifica tutti", il modello sia già addestrato a ridurre al minimo l'intervento umano.

## 🧩 Logica del Sistema

### 📊 Estrazione e Clustering Completo
1. **Estrazione completa**: Il sistema estrae SEMPRE tutte le discussioni dal database
2. **Clustering completo**: Il clustering viene eseguito su tutto il dataset disponibile  
3. **Nessun limite sul clustering**: Il processo di analisi semantica considera tutti i dati

### 👤 Selezione Intelligente per Review Umana
4. **Limite umano**: Solo un numero massimo configurabile di sessioni rappresentative viene sottoposto all'umano
5. **Selezione prioritaria**: Il sistema seleziona i cluster più importanti (grandi, bassa confidenza, nuovi pattern)
6. **Rappresentanti diversificati**: Per ogni cluster selezionato, vengono scelti i rappresentanti più diversi

## 🎛️ Interfaccia Utente Semplificata

L'interfaccia grafica deve permettere all'utente di configurare solo **2 parametri**:

### 1. 📝 Numero Massimo Sessioni per Review
- **Campo**: `max_sessions` 
- **Default**: 500
- **Descrizione**: Numero massimo di sessioni rappresentative da analizzare e potenzialmente sottoporre all'utente
- **Comportamento**:
  - Se tutte le 500 sessioni ricadono nei casi di disaccordo ML/LLM o bassa confidenza → l'utente dovrà etichettarle tutte
  - Se sono meno → l'utente supervisionerà solo quelle necessarie

### 2. 🎯 Soglia di Confidenza
- **Campo**: `confidence_threshold`
- **Default**: 0.7 (70%)
- **Descrizione**: Soglia di confidenza oltre la quale non è necessario l'intervento umano
- **Comportamento**:
  - Classificazioni con confidenza > soglia → automatiche
  - Classificazioni con confidenza ≤ soglia → review umana

## 🔧 Configurazioni di Default (Non Modificabili dall'UI)

Tutti gli altri parametri rimangono nelle configurazioni di default del file `config.yaml`:

```yaml
supervised_training:
  human_review:
    representatives_per_cluster: 3        # Rappresentanti per cluster
    min_representatives_per_cluster: 1   # Minimo rappresentanti
    max_representatives_per_cluster: 5   # Massimo rappresentanti
    selection_strategy: "prioritize_by_size"  # Strategia selezione
    min_cluster_size_for_review: 2       # Dimensione minima cluster
    overflow_handling: "proportional"     # Gestione overflow
```

## 📡 API Endpoint

### POST `/train/supervised/{client_name}`

**Body JSON**:
```json
{
  "max_sessions": 500,         // Numero massimo sessioni per review umana
  "confidence_threshold": 0.7  // Soglia di confidenza
}
```

**Response di successo**:
```json
{
  "success": true,
  "message": "Training supervisionato completato per humanitas",
  "client": "humanitas",
  "user_configuration": {
    "max_sessions": 500,
    "confidence_threshold": 0.7
  },
  "extraction_stats": {
    "total_sessions_extracted": 8547,
    "extraction_mode": "FULL_DATASET"
  },
  "clustering_stats": {
    "total_sessions_clustered": 8547,
    "n_clusters": 47,
    "n_outliers": 234,
    "clustering_mode": "COMPLETE"
  },
  "human_review_stats": {
    "max_sessions_for_review": 500,
    "actual_sessions_for_review": 423,
    "clusters_reviewed": 38,
    "clusters_excluded": 9,
    "selection_strategy": "prioritize_by_size"
  },
  "training_metrics": { ... },
  "timestamp": "2025-08-20T08:16:04"
}
```

## 🚀 Flusso Operativo Completo

### 1. 📊 Estrazione Completa
```
📥 Database → [Tutte le discussioni] → 8547 sessioni estratte
```

### 2. 🧩 Clustering Completo  
```
🔍 8547 sessioni → Clustering → 47 cluster + 234 outlier
```

### 3. 🎯 Selezione Intelligente
```
📋 47 cluster → Prioritizzazione → 38 cluster selezionati
📝 38 cluster × 3 rappresentanti → 423 sessioni per review
✅ 423 ≤ 500 (limite utente) → Tutti inclusi
```

### 4. 👤 Review Umana
```
🔍 423 sessioni → Review umana → Etichettatura supervisionata
📚 Training modello → Modello addestrato
```

### 5. 🎯 Obiettivo Raggiunto
```
🤖 Modello addestrato → Modalità "Classifica tutti" → Intervento umano minimizzato
```

## 🎨 Mockup Interfaccia UI

```
┌─────────────────────────────────────────────────────────┐
│ 🎓 Training Supervisionato                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 📝 Numero massimo sessioni da revisionare              │
│ ┌─────────────────┐                                     │
│ │      500        │  [Default: 500]                     │
│ └─────────────────┘                                     │
│ 💡 Sessioni rappresentative sottoposte alla tua review │
│                                                         │
│ 🎯 Soglia di confidenza (%)                            │
│ ┌─────────────────┐                                     │
│ │       70        │  [Default: 70%]                     │
│ └─────────────────┘                                     │
│ 💡 Sotto questa soglia serve il tuo intervento         │
│                                                         │
│                                     ┌─────────────────┐ │
│                                     │ 🚀 Avvia       │ │
│                                     │    Training     │ │
│                                     └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## ✅ Vantaggi del Sistema Semplificato

1. **🧠 Intelligenza**: Estrazione e clustering su tutto il dataset
2. **⏱️ Efficienza**: Solo le sessioni più importanti per review umana  
3. **🎯 Semplicità**: UI con solo 2 parametri configurabili
4. **📊 Scalabilità**: Gestione di dataset di qualsiasi dimensione
5. **🤖 Ottimizzazione**: Modello addestrato per ridurre futuro intervento umano
