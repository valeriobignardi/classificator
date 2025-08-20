# Training Supervisionato Avanzato - Documentazione

## 📋 Panoramica

Il sistema di **Training Supervisionato Avanzato** implementa una logica migliorata per la supervisione umana nella classificazione delle discussioni. La principale innovazione consiste nella separazione tra:

1. **Estrazione e Clustering**: Sempre su TUTTO il dataset disponibile
2. **Review Umana**: Solo su un numero limitato di sessioni rappresentative

## 🚀 Nuova Logica Implementata

### Prima (Logica Vecchia):
```
Database (10.000 discussioni) → Limit 500 → Clustering su 500 → Review su 500
```
**Problema**: Il clustering viene fatto solo su un subset dei dati, perdendo informazioni.

### Ora (Logica Nuova):
```
Database (10.000 discussioni) → Estrazione COMPLETA → Clustering su 10.000 → Selezione intelligente → Review su 500 rappresentanti
```
**Vantaggi**: 
- Clustering accurato su tutto il dataset
- Review umana efficiente su campioni rappresentativi
- Migliore qualità dei modelli addestrati

## ⚙️ Configurazione

### File: `config.yaml`

```yaml
# =====================================
# CONFIGURAZIONE TRAINING SUPERVISIONATO
# =====================================
supervised_training:
  # Estrazione completa dal database
  extraction:
    use_full_dataset: true          # SEMPRE estrarre TUTTE le discussioni dal database
    ignore_session_limits: true     # Ignora qualsiasi limite su numero sessioni estratte
    full_clustering_required: true  # Il clustering deve essere fatto su tutto il dataset
  
  # Limiti per revisione umana
  human_review:
    max_total_sessions: 500         # Limite massimo sessioni da sottoporre all'umano
    representatives_per_cluster: 3  # Numero di rappresentanti per cluster (di default)
    min_representatives_per_cluster: 1  # Minimo rappresentanti per cluster piccoli
    max_representatives_per_cluster: 5  # Massimo rappresentanti per cluster grandi
    
    # Strategia di selezione quando ci sono troppi cluster
    selection_strategy: "prioritize_by_size"  # "prioritize_by_size", "prioritize_by_confidence", "balanced"
    
    # Prioritizzazione cluster per review umana
    priority_large_clusters: true   # Priorità ai cluster più grandi
    priority_low_confidence: true   # Priorità ai cluster a bassa confidenza
    confidence_threshold_priority: 0.7  # Soglia sotto cui il cluster ha priorità
    
    # Gestione overflow
    overflow_handling: "proportional"  # "proportional", "truncate", "ask_user"
    min_cluster_size_for_review: 2  # Dimensione minima cluster per essere incluso nella review
  
  # Configurazione output dettagliato
  reporting:
    show_extraction_stats: true     # Mostra statistiche estrazione completa
    show_clustering_stats: true     # Mostra statistiche clustering completo
    show_selection_stats: true      # Mostra statistiche selezione per review umana
    log_excluded_clusters: true     # Logga cluster esclusi dalla review umana
```

## 🌐 API Endpoints

### Nuovo Endpoint: `/train/supervised/advanced/<client_name>`

**Metodo**: `POST`

**Descrizione**: Esegue training supervisionato con estrazione completa del dataset.

#### Request Body (Opzionale):
```json
{
    "max_human_review_sessions": 500,
    "representatives_per_cluster": 3,
    "force_retrain": true
}
```

#### Response Success:
```json
{
    "success": true,
    "message": "Training supervisionato avanzato completato",
    "client": "humanitas", 
    "extraction_stats": {
        "total_sessions_extracted": 10000,
        "extraction_mode": "FULL_DATASET",
        "ignored_original_limit": null
    },
    "clustering_stats": {
        "total_sessions_clustered": 10000,
        "n_clusters": 45,
        "n_outliers": 120,
        "clustering_mode": "COMPLETE"
    },
    "human_review_stats": {
        "max_sessions_for_review": 500,
        "actual_sessions_for_review": 485,
        "clusters_reviewed": 42,
        "clusters_excluded": 3,
        "selection_strategy": "prioritize_by_size"
    },
    "training_metrics": {
        "training_accuracy": 0.89,
        "n_samples": 485,
        "human_feedback_stats": {
            "total_reviews": 42,
            "approved_labels": 35,
            "modified_labels": 7
        }
    }
}
```

### Endpoint Esistente (per confronto): `/train/supervised/<client_name>`

**Logica**: Estrazione limitata + clustering limitato (logica precedente)

## 📊 Strategie di Selezione

### 1. `prioritize_by_size` (Default)
- Priorità ai cluster più grandi
- Garantisce copertura dei pattern più comuni
- Ordina cluster per numero di sessioni (decrescente)

### 2. `prioritize_by_confidence`
- Priorità ai cluster con bassa confidenza
- Migliora la qualità in aree problematiche
- Ordina cluster per confidenza media (crescente)

### 3. `balanced`
- Strategia bilanciata tra dimensione e confidenza
- Alterna cluster grandi e a bassa confidenza

## 🔄 Workflow del Sistema

### Fase 1: Estrazione Completa
```
Database → Estrai TUTTE le discussioni → N discussioni totali
```

### Fase 2: Clustering Completo
```
N discussioni → Clustering HDBSCAN → M cluster + outliers
```

### Fase 3: Selezione Intelligente
```
M cluster → Applica strategia → Selezione fino a max_sessions rappresentanti
```

### Fase 4: Review Umana
```
Rappresentanti selezionati → Review interattiva → Etichette validate
```

### Fase 5: Training Modelli
```
Etichette validate → Training ML + LLM → Modelli aggiornati
```

## 📈 Esempi Pratici

### Scenario 1: Database Piccolo
- **Database**: 1.000 discussioni
- **Cluster**: 15 cluster identificati
- **Rappresentanti**: 3 per cluster = 45 sessioni
- **Limite**: 500 sessioni
- **Risultato**: Review di tutte le 45 sessioni (sotto il limite)

### Scenario 2: Database Grande
- **Database**: 50.000 discussioni 
- **Cluster**: 200 cluster identificati
- **Rappresentanti potenziali**: 3 per cluster = 600 sessioni
- **Limite**: 500 sessioni
- **Selezione**: Priorità ai 167 cluster più grandi (3 reps ciascuno = 501 ≈ 500)
- **Risultato**: Review di 500 sessioni dai cluster principali

### Scenario 3: Molti Cluster Piccoli
- **Database**: 20.000 discussioni
- **Cluster**: 300 cluster (molti piccoli)
- **Filtro**: Solo cluster con ≥ 2 sessioni (200 cluster rimangono)
- **Strategia**: Proporzionale basata su dimensione cluster
- **Risultato**: Review bilanciata rispettando il limite

## 🐛 Debugging e Monitoraggio

### Log di Debug Disponibili

1. **Estrazione**: 
   ```
   📊 MODALITÀ ESTRAZIONE COMPLETA ATTIVATA
   📋 Ignorando limite sessioni - estrazione di TUTTO il dataset
   ✅ ESTRAZIONE COMPLETA: 10000 sessioni totali dal database
   ```

2. **Clustering**:
   ```
   🧩 Clustering intelligente di 10000 sessioni...
   ✅ Clustering completo: 45 cluster, 120 outlier
   ```

3. **Selezione**:
   ```
   🔍 Selezione intelligente rappresentanti per review umana...
   📊 Analisi cluster: 45 totali, 42 eleggibili, 3 troppo piccoli
   ✅ Selezione completata: 485/500 sessioni, 42 cluster per review
   ```

### Metriche di Monitoraggio

- `total_sessions_extracted`: Sessioni totali estratte dal database
- `total_sessions_clustered`: Sessioni processate dal clustering  
- `clusters_reviewed`: Cluster sottoposti a review umana
- `clusters_excluded`: Cluster esclusi dalla review
- `actual_sessions_for_review`: Sessioni effettivamente riviste dall'umano

## 🚀 Come Utilizzare

### 1. Via API (Raccomandato)
```bash
curl -X POST http://localhost:5000/train/supervised/advanced/humanitas \
  -H "Content-Type: application/json" \
  -d '{
    "max_human_review_sessions": 500,
    "representatives_per_cluster": 3
  }'
```

### 2. Via Python Script
```python
from Pipeline.end_to_end_pipeline import EndToEndPipeline

pipeline = EndToEndPipeline(tenant_slug="humanitas")

# Nuovo metodo con estrazione completa
results = pipeline.esegui_training_interattivo(
    max_human_review_sessions=500
)

print(f"Sessioni estratte: {results['extraction_stats']['total_sessions_extracted']}")
print(f"Cluster trovati: {results['clustering_stats']['n_clusters']}")
print(f"Sessioni riviste: {results['human_review_stats']['actual_sessions_for_review']}")
```

### 3. Via UI React (Prossimo Step)
- Bottone "Training Avanzato" nell'interfaccia
- Configurazione parametri via form
- Progress indicator in tempo reale
- Visualizzazione risultati dettagliati

## ⚠️ Note Importanti

1. **Memoria**: L'estrazione completa può richiedere più RAM per dataset molto grandi
2. **Tempo**: Il clustering completo richiede più tempo computazionale
3. **GPU**: Assicurarsi che la GPU abbia memoria sufficiente per gli embeddings
4. **Backup**: Il sistema crea automaticamente backup dei modelli precedenti

## 📋 Migrazione dalla Logica Precedente

1. **Aggiorna configurazione**: Imposta `supervised_training.extraction.use_full_dataset: true`
2. **Testa nuovo endpoint**: Usa `/train/supervised/advanced/<client>` invece di `/train/supervised/<client>`
3. **Verifica risultati**: Confronta metriche di qualità prima/dopo
4. **Graduale adozione**: Mantieni endpoint vecchio per retrocompatibilità

## 🎯 Benefici Attesi

- **Miglior Clustering**: Basato su tutto il dataset disponibile
- **Efficiency Umana**: Review solo su campioni rappresentativi
- **Qualità Modelli**: Training su etichette più accurate e complete
- **Scalabilità**: Gestisce dataset di qualsiasi dimensione
- **Flessibilità**: Configurabile per diverse esigenze operative

---

**Autore**: GitHub Copilot  
**Data**: 20 Agosto 2025  
**Versione**: 1.0  
**Status**: ✅ Pronto per produzione
