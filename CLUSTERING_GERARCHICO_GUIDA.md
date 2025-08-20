# Clustering Gerarchico Adattivo - Guida Completa

## 📋 Panoramica

Il sistema di **Clustering Gerarchico Adattivo** è stato implementato per risolvere automaticamente i conflitti di etichette che possono verificarsi quando due casi simili vengono assegnati a etichette diverse all'interno dello stesso cluster.

### 🎯 Problema Risolto

Nel clustering standard HDBSCAN, quando:
- Due sessioni finiscono nello stesso cluster
- Ma il classificatore LLM assegna loro etichette diverse
- Il sistema originale usava un meccanismo di consenso che poteva perdere informazioni importanti

Il nuovo sistema **preserva l'incertezza** invece di forzare una decisione, utilizzando regioni gerarchiche con membership probabilistiche.

## 🏗️ Architettura del Sistema

### Componenti Principali

1. **HierarchicalAdaptiveClusterer** (`Clustering/hierarchical_adaptive_clusterer.py`)
   - Classe principale per clustering gerarchico
   - Gestisce regioni di confidenza (core/boundary/outlier)
   - Risolve conflitti attraverso split semantici

2. **Integrazione Pipeline** (`Pipeline/end_to_end_pipeline.py`)
   - Nuovi metodi per clustering gerarchico
   - Funzioni di analisi e risoluzione conflitti
   - Compatibilità con sistema esistente

3. **Configurazione** (`config.yaml`)
   - Parametri per clustering gerarchico
   - Soglie di confidenza configurabili
   - Strategie di risoluzione conflitti

## 🚀 Utilizzo del Sistema

### 1. Analisi Conflitti Esistenti

```python
from Pipeline.end_to_end_pipeline import EndToEndPipeline

# Inizializza pipeline
pipeline = EndToEndPipeline(config_path='config.yaml')

# Carica le tue sessioni
sessioni = {...}  # Il tuo dataset

# Analizza conflitti nel clustering attuale
risultati = pipeline.analizza_conflitti_etichette(sessioni, show_details=True)

print(f"Cluster con conflitti: {risultati['statistiche']['cluster_con_conflitti']}")
print(f"Raccomandazione: {risultati['statistiche']['raccomandazione_principale']}")
```

### 2. Clustering Gerarchico Diretto

```python
# Esegui clustering gerarchico con parametri personalizzati
embeddings, session_memberships, cluster_info, structure = pipeline.esegui_clustering_gerarchico_avanzato(
    sessioni,
    confidence_threshold=0.8,    # Soglia alta per regioni pure
    boundary_threshold=0.4,      # Soglia per zone di incertezza  
    max_iterations=5             # Iterazioni per convergenza
)

# Analizza risultati
print(f"Regioni create: {len(structure['regions'])}")
print(f"Sessioni multi-membership: {structure['statistics']['multi_membership_sessions']}")
```

### 3. Risoluzione Automatica Conflitti

```python
# Risoluzione automatica con strategia ottimale
risultati = pipeline.risolvi_conflitti_automaticamente(
    sessioni,
    strategia='auto'  # 'auto', 'gerarchico', 'refinement', 'ibrido'
)

print(f"Conflitti risolti: {risultati['conflitti_risolti']}/{risultati['conflitti_iniziali']}")
print(f"Miglioramento: {risultati['miglioramento_percentuale']:.1f}%")
```

### 4. Configurazione via YAML

```yaml
# config.yaml
hierarchical_clustering:
  enabled: true                    # Abilita clustering gerarchico
  confidence_threshold: 0.75       # Soglia per regioni core
  boundary_threshold: 0.45         # Soglia per regioni boundary
  max_iterations: 3                # Iterazioni adattive
  max_hierarchy_depth: 3           # Profondità massima gerarchia
  conflict_resolution_strategy: 'split'  # Strategia risoluzione conflitti
```

## 📊 Tipi di Regioni

### 1. **Regioni Core** (Alta Confidenza)
- Confidenza > `confidence_threshold` (default: 0.75)
- Clustering semanticamente puro
- Etichette coerenti tra membri

### 2. **Regioni Boundary** (Incertezza)
- Confidenza tra `boundary_threshold` e `confidence_threshold`
- Zone di transizione tra categorie
- Possibile multi-membership

### 3. **Regioni Outlier** (Bassa Confidenza)
- Confidenza < `boundary_threshold` (default: 0.45)
- Casi isolati o anomali
- Richiedono review umana

## 🔧 Strategie di Risoluzione Conflitti

### 1. **Split Semantico**
```python
conflict_resolution_strategy: 'split'
```
- Divide cluster conflittuali in sotto-regioni
- Usa analisi semantica LLM per separazione
- Preserva granularità informazioni

### 2. **Boundary Expansion**
```python
conflict_resolution_strategy: 'boundary'
```
- Espande regioni boundary per catturare incertezza
- Gestisce casi ambigui senza forzare decisioni
- Mantiene soft clustering

### 3. **Voting Pesato**
```python
conflict_resolution_strategy: 'voting'
```
- Combina predizioni multiple con pesi
- Usa confidenza per pesare contributi
- Fallback per casi complessi

## 📈 Metriche e Monitoraggio

### Metriche Chiave

1. **Risoluzione Conflitti**
   - `conflitti_risolti`: Numero conflitti eliminati
   - `miglioramento_percentuale`: % miglioramento
   - `tempo_esecuzione`: Performance temporale

2. **Qualità Clustering**
   - `regioni_create`: Numero regioni gerarchiche
   - `multi_membership_sessions`: Sessioni con membership multiple
   - `profondita_massima`: Profondità gerarchia

3. **Distribuzione Confidenza**
   - `core_regions`: Regioni ad alta confidenza
   - `boundary_regions`: Regioni di incertezza
   - `outlier_sessions`: Sessioni outlier

### Monitoraggio Performance

```python
# Confronto performance standard vs gerarchico
performance = pipeline.test_comparativo_performance(sessioni)

print(f"Overhead gerarchico: {performance['gerarchico']['overhead']:+.1f}%")
print(f"Cluster standard: {performance['standard']['clusters']}")
print(f"Regioni gerarchiche: {performance['gerarchico']['regions']}")
```

## 🧪 Testing e Validazione

### Suite Test Completa

```bash
# Esegui tutti i test del clustering gerarchico
python tests/test_hierarchical_clustering.py
```

La suite include:
1. **Test Baseline**: Analisi conflitti con clustering standard
2. **Test Gerarchico**: Funzionalità clustering gerarchico
3. **Test Automatico**: Risoluzione automatica conflitti
4. **Test Performance**: Confronto prestazioni
5. **Test Configurazione**: Validazione YAML

### Test Personalizzati

```python
# Crea dataset con conflitti controllati
sessioni_test = crea_sessioni_mock_con_conflitti()

# Test analisi conflitti
risultati = pipeline.analizza_conflitti_etichette(sessioni_test)

# Verifica miglioramenti
assert risultati['statistiche']['cluster_con_conflitti'] >= 0
```

## ⚙️ Configurazione Avanzata

### Parametri di Fine-Tuning

```yaml
hierarchical_clustering:
  # Soglie di qualità
  min_region_size: 2              # Dimensione minima regione
  min_membership_confidence: 0.3   # Confidenza minima membership
  
  # Controllo iterazioni
  convergence_threshold: 0.05      # Soglia convergenza
  max_iterations: 5                # Iterazioni massime
  
  # Gestione gerarchia
  split_threshold: 0.6             # Soglia per split regioni
  merge_threshold: 0.85            # Soglia per merge regioni
  
  # Debug e logging
  log_conflict_details: true       # Log dettagliato conflitti
  export_hierarchy_structure: true # Esporta struttura completa
```

### Modalità Operative

1. **Produzione** (Performance ottimizzata)
```yaml
hierarchical_clustering:
  enabled: true
  max_iterations: 3
  confidence_threshold: 0.8
  generate_region_reports: false
```

2. **Sviluppo** (Debug completo)
```yaml
hierarchical_clustering:
  enabled: true
  max_iterations: 5
  log_conflict_details: true
  export_hierarchy_structure: true
  generate_region_reports: true
```

3. **Test** (Validazione estesa)
```yaml
hierarchical_clustering:
  enabled: true
  max_iterations: 10
  convergence_threshold: 0.01
  preserve_outliers: true
```

## 🚨 Troubleshooting

### Problemi Comuni

1. **Performance Lenta**
   - Riduci `max_iterations`
   - Aumenta `convergence_threshold`
   - Disabilita `generate_region_reports`

2. **Troppi Outlier**
   - Riduci `confidence_threshold`
   - Riduci `boundary_threshold`
   - Usa `conflict_resolution_strategy: 'boundary'`

3. **Regioni Troppo Frammentate**
   - Aumenta `min_region_size`
   - Aumenta `merge_threshold`
   - Riduci `max_hierarchy_depth`

### Debug e Logging

```python
# Abilita logging dettagliato
import logging
logging.basicConfig(level=logging.DEBUG)

# Esegui con debug abilitato
pipeline.esegui_clustering_gerarchico_avanzato(
    sessioni,
    confidence_threshold=0.75,
    boundary_threshold=0.45,
    max_iterations=3
)
```

## 🔄 Integrazione con Sistema Esistente

### Compatibilità Backward

Il sistema gerarchico è **completamente compatibile** con il sistema esistente:

```python
# Funziona con entrambi i sistemi
if config['hierarchical_clustering']['enabled']:
    # Usa clustering gerarchico
    results = pipeline.esegui_clustering_gerarchico_avanzato(sessioni)
else:
    # Usa clustering standard
    results = pipeline.esegui_clustering(sessioni)
```

### Migrazione Graduale

1. **Fase 1**: Test su dataset ridotto
2. **Fase 2**: Confronto performance
3. **Fase 3**: Migrazione configurazione
4. **Fase 4**: Deployment produzione

## 📚 Best Practices

### 1. Configurazione Iniziale
- Inizia con parametri conservativi
- Testa su dataset rappresentativo
- Monitora metriche qualità

### 2. Tuning Parametri
- `confidence_threshold`: Start 0.75, adjust based on precision needs
- `boundary_threshold`: Start 0.45, adjust for uncertainty tolerance
- `max_iterations`: Start 3, increase if conflicts persist

### 3. Monitoraggio Produzione
- Track `miglioramento_percentuale` over time
- Monitor `tempo_esecuzione` for performance
- Watch `multi_membership_sessions` for complexity

### 4. Manutenzione
- Review `outlier_sessions` periodically
- Update thresholds based on domain evolution
- Retrain LLM classifier with new conflict patterns

---

**🎯 Obiettivo**: Trasformare conflitti di etichette da problemi in opportunità di maggiore precisione e comprensione semantica del dataset.
