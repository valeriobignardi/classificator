# 🔧 CORREZIONE BUG: Ensemble ML+LLM per Outlier in Intelligent Clustering

**Data**: 2025-09-04  
**Autore**: Valerio Bignardi  
**Problema**: Gli outlier individuali (cluster_id = -1) in modalità `intelligent_clustering` usavano solo LLM invece dell'ensemble ML+LLM

## 📋 **PROBLEMA IDENTIFICATO**

### Sintomi:
- Nel `predict.log` si vedeva solo `LLM DEBUG` per gli outlier
- Non compariva mai `ML Ensemble Debugger` o `predict_with_ensemble` per outlier
- Il metodo `intelligent_clustering` bypassava completamente l'ensemble ML

### Causa Profonda:
Il `IntelligentIntentClusterer` riceveva solo `llm_classifier` invece dell'`ensemble_classifier` completo, causando:
1. **Outlier classificati solo con LLM** invece di ML+LLM ensemble
2. **Perdita delle capacità ML** per outlier individuali
3. **Inconsistenza**: rappresentanti dei cluster usavano ensemble, outlier solo LLM

## 🔧 **SOLUZIONE IMPLEMENTATA**

### Modifiche apportate:

#### 1. **IntelligentIntentClusterer** (`Clustering/intelligent_intent_clusterer.py`)

**Aggiunto parametro `ensemble_classifier`:**
```python
def __init__(self, tenant=None, config_path=None, llm_classifier=None, ensemble_classifier=None):
    # 🔧 CORREZIONE: Priorità all'ensemble_classifier se disponibile
    self.ensemble_classifier = ensemble_classifier
    self.llm_classifier = llm_classifier
    
    # Determina quale classificatore usare
    if self.ensemble_classifier is not None:
        self.use_ensemble = True
        print(f"✅ [INTELLIGENT_CLUSTERER] Usando ensemble_classifier (ML+LLM)")
    elif self.llm_classifier is not None:
        self.use_ensemble = False
        print(f"⚠️ [INTELLIGENT_CLUSTERER] Usando solo llm_classifier (fallback)")
```

**Modificato metodo di analisi per usare ensemble:**
```python
if self.use_ensemble and self.ensemble_classifier:
    # Usa ensemble ML+LLM per classificazione completa
    ensemble_result = self.ensemble_classifier.predict_with_ensemble(
        text,
        return_details=True,
        embedder=getattr(self.ensemble_classifier, 'embedder', None)
    )
    
    intent_info = {
        'intent': ensemble_result['predicted_label'],
        'confidence': ensemble_result['confidence'],
        'reasoning': f"Ensemble {ensemble_result['method']} - conf: {ensemble_result['confidence']:.3f}",
        'method': f"ensemble_{ensemble_result['method'].lower()}",
        'ensemble_details': ensemble_result  # 🆕 Dettagli completi ensemble
    }
```

#### 2. **Pipeline** (`Pipeline/end_to_end_pipeline.py`)

**Modificato passaggio parametri:**
```python
intelligent_clusterer = IntelligentIntentClusterer(
    tenant=self.tenant,
    config_path=self.clusterer.config_path,
    llm_classifier=self.ensemble_classifier.llm_classifier if self.ensemble_classifier else None,
    ensemble_classifier=self.ensemble_classifier  # 🆕 Passa ensemble completo
)
```

## ✅ **RISULTATI ATTESI**

### Prima della correzione:
```log
🤖 LLM DEBUG - CLASSIFICATION | outlier
📋 classified_by: 'intelligent_clustering'
📋 method: intelligent_clustering
```

### Dopo la correzione:
```log
🧪 DEBUG ML ENSEMBLE: | outlier 
📊 ML Ensemble addestrato - classi disponibili: [...]
📊 ML Predizione: 'prenotazione_esami' (conf: 0.85)
✅ Entrambi disponibili - uso ENSEMBLE voting
🎯 Final label: prenotazione_esami (method: ENSEMBLE)
```

## 🧪 **TESTING**

**Test eseguito con successo:**
- ✅ Backward compatibility: funziona con solo `llm_classifier`
- ✅ Nuova funzionalità: funziona con `ensemble_classifier`
- ✅ Priorità corretta: ensemble > llm > fallback
- ✅ Import e inizializzazione senza errori

## 📊 **IMPATTO**

### Vantaggi:
1. **Outlier ora usano ML+LLM ensemble** invece di solo LLM
2. **Maggiore accuratezza** per outlier individuali
3. **Consistenza**: tutti i tipi di sessioni usano stesso classificatore
4. **Mantiene compatibilità** con codice esistente

### Rischi mitigati:
- ✅ Mantiene fallback a solo LLM se ensemble non disponibile
- ✅ Nessuna breaking change per codice esistente
- ✅ Gestione errori robusta

## 🔄 **PROSSIMI PASSI**

1. **Testare in produzione** con sessioni reali
2. **Monitorare performance** degli outlier ML vs LLM-only
3. **Analizzare accuracy** su outlier complex vs semplici
4. **Considerare ottimizzazioni** basate sui risultati

---

**Questa correzione risolve definitivamente il bug che impediva agli outlier di utilizzare le capacità ML dell'ensemble, garantendo ora una classificazione consistente e più accurata per tutti i tipi di sessioni.**
