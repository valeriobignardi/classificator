# ANALISI COMPLETA: Storage e Identificazione Cluster Metadata in MongoDB

**Autore:** Valerio Bignardi  
**Data creazione:** 2025-01-23  
**Ultimo aggiornamento:** 2025-01-23  

## PANORAMICA SISTEMA

Il sistema di classificazione utilizza una logica complessa per identificare e memorizzare 3 tipi di conversazioni nel MongoDB:
1. **RAPPRESENTANTI** (Representatives): Conversazioni scelte come rappresentative di un cluster
2. **PROPAGATI** (Propagated): Conversazioni classificate tramite propagazione da un rappresentante 
3. **OUTLIERS**: Conversazioni che non appartengono a nessun cluster ben definito

## STATISTICHE ATTUALI MONGODB (Tenant Humanitas)

```
Totale documenti: 4,269
- Rappresentanti: 0
- Propagati: 3,269 (76.6%)
- Outliers (string "outlier_X"): 1,000 (23.4%)
- Outliers (numeric -1): 384
```

## FLUSSO DATI: Pipeline → MongoDB

### 1. COSTRUZIONE CLUSTER_METADATA (end_to_end_pipeline.py)

La pipeline costruisce `cluster_metadata` analizzando il campo `method` della classificazione:

```python
# Linee 1590-1620 di end_to_end_pipeline.py
if 'REPRESENTATIVE' in method:
    cluster_metadata = {
        'cluster_id': cluster_id,
        'is_representative': True,
        'cluster_size': None,
        'confidence': confidence,
        'method': method
    }
elif method == 'CLUSTER_PROPAGATED':
    cluster_metadata = {
        'cluster_id': cluster_id,
        'is_representative': False,
        'propagated_from': prediction.get('source_representative'),
        'propagation_confidence': confidence,
        'method': method
    }
elif 'OUTLIER' in method:
    cluster_metadata = {
        'cluster_id': -1,
        'is_representative': False,
        'outlier_score': 1.0 - confidence,
        'method': method
    }
```

### 2. SALVATAGGIO IN MONGODB (mongo_classification_reader.py)

La funzione `save_classification_result` gestisce 3 scenari:

#### A) CON CLUSTER_METADATA (Rappresentanti/Propagati)
```python
# Linee 1440-1475
if cluster_metadata:
    doc["metadata"]["cluster_id"] = cluster_metadata["cluster_id"]
    if "is_representative" in cluster_metadata:
        doc["metadata"]["is_representative"] = bool(cluster_metadata["is_representative"])
    if "propagated_from" in cluster_metadata:
        doc["metadata"]["propagated_from"] = str(cluster_metadata["propagated_from"])
    if "propagation_confidence" in cluster_metadata:
        doc["metadata"]["propagation_confidence"] = float(cluster_metadata["propagation_confidence"])
```

#### B) SENZA CLUSTER_METADATA (Auto-Outliers)
```python
# Linee 1476-1495
else:
    # Genera cluster_id progressivo per outlier
    outlier_counter = self._get_next_outlier_counter()
    outlier_cluster_id = f"outlier_{outlier_counter}"
    
    doc["metadata"]["cluster_id"] = outlier_cluster_id
    doc["metadata"]["is_representative"] = False
    doc["metadata"]["outlier_score"] = 1.0
    doc["metadata"]["method"] = "auto_outlier_assignment"
```

## SCHEMA MONGODB RISULTANTE

### Struttura Documento Tipo:
```json
{
    "_id": ObjectId,
    "session_id": "uuid-string",
    "classification": "etichetta_finale",
    "confidence": 0.85,
    "metadata": {
        "cluster_id": "numero_o_outlier_X",
        "is_representative": true/false,
        "propagated_from": "cluster_propagation",
        "propagation_confidence": 0.85,
        "outlier_score": 1.0,
        "method": "REPRESENTATIVE|CLUSTER_PROPAGATED|auto_outlier_assignment"
    }
}
```

### Esempi Reali dal Database:

#### PROPAGATI (3,269 casi):
```json
{
    "session_id": "0008ef4f-c2f4-4f5e-933e-646574f3638c",
    "classification": "info_fattura",
    "metadata": {
        "cluster_id": 35,
        "is_representative": false,
        "propagated_from": "cluster_propagation",
        "propagation_confidence": 0.85
    }
}
```

#### OUTLIERS STRING (1,000 casi):
```json
{
    "session_id": "3baa0f4249241bcad1f6af9b8856f5a2",
    "classification": "prenotazione_oncologica", 
    "metadata": {
        "cluster_id": "outlier_1",
        "is_representative": false,
        "outlier_score": 1.0,
        "method": "auto_outlier_assignment"
    }
}
```

#### OUTLIERS NUMERICI (384 casi):
```json
{
    "session_id": "00a733b6-1cd0-4030-a0d3-000c8419331c",
    "classification": "altro",
    "metadata": {
        "cluster_id": -1,
        "is_representative": false,
        "propagated_from": "cluster_propagation",
        "propagation_confidence": 0.3
    }
}
```

## LOGICA DI IDENTIFICAZIONE A RUNTIME

### get_review_queue_sessions (Linee 1570-1720)

La funzione identifica il tipo di sessione utilizzando i metadata:

```python
# Determina il tipo di sessione per metadata UI
session_type = "unknown"
metadata = doc.get('metadata', {})

if metadata.get('is_representative', False):
    session_type = "representative"
elif metadata.get('propagated_from'):
    session_type = "propagated"
elif metadata.get('cluster_id') in [-1, "-1"] or not metadata.get('cluster_id'):
    session_type = "outlier"
```

### Query MongoDB per Review Queue:

```python
# 1. RAPPRESENTANTI
{
    "review_status": "pending",
    "$or": [
        {"metadata.is_representative": True},
        {"metadata.is_representative": {"$exists": False}}
    ]
}

# 2. PROPAGATE  
{
    "metadata.propagated_from": {"$exists": True, "$ne": None},
    "metadata.is_representative": {"$ne": True}
}

# 3. OUTLIERS
{
    "$or": [
        {"metadata.cluster_id": {"$regex": "^outlier_"}},
        {"metadata.cluster_id": -1},
        {"metadata.cluster_id": "-1"},
        {"metadata.cluster_id": {"$exists": False}}
    ]
}
```

## GESTIONE CONTATORI OUTLIER

La funzione `_get_next_outlier_counter()` (linee 1529-1569) genera ID progressivi:

```python
# Pipeline per trovare il numero massimo esistente
pipeline = [
    {"$match": {"metadata.cluster_id": {"$regex": "^outlier_"}}},
    {"$project": {"cluster_id": "$metadata.cluster_id"}},
    {"$group": {"_id": "$cluster_id"}}
]

# Estrae numeri ed calcola il prossimo
max_counter = 0
for outlier_doc in existing_outliers:
    cluster_id = outlier_doc.get('_id', '')
    if cluster_id.startswith('outlier_'):
        counter = int(cluster_id.replace('outlier_', ''))
        max_counter = max(max_counter, counter)

return max_counter + 1
```

## DISTRIBUZIONE CLUSTER_ID ATTUALI

Top 10 cluster per numero di documenti:
```
outlier_1: 1,000 documenti
-1: 384 documenti  
0: 255 documenti
1: 188 documenti
19: 176 documenti
15: 149 documenti
4: 146 documenti
23: 124 documenti
16: 105 documenti
25: 91 documenti
```

## CONSIDERAZIONI TECNICHE

### 1. MANCANZA DI RAPPRESENTANTI
Attualmente non ci sono rappresentanti salvati (0 documenti con `is_representative: true`). Questo indica che:
- Il sistema non ha ancora eseguito un training con selezione di rappresentanti
- Tutte le classificazioni sono propagate o outliers

### 2. DUE TIPI DI OUTLIERS
- **String**: `outlier_1`, `outlier_2`, etc. (1,000 casi)
- **Numerici**: `cluster_id: -1` (384 casi)

Questo suggerisce due diverse logiche di assegnazione outlier in base al contesto.

### 3. PROPAGATED CON CLUSTER_ID = -1
Esistono 384 casi con `cluster_id: -1` ma anche `propagated_from: "cluster_propagation"`. Questi potrebbero essere:
- Propagazioni fallite (confidence bassa)
- Casi limite assegnati come outliers durante la propagazione

### 4. ROBUSTEZZA QUERY
Le query di retrieval usano pattern flessibili per gestire:
- Valori legacy (cluster_id mancanti)
- Diversi formati (string vs numeric)
- Fallback per compatibilità backward

## CONCLUSIONI

Il sistema implementa una logica sofisticata di classificazione cluster-based con storage strutturato in MongoDB. La pipeline è progettata per essere fault-tolerant e gestire diversi scenari di classificazione, dall'assegnazione automatica di outliers alla propagazione da rappresentanti di cluster.

I metadata sono completamente tracciabili e permettono sia l'identificazione runtime che il debugging offline delle decisioni di classificazione.
