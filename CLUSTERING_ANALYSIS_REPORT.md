# ANALISI CONFIGURAZIONI CLUSTERING - HUMANITAS
## Report Completo - 28 Agosto 2025

---

## 📊 EXECUTIVE SUMMARY

**Tenant:** Humanitas (015007d9-d413-11ef-86a5-96000228e7fe)  
**Dataset:** 3,235 conversazioni  
**Configurazioni testate:** 22 varianti  
**Periodo analisi:** 27-28 Agosto 2025

### 🎯 Risultati Principali

- **Migliore configurazione attuale:** Versione 22 (per copertura) e Versione 16 (per qualità)
- **Quality Score massimo:** 0.0854 (Versione 16)
- **Migliore Silhouette Score:** 0.2610 (Versione 17, ma con 76.6% outliers)
- **Migliore copertura:** Versione 22 con solo 12.9% outliers

---

## 🏆 TOP 3 CONFIGURAZIONI RACCOMANDATE

### 1. 🥇 Versione 22 - **MIGLIORE COMPLESSIVA**
- **Quality Score:** 0.0728
- **Silhouette Score:** 0.0836  
- **Cluster:** 51
- **Outliers:** 417 (12.9%) ✅
- **Tempo:** 42.35s
- **Pro:** Ottima copertura, buon bilanciamento
- **Contro:** Qualità cluster migliorabile

### 2. 🥈 Versione 21 - **SECONDA SCELTA**  
- **Quality Score:** 0.0752
- **Silhouette Score:** 0.0888
- **Cluster:** 58
- **Outliers:** 498 (15.4%) ✅
- **Tempo:** 42.85s
- **Pro:** Buona copertura e granularità
- **Contro:** Qualità cluster migliorabile

### 3. 🥉 Versione 16 - **MASSIMA QUALITÀ**
- **Quality Score:** 0.0854
- **Silhouette Score:** 0.1638 ✅
- **Cluster:** 19  
- **Outliers:** 1549 (47.9%) ❌
- **Tempo:** 18.70s ✅
- **Pro:** Eccellente qualità cluster, veloce
- **Contro:** Troppi outliers

---

## 🔬 CONFIGURAZIONI OTTIMIZZATE PROPOSTE

### 1. CONFIGURAZIONE "BILANCIATA" - **RACCOMANDATA**
```yaml
alpha: 0.45
metric: euclidean
use_umap: true
min_cluster_size: 10
min_samples: 8
cluster_selection_method: eom
cluster_selection_epsilon: 0.18
```
**Obiettivo:** Bilanciare qualità e copertura  
**Risultato atteso:** 40-50 clusters, 25-30% outliers, Silhouette 0.10-0.15

### 2. CONFIGURAZIONE "QUALITÀ_MASSIMA"
```yaml
alpha: 0.6
metric: euclidean
use_umap: false
min_cluster_size: 12
min_samples: 10
cluster_selection_method: eom
cluster_selection_epsilon: 0.12
```
**Obiettivo:** Massimizzare Silhouette Score (>0.20)  
**Risultato atteso:** 25-30 clusters, 35-45% outliers

### 3. CONFIGURAZIONE "COPERTURA_MASSIMA"
```yaml
alpha: 0.3
metric: euclidean
use_umap: true
min_cluster_size: 8
min_samples: 6
cluster_selection_method: leaf
cluster_selection_epsilon: 0.25
```
**Obiettivo:** Minimizzare outliers (<20%)  
**Risultato atteso:** 60-80 clusters, 15-25% outliers

---

## 📈 INSIGHTS E PATTERN IDENTIFICATI

### Parametri Chiave
- **`use_umap: false`** → Migliore Silhouette Score
- **`use_umap: true`** → Migliore copertura (meno outliers)
- **`metric: euclidean`** → Funziona meglio per questo dataset
- **`cluster_selection_method: eom`** → Migliore per qualità cluster
- **`cluster_selection_method: leaf`** → Migliore per copertura
- **`alpha: 0.4-0.6`** → Range ottimale per il bilanciamento

### Correlazioni Osservate
- **Silhouette Score alto** ↔ **Più outliers**
- **Meno outliers** ↔ **Più cluster piccoli**
- **UMAP abilitato** ↔ **Migliore copertura**
- **UMAP disabilitato** ↔ **Migliore qualità cluster**

---

## 🎯 RACCOMANDAZIONI OPERATIVE

### Raccomandazione Principale
**Utilizzare la CONFIGURAZIONE BILANCIATA** come punto di partenza per la produzione.

### Piano di Implementazione
1. **Fase 1:** Testare configurazione BILANCIATA
2. **Fase 2:** Se Silhouette < 0.12 → testare QUALITÀ_MASSIMA  
3. **Fase 3:** Se Outliers > 30% → testare COPERTURA_MASSIMA
4. **Fase 4:** Fine-tuning incrementale sui parametri migliori

### KPI di Monitoraggio
- **Silhouette Score:** Target > 0.15 (ottimo > 0.20)
- **Outlier Ratio:** Target < 25% (ottimo < 15%)
- **Numero Cluster:** Range ottimale 30-60
- **Tempo Esecuzione:** Target < 3 minuti

---

## 🔧 STRUMENTI FORNITI

### File Generati
1. **`humanitas_clustering_analysis.json`** - Analisi completa in formato JSON
2. **`apply_optimal_clustering.py`** - Script interattivo per applicare configurazioni
3. **`optimal_clustering_recommendations.py`** - Script di analisi dettagliata
4. **`analyze_clustering_configs.py`** - Script di analisi configurazioni esistenti

### Utilizzo
```bash
# Analisi configurazioni esistenti
python analyze_clustering_configs.py

# Raccomandazioni ottimizzate  
python optimal_clustering_recommendations.py

# Applicazione configurazione ottimale (interattivo)
python apply_optimal_clustering.py
```

---

## 📊 METRICHE COMPARATIVE

| Versione | Quality Score | Silhouette | Outliers | Cluster | Tempo | Note |
|----------|---------------|------------|----------|---------|-------|------|
| **22** ⭐ | 0.0728 | 0.0836 | **12.9%** | 51 | 42s | **Migliore complessiva** |
| **21** | 0.0752 | 0.0888 | 15.4% | 58 | 43s | Buona alternativa |
| **16** | **0.0854** | **0.1638** | 47.9% | 19 | **19s** | Massima qualità |
| 1 | 0.0720 | 0.1263 | 43.0% | 31 | 20s | Primo tentativo |
| 15 | 0.0644 | 0.0758 | 15.1% | 83 | 42s | Troppi cluster |

---

## 💡 CONCLUSIONI E NEXT STEPS

### Stato Attuale
✅ **La configurazione Versione 22 è già OTTIMA** per l'uso in produzione  
✅ Ha il miglior bilanciamento tra qualità e copertura  
✅ Solo 12.9% di outliers è un risultato eccellente  

### Opportunità di Miglioramento
🔬 Testare la configurazione BILANCIATA proposta per possibili miglioramenti  
📈 Monitorare performance con nuovi dati  
🎯 Fine-tuning graduale dei parametri

### Raccomandazione Finale
**Implementare la Versione 22 in produzione** come configurazione principale, con la configurazione BILANCIATA come backup per futuri test.

---

*Report generato il 28 Agosto 2025 - Valerio Bignardi*
