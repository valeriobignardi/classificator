# Analisi Parametri Clustering: React vs Database

## 📊 PARAMETRI DEFINITI IN REACT (ClusteringParametersManager.tsx)

### 🔧 PARAMETRI HDBSCAN BASE
1. **min_cluster_size** - Dimensione minima cluster
2. **min_samples** - Numero minimo campioni  
3. **cluster_selection_epsilon** - Soglia selezione cluster
4. **metric** - Metrica di distanza

### 🆕 PARAMETRI AVANZATI HDBSCAN  
5. **cluster_selection_method** - Metodo selezione cluster
6. **alpha** - Parametro alpha per controllo noise
7. **max_cluster_size** - Dimensione massima cluster
8. **allow_single_cluster** - Permetti cluster singolo

### 🎯 PARAMETRI PREPROCESSING
9. **only_user** - Filtra solo messaggi utente

### 🗂️ PARAMETRI UMAP
10. **use_umap** - Abilita/disabilita UMAP
11. **umap_n_neighbors** - Numero di vicini UMAP
12. **umap_min_dist** - Distanza minima UMAP  
13. **umap_metric** - Metrica distanza UMAP
14. **umap_n_components** - Dimensioni output UMAP
15. **umap_random_state** - Seed random UMAP

### 🎯 PARAMETRI REVIEW QUEUE - SOGLIE CONFIDENZA
16. **outlier_confidence_threshold** - Soglia confidenza OUTLIER
17. **propagated_confidence_threshold** - Soglia confidenza PROPAGATO
18. **representative_confidence_threshold** - Soglia confidenza RAPPRESENTATIVO

### 🎯 PARAMETRI REVIEW QUEUE - CONFIGURAZIONE
19. **minimum_consensus_threshold** - Soglia consenso minimo
20. **enable_smart_review** - Abilita review intelligente
21. **max_pending_per_batch** - Massimo casi pending per batch

---

## 🗄️ CAMPI PRESENTI NELLA TABELLA DB `soglie` (TAG)

### ✅ CAMPI CORRISPONDENTI DIRETTI (React → DB)
1. **min_cluster_size** → `min_cluster_size` ✅
2. **min_samples** → `min_samples` ✅
3. **cluster_selection_epsilon** → `cluster_selection_epsilon` ✅
4. **metric** → `metric` ✅
5. **cluster_selection_method** → `cluster_selection_method` ✅  
6. **alpha** → `alpha` ✅
7. **max_cluster_size** → `max_cluster_size` ✅
8. **allow_single_cluster** → `allow_single_cluster` ✅
9. **only_user** → `only_user` ✅
10. **use_umap** → `use_umap` ✅
11. **umap_n_neighbors** → `umap_n_neighbors` ✅
12. **umap_min_dist** → `umap_min_dist` ✅
13. **umap_metric** → `umap_metric` ✅
14. **umap_n_components** → `umap_n_components` ✅
15. **umap_random_state** → `umap_random_state` ✅
16. **outlier_confidence_threshold** → `outlier_confidence_threshold` ✅
17. **propagated_confidence_threshold** → `propagated_confidence_threshold` ✅
18. **representative_confidence_threshold** → `representative_confidence_threshold` ✅
19. **minimum_consensus_threshold** → `minimum_consensus_threshold` ✅
20. **enable_smart_review** → `enable_smart_review` ✅
21. **max_pending_per_batch** → `max_pending_per_batch` ✅

---

## 🆕 CAMPI AGGIUNTIVI PRESENTI SOLO NEL DATABASE

### 📊 CAMPI DI SISTEMA/METADATA
- **id** (PK auto_increment) - ID univoco record
- **tenant_id** - ID del tenant
- **config_source** - Fonte configurazione ('custom', 'default')
- **last_updated** - Timestamp ultimo aggiornamento  
- **created_at** - Timestamp creazione record

### 📋 CAMPI SUPERVISED TRAINING (non in React)
- **confidence_threshold_priority** (decimal 3,2, default 0.70) - Soglia priorità confidenza
- **max_representatives_per_cluster** (int, default 5) - Max rappresentanti per cluster
- **max_total_sessions** (int, default 500) - Max sessioni totali
- **min_representatives_per_cluster** (int, default 1) - Min rappresentanti per cluster  
- **overflow_handling** (varchar 50, default 'proportional') - Gestione overflow
- **representatives_per_cluster** (int, default 3) - Rappresentanti per cluster
- **selection_strategy** (varchar 50, default 'prioritize_by_size') - Strategia selezione

---

## 🔍 IDENTIFICAZIONE DOPPIONI

### ⚠️ POSSIBILI DOPPIONI IDENTIFICATI

1. **Rappresentanti per cluster - TRIPLO DOPPIONE:**
   - `max_representatives_per_cluster` (DB) 
   - `min_representatives_per_cluster` (DB)
   - `representatives_per_cluster` (DB)
   - **Problema:** Tre campi che gestiscono lo stesso concetto con logiche potenzialmente conflittuali

2. **Confidenza/Soglie - DOPPIONE POTENZIALE:**
   - `confidence_threshold_priority` (DB solo)
   - `outlier_confidence_threshold` (React + DB)
   - `propagated_confidence_threshold` (React + DB) 
   - `representative_confidence_threshold` (React + DB)
   - **Problema:** Quattro soglie di confidenza diverse potrebbero sovrapporsi

---

## ❌ PARAMETRI REACT NON SALVATI NEL DATABASE

**Risultato: TUTTI i parametri React hanno corrispondenza diretta nel database!**

Tutti i 21 parametri definiti nell'interfaccia React `ClusteringParametersManager.tsx` sono presenti come campi nella tabella `soglie` del database TAG con nomi identici.

---

## 🔧 PARAMETRI DB NON GESTITI DA REACT

### 📊 CAMPI SUPERVISED TRAINING MANCANTI IN REACT:
1. ~~**confidence_threshold_priority** - Soglia priorità confidenza~~ **RIMOSSO - Era duplicato non utilizzato**
2. **max_representatives_per_cluster** - Max rappresentanti per cluster
3. **max_total_sessions** - Max sessioni totali  
4. **min_representatives_per_cluster** - Min rappresentanti per cluster
5. **overflow_handling** - Gestione overflow sessioni
6. **representatives_per_cluster** - Rappresentanti per cluster
7. **selection_strategy** - Strategia selezione rappresentanti

### 🚨 IMPATTO: 
- Questi 6 parametri esistono nel database ma NON sono modificabili dall'interfaccia React
- Sono relativi alla fase di Supervised Training e selezione dei rappresentanti
- Attualmente vengono utilizzati con valori di default fissi

---

## 📋 RACCOMANDAZIONI

### ✅ AZIONI IMMEDIATE:
1. **Risolvere doppioni rappresentanti:** Unificare `max_representatives_per_cluster`, `min_representatives_per_cluster`, `representatives_per_cluster` in una logica coerente
2. **Aggiungere sezione Supervised Training in React** per gestire i 6 parametri rimanenti

### 🔧 AZIONI FUTURE:
1. **Audit completo logica rappresentanti** per eliminare conflitti
2. **Implementare validazione incrociata** per parametri correlati

### ✅ AZIONI COMPLETATE:
- ~~**confidence_threshold_priority rimosso** (27/01/2025) - Confermato come duplicato non utilizzato in logica condizionale~~