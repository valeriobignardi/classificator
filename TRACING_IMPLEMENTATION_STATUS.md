# 🔍 SISTEMA DI TRACING PIPELINE - IMPLEMENTAZIONE COMPLETA

## 📋 **PANORAMICA**

Il sistema di tracing è stato implementato per tracciare completamente il flusso della pipeline `end_to_end_pipeline.py`.

### **📁 File Modificati:**
- `config.yaml` - Aggiunta sezione `tracing`
- `Pipeline/end_to_end_pipeline.py` - Aggiunta funzione `trace_all()` e tracing funzioni
- `test_tracing.py` - Script di test del sistema

---

## ⚙️ **CONFIGURAZIONE**

### **config.yaml - Sezione tracing:**
```yaml
tracing:
  enabled: true
  log_file: tracing.log
  max_file_size_mb: 100
  include_parameters: true
  include_return_values: true
  include_exceptions: true
```

---

## 🔧 **FUNZIONE `trace_all()`**

### **Localizzazione:** `Pipeline/end_to_end_pipeline.py` linee 78-196

### **Funzionalità:**
- ✅ **Controllo configurazione**: Legge `tracing.enabled` da config.yaml
- ✅ **Rotazione file**: Auto-rotazione quando supera `max_file_size_mb`
- ✅ **Timestamp preciso**: Millisecondi per debugging accurato
- ✅ **Gestione parametri**: Serializzazione intelligente di oggetti complessi
- ✅ **Gestione return values**: Tracciamento valori di ritorno
- ✅ **Gestione eccezioni**: Tracciamento errori e stack traces
- ✅ **Fallback silenzioso**: Non interrompe mai la pipeline

### **Signature:**
```python
def trace_all(function_name: str, action: str = "ENTER", **kwargs)
```

### **Parametri:**
- `function_name`: Nome della funzione da tracciare
- `action`: "ENTER", "EXIT", "ERROR"
- `**kwargs`: Parametri da tracciare (input, return_value, exception, etc.)

---

## 📊 **FUNZIONI CON TRACING IMPLEMENTATO**

### ✅ **COMPLETATO (5 funzioni):**

| # | Funzione | ENTER | EXIT | ERROR | Note |
|---|----------|-------|------|-------|------|
| 1 | `trace_all()` | N/A | N/A | N/A | Funzione base |
| 2 | `get_supervised_training_params_from_db` | ✅ | ✅ | ✅ | Completo |
| 3 | `__init__` | ✅ | 🔄 | 🔄 | ENTER implementato |
| 4 | `esegui_training_interattivo` | ✅ | 🔄 | 🔄 | ENTER implementato |
| 5 | `esegui_clustering` | ✅ | 🔄 | 🔄 | ENTER implementato |

### 🔄 **DA COMPLETARE (17 funzioni):**

| # | Funzione | Priorità | Status |
|---|----------|----------|--------|
| 6 | `allena_classificatore` | 🔥 Alta | Da implementare |
| 7 | `classifica_e_salva_sessioni` | 🔥 Alta | Da implementare |
| 8 | `esegui_pipeline_completa` | 🔥 Alta | Da implementare |
| 9 | `_classify_and_save_representatives_post_training` | 🟡 Media | Da implementare |
| 10 | `estrai_sessioni` | 🟡 Media | Da implementare |
| 11-22 | Altre funzioni helper | 🟢 Bassa | Da implementare |

---

## 🧪 **TEST E VALIDAZIONE**

### **Test Eseguito:**
```bash
python3 test_tracing.py
```

### **Risultati Test:**
- ✅ Import funzione `trace_all` riuscito
- ✅ Test ENTER con parametri
- ✅ Test EXIT con return value
- ✅ Test ERROR con eccezione
- ✅ Test funzione reale con database
- ✅ File `tracing.log` creato correttamente

### **Output Tracing.log:**
```log
[2025-09-06 12:19:19.439] ENTER -> test_function (param1=valore1, param2=123)
[2025-09-06 12:19:19.460]  EXIT -> test_function RETURN: {"result": "success", "data": [1, 2, 3]}
[2025-09-06 12:19:19.481] ERROR -> test_function EXCEPTION: ValueError: Test exception
[2025-09-06 12:19:19.502] ENTER -> get_supervised_training_params_from_db (tenant_id=test-tenant-id)
[2025-09-06 12:19:19.601]  EXIT -> get_supervised_training_params_from_db RETURN: {...}
```

---

## 🎯 **FORMATO OUTPUT TRACING**

### **Struttura Log:**
```
[TIMESTAMP] ACTION -> FUNCTION_NAME (DETAILS)
```

### **Esempi:**
```log
# Ingresso funzione con parametri
[2025-09-06 12:19:19.439] ENTER -> esegui_training_interattivo (giorni_indietro=7, limit=100, ...)

# Uscita funzione con valore ritorno
[2025-09-06 12:19:20.123]  EXIT -> esegui_clustering RETURN: tuple(size=4)

# Errore con eccezione
[2025-09-06 12:19:20.456] ERROR -> allena_classificatore EXCEPTION: ConnectionError: Database unreachable
```

---

## 🚀 **VANTAGGI IMPLEMENTAZIONE**

### **1. Debug Completo:**
- Traccia **completa** del flusso di esecuzione
- **Timing preciso** di ogni funzione
- **Parametri di input** e **valori di ritorno**
- **Stack trace** completo degli errori

### **2. Performance Monitoring:**
- **Durata** di ogni funzione calcolabile dai timestamp
- **Identificazione bottleneck** nel flusso
- **Pattern di errore** ricorrenti

### **3. Manutenibilità:**
- **Non invasivo**: Fallback silenzioso se disabilitato
- **Configurabile**: Abilitazione via config.yaml
- **Auto-rotation**: Gestione automatica dimensione file
- **Retrocompatibilità**: Non rompe codice esistente

---

## 🔄 **PROSSIMI PASSI**

### **Immediate (Alta Priorità):**
1. **Completare EXIT/ERROR** nelle 5 funzioni implementate
2. **Aggiungere tracing** alle 3 funzioni critiche rimanenti:
   - `allena_classificatore`
   - `classifica_e_salva_sessioni` 
   - `esegui_pipeline_completa`

### **Successive (Media Priorità):**
3. **Aggiungere tracing** alle 14 funzioni helper rimanenti
4. **Test completo** con training supervisionato reale
5. **Analisi performance** basata su tracing.log

### **Opzionali (Bassa Priorità):**
6. **Dashboard web** per visualizzare tracing in tempo reale
7. **Alerting** per errori critici
8. **Compressione** automatica file log storici

---

## 💡 **UTILIZZO PRATICO**

### **Abilitazione/Disabilitazione:**
```yaml
# config.yaml
tracing:
  enabled: true  # false per disabilitare
```

### **Analisi Logs:**
```bash
# Mostra tutti gli errori
grep "ERROR" tracing.log

# Mostra flusso di una funzione specifica
grep "esegui_training_interattivo" tracing.log

# Mostra solo ENTER per vedere sequenza chiamate
grep "ENTER" tracing.log
```

### **Monitoring Performance:**
```bash
# Calcola durata funzioni (manuale o con script)
grep -A1 "ENTER.*esegui_clustering" tracing.log
```

---

**Autore**: Valerio Bignardi  
**Data**: 2025-09-06  
**Status**: ✅ **IMPLEMENTAZIONE BASE COMPLETATA**  
**Test**: ✅ **VALIDATO E FUNZIONANTE**
