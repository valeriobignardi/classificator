# Fix per Bug Etichette con Caratteri Speciali

**Data**: 2025-09-04  
**Autore**: Valerio Bignardi  
**Problema**: L'LLM restituiva etichette con virgolette e caratteri speciali, causando problemi nel salvataggio e nella visualizzazione

## Problema Identificato

Nel training supervisionato, il sistema utilizzava correttamente Machine Learning e LLM per la classificazione, ma si verificava un bug nel post-processing:

### Sintomi
- Il training e la classificazione funzionavano correttamente
- L'LLM assegnava etichette corrette con alta confidenza
- Il frontend mostrava "N/A" invece delle etichette classificate

### Causa Profonda
Dal log `predict.log` è emerso che l'LLM a volte restituiva etichette con virgolette esterne:

```log
🔍 STRUCTURED RESPONSE DEBUG: {'predicted_label': '"convenzioni_viaggio_strutture_alberghiere"', ...}
✅ Nuovo tag '"convenzioni_viaggio_strutture_alberghiere"' aggiunto per tenant ...
```

L'etichetta veniva salvata nel database **con le virgolette** (`"convenzioni_viaggio_strutture_alberghiere"` invece di `convenzioni_viaggio_strutture_alberghiere`), causando mancate corrispondenze nella ricerca.

## Soluzione Implementata

### 1. Funzione di Pulizia Centralizzata
Aggiunta la funzione `clean_label_text()` in `Classification/intelligent_classifier.py`:

```python
def clean_label_text(label: str) -> str:
    """
    Pulisce l'etichetta da caratteri speciali che possono causare problemi nel salvataggio
    
    - Rimuove virgolette esterne (doppie e singole)
    - Rimuove backslash di escape
    - Rimuove spazi extra
    """
```

### 2. Applicazione della Pulizia

La pulizia è stata applicata in **3 punti strategici**:

1. **Structured Output Processing** (riga ~3100):
   ```python
   raw_predicted_label = structured_result["predicted_label"]
   predicted_label = clean_label_text(raw_predicted_label)  # 🧹 PULIZIA ETICHETTA
   ```

2. **Fallback JSON Parsing** (riga ~2077):
   ```python
   raw_predicted_label = result.get('predicted_label', '').strip()
   predicted_label = clean_label_text(raw_predicted_label).lower()  # 🧹 PULIZIA ETICHETTA
   ```

3. **Fallback Response Processing** (riga ~2150):
   ```python
   raw_predicted_label = mentioned_labels[0]
   predicted_label = clean_label_text(raw_predicted_label)  # 🧹 PULIZIA ETICHETTA
   ```

4. **Sicurezza aggiuntiva nel salvataggio** (riga ~4087):
   ```python
   clean_tag_name = clean_label_text(tag_name)
   if clean_tag_name != tag_name:
       self.logger.info(f"🧹 Etichetta pulita prima del salvataggio: '{tag_name}' → '{clean_tag_name}'")
       tag_name = clean_tag_name
   ```

## File Modificati

1. **`Classification/intelligent_classifier.py`**
   - Aggiunta funzione `clean_label_text()`
   - Applicazione pulizia in 4 punti del flusso di classificazione
   - Logging per debug delle pulizie applicate

## Backup Creato

```
/home/ubuntu/classificatore/backup/intelligent_classifier_20250904_HHMMSS.py
```

## Test di Verifica

Creati due script di test:
1. **`test_label_cleaning.py`** - Test unitari della funzione di pulizia
2. **`test_bug_fix_simulation.py`** - Simulazione del bug e della risoluzione

## Risultato

- ✅ **PRIMA**: `"convenzioni_viaggio_strutture_alberghiere"` (con virgolette) → salvato con virgolette → Frontend: "N/A"
- ✅ **DOPO**: `"convenzioni_viaggio_strutture_alberghiere"` (con virgolette) → pulito → `convenzioni_viaggio_strutture_alberghiere` → salvato correttamente → Frontend: mostra etichetta

## Compatibilità

- ✅ Mantiene piena compatibilità con etichette già esistenti senza virgolette
- ✅ Non modifica la logica di classificazione ML/LLM
- ✅ Non interferisce con il training o l'ensemble
- ✅ Safe: se l'etichetta è già pulita, non viene modificata

## Note Tecniche

La pulizia viene applicata **prima** di:
- Salvataggio nel database
- Confronti semantici
- Validazioni delle etichette
- Aggiornamento della cache in memoria

Questo garantisce che tutti i componenti downstream ricevano etichette normalizzate.
