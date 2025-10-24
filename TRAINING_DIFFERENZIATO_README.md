# Training Differenziato ML - Implementazione

## Panoramica

Implementata la logica di training differenziato per il modello ML nel `QualityGateEngine`:

- **Primo addestramento**: Usa tutte le frasi etichettate (review umane + classificazioni LLM)
- **Riaddestramento**: Usa solo le review umane

## Modifiche Implementate

### 1. Funzione Principale: `_retrain_ml_model()`

Modificata per distinguere automaticamente tra primo addestramento e riaddestramento:

```python
def _retrain_ml_model(self) -> bool:
    # Determina se è il primo addestramento o un riaddestramento
    is_first_training = self._is_first_ml_training()
    
    if is_first_training:
        self.logger.info("🚀 PRIMO ADDESTRAMENTO: Uso review umane + classificazioni LLM")
        training_data = self._load_all_training_data_for_first_training()
    else:
        self.logger.info("🔄 RIADDESTRAMENTO: Uso solo review umane")
        training_data = self._load_human_decisions_for_training()
```

### 2. Nuove Funzioni Aggiunte

#### `_is_first_ml_training()`
Determina se è il primo addestramento verificando:
- Esistenza di modelli ML salvati precedentemente
- File di training log esistenti e dimensione
- Classificazioni ML nel database MongoDB

#### `_load_all_training_data_for_first_training()`
Per il primo addestramento, carica:
- Review umane dal training log
- Classificazioni LLM dal database MongoDB
- Rimuove duplicati (priorità alle review umane)

#### `_load_llm_classifications_from_mongodb()`
Estrae classificazioni LLM dal database MongoDB:
```python
query = {
    'client_name': self.tenant.tenant_slug,
    '$and': [
        {'classification_method': {'$regex': 'llm', '$options': 'i'}},
        {'classification_method': {'$not': {'$regex': 'human', '$options': 'i'}}},
        {'classification_method': {'$not': {'$regex': 'ml', '$options': 'i'}}},
        {'llm_prediction': {'$exists': True, '$ne': None}},
        {'needs_review': {'$ne': True}}
    ]
}
```

#### `_remove_duplicate_training_data()`
Rimuove duplicati per session_id, dando priorità alle review umane.

#### `_check_if_ml_has_ever_classified()`
Verifica se il ML ha mai fatto classificazioni controllando il database.

## Flusso di Decisione

```
Training Richiesto
        ↓
_is_first_ml_training()
        ↓
    Prima volta?
    ├─ Sì → Carica review umane + LLM
    └─ No  → Carica solo review umane
        ↓
    Prepara dati training
        ↓
    Addestra modello ML
```

## Criteri di Identificazione

### Primo Addestramento
- Nessun file di modelli ML esistente
- Nessuna classificazione ML nel database
- Training log assente o molto piccolo

### Riaddestramento  
- File di modelli ML esistenti
- Classificazioni ML precedenti nel database
- Training log sostanzioso (>10 righe)

## Vantaggi

1. **Massima Utilizzazione Dati**: Il primo addestramento usa tutto il materiale disponibile
2. **Qualità Progressiva**: I riaddestamenti usano solo dati validati umani
3. **Automatismo**: La decisione è completamente automatica
4. **Compatibilità**: Funziona sia con `_retrain_ml_model()` che `trigger_manual_retraining()`

## Testing

Creato script di test `test_training_logic.py` che verifica:
- Identificazione corretta primo/riaddestramento
- Formato dati training
- Comportamento con log di diverse dimensioni

La logica è ora implementata e testata con successo! 🎯