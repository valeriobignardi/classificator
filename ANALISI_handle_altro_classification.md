"""
ANALISI: Quando viene chiamato handle_altro_classification

Autore: Valerio Bignardi
Data: 2025-01-27
"""

## 📋 CONDIZIONI PER L'ATTIVAZIONE DI handle_altro_classification

### 🔗 CATENA DI CHIAMATE:

1. **Pipeline completa**: `esegui_pipeline_completa()`
   ↓
2. **Classificazione**: `classifica_e_salva_sessioni()`
   ↓
3. **Validazione ALTRO**: `handle_altro_classification()`

### ✅ CONDIZIONI NECESSARIE (tutte devono essere vere):

#### 1. **Predicted Label = "altro"**
```python
if predicted_label == 'altro' and hasattr(self, 'interactive_trainer') and self.interactive_trainer.altro_validator:
```

#### 2. **Interactive Trainer disponibile**
- `self.interactive_trainer` deve esistere
- Deve avere l'attributo `altro_validator` non None

#### 3. **AltroTagValidator inizializzato**
- `self.interactive_trainer.altro_validator` deve essere stato creato con successo
- Richiede un oggetto `Tenant` valido

#### 4. **Testo conversazione disponibile**
```python
conversation_text = sessioni[session_id].get('testo_completo', '')
if conversation_text:
```

### 🎯 DUE PUNTI DI ATTIVAZIONE nella pipeline:

#### **PUNTO 1: Classificazione Ensemble**
```python
# Linea ~2548 in classifica_e_salva_sessioni()
if use_ensemble:
    # ... ensemble classification ...
    if predicted_label == 'altro' and hasattr(self, 'interactive_trainer') and self.interactive_trainer.altro_validator:
        validated_label, validated_confidence, validation_info = self.interactive_trainer.handle_altro_classification(
            conversation_text=conversation_text,
            force_human_decision=False  # Automatico durante training
        )
```

#### **PUNTO 2: Classificazione ML_AUTO**
```python
# Linea ~2607 in classifica_e_salva_sessioni()
else:
    # ... ML singolo classification ...
    if predicted_label == 'altro' and hasattr(self, 'interactive_trainer') and self.interactive_trainer.altro_validator:
        validated_label, validated_confidence, validation_info = self.interactive_trainer.handle_altro_classification(
            conversation_text=conversation_text,
            force_human_decision=False  # Automatico durante training
        )
```

### 🔄 FLUSSO COMPLETO:

1. **Pipeline avviata** (esegui_pipeline_completa)
2. **Sessioni estratte** dal database MySQL
3. **Classificazione eseguita** (Ensemble o ML singolo)
4. **Per ogni sessione**:
   - Se `predicted_label == 'altro'` ✅
   - E `interactive_trainer` disponibile ✅
   - E `altro_validator` inizializzato ✅
   - E `conversation_text` non vuoto ✅
   - **ALLORA**: Chiama `handle_altro_classification()`

### 🧪 VALIDAZIONE INTERNA (in handle_altro_classification):

```python
def handle_altro_classification(self, conversation_text: str, force_human_decision: bool = False):
    # VERIFICA PRELIMINARE
    if not self.altro_validator or not self.llm_classifier:
        # Fallback: restituisci "altro" senza validazione
        return "altro", 0.3, {"validation_path": "no_validator", "needs_human_review": True}
    
    # VALIDAZIONE EFFETTIVA
    validation_result = self.altro_validator.validate_altro_classification(
        conversation_text=conversation_text,
        llm_classifier=self.llm_classifier,
        bertopic_model=self.bertopic_model,
        force_human_decision=force_human_decision
    )
```

### 🎯 RISULTATO DELLA VALIDAZIONE:

Se `validated_label != 'altro'`:
- Il tag viene **pulito** con `clean_label_text()`
- La classificazione viene **aggiornata** con il nuovo tag
- Il metodo viene marcato come `{method}_ALTRO_VAL`

### 📊 PARAMETRI CHIAVE:

- **force_human_decision**: Sempre `False` nelle chiamate pipeline (modalità automatica)
- **conversation_text**: Testo completo della conversazione
- **Risultato**: `(validated_label, confidence, validation_info)`

### 🚫 CASI IN CUI NON VIENE CHIAMATO:

1. `predicted_label != 'altro'` (tag normale trovato)
2. `interactive_trainer` non inizializzato
3. `altro_validator` non creato (problemi inizializzazione)
4. `conversation_text` vuoto o mancante
5. LLM classifier non disponibile

### 💡 IMPLICAZIONI:

Il metodo viene chiamato **solo** quando il sistema classifica automaticamente come "altro", permettendo di:

1. **Rivalutare** la classificazione con LLM + BERTopic
2. **Trovare tag esistenti simili** via embedding similarity
3. **Proporre nuovi tag** se necessario
4. **Pulire automaticamente** i tag da caratteri speciali
5. **Ridurre** i falsi positivi di "altro"

Questo è il sistema di **"secondo livello"** per gestire le classificazioni incerte.
