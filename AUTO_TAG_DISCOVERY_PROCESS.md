# Processo di Auto-Scoperta e Creazione Tag

Questo documento spiega in dettaglio come il sistema scopre e crea automaticamente nuovi tag nel database `TAG` durante il processo di classificazione.

## 🎯 Obiettivo

L'obiettivo è arricchire dinamicamente il set di tag disponibili basandosi sulle classificazioni ad alta confidenza del modello LLM, senza richiedere un intervento manuale per ogni nuovo concetto emerso.

## 🔄 Flusso del Processo (Passo-Passo)

Il processo si attiva durante la classificazione di una conversazione. Ecco la sequenza degli eventi:

### Fase 1: Avvio della Classificazione

Tutto inizia quando una sessione viene inviata per la classificazione.

1.  **Trigger API**: Un client chiama l'endpoint del server, ad esempio `POST /classify/all/<client_name>`.
2.  **Servizio di Classificazione**: Il `server.py` avvia la pipeline di classificazione per il tenant specificato.
3.  **Chiamata all'LLM**: Il metodo `classify_with_motivation` in `Classification/intelligent_classifier.py` viene invocato. Questo costruisce un prompt e interroga il modello di linguaggio (LLM) per ottenere un'etichetta (`predicted_label`), una confidenza (`confidence`) e una motivazione.

### Fase 2: Il Momento della Scoperta

Questa è la fase cruciale dove avviene la "magia". Si svolge all'interno del metodo `_semantic_label_resolution` in `Classification/intelligent_classifier.py`.

1.  **Analisi Risposta LLM**: Il sistema riceve la risposta, ad esempio:
    *   `predicted_label`: "richiesta-fattura-dettagliata"
    *   `confidence`: 0.92

2.  **Controllo Esistenza Tag**: Il sistema verifica se l'etichetta `"richiesta-fattura-dettagliata"` esiste già nella tabella `tags` del database per il tenant corrente.
    ```sql
    SELECT id FROM tags WHERE tag_name = 'richiesta-fattura-dettagliata' AND tenant_id = '...'
    ```

3.  **Decisione di Auto-Creazione**: Se tutte le seguenti condizioni sono vere, il sistema decide di creare un nuovo tag:
    *   **Il tag NON esiste** nel database per quel tenant.
    *   La **confidenza dell'LLM è molto alta** (es. `0.92 >= 0.85`). La soglia `0.85` è definita in `config.yaml`.
    *   L'**auto-creazione è abilitata** nel file `config.yaml` (`auto_tag_creation: true`).

Il codice che governa questa logica è:
```python
# In Classification/intelligent_classifier.py

# Se il tag non esiste e l'LLM è molto sicuro...
if initial_confidence >= self.llm_confidence_threshold:
    self.logger.info(f"🎯 LLM confidence alta -> Tentativo auto-creazione tag '{original_label}'")
    
    if self.auto_tag_creation:
        # ...prova a creare il nuovo tag automaticamente
        success = self.add_new_label_to_database(original_label, tenant_id) # Passa anche il tenant_id
        if success:
            self.logger.info(f"✅ Nuovo tag creato automaticamente: '{original_label}'")
            # Ricarica le etichette per usarle subito
            self._reload_domain_labels() 
```

### Fase 3: Salvataggio nel Database

Una volta decisa la creazione, il metodo `add_new_label_to_database` viene eseguito.

1.  **Connessione al DB**: Si connette al database locale `TAG`.
2.  **Inserimento Sicuro**: Esegue una query `INSERT` per aggiungere il nuovo tag, **associandolo al tenant corretto** per garantire l'isolamento dei dati.

```sql
-- Query eseguita per salvare il nuovo tag
INSERT INTO tags (
    tag_name, 
    tag_description, 
    tenant_id, 
    tenant_name
) VALUES (
    'richiesta-fattura-dettagliata', 
    'Etichetta generata automaticamente...', 
    'a0fd7600-f4f7-11ef-9315-96000228e7fe', -- Esempio tenant_id
    'Alleanza'                                -- Esempio tenant_name
);
```

## ⚙️ Configurazione

Questo comportamento è controllato da due parametri nel file `config.yaml`:

```yaml
# in config.yaml

# Soglia di confidenza minima dell'LLM per considerare la creazione di un nuovo tag.
# Valori più alti = più cautela. Valori più bassi = più aggressivo.
llm_confidence_threshold: 0.85

# Flag per abilitare (true) o disabilitare (false) la funzione di auto-creazione.
# Utile per bloccare la creazione di nuovi tag in ambienti di produzione stabili.
auto_tag_creation: true
```

## 📄 Riepilogo

In sintesi, un nuovo tag viene creato e salvato **automaticamente durante la classificazione** solo se:
- **L'LLM è molto sicuro** (confidenza > 85%).
- **Il tag non esiste già** per quel tenant.
- La funzione è **abilitata** in `config.yaml`.

Questo meccanismo permette al sistema di apprendere e adattarsi a nuovi tipi di conversazione in modo autonomo e sicuro, mantenendo i dati di ogni tenant separati.
