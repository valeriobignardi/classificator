# Analisi Dettagliata del Sistema di Scoperta Nuovi Tag

Questo documento analizza l'attuale implementazione del sistema di scoperta automatica di nuovi tag, valuta la sua efficacia e propone un'architettura alternativa più robusta e metodologicamente corretta.

---

### 1. Come Funziona Attualmente: Un Fallback in Tempo Reale

L'attuale strategia di scoperta di nuovi tag è implementata come un **meccanismo di fallback in tempo reale** all'interno della classe `IntelligentClassifier`. Non è un processo di analisi offline, ma un tentativo di "salvare" una classificazione incerta.

Ecco il flusso esatto, con esempi basati sul codice:

1.  **Classificazione Iniziale LLM:**
    - Un testo viene inviato al modello LLM per la classificazione (metodo `classify_with_motivation`).
    - L'LLM restituisce un'etichetta e una confidenza. Esempio: `{"predicted_label": "info_medicinali", "confidence": 0.9}`.

2.  **Controllo della Soglia di Fallback:**
    - Il sistema controlla se la confidenza dell'LLM è inferiore a una soglia definita in `config.yaml` (`bertopic_fallback_threshold`, attualmente `0.70`).
    - **Scenario A (Confidenza Alta):** Se la confidenza è `>= 0.70` (es. 0.9), il sistema **NON attiva il fallback BERTopic**. Procede con la validazione semantica contro i tag esistenti. Se non trova un match, il risultato finale è `altro`. **Questo è il problema principale che hai riscontrato.**
    - **Scenario B (Confidenza Bassa):** Se la confidenza è `< 0.70`, il sistema attiva il fallback. Esempio: l'LLM restituisce `{"predicted_label": "gestione_rifiuti", "confidence": 0.6}`.

3.  **Attivazione del Fallback BERTopic (Il Cuore del Problema):**
    - Il metodo `_evaluate_new_category_with_bertopic` viene invocato.
    - **ERRORE CONCETTUALE:** Questo metodo prende il **singolo testo** della conversazione e chiama `bertopic_provider.model.transform([conversation_text])`.

---

### 2. Valutazione Critica: Perché l'Approccio Attuale è Sbagliato

L'uso di BERTopic in questo modo è **metodologicamente errato** e destinato a fallire.

**BERTopic non scopre topic da un singolo documento.**

-   **Scopo di BERTopic:** BERTopic è una libreria di *topic modeling*. Il suo scopo è analizzare un **corpus di documenti (centinaia o migliaia)** per trovare cluster semantici (i "topic"). L'addestramento (`.fit()` o `.fit_transform()`) è il processo in cui questi topic vengono scoperti.
-   **Cosa fa `.transform()`:** Il metodo `.transform()`, usato nel codice attuale, non scopre nuovi topic. Serve a prendere uno o più nuovi documenti e **assegnarli ai topic già scoperti durante l'addestramento**.

Di conseguenza, l'attuale implementazione non può scoprire un "nuovo tag". Può solo fare una delle seguenti cose:
1.  Assegnare il testo a un topic **già esistente** nel modello BERTopic pre-addestrato.
2.  Classificare il testo come un **outlier** (topic `-1`), che il codice interpreta come "nessuna nuova categoria trovata".

> **In sintesi: il sistema attuale non scopre nulla di nuovo. Cerca di usare uno strumento di analisi di corpus per analizzare un singolo punto dati, il che è come cercare di calcolare la densità di una foresta guardando un solo albero.**

---

### 3. La Strategia Corretta: Scoperta Offline Basata su Cluster

Un sistema di scoperta di tag efficace non deve avvenire in tempo reale, ma come un **processo di training e analisi offline**, eseguito periodicamente (es. ogni notte o una volta a settimana).

Il file `proposed_new_tag_discovery.py` suggerisce un approccio che va in questa direzione, ma non è integrato correttamente nel flusso principale.

Ecco come dovrebbe funzionare un'implementazione robusta:

**Fase 1: Raccolta Dati**
- Si raccolgono tutte le conversazioni che sono state classificate come `altro` o che hanno avuto una bassa confidenza dall'LLM nell'ultimo periodo.
- Questo crea un **corpus di documenti "incerti" o "sconosciuti"**.

**Fase 2: Topic Modeling con BERTopic sul Corpus**
- Si addestra un nuovo modello BERTopic su questo corpus (`bertopic.fit_transform(corpus_incerto)`).
- L'output di questa fase è un insieme di **cluster semantici (topic)**. Ogni cluster rappresenta un potenziale nuovo tag. Ad esempio, BERTopic potrebbe creare un cluster di testi che parlano tutti di "smaltimento oli esausti" e "rifiuti speciali".

**Fase 3: Generazione e Proposta di Nuovi Tag**
- Per ogni topic significativo scoperto da BERTopic (ignorando gli outlier):
    1.  **Estrai Parole Chiave:** Si prendono le parole chiave che definiscono il topic (es. "olio", "smaltimento", "rifiuti", "sentine").
    2.  **Genera Nome e Descrizione con LLM:** Si usa un LLM con un prompt specifico per generare un nome di tag leggibile e una descrizione chiara basandosi sulle parole chiave e su alcuni documenti rappresentativi del cluster.
        -   *Prompt Esempio:* `"Data la seguente lista di parole chiave ('olio', 'smaltimento', 'rifiuti') e questi esempi di testo, genera un nome di tag breve (in formato snake_case) e una descrizione per questa categoria."*
        -   *Output LLM:* `{"tag_name": "gestione_rifiuti_speciali", "description": "Informazioni relative allo smaltimento di oli esausti, batterie e altri rifiuti speciali all'interno dell'area portuale."}`
    3.  **Salva come "Proposto":** Questo nuovo tag viene salvato nel database `TAG.tags` con uno stato speciale, ad esempio `status = 'proposed'` o `is_approved = false`.

**Fase 4: Validazione Umana**
- Si crea un'interfaccia di amministrazione dove un operatore umano può vedere i tag proposti.
- L'operatore può:
    - **Approvare** il nuovo tag (cambiando lo stato in `approved`).
    - **Modificare** il nome o la descrizione.
    - **Unire** il nuovo topic a un tag esistente (es. il topic "smaltimento" potrebbe essere unito al tag esistente "servizi_portuali").
    - **Rifiutare** il topic se non è rilevante.

**Vantaggi di Questo Approccio:**
-   **Metodologicamente Corretto:** Sfrutta BERTopic per il suo vero scopo.
-   **Robusto:** Scopre temi reali e ricorrenti, non basandosi su singole conversazioni.
-   **Controllato:** Il coinvolgimento umano (Human-in-the-loop) garantisce che i nuovi tag siano di alta qualità e coerenti con la tassonomia esistente.
-   **Automatizzato:** Riduce drasticamente il lavoro manuale di analisi delle conversazioni non classificate.

**Conclusione:** L'idea di usare BERTopic è eccellente, ma l'implementazione attuale è inefficace. È necessario passare da un modello di "fallback in tempo reale" a un modello di "analisi e scoperta offline" per rendere il sistema veramente intelligente e autonomo.
