# Gestione Conversazioni Troppo Lunghe - OpenAI Token Limit

## Problema Risolto

Quando il sistema di clustering elabora conversazioni molto lunghe (oltre 8192 token per OpenAI), l'API OpenAI restituisce un errore 400 per superamento del limite di token. Questo blocca completamente il processo di clustering.

## Soluzione Implementata

### 1. Salvataggio Automatico delle Conversazioni Problematiche

Quando si verifica un errore di token limit, il sistema ora:

- **Salva automaticamente** tutte le conversazioni del batch problematico in un file di log nella root del progetto
- **Include il Session ID** per ogni conversazione (elemento chiave per l'analisi)
- **Stima i token** per identificare rapidamente le conversazioni più lunghe
- **Aggiunge metadata** completi (data, modello, errore originale)

### 2. File di Log Generato

Nome formato: `conversazioni_troppo_lunghe_YYYYMMDD_HHMMSS.txt`

Contenuto del file:
```
CONVERSAZIONI TROPPO LUNGHE - LOG AUTOMATICO
Data: 2025-08-26 12:11:27
Modello OpenAI: text-embedding-3-large
Limite token: 8000
Errore originale: Error code: 400 - {'error': {'message': "This model's maximum context length is 8192 tokens..."}}
Numero conversazioni nel batch: 3
================================================================================

CONVERSAZIONE #1
Session ID: session_001
Lunghezza caratteri: 27
Token stimati: 7
Problematica: NO
------------------------------------------------------------
TESTO COMPLETO:
[Testo completo della conversazione...]
------------------------------------------------------------
```

### 3. Modifiche al Codice

#### File: `EmbeddingEngine/openai_embedder.py`
- **Nuovo parametro `session_ids`** nel metodo `encode()` 
- **Nuova funzione `_save_problematic_conversations()`** per salvare le conversazioni problematiche
- **Logging avanzato** con Session ID negli errori di token limit
- **Gestione robusta degli errori** senza bloccare il sistema

#### File: `Pipeline/end_to_end_pipeline.py`
- **Passaggio dei session_ids** alle chiamate `embedder.encode()`
- **Supporto per identificazione** delle conversazioni nei log di errore

#### File: `Clustering/clustering_test_service_new.py`
- **Integrazione dei session_ids** nel processo di clustering
- **Tracciabilità completa** delle conversazioni problematiche

### 4. Vantaggi della Soluzione

1. **Non blocca il sistema**: Gli errori vengono gestiti e loggati senza fermare il processo
2. **Tracciabilità completa**: Ogni conversazione problematica è identificabile tramite Session ID
3. **Analisi facilitata**: File di log strutturati per facile consultazione
4. **Compatibilità**: Funziona con qualsiasi tenant (non solo alleanza)
5. **Debugging avanzato**: Token stimati e metadata completi per ogni conversazione

### 5. Utilizzo

Il sistema funziona automaticamente:
1. Durante il clustering, se una conversazione è troppo lunga
2. Il sistema cattura l'errore OpenAI
3. Salva automaticamente tutte le conversazioni del batch in un file di log
4. Mostra l'errore nel terminale con riferimento al Session ID
5. Il file di log viene creato nella root del progetto per analisi successiva

### 6. Esempio di Errore Gestito

```bash
🚨 ERRORE TOKEN LIMIT OPENAI:
   Errore: Error code: 400 - {'error': {'message': "This model's maximum context length is 8192 tokens, however you requested 8588 tokens"}}
   Numero testi nel batch: 100
   📝 Testo 89 sospetto (7277 token stimati) (Session ID: abc123-def456-ghi789):
      Inizio: '[UTENTE] Ciao sono Giulia sto veramente molto male...'
      Fine: '...Se hai bisogno di altre idee, sono qui per aiutarti.'
      Lunghezza: 29104 caratteri
💾 Salvataggio conversazioni problematiche in: conversazioni_troppo_lunghe_20250826_121127.txt
✅ File salvato: /home/ubuntu/classificatore/conversazioni_troppo_lunghe_20250826_121127.txt
```

## Ultima modifica
Data: 26 agosto 2025
Autore: GitHub Copilot
