# Problema: Frontend non riceve risposta Test Clustering

## Data: 2025-10-24

## Sintomi
- Il test clustering viene completato con successo sul backend
- Il risultato viene salvato nel database (versione 55, ID 99)
- Il frontend non riceve mai la risposta HTTP
- Nel browser appare errore 400 o timeout

## Causa Root
Il test clustering impiega circa **20-21 minuti** per completare:
- **Fase Embedding**: ~595 secondi (~10 minuti) per 8235 conversazioni
- **Fase Clustering HDBSCAN**: ~661 secondi (~11 minuti)
- **Totale**: ~1256 secondi (~21 minuti)

Il frontend (o un proxy intermediario come Nginx) ha probabilmente un timeout di default (es. 60 secondi, 5 minuti, o 10 minuti) che è inferiore al tempo necessario per completare l'operazione.

## Evidenze dai Log Backend

```
2025-10-24 22:34:41,115 - root - INFO - ✅ Estratte 8235 sessioni valide per 'Humanitas'
🔍 Generazione embeddings per 8235 conversazioni...
✅ Embedding remoto completato in 595.262s
🔍 Clustering HDBSCAN su 8235 embedding (48.3MB)...
✅ Clustering completato in 661.58s
2025-10-24 22:45:35,896 - ClusteringResultsDB - INFO - ✅ Connesso al database TAG locale
2025-10-24 22:45:36,367 - ClusteringResultsDB - INFO - ✅ Risultato clustering salvato: tenant=015007d9-d413-11ef-86a5-96000228e7fe, version=55, id=99
✅ [API] Test clustering completato con successo
```

## Errore Successivo

Dopo il completamento, un nuovo tentativo ha mostrato:
```
❌ [API] Test clustering fallito: Errore generazione embeddings: Servizio embedding non disponibile e fallback disabilitato: Errore connessione: HTTPConnectionPool(host='localhost', port=8081): Max retries exceeded with url: /embed (Caused by ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
```

Questo suggerisce che LaBSE era in riavvio o sovraccarico.

## Soluzioni Possibili

### Soluzione 1: Aumentare timeout Frontend (RACCOMANDATO)
Se il frontend ha un file di configurazione con timeout HTTP, aumentarlo a:
- **Minimo**: 1800 secondi (30 minuti)
- **Raccomandato**: 3600 secondi (1 ora)

### Soluzione 2: Implementare Pattern Asincrono (MIGLIORE)
Invece di aspettare la risposta sincrona:

1. **Frontend**: Invia richiesta POST e riceve subito un `job_id`
2. **Backend**: Avvia clustering in background, restituisce subito
3. **Frontend**: Polling periodico su `/api/clustering/<tenant_id>/job/<job_id>` ogni 10-15 secondi
4. **Backend**: Restituisce stato: `in_progress`, `completed`, `failed`
5. **Frontend**: Quando `completed`, mostra risultati

Esempio implementazione:

```python
# Backend endpoint sincrono diventa asincrono
@app.route('/api/clustering/<tenant_id>/test', methods=['POST'])
def test_clustering_async(tenant_id):
    job_id = str(uuid.uuid4())
    
    # Avvia task in background (usa threading o celery)
    threading.Thread(
        target=run_clustering_background,
        args=(tenant_id, job_id, custom_parameters)
    ).start()
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'status': 'in_progress',
        'message': 'Clustering avviato in background'
    }), 202  # Accepted

# Nuovo endpoint per polling stato
@app.route('/api/clustering/<tenant_id>/job/<job_id>', methods=['GET'])
def get_clustering_job_status(tenant_id, job_id):
    # Leggi stato da cache/db
    job_status = get_job_from_cache(job_id)
    
    if job_status['status'] == 'completed':
        return jsonify({
            'success': True,
            'status': 'completed',
            'result': job_status['result']
        }), 200
    elif job_status['status'] == 'in_progress':
        return jsonify({
            'success': True,
            'status': 'in_progress',
            'progress': job_status.get('progress', 0)
        }), 200
    else:
        return jsonify({
            'success': False,
            'status': 'failed',
            'error': job_status.get('error')
        }), 500
```

### Soluzione 3: Ottimizzare Performance (LUNGO TERMINE)
- Ridurre numero conversazioni per test (sample intelligente)
- Caching embeddings già calcolati
- Parallelizzare generazione embeddings (batch multipli)
- Usare RAPIDS cuML per clustering GPU-accelerato (già fatto)

## Test Effettuati

### Test LaBSE Timeout
```bash
python3 test_labse_timeout.py
```

Risultato: ✅ LaBSE funziona perfettamente con timeout 14400s
- 500 testi da tracing.log processati in 1.67 secondi
- Embeddings 768-dimensionali generati correttamente

### Test Backend Timeout
Modificati file con timeout hardcoded 300s → 14400s:
- `EmbeddingEngine/simple_embedding_manager.py`
- `EmbeddingEngine/embedding_engine_factory.py`
- `EmbeddingEngine/embedder_factory.py`

Risultato: ✅ Backend processa batch lunghi senza timeout

## Raccomandazioni

1. **Immediato**: Verificare timeout configurato nel frontend
   - Cerca file config.js, .env, o settings con parametri timeout
   - Se presente Nginx/Apache, verifica proxy_read_timeout

2. **Breve termine**: Implementare pattern asincrono con polling
   - Migliore UX: mostra progress bar
   - Nessun timeout: frontend controlla periodicamente
   - Scalabile: backend può gestire multiple richieste

3. **Lungo termine**: Ottimizzare performance clustering
   - Ridurre dataset di test (es. sample 2000 conversazioni)
   - Implementare caching intelligente embeddings
   - Monitoring e alerting per operazioni lunghe

## File Modificati (Backup disponibili)

Tutti i file modificati hanno backup in `./backup/` con timestamp `20251024_180500`:
- `simple_embedding_manager.py`
- `embedding_engine_factory.py`
- `embedder_factory.py`

## Autore
Sistema di Classificazione - Fix Timeout LaBSE
