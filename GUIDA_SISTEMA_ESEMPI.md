# 📚 GUIDA SISTEMA GESTIONE ESEMPI MULTI-TENANT

## 🎯 PANORAMICA
Il sistema di gestione esempi permette di creare e gestire esempi di conversazioni che vengono automaticamente inseriti nei prompt tramite il placeholder `{{examples_text}}`.

## 🚀 AVVIO SISTEMA

### 1. Avvia API Server Esempi
```bash
cd /home/ubuntu/classificazione_discussioni_bck_23_08_2025
python esempi_api_server.py
```
Server disponibile su: `http://localhost:5001`

### 2. Avvia Frontend React
```bash
cd human-review-ui
npm start
```
Frontend disponibile su: `http://localhost:3000`

## 📋 UTILIZZO INTERFACCIA WEB

### 1. **Selezione Tenant**
- Apri l'interfaccia web
- Seleziona il tenant dalla sidebar sinistra
- Il sistema caricherà automaticamente gli esempi del tenant

### 2. **Gestione Esempi**
- Vai alla sezione **"Configurazione"** → **"Gestione Esempi"**
- Visualizza esempi esistenti
- Crea nuovi esempi con il pulsante **"Nuovo Esempio"**

### 3. **Creazione Nuovo Esempio**
- **Nome Esempio**: Identificativo unico (es. `richiesta_preventivo_auto`)
- **Categoria**: Classificazione tematica (es. `preventivi_auto`)
- **Livello Difficoltà**: `FACILE`, `MEDIO`, `DIFFICILE`
- **Descrizione**: Breve descrizione dell'esempio
- **Contenuto Conversazione**: Formato standardizzato

#### ✅ **Formato Conversazione Corretto:**
```
UTENTE: La mia domanda qui

ASSISTENTE: La risposta dell'assistente qui

UTENTE: Un'altra domanda

ASSISTENTE: Un'altra risposta
```

### 4. **Preview Placeholder**
- Usa il pulsante **"Preview examples_text"**
- Visualizza come appariranno gli esempi nel prompt finale
- Vedi statistiche: numero conversazioni e caratteri totali

## 🔧 API ENDPOINTS

### Gestione Esempi
- `GET /api/examples?tenant_id=<uuid>` - Lista esempi
- `POST /api/examples` - Crea esempio
- `PUT /api/examples/<id>` - Aggiorna esempio  
- `DELETE /api/examples/<id>` - Elimina esempio

### Placeholder System
- `GET /api/examples/placeholder?tenant_id=<uuid>&limit=<n>` - Content per `{{examples_text}}`
- `GET /api/prompts/with-examples` - Prompt con esempi sostituiti

### Sistema
- `GET /health` - Status server
- `GET /api/info` - Informazioni API

## 📊 ESEMPIO PRATICO

### 1. Crea Esempio Preventivo Auto:
```json
{
  "tenant_id": "16c222a9-f293-11ef-9315-96000228e7fe",
  "esempio_name": "richiesta_preventivo_auto",
  "esempio_content": "UTENTE: Vorrei un preventivo per l'assicurazione auto\n\nASSISTENTE: Sarò felice di aiutarti. Che tipo di veicolo possiedi?\n\nUTENTE: Una Fiat 500 del 2020\n\nASSISTENTE: Perfetto. Il preventivo per la tua Fiat 500 è di €45 al mese per la polizza base.",
  "categoria": "preventivi_auto",
  "livello_difficolta": "MEDIO",
  "description": "Esempio di richiesta preventivo assicurazione auto"
}
```

### 2. Il Placeholder `{{examples_text}}` Diventa:
```
UTENTE: Vorrei un preventivo per l'assicurazione auto

ASSISTENTE: Sarò felice di aiutarti. Che tipo di veicolo possiedi?

UTENTE: Una Fiat 500 del 2020

ASSISTENTE: Perfetto. Il preventivo per la tua Fiat 500 è di €45 al mese per la polizza base.
```

## 🎛️ COMANDI UTILI

### Test Connessione API:
```bash
curl http://localhost:5001/health
```

### Lista Esempi Tenant:
```bash
curl "http://localhost:5001/api/examples?tenant_id=wopta" | jq .
```

### Preview Placeholder:
```bash
curl "http://localhost:5001/api/examples/placeholder?tenant_id=wopta&limit=2" | jq .
```

## ⚠️ NOTE IMPORTANTI

1. **Formato Rigido**: Rispetta esattamente il formato `UTENTE:` / `ASSISTENTE:` con a capo
2. **Multi-Tenant**: Gli esempi sono isolati per tenant
3. **No Fallback**: Se mancano esempi, il sistema **NON** usa fallback hardcoded
4. **Placeholder Automatico**: La sostituzione `{{examples_text}}` è completamente automatica

## 🐛 TROUBLESHOOTING

### Problema: "Nessun esempio trovato"
- Verifica che il tenant sia selezionato
- Controlla che esistano esempi per quel tenant
- Verifica connessione API server (porta 5001)

### Problema: "API Server non raggiungibile"
- Riavvia API server: `python esempi_api_server.py`
- Verifica porta 5001 libera
- Controlla log errori nel terminale

### Problema: "Placeholder non sostituito"
- Verifica che il prompt contenga esattamente `{{examples_text}}`
- Controlla che esistano esempi attivi per il tenant
- Usa il metodo `get_prompt_with_examples()` del PromptManager

## ✅ STATO SISTEMA

- ✅ Database esempi operativo (tabella `esempi`)
- ✅ API server attivo (porta 5001)  
- ✅ Frontend integrato (sezione Configurazioni)
- ✅ Multi-tenant isolation
- ✅ Placeholder substitution automatica
- ✅ CRUD operations complete
- ✅ 4+ esempi di test disponibili per tenant Wopta

---
**Sistema completamente operativo e pronto per uso produzione! 🚀**
