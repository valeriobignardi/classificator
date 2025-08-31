# Piano di Refactoring: Separazione Parametri Token e UI Dinamica

**Autore:** Valerio Bignardi  
**Data creazione:** 31 Agos## FASE 2: Implementazione API Layer

### FASE 2.1: API Endpoints per Gestione LLM ✅ COMPLETATA
- ✅ COMPLETATA: GET /api/llm/models/<tenant_id> - Lista modelli disponibili
- ✅ COMPLETATA: GET /api/llm/parameters/<tenant_id> - Parametri attuali tenant
- ✅ COMPLETATA: PUT /api/llm/parameters/<tenant_id> - Aggiorna parametri tenant
- ✅ COMPLETATA: GET /api/llm/model-info/<model_name> - Info modello specifico
- ✅ COMPLETATA: POST /api/llm/validate-parameters - Validazione parametri
- ✅ COMPLETATA: POST /api/llm/reset-parameters/<tenant_id> - Reset parametri
- ✅ COMPLETATA: POST /api/llm/test-model/<tenant_id> - Test modello con parametri
- ✅ COMPLETATA: Funzione validate_llm_parameters per controlli specifici per modello
- ✅ COMPLETATA: Gestione errori e logging per tutte le API
- ✅ COMPLETATA: Integrazione con IntelligentClassifier per caricamento tenant config*Obiettivo:** Separare i parametri di tokenizzazione per embedding e LLM, implementare UI dinamica per configurazione parametri LLM per tenant

---

## **Stato Attuale Analizzato**

### **Parametri Token Attuali**
- ❌ **Condiviso**: `tokenization.max_tokens: 8000` (usato da embedding E LLM input)
- ✅ **Separato**: `llm.generation.max_tokens: 150` (solo LLM output)

### **Parametri LLM Identificati**
**Generazione:**
- `max_tokens`: 150 (output)
- `temperature`: 0.1
- `top_k`: 40  
- `top_p`: 0.9
- `repeat_penalty`: 1.1

**Connessione:**
- `timeout`: 300
- `url`: http://localhost:11434

**Raw Mode:**
- `raw_mode.enabled`: true/false
- `models_requiring_raw`: array

### **Limiti Context per Modello**
- **Mistral 7B**: 8k context max
- **Mistral Nemo**: 128k context max
- **Llama3.3:70b**: 128k+ context
- **GPT-OSS:20b**: da verificare

---

## **FASE 1: Separazione Config Backend** ⏳

### **1.1 Modifica config.yaml** ✅ COMPLETATO
**File:** `/home/ubuntu/classificatore/config.yaml`

**Azioni:**
- [x] Backup config.yaml attuale → `backup/config_20250831_101120.yaml`
- [x] Separare `tokenization` in `embedding.tokenization` e `llm.tokenization`
- [x] Aggiungere definizioni complete per ogni modello con limiti specifici
- [x] Aggiungere struttura `tenant_configs` per parametri LLM personalizzati

**Struttura Target:**
```yaml
embedding:
  tokenization:
    max_tokens: 8000

llm:
  tokenization:
    max_tokens: 8000  # INPUT limit
  generation:
    max_tokens: 150   # OUTPUT limit
    temperature: 0.1
    top_k: 40
    top_p: 0.9
    repeat_penalty: 1.1
  connection:
    timeout: 300
  models:
    available:
      - name: "mistral:7b"
        display_name: "Mistral 7B v0.3"
        max_input_tokens: 8000
        max_output_tokens: 4000
        context_limit: 8192
        requires_raw_mode: true
      - name: "mistral-nemo:latest"
        display_name: "Mistral Nemo 12B"
        max_input_tokens: 128000
        max_output_tokens: 4000
        context_limit: 131072
        requires_raw_mode: false

tenant_configs:
  humanitas:
    llm_model: "mistral:7b"
    llm_parameters:
      tokenization:
        max_tokens: 7000
      generation:
        max_tokens: 200
        temperature: 0.2
        top_k: 50
        top_p: 0.85
        repeat_penalty: 1.0
      connection:
        timeout: 300
```

### **1.2 Aggiornamento TokenizationManager** ✅ COMPLETATO
**File:** `/home/ubuntu/classificatore/Utils/tokenization_utils.py`

**Azioni:**
- [x] Backup file esistente → `backup/tokenization_utils_20250831_101246.py`
- [x] Modificare `_load_config()` per leggere `embedding.tokenization` separatamente
- [x] Aggiungere metodo `load_llm_tokenization_config(tenant_id)`
- [x] Mantenere backward compatibility

### **1.3 Aggiornamento Embedding Engines** ✅ COMPLETATO
**File:** `/home/ubuntu/classificatore/EmbeddingEngine/*.py`

**Azioni:**
- [x] `bge_m3_embedder.py`: Già usa TokenizationManager → automaticamente compatibile
- [x] `openai_embedder.py`: Già usa TokenizationManager → automaticamente compatibile
- [x] `labse_embedder.py`: Verificato e non richiede modifiche specifiche

### **1.4 Aggiornamento IntelligentClassifier** ✅ COMPLETATO
**File:** `/home/ubuntu/classificatore/Classification/intelligent_classifier.py`

**Azioni:**
- [x] Backup file esistente → `backup/intelligent_classifier_20250831_101246.py`
- [x] Modificare caricamento config per leggere parametri tenant-specific
- [x] Implementare `load_tenant_llm_config(tenant_id)`
- [x] Aggiornare logica inizializzazione con parametri tenant

---

## **FASE 2: API Backend per Gestione Parametri** ⏳

### **2.1 Nuove API Endpoints** ✅ COMPLETATA
**File:** `/home/ubuntu/classificatore/server.py`

**Endpoints implementati:**
- [x] `GET /api/llm/models/{tenant_id}` - Lista modelli disponibili con limiti
- [x] `GET /api/llm/parameters/{tenant_id}` - Parametri LLM correnti del tenant
- [x] `PUT /api/llm/parameters/{tenant_id}` - Aggiorna parametri LLM tenant
- [x] `GET /api/llm/model-info/{model_name}` - Info specifiche modello (limiti, capabilities)
- [x] `POST /api/llm/validate-parameters` - Validazione parametri prima del salvataggio
- [x] `POST /api/llm/reset-parameters/{tenant_id}` - Reset parametri tenant a default
- [x] `POST /api/llm/test-model/{tenant_id}` - Test modello con parametri
- [x] `GET /api/llm/tenants` - Lista tenant con configurazioni personalizzate

### **2.2 Servizi Backend** ✅ COMPLETATA
**File:** `/home/ubuntu/classificatore/Services/llm_configuration_service.py`

**Funzioni implementate:**
- [x] `get_available_models(tenant_id)` - Lista modelli con info complete
- [x] `get_tenant_parameters(tenant_id)` - Parametri LLM tenant con metadati
- [x] `update_tenant_parameters(tenant_id, parameters)` - Aggiornamento sicuro con backup
- [x] `validate_parameters(parameters, model_name)` - Validazione avanzata con vincoli modello
- [x] `test_model_configuration(tenant_id, model_name, parameters)` - Test modelli
- [x] `reset_tenant_parameters(tenant_id)` - Reset a parametri default
- [x] `get_tenant_list()` - Lista tenant con config personalizzate
- [x] Cache intelligente con hot-reload automatico
- [x] Thread-safe operations con lock meccanismo
- [x] Backup automatico prima delle modifiche
- [x] Integrazione completa con API server.py

---

## **FASE 3: Componenti React Frontend** ⏳

### **3.1 Componente LLMModelSelector** ❌ TODO
**File:** `/home/ubuntu/classificatore/human-review-ui/src/components/LLMModelSelector.tsx`

**Funzionalità:**
- [ ] Dropdown modelli disponibili per tenant
- [ ] Caricamento dinamico limiti modello selezionato
- [ ] Aggiornamento automatico parametri quando cambia modello
- [ ] Visualizzazione info modello (context limit, capabilities)

### **3.2 Componente LLMParametersPanel** ❌ TODO
**File:** `/home/ubuntu/classificatore/human-review-ui/src/components/LLMParametersPanel.tsx`

**Controlli UI:**
- [ ] **Tokenization Slider**: `max_tokens` (range dinamico basato su modello)
- [ ] **Generation Sliders**: 
  - `max_tokens` (50-2000)
  - `temperature` (0.0-2.0)
  - `top_k` (1-100)
  - `top_p` (0.1-1.0) 
  - `repeat_penalty` (0.8-1.5)
- [ ] **Connection Settings**: `timeout` (30-600s)
- [ ] **Validazione real-time** con indicatori visivi
- [ ] **Reset to defaults** button

### **3.3 Integrazione in Pagina Configurazione** ❌ TODO
**File:** `/home/ubuntu/classificatore/human-review-ui/src/components/ConfigurationPage.tsx` (da creare o modificare esistente)

**Layout:**
- [ ] Sezione "Configurazione LLM" 
- [ ] LLMModelSelector in alto
- [ ] LLMParametersPanel sotto il selector
- [ ] Salvataggio automatico + feedback utente
- [ ] Anteprima configurazione JSON

### **3.4 Servizi API Frontend** ❌ TODO
**File:** `/home/ubuntu/classificatore/human-review-ui/src/services/llmConfigService.ts`

**Metodi da implementare:**
- [ ] `getAvailableModels(tenantId)`
- [ ] `getTenantLLMParameters(tenantId)`
- [ ] `updateTenantLLMParameters(tenantId, parameters)`
- [ ] `getModelConstraints(modelName)`
- [ ] `validateParameters(modelName, parameters)`

---

## **FASE 4: Integrazione e Testing** ⏳

### **4.1 Testing Backend** ❌ TODO
**File:** `/home/ubuntu/classificatore/tests/test_llm_config.py` (da creare)

**Test da implementare:**
- [ ] Test separazione parametri embedding/LLM
- [ ] Test caricamento configurazioni tenant
- [ ] Test validazione limiti per modello
- [ ] Test API endpoints nuovi
- [ ] Test backward compatibility

### **4.2 Testing Frontend** ❌ TODO
**File:** `/home/ubuntu/classificatore/tests/test_llm_ui.py` (da creare)

**Test da implementare:**
- [ ] Test componenti UI
- [ ] Test caricamento dinamico parametri
- [ ] Test validazione real-time
- [ ] Test cambio modello dinamico
- [ ] Test salvataggio configurazioni

### **4.3 Testing End-to-End** ❌ TODO
**Test completi:**
- [ ] Cambio modello → aggiornamento UI → classificazione con nuovi parametri
- [ ] Configurazione tenant diversi con parametri diversi
- [ ] Gestione errori e fallback
- [ ] Performance con nuova struttura

---

## **FASE 5: Deployment e Documentazione** ⏳

### **5.1 Migrazione Graduale** ❌ TODO
- [ ] Deploy con backward compatibility
- [ ] Migrazione configurazioni esistenti
- [ ] Verifica funzionamento tenant esistenti
- [ ] Attivazione nuove funzionalità

### **5.2 Documentazione** ❌ TODO
- [ ] Aggiornamento README con nuove funzionalità
- [ ] Documentazione API endpoints
- [ ] Guide utente per configurazione UI
- [ ] Troubleshooting common issues

---

## **Status Tracking**

### **Completato** ✅
- **FASE 1.1**: Separazione config.yaml - backup, tokenization separata, definizioni modelli, struttura tenant
- **FASE 1.2**: Aggiornamento TokenizationManager - backup, config separata, metodo tenant, backward compatibility  
- **FASE 1.3**: Aggiornamento Embedding Engines - compatibilità automatica con TokenizationManager
- **FASE 1.4**: Aggiornamento IntelligentClassifier - caricamento config tenant, parametri dinamici

### **In Corso** 🔄
- **FASE 2.1**: Nuove API Endpoints (prossimo step)

### **Da Fare** ❌
- Tutti gli elementi sopra elencati

### **Bloccato** 🚫
*Nessun elemento bloccato*

---

## **Note Tecniche**

### **Vincoli Identificati:**
1. **Mistral 7B**: Hard limit 8k context, richiede raw mode
2. **Backward Compatibility**: Config esistenti devono continuare a funzionare
3. **Tenant Isolation**: Ogni tenant deve avere configurazioni indipendenti
4. **Real-time Updates**: Cambio parametri deve applicarsi immediatamente

### **Rischi Potenziali:**
1. **Breaking Changes**: Modifiche config potrebbero rompere codice esistente
2. **Performance**: Caricamento dinamico potrebbe rallentare UI
3. **Validazione**: Parametri invalidi potrebbero causare errori classificazione
4. **Sincronizzazione**: Frontend e backend devono rimanere allineati

### **Strategie Mitigazione:**
1. **Backup automatico** di tutti i file modificati
2. **Testing incrementale** dopo ogni fase
3. **Rollback plan** per ogni modifica critica
4. **Feature flags** per attivazione graduale

---

**Prossima Azione:** Attendere conferma per iniziare FASE 1.1 - Modifica config.yaml
