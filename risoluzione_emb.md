# Soluzione al Problema di Cambio Embedding Engine

## ⚠️ AGGIORNAMENTO: CAUSA REALE IDENTIFICATA

**PROBLEMA CONFERMATO: INCONSISTENZA TRA `tenant_id` (UUID) E `tenant_slug`**

Dopo analisi approfondita del codice, il problema **NON** è solo un'incoerenza della cache, ma una **inconsistenza nell'uso degli identificatori tenant** durante il flusso di cambio engine:

### Flusso Problematico Identificato:

1. **API Change Engine (server.py:2430)**: Riceve `tenant_id` UUID (`16c222a9-f293-11ef-9315-96000228e7fe`)
2. **EmbeddingManager**: Usa UUID per normalizzazione e switch (`_normalize_tenant_id`)
3. **ClusteringTestService**: Chiama `get_shared_embedder(tenant_id='wopta')` - **USA SLUG!**
4. **Inconsistenza**: EmbeddingManager ha cache con chiave UUID, ma clustering service richiede con SLUG

**Nei log si vede chiaramente:**
- Cambio engine: `tenant 16c222a9-f293-11ef-9315-96000228e7fe` (UUID)
- Test clustering: `get_shared_embedder(tenant_id='wopta')` (SLUG)
- Risultato: Manager normalizza 'wopta' → '16c222a9-f293-11ef-9315-96000228e7fe', ma usa cache obsoleta

## 1. Analisi della Causa Profonda (Root Cause)

Il problema deriva da una **race condition nella normalizzazione degli identificatori tenant** combinata con incoerenza della cache. 

### Sequenza problematica nei log:

1. **22:18:10** - Cambio engine da `labse` a `openai_small` per `tenant_id=16c222a9-f293-11ef-9315-96000228e7fe`
2. **22:18:11** - EmbeddingManager forza reload con UUID: `✅ Embedder FORZATAMENTE ricaricato per tenant 16c222a9-f293-11ef-9315-96000228e7fe: OpenAIEmbedder`
3. **22:18:21** - ClusteringTestService richiede embedder per `tenant_id='wopta'` (SLUG)
4. **22:18:21** - EmbeddingManager normalizza: `🔄 EMBEDDING MANAGER: Normalizzato 'wopta' -> '16c222a9-f293-11ef-9315-96000228e7fe'`
5. **22:18:21** - **ERRORE CRITICO**: Invece di usare il nuovo OpenAIEmbedder, recupera vecchia istanza: `♻️ Riuso embedder esistente per tenant 16c222a9-f293-11ef-9315-96000228e7fe: LaBSEEmbedder`

### Causa Specifica:
Il problema è nella gestione della cache dell'`EmbeddingManager`. La cache usa come chiave il `tenant_id` normalizzato (UUID), ma nel processo di cambio engine:

1. La cache viene **parzialmente** invalidata (il riferimento viene aggiornato)
2. Ma l'**oggetto LaBSEEmbedder** rimane in memoria in uno stato "danneggiato" (senza `.model` per liberare GPU)
3. Quando il clustering service richiede l'embedder, la normalizzazione restituisce l'UUID corretto
4. Ma la cache manager restituisce l'**istanza obsoleta e danneggiata** invece di quella nuova

La causa radice è che **l'invalidazione della cache non rimuove completamente l'oggetto obsoleto**, che mantiene un riferimento "fantasma" e viene erroneamente riutilizzato.

## 2. Soluzione Proposta: Correzione della Gestione Cache nell'EmbeddingManager

La soluzione più robusta è correggere la logica di cache nell'`EmbeddingManager` per garantire che:

1. **Invalidazione completa**: Quando un embedder viene sostituito, l'istanza precedente viene completamente rimossa dalla cache
2. **Cleanup sicuro**: L'oggetto obsoleto viene correttamente deallocato prima di essere sostituito
3. **Prevenzione race condition**: Il lock assicura operazioni atomiche durante switch/reload

### Strategia di Fix:

**NON è necessario aggiungere nuovi metodi**, ma correggere il metodo esistente `switch_tenant_embedder` nell'`EmbeddingManager` per assicurare che la cache venga **veramente** pulita durante il `force_reload`.

Il problema specifico è nella logica di `_cleanup_current_embedder()` che non rimuove completamente l'oggetto dalla cache interna del manager.

## 3. Modifiche al Codice Suggerite

### Modifica UNICA: `EmbeddingEngine/embedding_manager.py`

Il fix richiede **una sola modifica** nel metodo `switch_tenant_embedder` per assicurare che il cleanup sia completo e che la cache venga veramente invalidata.

**Problema attuale (righe ~190-220):**
```python
def switch_tenant_embedder(self, tenant_id: str, force_reload: bool = False) -> BaseEmbedder:
    # ...
    with self._manager_lock:
        if force_reload:
            print(f"🔄 Force reload: cleanup embedder corrente e ricarica configurazione")
            self._cleanup_current_embedder()  # <- PROBLEMA: Non invalida cache completamente
            
            # Ottiene nuovo embedder
            self._current_embedder = embedding_factory.get_embedder_for_tenant(normalized_tenant_id, force_reload=True)
            self._current_tenant_id = normalized_tenant_id
            # ...
```

**Soluzione corretta:**
```python
def switch_tenant_embedder(self, tenant_id: str, force_reload: bool = False) -> BaseEmbedder:
    """
    Forza switch a embedder specifico per tenant
    
    CORREZIONE 2025-08-25: Fix cache inconsistency durante force_reload
    """
    # NORMALIZZA SEMPRE A UUID
    normalized_tenant_id = self._normalize_tenant_id(tenant_id)
    
    with self._manager_lock:
        print(f"🔄 Switch forzato embedder a tenant {tenant_id} -> {normalized_tenant_id} (force_reload={force_reload})")
        
        # Se force_reload=True, SEMPRE cleanup COMPLETO e ricarica
        if force_reload:
            print(f"🔄 Force reload: cleanup embedder corrente e ricarica configurazione")
            
            # *** CORREZIONE CRITICA: CLEANUP COMPLETO ***
            self._cleanup_current_embedder()
            
            # *** AGGIUNTA: RESET ESPLICITO CACHE INTERNA ***
            self._current_embedder = None
            self._current_tenant_id = None
            
            print(f"🧹 Cache interna EmbeddingManager completamente invalidata")
            
            # FORZA ANCHE LA FACTORY A RICARICARE BYPASSANDO LA SUA CACHE
            try:
                print(f"🔧 FORCE RELOAD: ordino alla factory di bypassare completamente la cache")
                
                # *** IMPORTANTE: OTTIENI NUOVO EMBEDDER FRESH ***
                new_embedder = embedding_factory.get_embedder_for_tenant(normalized_tenant_id, force_reload=True)
                
                # *** AGGIORNAMENTO ATOMICO CACHE ***
                self._current_embedder = new_embedder
                self._current_tenant_id = normalized_tenant_id
                
                print(f"✅ Embedder FORZATAMENTE ricaricato per tenant {normalized_tenant_id}: {type(self._current_embedder).__name__}")
                return self._current_embedder
                
            except Exception as e:
                # ... gestione errori
        
        # ... resto del codice invariato
```

**Punti chiave del fix:**
1. **Reset esplicito**: `self._current_embedder = None` e `self._current_tenant_id = None` **prima** di ottenere il nuovo
2. **Aggiornamento atomico**: Il nuovo embedder viene assegnato solo **dopo** che è stato ottenuto con successo
3. **Prevenzione race**: Il lock assicura che non ci siano accessi concorrenti durante la sostituzione

## 4. Vantaggi di Questa Soluzione

*   **Minimal Impact:** Richiede modifiche in **un solo metodo** di **un solo file**
*   **Root Cause Fix:** Risolve il problema alla fonte (gestione cache nell'EmbeddingManager) invece di aggiungere workaround
*   **Backward Compatible:** Non cambia l'API pubblica, solo la logica interna del cache cleanup
*   **Thread Safe:** La soluzione mantiene la sicurezza thread esistente tramite `_manager_lock`
*   **Robustezza:** Previene sia l'uso di modelli obsoleti sia crash dovuti a oggetti parzialmente deallocati

## 5. Test della Soluzione

Dopo aver implementato il fix, per verificare che funzioni:

1. **Avviare il server** con il codice modificato
2. **Cambiare engine** per un tenant via API POST `/api/ai-config/{tenant_id}/embedding-engines`
3. **Verificare nei log** che appaia: `🧹 Cache interna EmbeddingManager completamente invalidata`
4. **Eseguire test clustering** e verificare che usi il nuovo embedder corretto
5. **Controllare che i log mostrino** il nuovo tipo di embedder (es. `OpenAIEmbedder`) invece del vecchio

**Sequenza di log attesa dopo il fix:**
```
🔄 Switch forzato embedder a tenant wopta -> 16c222a9-f293-11ef-9315-96000228e7fe (force_reload=True)
🔄 Force reload: cleanup embedder corrente e ricarica configurazione
🧹 Cache interna EmbeddingManager completamente invalidata
🔧 FORCE RELOAD: ordino alla factory di bypassare completamente la cache
✅ Embedder FORZATAMENTE ricaricato per tenant 16c222a9-f293-11ef-9315-96000228e7fe: OpenAIEmbedder
...
♻️ Riuso embedder esistente per tenant 16c222a9-f293-11ef-9315-96000228e7fe: OpenAIEmbedder  <- CORRETTO!
```
