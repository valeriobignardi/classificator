# PIANO ELIMINAZIONE RETROCOMPATIBILITÀ tenant_or_id → SOLO oggetto Tenant

**Obiettivo**: Eliminare TUTTA la retrocompatibilità `tenant_or_id` sostituendola con l'obbligo di passare l'oggetto `Tenant`.

**Regola**: Ogni funzione deve ricevere come parametro l'oggetto `Tenant` e prelevare da esso ciò che serve (`tenant.tenant_id`, `tenant.tenant_name`, etc).

## STATUS PLAN

### ✅ COMPLETATI
1. `/home/ubuntu/classificatore/Clustering/clustering_test_service.py` ✅ COMPLETO
   - ✅ `_get_pipeline(self, tenant_or_id)` → `_get_pipeline(self, tenant)`
   - ✅ `get_sample_conversations(self, tenant_or_id, limit)` → `get_sample_conversations(self, tenant, limit)`

2. `/home/ubuntu/classificatore/EmbeddingEngine/embedding_manager.py` ✅ COMPLETO
   - ✅ `get_shared_embedder(self, tenant_or_id)` → `get_shared_embedder(self, tenant)`
   - ✅ `switch_tenant_embedder(self, tenant_or_id, force_reload)` → `switch_tenant_embedder(self, tenant, force_reload)`
   - ✅ `invalidate_cache_for_tenant(self, tenant_or_id)` → `invalidate_cache_for_tenant(self, tenant)`

3. `/home/ubuntu/classificatore/EmbeddingEngine/simple_embedding_manager.py` ✅ COMPLETO
   - ✅ `get_embedder_for_tenant(self, tenant_or_id)` → `get_embedder_for_tenant(self, tenant)`

4. `/home/ubuntu/classificatore/Utils/prompt_manager.py` ✅ COMPLETO - **TUTTI I METODI**
   - ✅ `validate_tenant_prompts_strict(self, tenant_id)` → `validate_tenant_prompts_strict(self, tenant)` 
   - ✅ `get_prompt_strict(self, tenant_id)` → `get_prompt_strict(self, tenant)`
   - ✅ `get_prompt(self, tenant_or_id, ...)` → `get_prompt(self, tenant, ...)`
   - ✅ `_load_prompt_from_db(self, tenant_or_id, ...)` → `_load_prompt_from_db(self, tenant, ...)`
   - ✅ `list_prompts_for_tenant(self, tenant_or_id)` → `list_prompts_for_tenant(self, tenant)`
   - ✅ `get_all_prompts_for_tenant(self, tenant_or_id)` → `get_all_prompts_for_tenant(self, tenant)`
   - ✅ `get_examples_for_placeholder(self, tenant_or_id, ...)` → `get_examples_for_placeholder(self, tenant, ...)`
   - ✅ `create_example(self, tenant_or_id, ...)` → `create_example(self, tenant, ...)`
   - ✅ `update_example(self, tenant_or_id, ...)` → `update_example(self, tenant, ...)`
   - ✅ `get_examples_list(self, tenant_or_id, ...)` → `get_examples_list(self, tenant, ...)`
   - ✅ `delete_example(self, esempio_id, tenant_or_id)` → `delete_example(self, esempio_id, tenant)`
   - ✅ `get_prompt_with_examples(self, tenant_id, ...)` → `get_prompt_with_examples(self, tenant, ...)`

5. `/home/ubuntu/classificatore/Utils/tool_manager.py` ✅ COMPLETO
   - ✅ `get_tool_by_name(self, tool_name, tenant_or_id)` → `get_tool_by_name(self, tool_name, tenant)`

6. `/home/ubuntu/classificatore/esempi_api_server.py` ✅ COMPLETO
   - ✅ Riga 129: `tenant_or_id=tenant_id` → `tenant=Tenant.from_uuid(tenant_id)` + passaggio oggetto tenant
   - ✅ Riga 208: `tenant_or_id=tenant_id` → `tenant=Tenant.from_uuid(tenant_id)` + passaggio oggetto tenant  
   - ✅ Riga 333: `tenant_or_id=tenant_id` → `tenant=Tenant.from_uuid(tenant_id)` + passaggio oggetto tenant

### 🎯 **RISULTATO FINALE**

**COMPLETAMENTO**: 6/6 file completati - **ARCHITETTURA TENANT COMPLETAMENTE REFACTORIZZATA!**

🚀 **SUCCESSO TOTALE**: TUTTI i metodi che usavano `tenant_or_id` sono stati convertiti per usare ESCLUSIVAMENTE oggetti `Tenant`.

## AZIONI IMMEDIATE

**✅ COMPLETATO**: Tutti i file refactorizzati con successo!

**PROSSIMO**: Test completo del sistema per verificare:
1. Cross-contamination ancora risolta (HUMANITAS ≠ WOPTA)
2. HDBSCAN clustering funziona con nuova architettura 
3. API cache ottimizzata funziona
4. Tenant resolution funziona in tutti i moduli

## PATTERN DI SOSTITUZIONE

**DA**:
```python
def metodo(self, tenant_or_id, ...):
    if TENANT_AVAILABLE and hasattr(tenant_or_id, 'tenant_id'):
        tenant = tenant_or_id
        resolved_tenant_id = tenant.tenant_id
    else:
        resolved_tenant_id = self._resolve_tenant_id(tenant_or_id)
```

**A**:
```python
def metodo(self, tenant: 'Tenant', ...):
    if not tenant or not hasattr(tenant, 'tenant_id'):
        raise ValueError("❌ ERRORE: Deve essere passato un oggetto Tenant valido!")
    
    # Usa direttamente i dati del tenant
    resolved_tenant_id = tenant.tenant_id
```

---

**AGGIORNATO**: 2025-08-31 22:59:30
**COMPLETAMENTO**: 2/6 file completati, 1 in corso
