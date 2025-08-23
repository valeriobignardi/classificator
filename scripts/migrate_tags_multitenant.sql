-- =====================================================================
-- MIGRAZIONE MULTI-TENANT PER TABELLA TAG.tags
-- =====================================================================
-- Autore: Sistema di Classificazione AI
-- Data: 2025-08-21
-- Descrizione: Aggiunge supporto multi-tenant alla tabella tags esistente
--              Preserva tutti i dati esistenti assegnandoli al tenant "humanitas"
-- =====================================================================

USE TAG;

-- Backup della tabella originale (per sicurezza)
CREATE TABLE IF NOT EXISTS tags_backup_20250821 AS SELECT * FROM tags;

-- Aggiunge colonne per multi-tenancy
ALTER TABLE tags 
ADD COLUMN tenant_id VARCHAR(50) NOT NULL DEFAULT 'humanitas' AFTER id,
ADD COLUMN tenant_name VARCHAR(100) NOT NULL DEFAULT 'Humanitas' AFTER tenant_id;

-- Crea indice composito per performance delle query multi-tenant
CREATE INDEX idx_tags_tenant_name ON tags(tenant_id, tag_name);

-- Aggiorna i record esistenti con il tenant predefinito
UPDATE tags 
SET 
    tenant_id = 'humanitas',
    tenant_name = 'Humanitas'
WHERE tenant_id = 'humanitas';  -- Condizione per sicurezza

-- Aggiunge constraint per evitare duplicati per tenant
ALTER TABLE tags 
ADD CONSTRAINT unique_tag_per_tenant 
UNIQUE (tenant_id, tag_name);

-- Verifica che la migrazione sia andata a buon fine
SELECT 
    COUNT(*) as total_tags,
    tenant_id,
    tenant_name,
    MIN(created_at) as first_tag,
    MAX(updated_at) as last_update
FROM tags 
GROUP BY tenant_id, tenant_name;

-- Mostra alcuni esempi di tag migrati
SELECT 
    id, 
    tenant_id, 
    tenant_name, 
    tag_name, 
    LEFT(tag_description, 50) as description_preview,
    created_at
FROM tags 
ORDER BY created_at DESC 
LIMIT 5;

SELECT '✅ Migrazione multi-tenant completata con successo!' as status;
