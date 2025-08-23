-- =====================================================================
-- CREAZIONE TABELLA TAG.prompts PER GESTIONE PROMPT MULTI-TENANT
-- Autore: Sistema di Classificazione AI
-- Data: 2025-08-21
-- Descrizione: Tabella per memorizzare prompt di ML e LLM per ogni tenant
-- =====================================================================

USE TAG;

-- Tabella principale per i prompt
CREATE TABLE IF NOT EXISTS prompts (
    -- Chiavi primarie e identificatori
    id INT AUTO_INCREMENT PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    tenant_name VARCHAR(255) NOT NULL,
    
    -- Tipo di engine e prompt
    engine ENUM('ML', 'LLM', 'FINETUNING') NOT NULL,
    prompt_type ENUM('SYSTEM', 'USER', 'TEMPLATE', 'SPECIALIZED') NOT NULL,
    
    -- Nome identificativo del prompt
    prompt_name VARCHAR(255) NOT NULL,
    
    -- Contenuto del prompt (supporta placeholder per variabili dinamiche)
    prompt_content TEXT NOT NULL,
    
    -- Metadati per variabili dinamiche
    dynamic_variables JSON DEFAULT NULL COMMENT 'Variabili che vengono sostituite a runtime',
    
    -- Configurazione aggiuntiva
    config_parameters JSON DEFAULT NULL COMMENT 'Parametri di configurazione specifici',
    
    -- Versioning e storico
    version INT DEFAULT 1 COMMENT 'Versione del prompt per tracking modifiche',
    is_active BOOLEAN DEFAULT TRUE COMMENT 'Se questo prompt è attivo',
    
    -- Metadata
    description TEXT DEFAULT NULL COMMENT 'Descrizione del prompt e del suo utilizzo',
    created_by VARCHAR(255) DEFAULT 'system',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(255) DEFAULT 'system',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indici per performance
    INDEX idx_tenant_engine (tenant_id, engine),
    INDEX idx_tenant_name_engine (tenant_name, engine),
    INDEX idx_prompt_type (prompt_type),
    INDEX idx_active (is_active),
    
    -- Chiave univoca per tenant + engine + tipo + nome
    UNIQUE KEY unique_tenant_prompt (tenant_id, engine, prompt_type, prompt_name, version)
) 
ENGINE=InnoDB 
DEFAULT CHARSET=utf8mb4 
COLLATE=utf8mb4_unicode_ci
COMMENT='Tabella per gestione prompt ML/LLM multi-tenant';

-- Tabella per storico modifiche prompt
CREATE TABLE IF NOT EXISTS prompt_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    prompt_id INT NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    
    -- Snapshot completo del prompt
    old_content TEXT,
    new_content TEXT,
    
    -- Dettagli modifica
    change_type ENUM('CREATE', 'UPDATE', 'DELETE', 'ACTIVATE', 'DEACTIVATE') NOT NULL,
    change_reason TEXT DEFAULT NULL,
    
    -- Chi ha fatto la modifica
    changed_by VARCHAR(255) NOT NULL,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Riferimenti
    FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE CASCADE,
    INDEX idx_prompt_history (prompt_id),
    INDEX idx_tenant_history (tenant_id),
    INDEX idx_change_date (changed_at)
) 
ENGINE=InnoDB 
DEFAULT CHARSET=utf8mb4 
COLLATE=utf8mb4_unicode_ci
COMMENT='Storico modifiche prompt per audit e rollback';

-- Vista per prompt attivi (per query semplificate)
CREATE OR REPLACE VIEW active_prompts AS
SELECT 
    p.*,
    t.nome as tenant_display_name
FROM prompts p
JOIN tenants t ON p.tenant_id = t.tenant_id
WHERE p.is_active = TRUE;

-- Commenti per documentazione
ALTER TABLE prompts COMMENT = 'Tabella principale per memorizzazione prompt ML/LLM multi-tenant con supporto variabili dinamiche';
ALTER TABLE prompt_history COMMENT = 'Storico completo delle modifiche ai prompt per audit e possibile rollback';
