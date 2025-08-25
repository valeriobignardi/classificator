-- =====================================================================
-- SCHEMA TABELLA ESEMPI MULTI-TENANT
-- =====================================================================
-- Autore: Sistema di Classificazione AI
-- Data: 2025-08-25
-- Descrizione: Creazione tabella per gestione esempi multi-tenant
--              per il sistema di placeholder {{examples_text}}
-- =====================================================================

USE TAG;

-- Creazione tabella esempi
CREATE TABLE IF NOT EXISTS esempi (
    -- ID primario auto-incrementale
    id INT AUTO_INCREMENT PRIMARY KEY,
    
    -- Identificativi tenant (seguono lo stesso pattern della tabella prompts)
    tenant_id VARCHAR(255) NOT NULL,
    tenant_name VARCHAR(255) NOT NULL,
    
    -- Classificazione dell'esempio
    engine ENUM('ML', 'LLM', 'FINETUNING') NOT NULL DEFAULT 'LLM',
    esempio_type ENUM('CLASSIFICATION', 'CONVERSATION', 'TEMPLATE') NOT NULL DEFAULT 'CONVERSATION',
    esempio_name VARCHAR(255) NOT NULL,
    
    -- Contenuto dell'esempio formattato con ruoli UTENTE/ASSISTENTE
    esempio_content TEXT NOT NULL,
    
    -- Metadati dell'esempio
    description TEXT,
    categoria VARCHAR(255),  -- Per raggruppare esempi simili
    livello_difficolta ENUM('FACILE', 'MEDIO', 'DIFFICILE') DEFAULT 'MEDIO',
    
    -- Controllo versioni e stato
    version INT DEFAULT 1,
    is_active TINYINT(1) DEFAULT 1,
    
    -- Auditing
    created_by VARCHAR(255) DEFAULT 'system',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(255) DEFAULT 'system',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Indici per performance
CREATE INDEX idx_esempi_tenant_id ON esempi(tenant_id);
CREATE INDEX idx_esempi_tenant_name ON esempi(tenant_name);
CREATE INDEX idx_esempi_engine_type ON esempi(engine, esempio_type);
CREATE INDEX idx_esempi_active ON esempi(is_active);
CREATE INDEX idx_esempi_categoria ON esempi(categoria);

-- Indice composito per ricerche comuni
CREATE INDEX idx_esempi_tenant_engine_type_name ON esempi(tenant_id, engine, esempio_type, esempio_name);

COMMIT;
