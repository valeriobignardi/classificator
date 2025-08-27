-- 
-- Script per creare la tabella clustering_test_results
-- Autore: Sistema di Classificazione  
-- Data: 2025-08-27
--

USE common;

CREATE TABLE IF NOT EXISTS clustering_test_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL COMMENT 'UUID del tenant',
    version_number INT NOT NULL COMMENT 'Numero progressivo versione per tenant',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Timestamp creazione test',
    execution_time FLOAT COMMENT 'Tempo esecuzione in secondi',
    
    -- Risultati clustering completi (JSON)
    results_json JSON NOT NULL COMMENT 'Risultati completi del clustering test',
    
    -- Parametri utilizzati per il clustering (JSON)  
    parameters_json JSON NOT NULL COMMENT 'Parametri HDBSCAN e UMAP utilizzati',
    
    -- Statistiche estratte per query rapide e filtri
    n_clusters INT COMMENT 'Numero di cluster generati',
    n_outliers INT COMMENT 'Numero di outliers',
    n_conversations INT COMMENT 'Numero totale conversazioni analizzate',
    clustering_ratio FLOAT COMMENT 'Rapporto conversazioni clusterizzate (0-1)',
    silhouette_score FLOAT COMMENT 'Punteggio silhouette per qualità clustering',
    calinski_harabasz_score FLOAT COMMENT 'Punteggio Calinski-Harabasz',
    davies_bouldin_score FLOAT COMMENT 'Punteggio Davies-Bouldin',
    
    -- Indici per performance
    INDEX idx_tenant_version (tenant_id, version_number),
    INDEX idx_tenant_created (tenant_id, created_at DESC),
    INDEX idx_created (created_at DESC),
    
    -- Constraint per unicità versione per tenant
    UNIQUE KEY unique_tenant_version (tenant_id, version_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci 
COMMENT='Storico risultati test clustering con versioning per tenant';
