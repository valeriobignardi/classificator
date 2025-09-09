/**
 * ============================================================================
 * Batch Processing Configuration Panel Component
 * ============================================================================
 * 
 * Autore: Valerio Bignardi
 * Data creazione: 2025-09-07
 * Ultima modifica: 2025-09-07
 * 
 * Descrizione:
 *     Componente React per configurazione parametri batch processing
 *     con slider interattivi, validazione real-time e salvataggio
 *     automatico nel database MySQL.
 * 
 * Props:
 *     - tenantId: ID del tenant
 *     - onConfigChange: Callback per cambio configurazione
 *     - autoSave: Flag per salvataggio automatico
 * 
 * ============================================================================
 */

import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

interface BatchProcessingConfig {
  classification_batch_size: number;
  max_parallel_calls: number;
  source?: string;
  updated_at?: string;
}

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  suggestions: string[];
}

interface BatchProcessingConfigPanelProps {
  tenantId: string;
  onConfigChange?: (config: BatchProcessingConfig) => void;
  autoSave?: boolean;
  disabled?: boolean;
}

/**
 * Componente per configurazione batch processing
 * 
 * Scopo:
 *     Fornisce interfaccia user-friendly per configurazione parametri
 *     batch processing con validazione real-time e persistenza database
 *     
 * Props:
 *     tenantId: ID tenant
 *     onConfigChange: Callback cambio configurazione
 *     autoSave: Flag salvataggio automatico
 *     disabled: Flag disabilitazione
 *     
 * Data ultima modifica: 2025-09-07
 */
const BatchProcessingConfigPanel: React.FC<BatchProcessingConfigPanelProps> = ({
  tenantId,
  onConfigChange,
  autoSave = true,
  disabled = false
}) => {
  // Stati componente
  const [config, setConfig] = useState<BatchProcessingConfig>({
    classification_batch_size: 32,
    max_parallel_calls: 200
  });
  
  const [validation, setValidation] = useState<ValidationResult>({
    valid: true,
    errors: [],
    warnings: [],
    suggestions: []
  });
  
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastSaved, setLastSaved] = useState<string | null>(null);
  
  // 🚨 FIX LOOP INFINITO: Stati per controllo retry e prevenzione loop
  const [retryCount, setRetryCount] = useState(0);
  const [isRetrying, setIsRetrying] = useState(false);
  const [lastLoadAttempt, setLastLoadAttempt] = useState<number>(0);
  const MAX_RETRIES = 3;
  const RETRY_DELAY = 5000; // 5 secondi tra retry

  /**
   * Carica configurazione batch processing per tenant
   * 🚨 FIX LOOP INFINITO: Aggiunto controllo retry e prevenzione loop
   */
  const loadConfig = useCallback(async (forceRetry: boolean = false) => {
    const now = Date.now();
    
    // 🚨 PREVENZIONE LOOP: Controlla se è troppo presto per un nuovo tentativo
    if (!forceRetry && (now - lastLoadAttempt) < 2000) {
      console.log('⚠️ [BatchConfig] Tentativo troppo frequente, skip per prevenire loop');
      return;
    }
    
    // 🚨 PREVENZIONE LOOP: Controlla limite retry
    if (!forceRetry && retryCount >= MAX_RETRIES && !isRetrying) {
      console.error('❌ [BatchConfig] Limite retry raggiunto, uso configurazione default');
      setError('Impossibile caricare configurazione batch, uso valori default');
      setLoading(false);
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      setLastLoadAttempt(now);
      
      console.log(`🔄 [BatchConfig] Tentativo caricamento ${retryCount + 1}/${MAX_RETRIES} per tenant ${tenantId}`);
      
      const response = await axios.get(`/api/ai-config/${tenantId}/batch-config`, {
        timeout: 10000 // 10 secondi timeout
      });
      
      if (response.data.success) {
        const loadedConfig = response.data.batch_config;
        setConfig(loadedConfig);
        setLastSaved(loadedConfig.updated_at);
        
        // Reset retry counter su successo
        setRetryCount(0);
        setIsRetrying(false);
        
        // Notifica parent component SOLO se i valori sono effettivamente cambiati
        if (onConfigChange) {
          onConfigChange(loadedConfig);
        }
        
        console.log('✅ [BatchConfig] Configurazione caricata:', loadedConfig);
      } else {
        throw new Error(response.data.error || 'Errore caricamento configurazione');
      }
      
    } catch (err: any) {
      console.error('❌ [BatchConfig] Errore caricamento:', err);
      
      // 🚨 GESTIONE RETRY CONTROLLATA
      if (retryCount < MAX_RETRIES && !isRetrying) {
        setRetryCount(prev => prev + 1);
        setError(`Tentativo ${retryCount + 1}/${MAX_RETRIES} fallito. Nuovo tentativo tra ${RETRY_DELAY/1000}s...`);
        
        setIsRetrying(true);
        setTimeout(() => {
          setIsRetrying(false);
          loadConfig(true); // Force retry
        }, RETRY_DELAY);
      } else {
        setError(err.message || 'Errore caricamento configurazione batch');
        setLoading(false);
      }
    } finally {
      if (retryCount >= MAX_RETRIES || !isRetrying) {
        setLoading(false);
      }
    }
  }, [tenantId, retryCount, isRetrying, lastLoadAttempt]); // 🚨 DIPENDENZE CONTROLLATE

  /**
   * Valida configurazione senza salvarla
   */
  const validateConfig = useCallback(async (configToValidate: BatchProcessingConfig) => {
    try {
      const response = await axios.post(
        `/api/ai-config/${tenantId}/batch-config/validate`,
        configToValidate
      );
      
      if (response.data.success) {
        setValidation(response.data);
      }
      
    } catch (err: any) {
      console.error('❌ [BatchConfig] Errore validazione:', err);
      setValidation({
        valid: false,
        errors: ['Errore validazione'],
        warnings: [],
        suggestions: []
      });
    }
  }, [tenantId]);

  /**
   * Salva configurazione nel database
   */
  const saveConfig = useCallback(async (configToSave: BatchProcessingConfig) => {
    if (!autoSave || disabled) return;
    
    try {
      setSaving(true);
      setError(null);
      
      const response = await axios.post(
        `/api/ai-config/${tenantId}/batch-config`,
        configToSave
      );
      
      if (response.data.success) {
        setLastSaved(response.data.timestamp);
        console.log('✅ [BatchConfig] Configurazione salvata');
      } else {
        throw new Error(response.data.error || 'Errore salvataggio');
      }
      
    } catch (err: any) {
      setError(err.message || 'Errore salvataggio configurazione');
      console.error('❌ [BatchConfig] Errore salvataggio:', err);
    } finally {
      setSaving(false);
    }
  }, [tenantId, autoSave, disabled]);

  /**
   * Aggiorna parametro specifico con validazione e salvataggio
   */
  const updateParameter = useCallback((
    parameter: keyof BatchProcessingConfig,
    value: number
  ) => {
    const newConfig = { ...config, [parameter]: value };
    
    setConfig(newConfig);
    
    // Validazione real-time
    validateConfig(newConfig);
    
    // Salvataggio automatico (debounced)
    setTimeout(() => {
      saveConfig(newConfig);
    }, 1000);
    
    // Notifica parent component
    if (onConfigChange) {
      onConfigChange(newConfig);
    }
    
  }, [config, validateConfig, saveConfig, onConfigChange]);

  /**
   * Salvataggio manuale
   */
  const handleManualSave = useCallback(async () => {
    await saveConfig(config);
  }, [config, saveConfig]);

  // Carica configurazione al mount
  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  // Rendering condizionale per stati di caricamento/errore
  if (loading) {
    return (
      <div className="batch-config-panel loading">
        <h3>🔄 Caricamento Configurazione Batch...</h3>
      </div>
    );
  }

  if (error && !config.classification_batch_size) {
    return (
      <div className="batch-config-panel error">
        <h3>❌ Errore Configurazione Batch</h3>
        <p>{error}</p>
        <button onClick={() => loadConfig()} className="retry-button">
          🔄 Riprova
        </button>
      </div>
    );
  }

  return (
    <div className="batch-config-panel">
      <h3>⚡ Configurazione Batch Processing - {tenantId}</h3>
      
      {/* Indicatore stato */}
      <div className="config-status">
        <span className={`status-indicator ${config.source === 'database' ? 'database' : 'default'}`}>
          📊 Fonte: {config.source === 'database' ? 'Database' : 'Default'}
        </span>
        {lastSaved && (
          <span className="last-saved">
            💾 Ultimo salvataggio: {new Date(lastSaved).toLocaleString()}
          </span>
        )}
        {saving && <span className="saving">💾 Salvataggio...</span>}
      </div>

      {/* Errori/Warning */}
      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}
      
      {validation.errors.length > 0 && (
        <div className="validation-errors">
          {validation.errors.map((err, idx) => (
            <div key={idx} className="error">❌ {err}</div>
          ))}
        </div>
      )}
      
      {validation.warnings.length > 0 && (
        <div className="validation-warnings">
          {validation.warnings.map((warn, idx) => (
            <div key={idx} className="warning">⚠️ {warn}</div>
          ))}
        </div>
      )}

      {/* Controlli parametri */}
      <div className="parameters-section">
        
        {/* Classification Batch Size */}
        <div className="param-control">
          <label>
            📦 Batch Size Classificazione: <strong>{config.classification_batch_size}</strong>
            <span className="param-desc">(Numero conversazioni per batch)</span>
          </label>
          <input
            type="range"
            min="1"
            max="100"
            step="1"
            value={config.classification_batch_size}
            onChange={(e) => updateParameter('classification_batch_size', parseInt(e.target.value))}
            disabled={disabled}
            className="slider"
          />
          <div className="range-labels">
            <span>1</span>
            <span>50</span>
            <span>100</span>
          </div>
        </div>

        {/* Max Parallel Calls */}
        <div className="param-control">
          <label>
            ⚡ Chiamate Parallele Max: <strong>{config.max_parallel_calls}</strong>
            <span className="param-desc">(Limite simultaneità API)</span>
          </label>
          <input
            type="range"
            min="1"
            max="500"
            step="5"
            value={config.max_parallel_calls}
            onChange={(e) => updateParameter('max_parallel_calls', parseInt(e.target.value))}
            disabled={disabled}
            className="slider"
          />
          <div className="range-labels">
            <span>1</span>
            <span>200</span>
            <span>500</span>
          </div>
        </div>
      </div>

      {/* Suggerimenti */}
      {validation.suggestions.length > 0 && (
        <div className="validation-suggestions">
          <h4>💡 Suggerimenti:</h4>
          {validation.suggestions.map((suggestion, idx) => (
            <div key={idx} className="suggestion">💡 {suggestion}</div>
          ))}
        </div>
      )}

      {/* Anteprima configurazione */}
      <div className="config-preview">
        <h4>📋 Configurazione Corrente</h4>
        <pre className="json-preview">
          {JSON.stringify(config, null, 2)}
        </pre>
      </div>

      {/* Controlli manuali */}
      {!autoSave && (
        <div className="manual-controls">
          <button 
            onClick={handleManualSave}
            disabled={disabled || saving || !validation.valid}
            className="save-button"
          >
            {saving ? '💾 Salvataggio...' : '💾 Salva Configurazione'}
          </button>
        </div>
      )}

      <style>{`
        .batch-config-panel {
          background: #ffffff;
          border: 1px solid #dee2e6;
          border-radius: 8px;
          padding: 20px;
          margin: 20px 0;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .config-status {
          display: flex;
          gap: 15px;
          margin-bottom: 15px;
          padding: 10px;
          background: #f8f9fa;
          border-radius: 6px;
          font-size: 0.9em;
        }

        .status-indicator.database {
          color: #28a745;
          font-weight: bold;
        }

        .status-indicator:not(.database) {
          color: #6c757d;
        }

        .last-saved, .saving {
          color: #6c757d;
          font-size: 0.85em;
        }

        .error-message, .validation-errors .error {
          background: #f8d7da;
          color: #721c24;
          padding: 8px 12px;
          border-radius: 4px;
          margin: 10px 0;
        }

        .validation-warnings .warning {
          background: #fff3cd;
          color: #856404;
          padding: 8px 12px;
          border-radius: 4px;
          margin: 5px 0;
        }

        .validation-suggestions {
          background: #d1ecf1;
          padding: 15px;
          border-radius: 6px;
          margin: 15px 0;
        }

        .validation-suggestions h4 {
          margin: 0 0 10px 0;
          color: #0c5460;
        }

        .suggestion {
          color: #0c5460;
          margin: 5px 0;
        }

        .parameters-section {
          margin: 20px 0;
        }

        .param-control {
          margin: 25px 0;
          padding: 15px;
          border: 1px solid #e9ecef;
          border-radius: 6px;
          background: #f8f9fa;
        }

        .param-control label {
          display: block;
          margin-bottom: 10px;
          font-weight: 600;
          color: #495057;
        }

        .param-desc {
          display: block;
          font-weight: normal;
          font-size: 0.85em;
          color: #6c757d;
          margin-top: 4px;
        }

        .slider {
          width: 100%;
          height: 6px;
          border-radius: 3px;
          background: #dee2e6;
          outline: none;
          margin: 15px 0 10px 0;
        }

        .slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #007bff;
          cursor: pointer;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        .slider::-moz-range-thumb {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #007bff;
          cursor: pointer;
          border: none;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        .range-labels {
          display: flex;
          justify-content: space-between;
          font-size: 0.8em;
          color: #6c757d;
        }

        .config-preview {
          margin: 20px 0;
          padding: 15px;
          background: #f8f9fa;
          border-radius: 6px;
        }

        .config-preview h4 {
          margin: 0 0 10px 0;
          color: #495057;
        }

        .json-preview {
          background: #2d3748;
          color: #e2e8f0;
          padding: 15px;
          border-radius: 4px;
          font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
          font-size: 0.85em;
          overflow-x: auto;
          white-space: pre;
        }

        .manual-controls {
          margin-top: 20px;
          text-align: center;
        }

        .save-button {
          background: #28a745;
          color: white;
          border: none;
          padding: 12px 24px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 1em;
          font-weight: 600;
        }

        .save-button:disabled {
          background: #6c757d;
          cursor: not-allowed;
        }

        .save-button:hover:not(:disabled) {
          background: #218838;
        }

        .retry-button {
          background: #007bff;
          color: white;
          border: none;
          padding: 10px 20px;
          border-radius: 4px;
          cursor: pointer;
          margin-top: 10px;
        }

        .loading, .error {
          text-align: center;
          padding: 40px;
        }
      `}</style>
    </div>
  );
};

export default BatchProcessingConfigPanel;
