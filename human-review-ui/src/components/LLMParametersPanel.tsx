/**
 * ============================================================================
 * LLM Parameters Panel Component
 * ============================================================================
 * 
 * Autore: Valerio Bignardi
 * Data creazione: 2025-08-31
 * Ultima modifica: 2025-08-31
 * 
 * Descrizione:
 *     Componente React per configurazione parametri LLM con slider
 *     interattivi, validazione real-time e feedback visivo.
 *     Supporta vincoli specifici per modello e salvataggio automatico.
 * 
 * Props:
 *     - tenantId: ID del tenant
 *     - selectedModel: Modello LLM selezionato
 *     - initialParameters: Parametri iniziali
 *     - onParametersChange: Callback per cambio parametri
 *     - autoSave: Flag per salvataggio automatico
 * 
 * ============================================================================
 */

import React, { useState, useEffect, useCallback } from 'react';
import { LLMModel, LLMParameters, ValidationResult, LLMConfigService } from '../services/llmConfigService';
import BatchProcessingConfigPanel from './BatchProcessingConfigPanel';

interface LLMParametersPanelProps {
  tenantId: string;
  selectedModel: LLMModel | null;
  initialParameters?: Partial<LLMParameters>;
  onParametersChange?: (parameters: Partial<LLMParameters>) => void;
  autoSave?: boolean;
  disabled?: boolean;
}

/**
 * Componente per configurazione parametri LLM con slider
 * 
 * Scopo:
 *     Fornisce interfaccia user-friendly per configurazione parametri
 *     LLM con validazione real-time e vincoli specifici per modello
 *     
 * Props:
 *     tenantId: ID tenant
 *     selectedModel: Modello selezionato
 *     initialParameters: Parametri iniziali
 *     onParametersChange: Callback cambio parametri
 *     autoSave: Flag salvataggio automatico
 *     disabled: Flag disabilitazione
 *     
 * Data ultima modifica: 2025-08-31
 */
const LLMParametersPanel: React.FC<LLMParametersPanelProps> = ({
  tenantId,
  selectedModel,
  initialParameters,
  onParametersChange,
  autoSave = false,
  disabled = false
}) => {
  const [parameters, setParameters] = useState<Partial<LLMParameters>>({
    tokenization: {
      max_tokens: 8000,
      model_name: 'cl100k_base',
      truncation_strategy: 'start'
    },
    generation: {
      max_tokens: 150,
      temperature: 0.1,
      top_k: 40,
      top_p: 0.9,
      repeat_penalty: 1.1
    },
    connection: {
      timeout: 300,
      url: 'http://localhost:11434'
    },
    openai: {
      parallel_calls_max: 200
    }
  });

  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'success' | 'error' | null>(null);
  const [lastSaved, setLastSaved] = useState<string | null>(null);

  /**
   * Inizializza parametri dal prop initialParameters
   * 
   * Data ultima modifica: 2025-08-31
   */
  useEffect(() => {
    if (initialParameters) {
      setParameters(prev => ({
        ...prev,
        ...initialParameters
      }));
    }
  }, [initialParameters]);

  // Clamp automatico quando cambia il modello selezionato
  useEffect(() => {
    if (!selectedModel) return;
    setParameters(prev => {
      const maxIn = selectedModel.max_input_tokens || 8000;
      const maxOut = selectedModel.max_output_tokens || 4000;
      const curIn = prev.tokenization?.max_tokens ?? 8000;
      const curOut = prev.generation?.max_tokens ?? 150;
      const nextIn = Math.min(curIn, maxIn);
      const nextOut = Math.min(curOut, maxOut);
      if (nextIn !== curIn || nextOut !== curOut) {
        const updatedTokenization = {
          max_tokens: nextIn,
          model_name: prev.tokenization?.model_name ?? 'cl100k_base',
          truncation_strategy: prev.tokenization?.truncation_strategy ?? 'start'
        };

        const updatedGeneration = {
          max_tokens: nextOut,
          temperature: prev.generation?.temperature ?? 0.1,
          top_k: prev.generation?.top_k ?? 40,
          top_p: prev.generation?.top_p ?? 0.9,
          repeat_penalty: prev.generation?.repeat_penalty ?? 1.1
        };

        const updated: Partial<LLMParameters> = {
          ...prev,
          tokenization: updatedTokenization,
          generation: updatedGeneration
        };
        if (onParametersChange) onParametersChange(updated);
        return updated;
      }
      return prev;
    });
  }, [selectedModel, onParametersChange]);

  /**
   * Valida parametri in real-time quando cambiano
   * 
   * Data ultima modifica: 2025-08-31
   */
  useEffect(() => {
    const validateParams = async () => {
      if (selectedModel) {
        try {
          const result = await LLMConfigService.validateParameters(
            parameters,
            selectedModel.name
          );
          setValidation(result);
        } catch (error) {
          console.error('❌ [LLMParametersPanel] Errore validazione:', error);
        }
      }
    };

    validateParams();
  }, [parameters, selectedModel]);

  /**
   * Aggiorna parametro specifico con validazione
   * 
   * Args:
   *     section: Sezione parametri (tokenization, generation, connection)
   *     key: Chiave parametro
   *     value: Nuovo valore
   *     
   * Data ultima modifica: 2025-08-31
   */
  const updateParameter = useCallback((section: string, key: string, value: any) => {
    setParameters(prev => {
      const updated = {
        ...prev,
        [section]: {
          ...prev[section as keyof LLMParameters],
          [key]: value
        }
      };

      // Notifica cambio parametri
      if (onParametersChange) {
        onParametersChange(updated);
      }

      // Auto-save se abilitato
      if (autoSave) {
        handleSave(updated);
      }

      return updated;
    });
  }, [onParametersChange, autoSave]);

  /**
   * Salva parametri sul backend
   * 
   * Args:
   *     paramsToSave: Parametri da salvare (default: parametri correnti)
   *     
   * Data ultima modifica: 2025-08-31
   */
  const handleSave = async (paramsToSave?: Partial<LLMParameters>) => {
    try {
      setSaving(true);
      setSaveStatus(null);

      const result = await LLMConfigService.updateTenantParameters(
        tenantId,
        paramsToSave || parameters,
        selectedModel?.name
      );

      setSaveStatus('success');
      setLastSaved(new Date().toLocaleTimeString());
      
    } catch (error: any) {
      setSaveStatus('error');
      console.error('❌ [LLMParametersPanel] Errore salvataggio:', error);
    } finally {
      setSaving(false);
    }
  };

  /**
   * Reset parametri ai valori default
   * 
   * Data ultima modifica: 2025-08-31
   */
  const handleReset = async () => {
    try {
      setSaving(true);
      const result = await LLMConfigService.resetTenantParameters(tenantId);
      
      if (result.default_parameters) {
        setParameters(result.default_parameters);
        if (onParametersChange) {
          onParametersChange(result.default_parameters);
        }
      }
      
      setSaveStatus('success');
      setLastSaved(new Date().toLocaleTimeString());
      
    } catch (error: any) {
      setSaveStatus('error');
      console.error('❌ [LLMParametersPanel] Errore reset:', error);
    } finally {
      setSaving(false);
    }
  };

  /**
   * Calcola range max tokens basato su modello selezionato
   * 
   * Returns:
   *     Numero massimo token permessi
   *     
   * Data ultima modifica: 2025-08-31
   */
  const getMaxTokensLimit = (): number => {
    return selectedModel?.max_input_tokens || 8000;
  };

  return (
    <div className="llm-parameters-panel">
      <div className="panel-header">
        <h3>⚙️ Configurazione Parametri LLM</h3>
        <div className="header-actions">
          {!autoSave && (
            <button
              onClick={() => handleSave()}
              disabled={disabled || saving || (validation ? !validation.valid : false)}
              className="save-btn"
            >
              {saving ? '💾 Salvando...' : '💾 Salva'}
            </button>
          )}
          
          <button
            onClick={handleReset}
            disabled={disabled || saving}
            className="reset-btn"
          >
            🔄 Reset Default
          </button>
        </div>
      </div>

      {saveStatus && (
        <div className={`save-status ${saveStatus}`}>
          {saveStatus === 'success' ? (
            <span>✅ Salvato con successo {lastSaved && `alle ${lastSaved}`}</span>
          ) : (
            <span>❌ Errore durante il salvataggio</span>
          )}
        </div>
      )}

      {validation && validation.errors.length > 0 && (
        <div className="validation-errors">
          <h4>❌ Errori Validazione:</h4>
          <ul>
            {validation.errors.map((error, index) => (
              <li key={index}>{error}</li>
            ))}
          </ul>
        </div>
      )}

      {validation && validation.warnings.length > 0 && (
        <div className="validation-warnings">
          <h4>⚠️ Avvertimenti:</h4>
          <ul>
            {validation.warnings.map((warning, index) => (
              <li key={index}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="parameters-sections">
        {/* Sezione Tokenization */}
        <div className="param-section">
          <h4>📝 Tokenization</h4>
          
          <div className="param-control">
            <label>
              Max Input Tokens: <strong>{parameters.tokenization?.max_tokens}</strong>
              <span className="limit-info">
                (Limite modello: {getMaxTokensLimit().toLocaleString()})
              </span>
            </label>
            <input
              type="range"
              min="100"
              max={getMaxTokensLimit()}
              step="100"
              value={parameters.tokenization?.max_tokens || 8000}
              onChange={(e) => updateParameter('tokenization', 'max_tokens', parseInt(e.target.value))}
              disabled={disabled}
              className="slider"
            />
            <div className="range-labels">
              <span>100</span>
              <span>{getMaxTokensLimit().toLocaleString()}</span>
            </div>
          </div>
        </div>

        {/* Sezione Generation */}
        <div className="param-section">
          <h4>🎯 Generation</h4>
          
          <div className="param-control">
            <label>
              Max Output Tokens: <strong>{parameters.generation?.max_tokens}</strong>
            </label>
            <input
              type="range"
              min="50"
              max={selectedModel?.max_output_tokens || 2000}
              step="25"
              value={parameters.generation?.max_tokens || 150}
              onChange={(e) => updateParameter('generation', 'max_tokens', parseInt(e.target.value))}
              disabled={disabled}
              className="slider"
            />
            <div className="range-labels">
              <span>50</span>
              <span>{(selectedModel?.max_output_tokens || 2000).toLocaleString()}</span>
            </div>
          </div>

          {/* 🆕 Nasconde temperature per GPT-5 (non supportato) */}
          {selectedModel?.name?.toLowerCase() !== 'gpt-5' && (
            <div className="param-control">
              <label>
                Temperature: <strong>{(parameters.generation?.temperature || 0.1).toFixed(2)}</strong>
                <span className="param-desc">(Creatività: 0=rigido, 2=creativo)</span>
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.01"
                value={parameters.generation?.temperature || 0.1}
                onChange={(e) => updateParameter('generation', 'temperature', parseFloat(e.target.value))}
                disabled={disabled}
                className="slider"
              />
              <div className="range-labels">
                <span>0.0</span>
                <span>2.0</span>
              </div>
            </div>
          )}
          
          {/* 🆕 Mostra avviso GPT-5 */}
          {selectedModel?.name?.toLowerCase() === 'gpt-5' && (
            <div className="info-box" style={{ marginTop: '10px', padding: '12px', backgroundColor: '#e3f2fd', borderLeft: '4px solid #2196f3', borderRadius: '4px' }}>
              <strong>ℹ️ GPT-5:</strong> Questo modello non supporta temperature, top_p, top_k e repeat_penalty. I parametri vengono gestiti automaticamente dall'API.
            </div>
          )}

          {/* 🆕 Nasconde top_k per GPT-5 */}
          {selectedModel?.name?.toLowerCase() !== 'gpt-5' && (
            <div className="param-control">
              <label>
                Top K: <strong>{parameters.generation?.top_k}</strong>
                <span className="param-desc">(Token considerati)</span>
              </label>
              <input
                type="range"
                min="1"
                max="100"
                step="1"
                value={parameters.generation?.top_k || 40}
                onChange={(e) => updateParameter('generation', 'top_k', parseInt(e.target.value))}
                disabled={disabled}
                className="slider"
              />
              <div className="range-labels">
                <span>1</span>
                <span>100</span>
              </div>
            </div>
          )}

          {/* 🆕 Nasconde top_p per GPT-5 */}
          {selectedModel?.name?.toLowerCase() !== 'gpt-5' && (
            <div className="param-control">
              <label>
                Top P: <strong>{(parameters.generation?.top_p || 0.9).toFixed(2)}</strong>
                <span className="param-desc">(Soglia probabilità cumulativa)</span>
              </label>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.01"
                value={parameters.generation?.top_p || 0.9}
                onChange={(e) => updateParameter('generation', 'top_p', parseFloat(e.target.value))}
                disabled={disabled}
                className="slider"
              />
              <div className="range-labels">
                <span>0.1</span>
                <span>1.0</span>
              </div>
            </div>
          )}

          {/* 🆕 Nasconde repeat_penalty per GPT-5 */}
          {selectedModel?.name?.toLowerCase() !== 'gpt-5' && (
            <div className="param-control">
              <label>
                Repeat Penalty: <strong>{(parameters.generation?.repeat_penalty || 1.1).toFixed(2)}</strong>
                <span className="param-desc">(Penalità ripetizione)</span>
              </label>
              <input
                type="range"
                min="0.8"
                max="1.5"
                step="0.01"
                value={parameters.generation?.repeat_penalty || 1.1}
                onChange={(e) => updateParameter('generation', 'repeat_penalty', parseFloat(e.target.value))}
                disabled={disabled}
                className="slider"
              />
              <div className="range-labels">
                <span>0.8</span>
                <span>1.5</span>
              </div>
            </div>
          )}
        </div>

        {/* Sezione Connection */}
        <div className="param-section">
          <h4>🔗 Connection</h4>
          
          <div className="param-control">
            <label>
              Timeout: <strong>{parameters.connection?.timeout}s</strong>
            </label>
            <input
              type="range"
              min="30"
              max="600"
              step="30"
              value={parameters.connection?.timeout || 300}
              onChange={(e) => updateParameter('connection', 'timeout', parseInt(e.target.value))}
              disabled={disabled}
              className="slider"
            />
            <div className="range-labels">
              <span>30s</span>
              <span>600s</span>
            </div>
          </div>
        </div>

        {/* Sezione OpenAI - Solo per modelli OpenAI */}
        {selectedModel?.provider === 'openai' && (
          <div className="param-section openai-section">
            <h4>🤖 OpenAI Configurazione</h4>
            
            <div className="param-control">
              <label>
                Max Chiamate Parallele: <strong>{parameters.openai?.parallel_calls_max || 200}</strong>
                <span className="param-desc">(Numero massimo di chiamate API simultanee)</span>
              </label>
              <input
                type="range"
                min="1"
                max="500"
                step="1"
                value={parameters.openai?.parallel_calls_max || 200}
                onChange={(e) => updateParameter('openai', 'parallel_calls_max', parseInt(e.target.value))}
                disabled={disabled}
                className="slider"
              />
              <div className="range-labels">
                <span>1</span>
                <span>500</span>
              </div>
            </div>
            
            {selectedModel.parallel_calls_max && (
              <div className="model-limit-info">
                <span className="info-text">
                  ℹ️ Limite modello: {selectedModel.parallel_calls_max} chiamate parallele
                </span>
              </div>
            )}
          </div>
        )}

        {/* Sezione Batch Processing - Solo per modelli OpenAI */}
        {selectedModel?.provider === 'openai' && (
          <div className="batch-processing-section">
            <BatchProcessingConfigPanel
              tenantId={tenantId}
              autoSave={autoSave}
              disabled={disabled}
              onConfigChange={(batchConfig) => {
                console.log('🔄 [LLMParametersPanel] Batch config aggiornata:', batchConfig);
              }}
            />
          </div>
        )}
      </div>

      {/* Anteprima JSON configurazione */}
      <div className="config-preview">
        <h4>📋 Anteprima Configurazione</h4>
        <pre className="json-preview">
          {JSON.stringify(parameters, null, 2)}
        </pre>
      </div>

      <style>{`
        .llm-parameters-panel {
          background: #ffffff;
          border: 1px solid #dee2e6;
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 20px;
        }

        .llm-parameters-panel .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          padding-bottom: 15px;
          border-bottom: 1px solid #e9ecef;
        }

        .llm-parameters-panel .panel-header h3 {
          margin: 0;
          color: #495057;
          font-size: 1.2em;
        }

        .llm-parameters-panel .header-actions {
          display: flex;
          gap: 10px;
        }

        .llm-parameters-panel .save-btn, .llm-parameters-panel .reset-btn {
          padding: 8px 16px;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 0.9em;
          transition: all 0.2s;
        }

        .llm-parameters-panel .save-btn {
          background: #007bff;
          color: white;
        }

        .llm-parameters-panel .save-btn:hover:not(:disabled) {
          background: #0056b3;
        }

        .llm-parameters-panel .save-btn:disabled {
          background: #6c757d;
          cursor: not-allowed;
        }

        .llm-parameters-panel .reset-btn {
          background: #6c757d;
          color: white;
        }

        .llm-parameters-panel .reset-btn:hover:not(:disabled) {
          background: #545b62;
        }

        .llm-parameters-panel .save-status {
          padding: 10px;
          border-radius: 4px;
          margin-bottom: 15px;
          font-weight: 500;
        }

        .llm-parameters-panel .save-status.success {
          background: #d4edda;
          color: #155724;
          border: 1px solid #c3e6cb;
        }

        .llm-parameters-panel .save-status.error {
          background: #f8d7da;
          color: #721c24;
          border: 1px solid #f5c6cb;
        }

        .llm-parameters-panel .validation-errors, .llm-parameters-panel .validation-warnings {
          margin-bottom: 15px;
          padding: 12px;
          border-radius: 4px;
        }

        .llm-parameters-panel .validation-errors {
          background: #f8d7da;
          border: 1px solid #f5c6cb;
        }

        .llm-parameters-panel .validation-warnings {
          background: #fff3cd;
          border: 1px solid #ffeaa7;
        }

        .llm-parameters-panel .validation-errors h4, .llm-parameters-panel .validation-warnings h4 {
          margin: 0 0 8px 0;
          font-size: 1em;
        }

        .llm-parameters-panel .validation-errors ul, .llm-parameters-panel .validation-warnings ul {
          margin: 0;
          padding-left: 20px;
        }

        .llm-parameters-panel .parameters-sections {
          display: grid;
          gap: 20px;
        }

        .llm-parameters-panel .param-section {
          background: #f8f9fa;
          border: 1px solid #e9ecef;
          border-radius: 6px;
          padding: 16px;
        }

        .llm-parameters-panel .param-section.openai-section {
          background: #f0f8ff;
          border-color: #007bff;
          border-width: 1px;
          box-shadow: 0 2px 4px rgba(0, 123, 255, 0.1);
        }

        .llm-parameters-panel .openai-section h4 {
          color: #007bff;
        }

        .llm-parameters-panel .model-limit-info {
          margin-top: 8px;
          padding: 6px 10px;
          background: #e7f3ff;
          border: 1px solid #b3d9ff;
          border-radius: 4px;
        }

        .llm-parameters-panel .info-text {
          font-size: 0.85em;
          color: #0056b3;
        }

        .llm-parameters-panel .param-section h4 {
          margin: 0 0 15px 0;
          color: #495057;
          font-size: 1.1em;
        }

        .llm-parameters-panel .param-control {
          margin-bottom: 20px;
        }

        .llm-parameters-panel .param-control:last-child {
          margin-bottom: 0;
        }

        .llm-parameters-panel .param-control label {
          display: block;
          margin-bottom: 8px;
          font-weight: 500;
          color: #495057;
        }

        .llm-parameters-panel .limit-info, .llm-parameters-panel .param-desc {
          font-size: 0.85em;
          color: #6c757d;
          font-weight: normal;
          margin-left: 8px;
        }

        .llm-parameters-panel .slider {
          width: 100%;
          height: 6px;
          border-radius: 3px;
          background: #dee2e6;
          outline: none;
          margin-bottom: 8px;
        }

        .llm-parameters-panel .slider:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .llm-parameters-panel .range-labels {
          display: flex;
          justify-content: space-between;
          font-size: 0.8em;
          color: #6c757d;
        }

        .llm-parameters-panel .config-preview {
          margin-top: 20px;
          padding: 15px;
          background: #f8f9fa;
          border: 1px solid #e9ecef;
          border-radius: 6px;
        }

        .llm-parameters-panel .config-preview h4 {
          margin: 0 0 10px 0;
          color: #495057;
        }

        .llm-parameters-panel .json-preview {
          background: #2d3748;
          color: #e2e8f0;
          padding: 15px;
          border-radius: 4px;
          font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
          font-size: 0.85em;
          overflow-x: auto;
          margin: 0;
        }
      `}</style>
    </div>
  );
};

export default LLMParametersPanel;
