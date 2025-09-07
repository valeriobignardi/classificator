/**
 * ============================================================================
 * LLM Model Selector Component
 * ============================================================================
 * 
 * Autore: Valerio Bignardi
 * Data creazione: 2025-08-31
 * Ultima modifica: 2025-08-31
 * 
 * Descrizione:
 *     Componente React per selezione modello LLM con visualizzazione
 *     delle informazioni tecniche del modello (limiti, capabilities).
 *     Supporta caricamento dinamico e aggiornamento automatico parametri.
 * 
 * Props:
 *     - tenantId: ID del tenant per cui selezionare il modello
 *     - currentModel: Modello attualmente selezionato
 *     - onModelChange: Callback per cambio modello
 *     - disabled: Flag per disabilitare il componente
 * 
 * ============================================================================
 */

import React, { useState, useEffect } from 'react';
import { LLMModel, LLMConfigService } from '../services/llmConfigService';

interface LLMModelSelectorProps {
  tenantId: string;
  currentModel?: string;
  onModelChange: (model: LLMModel) => void;
  disabled?: boolean;
}

/**
 * Componente per selezione modello LLM con info tecniche
 * 
 * Scopo:
 *     Permette selezione modello LLM per tenant con visualizzazione
 *     limiti e capabilities del modello selezionato
 *     
 * Props:
 *     tenantId: ID tenant
 *     currentModel: Modello corrente
 *     onModelChange: Callback cambio modello
 *     disabled: Flag disabilitazione
 *     
 * Data ultima modifica: 2025-08-31
 */
const LLMModelSelector: React.FC<LLMModelSelectorProps> = ({
  tenantId,
  currentModel,
  onModelChange,
  disabled = false
}) => {
  const [models, setModels] = useState<LLMModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<LLMModel | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  /**
   * Carica modelli disponibili dal backend
   * 
   * Data ultima modifica: 2025-08-31
   */
  useEffect(() => {
    const loadModels = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const availableModels = await LLMConfigService.getAvailableModels(tenantId);
        setModels(availableModels);
        
        // Seleziona modello corrente se specificato
        if (currentModel) {
          const current = availableModels.find(m => m.name === currentModel);
          if (current) {
            setSelectedModel(current);
          }
        }
        
      } catch (err: any) {
        setError(err.message || 'Errore caricamento modelli');
        console.error('❌ [LLMModelSelector] Errore caricamento modelli:', err);
      } finally {
        setLoading(false);
      }
    };

    if (tenantId) {
      loadModels();
    }
  }, [tenantId, currentModel]);

  /**
   * Gestisce cambio selezione modello
   * 
   * Args:
   *     modelName: Nome del nuovo modello selezionato
   *     
   * Data ultima modifica: 2025-08-31
   */
  const handleModelChange = async (modelName: string) => {
    try {
      const model = models.find(m => m.name === modelName);
      if (model) {
        setSelectedModel(model);
        onModelChange(model);
      }
    } catch (err: any) {
      setError(err.message || 'Errore cambio modello');
      console.error('❌ [LLMModelSelector] Errore cambio modello:', err);
    }
  };

  /**
   * Formatta dimensione context limit per display
   * 
   * Args:
   *     limit: Limite in token
   *     
   * Returns:
   *     Stringa formattata con unità
   *     
   * Data ultima modifica: 2025-08-31
   */
  const formatContextLimit = (limit: number): string => {
    if (limit >= 100000) {
      return `${Math.round(limit / 1000)}K token`;
    }
    return `${limit.toLocaleString()} token`;
  };

  if (loading) {
    return (
      <div className="llm-model-selector loading">
        <div className="selector-header">
          <h3>🔄 Caricamento Modelli LLM...</h3>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="llm-model-selector error">
        <div className="selector-header">
          <h3>❌ Errore Caricamento Modelli</h3>
          <p className="error-message">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="llm-model-selector">
      <div className="selector-header">
        <h3>🤖 Selezione Modello LLM</h3>
        <p className="tenant-info">Tenant: <strong>{tenantId}</strong></p>
      </div>

      <div className="model-selection">
        <label htmlFor="model-select" className="select-label">
          Modello LLM:
        </label>
        
        <select
          id="model-select"
          value={selectedModel?.name || ''}
          onChange={(e) => handleModelChange(e.target.value)}
          disabled={disabled || loading}
          className="model-dropdown"
        >
          <option value="">-- Seleziona Modello --</option>
          {models.map((model) => (
            <option key={model.name} value={model.name}>
              {model.display_name}
              {model.provider === 'openai' ? ' 🤖 (OpenAI)' : ''}
              {model.requires_raw_mode ? ' (Raw Mode)' : ''}
              {model.parallel_calls_max ? ` [Max ${model.parallel_calls_max} calls]` : ''}
            </option>
          ))}
        </select>
      </div>

      {selectedModel && (
        <div className="model-info">
          <div className="info-header">
            <h4>📊 Informazioni Modello: {selectedModel.display_name}</h4>
          </div>
          
          <div className="info-grid">
            <div className="info-item">
              <span className="label">Provider:</span>
              <span className={`value provider-${selectedModel.provider || 'ollama'}`}>
                {selectedModel.provider === 'openai' ? '🤖 OpenAI' : '🦙 Ollama'}
              </span>
            </div>
            
            <div className="info-item">
              <span className="label">Context Limit:</span>
              <span className="value">{formatContextLimit(selectedModel.context_limit)}</span>
            </div>
            
            <div className="info-item">
              <span className="label">Max Input Tokens:</span>
              <span className="value">{selectedModel.max_input_tokens.toLocaleString()}</span>
            </div>
            
            <div className="info-item">
              <span className="label">Max Output Tokens:</span>
              <span className="value">{selectedModel.max_output_tokens.toLocaleString()}</span>
            </div>
            
            {selectedModel.provider === 'openai' && selectedModel.parallel_calls_max && (
              <div className="info-item">
                <span className="label">Max Parallel Calls:</span>
                <span className="value parallel-calls">⚡ {selectedModel.parallel_calls_max}</span>
              </div>
            )}
            
            {selectedModel.provider === 'openai' && selectedModel.rate_limit_per_minute && (
              <div className="info-item">
                <span className="label">Rate Limit:</span>
                <span className="value rate-limit">
                  📊 {selectedModel.rate_limit_per_minute.toLocaleString()}/min
                </span>
              </div>
            )}
            
            <div className="info-item">
              <span className="label">Raw Mode:</span>
              <span className={`value ${selectedModel.requires_raw_mode ? 'required' : 'optional'}`}>
                {selectedModel.requires_raw_mode ? '🔴 Richiesto' : '🟢 Opzionale'}
              </span>
            </div>
          </div>

          {selectedModel.default_generation && (
            <div className="default-params">
              <h5>⚙️ Parametri Default:</h5>
              <div className="params-grid">
                <span>Temperature: <strong>{selectedModel.default_generation.temperature}</strong></span>
                <span>Top K: <strong>{selectedModel.default_generation.top_k}</strong></span>
                <span>Top P: <strong>{selectedModel.default_generation.top_p}</strong></span>
                <span>Max Tokens: <strong>{selectedModel.default_generation.max_tokens}</strong></span>
              </div>
            </div>
          )}
        </div>
      )}

      <style>{`
        .llm-model-selector {
          background: #f8f9fa;
          border: 1px solid #dee2e6;
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 20px;
        }

        .llm-model-selector .selector-header h3 {
          margin: 0 0 10px 0;
          color: #495057;
          font-size: 1.2em;
        }

        .llm-model-selector .tenant-info {
          margin: 0 0 15px 0;
          font-size: 0.9em;
          color: #6c757d;
        }

        .llm-model-selector .model-selection {
          margin-bottom: 20px;
        }

        .llm-model-selector .select-label {
          display: block;
          margin-bottom: 8px;
          font-weight: 600;
          color: #495057;
        }

        .llm-model-selector .model-dropdown {
          width: 100%;
          padding: 10px;
          border: 1px solid #ced4da;
          border-radius: 4px;
          font-size: 1em;
          background: white;
        }

        .llm-model-selector .model-dropdown:disabled {
          background: #f8f9fa;
          color: #6c757d;
        }

        .llm-model-selector .model-info {
          background: white;
          border: 1px solid #e9ecef;
          border-radius: 6px;
          padding: 16px;
        }

        .llm-model-selector .info-header h4 {
          margin: 0 0 15px 0;
          color: #495057;
        }

        .llm-model-selector .info-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
          margin-bottom: 15px;
        }

        .llm-model-selector .info-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px;
          background: #f8f9fa;
          border-radius: 4px;
        }

        .llm-model-selector .label {
          font-weight: 600;
          color: #495057;
        }

        .llm-model-selector .value {
          font-weight: 500;
        }

        .llm-model-selector .value.required {
          color: #dc3545;
        }

        .llm-model-selector .value.optional {
          color: #28a745;
        }

        .llm-model-selector .value.provider-openai {
          color: #0066cc;
          font-weight: 600;
        }

        .llm-model-selector .value.provider-ollama {
          color: #ff6b35;
          font-weight: 600;
        }

        .llm-model-selector .value.parallel-calls {
          color: #17a2b8;
          font-weight: 600;
        }

        .llm-model-selector .value.rate-limit {
          color: #6610f2;
          font-weight: 600;
        }

        .llm-model-selector .default-params h5 {
          margin: 15px 0 10px 0;
          color: #495057;
        }

        .llm-model-selector .params-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
          font-size: 0.9em;
        }

        .llm-model-selector .params-grid span {
          padding: 6px;
          background: #e9ecef;
          border-radius: 4px;
        }

        .llm-model-selector .error-message {
          color: #dc3545;
          font-style: italic;
          margin: 10px 0;
        }

        .llm-model-selector.loading {
          opacity: 0.6;
        }
      `}</style>
    </div>
  );
};

export default LLMModelSelector;
