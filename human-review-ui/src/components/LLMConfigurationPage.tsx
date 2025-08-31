/**
 * ============================================================================
 * LLM Configuration Page Component
 * ============================================================================
 * 
 * Autore: Valerio Bignardi
 * Data creazione: 2025-08-31
 * Ultima modifica: 2025-08-31
 * 
 * Descrizione:
 *     Pagina completa per configurazione LLM dei tenant che combina
 *     selezione modello e pannello parametri con funzionalità di test
 *     e anteprima configurazione in tempo reale.
 * 
 * Props:
 *     - tenantId: ID del tenant da configurare
 *     - onConfigurationChange: Callback per notificare cambi config
 * 
 * ============================================================================
 */

import React, { useState, useEffect } from 'react';
import LLMModelSelector from './LLMModelSelector';
import LLMParametersPanel from './LLMParametersPanel';
import { LLMModel, LLMParameters, TenantLLMConfig, LLMConfigService } from '../services/llmConfigService';

interface LLMConfigurationPageProps {
  tenantId: string;
  onConfigurationChange?: (config: TenantLLMConfig) => void;
}

/**
 * Pagina completa per configurazione LLM tenant
 * 
 * Scopo:
 *     Fornisce interfaccia completa per gestione configurazione LLM
 *     con selezione modello, parametri e test funzionalità
 *     
 * Props:
 *     tenantId: ID tenant da configurare
 *     onConfigurationChange: Callback notifica cambi
 *     
 * Data ultima modifica: 2025-08-31
 */
const LLMConfigurationPage: React.FC<LLMConfigurationPageProps> = ({
  tenantId,
  onConfigurationChange
}) => {
  const [selectedModel, setSelectedModel] = useState<LLMModel | null>(null);
  const [tenantConfig, setTenantConfig] = useState<TenantLLMConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [testingModel, setTestingModel] = useState(false);
  const [testResult, setTestResult] = useState<any>(null);

  /**
   * Carica configurazione tenant corrente
   * 
   * Data ultima modifica: 2025-08-31
   */
  useEffect(() => {
    const loadTenantConfig = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const config = await LLMConfigService.getTenantParameters(tenantId);
        setTenantConfig(config);
        
        // Carica info modello corrente
        if (config.current_model) {
          const modelInfo = await LLMConfigService.getModelInfo(config.current_model);
          setSelectedModel(modelInfo);
        }
        
      } catch (err: any) {
        setError(err.message || 'Errore caricamento configurazione');
        console.error('❌ [LLMConfigPage] Errore caricamento config:', err);
      } finally {
        setLoading(false);
      }
    };

    if (tenantId) {
      loadTenantConfig();
    }
  }, [tenantId]);

  /**
   * Gestisce cambio modello selezionato
   * 
   * Args:
   *     model: Nuovo modello selezionato
   *     
   * Data ultima modifica: 2025-08-31
   */
  const handleModelChange = (model: LLMModel) => {
    setSelectedModel(model);
    setTestResult(null); // Reset risultati test precedenti
    
    // Se il modello ha parametri default, li applica
    if (model.default_generation && tenantConfig) {
      const updatedConfig = {
        ...tenantConfig,
        current_model: model.name,
        parameters: {
          ...tenantConfig.parameters,
          generation: {
            ...tenantConfig.parameters.generation,
            ...model.default_generation
          }
        }
      };
      
      setTenantConfig(updatedConfig);
      
      if (onConfigurationChange) {
        onConfigurationChange(updatedConfig);
      }
    }
  };

  /**
   * Gestisce cambio parametri dal pannello
   * 
   * Args:
   *     parameters: Nuovi parametri
   *     
   * Data ultima modifica: 2025-08-31
   */
  const handleParametersChange = (parameters: Partial<LLMParameters>) => {
    if (tenantConfig) {
      const updatedConfig = {
        ...tenantConfig,
        parameters: {
          ...tenantConfig.parameters,
          ...parameters
        }
      };
      
      setTenantConfig(updatedConfig);
      
      if (onConfigurationChange) {
        onConfigurationChange(updatedConfig);
      }
    }
  };

  /**
   * Testa configurazione corrente
   * 
   * Data ultima modifica: 2025-08-31
   */
  const handleTestConfiguration = async () => {
    if (!selectedModel || !tenantConfig) {
      return;
    }

    try {
      setTestingModel(true);
      setTestResult(null);
      
      const result = await LLMConfigService.testModelConfiguration(
        tenantId,
        selectedModel.name,
        tenantConfig.parameters,
        'Test di configurazione LLM. Rispondi con una breve conferma che tutto funziona correttamente.'
      );
      
      setTestResult(result);
      
    } catch (err: any) {
      setTestResult({
        success: false,
        error: err.message || 'Errore test configurazione'
      });
      console.error('❌ [LLMConfigPage] Errore test:', err);
    } finally {
      setTestingModel(false);
    }
  };

  if (loading) {
    return (
      <div className="llm-config-page loading">
        <h2>🔄 Caricamento Configurazione LLM...</h2>
      </div>
    );
  }

  if (error) {
    return (
      <div className="llm-config-page error">
        <h2>❌ Errore Configurazione LLM</h2>
        <p className="error-message">{error}</p>
        <button onClick={() => window.location.reload()} className="retry-btn">
          🔄 Riprova
        </button>
      </div>
    );
  }

  return (
    <div className="llm-config-page">
      <div className="page-header">
        <h2>🤖 Configurazione LLM - {tenantId}</h2>
        <p className="page-description">
          Configura il modello LLM e i parametri di generazione per il tenant.
          Le modifiche vengono applicate immediatamente al sistema di classificazione.
        </p>
        
        {tenantConfig && (
          <div className="config-status">
            <span className="status-label">Stato:</span>
            <span className={`status-value ${tenantConfig.source}`}>
              {tenantConfig.source === 'tenant_specific' ? '🎯 Personalizzato' : '🌐 Default'}
            </span>
            {tenantConfig.last_modified && (
              <span className="last-modified">
                (Ultimo aggiornamento: {new Date(tenantConfig.last_modified).toLocaleString()})
              </span>
            )}
          </div>
        )}
      </div>

      <LLMModelSelector
        tenantId={tenantId}
        currentModel={tenantConfig?.current_model}
        onModelChange={handleModelChange}
        disabled={loading}
      />

      {tenantConfig && (
        <LLMParametersPanel
          tenantId={tenantId}
          selectedModel={selectedModel}
          initialParameters={tenantConfig.parameters}
          onParametersChange={handleParametersChange}
          autoSave={false}
          disabled={loading}
        />
      )}

      <div className="test-section">
        <div className="test-header">
          <h3>🧪 Test Configurazione</h3>
          <button
            onClick={handleTestConfiguration}
            disabled={!selectedModel || testingModel || loading}
            className="test-btn"
          >
            {testingModel ? '🔄 Testing...' : '🧪 Testa Configurazione'}
          </button>
        </div>

        {testResult && (
          <div className={`test-result ${testResult.success ? 'success' : 'error'}`}>
            {testResult.success ? (
              <div>
                <h4>✅ Test Completato con Successo</h4>
                <div className="test-details">
                  <p><strong>Durata:</strong> {testResult.test_duration}s</p>
                  <p><strong>Modello:</strong> {testResult.model_name}</p>
                  <p><strong>Lunghezza Risposta:</strong> {testResult.response_length} caratteri</p>
                </div>
                <div className="response-preview">
                  <h5>📄 Anteprima Risposta:</h5>
                  <pre>{testResult.response_preview}</pre>
                </div>
              </div>
            ) : (
              <div>
                <h4>❌ Test Fallito</h4>
                <p className="error-details">{testResult.error}</p>
              </div>
            )}
          </div>
        )}
      </div>

      <style>{`
        .llm-config-page {
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
        }

        .llm-config-page .page-header {
          margin-bottom: 30px;
          padding-bottom: 20px;
          border-bottom: 2px solid #e9ecef;
        }

        .llm-config-page .page-header h2 {
          margin: 0 0 10px 0;
          color: #495057;
          font-size: 1.8em;
        }

        .llm-config-page .page-description {
          margin: 0 0 15px 0;
          color: #6c757d;
          font-size: 1em;
          line-height: 1.5;
        }

        .llm-config-page .config-status {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 0.9em;
        }

        .llm-config-page .status-label {
          font-weight: 600;
          color: #495057;
        }

        .llm-config-page .status-value {
          font-weight: 500;
          padding: 4px 8px;
          border-radius: 4px;
        }

        .llm-config-page .status-value.tenant_specific {
          background: #d4edda;
          color: #155724;
        }

        .llm-config-page .status-value.global {
          background: #cce5ff;
          color: #004085;
        }

        .llm-config-page .last-modified {
          color: #6c757d;
          font-size: 0.85em;
        }

        .llm-config-page .test-section {
          background: #f8f9fa;
          border: 1px solid #dee2e6;
          border-radius: 8px;
          padding: 20px;
          margin-top: 20px;
        }

        .llm-config-page .test-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 15px;
        }

        .llm-config-page .test-header h3 {
          margin: 0;
          color: #495057;
        }

        .llm-config-page .test-btn {
          padding: 10px 20px;
          background: #17a2b8;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 0.9em;
          transition: all 0.2s;
        }

        .llm-config-page .test-btn:hover:not(:disabled) {
          background: #138496;
        }

        .llm-config-page .test-btn:disabled {
          background: #6c757d;
          cursor: not-allowed;
        }

        .llm-config-page .test-result {
          margin-top: 15px;
          padding: 15px;
          border-radius: 6px;
        }

        .llm-config-page .test-result.success {
          background: #d4edda;
          border: 1px solid #c3e6cb;
        }

        .llm-config-page .test-result.error {
          background: #f8d7da;
          border: 1px solid #f5c6cb;
        }

        .llm-config-page .test-result h4 {
          margin: 0 0 10px 0;
        }

        .llm-config-page .test-details {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 10px;
          margin-bottom: 15px;
        }

        .llm-config-page .test-details p {
          margin: 0;
          font-size: 0.9em;
        }

        .llm-config-page .response-preview h5 {
          margin: 0 0 10px 0;
          color: #495057;
        }

        .llm-config-page .response-preview pre {
          background: #2d3748;
          color: #e2e8f0;
          padding: 15px;
          border-radius: 4px;
          font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
          font-size: 0.85em;
          overflow-x: auto;
          margin: 0;
          white-space: pre-wrap;
        }

        .llm-config-page .error-details {
          color: #721c24;
          font-weight: 500;
        }

        .llm-config-page .error-message {
          color: #dc3545;
          font-style: italic;
        }

        .llm-config-page .retry-btn {
          padding: 10px 20px;
          background: #007bff;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          margin-top: 15px;
        }

        .llm-config-page .retry-btn:hover {
          background: #0056b3;
        }

        .llm-config-page.loading {
          opacity: 0.6;
        }
      `}</style>
    </div>
  );
};

export default LLMConfigurationPage;
