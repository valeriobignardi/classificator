/**
 * ============================================================================
 * LLM Configuration Service
 * ============================================================================
 * 
 * Autore: Valerio Bignardi
 * Data creazione: 2025-08-31
 * Ultima modifica: 2025-08-31
 * 
 * Descrizione:
 *     Servizio frontend per gestione configurazione LLM dei tenant.
 *     Fornisce interfaccia per recupero modelli, gestione parametri,
 *     validazione e test delle configurazioni.
 * 
 * Funzionalità principali:
 *     - Recupero modelli LLM disponibili
 *     - Gestione parametri LLM tenant-specific
 *     - Validazione real-time parametri
 *     - Test configurazioni modelli
 * 
 * ============================================================================
 */

import axios from 'axios';

// Usa URL relativi in development con proxy, URL assoluti in production
const API_BASE_URL = process.env.NODE_ENV === 'development' 
  ? '/api/llm' 
  : 'http://localhost:5000/api/llm';

/**
 * Interfacce TypeScript per tipizzazione forte
 */
export interface LLMModel {
  name: string;
  display_name: string;
  provider?: string;  // 'ollama' | 'openai'
  max_input_tokens: number;
  max_output_tokens: number;
  context_limit: number;
  requires_raw_mode: boolean;
  parallel_calls_max?: number;
  rate_limit_per_minute?: number;
  rate_limit_per_day?: number;
  default_generation?: {
    max_tokens: number;
    temperature: number;
    top_k: number;
    top_p: number;
    repeat_penalty: number;
  };
}

export interface LLMParameters {
  tokenization: {
    max_tokens: number;
    model_name: string;
    truncation_strategy: string;
  };
  generation: {
    max_tokens: number;
    temperature: number;
    top_k: number;
    top_p: number;
    repeat_penalty: number;
  };
  connection: {
    timeout: number;
    url: string;
  };
}

export interface TenantLLMConfig {
  tenant_id: string;
  current_model: string;
  parameters: LLMParameters;
  source: 'global' | 'tenant_specific';
  last_modified?: string;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  model_constraints?: LLMModel;
}

export interface TestResult {
  success: boolean;
  tenant_id: string;
  model_name: string;
  test_duration: number;
  response_preview: string;
  response_length: number;
  parameters_used: Partial<LLMParameters>;
  validation: ValidationResult;
}

/**
 * Classe servizio per gestione configurazione LLM
 * 
 * Scopo:
 *     Fornisce interfaccia frontend per interazione con API backend
 *     di gestione configurazione LLM dei tenant
 *     
 * Data ultima modifica: 2025-08-31
 */
export class LLMConfigService {
  
  /**
   * Recupera lista modelli LLM disponibili per un tenant
   * 
   * Args:
   *     tenantId: ID del tenant
   *     
   * Returns:
   *     Promise con lista modelli disponibili
   *     
   * Data ultima modifica: 2025-08-31
   */
  static async getAvailableModels(tenantId: string): Promise<LLMModel[]> {
    try {
      const response = await axios.get(`${API_BASE_URL}/models/${tenantId}`);
      
      if (response.data.success) {
        return response.data.models;
      } else {
        throw new Error(response.data.error || 'Errore recupero modelli');
      }
      
    } catch (error: any) {
      console.error(`❌ [LLMConfigService] Errore recupero modelli per ${tenantId}:`, error);
      throw error;
    }
  }

  /**
   * Recupera parametri LLM correnti per un tenant
   * 
   * Args:
   *     tenantId: ID del tenant
   *     
   * Returns:
   *     Promise con configurazione LLM del tenant
   *     
   * Data ultima modifica: 2025-08-31
   */
  static async getTenantParameters(tenantId: string): Promise<TenantLLMConfig> {
    try {
      const response = await axios.get(`${API_BASE_URL}/parameters/${tenantId}`);
      
      if (response.data.success) {
        return {
          tenant_id: response.data.tenant_id,
          current_model: response.data.current_model,
          parameters: response.data.parameters,
          source: response.data.source,
          last_modified: response.data.last_modified
        };
      } else {
        throw new Error(response.data.error || 'Errore recupero parametri');
      }
      
    } catch (error: any) {
      console.error(`❌ [LLMConfigService] Errore recupero parametri per ${tenantId}:`, error);
      throw error;
    }
  }

  /**
   * Aggiorna parametri LLM per un tenant
   * 
   * Args:
   *     tenantId: ID del tenant
   *     parameters: Nuovi parametri LLM
   *     modelName: Nome modello per validazione
   *     
   * Returns:
   *     Promise con risultato operazione
   *     
   * Data ultima modifica: 2025-08-31
   */
  static async updateTenantParameters(
    tenantId: string,
    parameters: Partial<LLMParameters>,
    modelName?: string
  ): Promise<any> {
    try {
      const payload: any = { parameters };
      if (modelName) {
        payload.model_name = modelName;
      }
      
      const response = await axios.put(`${API_BASE_URL}/parameters/${tenantId}`, payload);
      
      if (response.data.success) {
        return response.data;
      } else {
        throw new Error(response.data.error || 'Errore aggiornamento parametri');
      }
      
    } catch (error: any) {
      console.error(`❌ [LLMConfigService] Errore aggiornamento parametri per ${tenantId}:`, error);
      throw error;
    }
  }

  /**
   * Ripristina parametri default per un tenant
   * 
   * Args:
   *     tenantId: ID del tenant
   *     
   * Returns:
   *     Promise con parametri default ripristinati
   *     
   * Data ultima modifica: 2025-08-31
   */
  static async resetTenantParameters(tenantId: string): Promise<any> {
    try {
      const response = await axios.post(`${API_BASE_URL}/reset-parameters/${tenantId}`);
      
      if (response.data.success) {
        return response.data;
      } else {
        throw new Error(response.data.error || 'Errore reset parametri');
      }
      
    } catch (error: any) {
      console.error(`❌ [LLMConfigService] Errore reset parametri per ${tenantId}:`, error);
      throw error;
    }
  }

  /**
   * Recupera informazioni specifiche per un modello
   * 
   * Args:
   *     modelName: Nome del modello
   *     
   * Returns:
   *     Promise con informazioni del modello
   *     
   * Data ultima modifica: 2025-08-31
   */
  static async getModelInfo(modelName: string): Promise<LLMModel> {
    try {
      const response = await axios.get(`${API_BASE_URL}/model-info/${modelName}`);
      
      if (response.data.success) {
        return response.data.model_info;
      } else {
        throw new Error(response.data.error || 'Errore recupero info modello');
      }
      
    } catch (error: any) {
      console.error(`❌ [LLMConfigService] Errore recupero info modello ${modelName}:`, error);
      throw error;
    }
  }

  /**
   * Valida parametri LLM prima del salvataggio
   * 
   * Args:
   *     parameters: Parametri da validare
   *     modelName: Nome modello per vincoli specifici
   *     
   * Returns:
   *     Promise con risultato validazione
   *     
   * Data ultima modifica: 2025-08-31
   */
  static async validateParameters(
    parameters: Partial<LLMParameters>,
    modelName?: string
  ): Promise<ValidationResult> {
    try {
      const payload: any = { parameters };
      if (modelName) {
        payload.model_name = modelName;
      }
      
      const response = await axios.post(`${API_BASE_URL}/validate-parameters`, payload);
      
      if (response.data.success) {
        return response.data.validation;
      } else {
        throw new Error(response.data.error || 'Errore validazione');
      }
      
    } catch (error: any) {
      console.error(`❌ [LLMConfigService] Errore validazione parametri:`, error);
      throw error;
    }
  }

  /**
   * Testa configurazione modello con parametri specifici
   * 
   * Args:
   *     tenantId: ID del tenant
   *     modelName: Nome del modello
   *     parameters: Parametri di test
   *     testPrompt: Prompt personalizzato per test
   *     
   * Returns:
   *     Promise con risultato test
   *     
   * Data ultima modifica: 2025-08-31
   */
  static async testModelConfiguration(
    tenantId: string,
    modelName: string,
    parameters: Partial<LLMParameters>,
    testPrompt?: string
  ): Promise<TestResult> {
    try {
      const payload: any = {
        model_name: modelName,
        parameters
      };
      
      if (testPrompt) {
        payload.test_prompt = testPrompt;
      }
      
      const response = await axios.post(`${API_BASE_URL}/test-model/${tenantId}`, payload);
      
      if (response.data.success) {
        return response.data;
      } else {
        throw new Error(response.data.error || 'Errore test modello');
      }
      
    } catch (error: any) {
      console.error(`❌ [LLMConfigService] Errore test modello per ${tenantId}:`, error);
      throw error;
    }
  }

  /**
   * Recupera lista tenant con configurazioni LLM personalizzate
   * 
   * Returns:
   *     Promise con lista tenant
   *     
   * Data ultima modifica: 2025-08-31
   */
  static async getLLMTenants(): Promise<string[]> {
    try {
      const response = await axios.get(`${API_BASE_URL}/tenants`);
      
      if (response.data.success) {
        return response.data.tenants;
      } else {
        throw new Error(response.data.error || 'Errore recupero tenant');
      }
      
    } catch (error: any) {
      console.error(`❌ [LLMConfigService] Errore recupero tenant:`, error);
      throw error;
    }
  }
}

export default LLMConfigService;
