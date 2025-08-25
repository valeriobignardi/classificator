import axios from 'axios';
import { ReviewCase, ReviewStats } from '../types/ReviewCase';
import { Tenant } from '../types/Tenant';

// Usa URL relativi in development con proxy, URL assoluti in production
const API_BASE_URL = process.env.NODE_ENV === 'development' 
  ? '/api' 
  : 'http://localhost:5000/api';

const DEV_BASE_URL = process.env.NODE_ENV === 'development' 
  ? '' 
  : 'http://localhost:5000';

class ApiService {
  private async handleRequest<T>(request: Promise<any>): Promise<T> {
    try {
      const response = await request;
      if (response.data.success) {
        return response.data;
      } else {
        throw new Error(response.data.error || 'API request failed');
      }
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.error || error.message);
      }
      throw error;
    }
  }

  async getReviewCases(tenant: string, limit: number = 20, includePropagated: boolean = false, includeOutliers: boolean = false): Promise<{ cases: ReviewCase[]; total: number }> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/review/${tenant}/cases`, {
        params: { 
          limit,
          include_propagated: includePropagated,
          include_outliers: includeOutliers
        }
      })
    );
  }

  // 🆕 Nuovo metodo per cluster view - mostra solo rappresentanti per default
  async getClusterCases(tenant: string, limit: number = 20, includePropagated: boolean = false, includeOutliers: boolean = false): Promise<{ clusters: any[]; total: number }> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/review/${tenant}/clusters`, {
        params: { 
          limit,
          include_propagated: includePropagated,
          include_outliers: includeOutliers
        }
      })
    );
  }

  async getCaseDetail(tenant: string, caseId: string): Promise<{ case: ReviewCase }> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/review/${tenant}/cases/${caseId}`)
    );
  }

  async resolveCase(
    tenant: string, 
    caseId: string, 
    humanDecision: string, 
    confidence: number, 
    notes?: string
  ): Promise<{ message: string }> {
    return this.handleRequest(
      axios.post(`${API_BASE_URL}/review/${tenant}/cases/${caseId}/resolve`, {
        human_decision: humanDecision,
        confidence,
        notes
      })
    );
  }

  async getReviewStats(tenant: string): Promise<{ stats: ReviewStats }> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/review/${tenant}/stats`)
    );
  }

  async createMockCases(tenant: string, count: number = 3): Promise<{ created_cases: string[]; total_pending: number }> {
    return this.handleRequest(
      axios.post(`${DEV_BASE_URL}/dev/create-mock-cases/${tenant}`, {
        count
      })
    );
  }

  // NUOVI METODI PER STATISTICHE ETICHETTE
  async getAvailableTenants(): Promise<{ tenants: string[]; total: number }> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/stats/tenants`)
    );
  }

  // Metodo per recuperare l'elenco dei tenant dal server
  async getTenants(): Promise<Tenant[]> {
    console.log('🔍 [DEBUG] ApiService.getTenants() - Avvio richiesta');
    console.log('🔍 [DEBUG] URL chiamata:', `${API_BASE_URL}/tenants`);
    
    try {
      console.log('🔍 [DEBUG] Eseguo axios.get...');
      const axiosResponse = await axios.get(`${API_BASE_URL}/tenants`);
      console.log('✅ [DEBUG] Risposta axios ricevuta:', axiosResponse.status);
      console.log('✅ [DEBUG] Dati risposta:', axiosResponse.data);
      
      console.log('🔍 [DEBUG] Chiamo handleRequest...');
      const response = await this.handleRequest<{ tenants: Tenant[] }>(
        Promise.resolve(axiosResponse)
      );
      console.log('✅ [DEBUG] HandleRequest completato:', response);
      console.log('✅ [DEBUG] Restituisco tenant:', response.tenants.length, 'elementi');
      return response.tenants;
      
    } catch (error) {
      console.error('❌ [DEBUG] Errore in getTenants():', error);
      console.error('❌ [DEBUG] Tipo errore:', typeof error);
      console.error('❌ [DEBUG] Stack:', error instanceof Error ? error.stack : 'No stack');
      throw error;
    }
  }

  async getLabelStatistics(tenant: string): Promise<any> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/stats/labels/${tenant}`)
    );
  }

  async startSupervisedTraining(tenant: string, options?: {
    batch_size?: number;
    min_confidence?: number;
    disagreement_threshold?: number;
    force_review?: boolean;
    max_review_cases?: number | null;
  }): Promise<any> {
    return this.handleRequest(
      axios.post(`${DEV_BASE_URL}/train/supervised/${tenant}`, options || {})
    );
  }

  async startFullClassification(tenant: string, options?: {
    confidence_threshold?: number;
    force_retrain?: boolean;
    max_sessions?: number;
    debug_mode?: boolean;
    force_review?: boolean;
    force_reprocess_all?: boolean;  // NUOVO: Riclassifica tutto dall'inizio
  }): Promise<any> {
    return this.handleRequest(
      axios.post(`${DEV_BASE_URL}/classify/all/${tenant}`, options || {})
    );
  }

  async getUIConfig(): Promise<any> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/config/ui`)
    );
  }

  async getAvailableTags(tenant: string): Promise<{ tags: Array<{tag: string, count: number, source: string, avg_confidence: number}> }> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/review/${tenant}/available-tags`)
    );
  }

  async triggerManualRetraining(tenant: string): Promise<{
    success: boolean;
    message: string;
    decision_count: number;
    timestamp: string;
  }> {
    return this.handleRequest(
      axios.post(`${API_BASE_URL}/retrain/${tenant}`)
    );
  }

  async getAllSessions(tenant: string, includeReviewed: boolean = false): Promise<{
    sessions: Array<{
      session_id: string;
      conversation_text: string;
      full_text: string;
      num_messages: number;
      num_user_messages: number;
      status: 'available' | 'in_review_queue' | 'reviewed';
      created_at: string;
      last_activity: string;
      classifications: Array<{
        tag_name: string;
        confidence: number;
        method: string;
        created_at: string;
      }>;
    }>;
    count: number;
    total_valid_sessions: number;
    breakdown: {
      available: number;
      in_review_queue: number;
      reviewed: number;
    };
  }> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/review/${tenant}/all-sessions`, {
        params: { include_reviewed: includeReviewed }
      })
    );
  }

  async addSessionToQueue(tenant: string, sessionId: string, reason?: string): Promise<{
    message: string;
    case_id: string;
    queue_size: number;
  }> {
    return this.handleRequest(
      axios.post(`${API_BASE_URL}/review/${tenant}/add-to-queue`, {
        session_id: sessionId,
        reason: reason || 'manual_addition'
      })
    );
  }

  // NUOVO: Metodo per cancellare tutte le classificazioni esistenti
  async clearAllClassifications(tenant: string): Promise<{
    success: boolean;
    message: string;
    deleted_count: number;
    timestamp: string;
  }> {
    return this.handleRequest(
      axios.delete(`${API_BASE_URL}/classifications/${tenant}/clear-all`)
    );
  }

  // FINE-TUNING METHODS
  
  async get<T = any>(url: string): Promise<T> {
    const response = await axios.get(url);
    if (response.data.success !== undefined) {
      if (response.data.success) {
        return response.data;
      } else {
        throw new Error(response.data.error || 'API request failed');
      }
    }
    return response.data;
  }

  async post<T = any>(url: string, data?: any): Promise<T> {
    const response = await axios.post(url, data);
    if (response.data.success !== undefined) {
      if (response.data.success) {
        return response.data;
      } else {
        throw new Error(response.data.error || 'API request failed');
      }
    }
    return response.data;
  }

  async getFineTuningStatus(tenant: string): Promise<any> {
    return this.get(`${API_BASE_URL}/finetuning/${tenant}/status`);
  }

  async createFineTuning(tenant: string, config: {
    min_confidence?: number;
    force_retrain?: boolean;
  }): Promise<any> {
    return this.post(`${API_BASE_URL}/finetuning/${tenant}/create`, config);
  }

  async switchModel(tenant: string, modelType: 'finetuned' | 'base'): Promise<any> {
    return this.post(`${API_BASE_URL}/finetuning/${tenant}/switch`, {
      model_type: modelType
    });
  }

  async listAllModels(): Promise<any> {
    return this.get(`${API_BASE_URL}/finetuning/models`);
  }

  // PROMPT MANAGEMENT METHODS
  
  async checkPromptStatus(tenant: string): Promise<{
    canOperate: boolean;
    requiredPrompts: Array<{
      name: string;
      type: string;
      description: string;
      exists: boolean;
    }>;
    missingCount: number;
  }> {
    console.log('🔍 [DEBUG] ApiService.checkPromptStatus() - Avvio richiesta');
    console.log('🔍 [DEBUG] Tenant:', tenant);
    console.log('🔍 [DEBUG] URL chiamata:', `${API_BASE_URL}/prompts/${tenant}/status`);
    
    try {
      console.log('🔍 [DEBUG] Eseguo axios.get per prompt status...');
      const axiosResponse = await axios.get(`${API_BASE_URL}/prompts/${tenant}/status`);
      console.log('✅ [DEBUG] Risposta axios per prompt status:', axiosResponse.status);
      console.log('✅ [DEBUG] Dati prompt status:', axiosResponse.data);
      
      console.log('🔍 [DEBUG] Chiamo handleRequest per prompt status...');
      const result = await this.handleRequest<{
        canOperate: boolean;
        requiredPrompts: Array<{
          name: string;
          type: string;
          description: string;
          exists: boolean;
        }>;
        missingCount: number;
      }>(Promise.resolve(axiosResponse));
      console.log('✅ [DEBUG] HandleRequest per prompt status completato:', result);
      return result;
      
    } catch (error) {
      console.error('❌ [DEBUG] Errore in checkPromptStatus():', error);
      console.error('❌ [DEBUG] Tipo errore:', typeof error);
      console.error('❌ [DEBUG] Stack:', error instanceof Error ? error.stack : 'No stack');
      throw error;
    }
  }

  async createPromptFromTemplate(
    tenant: string,
    tenantName: string,
    config: {
      customize_prompts: boolean;
      system_customization: string;
    }
  ): Promise<{
    success: boolean;
    message: string;
    created_prompts: string[];
  }> {
    return this.handleRequest(
      axios.post(`${API_BASE_URL}/prompts/${tenant}/from-template`, {
        tenant_name: tenantName,
        ...config
      })
    );
  }

  /**
   * Copia tutti i prompt dal tenant Humanitas al tenant specificato
   * 
   * @param targetTenantId - ID del tenant di destinazione
   * @returns Risultato della copia con lista prompt copiati
   * 
   * Autore: Sistema 
   * Data: 2025-08-24
   * Descrizione: Copia automatica prompt template da Humanitas
   */
  async copyPromptsFromHumanitas(targetTenantId: string): Promise<{
    success: boolean;
    copied_prompts: number;
    prompts: any[];
    message: string;
  }> {
    console.log('🔄 [DEBUG] ApiService.copyPromptsFromHumanitas() - Avvio copia');
    console.log('🔄 [DEBUG] Target tenant:', targetTenantId);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/prompts/copy-from-humanitas`, {
        target_tenant_id: targetTenantId
      });
      
      console.log('✅ [DEBUG] Copia prompt completata:', response.data);
      return response.data;
      
    } catch (error) {
      console.error('❌ [DEBUG] Errore copia prompt:', error);
      throw error;
    }
  }

  // =====================================================================
  // METODI PER GESTIONE ESEMPI
  // =====================================================================

  /**
   * Recupera la lista degli esempi per un tenant
   * @param tenantId ID del tenant
   * @returns Lista degli esempi
   */
  async getExamples(tenantId: string): Promise<any[]> {
    console.log('🔍 [DEBUG] ApiService.getExamples() - Avvio richiesta');
    console.log('🔍 [DEBUG] Tenant:', tenantId);
    
    const url = `${API_BASE_URL}/examples?tenant_id=${tenantId}`;
    console.log('🔍 [DEBUG] URL chiamata:', url);

    try {
      const response = await axios.get(url);
      console.log('✅ [DEBUG] Risposta esempi ricevuta:', response.status);
      console.log('✅ [DEBUG] Dati esempi:', response.data);
      
      return response.data.data || [];
    } catch (error) {
      console.error('❌ [DEBUG] Errore get esempi:', error);
      throw error;
    }
  }

  /**
   * Crea un nuovo esempio
   * @param example Dati dell'esempio
   * @returns Risultato operazione
   */
  async createExample(example: any): Promise<any> {
    console.log('🔍 [DEBUG] ApiService.createExample() - Avvio richiesta');
    console.log('🔍 [DEBUG] Esempio:', example);
    
    const url = `${API_BASE_URL}/examples`;
    console.log('🔍 [DEBUG] URL chiamata:', url);

    try {
      const response = await axios.post(url, example);
      console.log('✅ [DEBUG] Esempio creato:', response.status);
      console.log('✅ [DEBUG] Risultato:', response.data);
      
      return response.data;
    } catch (error) {
      console.error('❌ [DEBUG] Errore creazione esempio:', error);
      throw error;
    }
  }

  /**
   * Elimina un esempio
   * @param exampleId ID dell'esempio
   * @param tenantId ID del tenant
   * @returns Risultato operazione
   */
  async deleteExample(exampleId: number, tenantId: string): Promise<any> {
    console.log('🔍 [DEBUG] ApiService.deleteExample() - Avvio richiesta');
    console.log('🔍 [DEBUG] ID esempio:', exampleId);
    console.log('🔍 [DEBUG] Tenant:', tenantId);
    
    const url = `${API_BASE_URL}/examples/${exampleId}?tenant_id=${tenantId}`;
    console.log('🔍 [DEBUG] URL chiamata:', url);

    try {
      const response = await axios.delete(url);
      console.log('✅ [DEBUG] Esempio eliminato:', response.status);
      console.log('✅ [DEBUG] Risultato:', response.data);
      
      return response.data;
    } catch (error) {
      console.error('❌ [DEBUG] Errore eliminazione esempio:', error);
      throw error;
    }
  }

  /**
   * Ottiene esempi formattati per il placeholder {{examples_text}}
   * @param tenantId ID del tenant
   * @param limit Numero massimo di esempi
   * @returns Oggetto con esempi formattati e statistiche
   */
  async getExamplesForPlaceholder(tenantId: string, limit: number = 3): Promise<{
    examples_text: string;
    num_conversations: number;
    length: number;
  }> {
    console.log('🔍 [DEBUG] ApiService.getExamplesForPlaceholder() - Avvio richiesta');
    console.log('🔍 [DEBUG] Tenant:', tenantId);
    console.log('🔍 [DEBUG] Limit:', limit);
    
    const url = `${API_BASE_URL}/examples/placeholder?tenant_id=${tenantId}&limit=${limit}`;
    console.log('🔍 [DEBUG] URL chiamata:', url);

    try {
      const response = await axios.get(url);
      console.log('✅ [DEBUG] Placeholder esempi ricevuto:', response.status);
      console.log('✅ [DEBUG] Esempi formattati:', response.data);
      
      return response.data.data || { examples_text: '', num_conversations: 0, length: 0 };
    } catch (error) {
      console.error('❌ [DEBUG] Errore get placeholder esempi:', error);
      throw error;
    }
  }
}

export const apiService = new ApiService();
export { ApiService };
