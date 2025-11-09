import axios from 'axios';
import { ReviewCase, ReviewStats } from '../types/ReviewCase';
import { TrainingFileListResponse, TrainingFileContentResponse } from '../types/TrainingFile';
import { Tenant } from '../types/Tenant';

// Usa sempre URL relativi per passare attraverso il proxy nginx
const API_BASE_URL = '/api';

const DEV_BASE_URL = '';

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

  async getReviewCases(tenant_id: string, limit: number = 20, includePropagated: boolean = false, includeOutliers: boolean = false, includeRepresentatives: boolean = true): Promise<{ cases: ReviewCase[]; total: number }> {
    console.log('🔍 [DEBUG] ApiService.getReviewCases() - Avvio richiesta');
    console.log('🔍 [DEBUG] Tenant ID:', tenant_id);
    console.log('🔍 [DEBUG] URL chiamata:', `${API_BASE_URL}/review/${tenant_id}/cases`);
    
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/review/${tenant_id}/cases`, {
        params: { 
          limit,
          include_propagated: includePropagated,
          include_outliers: includeOutliers,
          include_representatives: includeRepresentatives
        }
      })
    );
  }

  // 🆕 Nuovo metodo per cluster view - mostra solo rappresentanti per default
  async getClusterCases(tenant_id: string, limit: number = 20, includePropagated: boolean = false, includeOutliers: boolean = false): Promise<{ clusters: any[]; total: number }> {
    console.log('🔍 [DEBUG] ApiService.getClusterCases() - Avvio richiesta');
    console.log('🔍 [DEBUG] Tenant ID:', tenant_id);
    
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/review/${tenant_id}/clusters`, {
        params: { 
          limit,
          include_propagated: includePropagated,
          include_outliers: includeOutliers
        }
      })
    );
  }

  async resolveClusterMajority(tenant_id: string, cluster_id: string, options?: { notes?: string }): Promise<{
    success: boolean;
    tenant_id: string;
    cluster_id: string;
    majority_label: string;
    resolved_count: number;
    total_candidates: number;
    errors: Array<{ case_id: string; error: string }>;
  }> {
    return this.handleRequest(
      axios.post(`${API_BASE_URL}/review/${tenant_id}/clusters/${cluster_id}/resolve-majority`, options || {})
    );
  }

  async getCaseDetail(tenant_id: string, caseId: string): Promise<{ case: ReviewCase }> {
    console.log('🔍 [DEBUG] ApiService.getCaseDetail() - Avvio richiesta');
    console.log('🔍 [DEBUG] Tenant ID:', tenant_id, 'Case ID:', caseId);
    
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/review/${tenant_id}/cases/${caseId}`)
    );
  }

  async resolveCase(
    tenant_id: string, 
    caseId: string, 
    humanDecision: string, 
    confidence: number, 
    notes?: string
  ): Promise<{ message: string }> {
    console.log('🔍 [DEBUG] ApiService.resolveCase() - Avvio richiesta');
    console.log('🔍 [DEBUG] Tenant ID:', tenant_id, 'Case ID:', caseId);
    console.log('🔍 [DEBUG] URL chiamata:', `${API_BASE_URL}/review/${tenant_id}/cases/${caseId}/resolve`);
    
    return this.handleRequest(
      axios.post(`${API_BASE_URL}/review/${tenant_id}/cases/${caseId}/resolve`, {
        human_decision: humanDecision,
        confidence,
        notes
      })
    );
  }

  async getReviewStats(tenant_id: string): Promise<{ stats: ReviewStats }> {
    console.log('🔍 [DEBUG] ApiService.getReviewStats() - Avvio richiesta');
    console.log('🔍 [DEBUG] Tenant ID:', tenant_id);
    
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/review/${tenant_id}/stats`)
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
    const response = await this.handleRequest<{ tenants: Tenant[] }>(
      axios.get(`${API_BASE_URL}/tenants`)
    );
    return response.tenants;
  }

  async getLabelStatistics(tenant_id: string): Promise<any> {
    console.log('🔍 [DEBUG] ApiService.getLabelStatistics() - Avvio richiesta');
    console.log('🔍 [DEBUG] Tenant ID:', tenant_id);
    
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/stats/labels/${tenant_id}`)
    );
  }

  async startSupervisedTraining(tenant: string, options?: {
    batch_size?: number;
    min_confidence?: number;
    disagreement_threshold?: number;
    force_review?: boolean;
    max_review_cases?: number | null;
  }): Promise<any> {
    // Instrada sempre tramite proxy /api per evitare problemi di CORS/proxy
    return this.handleRequest(
      axios.post(`${API_BASE_URL}/train/supervised/${tenant}`, options || {})
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

  // SCHEDULER CONFIG PER-TENANT
  async getSchedulerConfig(tenantId: string): Promise<{ config: any }> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/scheduler/config/${tenantId}`)
    );
  }

  async setSchedulerConfig(tenantId: string, config: {
    enabled: boolean;
    frequency_unit: 'minutes' | 'hours' | 'days' | 'weeks';
    frequency_value: number;
    start_at?: string | null; // ISO or YYYY-MM-DDTHH:MM
  }): Promise<{ config: any }> {
    return this.handleRequest(
      axios.post(`${API_BASE_URL}/scheduler/config/${tenantId}`, config)
    );
  }

  async runSchedulerNow(tenantId: string): Promise<{ result: any }> {
    return this.handleRequest(
      axios.post(`${API_BASE_URL}/scheduler/run-now/${tenantId}`)
    );
  }

  // TRAINING FILES
  async listTrainingFiles(tenant_id: string): Promise<TrainingFileListResponse> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/training-files/${tenant_id}`)
    );
  }

  async getTrainingFileContent(tenant_id: string, fileName: string, limit: number = 500): Promise<TrainingFileContentResponse> {
    const params = new URLSearchParams({ file: fileName, limit: String(limit) });
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/training-files/${tenant_id}/content?${params.toString()}`)
    );
  }

  async getAvailableTags(tenant_id: string): Promise<{ tags: Array<{tag: string, count: number, source: string, avg_confidence: number}> }> {
    console.log('🔍 [DEBUG] ApiService.getAvailableTags() - Avvio richiesta');
    console.log('🔍 [DEBUG] Tenant ID:', tenant_id);
    console.log('🔍 [DEBUG] URL chiamata:', `${API_BASE_URL}/review/${tenant_id}/available-tags`);
    
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/review/${tenant_id}/available-tags`)
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
        cluster_id?: string; // 🆕 AGGIUNTO CLUSTER ID
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
    return await this.handleRequest<{
      canOperate: boolean;
      requiredPrompts: Array<{
        name: string;
        type: string;
        description: string;
        exists: boolean;
      }>;
      missingCount: number;
    }>(axios.get(`${API_BASE_URL}/prompts/${tenant}/status`));
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
   * Aggiorna un esempio esistente
   * @param exampleId ID dell'esempio da aggiornare
   * @param data Dati dell'esempio da aggiornare
   * Autore: Valerio Bignardi
   * Data: 2025-08-30
   */
  async updateExample(exampleId: number, data: {
    esempio_name: string;
    description: string;
    categoria: string;
    livello_difficolta: string;
    tenant_id: string;
  }): Promise<any> {
    console.log('🔍 [DEBUG] ApiService.updateExample() - Avvio richiesta');
    console.log('🔍 [DEBUG] ID esempio:', exampleId);
    console.log('🔍 [DEBUG] Dati:', data);
    
    const url = `${API_BASE_URL}/examples/${exampleId}`;
    console.log('🔍 [DEBUG] URL chiamata:', url);

    try {
      const response = await axios.put(url, data);
      console.log('✅ [DEBUG] Esempio aggiornato:', response.status);
      console.log('✅ [DEBUG] Risultato:', response.data);
      
      return response.data;
    } catch (error) {
      console.error('❌ [DEBUG] Errore aggiornamento esempio:', error);
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

  /**
   * Testa i parametri di clustering senza LLM per ottimizzazione
   * 
   * @param tenantId ID del tenant
   * @param parameters Parametri clustering personalizzati (opzionale)
   * @param sampleSize Numero conversazioni da testare (opzionale)
   * @returns Risultati del test clustering con statistiche e cluster
   * 
   * Autore: Sistema di Classificazione
   * Data: 2025-08-25
   * Descrizione: Test rapido clustering per validazione parametri
   */
  async testClustering(
    tenantId: string, 
    parameters?: {
      min_cluster_size?: number;
      min_samples?: number;
      cluster_selection_epsilon?: number;
      metric?: string;
      
      // 🆕 NUOVI PARAMETRI AVANZATI HDBSCAN
      cluster_selection_method?: string;
      alpha?: number;
      max_cluster_size?: number;
      allow_single_cluster?: boolean;
    },
    sampleSize?: number
  ): Promise<{
    success: boolean;
    tenant_id: string;
    execution_time: number;
    statistics: {
      total_conversations: number;
      n_clusters: number;
      n_outliers: number;
      clustering_ratio: number;
      parameters_used: any;
    };
    detailed_clusters: Array<{
      cluster_id: number;
      label: string;
      size: number;
      conversations: Array<{
        session_id: string;
        testo_completo: string;
        is_representative: boolean;
      }>;
      representatives: any[];
      representative_count: number;
    }>;
    quality_metrics: {
      silhouette_score: number;
      davies_bouldin_score: number;
      calinski_harabasz_score: number;
      outlier_ratio: number;
      cluster_balance: string;
      quality_assessment: string;
    };
    outlier_analysis: {
      count: number;
      ratio: number;
      analysis: string;
      recommendation: string;
      sample_outliers: any[];
    };
    // 🆕 Dati visualizzazione per grafici 2D/3D
    visualization_data?: {
      points: Array<{
        x: number;
        y: number;
        z?: number;
        cluster_id: number;
        cluster_label: string;
        session_id: string;
        text_preview: string;
      }>;
      cluster_colors: Record<number, string>;
      statistics: {
        total_points: number;
        n_clusters: number;
        n_outliers: number;
        dimensions: number;
      };
      coordinates: {
        tsne_2d: Array<[number, number]>;
        pca_2d: Array<[number, number]>;
        pca_3d: Array<[number, number, number]>;
      };
    };
    error?: string;
  }> {
    console.log('🧪 [DEBUG] ApiService.testClustering() - Avvio richiesta');
    console.log('🧪 [DEBUG] Tenant:', tenantId);
    console.log('🧪 [DEBUG] Parameters:', parameters);
    console.log('🧪 [DEBUG] Sample size:', sampleSize);
    
    const url = `${API_BASE_URL}/clustering/${tenantId}/test`;
    console.log('🧪 [DEBUG] URL chiamata:', url);

    try {
      const payload: any = {};
      if (parameters) {
        payload.parameters = parameters;
      }
      if (sampleSize) {
        payload.sample_size = sampleSize;
      }
      
      console.log('🧪 [DEBUG] Payload:', payload);
      
      const response = await axios.post(url, payload, {
        timeout: 0, // Nessun timeout - può durare quanto necessario
      });
      console.log('✅ [DEBUG] Test clustering - risposta iniziale:', response.status);
      console.log('✅ [DEBUG] Payload risposta:', response.data);

      const data = response.data;

      // Backend asincrono: restituisce job_id da pollare
      if (data && data.job_id) {
        const jobId = data.job_id;
        console.log(`⏱️ [DEBUG] Test clustering in corso - Job ID: ${jobId}`);
        const jobResult = await this.pollClusteringJob(tenantId, jobId);
        console.log('✅ [DEBUG] Test clustering completato (polling):', jobResult);
        return jobResult;
      }
      
      // Backend sincrono (compatibilità retro)
      console.log('✅ [DEBUG] Test clustering completato (sincrono):', data);
      return data;
    } catch (error) {
      console.error('❌ [DEBUG] Errore test clustering:', error);
      
      // Gestione errori specifici
      if (axios.isAxiosError(error)) {
        const errorData = error.response?.data;
        return {
          success: false,
          error: errorData?.error || error.message,
          tenant_id: tenantId,
          execution_time: 0,
          statistics: { total_conversations: 0, n_clusters: 0, n_outliers: 0, clustering_ratio: 0, parameters_used: {} },
          detailed_clusters: [],
          quality_metrics: { 
            silhouette_score: 0, 
            davies_bouldin_score: 0,
            calinski_harabasz_score: 0,
            outlier_ratio: 0, 
            cluster_balance: 'error', 
            quality_assessment: 'error' 
          },
          outlier_analysis: { count: 0, ratio: 0, analysis: 'error', recommendation: '', sample_outliers: [] }
        };
      }
      throw error;
    }
  }

  private async pollClusteringJob(tenantId: string, jobId: string, intervalMs: number = 5000, timeoutMs: number = 20 * 60 * 1000): Promise<any> {
    const pollUrl = `${API_BASE_URL}/clustering/${tenantId}/job/${jobId}`;
    const startTime = Date.now();
    console.log(`📡 [DEBUG] Polling stato job clustering (${jobId})`);

    while (true) {
      if (Date.now() - startTime > timeoutMs) {
        throw new Error('Timeout test clustering: il job richiede troppo tempo');
      }

      try {
        const response = await axios.get(pollUrl, { timeout: intervalMs });
        const jobData = response.data;
        console.log('📡 [DEBUG] Stato job clustering:', jobData);

        if (!jobData.success) {
          throw new Error(jobData.error || 'Job clustering non riuscito');
        }

        if (jobData.status === 'completed' && jobData.result) {
          return jobData.result;
        }

        if (jobData.status === 'failed') {
          const errorMessage = jobData.error || jobData.result?.error || 'Test clustering fallito';
          return {
            success: false,
            error: errorMessage,
            tenant_id: tenantId,
            execution_time: jobData.elapsed_time_seconds || 0,
            statistics: undefined,
            detailed_clusters: [],
            quality_metrics: undefined,
            outlier_analysis: undefined
          };
        }

        await this.delay(intervalMs);
      } catch (err) {
        if (axios.isAxiosError(err) && err.response?.status === 404) {
          throw new Error('Job clustering non trovato o scaduto');
        }
        console.warn('⚠️ [DEBUG] Polling job clustering ha incontrato un errore, ritento...', err);
        await this.delay(intervalMs);
      }
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Ottiene configurazione AI completa per tenant
   * @param tenantId ID del tenant
   * @returns Configurazione AI corrente
   */
  async getAIConfiguration(tenantId: string): Promise<any> {
    console.log('🤖 [DEBUG] ApiService.getAIConfiguration() - Avvio richiesta');
    console.log('🤖 [DEBUG] Tenant:', tenantId);
    
    const url = `${API_BASE_URL}/ai-config/${tenantId}/configuration`;
    console.log('🤖 [DEBUG] URL chiamata:', url);

    try {
      const response = await axios.get(url);
      console.log('✅ [DEBUG] Configurazione AI ricevuta:', response.status);
      console.log('✅ [DEBUG] Dati configurazione:', response.data);
      
      return response.data;
    } catch (error) {
      console.error('❌ [DEBUG] Errore get configurazione AI:', error);
      throw error;
    }
  }

  /**
   * Ottiene embedding engines disponibili
   * @param tenantId ID del tenant
   * @returns Lista embedding engines
   */
  async getEmbeddingEngines(tenantId: string): Promise<any> {
    console.log('🔧 [DEBUG] ApiService.getEmbeddingEngines() - Avvio richiesta');
    console.log('🔧 [DEBUG] Tenant:', tenantId);
    
    const url = `${API_BASE_URL}/ai-config/${tenantId}/embedding-engines`;
    console.log('🔧 [DEBUG] URL chiamata:', url);

    try {
      const response = await axios.get(url);
      console.log('✅ [DEBUG] Embedding engines ricevuti:', response.status);
      console.log('✅ [DEBUG] Dati engines:', response.data);
      
      return response.data;
    } catch (error) {
      console.error('❌ [DEBUG] Errore get embedding engines:', error);
      throw error;
    }
  }

  /**
   * Imposta nuovo embedding engine
   * @param tenantId ID del tenant
   * @param engineType Tipo di engine
   * @param config Configurazione engine
   * @returns Risultato operazione
   */
  async setEmbeddingEngine(tenantId: string, engineType: string, config: any = {}): Promise<any> {
    console.log('🔧 [DEBUG] ApiService.setEmbeddingEngine() - Avvio richiesta');
    console.log('🔧 [DEBUG] Tenant:', tenantId);
    console.log('🔧 [DEBUG] Engine type:', engineType);
    
    const url = `${API_BASE_URL}/ai-config/${tenantId}/embedding-engines`;
    console.log('🔧 [DEBUG] URL chiamata:', url);

    try {
      const payload = {
        engine_type: engineType,
        config: config
      };
      
      const response = await axios.post(url, payload);
      console.log('✅ [DEBUG] Embedding engine impostato:', response.status);
      console.log('✅ [DEBUG] Risultato:', response.data);
      
      return response.data;
    } catch (error) {
      console.error('❌ [DEBUG] Errore set embedding engine:', error);
      throw error;
    }
  }

  /**
   * Ottiene modelli LLM disponibili (Ollama + OpenAI)
   * CORREZIONE: Usa nuova API unificata che include modelli OpenAI
   * @param tenantId ID del tenant
   * @returns Lista modelli LLM con supporto OpenAI e chiamate parallele
   */
  async getLLMModels(tenantId: string): Promise<any> {
    console.log('🧠 [DEBUG] ApiService.getLLMModels() - Avvio richiesta (API unificata)');
    console.log('🧠 [DEBUG] Tenant:', tenantId);
    
    const url = `${API_BASE_URL}/llm/models/${tenantId}`;
    console.log('🧠 [DEBUG] URL chiamata:', url);

    try {
      const response = await axios.get(url);
      console.log('✅ [DEBUG] Modelli LLM ricevuti:', response.status);
      console.log('✅ [DEBUG] Dati modelli (include OpenAI):', response.data);
      
      return response.data;
    } catch (error) {
      console.error('❌ [DEBUG] Errore get modelli LLM:', error);
      throw error;
    }
  }

  /**
   * Imposta nuovo modello LLM
   * @param tenantId ID del tenant
   * @param modelName Nome del modello
   * @returns Risultato operazione
   */
  async setLLMModel(tenantId: string, modelName: string): Promise<any> {
    console.log('🧠 [DEBUG] ApiService.setLLMModel() - Avvio richiesta');
    console.log('🧠 [DEBUG] Tenant:', tenantId);
    console.log('🧠 [DEBUG] Model name:', modelName);
    
    const url = `${API_BASE_URL}/ai-config/${tenantId}/llm-models`;
    console.log('🧠 [DEBUG] URL chiamata:', url);

    try {
      const payload = {
        model_name: modelName
      };
      
      const response = await axios.post(url, payload);
      console.log('✅ [DEBUG] Modello LLM impostato:', response.status);
      console.log('✅ [DEBUG] Risultato:', response.data);
      
      return response.data;
    } catch (error) {
      console.error('❌ [DEBUG] Errore set modello LLM:', error);
      throw error;
    }
  }

  /**
   * Ricarica configurazione modello LLM per un tenant
   * @param tenantId ID del tenant
   * @returns Risultato del reload
   */
  async reloadLLMConfiguration(tenantId: string): Promise<any> {
    console.log('🔄 [DEBUG] ApiService.reloadLLMConfiguration() - Avvio richiesta');
    console.log('🔄 [DEBUG] Tenant:', tenantId);
    
    const url = `${API_BASE_URL}/llm/${tenantId}/reload`;
    console.log('🔄 [DEBUG] URL chiamata:', url);

    try {
      const response = await axios.post(url, {});
      console.log('✅ [DEBUG] LLM configuration reload:', response.status);
      console.log('✅ [DEBUG] Risultato:', response.data);
      
      return response.data;
    } catch (error) {
      console.error('❌ [DEBUG] Errore reload LLM configuration:', error);
      throw error;
    }
  }

  /**
   * Ottiene informazioni sul modello LLM corrente per un tenant
   * @param tenantId ID del tenant
   * @returns Informazioni LLM corrente
   */
  async getCurrentLLMInfo(tenantId: string): Promise<any> {
    console.log('ℹ️ [DEBUG] ApiService.getCurrentLLMInfo() - Avvio richiesta');
    console.log('ℹ️ [DEBUG] Tenant:', tenantId);
    
    const url = `${API_BASE_URL}/llm/${tenantId}/info`;
    console.log('ℹ️ [DEBUG] URL chiamata:', url);

    try {
      const response = await axios.get(url);
      console.log('✅ [DEBUG] LLM info ricevute:', response.status);
      console.log('✅ [DEBUG] Info LLM:', response.data);
      
      return response.data;
    } catch (error) {
      console.error('❌ [DEBUG] Errore get LLM info:', error);
      throw error;
    }
  }

  /**
   * Ottiene informazioni debug per configurazione AI
   * @param tenantId ID del tenant
   * @returns Debug info dettagliate
   */
  async getAIDebugInfo(tenantId: string): Promise<any> {
    console.log('🐛 [DEBUG] ApiService.getAIDebugInfo() - Avvio richiesta');
    console.log('🐛 [DEBUG] Tenant:', tenantId);
    
    const url = `${API_BASE_URL}/ai-config/${tenantId}/debug`;
    console.log('🐛 [DEBUG] URL chiamata:', url);

    try {
      const response = await axios.get(url);
      console.log('✅ [DEBUG] Debug info ricevute:', response.status);
      console.log('✅ [DEBUG] Dati debug:', response.data);
      
      return response.data;
    } catch (error) {
      console.error('❌ [DEBUG] Errore get debug info:', error);
      throw error;
    }
  }
  /**
   * 🆕 Ottiene statistiche avanzate con dati visualizzazione clustering
   * @param tenantId ID del tenant
   * @param timeRange Periodo di analisi (opzionale)
   * @returns Dati clustering + classificazioni per visualizzazioni
   */
  async getClusteringStatistics(
    tenantId: string, 
    timeRange?: { start_date: string; end_date: string }
  ): Promise<{
    success: boolean;
    visualization_data: {
      points: Array<{
        x: number;
        y: number;
        z?: number;
        cluster_id: number;
        cluster_label: string;
        session_id: string;
        text_preview: string;
        classification: string;
        confidence: number;
      }>;
      cluster_colors: Record<number, string>;
      statistics: {
        total_points: number;
        n_clusters: number;
        n_outliers: number;
        dimensions: number;
      };
      coordinates: {
        tsne_2d: Array<[number, number]>;
        pca_2d: Array<[number, number]>;
        pca_3d: Array<[number, number, number]>;
      };
    };
    tenant_id: string;
    execution_time: number;
    error?: string;
  }> {
    console.log('📊 [DEBUG] ApiService.getClusteringStatistics() - Avvio richiesta');
    console.log('📊 [DEBUG] Tenant:', tenantId);
    console.log('📊 [DEBUG] Time range:', timeRange);
    
    try {
      // Costruisci URL con query parameters per richiesta GET
      let url = `${API_BASE_URL}/statistics/${tenantId}/clustering`;
      const params = new URLSearchParams();
      
      // Calcola giorni se timeRange è fornito
      if (timeRange?.start_date && timeRange?.end_date) {
        const startDate = new Date(timeRange.start_date);
        const endDate = new Date(timeRange.end_date);
        const diffTime = Math.abs(endDate.getTime() - startDate.getTime());
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        params.append('days_back', diffDays.toString());
      } else {
        // Default: ultimi 30 giorni
        params.append('days_back', '30');
      }
      
      // Parametri di default per le statistiche clustering
      params.append('include_visualizations', 'true');
      params.append('sample_limit', '5000');
      
      url += `?${params.toString()}`;
      
      console.log('📊 [DEBUG] URL chiamata:', url);
      console.log('📊 [DEBUG] Metodo: GET (correzione da POST)');
      
      const response = await axios.get(url);
      console.log('✅ [DEBUG] Statistiche clustering completate:', response.status);
      console.log('✅ [DEBUG] Dati statistiche:', response.data);
      
      return response.data;
    } catch (error) {
      console.error('❌ [DEBUG] Errore statistiche clustering:', error);
      
      if (axios.isAxiosError(error)) {
        const errorData = error.response?.data;
        return {
          success: false,
          error: errorData?.error || error.message,
          tenant_id: tenantId,
          execution_time: 0,
          visualization_data: {
            points: [],
            cluster_colors: {},
            statistics: { total_points: 0, n_clusters: 0, n_outliers: 0, dimensions: 0 },
            coordinates: { tsne_2d: [], pca_2d: [], pca_3d: [] }
          }
        };
      }
      throw error;
    }
  }

  /**
   * Sincronizza tenant dal database remoto al database locale
   * Importa in locale MySQL i tenant che non sono già presenti
   * 
   * @returns Risultato della sincronizzazione con statistiche
   * 
   * Autore: Valerio Bignardi
   * Data: 2025-08-27
   * Descrizione: Sincronizzazione automatica tenant remoti in locale
   */
  async syncTenants(): Promise<{
    success: boolean;
    imported_count: number;
    total_remote: number;
    total_local_before: number;
    total_local_after: number;
    errors: string[];
    timestamp: string;
    error?: string;
  }> {
    console.log('🔄 [DEBUG] ApiService.syncTenants() - Avvio sincronizzazione');
    
    try {
      const response = await axios.post(`${API_BASE_URL}/tenants/sync`, {});
      console.log('✅ [DEBUG] Sincronizzazione tenant completata:', response.data);
      
      return response.data;
      
    } catch (error) {
      console.error('❌ [DEBUG] Errore sincronizzazione tenant:', error);
      
      if (axios.isAxiosError(error)) {
        const errorData = error.response?.data;
        return {
          success: false,
          error: errorData?.error || error.message,
          imported_count: 0,
          total_remote: 0,
          total_local_before: 0,
          total_local_after: 0,
          errors: [errorData?.error || error.message],
          timestamp: new Date().toISOString()
        };
      }
      throw error;
    }
  }

  // 🆕 CLUSTERING VERSIONING ENDPOINTS

  /**
   * Ottiene la cronologia dei risultati di clustering per un tenant
   * 
   * @param tenantId - ID del tenant
   * @param limit - Numero massimo di record da recuperare (default: 50)
   * @returns Promise con lista di risultati clustering storici
   */
  async getClusteringHistory(tenantId: string, limit: number = 50): Promise<{
    success: boolean;
    data: Array<{
      id: number;
      version_number: number;
      created_at: string;
      n_clusters: number;
      n_outliers: number;
      silhouette_score: number;
      execution_time: number;
      parameters_summary: string;
    }>;
    total_versions: number;
    error?: string;
  }> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/clustering/${tenantId}/history`, {
        params: { limit }
      })
    );
  }

  /**
   * Recupera un risultato specifico di clustering per tenant e versione
   * 
   * @param tenantId - ID del tenant
   * @param versionNumber - Numero della versione
   * @returns Promise con risultato clustering completo
   */
  async getClusteringVersion(tenantId: string, versionNumber: number): Promise<{
    success: boolean;
    data: {
      id: number;
      version_number: number;
      tenant_id: string;
      created_at: string;
      results_data: any;  // Cambiato da results_json a results_data
      parameters_data: any; // Cambiato da parameters_json a parameters_data
      n_clusters: number;
      n_outliers: number;
      silhouette_score: number;
      execution_time: number;
    };
    error?: string;
  }> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/clustering/${tenantId}/version/${versionNumber}`)
    );
  }

  /**
   * Ottiene l'ultimo risultato di clustering per un tenant
   * 
   * @param tenantId - ID del tenant
   * @returns Promise con ultimo risultato clustering
   */
  async getLatestClusteringResult(tenantId: string): Promise<{
    success: boolean;
    data: {
      id: number;
      version_number: number;
      tenant_id: string;
      created_at: string;
      results_json: any;
      parameters_json: any;
      n_clusters: number;
      n_outliers: number;
      silhouette_score: number;
      execution_time: number;
    } | null;
    error?: string;
  }> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/clustering/${tenantId}/latest`)
    );
  }

  /**
   * Confronta due versioni di risultati clustering
   * 
   * @param tenantId - ID del tenant
   * @param version1 - Prima versione da confrontare
   * @param version2 - Seconda versione da confrontare  
   * @returns Promise con confronto dettagliato
   */
  async compareClusteringVersions(tenantId: string, version1: number, version2: number): Promise<{
    success: boolean;
    tenant_id: string;
    version1: {
      number: number;
      data: any;
      parameters: any;
      metadata: {
        created_at: string;
        execution_time: number;
      };
    };
    version2: {
      number: number;
      data: any;
      parameters: any;
      metadata: {
        created_at: string;
        execution_time: number;
      };
    };
    comparison_metrics: {
      clusters_diff: number;
      outliers_diff: number;
      ratio_diff: number;
      silhouette_diff: number;
      execution_time_diff: number;
    };
    error?: string;
  }> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/clustering/${tenantId}/compare/${version1}/${version2}`)
    );
  }

  /**
   * Ottiene trend delle metriche clustering nel tempo
   * 
   * @param tenantId - ID del tenant
   * @param days - Numero di giorni da analizzare (default: 30)
   * @returns Promise con dati trend
   */
  async getClusteringMetricsTrend(tenantId: string, days: number = 30): Promise<{
    success: boolean;
    data: {
      trend_data: Array<{
        version_number: number;
        created_at: string;
        n_clusters: number;
        n_outliers: number;
        silhouette_score: number;
        davies_bouldin_score?: number;
        calinski_harabasz_score?: number;
        execution_time: number;
        clustering_ratio: number;
        n_conversations: number;
      }>;
      metrics_summary: {
        avg_clusters: number;
        avg_outliers: number;
        avg_silhouette: number;
        best_silhouette: number;
        best_version: number;
        total_versions: number;
      };
      has_data: boolean;
      tenant_id: string;
    };
    error?: string;
  }> {
    try {
      const response = await axios.get(`${API_BASE_URL}/clustering/${tenantId}/metrics-trend`, {
        params: { days }
      });
      
      // Il backend restituisce direttamente la struttura, la adattiamo al formato atteso
      const backendData = response.data;
      
      return {
        success: backendData.success || false,
        data: {
          trend_data: backendData.trend_data || [],
          metrics_summary: backendData.metrics_summary || {},
          has_data: backendData.has_data || false,
          tenant_id: backendData.tenant_id || tenantId
        }
      };
    } catch (error: any) {
      return {
        success: false,
        data: {
          trend_data: [],
          metrics_summary: {
            avg_clusters: 0,
            avg_outliers: 0,
            avg_silhouette: 0,
            best_silhouette: 0,
            best_version: 0,
            total_versions: 0
          },
          has_data: false,
          tenant_id: tenantId
        },
        error: error.response?.data?.message || error.message || 'Errore chiamata API'
      };
    }
  }

  /**
   * Carica le soglie Review Queue per un tenant
   * 
   * Args:
   *   tenantId: ID del tenant
   *   
   * Returns:
   *   Promise con dati soglie o errore
   *   
   * Data ultima modifica: 2025-09-03
   */
  async getReviewQueueThresholds(tenantId: string): Promise<{
    success: boolean;
    thresholds: Record<string, any>;
    tenant_id: string;
    config_source: 'default' | 'custom';
    last_updated: string;
  }> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/review-queue/${tenantId}/thresholds`)
    );
  }

  /**
   * Aggiorna le soglie Review Queue per un tenant
   * 
   * Args:
   *   tenantId: ID del tenant
   *   thresholds: Record con le nuove soglie
   *   
   * Returns:
   *   Promise con risultato aggiornamento
   *   
   * Data ultima modifica: 2025-09-03
   */
  async updateReviewQueueThresholds(tenantId: string, thresholds: Record<string, any>): Promise<{
    success: boolean;
    message: string;
    tenant_id: string;
    updated_thresholds: Record<string, any>;
    timestamp: string;
  }> {
    return this.handleRequest(
      axios.post(`${API_BASE_URL}/review-queue/${tenantId}/thresholds`, {
        thresholds
      })
    );
  }

  /**
   * Reset delle soglie Review Queue ai valori default
   * 
   * Args:
   *   tenantId: ID del tenant
   *   
   * Returns:
   *   Promise con risultato reset
   *   
   * Data ultima modifica: 2025-09-03
   */
  async resetReviewQueueThresholds(tenantId: string): Promise<{
    success: boolean;
    message: string;
    tenant_id: string;
    timestamp: string;
  }> {
    return this.handleRequest(
      axios.post(`${API_BASE_URL}/review-queue/${tenantId}/thresholds/reset`, {})
    );
  }
}

export const apiService = new ApiService();
export { ApiService };
