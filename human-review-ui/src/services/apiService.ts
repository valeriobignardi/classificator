import axios from 'axios';
import { ReviewCase, ReviewStats } from '../types/ReviewCase';

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

  async getReviewCases(tenant: string, limit: number = 20): Promise<{ cases: ReviewCase[]; total: number }> {
    return this.handleRequest(
      axios.get(`${API_BASE_URL}/review/${tenant}/cases`, {
        params: { limit }
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
}

export const apiService = new ApiService();
export { ApiService };
