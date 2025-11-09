import axios from 'axios';
import { API_BASE_URL } from '../config';
import { ReviewCase, ReviewStats } from '../types/ReviewCase';
import { Tenant } from '../types/Tenant';

const http = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
});

async function handle<T>(p: Promise<any>): Promise<T> {
  const res = await p;
  return res.data?.success !== undefined ? (res.data.success ? res.data : Promise.reject(new Error(res.data.error || 'API error'))) : res.data;
}

export const api = {
  // Tenants
  async getTenants(): Promise<Tenant[]> {
    const data = await handle<{ tenants: Tenant[] }>(http.get('/tenants'));
    return data.tenants;
  },

  // Review
  async getReviewStats(tenant_id: string): Promise<{ stats: ReviewStats }> {
    return handle(http.get(`/review/${tenant_id}/stats`));
  },

  async getReviewCases(
    tenant_id: string,
    options?: { limit?: number; include_representatives?: boolean; include_propagated?: boolean; include_outliers?: boolean }
  ): Promise<{ cases: ReviewCase[]; total: number }>
  {
    const params = {
      limit: options?.limit ?? 20,
      include_representatives: options?.include_representatives ?? true,
      include_propagated: options?.include_propagated ?? true,
      include_outliers: options?.include_outliers ?? true,
    };
    return handle(http.get(`/review/${tenant_id}/cases`, { params }));
  },

  async getCaseDetail(tenant_id: string, case_id: string): Promise<{ case: ReviewCase }>{
    return handle(http.get(`/review/${tenant_id}/cases/${case_id}`));
  },

  async resolveCase(
    tenant_id: string,
    case_id: string,
    human_decision: string,
    confidence: number,
    notes?: string
  ): Promise<{ message: string }>{
    return handle(http.post(`/review/${tenant_id}/cases/${case_id}/resolve`, {
      human_decision,
      confidence,
      notes,
    }));
  },

  // Clustering params
  async getClusteringParameters(tenant_id: string): Promise<{ parameters: any }>{
    return handle(http.get(`/clustering/${tenant_id}/parameters`));
  },

  async updateClusteringParameters(tenant_id: string, params: any): Promise<{ parameters: any }>{
    return handle(http.post(`/clustering/${tenant_id}/parameters`, params));
  },

  // LLM configuration
  async getLLMModels(tenant_id: string): Promise<{ models: any[] }>{
    return handle(http.get(`/llm/models/${tenant_id}`));
  },
  async getLLMParameters(tenant_id: string): Promise<{ parameters: any; current_model: string; source: string }>{
    return handle(http.get(`/llm/parameters/${tenant_id}`));
  },
  async updateLLMParameters(tenant_id: string, parameters: any, model_name?: string): Promise<any>{
    return handle(http.put(`/llm/parameters/${tenant_id}`, { parameters, model_name }));
  },
  async resetLLMParameters(tenant_id: string): Promise<any>{
    return handle(http.post(`/llm/reset-parameters/${tenant_id}`));
  },
  async testLLMModel(tenant_id: string, model_name: string, parameters: any, test_prompt?: string): Promise<any>{
    return handle(http.post(`/llm/test-model/${tenant_id}`, { model_name, parameters, test_prompt }));
  },

  // Scheduler
  async getSchedulerStatus(): Promise<any>{
    return handle(http.get(`/scheduler/status`));
  },
  async startScheduler(): Promise<any>{
    return handle(http.post(`/scheduler/start`));
  },
  async stopScheduler(): Promise<any>{
    return handle(http.post(`/scheduler/stop`));
  },
  async runSchedulerNow(client_slug: string): Promise<any>{
    return handle(http.post(`/scheduler/run-now/${client_slug}`));
  },
  async getSchedulerConfig(tenant_identifier: string): Promise<{ config: any }>{
    return handle(http.get(`/scheduler/config/${tenant_identifier}`));
  },
  async setSchedulerConfig(tenant_identifier: string, cfg: { enabled: boolean; frequency_unit: string; frequency_value: number; start_at?: string | null }): Promise<{ config: any }>{
    return handle(http.post(`/scheduler/config/${tenant_identifier}`, cfg));
  },

  // Training files
  async listTrainingFiles(tenant_id: string): Promise<{ success: boolean; files: Array<{name: string; size: number; modified_at: string}> }>{
    return handle(http.get(`/training-files/${tenant_id}`));
  },
  async getTrainingFileContent(tenant_id: string, file: string, limit: number = 500): Promise<{ success: boolean; content: string }>{
    return handle(http.get(`/training-files/${tenant_id}/content`, { params: { file, limit } }));
  },
};
