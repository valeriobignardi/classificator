export interface ReviewCase {
  case_id: string;
  session_id: string;
  conversation_text: string;
  ml_prediction: string;
  ml_confidence: number;
  llm_prediction: string;
  llm_confidence: number;
  uncertainty_score: number;
  novelty_score: number;
  reason: string;
  created_at: string;
  tenant: string;
  cluster_id?: string;
  // 🆕 Nuovi campi per cluster organization
  is_representative?: boolean;
  propagated_from?: string;
  propagation_indicator?: string;
}

// 🆕 Nuovo tipo per cluster view
export interface ClusterCase {
  cluster_id: string;
  representative: ReviewCase;
  propagated_sessions: ReviewCase[];
  total_sessions: number;
  cluster_size: number;
}

export interface ReviewStats {
  review_queue: {
    pending_cases: number;
    cases_by_reason: Record<string, number>;
    total_capacity: number;
    queue_utilization: number;
    last_updated: string;
  };
  general: {
    total_cases: number;
    resolved_cases: number;
    disagreement_cases: number;
    low_confidence_cases: number;
  };
  novelty_detection: {
    total_embeddings: number;
    tenant_embeddings: number;
    memory_utilization: number;
    avg_embedding_dimension: number;
  };
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}
