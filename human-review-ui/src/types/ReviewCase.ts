export interface ReviewCase {
  case_id: string;
  session_id: string;
  conversation_text: string;
  classification: string; // 🚨 FIX CRITICO: Campo classificazione principale dall'API
  classification_method?: string; // 🆕 Metodo di classificazione (LLM_STRUCTURED, ENSEMBLE, etc.)
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
  classification_type?: string; // 🆕 RAPPRESENTANTE, OUTLIER, PROPAGATO, NORMALE
  // 🆕 Nuovi campi per cluster organization
  is_representative?: boolean;
  propagated_from?: string;
  propagation_indicator?: string;
}

// 🆕 Nuovo tipo per cluster view
export interface ClusterCase {
  cluster_id: string;
  representative: ReviewCase;
  representatives?: ReviewCase[];
  propagated_sessions: ReviewCase[];
  total_sessions: number;
  cluster_size: number;
}

// UI helper types for cluster context navigation
export interface ClusterContextCase {
  case_id?: string;
  session_id: string;
  label: string;
  raw_label?: string;
  display_label?: string;
  status?: 'available' | 'in_review_queue' | 'reviewed';
  is_representative?: boolean;
  classification_type?: string;
  propagated_from?: string;
  confidence?: number;
  method?: string;
  human_decision?: string;
  resolved_at?: string;
  updated_at?: string;
  created_at?: string;
  conversation_text?: string;
}

export interface ClusterContextSummary {
  cluster_id: string;
  majority_label?: string;
  majority_label_raw?: string;
  majority_label_display?: string;
  total_cases: number;
  reviewed_cases: number;
  pending_cases: number;
  representatives: number;
  propagated: number;
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
