export interface ReviewCase {
  case_id: string;
  session_id: string;
  conversation_text: string;
  classification: string;
  classification_method?: string;
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
  classification_type?: string;
  is_representative?: boolean;
  propagated_from?: string;
  propagation_indicator?: string;
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

