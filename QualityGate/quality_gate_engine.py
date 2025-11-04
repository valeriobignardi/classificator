"""
QualityGate Engine - Sistema di controllo qualità per classificazioni
Determina quando una classificazione necessita di revisione umana basandosi su:
- Disaccordo ensemble ML vs LLM
- Soglie di confidenza
- Active learning (incertezza)
- Novità semantica
"""

import logging
import numpy as np
import sys
import os
import traceback
import shutil
import joblib
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Aggiungi path per la classe Tenant
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from tenant import Tenant

# Import della funzione centralizzata per conversione NumPy
from Utils.numpy_serialization import convert_numpy_types
import json

# Import EmbeddingManager per gestione dinamica embedder
try:
    from EmbeddingEngine.embedding_manager import embedding_manager
except ImportError:
    # Fallback per compatibilità
    embedding_manager = None

# Importa SemanticMemoryManager per novelty detection
try:
    from SemanticMemory.semantic_memory_manager import SemanticMemoryManager
except ImportError:
    # Se il modulo non è disponibile, usa None
    SemanticMemoryManager = None

@dataclass
class QualityDecision:
    """Decisione del quality gate per una classificazione"""
    needs_review: bool
    confidence_score: float
    reason: str
    ml_prediction: Optional[str]
    llm_prediction: Optional[str]
    uncertainty_score: float
    novelty_score: float
    metadata: Dict[str, Any]

@dataclass
class ReviewCase:
    """Caso che necessita di revisione umana"""
    case_id: str
    session_id: str
    conversation_text: str
    ml_prediction: str
    ml_confidence: float
    llm_prediction: str
    llm_confidence: float
    uncertainty_score: float
    novelty_score: float
    reason: str
    created_at: datetime
    tenant: str
    cluster_id: Optional[str] = None
    embedding: Optional[np.ndarray] = None

class QualityGateEngine:
    """
    Engine per il controllo qualità delle classificazioni.
    Implementa logiche di ensemble disagreement e active learning.
    """
    
    def __init__(self, 
                 tenant,  # OGGETTO TENANT OBBLIGATORIO
                 training_log_path: str = None,
                 config: Dict[str, Any] = None,
                 confidence_threshold: float = None,
                 disagreement_threshold: float = None,
                 uncertainty_threshold: float = None,
                 novelty_threshold: float = None):
        """
        Inizializza il QualityGateEngine per un specifico tenant.
        CAMBIO RADICALE: USA OGGETTO TENANT
        
        Args:
            tenant: Oggetto Tenant completo (OBBLIGATORIO)
            training_log_path: Percorso del file di log per le decisioni di training (opzionale)
            config: Configurazione aggiuntiva (opzionale)
            confidence_threshold: Soglia di confidenza personalizzata (sovrascrive config)
            disagreement_threshold: Soglia di disagreement personalizzata (sovrascrive config)
            uncertainty_threshold: Soglia di incertezza personalizzata (sovrascrive config)
            novelty_threshold: Soglia di novità personalizzata (sovrascrive config)
        """
        # VALIDA OGGETTO TENANT
        if not hasattr(tenant, 'tenant_id') or not hasattr(tenant, 'tenant_name') or not hasattr(tenant, 'tenant_slug'):
            raise TypeError("Il parametro 'tenant' deve essere un oggetto Tenant valido")
            
        self.tenant = tenant
        self.tenant_name = tenant.tenant_name  # Mantieni per compatibilità 
        self.training_log_path = training_log_path or f"training_decisions_{tenant.tenant_slug}.jsonl"
        
        # Usa config fornito o carica dal file
        if config is None:
            # Carica il config.yaml se non fornito
            try:
                import yaml
                import os
                config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
                with open(config_path, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
            except Exception as e:
                print(f"⚠️ Impossibile caricare config.yaml: {e}. Uso valori di default.")
                config = {}
        
        self.config = config
        self.quality_config = self.config.get('quality_gate', {})
        self.logger = logging.getLogger(__name__)
        
        # Soglie configurabili - i parametri espliciti sovrascrivono il config
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else self.quality_config.get('confidence_threshold', 0.7)
        self.disagreement_threshold = disagreement_threshold if disagreement_threshold is not None else self.quality_config.get('disagreement_threshold', 0.3)
        self.uncertainty_threshold = uncertainty_threshold if uncertainty_threshold is not None else self.quality_config.get('uncertainty_threshold', 0.5)
        self.novelty_threshold = novelty_threshold if novelty_threshold is not None else self.quality_config.get('novelty_threshold', 0.8)
        
        # Inizializza MongoReader per gestire review cases con OGGETTO TENANT
        from mongo_classification_reader import MongoClassificationReader
        self.mongo_reader = MongoClassificationReader(tenant=self.tenant)
        self.mongo_reader.connect()
        
        # Inizializza dizionario per pending reviews (in-memory cache)
        self.pending_reviews = {}
        
        # Inizializza SemanticMemoryManager per novelty detection
        try:
            if SemanticMemoryManager is not None:
                self.semantic_memory = SemanticMemoryManager(tenant=self.tenant)
                self.semantic_memory.load_semantic_memory()
                self.logger.info(f"SemanticMemoryManager inizializzato per novelty detection - Tenant: {self.tenant_name}")
            else:
                self.semantic_memory = None
                self.logger.info("SemanticMemoryManager non disponibile, novelty detection disabilitata")
        except Exception as e:
            self.logger.warning(f"Errore inizializzazione SemanticMemoryManager: {e}")
            self.semantic_memory = None
        
        self.logger.info(f"QualityGate inizializzato per tenant {self.tenant.tenant_name} con soglie: "
                        f"confidence={self.confidence_threshold}, disagreement={self.disagreement_threshold}, "
                        f"uncertainty={self.uncertainty_threshold}")
        self.logger.info(f"Log training: {self.training_log_path}")

    def _get_dynamic_embedder(self):
        """
        Ottiene embedder dinamico configurato per il tenant
        
        Returns:
            Embedder configurato per il tenant
        """
        # Usa direttamente l'oggetto Tenant della classe
        if not self.tenant:
            raise ValueError("QualityGateEngine deve essere inizializzato con oggetto Tenant valido")
        
        try:
            from EmbeddingEngine.simple_embedding_manager import simple_embedding_manager
            return simple_embedding_manager.get_embedder_for_tenant(self.tenant)
        except Exception as e:
            # AGGIORNAMENTO 2025-08-29: Fallback coerente con servizio Docker
            print(f"⚠️ QualityGate fallback embedder: {e}")
            try:
                from EmbeddingEngine.labse_remote_client import LaBSERemoteClient
                return LaBSERemoteClient(
                    service_url="http://localhost:8081", 
                    fallback_local=False  # 🚫 NESSUN FALLBACK LOCALE
                )
            except Exception as e:
                # 🚫 NESSUN FALLBACK LOCALE - Solo Docker service
                raise RuntimeError(f"Servizio Docker LaBSE richiesto ma non disponibile: {e}")

    def evaluate_classification(self, 
                              session_id: str,
                              conversation_text: str,
                              ml_result: Dict[str, Any],
                              llm_result: Dict[str, Any],
                              tenant: str,
                              embedding: Optional[np.ndarray] = None,
                              cluster_id: Optional[str] = None,
                              cluster_metadata: Optional[Dict[str, Any]] = None) -> QualityDecision:
        """
        Valuta se una classificazione necessita di revisione umana.
        
        Args:
            session_id: ID della sessione
            conversation_text: Testo della conversazione
            ml_result: Risultato del classificatore ML (label, confidence, probabilities)
            llm_result: Risultato del classificatore LLM (label, confidence, reasoning)
            tenant: Tenant di appartenenza
            embedding: Embedding della conversazione per novelty detection
            cluster_id: ID del cluster se disponibile
            cluster_metadata: Metadati completi del cluster (cluster_id, is_representative, propagated_from, etc.)
            
        Returns:
            QualityDecision con la decisione e i dettagli
        """
        try:
            ml_label = ml_result.get('predicted_label', '')
            ml_confidence = ml_result.get('confidence', 0.0)
            llm_label = llm_result.get('predicted_label', '')
            llm_confidence = llm_result.get('confidence', 0.0)
            
            # 1. Calcola disaccordo tra ensemble ML e LLM
            disagreement_score = self._calculate_disagreement(ml_result, llm_result)
            
            # 2. Calcola incertezza del modello ML (active learning)
            uncertainty_score = self._calculate_uncertainty(ml_result)
            
            # 3. Calcola novelty score se disponibile embedding
            novelty_score = self._calculate_novelty(embedding, tenant) if embedding is not None else 0.0
            
            # 4. Confidenza generale (minimo tra ML e LLM)
            overall_confidence = min(ml_confidence, llm_confidence)
            
            # 5. Determina se necessita revisione
            needs_review, reason = self._needs_human_review(
                disagreement_score, uncertainty_score, novelty_score, overall_confidence
            )
            
            # Converte tutti i valori NumPy in tipi Python nativi
            decision_data = convert_numpy_types({
                'needs_review': needs_review,
                'confidence_score': overall_confidence,
                'reason': reason,
                'ml_prediction': ml_label,
                'llm_prediction': llm_label,
                'uncertainty_score': uncertainty_score,
                'novelty_score': novelty_score,
                'metadata': {
                    'disagreement_score': disagreement_score,
                    'ml_confidence': ml_confidence,
                    'llm_confidence': llm_confidence,
                    'session_id': session_id,
                    'tenant': tenant,
                    'cluster_id': cluster_id
                }
            })

            # Determina la decisione finale basata su confidenza e disaccordo
            if needs_review:
                # Se necessita review, la decisione finale è incerta
                final_decision = {
                    "predicted_label": ml_label if ml_confidence > llm_confidence else llm_label,
                    "confidence": overall_confidence,
                    "method": "quality_gate_review_needed",
                    "reasoning": reason
                }
            else:
                # Auto-classificazione: usa la predizione con confidenza maggiore
                if ml_confidence >= llm_confidence:
                    final_decision = {
                        "predicted_label": ml_label,
                        "confidence": ml_confidence,
                        "method": "ensemble_ml_primary",
                        "reasoning": f"ML prediction selected (conf: {ml_confidence:.3f} vs LLM: {llm_confidence:.3f})"
                    }
                else:
                    final_decision = {
                        "predicted_label": llm_label,
                        "confidence": llm_confidence,
                        "method": "ensemble_llm_primary",
                        "reasoning": f"LLM prediction selected (conf: {llm_confidence:.3f} vs ML: {ml_confidence:.3f})"
                    }

            # NUOVO: Salva SEMPRE il risultato completo in MongoDB (includi embedding, se presente)
            # Determina un nome modello minimale per tracciare la provenienza embedding
            emb_model_name = 'no_embedding' if embedding is None else 'unknown_embedder'
            mongo_saved = self.mongo_reader.save_classification_result(
                session_id=session_id,
                client_name=tenant,
                ml_result=ml_result,
                llm_result=llm_result,
                final_decision=final_decision,
                conversation_text=conversation_text,
                needs_review=needs_review,
                review_reason=reason if needs_review else None,
                cluster_metadata=cluster_metadata,  # 🔧 FIX: Aggiunti metadati cluster
                embedding=embedding,
                embedding_model=emb_model_name
            )
            
            if mongo_saved:
                self.logger.debug(f"✅ Risultato classificazione salvato in MongoDB: {session_id} -> {final_decision['predicted_label']}")
            else:
                self.logger.warning(f"⚠️ Impossibile salvare risultato in MongoDB per {session_id}")

            decision = QualityDecision(
                needs_review=decision_data['needs_review'],
                confidence_score=decision_data['confidence_score'],
                reason=decision_data['reason'],
                ml_prediction=decision_data['ml_prediction'],
                llm_prediction=decision_data['llm_prediction'],
                uncertainty_score=decision_data['uncertainty_score'],
                novelty_score=decision_data['novelty_score'],
                metadata=decision_data['metadata']
            )
            
            # Se necessita revisione, crea anche il caso di review legacy (per backward compatibility)
            if needs_review:
                self._create_review_case(session_id, conversation_text, ml_result, llm_result,
                                       uncertainty_score, novelty_score, reason, tenant,
                                       cluster_id, embedding)
            
            self.logger.debug(f"QualityGate decision per {session_id}: needs_review={needs_review}, "
                            f"reason='{reason}', confidence={overall_confidence:.3f}")
            
            # Aggiorna la memoria semantica con questo nuovo embedding per future novelty detection
            self._update_semantic_memory(session_id, embedding, tenant, ml_label if overall_confidence > 0.7 else None)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Errore in evaluate_classification per {session_id}: {e}")
            # In caso di errore, richiedi sempre revisione per sicurezza
            return QualityDecision(
                needs_review=True,
                confidence_score=0.0,
                reason=f"Errore nel quality gate: {str(e)}",
                ml_prediction=ml_result.get('label', ''),
                llm_prediction=llm_result.get('label', ''),
                uncertainty_score=1.0,
                novelty_score=1.0,
                metadata={'error': str(e)}
            )

    def _calculate_disagreement(self, ml_result: Dict[str, Any], llm_result: Dict[str, Any]) -> float:
        """
        Calcola il grado di disaccordo tra ML e LLM.
        Considera sia le etichette che le distribuzioni di probabilità.
        """
        ml_label = ml_result.get('predicted_label', '') or ml_result.get('label', '')
        llm_label = llm_result.get('predicted_label', '') or llm_result.get('label', '')
        
        # Disaccordo binario sulle etichette
        label_disagreement = 1.0 if ml_label != llm_label else 0.0
        
        # Se abbiamo probabilità ML, calcola divergenza
        ml_probs = ml_result.get('probabilities', {})
        if ml_probs and llm_label in ml_probs:
            # Probabilità che ML assegna alla predizione LLM
            ml_prob_for_llm = ml_probs.get(llm_label, 0.0)
            # Se ML assegna bassa probabilità alla predizione LLM, c'è disaccordo
            prob_disagreement = 1.0 - ml_prob_for_llm
            return max(label_disagreement, prob_disagreement)
        
        return label_disagreement

    def _calculate_uncertainty(self, ml_result: Dict[str, Any]) -> float:
        """
        Calcola l'incertezza del modello ML usando l'entropia delle probabilità.
        Alta entropia = alta incertezza = candidato per active learning.
        """
        probs = ml_result.get('probabilities', {})
        if not probs:
            return 0.0
        
        # Calcola entropia normalizzata
        prob_values = list(probs.values())
        if len(prob_values) <= 1:
            return 0.0
        
        # Entropia di Shannon
        entropy = -sum(p * np.log2(p + 1e-10) for p in prob_values if p > 0)
        # Normalizza per il numero di classi
        max_entropy = np.log2(len(prob_values))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy

    def _calculate_novelty(self, embedding: np.ndarray, tenant: str) -> float:
        """
        Calcola novelty score basandosi sulla distanza semantica dagli embeddings esistenti
        nella memoria semantica. Utilizza il SemanticMemoryManager per determinare
        quanto una nuova conversazione è semanticamente diversa da quelle già viste.
        
        Args:
            embedding: Vector embedding della conversazione
            tenant: Nome del tenant per filtrare la memoria
            
        Returns:
            float: Score di novelty [0.0, 1.0] dove 1.0 = completamente nuovo
        """
        if self.semantic_memory is None:
            # Se non abbiamo memoria semantica disponibile, considera tutto come poco nuovo
            # per evitare troppi falsi positivi
            self.logger.debug("SemanticMemoryManager non disponibile, assumo bassa novelty")
            return 0.2
        
        if embedding is None or len(embedding) == 0:
            self.logger.warning("Embedding vuoto o nullo per calcolo novelty")
            return 0.0
        
        try:
            # Usa il SemanticMemoryManager per calcolare la novelty
            novelty_score = self.semantic_memory.calculate_novelty_score(
                embedding=embedding, 
                tenant=tenant
            )
            
            self.logger.debug(f"Novelty score calcolato: {novelty_score:.3f} per tenant {tenant}")
            return novelty_score
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo novelty tramite SemanticMemoryManager: {e}")
            # In caso di errore, restituisci un valore conservativo
            return 0.3

    def _update_semantic_memory(self, session_id: str, embedding: np.ndarray, 
                               tenant: str, label: str = None):
        """
        Aggiorna la memoria semantica con un nuovo embedding per migliorare
        future analisi di novelty detection.
        
        Args:
            session_id: ID della sessione
            embedding: Vector embedding della conversazione
            tenant: Nome del tenant
            label: Etichetta assegnata (se disponibile e confidenza alta)
        """
        if self.semantic_memory is None or embedding is None:
            return
        
        try:
            success = self.semantic_memory.add_embedding_to_memory(
                session_id=session_id,
                embedding=embedding,
                tenant=tenant,
                label=label
            )
            
            if success:
                self.logger.debug(f"Embedding aggiunto alla memoria semantica: {session_id}")
            else:
                self.logger.warning(f"Fallimento aggiunta embedding alla memoria: {session_id}")
                
        except Exception as e:
            self.logger.error(f"Errore aggiornamento memoria semantica per {session_id}: {e}")

    def _needs_human_review(self, disagreement: float, uncertainty: float, 
                          novelty: float, confidence: float) -> Tuple[bool, str]:
        """
        Determina se la classificazione necessita di revisione umana.
        
        Returns:
            Tuple[bool, str]: (needs_review, reason)
        """
        reasons = []
        
        # Controllo 1: Disaccordo tra ensemble
        if disagreement > self.disagreement_threshold:
            reasons.append(f"disaccordo ensemble (score: {disagreement:.3f})")
        
        # Controllo 2: Bassa confidenza generale
        if confidence < self.confidence_threshold:
            reasons.append(f"bassa confidenza (score: {confidence:.3f})")
        
        # Controllo 3: Alta incertezza (active learning)
        if uncertainty > self.uncertainty_threshold:
            reasons.append(f"alta incertezza modello (score: {uncertainty:.3f})")
        
        # Controllo 4: Novità semantica
        if novelty > self.novelty_threshold:
            reasons.append(f"alta novità semantica (score: {novelty:.3f})")
        
        needs_review = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "classificazione affidabile"
        
        return needs_review, reason

    def _create_review_case(self, session_id: str, conversation_text: str,
                          ml_result: Dict[str, Any], llm_result: Dict[str, Any],
                          uncertainty_score: float, novelty_score: float,
                          reason: str, tenant: str, cluster_id: Optional[str],
                          embedding: Optional[np.ndarray]):
        """Crea un caso per la revisione umana"""
        case_id = f"{tenant}_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Converte tutti i valori NumPy prima di creare il ReviewCase
        case_data = convert_numpy_types({
            'ml_prediction': ml_result.get('predicted_label', ''),
            'ml_confidence': ml_result.get('confidence', 0.0),
            'llm_prediction': llm_result.get('predicted_label', ''),
            'llm_confidence': llm_result.get('confidence', 0.0),
            'uncertainty_score': uncertainty_score,
            'novelty_score': novelty_score
        })
        
        review_case = ReviewCase(
            case_id=case_id,
            session_id=session_id,
            conversation_text=conversation_text,
            ml_prediction=case_data['ml_prediction'],
            ml_confidence=case_data['ml_confidence'],
            llm_prediction=case_data['llm_prediction'],
            llm_confidence=case_data['llm_confidence'],
            uncertainty_score=case_data['uncertainty_score'],
            novelty_score=case_data['novelty_score'],
            reason=reason,
            created_at=datetime.now(),
            tenant=tenant,
            cluster_id=cluster_id,
            embedding=embedding
        )
        
        # MODIFICATO: Non salviamo più in memory, tutto va in MongoDB tramite il nuovo save_classification_result
        # Il caso è già stato creato dal calling method evaluate_classification()
        self.logger.info(f"Creato caso di revisione {case_id} per sessione {session_id}: {reason}")

    def get_pending_reviews(self, tenant: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Scopo: Ottiene i casi in attesa di revisione da MongoDB
        
        Parametri input:
            - tenant: Nome del tenant (opzionale)
            - limit: Numero massimo di casi da recuperare
            
        Output:
            - Lista di dizionari con i casi pending review
            
        Ultimo aggiornamento: 2025-08-21
        """
        try:
            client_name = tenant or self.tenant_name
            sessions = self.mongo_reader.get_pending_review_sessions(client_name, limit)
            
            # Converte in formato compatibile con ReviewCase per l'API
            review_cases = []
            for session in sessions:
                review_case = {
                    'case_id': session['case_id'],
                    'session_id': session['session_id'],
                    'tenant': client_name,
                    'conversation_text': session['conversation_text'],
                    'ml_prediction': session.get('classification', ''),
                    'ml_confidence': session.get('confidence', 0.0),
                    'llm_prediction': session.get('classification', ''),
                    'llm_confidence': session.get('confidence', 0.0),
                    'uncertainty_score': 0.8,  # Default per ora
                    'novelty_score': 0.5,      # Default per ora  
                    'reason': session.get('review_reason', 'manual_review'),
                    'created_at': session.get('timestamp'),
                    'review_status': session.get('review_status', 'pending')
                }
                review_cases.append(review_case)
            
            return review_cases
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero pending reviews: {e}")
            return []

    def resolve_review_case(self, case_id: str, human_decision: str, 
                          human_confidence: float = 1.0, notes: str = "") -> Dict[str, Any]:
        """
        Scopo: Risolve un caso di revisione con la decisione umana usando MongoDB
               e propaga la decisione ai membri dello stesso cluster se è rappresentante
        
        Parametri input:
            - case_id: ID del caso (MongoDB _id)
            - human_decision: Etichetta scelta dall'umano
            - human_confidence: Confidenza della decisione umana
            - notes: Note aggiuntive
            
        Output:
            - Dizionario con risultati dettagliati della risoluzione e propagazione
            
        Ultimo aggiornamento: 2025-08-27
        """
        
        # Usa il nuovo metodo con propagazione cluster
        result = self.mongo_reader.resolve_review_session_with_cluster_propagation(
            case_id=case_id,
            client_name=self.tenant_name,
            human_decision=human_decision,
            human_confidence=human_confidence,
            human_notes=notes
        )
        
        if not result.get("case_resolved", False):
            error = result.get("error", "Motivo sconosciuto")
            self.logger.warning(f"Caso di revisione {case_id} non risolto: {error}")
            return result
        
        # Log della decisione per training futuro (manteniamo il log file)
        decision_log = convert_numpy_types({
            'case_id': case_id,
            'session_id': case_id,  # Per ora usiamo case_id come session_id
            'tenant': self.tenant_name,
            'ml_prediction': '',  # Recupereremo dai dati MongoDB se necessario
            'ml_confidence': 0.0,
            'llm_prediction': '',
            'llm_confidence': 0.0,
            'human_decision': human_decision,
            'human_confidence': human_confidence,
            'uncertainty_score': 0.8,  # Default
            'novelty_score': 0.5,      # Default
            'reason': 'mongo_review_with_cluster_propagation',
            'notes': notes,
            'resolved_at': datetime.now().isoformat(),
            'cluster_propagation': {
                'is_representative': result.get("is_representative", False),
                'propagated_cases': result.get("propagated_cases", 0),
                'cluster_id': result.get("cluster_id")
            }
        })
        
        # Log delle informazioni di propagazione
        propagated_count = result.get("propagated_cases", 0)
        is_representative = result.get("is_representative", False)
        cluster_id = result.get("cluster_id")
        
        if is_representative and propagated_count > 0:
            self.logger.info(f"🔄 Caso {case_id} risolto come RAPPRESENTANTE: '{human_decision}' "
                           f"propagato a {propagated_count} membri del cluster {cluster_id}")
        elif is_representative and propagated_count == 0:
            self.logger.info(f"🎯 Caso {case_id} risolto come RAPPRESENTANTE: '{human_decision}' "
                           f"(nessun membro da propagare nel cluster {cluster_id})")
        else:
            self.logger.info(f"📝 Caso {case_id} risolto: '{human_decision}' "
                           f"(caso non-rappresentante)")
        
        # Log in JSON per training futuro
        with open(self.training_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(decision_log, ensure_ascii=False) + '\n')
        
        # Aggiorna cache dei tag umani
        self._update_human_tags_cache(human_decision)
        
        self.logger.info(f"Caso {case_id} risolto: {human_decision} (confidenza: {human_confidence})")
        
        return result
    
    def _save_human_decision_to_database(self, case: ReviewCase, human_decision: str, 
                                       human_confidence: float, notes: str) -> bool:
        """
        Salva la decisione umana in MongoDB (sistema unificato).
        
        Args:
            case: Il caso di revisione risolto
            human_decision: Etichetta scelta dall'umano
            human_confidence: Confidenza della decisione umana
            notes: Note aggiuntive
            
        Returns:
            bool: True se salvato con successo
        """
        try:
            # 🔧 FIX: Ricostruisci cluster_metadata da cluster_id disponibile
            cluster_metadata = None
            if hasattr(case, 'cluster_id') and case.cluster_id:
                cluster_metadata = {
                    'cluster_id': case.cluster_id,
                    'method': 'human_review_preserved'  # Metodo per tracciabilità
                }
                # TODO: In futuro si potrebbe recuperare metadata completi dal DB
            
            # Usa MongoDB come sistema unificato per tutte le classificazioni
            # Includi embedding se presente nel caso di review
            emb_model_name = 'no_embedding' if not getattr(case, 'embedding', None) else 'unknown_embedder'
            success = self.mongo_reader.save_classification_result(
                session_id=case.session_id,
                client_name=case.tenant,
                ml_result={
                    'predicted_label': case.ml_prediction,
                    'confidence': case.ml_confidence
                } if case.ml_prediction else None,
                llm_result={
                    'predicted_label': case.llm_prediction,
                    'confidence': case.llm_confidence
                } if case.llm_prediction else None,
                final_decision={
                    'predicted_label': human_decision,
                    'confidence': human_confidence,
                    'method': 'HUMAN_REVIEW',
                    'reasoning': f"Human review decision. Original ML: {case.ml_prediction} "
                                f"(conf: {case.ml_confidence:.3f}), LLM: {case.llm_prediction} "
                                f"(conf: {case.llm_confidence:.3f}). Reason: {case.reason}"
                },
                conversation_text=case.conversation_text,
                needs_review=False,  # Risolto dall'umano
                classified_by='human_supervisor',
                notes=notes,
                cluster_metadata=cluster_metadata,  # 🔧 FIX: Aggiunti metadati cluster
                embedding=getattr(case, 'embedding', None),
                embedding_model=emb_model_name
            )
                
            if success:
                self.logger.info(f"✅ Decisione umana salvata in MongoDB: "
                               f"session_id={case.session_id}, tag={human_decision}, "
                               f"confidence={human_confidence:.3f}")
            else:
                self.logger.error(f"❌ Errore nel salvataggio MongoDB per "
                                f"session_id={case.session_id}")
                
            return success
                
        except Exception as e:
            self.logger.error(f"❌ Errore nel salvataggio decisione umana: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def add_to_review_queue(self, session_id: str, conversation_text: str, 
                           reason: str = "manual_addition",
                           ml_prediction: str = "",
                           ml_confidence: float = 0.0,
                           llm_prediction: str = "manual_review",
                           llm_confidence: float = 0.0) -> str:
        """
        Aggiunge manualmente una sessione alla coda di revisione.
        
        Args:
            session_id: ID della sessione
            conversation_text: Testo della conversazione
            reason: Motivo dell'aggiunta alla coda
            ml_prediction: Predizione ML (opzionale)
            ml_confidence: Confidenza ML (opzionale)
            llm_prediction: Predizione LLM (opzionale)
            llm_confidence: Confidenza LLM (opzionale)
            
        Returns:
            case_id: ID del caso creato nella review queue
        """
        try:
            # Genera un case_id unico
            case_id = f"{self.tenant_name}_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_manual"
            
            # Crea il caso di review
            review_case = ReviewCase(
                case_id=case_id,
                session_id=session_id,
                conversation_text=conversation_text,
                ml_prediction=ml_prediction,
                ml_confidence=ml_confidence,
                llm_prediction=llm_prediction,
                llm_confidence=llm_confidence,
                uncertainty_score=0.5,
                novelty_score=0.3,
                reason=reason,
                created_at=datetime.now(),
                tenant=self.tenant_name,
                cluster_id=-1
            )
            
            # Aggiungi alla cache in-memory
            self.pending_reviews[case_id] = review_case
            
            # Salva anche in MongoDB per persistenza
            success = self.mongo_reader.mark_session_for_review(
                session_id=session_id,
                client_name=self.tenant_name,
                review_reason=reason,
                conversation_text=conversation_text
            )
            
            if success:
                self.logger.info(f"✅ Sessione {session_id} aggiunta alla review queue come {case_id}")
            else:
                self.logger.warning(f"⚠️ Sessione {session_id} aggiunta alla cache ma non salvata in MongoDB")
            
            return case_id
            
        except Exception as e:
            self.logger.error(f"❌ Errore nell'aggiunta alla review queue: {e}")
            raise

    def _update_human_tags_cache(self, tag: str):
        """
        Aggiorna la cache locale dei tag utilizzati nelle decisioni umane.
        
        Args:
            tag: Tag utilizzato nella decisione umana
        """
        if not hasattr(self, '_human_tags_cache'):
            self._human_tags_cache = {}
        
        # Normalizza il tag
        normalized_tag = tag.strip()
        if normalized_tag:
            if normalized_tag in self._human_tags_cache:
                self._human_tags_cache[normalized_tag] += 1
            else:
                self._human_tags_cache[normalized_tag] = 1
        
        # RIADDESTRAMENTO AUTOMATICO RIABILITATO per ML training
        self._check_and_trigger_retraining()
    
    def _check_and_trigger_retraining(self):
        """
        Controlla se è necessario riaddestramento automatico basato sul numero
        di decisioni umane nel log. Riaddestra ogni 5 decisioni.
        """
        try:
            # Conta le decisioni nel log
            decision_count = self._count_human_decisions()
            
            # Controlla se abbiamo abbastanza decisioni per il riaddestramento
            retraining_threshold = self.quality_config.get('retraining_threshold', 5)
            
            if decision_count > 0 and decision_count % retraining_threshold == 0:
                # Verifica che il file di log esista e abbia dati recenti
                if self._has_new_training_data():
                    self.logger.info(f"Trigger riaddestramento automatico: {decision_count} decisioni umane nel log")
                    success = self._retrain_ml_model()
                    if success:
                        self.logger.info("✅ Riaddestramento automatico completato con successo")
                    else:
                        self.logger.error("❌ Riaddestramento automatico fallito")
        except Exception as e:
            self.logger.error(f"Errore nel controllo riaddestramento automatico: {e}")
    
    def _count_human_decisions(self) -> int:
        """
        Conta il numero totale di decisioni umane nel log.
        
        Returns:
            Numero di decisioni nel log
        """
        try:
            if not os.path.exists(self.training_log_path):
                return 0
            
            count = 0
            with open(self.training_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        count += 1
            
            return count
        except Exception as e:
            self.logger.error(f"Errore nel conteggio decisioni umane: {e}")
            return 0
    
    def _has_new_training_data(self) -> bool:
        """
        Verifica se ci sono nuovi dati di training dal ultimo riaddestramento.
        
        Returns:
            True se ci sono nuovi dati
        """
        try:
            if not os.path.exists(self.training_log_path):
                return False
            
            # Controlla se il file è stato modificato di recente
            from pathlib import Path
            log_file = Path(self.training_log_path)
            
            # Se il file è più piccolo di 10 bytes, non ha abbastanza dati
            if log_file.stat().st_size < 10:
                return False
            
            # Verifica se abbiamo almeno alcune decisioni valide
            valid_decisions = 0
            with open(self.training_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        decision = json.loads(line.strip())
                        if decision.get('human_decision') and decision.get('session_id'):
                            valid_decisions += 1
                    except json.JSONDecodeError:
                        continue
            
            return valid_decisions >= 2  # Almeno 2 decisioni valide
        except Exception as e:
            self.logger.error(f"Errore nel controllo nuovi dati di training: {e}")
            return False
    
    def _retrain_ml_model(self) -> bool:
        """
        Riaddestra il modello ML usando le decisioni umane dal log.
        
        Returns:
            True se il riaddestramento è completato con successo
        """
        try:
            self.logger.info("🔄 Inizio riaddestramento automatico del modello ML...")
            
            # Determina se è il primo addestramento o un riaddestramento
            is_first_training = self._is_first_ml_training()
            
            if is_first_training:
                self.logger.info("🚀 PRIMO ADDESTRAMENTO: Uso review umane + classificazioni LLM")
                training_data = self._load_all_training_data_for_first_training()
            else:
                self.logger.info("🔄 RIADDESTRAMENTO: Uso solo review umane")
                training_data = self._load_human_decisions_for_training()
            
            if not training_data:
                self.logger.warning("Nessun dato di training trovato")
                return False
            
            # 2. Prepara dati per training
            X_new, y_new = self._prepare_training_data(training_data)
            if X_new is None or len(X_new) == 0:
                self.logger.warning("Nessun dato di training valido preparato")
                return False
            
            # 3. Carica modello esistente e combinare con nuovi dati
            success = self._update_ml_model_with_new_data(X_new, y_new)
            
            if success:
                training_type = "primo addestramento" if is_first_training else "riaddestramento"
                self.logger.info(f"✅ Modello {training_type} completato con {len(X_new)} esempi")
                # Marca il log come processato (opzionale)
                self._mark_training_data_processed()
                return True
            else:
                self.logger.error("❌ Fallimento nel riaddestramento del modello")
                return False
                
        except Exception as e:
            self.logger.error(f"Errore nel riaddestramento del modello: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _is_first_ml_training(self) -> bool:
        """
        Determina se questo è il primo addestramento ML o un riaddestramento.
        
        Verifica:
        1. Esistenza di modelli ML salvati precedentemente
        2. Se il ML ensemble è stato allenato in precedenza
        
        Returns:
            True se è il primo addestramento, False se è un riaddestramento
        """
        try:
            # Verifica se il ML ensemble ha mai fatto training
            if hasattr(self.ensemble_classifier, 'ml_classifier') and self.ensemble_classifier.ml_classifier:
                # Verifica se il classificatore ML ha modelli salvati
                ml_classifier = self.ensemble_classifier.ml_classifier
                
                # Controlla se esistono file di modelli salvati
                if hasattr(ml_classifier, 'models_dir') and ml_classifier.models_dir:
                    models_dir = Path(ml_classifier.models_dir)
                    if models_dir.exists():
                        # Cerca file di modelli (potrebbero essere .pkl, .joblib, .json, etc.)
                        model_files = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.joblib')) + list(models_dir.glob('*.json'))
                        if model_files:
                            self.logger.info(f"Trovati {len(model_files)} file di modelli esistenti - RIADDESTRAMENTO")
                            return False
                
                # Verifica se il classificatore ML ha attributi che indicano training precedente
                if hasattr(ml_classifier, 'is_trained') and callable(ml_classifier.is_trained):
                    if ml_classifier.is_trained():
                        self.logger.info("ML classifier è già stato addestrato - RIADDESTRAMENTO")
                        return False
                
                # Verifica se ci sono attributi di modelli salvati in memoria
                if hasattr(ml_classifier, 'models') and ml_classifier.models:
                    if isinstance(ml_classifier.models, dict) and ml_classifier.models:
                        self.logger.info("Trovati modelli in memoria - RIADDESTRAMENTO")
                        return False
            
            # Verifica file di training log per determinare se ci sono stati addestramenti precedenti
            # Se esiste il log e ha voci, potrebbe essere un riaddestramento
            if os.path.exists(self.training_log_path):
                try:
                    with open(self.training_log_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # Se il log è sostanzioso (> 10 righe), probabilmente non è il primo training
                        if len(lines) > 10:
                            self.logger.info(f"Log training esistente con {len(lines)} righe - Possibile RIADDESTRAMENTO")
                            # Non possiamo essere certi, verifichiamo se ci sono state classificazioni ML
                            return self._check_if_ml_has_ever_classified()
                except Exception as e:
                    self.logger.warning(f"Errore lettura training log: {e}")
            
            self.logger.info("Nessun evidence di training precedente - PRIMO ADDESTRAMENTO")
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nel determinare tipo di training: {e}")
            # In caso di errore, assume primo addestramento per sicurezza
            return True
    
    def _check_if_ml_has_ever_classified(self) -> bool:
        """
        Verifica se il ML ha mai fatto classificazioni controllando il database MongoDB.
        
        Returns:
            True se è il primo training (ML non ha mai classificato)
            False se è un riaddestramento (ML ha già classificato)
        """
        try:
            from mongo_classification_reader import MongoClassificationReader
            mongo_reader = MongoClassificationReader(tenant=self.tenant)
            mongo_reader.connect()
            
            try:
                # Cerca classificazioni fatte dal ML (non solo LLM)
                # Cerca documenti con classification_method che contiene 'ml' o 'ensemble'
                query = {
                    'client_name': self.tenant.tenant_slug,
                    '$or': [
                        {'classification_method': {'$regex': 'ml', '$options': 'i'}},
                        {'classification_method': {'$regex': 'ensemble', '$options': 'i'}},
                        {'ml_prediction': {'$exists': True, '$ne': None}}
                    ]
                }
                
                ml_classifications = mongo_reader.collection.count_documents(query)
                
                if ml_classifications > 0:
                    self.logger.info(f"Trovate {ml_classifications} classificazioni ML esistenti - RIADDESTRAMENTO")
                    return False
                else:
                    self.logger.info("Nessuna classificazione ML precedente trovata - PRIMO ADDESTRAMENTO")
                    return True
                    
            finally:
                mongo_reader.disconnect()
                
        except Exception as e:
            self.logger.error(f"Errore nel controllare classificazioni ML precedenti: {e}")
            # In caso di errore, assume primo addestramento
            return True
    
    def _load_all_training_data_for_first_training(self) -> List[Dict[str, Any]]:
        """
        Carica TUTTI i dati di training per il primo addestramento ML:
        - Review umane dal training log
        - Classificazioni LLM dal database MongoDB
        
        Returns:
            Lista combinata di decisioni per training
        """
        training_data = []
        
        try:
            # 1. Carica review umane dal log (come sempre)
            human_decisions = self._load_human_decisions_for_training()
            if human_decisions:
                self.logger.info(f"Caricate {len(human_decisions)} decisioni umane dal log")
                training_data.extend(human_decisions)
            
            # 2. Carica classificazioni LLM dal database MongoDB
            llm_classifications = self._load_llm_classifications_from_mongodb()
            if llm_classifications:
                self.logger.info(f"Caricate {len(llm_classifications)} classificazioni LLM dal database")
                training_data.extend(llm_classifications)
            
            # 3. Rimuovi duplicati per session_id (le review umane hanno priorità)
            training_data = self._remove_duplicate_training_data(training_data)
            
            self.logger.info(f"Dataset primo addestramento: {len(training_data)} esempi totali")
            return training_data
            
        except Exception as e:
            self.logger.error(f"Errore nel caricamento dati primo addestramento: {e}")
            return []
    
    def _load_llm_classifications_from_mongodb(self) -> List[Dict[str, Any]]:
        """
        Carica classificazioni LLM dal database MongoDB per il primo addestramento.
        
        Returns:
            Lista di classificazioni LLM formattate come decisioni di training
        """
        llm_data = []
        
        try:
            from mongo_classification_reader import MongoClassificationReader
            mongo_reader = MongoClassificationReader(tenant=self.tenant)
            mongo_reader.connect()
            
            try:
                # Cerca classificazioni fatte SOLO da LLM (non review umane e non ML)
                query = {
                    'client_name': self.tenant.tenant_slug,
                    '$and': [
                        {'classification_method': {'$regex': 'llm', '$options': 'i'}},
                        {'classification_method': {'$not': {'$regex': 'human', '$options': 'i'}}},
                        {'classification_method': {'$not': {'$regex': 'ml', '$options': 'i'}}},
                        {'llm_prediction': {'$exists': True, '$ne': None}},
                        {'needs_review': {'$ne': True}}  # Non prendere quelli ancora in review
                    ]
                }
                
                llm_classifications = list(mongo_reader.collection.find(query))
                
                # Converte in formato compatibile con training log
                for classification in llm_classifications:
                    session_id = classification.get('session_id')
                    llm_prediction = classification.get('llm_prediction')
                    conversation_text = classification.get('conversation_text')
                    
                    if session_id and llm_prediction and conversation_text:
                        training_entry = {
                            'session_id': session_id,
                            'human_decision': llm_prediction,  # Usa LLM prediction come "decisione"
                            'conversation_text': conversation_text,  # 🔧 AGGIUNTO: Include il testo direttamente
                            'source': 'llm_classification',
                            'original_confidence': classification.get('confidence', 0.5)
                        }
                        llm_data.append(training_entry)
                    elif session_id and llm_prediction:
                        # Se manca conversation_text, logga per debug
                        self.logger.warning(f"Classificazione LLM {session_id} senza conversation_text - saltata")
                
                self.logger.info(f"Trovate {len(llm_data)} classificazioni LLM utilizzabili per training")
                
            finally:
                mongo_reader.disconnect()
                
        except Exception as e:
            self.logger.error(f"Errore nel caricamento classificazioni LLM: {e}")
            
        return llm_data
    
    def _remove_duplicate_training_data(self, training_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rimuove duplicati dal training data, dando priorità alle review umane.
        
        Args:
            training_data: Lista di decisioni di training
            
        Returns:
            Lista senza duplicati per session_id
        """
        seen_sessions = set()
        unique_data = []
        
        # Prima passata: prendi tutte le review umane (senza 'source' o source != 'llm_classification')
        for item in training_data:
            session_id = item.get('session_id')
            source = item.get('source', 'human_review')
            
            if session_id and session_id not in seen_sessions and source != 'llm_classification':
                unique_data.append(item)
                seen_sessions.add(session_id)
        
        # Seconda passata: aggiungi classificazioni LLM solo se session_id non già presente
        for item in training_data:
            session_id = item.get('session_id')
            source = item.get('source', 'human_review')
            
            if session_id and session_id not in seen_sessions and source == 'llm_classification':
                unique_data.append(item)
                seen_sessions.add(session_id)
        
        removed_count = len(training_data) - len(unique_data)
        if removed_count > 0:
            self.logger.info(f"Rimossi {removed_count} duplicati dal training data")
        
        return unique_data

    def _load_human_decisions_for_training(self) -> List[Dict[str, Any]]:
        """
        Carica le decisioni umane dal log per il training.
        
        Returns:
            Lista di decisioni umane valide
        """
        training_data = []
        
        try:
            if not os.path.exists(self.training_log_path):
                return training_data
            
            with open(self.training_log_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        
                        decision = json.loads(line)
                        
                        # Valida che la decisione abbia i campi necessari
                        required_fields = ['session_id', 'human_decision']
                        if all(field in decision for field in required_fields):
                            training_data.append(decision)
                        else:
                            self.logger.debug(f"Decisione incompleta alla riga {line_num}: campi mancanti")
                            
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Errore parsing JSON alla riga {line_num}: {e}")
                        continue
            
            self.logger.info(f"Caricate {len(training_data)} decisioni umane TOTALI dal log per riaddestramento")
            return training_data
            
        except Exception as e:
            self.logger.error(f"Errore nel caricamento decisioni umane: {e}")
            return []
    
    def _prepare_training_data(self, decisions: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara i dati di training dalle decisioni umane.
        Se i session_id sono mock, usa campioni di dati reali dal database MongoDB per quella categoria.
        
        Args:
            decisions: Lista di decisioni umane dal log
            
        Returns:
            Tuple (X, y) con features ed etichette
        """
        try:
            # Estrai conversazioni, etichette e mantieni mapping session_id per features
            conversations = []
            labels = []
            session_ids_for_X = []
            
            # USA MONGODB CLASSIFICATION READER per ottenere conversazioni classificate
            from mongo_classification_reader import MongoClassificationReader
            
            # Inizializza il reader MongoDB con OGGETTO TENANT
            mongo_reader = MongoClassificationReader(tenant=self.tenant)
            mongo_reader.connect()
            
            try:
                # Ottieni tutte le sessioni classificate dal MongoDB per questo tenant
                all_sessions = mongo_reader.get_all_sessions()
                
                # Crea un dizionario per accesso rapido alle sessioni per session_id
                sessions_dict = {}
                for session in all_sessions:
                    session_id = session.get('session_id', '')
                    if session_id:
                        sessions_dict[session_id] = session
                
                self.logger.info(f"Caricate {len(sessions_dict)} sessioni classificate da MongoDB")
                
                for decision in decisions:
                    session_id = decision['session_id']
                    human_label = decision['human_decision']
                    
                    # 🔧 OTTIMIZZAZIONE: Usa conversation_text se già presente (es. da LLM classifications)
                    if 'conversation_text' in decision and decision['conversation_text']:
                        conversation_text = decision['conversation_text']
                        conversations.append(conversation_text)
                        labels.append(human_label)
                        session_ids_for_X.append(session_id)
                        self.logger.debug(f"Usato conversation_text diretto per {session_id} (source: {decision.get('source', 'unknown')})")
                    # Altrimenti, cerca la conversazione originale per session_id
                    elif session_id in sessions_dict:
                        # Trovata conversazione originale nel database
                        session_data = sessions_dict[session_id]
                        conversation_text = session_data['conversation_text']
                        conversations.append(conversation_text)
                        labels.append(human_label)
                        session_ids_for_X.append(session_id)
                        self.logger.debug(f"Usata conversazione da database per {session_id}")
                    else:
                        # Session_id non trovato (probabilmente mock), usa campione reale per questa categoria
                        self.logger.debug(f"Session_id {session_id} non trovato, cerco campione reale per categoria {human_label}")
                        
                        # Cerca conversazioni reali già classificate con questa etichetta
                        sample_sessions = mongo_reader.get_sessions_by_label(human_label, limit=1)
                        
                        if sample_sessions and len(sample_sessions) > 0:
                            # Trovato campione reale per questa categoria
                            sample_conversation = sample_sessions[0]['conversation_text']
                            if sample_conversation and len(sample_conversation.strip()) > 20:
                                conversations.append(sample_conversation)
                                labels.append(human_label)
                                session_ids_for_X.append(None)  # nessun session_id valido per feature store
                                self.logger.info(f"Usato campione reale per categoria {human_label}")
                            else:
                                # Testo troppo corto, usa testo sintetico
                                synthetic_text = self._create_synthetic_conversation_for_label(human_label)
                                conversations.append(synthetic_text)
                                labels.append(human_label)
                                session_ids_for_X.append(None)
                                self.logger.info(f"Campione reale troppo corto, creato testo sintetico per {human_label}")
                        else:
                            # Nessun campione reale trovato, crea testo sintetico
                            synthetic_text = self._create_synthetic_conversation_for_label(human_label)
                            conversations.append(synthetic_text)
                            labels.append(human_label)
                            session_ids_for_X.append(None)
                            self.logger.info(f"Creato testo sintetico per categoria {human_label}")
                
            finally:
                mongo_reader.disconnect()
            
            if not conversations:
                self.logger.warning("Nessuna conversazione preparata per il training")
                return None, None
            
            # Genera/recupera embeddings usando feature store quando disponibile
            embedder = self._get_dynamic_embedder()

            # Calcola nome embedder corrente per validare cache
            def _current_embedder_name() -> str:
                embedder_type = type(embedder).__name__
                if hasattr(embedder, 'model_name') and getattr(embedder, 'model_name'):
                    return f"{embedder_type}_{getattr(embedder, 'model_name')}"
                if hasattr(embedder, 'model_path') and getattr(embedder, 'model_path'):
                    import os as _os
                    mp = str(getattr(embedder, 'model_path'))
                    model_name = mp.split('/')[-1] if '/' in mp else mp
                    return f"{embedder_type}_{model_name}"
                return embedder_type

            current_model_name = _current_embedder_name()

            # Prova a leggere dallo store per sessioni reali (non None)
            store = None
            try:
                import sys as _sys, os as _os
                _sys.path.append(_os.path.join(_os.path.dirname(__file__), '..', 'MongoDB'))
                from embedding_store import EmbeddingStore  # type: ignore
                store = EmbeddingStore()
            except Exception as _e:
                self.logger.debug(f"Feature store non disponibile: {_e}")

            X_list: List[Optional[np.ndarray]] = [None] * len(conversations)
            missing_indices: List[int] = []
            if store is not None and hasattr(self.tenant, 'tenant_id'):
                tid = self.tenant.tenant_id
                for idx, sid in enumerate(session_ids_for_X):
                    if not sid:
                        missing_indices.append(idx)
                        continue
                    try:
                        fetched = store.get_embedding(tid, sid, consume=False)
                        if fetched:
                            emb_vec, emb_model = fetched
                            # Valida coerenza modello corrente; se mismatch, marca come missing per rigenerazione
                            if emb_model and emb_model != 'unknown_embedder' and not str(emb_model).startswith(current_model_name):
                                missing_indices.append(idx)
                            else:
                                X_list[idx] = np.array(emb_vec, dtype=float)
                        else:
                            missing_indices.append(idx)
                    except Exception as _fe:
                        self.logger.debug(f"Errore fetch embedding cache per {sid}: {_fe}")
                        missing_indices.append(idx)
            else:
                # Nessuno store: tutti mancanti
                missing_indices = list(range(len(conversations)))

            # Calcola embeddings per i mancanti in un singolo batch
            if missing_indices:
                texts_to_encode = [conversations[i] for i in missing_indices]
                self.logger.info(f"Generazione embeddings per {len(texts_to_encode)} conversazioni (non presenti in cache)...")
                enc = embedder.encode(texts_to_encode)
                for j, idx in enumerate(missing_indices):
                    X_list[idx] = enc[j]

            # Concatena in array numpy
            X = np.vstack([np.array(v, dtype=float) for v in X_list])
            y = np.array(labels)
            
            self.logger.info(f"Preparati {len(X)} esempi di training con {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Errore nella preparazione dati di training: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None

    def _get_ml_features_for_session(self, session_id: str, conversation_text: str, embedder) -> np.ndarray:
        """
        Recupera ML features pre-calcolate per una sessione, privilegiando il feature store
        degli embedding. In assenza, calcola l'embedding al volo.

        Returns:
            np.ndarray shape (1, D)
        """
        try:
            # Prova a leggere dal feature store degli embedding
            tenant_id = getattr(self.tenant, 'tenant_id', None)
            if tenant_id:
                try:
                    import os as _os, sys as _sys
                    _sys.path.append(_os.path.join(_os.path.dirname(__file__), '..', 'MongoDB'))
                    from embedding_store import EmbeddingStore  # type: ignore
                    store = EmbeddingStore()
                    fetched = store.get_embedding(tenant_id, session_id, consume=False)
                    if fetched:
                        emb_vec, _ = fetched
                        return np.array(emb_vec, dtype=float).reshape(1, -1)
                except Exception as _e:
                    self.logger.debug(f"Feature store indisponibile per {session_id}: {_e}")
            # Fallback: calcolo embedding
            if embedder is None:
                embedder = self._get_dynamic_embedder()
            return embedder.encode([conversation_text])
        except Exception as e:
            self.logger.debug(f"Errore ML features per {session_id}: {e}")
            # Ultimo fallback: vettore vuoto coerente
            try:
                if embedder is None:
                    embedder = self._get_dynamic_embedder()
                return embedder.encode([conversation_text])
            except Exception:
                # Ritorna placeholder 1x1 per non bloccare il flusso
                return np.zeros((1, 1), dtype=float)
    
    def _create_synthetic_conversation_for_label(self, label: str) -> str:
        """
        Crea una conversazione sintetica per una data etichetta quando non ci sono campioni reali.
        
        Args:
            label: Etichetta per cui creare il testo sintetico
            
        Returns:
            Testo della conversazione sintetica
        """
        # Template di conversazioni per diverse categorie
        templates = {
            'prenotazione_visite': "Buongiorno, vorrei prenotare una visita specialistica. Quando è possibile?",
            'ritiro_referti': "Salve, devo ritirare i referti degli esami. Come posso fare?",
            'informazioni_costi': "Vorrei sapere i costi per una visita specialistica. Potete aiutarmi?",
            'orari_apertura': "Quali sono gli orari di apertura della struttura?",
            'prenotazione_urgente': "Ho bisogno di una visita urgente. È possibile prenotare?",
            'gestione_appuntamenti': "Vorrei modificare il mio appuntamento. Come posso fare?",
            'pagamenti': "Come posso effettuare il pagamento della visita?",
            'info_procedurali': "Che documenti devo portare per la visita?",
            'info_logistiche': "Dove si trova l'ufficio? C'è un parcheggio?",
            'altro': "Ho una domanda generale sui vostri servizi."
        }
        
        # Restituisci template specifico o generico
        return templates.get(label, f"Richiesta relativa a {label}. Potete aiutarmi?")
    
    def _update_ml_model_with_new_data(self, X_new: np.ndarray, y_new: np.ndarray) -> bool:
        """
        Aggiorna il modello ML esistente con nuovi dati di training.
        
        Args:
            X_new: Nuove features di training
            y_new: Nuove etichette di training
            
        Returns:
            True se l'aggiornamento è riuscito
        """
        try:
            # 1. Carica il modello esistente
            from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier
            
            # Usa embedder dinamico condiviso
            embedder = self._get_dynamic_embedder()
            
            # Trova il modello più recente - supporta sia slug che name per compatibilità
            model_files = []
            models_dir = os.path.join(os.path.dirname(self.training_log_path), 'models')
            
            if os.path.exists(models_dir):
                for f in os.listdir(models_dir):
                    # 🔧 FIX: Cerca sia con tenant_slug che con tenant_name (retrocompatibilità)
                    if ((f.startswith(f'{self.tenant.tenant_slug}_classifier_') or 
                         f.startswith(f'{self.tenant_name}_classifier_')) and 
                        f.endswith('_ml_ensemble.pkl')):
                        model_files.append(os.path.join(models_dir, f))
            
            if not model_files:
                self.logger.warning("Nessun modello esistente trovato, creo un nuovo modello")
                return self._train_new_model_from_scratch(X_new, y_new)
            
            # Prendi il modello più recente
            latest_model = max(model_files, key=os.path.getmtime)
            self.logger.info(f"Caricamento modello esistente: {latest_model}")
            
            # Carica il modello ML
            import joblib
            ml_ensemble = joblib.load(latest_model)
            
            # 2. Carica dati di training originali (se disponibili)
            X_original, y_original = self._load_original_training_data()
            
            # 3. Combina dati originali e nuovi
            if X_original is not None and len(X_original) > 0:
                X_combined = np.vstack([X_original, X_new])
                y_combined = np.hstack([y_original, y_new])
                self.logger.info(f"Combinati {len(X_original)} esempi originali + {len(X_new)} nuovi")
            else:
                X_combined = X_new
                y_combined = y_new
                self.logger.info(f"Solo nuovi esempi: {len(X_new)}")
            
            # 4. Riaddestra il modello con modalità Mistral corretta
            classifier = self._create_ensemble_classifier_with_correct_mistral_mode()
            classifier.ml_ensemble = ml_ensemble
            
            # Riallena con tutti i dati con validazione out-of-sample
            try:
                training_result = classifier.train_ml_ensemble_with_validation(
                    X_combined, y_combined
                )
            except Exception as e:
                self.logger.warning(
                    f"Validazione non disponibile ({e}), fallback training diretto"
                )
                training_result = classifier.train_ml_ensemble(
                    X_combined, y_combined
                )
            
            # 5. Salva il modello aggiornato con tenant_slug (nuovo standard)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_model_path = os.path.join(models_dir, f'{self.tenant.tenant_slug}_classifier_{timestamp}_ml_ensemble.pkl')
            
            # Backup del modello precedente
            backup_path = latest_model + '.backup'
            import shutil
            shutil.copy2(latest_model, backup_path)
            
            # Salva nuovo modello
            joblib.dump(classifier.ml_ensemble, new_model_path)
            
            # Salva config del modello
            config_path = new_model_path.replace('_ml_ensemble.pkl', '_config.json')
            config = {
                'timestamp': timestamp,
                'training_samples': len(X_combined),
                'new_samples': len(X_new),
                'classes': list(classifier.ml_ensemble.classes_),
                'training_accuracy': training_result.get('training_accuracy', 0.0),
                'validation_metrics': training_result.get('validation_metrics', {}),
                'previous_model': os.path.basename(latest_model),
                'retraining_trigger': 'automatic_human_feedback'
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"💾 Nuovo modello salvato: {new_model_path}")
            self.logger.info(f"📊 Accuracy: {training_result.get('training_accuracy', 0.0):.3f}")
            self.logger.info(f"🔄 Esempi totali: {len(X_combined)} (nuovi: {len(X_new)})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiornamento del modello: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _load_original_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Carica i dati di training originali se disponibili.
        Questo è un placeholder - in un sistema completo dovresti salvare 
        i dati di training originali per permettere il riaddestramento incrementale.
        
        Returns:
            Tuple (X_original, y_original) o (None, None) se non disponibili
        """
        # TODO: Implementare il salvataggio/caricamento dei dati di training originali
        # Per ora ritorniamo None, il che significa che usiamo solo i nuovi dati
        self.logger.info("Dati di training originali non disponibili, uso solo nuovi dati")
        return None, None
    
    def _train_new_model_from_scratch(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Crea un nuovo modello da zero quando non esiste un modello precedente.
        
        Args:
            X: Features di training
            y: Etichette di training
            
        Returns:
            True se il training è riuscito
        """
        try:
            self.logger.info("Creazione nuovo modello da zero...")
            
            from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier
            
            classifier = self._create_ensemble_classifier_with_correct_mistral_mode()
            try:
                training_result = classifier.train_ml_ensemble_with_validation(X, y)
            except Exception as e:
                self.logger.warning(
                    f"Validazione non disponibile ({e}), fallback training diretto"
                )
                training_result = classifier.train_ml_ensemble(X, y)
            
            # Salva il nuovo modello con tenant_slug (nuovo standard)
            models_dir = os.path.join(os.path.dirname(self.training_log_path), 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(models_dir, f'{self.tenant.tenant_slug}_classifier_{timestamp}_ml_ensemble.pkl')
            
            import joblib
            joblib.dump(classifier.ml_ensemble, model_path)
            
            # Salva config
            config_path = model_path.replace('_ml_ensemble.pkl', '_config.json')
            config = {
                'timestamp': timestamp,
                'training_samples': len(X),
                'classes': list(classifier.ml_ensemble.classes_),
                'training_accuracy': training_result.get('training_accuracy', 0.0),
                'validation_metrics': training_result.get('validation_metrics', {}),
                'created_from': 'human_feedback_only'
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"✅ Nuovo modello creato: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nella creazione nuovo modello: {e}")
            return False
    
    def _mark_training_data_processed(self):
        """
        Marca i dati di training come processati (opzionale).
        Per ora non facciamo nulla, ma potremmo aggiungere un timestamp
        o spostare i dati processati in un file separato.
        """
        pass
    
    def trigger_manual_retraining(self) -> Dict[str, Any]:
        """
        Trigger manuale per il riaddestramento del modello.
        Può essere chiamato dall'API per forzare un riaddestramento.
        
        Returns:
            Risultato del riaddestramento
        """
        try:
            self.logger.info("🔄 Riaddestramento manuale richiesto...")
            
            decision_count = self._count_human_decisions()
            if decision_count == 0:
                return {
                    'success': False,
                    'message': 'Nessuna decisione umana disponibile per il riaddestramento',
                    'decision_count': 0
                }
            
            success = self._retrain_ml_model()
            
            return {
                'success': success,
                'message': f'Riaddestramento {"completato" if success else "fallito"}',
                'decision_count': decision_count,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Errore nel riaddestramento manuale: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'decision_count': 0
            }

    def _is_first_supervised_training(self) -> bool:
        """
        Determina se questo è il primo training supervisionato per il tenant.
        
        FLUSSO CORRETTO:
        - Primo training supervisionato: Mistral BASE + ML
        - Training successivi: Mistral FINE-TUNED + ML trained
        
        Returns:
            True se è il primo training supervisionato
        """
        try:
            # Controlla se esiste un modello Mistral fine-tuned per questo cliente
            from FineTuning.mistral_finetuning_manager import MistralFineTuningManager
            
            finetuning_manager = MistralFineTuningManager(tenant=self.tenant)
            has_finetuned_mistral = finetuning_manager.has_finetuned_model(self.tenant.tenant_name)
            
            # Controlla se esistono decisioni umane precedenti
            decision_count = self._count_human_decisions()
            
            # È il primo training se:
            # 1. Non esiste un modello fine-tuned
            # 2. O se non ci sono decisioni umane precedenti (reset completo)
            is_first = not has_finetuned_mistral or decision_count == 0
            
            self.logger.info(f"🔍 First training check per {self.tenant_name}:")
            self.logger.info(f"   - Modello fine-tuned esistente: {has_finetuned_mistral}")
            self.logger.info(f"   - Decisioni umane: {decision_count}")
            self.logger.info(f"   - È primo training: {is_first}")
            
            return is_first
            
        except Exception as e:
            self.logger.warning(f"Errore nel controllo primo training: {e}")
            # Default: considera come primo training se in dubbio
            return True

    def _create_ensemble_classifier_with_correct_mistral_mode(self, force_base_model: bool = False):
        """
        Crea AdvancedEnsembleClassifier con la modalità Mistral corretta.
        
        LOGICA:
        - Primo training supervisionato → Mistral BASE
        - Training successivi/classificazione automatica → Mistral FINE-TUNED
        
        Args:
            force_base_model: Se True, forza l'uso del modello base
            
        Returns:
            AdvancedEnsembleClassifier configurato correttamente
        """
        # Estrae tenant_name dall'oggetto Tenant della classe
        tenant_name = self.tenant.tenant_name if self.tenant else "humanitas"
        try:
            from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier
            
            # Determina se usare modello base o fine-tuned
            is_first_training = self._is_first_supervised_training()
            use_base_model = force_base_model or is_first_training
            
            if use_base_model:
                self.logger.info(f"🤖 Creazione ensemble con Mistral BASE per {tenant_name} (primo training)")
                
                # Crea classifier con modello base forzato
                ensemble = AdvancedEnsembleClassifier(client_name=tenant_name)
                
                # Forza switch al modello base se l'LLM classifier esiste
                if ensemble.llm_classifier and hasattr(ensemble.llm_classifier, 'switch_to_base_model'):
                    ensemble.llm_classifier.switch_to_base_model()
                    self.logger.info("✅ LLM classifier impostato su modello BASE")
                
            else:
                self.logger.info(f"🎯 Creazione ensemble con Mistral FINE-TUNED per {tenant_name} (training successivi)")
                
                # Crea classifier che usa automaticamente il modello fine-tuned
                ensemble = AdvancedEnsembleClassifier(client_name=tenant_name)
                
                # Verifica che sia attivo il modello fine-tuned
                if ensemble.llm_classifier and hasattr(ensemble.llm_classifier, 'switch_to_finetuned_model'):
                    switch_success = ensemble.llm_classifier.switch_to_finetuned_model()
                    if switch_success:
                        self.logger.info("✅ LLM classifier impostato su modello FINE-TUNED")
                    else:
                        self.logger.warning("⚠️ Switch a fine-tuned fallito, uso modello disponibile")
                        
            return ensemble
            
        except Exception as e:
            self.logger.error(f"Errore nella creazione ensemble: {e}")
            # Fallback: crea ensemble standard
            from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier
            return AdvancedEnsembleClassifier(client_name=tenant_name)

    def create_mock_review_cases(self, count: int = 3) -> List[str]:
        """
        Crea casi mock per testing della coda di revisione.
        
        Args:
            count: Numero di casi mock da creare
            
        Returns:
            Lista degli ID dei casi creati
        """
        created_case_ids = []
        
        try:
            # Template di conversazioni mock
            mock_conversazioni = [
                "Buongiorno, vorrei prenotare una visita cardiologica per mia madre di 75 anni.",
                "Salve, devo ritirare i referti degli esami del sangue. Come posso fare?",
                "Ciao, ho bisogno di informazioni sui costi della risonanza magnetica.",
                "Buonasera, vorrei sapere gli orari di apertura del laboratorio analisi.",
                "Salve, mio figlio ha bisogno di una visita pediatrica urgente. È possibile?",
                "Vorrei cancellare l'appuntamento di domani e spostarlo alla prossima settimana.",
                "Ho problemi con il pagamento online. Posso pagare di persona?",
                "Serve la ricetta medica per fare le analisi del sangue?",
                "Quanti giorni prima posso prenotare una visita specialistica?",
                "Dove si trova il parcheggio dell'ospedale? È gratuito?"
            ]
            
            # Predizioni mock per ML e LLM
            mock_predictions = [
                ("prenotazione_visite", "prenotazione_appuntamenti"),
                ("ritiro_referti", "gestione_documenti"),
                ("informazioni_costi", "tariffe_servizi"),
                ("orari_apertura", "info_generali"),
                ("prenotazione_urgente", "prenotazione_visite"),
                ("gestione_appuntamenti", "modifica_prenotazioni"),
                ("pagamenti", "modalita_pagamento"),
                ("info_procedurali", "informazioni_procedure"),
                ("prenotazione_visite", "gestione_appuntamenti"),
                ("info_logistiche", "informazioni_generali")
            ]
            
            for i in range(min(count, len(mock_conversazioni))):
                case_id = f"mock_{self.tenant_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}"
                
                # Simula disaccordo o bassa confidenza
                ml_pred, llm_pred = mock_predictions[i % len(mock_predictions)]
                ml_confidence = 0.4 + (i % 3) * 0.1  # 0.4, 0.5, 0.6
                llm_confidence = 0.6 + (i % 3) * 0.1  # 0.6, 0.7, 0.8
                
                # Calcola metriche per determinare la ragione
                disagreement = 1.0 if ml_pred != llm_pred else 0.0
                uncertainty = 1.0 - ml_confidence
                novelty = 0.3 + (i % 4) * 0.2  # 0.3, 0.5, 0.7, 0.9
                
                reason_parts = []
                if disagreement > 0.5:
                    reason_parts.append(f"ensemble disagreement ({disagreement:.2f})")
                if ml_confidence < 0.7:
                    reason_parts.append(f"low ML confidence ({ml_confidence:.2f})")
                if uncertainty > 0.4:
                    reason_parts.append(f"high uncertainty ({uncertainty:.2f})")
                if novelty > 0.6:
                    reason_parts.append(f"novel content ({novelty:.2f})")
                
                reason = "; ".join(reason_parts) if reason_parts else "mock test case"
                
                # Crea il caso di revisione
                review_case = ReviewCase(
                    case_id=case_id,
                    session_id=f"mock_session_{i+1}",
                    conversation_text=mock_conversazioni[i],
                    ml_prediction=ml_pred,
                    ml_confidence=ml_confidence,
                    llm_prediction=llm_pred,
                    llm_confidence=llm_confidence,
                    uncertainty_score=uncertainty,
                    novelty_score=novelty,
                    reason=reason,
                    created_at=datetime.now(),
                    tenant=self.tenant_name
                )
                
                # Per i casi mock, crea direttamente in MongoDB per realismo
                success = self.mongo_reader.mark_session_for_review(
                    session_id=f"mock_session_{i+1}",
                    client_name=self.tenant_name,
                    review_reason=reason
                )
                
                if success:
                    created_case_ids.append(case_id)
                    self.logger.info(f"Creato caso mock {case_id} in MongoDB: {reason}")
                else:
                    self.logger.warning(f"Impossibile creare caso mock {case_id} in MongoDB")
            
            self.logger.info(f"Creati {len(created_case_ids)} casi mock in MongoDB per testing")
            return created_case_ids
            
        except Exception as e:
            self.logger.error(f"Errore nella creazione casi mock: {e}")
            return created_case_ids

    def analyze_classifications_for_review(self, 
                                         batch_size: int = 100,
                                         min_confidence: float = 0.7,
                                         disagreement_threshold: float = 0.3,
                                         force_review: bool = False,
                                         max_review_cases: Optional[int] = None,
                                         use_optimal_selection: Optional[bool] = None,
                                         analyze_all_or_new_only: str = 'ask_user') -> Dict[str, Any]:
        """
        WORKFLOW UNIFICATO REFACTORATO - Basato sulla modalità di analisi richiesta:
        
        MODALITÀ SUPPORTATE:
        1. 'optimized_representatives' - Analisi ottimizzata con clustering e rappresentanti
        2. 'full_analysis' - Analisi completa di tutte le sessioni  
        3. 'new_only' - Solo sessioni non ancora classificate
        4. 'ask_user' - Auto-decide in base alla presenza di modello ML
        
        LOGICA:
        - Se l'utente sceglie esplicitamente una modalità (!= 'ask_user'), usa quella
        - Se 'ask_user', decide automaticamente: nuovo cliente = optimized, esistente = full
        - Ogni modalità può essere usata sia per nuovi che esistenti clienti
        - NON salva auto-classificazioni fino a revisione umana completata
        
        Returns:
            Dict con: auto_classified, needs_review, total_analyzed, warning (se applicabile)
        """
        try:
            from MySql.connettore import MySqlConnettore
            from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier
            from Preprocessing.session_aggregator import SessionAggregator

            # Ottieni embedder dinamico configurato per il tenant
            embedder = self._get_dynamic_embedder()

            # ⚠️ IMPORTANTE: Aggiorna le soglie con i valori dell'utente
            old_confidence_threshold = self.confidence_threshold
            old_disagreement_threshold = self.disagreement_threshold
            
            self.confidence_threshold = min_confidence
            self.disagreement_threshold = disagreement_threshold
            
            self.logger.info(f"🎯 Soglie aggiornate con valori utente:")
            self.logger.info(f"  📊 Confidence: {old_confidence_threshold} -> {self.confidence_threshold}")
            self.logger.info(f"  ⚖️ Disagreement: {old_disagreement_threshold} -> {self.disagreement_threshold}")

            self.logger.info(f"🔍 WORKFLOW UNIFICATO - Analisi classificazioni dal database {self.tenant_name}")
            self.logger.info(f"Parametri: batch_size={batch_size}, min_confidence={min_confidence}")
            self.logger.info(f"Modalità richiesta: {analyze_all_or_new_only}")

            # Controlla se esiste un modello ML per questo cliente - supporta slug e name
            models_dir = os.path.join(os.path.dirname(self.training_log_path), 'models')
            model_files = []
            
            if os.path.exists(models_dir):
                for f in os.listdir(models_dir):
                    # 🔧 FIX: Cerca sia con tenant_slug che con tenant_name (retrocompatibilità)
                    if ((f.startswith(f'{self.tenant.tenant_slug}_classifier_') or 
                         f.startswith(f'{self.tenant_name}_classifier_')) and 
                        f.endswith('_ml_ensemble.pkl')):
                        model_files.append(os.path.join(models_dir, f))
            
            has_ml_model = len(model_files) > 0
            
            # NUOVA LOGICA: Determina modalità di analisi basata su richiesta utente
            analysis_mode = self._determine_analysis_mode(
                analyze_all_or_new_only, 
                has_ml_model, 
                use_optimal_selection,
                max_review_cases
            )
            
            self.logger.info(f"🎯 MODALITÀ SELEZIONATA: {analysis_mode}")
            
            # Dispatching alla funzione appropriata
            result = None
            if analysis_mode == 'optimized_representatives':
                result = self._analyze_with_optimized_representatives(
                    batch_size=batch_size,
                    min_confidence=min_confidence,
                    disagreement_threshold=disagreement_threshold,
                    max_review_cases=max_review_cases,
                    has_ml_model=has_ml_model,
                    model_files=model_files
                )
            elif analysis_mode == 'full_analysis':
                result = self._analyze_with_full_analysis(
                    batch_size=batch_size,
                    min_confidence=min_confidence,
                    disagreement_threshold=disagreement_threshold,
                    max_review_cases=max_review_cases,
                    has_ml_model=has_ml_model,
                    model_files=model_files
                )
            elif analysis_mode == 'new_only':
                result = self._analyze_new_sessions_only(
                    batch_size=batch_size,
                    min_confidence=min_confidence,
                    disagreement_threshold=disagreement_threshold,
                    max_review_cases=max_review_cases,
                    has_ml_model=has_ml_model,
                    model_files=model_files
                )
            else:
                raise ValueError(f"Modalità di analisi non supportata: {analysis_mode}")
            
            # Ripristina soglie originali prima di ritornare
            self.confidence_threshold = old_confidence_threshold
            self.disagreement_threshold = old_disagreement_threshold
            self.logger.info(f"🔄 Soglie ripristinate ai valori originali:")
            self.logger.info(f"  📊 Confidence: {self.confidence_threshold}")
            self.logger.info(f"  ⚖️ Disagreement: {self.disagreement_threshold}")
            
            return result

        except Exception as e:
            # Ripristina soglie originali in caso di errore
            self.confidence_threshold = old_confidence_threshold
            self.disagreement_threshold = old_disagreement_threshold
            self.logger.error(f"Errore nell'analisi classificazioni: {e}")
            self.logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'analyzed_count': 0,
                'auto_classified': 0,
                'needs_review': 0
            }

    def _analyze_with_optimized_representatives(self, 
                                              batch_size: int,
                                              min_confidence: float,
                                              disagreement_threshold: float,
                                              max_review_cases: Optional[int],
                                              has_ml_model: bool,
                                              model_files: List[str]) -> Dict[str, Any]:
        """
        MODALITÀ: Analisi ottimizzata con clustering e rappresentanti.
        
        - Clusterizza tutte le sessioni semanticamente
        - Seleziona rappresentanti diversificati per ogni cluster  
        - Analizza solo i rappresentanti (più efficiente)
        - Propaga etichette a tutto il cluster
        - Utilizzabile sia per nuovi che esistenti clienti
        
        Vantaggi: Veloce, efficiente, buona copertura semantica
        Svantaggi: Non analizza ogni singola sessione individualmente
        """
        # AGGIORNAMENTO 2025-08-29: Usa embedder dinamico invece di hardcode
        try:
            embedder = self._get_dynamic_embedder()
        except Exception as e:
            # 🚫 NESSUN FALLBACK LOCALE - Solo Docker service
            raise RuntimeError(f"Embedder Docker richiesto ma non disponibile: {e}")
            
        from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier
        from Preprocessing.session_aggregator import SessionAggregator
        import numpy as np

        # Estrai TUTTE le sessioni (senza limite per analisi rappresentanti)
        aggregator = SessionAggregator(tenant=self.tenant)
        sessioni_aggregate = aggregator.estrai_sessioni_aggregate(limit=None)  # TUTTE
        if not sessioni_aggregate:
            return {
                'success': False,
                'error': f'Nessuna conversazione disponibile nel database {self.tenant_name}',
                'analyzed_count': 0,
                'auto_classified': 0,
                'needs_review': 0
            }

        # Filtra conversazioni valide
        sessioni_filtrate = {}
        for session_id, dati in sessioni_aggregate.items():
            if dati['num_messaggi_user'] > 1:
                sessioni_filtrate[session_id] = dati

        self.logger.info(f"📊 ANALISI RAPPRESENTANTI: {len(sessioni_filtrate)} sessioni totali per clustering")
        # 1. CLUSTERING con modalità Mistral corretta
        embedder = self._get_dynamic_embedder()
        classifier = self._create_ensemble_classifier_with_correct_mistral_mode()
        
        # Se abbiamo un modello ML esistente, caricalo nell'ensemble
        if has_ml_model and model_files:
            latest_model = max(model_files, key=os.path.getmtime)
            self.logger.info(f"Caricamento modello ML esistente: {latest_model}")
            classifier.ml_ensemble = joblib.load(latest_model)
        
        # CLUSTERING tramite pipeline intelligente  
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        pipeline = EndToEndPipeline(
            tenant_slug=self.tenant_name, 
            confidence_threshold=0.7, 
            auto_mode=True, 
            shared_embedder=embedder
            # auto_retrain ora gestito da config.yaml
        )
        embeddings, cluster_labels, representatives, suggested_labels = pipeline.esegui_clustering(sessioni_filtrate)
        
        # Ricostruisci session_ids e conversazioni in base all'ordine degli embeddings
        session_ids = list(sessioni_filtrate.keys())
        conversazioni = [sessioni_filtrate[sid]['testo_completo'] for sid in session_ids]

        # 2. ANALISI SOLO DEI RAPPRESENTANTI (strategia ottimizzata)
        rappresentanti_analizzati = set()
        if representatives:
            self.logger.debug(f"🔍 Struttura representatives: {type(representatives)}")
            for cluster_id, cluster_reps in representatives.items():
                if cluster_reps:
                    self.logger.debug(f"🔍 Cluster {cluster_id}: {len(cluster_reps)} reps, tipo: {type(cluster_reps[0]) if cluster_reps else 'vuoto'}")
                    # Estrai gli ID dei rappresentanti (se sono dict con 'id' o session_id)
                    for rep in cluster_reps:
                        if isinstance(rep, dict):
                            # Prova diversi nomi di campo per l'ID
                            rep_id = rep.get('session_id') or rep.get('id') or rep.get('ID')
                            if rep_id:
                                rappresentanti_analizzati.add(rep_id)
                            else:
                                self.logger.warning(f"⚠️ Rappresentante senza ID riconoscibile: {list(rep.keys())[:3]}")
                        else:
                            # Se è già un ID semplice
                            rappresentanti_analizzati.add(rep)
        
        self.logger.info(f"🎯 Analizzando solo {len(rappresentanti_analizzati)} rappresentanti di cluster")
        
        # Mappa rappresentanti -> etichette
        representative_labels = {}
        representative_confidences = {}
        
        for idx, session_id in enumerate(session_ids):
            if session_id in rappresentanti_analizzati:
                conversation_text = conversazioni[idx]
                try:
                    if has_ml_model:
                        # Usa ensemble ML+LLM se disponibile
                        # Prepara ML features (dal feature store o calcolate una volta sola)
                        ml_pre = self._get_ml_features_for_session(session_id, conversation_text, embedder)
                        result = classifier.predict_with_ensemble(
                            conversation_text,
                            return_details=True,
                            embedder=embedder,
                            ml_features_precalculated=ml_pre,
                            session_id=session_id
                        )
                        ml_result = result.get('ml_prediction', {})
                        llm_result = result.get('llm_prediction', {})
                        
                        # Usa predizione con confidenza maggiore
                        if ml_result.get('confidence', 0) >= llm_result.get('confidence', 0):
                            representative_labels[session_id] = ml_result.get('predicted_label', '')
                            representative_confidences[session_id] = ml_result.get('confidence', 0.0)
                        else:
                            representative_labels[session_id] = llm_result.get('predicted_label', '')
                            representative_confidences[session_id] = llm_result.get('confidence', 0.0)
                    else:
                        # Solo LLM per nuovi clienti
                        llm_result = classifier.predict_with_llm_only(conversation_text, return_details=True)
                        representative_labels[session_id] = llm_result.get('predicted_label', '')
                        representative_confidences[session_id] = llm_result.get('confidence', 0.0)
                        
                except Exception as e:
                    self.logger.error(f"Errore analisi rappresentante {session_id}: {e}")
                    representative_labels[session_id] = ''
                    representative_confidences[session_id] = 0.0

        # 3. PROPAGAZIONE ETICHETTE AI CLUSTER
        # Propaga le etichette dei rappresentanti a tutte le sessioni del cluster
        session_labels = {}
        session_confidences = {}
        
        for idx, session_id in enumerate(session_ids):
            cluster_id = cluster_labels[idx] if idx < len(cluster_labels) else -1
            
            if cluster_id >= 0 and cluster_id in representatives:
                # Trova il rappresentante del cluster per questa sessione
                cluster_reps = representatives[cluster_id]
                if cluster_reps:
                    # Usa il primo rappresentante come etichetta del cluster
                    rep_item = cluster_reps[0]
                    
                    # Estrai l'ID dal rappresentante (gestisce sia dict che string)
                    if isinstance(rep_item, dict):
                        rep_id = rep_item.get('session_id') or rep_item.get('id') or rep_item.get('ID')
                    else:
                        rep_id = rep_item
                    
                    if rep_id and rep_id in representative_labels:
                        session_labels[session_id] = representative_labels[rep_id]
                        session_confidences[session_id] = representative_confidences[rep_id]
                    else:
                        session_labels[session_id] = ''
                        session_confidences[session_id] = 0.0
                else:
                    session_labels[session_id] = ''
                    session_confidences[session_id] = 0.0
            else:
                # Outlier: analizza individualmente se è tra i rappresentanti, altrimenti bassa confidenza
                if session_id in representative_labels:
                    session_labels[session_id] = representative_labels[session_id]
                    session_confidences[session_id] = representative_confidences[session_id]
                else:
                    session_labels[session_id] = 'altro'
                    session_confidences[session_id] = 0.3

        # 4. TRAINING ML SE NUOVO CLIENTE
        if not has_ml_model:
            try:
                X = embeddings
                y = np.array([session_labels[sid] for sid in session_ids])
                if len(set(y)) > 1 and all(label != '' for label in y):
                    classifier.train_ml_ensemble(X, y)
                    self.logger.info(f"✅ Training ML eseguito su {len(y)} sessioni con etichette propagate")
                else:
                    self.logger.warning("⚠️ Training ML non eseguito: etichette insufficienti o vuote")
            except Exception as e:
                self.logger.error(f"Errore training ML: {e}")

        # 5. AUTO-CLASSIFICAZIONE/REVIEW QUEUE
        auto_classified_count = 0
        needs_review_count = 0
        analyzed_count = 0
        auto_classified_sessions = []
        
        for idx, session_id in enumerate(session_ids):
            conversation_text = conversazioni[idx]
            label = session_labels[session_id]
            confidence = session_confidences[session_id]
            analyzed_count += 1
            
            if confidence >= min_confidence and label and label != '':
                auto_classified_count += 1
                auto_classified_sessions.append(convert_numpy_types({
                    'session_id': session_id,
                    'conversation_text': conversation_text,
                    'predicted_label': label,
                    'confidence': confidence,
                    'method': f"{'ensemble' if has_ml_model else 'llm'}_representatives_propagated",
                    'tenant': self.tenant_name
                }))
                self.logger.debug(f"✅ Auto-classificata (propagata): {session_id} -> {label} (conf: {confidence:.3f})")
            else:
                needs_review_count += 1
                case_id = f"{self.tenant_name}_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                review_case = ReviewCase(
                    case_id=case_id,
                    session_id=session_id,
                    conversation_text=conversation_text,
                    ml_prediction=label if has_ml_model else "",
                    ml_confidence=confidence if has_ml_model else 0.0,
                    llm_prediction=label if not has_ml_model else "",
                    llm_confidence=confidence if not has_ml_model else 0.0,
                    uncertainty_score=1.0 - confidence,
                    novelty_score=0.5,
                    reason=f"rappresentanti_bassa_confidenza (conf: {confidence:.3f})",
                    created_at=datetime.now(),
                    tenant=self.tenant_name
                )
                
                # Aggiunge a MongoDB invece che alla memoria locale
                success = self.mongo_reader.mark_session_for_review(
                    session_id=session_id,
                    client_name=self.tenant_name,
                    review_reason=f"rappresentanti_bassa_confidenza (conf: {confidence:.3f})"
                )
                
                if success:
                    self.logger.debug(f"📝 Aggiunta a review MongoDB: {session_id} -> confidenza {confidence:.3f}")
                else:
                    self.logger.warning(f"⚠️ Impossibile aggiungere {session_id} a review MongoDB")
                    needs_review_count -= 1  # Decrementa se non è stato aggiunto

        # Recupera statistiche review da MongoDB
        review_stats = self.mongo_reader.get_review_statistics(self.tenant_name)
        current_queue_size = review_stats.get('pending', 0)

        return {
            'success': True,
            'analysis_mode': 'optimized_representatives',
            'client_type': 'new' if not has_ml_model else 'existing',
            'strategy_used': 'clustering_representatives_propagation',
            'analyzed_count': analyzed_count,
            'representatives_analyzed': len(rappresentanti_analizzati),
            'auto_classified': auto_classified_count,
            'needs_review': needs_review_count,
            'current_queue_size': current_queue_size,
            'message': f'Analisi rappresentanti - {len(rappresentanti_analizzati)} rappresentanti → {auto_classified_count} auto-classificate, {needs_review_count} in review',
            'auto_classified_sessions': len(auto_classified_sessions),
            'pending_save': True,
            'has_ml_model': has_ml_model
        }

    def _analyze_with_full_analysis(self,
                                  batch_size: int,
                                  min_confidence: float,
                                  disagreement_threshold: float,
                                  max_review_cases: Optional[int],
                                  has_ml_model: bool,
                                  model_files: List[str]) -> Dict[str, Any]:
        """
        MODALITÀ: Analisi completa di tutte le sessioni.
        
        - Analizza OGNI singola sessione individualmente
        - Usa ensemble ML+LLM se disponibile, altrimenti solo LLM
        - Nessun clustering o ottimizzazione
        - Massima precisione ma più lenta
        - Utilizzabile sia per nuovi che esistenti clienti
        
        Vantaggi: Analisi completa, massima precisione
        Svantaggi: Più lenta, richiede più risorse
        """
        # AGGIORNAMENTO 2025-08-29: Usa embedder dinamico
        try:
            embedder = self._get_dynamic_embedder()
        except Exception as e:
            # 🚫 NESSUN FALLBACK LOCALE - Solo Docker service
            raise RuntimeError(f"Embedder Docker richiesto ma non disponibile: {e}")
            
        from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier
        from Preprocessing.session_aggregator import SessionAggregator
        
        aggregator = SessionAggregator(tenant=self.tenant)
        
        # Per analisi completa, processa tutte le sessioni senza limite
        if max_review_cases is None:
            # "ANALIZZA TUTTO" - processa tutte le sessioni senza limite
            sessioni_aggregate = aggregator.estrai_sessioni_aggregate(limit=None)
            self.logger.info("🚀 ANALISI COMPLETA: processando tutte le sessioni disponibili")
        else:
            # Analisi con limite batch_size
            sessioni_aggregate = aggregator.estrai_sessioni_aggregate(limit=batch_size)
            self.logger.info(f"📊 ANALISI LIMITATA: processando {batch_size} sessioni")
        
        if not sessioni_aggregate:
            return {
                'success': False,
                'error': f'Nessuna conversazione disponibile nel database {self.tenant_name}',
                'analyzed_count': 0,
                'auto_classified': 0,
                'needs_review': 0
            }

        # Filtra conversazioni valide
        sessioni_filtrate = {}
        for session_id, dati in sessioni_aggregate.items():
            if dati['num_messaggi_user'] > 1:
                sessioni_filtrate[session_id] = dati

        self.logger.info(f"📊 ANALISI COMPLETA: {len(sessioni_filtrate)} sessioni da analizzare individualmente")

        # Inizializza ensemble con modalità Mistral corretta
        embedder = self._get_dynamic_embedder()
        ensemble = self._create_ensemble_classifier_with_correct_mistral_mode()
        
        # Carica modello ML se disponibile
        if has_ml_model and model_files:
            latest_model = max(model_files, key=os.path.getmtime)
            self.logger.info(f"Caricamento modello ML: {latest_model}")
            ensemble.ml_ensemble = joblib.load(latest_model)

        auto_classified_count = 0
        needs_review_count = 0
        analyzed_count = 0
        auto_classified_sessions = []

        # ANALISI INDIVIDUALE DI OGNI SESSIONE
        for session_id, dati_sessione in sessioni_filtrate.items():
            conversation_text = dati_sessione['testo_completo']
            
            if not conversation_text:
                continue

            try:
                if has_ml_model:
                    # Usa ensemble ML+LLM per clienti esistenti
                    ml_pre = self._get_ml_features_for_session(session_id, conversation_text, embedder)
                    result = ensemble.predict_with_ensemble(
                        conversation_text, 
                        return_details=True,
                        embedder=embedder,
                        ml_features_precalculated=ml_pre,
                        session_id=session_id
                    )
                    
                    ml_result = result.get('ml_prediction', {})
                    llm_result = result.get('llm_prediction', {})
                    
                    if not ml_result or not llm_result:
                        continue

                    analyzed_count += 1

                    # Valuta con quality gate se necessita review
                    decision = self.evaluate_classification(
                        session_id=session_id,
                        conversation_text=conversation_text,
                        ml_result=ml_result,
                        llm_result=llm_result,
                        tenant=self.tenant_name
                    )

                    if decision.needs_review:
                        # INVIA A REVIEW
                        needs_review_count += 1
                        # Il caso è già stato aggiunto dal evaluate_classification
                        
                    else:
                        # AUTO-CLASSIFICAZIONE (alta confidenza, accordo ensemble)
                        auto_classified_count += 1
                        
                        # Usa la predizione con confidenza maggiore
                        if ml_result.get('confidence', 0) >= llm_result.get('confidence', 0):
                            final_label = ml_result.get('predicted_label', '')
                            final_confidence = ml_result.get('confidence', 0.0)
                            method = 'ml_ensemble_full_analysis'
                        else:
                            final_label = llm_result.get('predicted_label', '')
                            final_confidence = llm_result.get('confidence', 0.0)
                            method = 'llm_ensemble_full_analysis'
                        
                        auto_classified_sessions.append(convert_numpy_types({
                            'session_id': session_id,
                            'conversation_text': conversation_text,
                            'predicted_label': final_label,
                            'confidence': final_confidence,
                            'method': method,
                            'tenant': self.tenant_name,
                            'ml_prediction': ml_result.get('predicted_label', ''),
                            'llm_prediction': llm_result.get('predicted_label', '')
                        }))
                else:
                    # Solo LLM per nuovi clienti
                    llm_result = ensemble.predict_with_llm_only(conversation_text, return_details=True)
                    
                    analyzed_count += 1
                    
                    llm_label = llm_result.get('predicted_label', '')
                    llm_confidence = llm_result.get('confidence', 0.0)
                    
                    if llm_confidence >= min_confidence and llm_label:
                        # AUTO-CLASSIFICAZIONE
                        auto_classified_count += 1
                        auto_classified_sessions.append(convert_numpy_types({
                            'session_id': session_id,
                            'conversation_text': conversation_text,
                            'predicted_label': llm_label,
                            'confidence': llm_confidence,
                            'method': 'llm_full_analysis',
                            'tenant': self.tenant_name
                        }))
                    else:
                        # INVIA A REVIEW
                        needs_review_count += 1
                        case_id = f"{self.tenant_name}_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        # Aggiunge a MongoDB invece che alla memoria locale
                        success = self.mongo_reader.mark_session_for_review(
                            session_id=session_id,
                            client_name=self.tenant_name,
                            review_reason=f"full_analysis_bassa_confidenza (LLM conf: {llm_confidence:.3f})"
                        )
                        
                        if not success:
                            self.logger.warning(f"⚠️ Impossibile aggiungere {session_id} a review MongoDB")
                            needs_review_count -= 1  # Decrementa se non è stato aggiunto
                    
            except Exception as e:
                self.logger.error(f"Errore classificazione sessione {session_id}: {e}")
                continue

        # TRAINING ML RIMOSSO - ora gestito tramite human supervision in resolve_review_case
        # Salva auto-classificazioni in cache temporanea
        self._cache_auto_classifications(auto_classified_sessions)

        # Recupera statistiche review da MongoDB
        review_stats = self.mongo_reader.get_review_statistics(self.tenant_name)
        current_queue_size = review_stats.get('pending', 0)

        return {
            'success': True,
            'analysis_mode': 'full_analysis',
            'client_type': 'existing' if has_ml_model else 'new',
            'strategy_used': f"{'ensemble_ml_llm' if has_ml_model else 'llm_only'}_full_analysis",
            'analyzed_count': analyzed_count,
            'auto_classified': auto_classified_count,
            'needs_review': needs_review_count,
            'current_queue_size': current_queue_size,
            'message': f'Analisi completa - {analyzed_count} sessioni analizzate individualmente: {auto_classified_count} auto-classificate, {needs_review_count} in review',
            'auto_classified_sessions': len(auto_classified_sessions),
            'pending_save': True,
            'has_ml_model': has_ml_model,
            'is_complete_analysis': max_review_cases is None
        }

    def _analyze_new_sessions_only(self,
                                 batch_size: int,
                                 min_confidence: float,
                                 disagreement_threshold: float,
                                 max_review_cases: Optional[int],
                                 has_ml_model: bool,
                                 model_files: List[str]) -> Dict[str, Any]:
        """
        MODALITÀ: Analisi solo delle sessioni non ancora classificate.
        
        - Filtra solo sessioni senza classificazione esistente nel DB
        - Usa stesso approccio di full_analysis ma su subset filtrato
        - Utilizzabile sia per nuovi che esistenti clienti
        - Ottimale per aggiornamenti incrementali
        
        Vantaggi: Efficiente per aggiornamenti, evita riprocessing
        Svantaggi: Richiede logica di tracking delle sessioni processate
        """
        # AGGIORNAMENTO 2025-08-29: Usa embedder dinamico
        try:
            embedder = self._get_dynamic_embedder()
        except Exception as e:
            # 🚫 NESSUN FALLBACK LOCALE - Solo Docker service
            raise RuntimeError(f"Embedder Docker richiesto ma non disponibile: {e}")
            
        from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier
        from Preprocessing.session_aggregator import SessionAggregator
        from TagDatabase.tag_database_connector import TagDatabaseConnector
        
        # TODO: Implementare logica reale per identificare sessioni non classificate
        # Per ora, usa un filtro basato su presenza nel database delle classificazioni
        
        aggregator = SessionAggregator(tenant=self.tenant)
        tag_db = TagDatabaseConnector(tenant=self.tenant)
        
        # Estrai tutte le sessioni disponibili
        sessioni_aggregate = aggregator.estrai_sessioni_aggregate(limit=None)
        
        if not sessioni_aggregate:
            return {
                'success': False,
                'error': f'Nessuna conversazione disponibile nel database {self.tenant_name}',
                'analyzed_count': 0,
                'auto_classified': 0,
                'needs_review': 0
            }
        
        # Filtra solo sessioni NON già classificate
        tag_db.connetti()
        try:
            session_ids_all = list(sessioni_aggregate.keys())
            # Query per trovare sessioni già classificate
            placeholders = ','.join(['%s'] * len(session_ids_all))
            query = f"""
            SELECT DISTINCT session_id 
            FROM session_classifications 
            WHERE tenant_name = %s AND session_id IN ({placeholders})
            """
            cursor = tag_db.connection.cursor()
            cursor.execute(query, [self.tenant_name] + session_ids_all)
            classified_session_ids = set(row[0] for row in cursor.fetchall())
            cursor.close()
            
            # Filtra sessioni NON classificate
            new_session_ids = set(session_ids_all) - classified_session_ids
            sessioni_nuove = {sid: sessioni_aggregate[sid] for sid in new_session_ids if sid in sessioni_aggregate}
            
        except Exception as e:
            self.logger.error(f"Errore nel filtrare sessioni nuove: {e}")
            # Fallback: usa tutte le sessioni
            sessioni_nuove = sessioni_aggregate
        finally:
            tag_db.disconnetti()
        
        # Filtra conversazioni valide
        sessioni_filtrate = {}
        for session_id, dati in sessioni_nuove.items():
            if dati['num_messaggi_user'] > 1:
                sessioni_filtrate[session_id] = dati

        self.logger.info(f"📊 NEW_ONLY: {len(sessioni_filtrate)} sessioni non classificate da analizzare")
        
        # Warning se poche sessioni nuove
        warning_message = None
        if len(sessioni_filtrate) < 100:
            warning_message = f"Solo {len(sessioni_filtrate)} nuove sessioni trovate. Considera di usare 'analizza tutto' per una copertura completa."

        # Se nessuna sessione nuova, ritorna subito
        if len(sessioni_filtrate) == 0:
            return {
                'success': True,
                'analysis_mode': 'new_only',
                'client_type': 'existing' if has_ml_model else 'new',
                'strategy_used': 'new_sessions_only',
                'analyzed_count': 0,
                'auto_classified': 0,
                'needs_review': 0,
                'current_queue_size': len(self.pending_reviews),
                'message': 'Nessuna nuova sessione da analizzare',
                'warning': 'Tutte le sessioni sono già state classificate',
                'auto_classified_sessions': 0,
                'pending_save': True,
                'has_ml_model': has_ml_model
            }

        # Inizializza ensemble con modalità Mistral corretta
        embedder = self._get_dynamic_embedder()
        ensemble = self._create_ensemble_classifier_with_correct_mistral_mode()
        
        # Carica modello ML se disponibile
        if has_ml_model and model_files:
            latest_model = max(model_files, key=os.path.getmtime)
            self.logger.info(f"Caricamento modello ML: {latest_model}")
            ensemble.ml_ensemble = joblib.load(latest_model)

        auto_classified_count = 0
        needs_review_count = 0
        analyzed_count = 0
        auto_classified_sessions = []

        # ANALISI INDIVIDUALE DELLE NUOVE SESSIONI (stesso algoritmo di full_analysis)
        for session_id, dati_sessione in sessioni_filtrate.items():
            conversation_text = dati_sessione['testo_completo']
            
            if not conversation_text:
                continue

            try:
                if has_ml_model:
                    # Usa ensemble ML+LLM
                    ml_pre = self._get_ml_features_for_session(session_id, conversation_text, embedder)
                    result = ensemble.predict_with_ensemble(
                        conversation_text, 
                        return_details=True,
                        embedder=embedder,
                        ml_features_precalculated=ml_pre,
                        session_id=session_id
                    )
                    
                    ml_result = result.get('ml_prediction', {})
                    llm_result = result.get('llm_prediction', {})
                    
                    if not ml_result or not llm_result:
                        continue

                    analyzed_count += 1

                    # Valuta con quality gate se necessita review
                    decision = self.evaluate_classification(
                        session_id=session_id,
                        conversation_text=conversation_text,
                        ml_result=ml_result,
                        llm_result=llm_result,
                        tenant=self.tenant_name
                    )

                    if decision.needs_review:
                        needs_review_count += 1
                    else:
                        auto_classified_count += 1
                        
                        # Usa predizione con confidenza maggiore
                        if ml_result.get('confidence', 0) >= llm_result.get('confidence', 0):
                            final_label = ml_result.get('predicted_label', '')
                            final_confidence = ml_result.get('confidence', 0.0)
                            method = 'ml_ensemble_new_only'
                        else:
                            final_label = llm_result.get('predicted_label', '')
                            final_confidence = llm_result.get('confidence', 0.0)
                            method = 'llm_ensemble_new_only'
                        
                        auto_classified_sessions.append(convert_numpy_types({
                            'session_id': session_id,
                                                       'conversation_text': conversation_text,
                            'predicted_label': final_label,
                            'confidence': final_confidence,
                            'method': method,
                            'tenant': self.tenant_name,
                            'ml_prediction': ml_result.get('predicted_label', ''),
                            'llm_prediction': llm_result.get('predicted_label', '')
                        }))
                else:
                    # Solo LLM per nuovi clienti
                    llm_result = ensemble.predict_with_llm_only(conversation_text, return_details=True)
                    
                    analyzed_count += 1
                    
                    llm_label = llm_result.get('predicted_label', '')
                    llm_confidence = llm_result.get('confidence', 0.0)
                    
                    if llm_confidence >= min_confidence and llm_label:
                        auto_classified_count += 1
                        auto_classified_sessions.append(convert_numpy_types({
                            'session_id': session_id,
                            'conversation_text': conversation_text,
                            'predicted_label': llm_label,
                            'confidence': llm_confidence,
                            'method': 'llm_new_only',
                            'tenant': self.tenant_name
                        }))
                    else:
                        needs_review_count += 1
                        case_id = f"{self.tenant_name}_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        # Aggiunge a MongoDB invece che alla memoria locale
                        success = self.mongo_reader.mark_session_for_review(
                            session_id=session_id,
                            client_name=self.tenant_name,
                            review_reason=f"new_only_bassa_confidenza (LLM conf: {llm_confidence:.3f})"
                        )
                        
                        if not success:
                            self.logger.warning(f"⚠️ Impossibile aggiungere {session_id} a review MongoDB")
                            needs_review_count -= 1  # Decrementa se non è stato aggiunto
                    
            except Exception as e:
                self.logger.error(f"Errore classificazione sessione {session_id}: {e}")
                continue

        # Salva auto-classificazioni in cache temporanea
        self._cache_auto_classifications(auto_classified_sessions)

        # Recupera statistiche review da MongoDB
        review_stats = self.mongo_reader.get_review_statistics(self.tenant_name)
        current_queue_size = review_stats.get('pending', 0)

        result = {
            'success': True,
            'analysis_mode': 'new_only',
            'client_type': 'existing' if has_ml_model else 'new',
            'strategy_used': f"{'ensemble_ml_llm' if has_ml_model else 'llm_only'}_new_sessions_only",
            'analyzed_count': analyzed_count,
            'auto_classified': auto_classified_count,
            'needs_review': needs_review_count,
            'current_queue_size': current_queue_size,
            'message': f'Analisi nuove sessioni - {analyzed_count} sessioni non classificate analizzate: {auto_classified_count} auto-classificate, {needs_review_count} in review',
            'auto_classified_sessions': len(auto_classified_sessions),
            'pending_save': True,
            'has_ml_model': has_ml_model
        }
        
        if warning_message:
            result['warning'] = warning_message
            
        return result
    
    def _determine_analysis_mode(self, 
                                analyze_all_or_new_only: str, 
                                has_ml_model: bool, 
                                use_optimal_selection: Optional[bool],
                                max_review_cases: Optional[int]) -> str:
        """
        Determina la modalità di analisi basata sui parametri dell'utente.
        
        Args:
            analyze_all_or_new_only: Scelta utente ('all', 'new_only', 'ask_user')
            has_ml_model: Se esiste un modello ML per questo cliente
            use_optimal_selection: Se forzare ottimizzazione cluster
            max_review_cases: Limite massimo casi (None = illimitato)
            
        Returns:
            str: Modalità di analisi ('optimized_representatives', 'full_analysis', 'new_only')
        """
        
        # Priorità 1: Se utente sceglie esplicitamente "new_only"
        if analyze_all_or_new_only == 'new_only':
            return 'new_only'
        
        # Priorità 2: Se use_optimal_selection è esplicitamente impostato
        if use_optimal_selection is True:
            return 'optimized_representatives'
        elif use_optimal_selection is False:
            return 'full_analysis'
        
        # Priorità 3: Se max_review_cases è None -> analizza tutto
        if max_review_cases is None:
            return 'full_analysis'
        
        # Priorità 4: Se analyze_all_or_new_only è "all" -> analizza tutto
        if analyze_all_or_new_only == 'all':
            return 'full_analysis'
        
        # Priorità 5: Default "ask_user" - decide in base allo stato cliente
        if analyze_all_or_new_only == 'ask_user':
            if has_ml_model:
                # Cliente esistente -> analisi completa
                return 'full_analysis' 
            else:
                # Nuovo cliente -> analisi ottimizzata
                return 'optimized_representatives'
        
        # Fallback (non dovrebbe mai arrivarci)
        self.logger.warning(f"Modalità non riconosciuta: {analyze_all_or_new_only}, uso optimized_representatives")
        return 'optimized_representatives'
    
    def _cache_auto_classifications(self, auto_classified_sessions: List[Dict[str, Any]]):
        """
        Memorizza le auto-classificazioni in cache temporanea.
        Non salva nel database fino a revisione umana completata.
        
        Args:
            auto_classified_sessions: Lista delle sessioni auto-classificate
        """
        tenant_key = self.tenant_name
        
        if not hasattr(self, '_auto_classification_cache'):
            self._auto_classification_cache = {}
        
        if tenant_key not in self._auto_classification_cache:
            self._auto_classification_cache[tenant_key] = []
        
        # Aggiungi le nuove auto-classificazioni
        self._auto_classification_cache[tenant_key].extend(auto_classified_sessions)
        
        self.logger.info(f"💾 Cached {len(auto_classified_sessions)} auto-classificazioni per {tenant_key}")
        self.logger.info(f"📊 Totale cached per {tenant_key}: {len(self._auto_classification_cache[tenant_key])}")

    def get_pending_auto_classifications(self, tenant: str = None) -> List[Dict[str, Any]]:
        """
        Ottieni le auto-classificazioni in cache per un tenant.
        
        Args:
            tenant: Nome del tenant (se None, usa self.tenant_name)
            
        Returns:
            Lista delle auto-classificazioni in cache
        """
        tenant_key = tenant or self.tenant_name
        
        if not hasattr(self, '_auto_classification_cache'):
            self._auto_classification_cache = {}
        
        return self._auto_classification_cache.get(tenant_key, [])

    def save_auto_classifications_to_db(self, tenant: str = None) -> Dict[str, Any]:
        """
        Salva le auto-classificazioni dalla cache al database.
        
        Args:
            tenant: Nome del tenant (se None, usa self.tenant_name)
            
        Returns:
            Risultato dell'operazione
        """
        tenant_key = tenant or self.tenant_name
        pending_auto = self.get_pending_auto_classifications(tenant_key)
        
        if not pending_auto:
            return {
                'success': True,
                'message': 'Nessuna auto-classificazione da salvare',
                'saved_count': 0
            }
        
        # Importa TagDatabaseConnector
        from TagDatabase.tag_database_connector import TagDatabaseConnector
        
        saved_count = 0
        errors = []
        
        tag_db = TagDatabaseConnector(tenant=self.tenant)
        tag_db.connetti()
        
        try:
            for classification in pending_auto:
                try:
                    success = tag_db.classifica_sessione(
                        session_id=classification['session_id'],
                        tag_name=classification['predicted_label'],
                        tenant_slug=classification['tenant'],
                        confidence_score=classification['confidence'],
                        method=classification['method'],
                        classified_by='auto_system',
                        notes=f"Auto-classified with {classification['method']}"
                    )
                    
                    if success:
                        saved_count += 1
                    else:
                        errors.append(f"Errore salvataggio {classification['session_id']}")
                        
                except Exception as e:
                    errors.append(f"Errore {classification['session_id']}: {str(e)}")
            
            # Pulisci cache dopo salvataggio
            if saved_count > 0:
                self.clear_auto_classifications_cache(tenant_key)
            
        finally:
            tag_db.disconnetti()
        
        self.logger.info(f"💾 Salvate {saved_count}/{len(pending_auto)} auto-classificazioni per {tenant_key}")
        
        return {
            'success': len(errors) == 0,
            'saved_count': saved_count,
            'total_pending': len(pending_auto),
            'errors': errors,
            'message': f'Salvate {saved_count} auto-classificazioni su {len(pending_auto)}'
        }

    def clear_auto_classifications_cache(self, tenant: str = None):
        """
        Pulisce la cache delle auto-classificazioni per un tenant.
        
        Args:
            tenant: Nome del tenant (se None, usa self.tenant_name)
        """
        tenant_key = tenant or self.tenant_name
        
        if not hasattr(self, '_auto_classification_cache'):
            self._auto_classification_cache = {}
        
        if tenant_key in self._auto_classification_cache:
            del self._auto_classification_cache[tenant_key]
            self.logger.info(f"🗑️ Cache auto-classificazioni pulita per {tenant_key}")

    def get_workflow_status(self, tenant: str = None) -> Dict[str, Any]:
        """
        Ottieni lo stato completo del workflow per un tenant.
        
        Args:
            tenant: Nome del tenant (se None, usa self.tenant_name)
            
        Returns:
            Stato completo del workflow
        """
        tenant_key = tenant or self.tenant_name
        
        # Statistiche review queue
        review_stats = self.get_review_stats()
        
        # Auto-classificazioni in cache
        pending_auto = self.get_pending_auto_classifications(tenant_key)
        
        # Determina tipo di cliente
        models_dir = os.path.join(os.path.dirname(self.training_log_path), 'models')
        has_ml_model = False
        
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) 
                          if f.startswith(f'{tenant_key}_classifier_') and f.endswith('_ml_ensemble.pkl')]
            has_ml_model = len(model_files) > 0
        
        client_type = 'existing' if has_ml_model else 'new'
        
        return {
            'tenant': tenant_key,
            'client_type': client_type,
            'has_ml_model': has_ml_model,
            'review_queue': {
                'total_pending': review_stats.get('total_pending', 0),
                'avg_confidence': review_stats.get('avg_confidence', 0.0),
                'avg_uncertainty': review_stats.get('avg_uncertainty', 0.0),
                'reason_distribution': review_stats.get('reason_distribution', {})
            },
            'auto_classifications': {
                'pending_count': len(pending_auto),
                'pending_save': len(pending_auto) > 0,
                'breakdown': self._get_auto_classification_breakdown(pending_auto)
            },
            'workflow_stage': self._determine_workflow_stage(review_stats.get('total_pending', 0), len(pending_auto))
        }

    def _get_auto_classification_breakdown(self, auto_classifications: List[Dict[str, Any]]) -> Dict[str, int]:
        """Ottieni breakdown delle auto-classificazioni per metodo."""
        breakdown = {}
        for classification in auto_classifications:
            method = classification.get('method', 'unknown')
            breakdown[method] = breakdown.get(method, 0) + 1
        return breakdown

    def _determine_workflow_stage(self, pending_reviews: int, pending_auto: int) -> str:
        """Determina lo stage del workflow basato sui conteggi."""
        if pending_reviews == 0 and pending_auto == 0:
            return 'idle'
        elif pending_reviews > 0 and pending_auto > 0:
            return 'analysis_completed_review_needed'
        elif pending_reviews > 0:
            return 'reviewing'
        elif pending_auto > 0:
            return 'pending_save'
        else:
            return 'unknown'

    def get_review_stats(self, tenant: Optional[str] = None) -> Dict[str, Any]:
        """
        Ottieni statistiche complete sulla coda di revisione.
        
        Args:
            tenant: Filtra per tenant specifico (opzionale)
            
        Returns:
            Dict con statistiche dei casi in review
        """
        cases = list(self.pending_reviews.values())
        
        if tenant:
            cases = [case for case in cases if case.tenant == tenant]
        
        if not cases:
            return {
                'total_pending': 0,
                'avg_confidence': 0.0,
                'avg_uncertainty': 0.0,
                'avg_novelty': 0.0,
                'reason_distribution': {},
                'tenant_distribution': {},
                'confidence_distribution': {
                    'high': 0, 'medium': 0, 'low': 0
                },
                'cases_by_disagreement': 0,
                'high_uncertainty_cases': 0,
                'high_novelty_cases': 0
            }
        
        # Calcoli statistici
        total_pending = len(cases)
        avg_confidence = np.mean([case.ml_confidence for case in cases])
        avg_uncertainty = np.mean([case.uncertainty_score for case in cases])
        avg_novelty = np.mean([case.novelty_score for case in cases])
        
        # Distribuzione dei motivi
        reason_distribution = {}
        for case in cases:
            reason = case.reason
            reason_distribution[reason] = reason_distribution.get(reason, 0) + 1
        
        # Distribuzione per tenant
        tenant_distribution = {}
        for case in cases:
            tenant_key = case.tenant
            tenant_distribution[tenant_key] = tenant_distribution.get(tenant_key, 0) + 1
        
        # Distribuzione per livello di confidenza
        confidence_distribution = {'high': 0, 'medium': 0, 'low': 0}
        for case in cases:
            if case.ml_confidence >= 0.8:
                confidence_distribution['high'] += 1
            elif case.ml_confidence >= 0.5:
                confidence_distribution['medium'] += 1
            else:
                confidence_distribution['low'] += 1
        
        return {
            'total_pending': total_pending,
            'avg_confidence': float(avg_confidence),
            'avg_uncertainty': float(avg_uncertainty),
            'avg_novelty': float(avg_novelty),
            'reason_distribution': reason_distribution,
            'tenant_distribution': tenant_distribution,
            'confidence_distribution': confidence_distribution,
            'cases_by_disagreement': len([c for c in cases if c.ml_prediction != c.llm_prediction]),
            'high_uncertainty_cases': len([c for c in cases if c.uncertainty_score > 0.7]),
            'high_novelty_cases': len([c for c in cases if c.novelty_score > 0.7])
        }
    
    def get_human_review_tags(self) -> List[Dict[str, Any]]:
        """
        Ottiene tutti i tag utilizzati nelle revisioni umane per questo tenant.
        
        Returns:
            Lista di dizionari con informazioni sui tag utilizzati nelle revisioni umane
        """
        try:
            tag_counts = {}
            
            # 1. Carica tag dalla cache locale se disponibile
            if hasattr(self, '_human_tags_cache') and self._human_tags_cache:
                for tag, count in self._human_tags_cache.items():
                    tag_counts[tag] = tag_counts.get(tag, 0) + count
                    
            # 2. Carica tag dal log delle decisioni di training
            try:
                if os.path.exists(self.training_log_path):
                    with open(self.training_log_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                decision = json.loads(line.strip())
                                if 'human_decision' in decision:
                                    tag = decision['human_decision'].strip()
                                    if tag:
                                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                self.logger.warning(f"Errore nel caricamento tag dal log training: {e}")
            
            # 3. Carica tag dal database locale tramite TagDatabaseConnector
            try:
                from TagDatabase.tag_database_connector import TagDatabaseConnector
                
                tag_db = TagDatabaseConnector(tenant=self.tenant)
                if tag_db.connetti():
                    # Query per ottenere i tag utilizzati nelle classificazioni manuali per questo tenant
                    query = """
                    SELECT tag_name, COUNT(*) as count
                    FROM session_classifications
                    WHERE tenant_name = %s AND classification_method = 'MANUAL'
                    GROUP BY tag_name
                    ORDER BY count DESC
                    """
                    
                    result = tag_db.esegui_query(query, (self.tenant_name,))
                    if result:
                        for row in result:
                            tag_name = row[0]
                            count = row[1]
                            tag_counts[tag_name] = tag_counts.get(tag_name, 0) + count
                    
                    tag_db.disconnetti()
                    
            except Exception as e:
                self.logger.warning(f"Errore nel caricamento tag dal database: {e}")
            
            # 4. Converte in formato richiesto dal frontend
            tags_list = []
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
                tags_list.append({
                    'tag': tag,
                    'count': count,
                    'source': 'human_review'
                })
            
            # 5. Se non ci sono tag, fornisce un set di default basato sui tag più comuni
            if not tags_list:
                default_tags = [
                    'accesso_portale', 'prenotazione_visita', 'informazioni_generali',
                    'referto_online', 'assistenza_tecnica', 'pagamenti', 'altro'
                ]
                for tag in default_tags:
                    tags_list.append({
                        'tag': tag,
                        'count': 0,
                        'source': 'default'
                    })
            
            self.logger.info(f"Ottenuti {len(tags_list)} tag per revisione umana del tenant {self.tenant_name}")
            return tags_list
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero tag per revisione umana: {e}")
            # Fallback con tag di default
            return [
                {'tag': 'accesso_portale', 'count': 0, 'source': 'default'},
                {'tag': 'prenotazione_visita', 'count': 0, 'source': 'default'},
                {'tag': 'informazioni_generali', 'count': 0, 'source': 'default'},
                {'tag': 'altro', 'count': 0, 'source': 'default'}
            ]
