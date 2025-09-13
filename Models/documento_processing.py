#!/usr/bin/env python3
"""
File: documento_processing.py
Autore: Valerio Bignardi
Data: 2025-09-08
Descrizione: Modello unificato per gestire documenti nel pipeline di classificazione

Storia delle modifiche:
2025-09-08 - Creazione classe DocumentoProcessing con metadati completi
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

@dataclass
class DocumentoProcessing:
    """
    Classe unificata per gestire documenti nel pipeline di classificazione
    
    Scopo della classe: Oggetto che mantiene tutti i metadati necessari 
    durante l'intero flusso di classificazione, clustering e salvataggio
    
    Attributi principali:
    - session_id: Identificativo univoco della sessione
    - testo_completo: Testo completo della conversazione
    - embedding: Vettore di embedding per clustering
    - cluster_info: Informazioni complete sul clustering
    - classification_info: Risultati della classificazione
    - status_info: Status di elaborazione (rappresentante/outlier/propagato)
    
    Autore: Valerio Bignardi
    Data: 2025-09-08
    """
    
    # ========== DATI ORIGINALI ==========
    session_id: str
    testo_completo: str
    embedding: Optional[List[float]] = None
    
    # ========== CLUSTERING METADATA ==========
    cluster_id: Optional[int] = None
    cluster_size: Optional[int] = None
    is_outlier: bool = False
    
    # ========== STATUS DOCUMENTO ==========
    is_representative: bool = False
    is_propagated: bool = False
    propagated_from_cluster: Optional[int] = None
    selection_reason: Optional[str] = None
    
    # ========== CLASSIFICAZIONE ==========
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    classification_method: Optional[str] = None
    reasoning: Optional[str] = None
    
    # ========== PREDIZIONI SPECIFICHE ==========
    ml_prediction: Optional[str] = None
    ml_confidence: Optional[float] = None
    llm_prediction: Optional[str] = None
    llm_confidence: Optional[float] = None
    
    # ========== REVIEW STATUS ==========
    needs_review: bool = False
    review_reason: Optional[str] = None
    human_reviewed: bool = False
    classification_final: Optional[str] = None
    
    # ========== PROPAGAZIONE ==========
    propagated_label: Optional[str] = None
    propagation_consensus: Optional[float] = None
    propagation_reason: Optional[str] = None
    
    # ========== METADATI TECNICI ==========
    processing_timestamp: Optional[str] = None
    classified_by: Optional[str] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        """
        Inizializzazione automatica dopo creazione oggetto
        
        Scopo: Imposta valori di default e timestamp di processing
        
        Autore: Valerio Bignardi
        Data: 2025-09-08
        """
        if self.processing_timestamp is None:
            self.processing_timestamp = datetime.now().isoformat()
    
    def set_clustering_info(self, 
                           cluster_id: int, 
                           cluster_size: int, 
                           is_outlier: bool = False):
        """
        Imposta informazioni di clustering
        
        Scopo: Aggiorna metadati dopo fase di clustering
        Parametri: cluster_id, cluster_size, is_outlier
        
        Args:
            cluster_id: ID del cluster assegnato (-1 per outlier)
            cluster_size: Dimensione del cluster
            is_outlier: True se il documento è un outlier
            
        Autore: Valerio Bignardi
        Data: 2025-09-08
        """
        self.cluster_id = cluster_id
        self.cluster_size = cluster_size
        self.is_outlier = is_outlier
        
        # Determina automaticamente il tipo di documento
        if cluster_id == -1:
            self.is_outlier = True
            self.selection_reason = "outlier_individual_classification"
        else:
            self.is_outlier = False
    
    def set_as_representative(self, selection_reason: str = "representative_selection"):
        """
        Marca il documento come rappresentante
        
        Scopo: Imposta status di rappresentante per review umana
        Parametri: selection_reason per tracciabilità
        
        Args:
            selection_reason: Motivo della selezione come rappresentante
            
        Autore: Valerio Bignardi
        Data: 2025-09-08
        """
        self.is_representative = True
        self.is_propagated = False
        self.selection_reason = selection_reason
        self.needs_review = True
        self.review_reason = "representative_human_review"
    
    def set_as_propagated(self, 
                         propagated_from: int, 
                         propagated_label: str,
                         consensus: float = 0.0,
                         reason: str = "cluster_propagation",
                         ml_prediction: Optional[str] = None,
                         ml_confidence: Optional[float] = None,
                         llm_prediction: Optional[str] = None,
                         llm_confidence: Optional[float] = None,
                         classification_method: Optional[str] = None):
        """
        Marca il documento come propagato ereditando TUTTI i campi dal rappresentante
        
        Scopo: Imposta status di propagato con COMPLETA ereditarietà dei campi classificazione
        Parametri: cluster di origine, label propagata, consenso, predizioni specifiche
        
        🚨 FIX CRITICO: Ora eredita anche ml_prediction e llm_prediction dal rappresentante
        
        Args:
            propagated_from: Cluster ID da cui eredita la classificazione
            propagated_label: Label propagata dai rappresentanti
            consensus: Livello di consenso tra rappresentanti (0-1)
            reason: Motivo della propagazione
            ml_prediction: Predizione ML ereditata dal rappresentante
            ml_confidence: Confidenza ML ereditata dal rappresentante
            llm_prediction: Predizione LLM ereditata dal rappresentante
            llm_confidence: Confidenza LLM ereditata dal rappresentante
            classification_method: Metodo classificazione ereditato dal rappresentante
            
        Autore: Valerio Bignardi
        Data: 2025-09-08
        """
        self.is_representative = False
        self.is_propagated = True
        self.propagated_from_cluster = propagated_from
        self.propagated_label = propagated_label
        self.propagation_consensus = consensus
        self.propagation_reason = reason
        self.selection_reason = "cluster_propagated"
        
        # I propagati sono auto-classificati (non vanno in review automaticamente)
        self.predicted_label = propagated_label
        self.confidence = 0.6 + (consensus * 0.3)  # 0.6-0.9 basato su consenso
        self.classification_method = classification_method or "propagated_from_representatives"
        self.needs_review = False  # Mai review automatico per propagati
        self.reasoning = f"Label propagata da cluster {propagated_from} con consenso {consensus:.1%}"
        
        # 🚨 FIX CRITICO: Eredita TUTTI i campi di classificazione dal rappresentante
        self.ml_prediction = ml_prediction
        self.ml_confidence = ml_confidence
        self.llm_prediction = llm_prediction
        self.llm_confidence = llm_confidence
    
    def set_classification_result(self, 
                                 predicted_label: str,
                                 confidence: float,
                                 method: str = "supervised_training",
                                 reasoning: str = ""):
        """
        Imposta risultato di classificazione
        
        Scopo: Salva risultato della classificazione ML
        Parametri: label predetta, confidenza, metodo, ragionamento
        
        Args:
            predicted_label: Label predetta dal modello
            confidence: Confidenza della predizione (0-1)
            method: Metodo di classificazione utilizzato
            reasoning: Spiegazione del risultato
            
        Autore: Valerio Bignardi
        Data: 2025-09-08
        """
        self.predicted_label = predicted_label
        self.confidence = confidence
        self.classification_method = method
        self.reasoning = reasoning
        self.classified_by = "supervised_training_pipeline"
        
        # 🚀 FIX REVIEW QUEUE: Valuta automaticamente se necessita review umana
        self.evaluate_review_needs()
    
    def evaluate_review_needs(self, 
                             representative_threshold: float = None,
                             outlier_threshold: float = None,
                             propagated_threshold: float = None) -> None:
        """
        🚀 FIX REVIEW QUEUE: Valuta se il documento necessita review umana
        
        Scopo: Implementa la logica mancante per determinare needs_review
        Parametri: soglie di confidence per diversi tipi di documento
        
        LOGICA IMPLEMENTATA:
        - Rappresentanti: confidence < representative_threshold (default 0.85)
        - Outlier: confidence < outlier_threshold (default 0.60) 
        - Propagati: NON vanno mai automaticamente in review
        - Altri documenti: confidence < 0.95 (soglia generale alta)
        
        Args:
            representative_threshold: Soglia confidence per rappresentanti
            outlier_threshold: Soglia confidence per outlier
            propagated_threshold: Soglia confidence per propagati
            
        Autore: Valerio Bignardi
        Data: 2025-09-12
        """
        if self.confidence is None:
            # Senza confidence, non possiamo valutare
            return
        
        # Soglie default se non fornite
        rep_threshold = representative_threshold or 0.85
        out_threshold = outlier_threshold or 0.60
        prop_threshold = propagated_threshold or 0.80
        general_threshold = 0.95
        
        # Reset stato review
        self.needs_review = False
        self.review_reason = None
        
        # LOGICA 1: RAPPRESENTANTI - Soglia più alta per massima qualità
        if self.is_representative:
            if self.confidence < rep_threshold:
                self.needs_review = True
                self.review_reason = f"representative_low_confidence_{self.confidence:.3f}"
                return
        
        # LOGICA 2: OUTLIER - Soglia più bassa ma controllo necessario
        elif self.is_outlier:
            if self.confidence < out_threshold:
                self.needs_review = True
                self.review_reason = f"outlier_low_confidence_{self.confidence:.3f}"
                return
        
        # LOGICA 3: PROPAGATI - Di solito auto-classificati, soglia alta
        elif self.is_propagated:
            # I propagati raramente vanno in review, solo se confidence molto bassa
            if self.confidence < prop_threshold:
                self.needs_review = True
                self.review_reason = f"propagated_very_low_confidence_{self.confidence:.3f}"
                return
        
        # LOGICA 4: DOCUMENTI GENERICI - Soglia alta per sicurezza
        else:
            if self.confidence < general_threshold:
                self.needs_review = True
                self.review_reason = f"general_low_confidence_{self.confidence:.3f}"
                return
        
        # LOGICA 5: CONTROLLI AGGIUNTIVI
        # Etichetta "altro" sempre da rivedere se confidence non altissima
        if (self.predicted_label and 
            self.predicted_label.lower() in ['altro', 'other', 'unknown'] and 
            self.confidence < 0.90):
            self.needs_review = True
            self.review_reason = f"altro_classification_{self.confidence:.3f}"
            return
    
    def evaluate_review_needs_with_db_thresholds(self, tenant_id: str) -> None:
        """
        🚀 FIX REVIEW QUEUE: Valuta needs_review usando soglie dal database MySQL
        
        Scopo: Integra con le soglie configurate nel database TAG per il tenant
        Parametri: tenant_id per recuperare soglie specifiche
        
        Args:
            tenant_id: ID del tenant per recuperare soglie personalizzate
            
        Autore: Valerio Bignardi
        Data: 2025-09-12
        """
        try:
            import yaml
            import mysql.connector
            import os
            
            # Carica configurazione database
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            db_config = config['tag_database']
            
            # Connessione al database
            connection = mysql.connector.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database'],
                autocommit=True
            )
            
            cursor = connection.cursor(dictionary=True)
            
            # Query per recuperare soglie review queue
            query = """
            SELECT 
                representative_confidence_threshold,
                outlier_confidence_threshold,
                propagated_confidence_threshold
            FROM soglie 
            WHERE tenant_id = %s 
            ORDER BY id DESC 
            LIMIT 1
            """
            
            cursor.execute(query, (tenant_id,))
            db_result = cursor.fetchone()
            
            if db_result:
                # Usa soglie dal database
                self.evaluate_review_needs(
                    representative_threshold=float(db_result['representative_confidence_threshold']),
                    outlier_threshold=float(db_result['outlier_confidence_threshold']),
                    propagated_threshold=float(db_result.get('propagated_confidence_threshold', 0.80))
                )
            else:
                # Fallback a soglie default
                self.evaluate_review_needs()
            
            cursor.close()
            connection.close()
            
        except Exception as e:
            # Fallback silenzioso a soglie default in caso di errore DB
            print(f"⚠️ Errore caricamento soglie DB per {tenant_id}: {e}")
            self.evaluate_review_needs()
    
    def get_document_type(self) -> str:
        """
        Determina il tipo di documento per categorizzazione
        
        Scopo: Classificare documento per filtri e interfaccia
        Ritorna: "RAPPRESENTANTE", "PROPAGATO", "OUTLIER"
        
        Returns:
            str: Tipo di documento per categorizzazione
            
        Autore: Valerio Bignardi
        Data: 2025-09-08
        """
        if self.is_representative:
            return "RAPPRESENTANTE"
        elif self.is_propagated:
            return "PROPAGATO"
        elif self.is_outlier:
            return "OUTLIER"
        else:
            return "NON_CLASSIFICATO"
    
    def to_mongo_metadata(self) -> Dict[str, Any]:
        """
        Converte l'oggetto in metadati per salvataggio MongoDB
        
        Scopo: Preparare dati per salvataggio in database
        Ritorna: Dizionario con metadati cluster per MongoDB
        
        Returns:
            Dict: Metadati formattati per MongoDB
            
        Autore: Valerio Bignardi
        Data: 2025-09-08
        """
        return {
            'cluster_id': self.cluster_id,
            'cluster_size': self.cluster_size,
            'is_representative': self.is_representative,
            'is_outlier': self.is_outlier,
            'selection_reason': self.selection_reason,
            'propagated_from': self.propagated_from_cluster,
            'propagation_consensus': self.propagation_consensus,
            'propagation_reason': self.propagation_reason,
        }
    
    def to_classification_decision(self) -> Dict[str, Any]:
        """
        Converte l'oggetto in decisione di classificazione COMPLETA
        
        Scopo: Preparare dati per salvataggio risultato classificazione
        🚨 FIX CRITICO: Ora include anche ml_prediction e llm_prediction
        Ritorna: Dizionario con decisione finale completa
        
        Returns:
            Dict: Decisione di classificazione formattata con TUTTI i campi
            
        Autore: Valerio Bignardi
        Data: 2025-09-08
        """
        return {
            'predicted_label': self.predicted_label or self.propagated_label,
            'confidence': self.confidence or 0.5,
            'method': self.classification_method or 'unified_pipeline',
            'reasoning': self.reasoning or f'Documento {self.get_document_type().lower()}',
            # 🚨 FIX CRITICO: Aggiunti campi mancanti per predizioni specifiche
            'ml_prediction': self.ml_prediction,
            'ml_confidence': self.ml_confidence,
            'llm_prediction': self.llm_prediction,
            'llm_confidence': self.llm_confidence
        }
    
    def get_review_info(self) -> Dict[str, Any]:
        """
        Ottiene informazioni per gestione review
        
        Scopo: Preparare dati per sistema di review
        Ritorna: Dizionario con info review
        
        Returns:
            Dict: Informazioni per sistema review
            
        Autore: Valerio Bignardi
        Data: 2025-09-08
        """
        return {
            'needs_review': self.needs_review,
            'review_reason': self.review_reason or 'pipeline_processing',
            'classified_by': self.classified_by or 'unified_pipeline',
            'notes': self.notes or f'{self.get_document_type()} - {self.selection_reason}'
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte l'oggetto in dizionario completo
        
        Scopo: Serializzazione completa per debug o export
        
        Returns:
            Dict: Rappresentazione completa dell'oggetto
            
        Autore: Valerio Bignardi
        Data: 2025-09-08
        """
        return asdict(self)
    
    def to_json(self) -> str:
        """
        Converte l'oggetto in stringa JSON
        
        Scopo: Serializzazione per logging o storage
        
        Returns:
            str: JSON string dell'oggetto
            
        Autore: Valerio Bignardi
        Data: 2025-09-08
        """
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def __str__(self) -> str:
        """
        Rappresentazione stringa per debug
        
        Scopo: Output leggibile per logging
        
        Returns:
            str: Descrizione dell'oggetto
            
        Autore: Valerio Bignardi
        Data: 2025-09-08
        """
        tipo = self.get_document_type()
        cluster_info = f"Cluster {self.cluster_id}" if self.cluster_id is not None else "No cluster"
        label_info = self.predicted_label or self.propagated_label or "No label"
        
        return f"DocumentoProcessing({self.session_id}: {tipo} - {cluster_info} - {label_info})"
