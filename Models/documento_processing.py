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
                         reason: str = "cluster_propagation"):
        """
        Marca il documento come propagato
        
        Scopo: Imposta status di propagato con label ereditata
        Parametri: cluster di origine, label propagata, consenso
        
        Args:
            propagated_from: Cluster ID da cui eredita la classificazione
            propagated_label: Label propagata dai rappresentanti
            consensus: Livello di consenso tra rappresentanti (0-1)
            reason: Motivo della propagazione
            
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
        self.classification_method = "propagated_from_representatives"
        self.needs_review = False  # Mai review automatico per propagati
        self.reasoning = f"Label propagata da cluster {propagated_from} con consenso {consensus:.1%}"
    
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
        Converte l'oggetto in decisione di classificazione
        
        Scopo: Preparare dati per salvataggio risultato classificazione
        Ritorna: Dizionario con decisione finale
        
        Returns:
            Dict: Decisione di classificazione formattata
            
        Autore: Valerio Bignardi
        Data: 2025-09-08
        """
        return {
            'predicted_label': self.predicted_label or self.propagated_label,
            'confidence': self.confidence or 0.5,
            'method': self.classification_method or 'unified_pipeline',
            'reasoning': self.reasoning or f'Documento {self.get_document_type().lower()}'
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
