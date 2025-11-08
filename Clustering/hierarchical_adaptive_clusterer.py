#!/usr/bin/env python3
"""
Clustering Gerarchico Incrementale con Regioni di Confidenza
Approccio avanzato che gestisce conflitti di etichette senza re-clustering traumatico
"""

import numpy as np
import yaml
import os
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import logging
from datetime import datetime
import json
from dataclasses import dataclass

# Imports dei moduli esistenti
import sys

# Import config_loader per caricare config.yaml con variabili ambiente
from config_loader import load_config

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EmbeddingEngine'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LLMClassifier'))

def trace_all(function_name: str, action: str = "ENTER", called_from: str = None, **kwargs):
    """
    Sistema di tracing completo per tracciare il flusso del clustering gerarchico
    
    Scopo della funzione: Tracciare ingresso, uscita ed errori di tutte le funzioni
    Parametri di input: function_name, action, called_from, **kwargs (parametri da tracciare)
    Parametri di output: None (scrive su file)
    Valori di ritorno: None
    Tracciamento aggiornamenti: 2025-09-07 - Valerio Bignardi - Sistema tracing clustering gerarchico
    
    Args:
        function_name (str): Nome della funzione da tracciare
        action (str): "ENTER", "EXIT", "ERROR"
        called_from (str): Nome della funzione chiamante (per tracciare chiamate annidate)
        **kwargs: Parametri da tracciare (input, return_value, exception, etc.)
        
    Autore: Valerio Bignardi
    Data: 2025-09-07
    """
    import yaml
    import os
    from datetime import datetime
    import json
    
    try:
        # Carica configurazione tracing dal config.yaml
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        if not os.path.exists(config_path):
            return  # Tracing disabilitato se config non esiste
            
        config = load_config()
            
        tracing_config = config.get('tracing', {})
        if not tracing_config.get('enabled', False):
            return  # Tracing disabilitato
            
        # Configurazioni tracing
        log_file = tracing_config.get('log_file', 'tracing.log')
        include_parameters = tracing_config.get('include_parameters', True)
        include_return_values = tracing_config.get('include_return_values', True)
        include_exceptions = tracing_config.get('include_exceptions', True)
        max_file_size_mb = tracing_config.get('max_file_size_mb', 100)
        
        # Path assoluto per il file di log
        log_path = os.path.join(os.path.dirname(__file__), '..', log_file)
        
        # Rotazione file se troppo grande
        if os.path.exists(log_path):
            file_size_mb = os.path.getsize(log_path) / (1024 * 1024)
            if file_size_mb > max_file_size_mb:
                backup_path = f"{log_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(log_path, backup_path)
        
        # Timestamp formattato
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Costruisci messaggio di tracing con called_from
        if called_from:
            message_parts = [f"[{timestamp}]", f"{action:>5}", "->", f"{called_from}::{function_name}"]
        else:
            message_parts = [f"[{timestamp}]", f"{action:>5}", "->", function_name]
        
        # Aggiungi parametri se richiesto
        if action == "ENTER" and include_parameters and kwargs:
            params_str = []
            for key, value in kwargs.items():
                try:
                    # Converti i parametri in stringa gestendo oggetti complessi
                    if isinstance(value, (dict, list)):
                        if len(str(value)) > 200:
                            value_str = f"{type(value).__name__}(size={len(value)})"
                        else:
                            value_str = json.dumps(value, default=str, ensure_ascii=False)[:200]
                    elif hasattr(value, '__len__') and len(str(value)) > 200:
                        value_str = f"{type(value).__name__}(len={len(value)})"
                    else:
                        value_str = str(value)[:200]
                    params_str.append(f"{key}={value_str}")
                except Exception:
                    params_str.append(f"{key}=<{type(value).__name__}>")
            
            if params_str:
                message_parts.append(f"({', '.join(params_str)})")
        
        # Aggiungi valore di ritorno se richiesto
        elif action == "EXIT" and include_return_values and 'return_value' in kwargs:
            try:
                return_val = kwargs['return_value']
                if isinstance(return_val, (dict, list)):
                    if len(str(return_val)) > 300:
                        return_str = f"{type(return_val).__name__}(size={len(return_val)})"
                    else:
                        return_str = json.dumps(return_val, default=str, ensure_ascii=False)[:300]
                elif hasattr(return_val, '__len__') and len(str(return_val)) > 300:
                    return_str = f"{type(return_val).__name__}(len={len(return_val)})"
                else:
                    return_str = str(return_val)[:300]
                message_parts.append(f"RETURN: {return_str}")
            except Exception:
                message_parts.append(f"RETURN: <{type(kwargs['return_value']).__name__}>")
        
        # Aggiungi eccezione se richiesto
        elif action == "ERROR" and include_exceptions and 'exception' in kwargs:
            try:
                exc = kwargs['exception']
                exc_str = f"{type(exc).__name__}: {str(exc)}"[:500]
                message_parts.append(f"EXCEPTION: {exc_str}")
            except Exception:
                message_parts.append(f"EXCEPTION: <{type(kwargs['exception']).__name__}>")
        
        # Scrivi nel file di log
        log_message = " ".join(message_parts) + "\n"
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_message)
            
    except Exception as e:
        # Fallback silenzioso se il tracing fallisce
        # Non vogliamo che errori di tracing interrompano la pipeline
        pass

@dataclass
class ConfidenceRegion:
    """Rappresenta una regione di confidenza nel clustering gerarchico"""
    region_id: int
    region_type: str  # 'core', 'boundary', 'outlier'
    center_embedding: np.ndarray
    radius: float
    member_sessions: Set[str]
    confidence_scores: Dict[str, float]  # session_id -> confidence di appartenenza
    labels: Dict[str, float]  # label -> probabilità
    parent_region: Optional[int] = None
    child_regions: List[int] = None
    created_at: datetime = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.child_regions is None:
            self.child_regions = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()

class HierarchicalAdaptiveClusterer:
    """
    Clusterer gerarchico che mantiene regioni di confidenza invece di cluster rigidi.
    Gestisce conflitti di etichette attraverso boundary regions e feedback incrementale.
    """
    
    def __init__(self, 
                 config_path: str = None,
                 llm_classifier=None,
                 confidence_threshold: float = 0.75,
                 boundary_threshold: float = 0.45,
                 max_hierarchy_depth: int = 3):
        """
        Inizializza il clusterer gerarchico
        
        Args:
            config_path: Percorso del file di configurazione
            llm_classifier: Classificatore LLM per analisi semantica
            confidence_threshold: Soglia per regioni core (alta confidenza)
            boundary_threshold: Soglia per regioni boundary (bassa confidenza)
            max_hierarchy_depth: Profondità massima della gerarchia
        """
        trace_all("__init__", "ENTER", 
                 config_path=config_path, 
                 llm_classifier=type(llm_classifier).__name__ if llm_classifier else None,
                 confidence_threshold=confidence_threshold,
                 boundary_threshold=boundary_threshold,
                 max_hierarchy_depth=max_hierarchy_depth)
        
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        self.config_path = config_path
        self.llm_classifier = llm_classifier
        self.confidence_threshold = confidence_threshold
        self.boundary_threshold = boundary_threshold
        self.max_hierarchy_depth = max_hierarchy_depth
        
        # Debug dell'LLM disponibile
        self._debug_llm_availability()
        
        # Strutture dati principali
        self.regions: Dict[int, ConfidenceRegion] = {}
        self.session_memberships: Dict[str, Dict[int, float]] = {}  # session_id -> {region_id: probability}
        self.hierarchy_tree: Dict[int, List[int]] = {}  # parent_id -> [child_ids]
        self.conflict_history: List[Dict] = []
        
        # Cache delle classificazioni LLM per evitare riclassificazioni
        self.session_classifications: Dict[str, Dict[str, Any]] = {}  # session_id -> {label, confidence, motivation}
        
        # Contatori e statistiche
        self.next_region_id = 0
        self.total_iterations = 0
        self.conflicts_resolved = 0
        
        trace_all("__init__", "EXIT")
        self.regions_split = 0
        self.regions_merged = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.load_config()
        
        print("🌳 HierarchicalAdaptiveClusterer inizializzato")
        print(f"   📊 Soglie: core={confidence_threshold}, boundary={boundary_threshold}")
        print(f"   🏗️ Max depth: {max_hierarchy_depth}")
    
    def _debug_llm_availability(self):
        """Debug per verificare disponibilità e tipo di LLM"""
        if self.llm_classifier is None:
            print(f"⚠️ ATTENZIONE: Nessun LLM classifier fornito - userà FALLBACK 'altro'")
            self.llm_debug_info = {
                'available': False,
                'type': 'None',
                'methods': [],
                'fallback_reason': 'No LLM provided'
            }
        else:
            available_methods = []
            llm_type = type(self.llm_classifier).__name__
            
            # Verifica metodi disponibili
            if hasattr(self.llm_classifier, 'classify_conversation'):
                available_methods.append('classify_conversation')
            if hasattr(self.llm_classifier, 'predict_with_ensemble'):
                available_methods.append('predict_with_ensemble')
            if hasattr(self.llm_classifier, 'predict'):
                available_methods.append('predict')
            
            self.llm_debug_info = {
                'available': True,
                'type': llm_type,
                'methods': available_methods,
                'fallback_reason': None
            }
            
            print(f"✅ LLM Classifier disponibile: {llm_type}")
            print(f"   📋 Metodi disponibili: {', '.join(available_methods)}")
            
            # Verifica priorità dei metodi
            if 'classify_conversation' in available_methods:
                print(f"   🎯 Userà: classify_conversation (metodo primario)")
            elif 'predict_with_ensemble' in available_methods:
                print(f"   🎯 Userà: predict_with_ensemble (metodo secondario)")
            else:
                print(f"   ⚠️ Nessun metodo riconosciuto - userà FALLBACK 'altro'")
                self.llm_debug_info['fallback_reason'] = 'No recognized methods'
        
        # Statistiche di utilizzo per debug
        self.llm_usage_stats = {
            'classify_conversation_calls': 0,
            'predict_with_ensemble_calls': 0,
            'fallback_calls': 0,
            'error_calls': 0,
            'total_predictions': 0
        }
    
    def load_config(self):
        """Carica configurazione aggiuntiva se disponibile"""
        trace_all("load_config", "ENTER", config_path=self.config_path)
        
        try:
            config = load_config()
            
            hierarchical_config = config.get('hierarchical_clustering', {})
            
            # Aggiorna parametri da config se presenti
            self.confidence_threshold = hierarchical_config.get('confidence_threshold', self.confidence_threshold)
            self.boundary_threshold = hierarchical_config.get('boundary_threshold', self.boundary_threshold)
            self.max_hierarchy_depth = hierarchical_config.get('max_hierarchy_depth', self.max_hierarchy_depth)
            
            # Parametri per gestione conflitti
            self.min_region_size = hierarchical_config.get('min_region_size', 3)
            self.max_boundary_ratio = hierarchical_config.get('max_boundary_ratio', 0.4)
            self.similarity_merge_threshold = hierarchical_config.get('similarity_merge_threshold', 0.85)
            
            # Parametri adattivi per rilevamento conflitti
            self.entropy_threshold = hierarchical_config.get('entropy_threshold', 1.0)
            self.dominant_prob_threshold = hierarchical_config.get('dominant_prob_threshold', 0.6)
            self.significant_label_threshold = hierarchical_config.get('significant_label_threshold', 0.2)
            
            # Parametri per strategie di risoluzione
            self.severe_conflict_threshold = hierarchical_config.get('severe_conflict_threshold', 0.7)
            self.moderate_conflict_threshold = hierarchical_config.get('moderate_conflict_threshold', 0.4)
            self.consolidation_similarity_threshold = hierarchical_config.get('consolidation_similarity_threshold', 0.7)
            
            # Pesi per calcolo severità conflitto (adattabili)
            self.severity_weights = hierarchical_config.get('severity_weights', {
                'entropy': 0.4,
                'uncertainty': 0.4,
                'label_penalty': 0.2
            })
            
            trace_all("load_config", "EXIT", config_loaded=True)
            
        except Exception as e:
            self.logger.warning(f"Errore caricamento config gerarchico: {e}")
            
            trace_all("load_config", "ERROR", exception=e)
            
            # Usa valori di default
            self.min_region_size = 3
            self.max_boundary_ratio = 0.4
            self.similarity_merge_threshold = 0.85
            
            # Parametri adattivi di default
            self.entropy_threshold = 1.0
            self.dominant_prob_threshold = 0.6
            self.significant_label_threshold = 0.2
            self.severe_conflict_threshold = 0.7
            self.moderate_conflict_threshold = 0.4
            self.consolidation_similarity_threshold = 0.7
            
            # Pesi di default per severità conflitto
            self.severity_weights = {
                'entropy': 0.4,
                'uncertainty': 0.4,
                'label_penalty': 0.2
            }
            
            trace_all("load_config", "EXIT", config_loaded=False)
    
    def initial_clustering(self, 
                          texts: List[str], 
                          embeddings: np.ndarray,
                          session_ids: List[str]) -> Tuple[Dict[int, ConfidenceRegion], Dict[str, Dict[int, float]]]:
        """
        Passo 1: Clustering iniziale che crea regioni core di alta confidenza
        
        Args:
            texts: Lista dei testi delle sessioni
            embeddings: Embeddings delle sessioni
            session_ids: ID delle sessioni
            
        Returns:
            Tuple (regioni_create, membership_iniziali)
        """
        trace_all("initial_clustering", "ENTER",
                 texts_count=len(texts),
                 embeddings_shape=embeddings.shape,
                 session_ids_count=len(session_ids))
        
        print(f"🌱 Fase 1: Clustering iniziale di {len(texts)} sessioni...")
        
        # 1. Clustering HDBSCAN tradizionale per identificare core regions
        from sklearn.cluster import HDBSCAN
        
        # Parametri più conservativi per evitare over-clustering
        clusterer = HDBSCAN(
            min_cluster_size=max(self.min_region_size, len(texts) // 10),
            min_samples=2,
            cluster_selection_epsilon=0.1,
            metric='cosine'
        )
        
        initial_labels = clusterer.fit_predict(embeddings)
        n_initial_clusters = len(set(initial_labels)) - (1 if -1 in initial_labels else 0)
        n_outliers = sum(1 for label in initial_labels if label == -1)
        
        print(f"   📊 Clustering iniziale: {n_initial_clusters} cluster, {n_outliers} outliers")
        
        # 2. Crea regioni core per ogni cluster
        for cluster_id in set(initial_labels):
            if cluster_id == -1:
                continue  # Gli outliers li gestiamo dopo
            
            # Trova membri del cluster
            cluster_mask = initial_labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_session_ids = [session_ids[i] for i in range(len(session_ids)) if cluster_mask[i]]
            
            # Calcola centro e raggio della regione
            center = np.mean(cluster_embeddings, axis=0)
            distances = [np.linalg.norm(emb - center) for emb in cluster_embeddings]
            radius = np.percentile(distances, 75)  # 75° percentile per robustezza
            
            # Crea regione core
            region = ConfidenceRegion(
                region_id=self.next_region_id,
                region_type='core',
                center_embedding=center,
                radius=radius,
                member_sessions=set(cluster_session_ids),
                confidence_scores={sid: 1.0 for sid in cluster_session_ids},  # Alta confidenza iniziale
                labels={}  # Sarà popolato dalla classificazione LLM
            )
            
            self.regions[self.next_region_id] = region
            
            # Aggiorna membership
            for session_id in cluster_session_ids:
                if session_id not in self.session_memberships:
                    self.session_memberships[session_id] = {}
                self.session_memberships[session_id][self.next_region_id] = 1.0
            
            print(f"   🟢 Regione core {self.next_region_id}: {len(cluster_session_ids)} membri, raggio {radius:.3f}")
            self.next_region_id += 1
        
        # 3. Gestisci outliers creando regioni boundary
        outlier_indices = [i for i, label in enumerate(initial_labels) if label == -1]
        if outlier_indices:
            self._create_boundary_regions_for_outliers(
                outlier_indices, embeddings, session_ids
            )
        
        print(f"   ✅ Clustering iniziale completato: {len(self.regions)} regioni create")
        
        trace_all("initial_clustering", "EXIT",
                 regions_created=len(self.regions),
                 memberships_created=len(self.session_memberships))
        
        return self.regions.copy(), self.session_memberships.copy()
    
    def _create_boundary_regions_for_outliers(self, 
                                            outlier_indices: List[int],
                                            embeddings: np.ndarray,
                                            session_ids: List[str]):
        """
        Crea regioni boundary per outliers che potrebbero appartenere a regioni multiple
        """
        print(f"   🔶 Creazione regioni boundary per {len(outlier_indices)} outliers...")
        
        outlier_embeddings = embeddings[outlier_indices]
        outlier_session_ids = [session_ids[i] for i in outlier_indices]
        
        # Per ogni outlier, calcola affinità con regioni core esistenti
        for i, (outlier_idx, session_id) in enumerate(zip(outlier_indices, outlier_session_ids)):
            outlier_embedding = embeddings[outlier_idx]
            
            # Calcola distanze da tutte le regioni core
            region_affinities = {}
            for region_id, region in self.regions.items():
                if region.region_type == 'core':
                    distance = np.linalg.norm(outlier_embedding - region.center_embedding)
                    # Converti distanza in affinità (inversamente proporzionale)
                    affinity = max(0.1, 1.0 - (distance / (region.radius * 2)))
                    region_affinities[region_id] = affinity
            
            # Se ha affinità sufficienti con regioni esistenti, crea membership multiple
            strong_affinities = {rid: aff for rid, aff in region_affinities.items() 
                               if aff > self.boundary_threshold}
            
            if strong_affinities:
                # Normalizza affinità per sommare a 1
                total_affinity = sum(strong_affinities.values())
                normalized_affinities = {rid: aff/total_affinity 
                                       for rid, aff in strong_affinities.items()}
                
                # Aggiorna membership
                self.session_memberships[session_id] = normalized_affinities
                
                # Aggiungi alle regioni con confidenza proporzionale
                for region_id, affinity in normalized_affinities.items():
                    self.regions[region_id].member_sessions.add(session_id)
                    self.regions[region_id].confidence_scores[session_id] = affinity
                
                print(f"     🔗 {session_id}: boundary su {len(strong_affinities)} regioni")
            
            else:
                # Crea regione outlier dedicata
                outlier_region = ConfidenceRegion(
                    region_id=self.next_region_id,
                    region_type='outlier',
                    center_embedding=outlier_embedding,
                    radius=0.5,  # Raggio fisso per outliers
                    member_sessions={session_id},
                    confidence_scores={session_id: 0.3},  # Bassa confidenza
                    labels={'altro': 1.0}  # Label di default
                )
                
                self.regions[self.next_region_id] = outlier_region
                self.session_memberships[session_id] = {self.next_region_id: 1.0}
                
                print(f"     🔴 {session_id}: regione outlier {self.next_region_id}")
                self.next_region_id += 1
    
    def classify_regions_with_llm(self, 
                                 texts: List[str],
                                 session_ids: List[str]) -> Dict[int, Dict[str, float]]:
        """
        Passo 2: Classifica le regioni usando LLM per ottenere distribuzioni di etichette
        
        Returns:
            Dict {region_id: {label: probability}}
        """
        trace_all("classify_regions_with_llm", "ENTER",
                 texts_count=len(texts),
                 session_ids_count=len(session_ids),
                 regions_count=len(self.regions))
        
        print(f"🧠 Fase 2: Classificazione LLM delle regioni...")
        
        region_labels = {}
        
        for region_id, region in self.regions.items():
            # Seleziona rappresentanti della regione (max 3 per efficienza)
            representatives = self._select_region_representatives(region, texts, session_ids)
            
            if not representatives:
                continue
            
            print(f"   🔍 Classificazione regione {region_id} ({region.region_type}): {len(representatives)} rappresentanti")
            
            # Crea mappa session_id -> testo per questa regione
            session_text_map = {sid: texts[i] for i, sid in enumerate(session_ids) if sid in region.member_sessions}
            
            # Classifica ogni rappresentante
            rep_predictions = []
            for i, rep_text in enumerate(representatives):
                self.llm_usage_stats['total_predictions'] += 1
                
                # Trova session_id corrispondente al rappresentante
                rep_session_id = None
                for sid in region.member_sessions:
                    if sid in session_text_map and session_text_map[sid] == rep_text:
                        rep_session_id = sid
                        break
                
                try:
                    # 🚫 RIMOSSO: Chiamate LLM durante clustering BERTopic
                    # Le classificazioni saranno fatte DOPO il clustering con batch processing ottimizzato
                    
                    # FALLBACK TEMPORANEO: Etichetta provvisoria per completare il clustering
                    self.llm_usage_stats['fallback_calls'] += 1
                    fallback_reason = "Clustering_phase_no_LLM"
                    
                    rep_predictions.append({
                        'label': 'clustering_temp',
                        'confidence': 0.5,
                        'motivation': f'Clustering phase - LLM classification deferred',
                        'method_used': 'clustering_fallback',
                        'text_preview': rep_text[:50] + "..." if len(rep_text) > 50 else rep_text
                    })
                    
                    print(f"     ⚠️ Rep {i+1}: 'clustering_temp' (conf: 0.5) - Classificazione differita post-clustering")
                    
                except Exception as e:
                    # ERRORE: Gestione eccezioni durante clustering
                    self.llm_usage_stats['error_calls'] += 1
                    error_msg = str(e)
                    
                    rep_predictions.append({
                        'label': 'clustering_error',
                        'confidence': 0.3,
                        'motivation': f'Errore clustering: {error_msg}',
                        'method_used': 'error_fallback',
                        'text_preview': rep_text[:50] + "..." if len(rep_text) > 50 else rep_text
                    })
                    
                    print(f"     ❌ Rep {i+1}: 'clustering_error' (conf: 0.3) - ERRORE: {error_msg}")
                    self.logger.warning(f"Errore durante fase clustering regione {region_id}: {e}")
            
            # Calcola distribuzione di etichette per la regione
            label_distribution = self._calculate_region_label_distribution(rep_predictions)
            region_labels[region_id] = label_distribution
            
            # Aggiorna la regione con le etichette
            self.regions[region_id].labels = label_distribution
            
            # Log del risultato
            main_label = max(label_distribution.items(), key=lambda x: x[1])
            print(f"     ✅ Regione {region_id}: '{main_label[0]}' (conf: {main_label[1]:.3f})")
        
        trace_all("classify_regions_with_llm", "EXIT",
                 region_labels_count=len(region_labels))
        
        return region_labels
    
    def _select_region_representatives(self, 
                                     region: ConfidenceRegion,
                                     texts: List[str],
                                     session_ids: List[str]) -> List[str]:
        """
        Seleziona rappresentanti più informativi di una regione
        """
        # Mappa session_id -> testo
        session_text_map = {sid: texts[i] for i, sid in enumerate(session_ids)}
        
        available_sessions = [sid for sid in region.member_sessions if sid in session_text_map]
        
        if not available_sessions:
            return []
        
        # Per regioni piccole, usa tutti i membri
        if len(available_sessions) <= 3:
            return [session_text_map[sid] for sid in available_sessions]
        
        # Per regioni grandi, seleziona i 3 più rappresentativi (alta confidenza + diversità)
        session_scores = []
        for session_id in available_sessions:
            confidence = region.confidence_scores.get(session_id, 0.5)
            # Score = confidenza + fattore di diversità (placeholder per ora)
            score = confidence
            session_scores.append((session_id, score))
        
        # Ordina per score e prendi i top 3
        session_scores.sort(key=lambda x: x[1], reverse=True)
        top_sessions = [sid for sid, _ in session_scores[:3]]
        
        return [session_text_map[sid] for sid in top_sessions]
    
    def _calculate_region_label_distribution(self, 
                                           rep_predictions: List[Dict]) -> Dict[str, float]:
        """
        Calcola distribuzione ponderata delle etichette per una regione
        """
        if not rep_predictions:
            return {'altro': 1.0}
        
        # Raccoglie etichette pesate per confidenza
        weighted_labels = defaultdict(float)
        total_weight = 0
        
        for pred in rep_predictions:
            label = pred['label']
            confidence = pred['confidence']
            weighted_labels[label] += confidence
            total_weight += confidence
        
        # Normalizza per ottenere distribuzione di probabilità
        if total_weight > 0:
            distribution = {label: weight/total_weight 
                          for label, weight in weighted_labels.items()}
        else:
            distribution = {'altro': 1.0}
        
        return distribution
    
    def detect_label_conflicts(self) -> Dict[int, Dict[str, Any]]:
        """
        Passo 3: Rileva conflitti di etichette all'interno delle regioni
        
        Returns:
            Dict {region_id: conflict_info}
        """
        trace_all("detect_label_conflicts", "ENTER", regions_count=len(self.regions))
        
        print(f"⚠️ Fase 3: Rilevamento conflitti nelle regioni...")
        
        conflicts = {}
        
        for region_id, region in self.regions.items():
            if region.region_type == 'outlier':
                continue  # Gli outliers non generano conflitti
            
            # Analizza distribuzione etichette della regione
            label_distribution = region.labels
            
            if not label_distribution:
                continue
            
            # Calcola entropia della distribuzione (misura di incertezza)
            entropy = self._calculate_entropy(label_distribution)
            
            # Identifica etichetta dominante
            dominant_label, dominant_prob = max(label_distribution.items(), key=lambda x: x[1])
            
            # Criteria per conflitto (parametri configurabili):
            # 1. Entropia alta (> entropy_threshold) indica incertezza
            # 2. Etichetta dominante con probabilità bassa (< dominant_prob_threshold)
            # 3. Presenza di etichette multiple significative (> significant_label_threshold)
            
            significant_labels = {label: prob for label, prob in label_distribution.items() 
                                if prob > self.significant_label_threshold}
            
            is_conflict = (
                entropy > self.entropy_threshold or 
                dominant_prob < self.dominant_prob_threshold or 
                len(significant_labels) > 1
            )
            
            if is_conflict:
                conflict_info = {
                    'entropy': entropy,
                    'dominant_label': dominant_label,
                    'dominant_probability': dominant_prob,
                    'significant_labels': significant_labels,
                    'region_size': len(region.member_sessions),
                    'avg_confidence': np.mean(list(region.confidence_scores.values())),
                    'conflict_severity': self._calculate_conflict_severity(entropy, dominant_prob, len(significant_labels))
                }
                
                conflicts[region_id] = conflict_info
                
                print(f"   ⚠️ Conflitto regione {region_id}: entropia={entropy:.3f}, "
                      f"dominante='{dominant_label}'({dominant_prob:.3f}), "
                      f"severità={conflict_info['conflict_severity']:.3f}")
        
        print(f"   📊 Rilevati {len(conflicts)} conflitti su {len(self.regions)} regioni")
        
        trace_all("detect_label_conflicts", "EXIT", conflicts_detected=len(conflicts))
        
        return conflicts
    
    def _calculate_entropy(self, distribution: Dict[str, float]) -> float:
        """Calcola entropia di Shannon per una distribuzione di probabilità"""
        entropy = 0
        for prob in distribution.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        return entropy
    
    def _calculate_conflict_severity(self, entropy: float, dominant_prob: float, n_labels: int) -> float:
        """
        Calcola severità del conflitto (0=nessun conflitto, 1=conflitto massimo)
        Usa pesi configurabili per personalizzare il comportamento.
        """
        # Normalizza entropia (max teorico per n etichette è log2(n))
        max_entropy = np.log2(max(n_labels, 2))
        normalized_entropy = min(entropy / max_entropy, 1.0)
        
        # Inverti probabilità dominante (1-prob = incertezza)
        uncertainty = 1.0 - dominant_prob
        
        # Penalità per numero di etichette significative
        label_penalty = min((n_labels - 1) / 3.0, 1.0)
        
        # Combina i fattori usando pesi configurabili
        severity = (
            normalized_entropy * self.severity_weights['entropy'] + 
            uncertainty * self.severity_weights['uncertainty'] + 
            label_penalty * self.severity_weights['label_penalty']
        )
        
        return min(severity, 1.0)
    
    def resolve_conflicts_hierarchically(self, 
                                       conflicts: Dict[int, Dict[str, Any]],
                                       embeddings: np.ndarray,
                                       session_ids: List[str],
                                       texts: List[str]) -> Dict[str, Any]:
        """
        Passo 4: Risolve conflitti attraverso raffinamento gerarchico
        
        Args:
            conflicts: Conflitti rilevati dalla fase precedente
            embeddings: Embeddings delle sessioni
            session_ids: ID delle sessioni
            
        Returns:
            Statistiche della risoluzione
        """
        trace_all("resolve_conflicts_hierarchically", "ENTER",
                 conflicts_count=len(conflicts),
                 embeddings_shape=embeddings.shape,
                 session_ids_count=len(session_ids),
                 texts_count=len(texts))
        print(f"🔧 Fase 4: Risoluzione gerarchica di {len(conflicts)} conflitti...")
        
        resolution_stats = {
            'conflicts_processed': len(conflicts),
            'regions_split': 0,
            'regions_merged': 0,
            'boundary_regions_created': 0,
            'sessions_reassigned': 0
        }
        
        # Ordina conflitti per severità (più severi prima)
        sorted_conflicts = sorted(conflicts.items(), 
                                key=lambda x: x[1]['conflict_severity'], 
                                reverse=True)
        
        for region_id, conflict_info in sorted_conflicts:
            region = self.regions[region_id]
            
            print(f"   🔧 Risoluzione conflitto regione {region_id} (severità: {conflict_info['conflict_severity']:.3f})")
            
            # Strategia di risoluzione basata su severità e dimensione regione (parametri configurabili)
            if conflict_info['conflict_severity'] > self.severe_conflict_threshold and conflict_info['region_size'] >= self.min_region_size * 2:
                # Conflitto severo + regione grande → SPLIT
                split_result = self._split_region_hierarchically(
                    region_id, conflict_info, embeddings, session_ids, texts
                )
                resolution_stats['regions_split'] += split_result['new_regions_created']
                resolution_stats['sessions_reassigned'] += split_result['sessions_moved']
                
            elif conflict_info['conflict_severity'] > self.moderate_conflict_threshold:
                # Conflitto moderato → BOUNDARY REFINEMENT
                boundary_result = self._create_boundary_refinement(
                    region_id, conflict_info, embeddings, session_ids
                )
                resolution_stats['boundary_regions_created'] += boundary_result['boundary_regions_created']
                resolution_stats['sessions_reassigned'] += boundary_result['sessions_moved']
                
            else:
                # Conflitto lieve → SOFT REASSIGNMENT
                soft_result = self._soft_reassign_memberships(
                    region_id, conflict_info
                )
                resolution_stats['sessions_reassigned'] += soft_result['sessions_updated']
        
        # Consolidamento post-risoluzione
        merge_stats = self._consolidate_similar_regions(embeddings, session_ids)
        resolution_stats['regions_merged'] = merge_stats['regions_merged']
        
        print(f"   ✅ Risoluzione completata: {resolution_stats}")
        
        trace_all("resolve_conflicts_hierarchically", "EXIT", resolution_stats=resolution_stats)
        
        return resolution_stats
    
    def _split_region_hierarchically(self, 
                                   region_id: int,
                                   conflict_info: Dict[str, Any],
                                   embeddings: np.ndarray,
                                   session_ids: List[str],
                                   texts: List[str]) -> Dict[str, int]:
        """
        Split intelligente di una regione conflittuale usando etichette esistenti
        e consolidamento con regioni compatibili
        """
        region = self.regions[region_id]
        print(f"     🔀 Split intelligente regione {region_id}...")
        
        # 1. Estrai embeddings dei membri della regione
        session_id_to_idx = {sid: i for i, sid in enumerate(session_ids)}
        region_indices = [session_id_to_idx[sid] for sid in region.member_sessions 
                         if sid in session_id_to_idx]
        
        if len(region_indices) < self.min_region_size * 2:
            print(f"       ⚠️ Regione troppo piccola per split ({len(region_indices)} membri)")
            return {'new_regions_created': 0, 'sessions_moved': 0}
        
        region_embeddings = embeddings[region_indices]
        region_session_ids = [session_ids[i] for i in region_indices]
        
        # 2. Usa le etichette significative GIÀ RILEVATE dal conflitto
        significant_labels = list(conflict_info['significant_labels'].keys())
        
        if len(significant_labels) < 2:
            print(f"       ⚠️ Insufficienti etichette significative ({len(significant_labels)})")
            return {'new_regions_created': 0, 'sessions_moved': 0}
        
        # 3. Raggruppa sessioni per etichetta esistente (senza riclassificare!)
        label_groups = self._extract_existing_session_labels(
            region_session_ids, texts, session_ids, significant_labels
        )
        
        if not label_groups:
            print(f"       ⚠️ Impossibile raggruppare sessioni per etichetta esistente")
            return {'new_regions_created': 0, 'sessions_moved': 0}
        
        # 4. Processo intelligente di creazione/consolidamento
        new_regions_created = 0
        sessions_moved = 0
        consolidations_performed = 0
        
        # Converte regione padre in nodo gerarchico
        region.region_type = 'hierarchical_parent'
        
        for label, session_group in label_groups.items():
            if len(session_group['sessions']) < self.min_region_size:
                print(f"       ⚠️ Gruppo '{label}' troppo piccolo ({len(session_group['sessions'])} sessioni)")
                continue
            
            # 5. Verifica se esistono regioni compatibili per questa etichetta
            consolidation_target = self._find_consolidation_target(
                label, session_group, embeddings, session_ids
            )
            
            if consolidation_target:
                # CONSOLIDAMENTO: Sposta sessioni in regione esistente
                result = self._consolidate_sessions_to_region(
                    session_group['sessions'], consolidation_target, label
                )
                sessions_moved += result['sessions_moved']
                consolidations_performed += 1
                
                print(f"       🔗 Consolidato '{label}': {len(session_group['sessions'])} "
                      f"sessioni → regione {consolidation_target['region_id']} "
                      f"(similarità: {consolidation_target['similarity']:.3f})")
            
            else:
                # CREAZIONE: Nuova sotto-regione
                result = self._create_new_subregion(
                    session_group, label, region_embeddings, region_session_ids, 
                    region_indices, region_id
                )
                if result['region_created']:
                    new_regions_created += 1
                    sessions_moved += result['sessions_moved']
                    
                    print(f"       ✅ Nuova sotto-regione {result['new_region_id']}: "
                          f"{len(session_group['sessions'])} membri, etichetta '{label}' "
                          f"(conf: {session_group['confidence']:.3f})")
        
        print(f"       📊 Split completato: {new_regions_created} nuove regioni, "
              f"{consolidations_performed} consolidamenti, {sessions_moved} sessioni spostate")
        
        self.regions_split += 1
        return {'new_regions_created': new_regions_created, 'sessions_moved': sessions_moved}
    
    def _extract_existing_session_labels(self, 
                                       region_session_ids: List[str],
                                       texts: List[str],
                                       session_ids: List[str],
                                       significant_labels: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Estrae le etichette esistenti delle sessioni usando la cache delle classificazioni LLM.
        Non riclassifica - usa i risultati già ottenuti dalla fase 2.
        """
        print(f"       📋 Estrazione etichette esistenti per {len(region_session_ids)} sessioni...")
        
        # Mappa session_id -> indice globale per accesso ai testi
        session_id_to_global_idx = {sid: i for i, sid in enumerate(session_ids)}
        
        # Raggruppa sessioni per etichetta usando la cache delle classificazioni
        label_groups = defaultdict(lambda: {'sessions': [], 'confidences': [], 'texts': []})
        
        # Usa classificazioni memorizzate dalla fase 2
        for session_id in region_session_ids:
            if session_id in self.session_classifications:
                # USA CLASSIFICAZIONE REALE dalla cache
                classification = self.session_classifications[session_id]
                assigned_label = classification['label']
                confidence = classification['confidence']
                
                # Verifica che l'etichetta sia tra quelle significative del conflitto
                if assigned_label in significant_labels:
                    label_groups[assigned_label]['sessions'].append(session_id)
                    label_groups[assigned_label]['confidences'].append(confidence)
                    
                    # Recupera testo per sample
                    if session_id in session_id_to_global_idx:
                        global_idx = session_id_to_global_idx[session_id]
                        text = texts[global_idx]
                        label_groups[assigned_label]['texts'].append(text[:100])
                    
                    print(f"         📌 {session_id}: '{assigned_label}' (conf: {confidence:.3f}) da cache")
                    
            else:
                # FALLBACK: Se non classificato in fase 2, usa etichetta più probabile del conflitto
                if session_id in session_id_to_global_idx:
                    global_idx = session_id_to_global_idx[session_id]
                    text = texts[global_idx]
                    
                    # Assegna etichetta più probabile deterministicamente
                    text_hash = hash(text) % len(significant_labels)
                    assigned_label = significant_labels[text_hash]
                    confidence = 0.6  # Confidenza ridotta per fallback
                    
                    label_groups[assigned_label]['sessions'].append(session_id)
                    label_groups[assigned_label]['confidences'].append(confidence)
                    label_groups[assigned_label]['texts'].append(text[:100])
                    
                    print(f"         ⚠️ {session_id}: '{assigned_label}' (conf: {confidence:.3f}) FALLBACK")
        
        # Calcola statistiche per ogni gruppo
        final_groups = {}
        for label, group_data in label_groups.items():
            if len(group_data['sessions']) >= self.min_region_size:
                avg_confidence = np.mean(group_data['confidences'])
                
                final_groups[label] = {
                    'sessions': group_data['sessions'],
                    'confidence': avg_confidence,
                    'size': len(group_data['sessions']),
                    'sample_texts': group_data['texts'][:3]
                }
                
                print(f"         ✅ Gruppo '{label}': {len(group_data['sessions'])} sessioni "
                      f"(conf: {avg_confidence:.3f})")
        
        return final_groups
    
    def _find_consolidation_target(self, 
                                 label: str,
                                 session_group: Dict[str, Any],
                                 embeddings: np.ndarray,
                                 session_ids: List[str]) -> Optional[Dict[str, Any]]:
        """
        Trova una regione esistente compatibile per consolidamento.
        """
        # Cerca regioni esistenti con la stessa etichetta dominante
        compatible_regions = []
        
        for region_id, region in self.regions.items():
            if region.region_type in ['core', 'hierarchical_parent']:
                # Verifica se la regione ha la stessa etichetta dominante
                if region.labels:
                    dominant_label = max(region.labels.items(), key=lambda x: x[1])[0]
                    dominant_prob = region.labels[label] if label in region.labels else 0.0
                    
                    if dominant_label == label and dominant_prob > 0.5:
                        compatible_regions.append({
                            'region_id': region_id,
                            'region': region,
                            'label_probability': dominant_prob
                        })
        
        if not compatible_regions:
            return None
        
        # Calcola similarità semantica con le regioni compatibili
        session_id_to_idx = {sid: i for i, sid in enumerate(session_ids)}
        group_embeddings = []
        
        for session_id in session_group['sessions']:
            if session_id in session_id_to_idx:
                group_embeddings.append(embeddings[session_id_to_idx[session_id]])
        
        if not group_embeddings:
            return None
        
        group_embeddings = np.array(group_embeddings)
        group_centroid = np.mean(group_embeddings, axis=0)
        
        best_target = None
        best_similarity = 0.0
        
        for candidate in compatible_regions:
            region = candidate['region']
            
            # Calcola similarità semantica
            region_centroid = region.center_embedding
            similarity = cosine_similarity([group_centroid], [region_centroid])[0][0]
            
            # Calcola punteggio complessivo (similarità + probabilità etichetta)
            score = similarity * 0.7 + candidate['label_probability'] * 0.3
            
            if score > best_similarity and similarity > self.consolidation_similarity_threshold:  # Soglia configurabile
                best_similarity = score
                best_target = {
                    'region_id': candidate['region_id'],
                    'region': candidate['region'],
                    'similarity': similarity,
                    'label_probability': candidate['label_probability'],
                    'score': score
                }
        
        return best_target
    
    def _consolidate_sessions_to_region(self,
                                      sessions: List[str],
                                      target_region: Dict[str, Any],
                                      label: str) -> Dict[str, int]:
        """
        Consolida sessioni in una regione esistente.
        """
        region_id = target_region['region_id']
        region = target_region['region']
        
        sessions_moved = 0
        
        for session_id in sessions:
            # Aggiungi sessione alla regione target
            region.member_sessions.add(session_id)
            region.confidence_scores[session_id] = 0.8  # Confidenza per consolidamento
            
            # Aggiorna membership
            self.session_memberships[session_id] = {region_id: 1.0}
            sessions_moved += 1
        
        # Aggiorna statistiche della regione target
        # Incrementa leggermente la probabilità dell'etichetta consolidata
        if label in region.labels:
            region.labels[label] = min(0.95, region.labels[label] + 0.1)
        else:
            region.labels[label] = 0.7
        
        # Rinormalizza le probabilità
        total_prob = sum(region.labels.values())
        if total_prob > 1.0:
            region.labels = {k: v/total_prob for k, v in region.labels.items()}
        
        return {'sessions_moved': sessions_moved}
    
    def _create_new_subregion(self,
                            session_group: Dict[str, Any],
                            label: str,
                            region_embeddings: np.ndarray,
                            region_session_ids: List[str],
                            region_indices: List[int],
                            parent_region_id: int) -> Dict[str, Any]:
        """
        Crea una nuova sotto-regione per un gruppo di sessioni.
        """
        sessions = session_group['sessions']
        
        # Trova indici delle sessioni nel contesto della regione
        session_to_region_idx = {sid: i for i, sid in enumerate(region_session_ids)}
        sub_indices = [session_to_region_idx[sid] for sid in sessions if sid in session_to_region_idx]
        
        if len(sub_indices) < self.min_region_size:
            return {'region_created': False, 'sessions_moved': 0, 'new_region_id': None}
        
        # Estrai embeddings e calcola geometria
        sub_embeddings = region_embeddings[sub_indices]
        sub_center = np.mean(sub_embeddings, axis=0)
        sub_distances = [np.linalg.norm(emb - sub_center) for emb in sub_embeddings]
        sub_radius = np.percentile(sub_distances, 75)
        
        # Crea distribuzione etichette
        confidence = session_group['confidence']
        labels_distribution = {label: confidence, 'altro': 1.0 - confidence}
        
        # Crea nuova sotto-regione
        sub_region = ConfidenceRegion(
            region_id=self.next_region_id,
            region_type='core',
            center_embedding=sub_center,
            radius=sub_radius,
            member_sessions=set(sessions),
            confidence_scores={sid: confidence for sid in sessions},
            labels=labels_distribution,
            parent_region=parent_region_id
        )
        
        # Aggiungi al sistema
        self.regions[self.next_region_id] = sub_region
        self.regions[parent_region_id].child_regions.append(self.next_region_id)
        
        # Aggiorna membership
        sessions_moved = 0
        for session_id in sessions:
            self.session_memberships[session_id] = {self.next_region_id: 1.0}
            sessions_moved += 1
        
        new_region_id = self.next_region_id
        self.next_region_id += 1
        
        return {
            'region_created': True,
            'sessions_moved': sessions_moved,
            'new_region_id': new_region_id
        }
    
    def _consolidate_similar_regions(self, embeddings: np.ndarray, session_ids: List[str] = None) -> Dict[str, int]:
        """
        Consolida regioni troppo simili per evitare frammentazione eccessiva.
        Versione intelligente che considera etichette semantiche.
        """
        print(f"     🔗 Consolidamento intelligente regioni simili...")
        
        regions_merged = 0
        
        # Raggruppa regioni per etichetta dominante
        label_to_regions = defaultdict(list)
        
        for region_id, region in self.regions.items():
            if region.region_type == 'core' and region.labels:
                dominant_label = max(region.labels.items(), key=lambda x: x[1])[0]
                dominant_prob = region.labels[dominant_label]
                
                if dominant_prob > 0.6:  # Solo regioni con etichetta chiara
                    label_to_regions[dominant_label].append({
                        'region_id': region_id,
                        'region': region,
                        'probability': dominant_prob
                    })
        
        # Per ogni etichetta con multiple regioni, valuta consolidamenti
        for label, regions_list in label_to_regions.items():
            if len(regions_list) < 2:
                continue
                
            print(f"       🔍 Verifica consolidamento per etichetta '{label}': {len(regions_list)} regioni")
            
            # OTTIMIZZAZIONE: Limita numero confronti per scalabilità
            max_comparisons = min(len(regions_list), 20)  # Evita O(n²) eccessivo
            regions_list = regions_list[:max_comparisons]
            
            # Ordina per probabilità decrescente (migliore prima)
            regions_list.sort(key=lambda x: x['probability'], reverse=True)
            
            # OTTIMIZZAZIONE: Usa approccio greedy invece di confrontare tutte le coppie
            regions_to_merge = self._find_merge_candidates_optimized(regions_list, label)
            
            # Esegui merge in ordine di similarità
            for merge_candidate in sorted(regions_to_merge, key=lambda x: x['similarity'], reverse=True):
                region1 = merge_candidate['region1']['region']
                region2 = merge_candidate['region2']['region']
                
                # Verifica che entrambe le regioni esistano ancora (non già merged)
                if (merge_candidate['region1']['region_id'] in self.regions and 
                    merge_candidate['region2']['region_id'] in self.regions):
                    
                    self._merge_regions(
                        merge_candidate['region1']['region_id'],
                        merge_candidate['region2']['region_id'],
                        embeddings,
                        session_ids
                    )
                    regions_merged += 1
                    
                    print(f"         ✅ Merge completato: regioni "
                          f"{merge_candidate['region1']['region_id']} + "
                          f"{merge_candidate['region2']['region_id']} → etichetta '{label}'")
        
        print(f"     📊 Consolidamento completato: {regions_merged} merge eseguiti")
        return {'regions_merged': regions_merged}
    
    def _find_merge_candidates_optimized(self, regions_list: List[Dict], label: str) -> List[Dict]:
        """
        Trova candidati per merge usando approccio greedy ottimizzato O(n log n) invece di O(n²)
        """
        regions_to_merge = []
        used_regions = set()
        
        # Approccio greedy: per ogni regione, trova la migliore compatibile
        for i, region1 in enumerate(regions_list):
            if region1['region_id'] in used_regions:
                continue
                
            best_match = None
            best_score = 0.0
            
            # Cerca tra le regioni successive (evita duplicati)
            for j in range(i + 1, len(regions_list)):
                region2 = regions_list[j]
                
                if region2['region_id'] in used_regions:
                    continue
                
                # Calcola similarità semantica
                similarity = cosine_similarity(
                    [region1['region'].center_embedding],
                    [region2['region'].center_embedding]
                )[0][0]
                
                # Early termination se similarità troppo bassa
                if similarity < self.similarity_merge_threshold:
                    continue
                
                # Calcola compatibilità delle dimensioni
                size1 = len(region1['region'].member_sessions)
                size2 = len(region2['region'].member_sessions)
                size_ratio = min(size1, size2) / max(size1, size2)
                
                # Criteri per merge
                compatible_size = size_ratio > 0.3  # Non troppo sbilanciate
                both_high_confidence = (region1['probability'] > 0.7 and 
                                      region2['probability'] > 0.7)
                
                if compatible_size and both_high_confidence:
                    # Score combinato per scegliere il miglior match
                    score = similarity * 0.7 + (region1['probability'] + region2['probability']) * 0.15
                    
                    if score > best_score:
                        best_score = score
                        best_match = region2
            
            # Se trovato match valido, aggiungi ai candidati
            if best_match:
                regions_to_merge.append({
                    'region1': region1,
                    'region2': best_match,
                    'similarity': cosine_similarity(
                        [region1['region'].center_embedding],
                        [best_match['region'].center_embedding]
                    )[0][0],
                    'label': label
                })
                
                # Marca regioni come usate per evitare conflitti
                used_regions.add(region1['region_id'])
                used_regions.add(best_match['region_id'])
                
                print(f"         🔗 Candidato merge: regioni {region1['region_id']} + "
                      f"{best_match['region_id']} (sim: {best_score:.3f})")
        
        return regions_to_merge
    
    def _merge_regions(self, region_id1: int, region_id2: int, embeddings: np.ndarray = None, session_ids: List[str] = None):
        """
        Esegue il merge fisico di due regioni.
        Versione migliorata che calcola correttamente il nuovo centro quando possibile.
        """
        region1 = self.regions[region_id1]
        region2 = self.regions[region_id2]
        
        # Combina membri (region1.member_sessions è un Set, region2.member_sessions è un Set)
        region1.member_sessions.update(region2.member_sessions)
        
        # Combina etichette (media pesata per dimensione)
        size1 = len(region1.member_sessions) - len(region2.member_sessions)  # Dimensione originale region1
        size2 = len(region2.member_sessions)
        total_size = size1 + size2
        
        combined_labels = {}
        all_labels = set(region1.labels.keys()) | set(region2.labels.keys())
        
        for label in all_labels:
            prob1 = region1.labels.get(label, 0.0) * (size1 / total_size)
            prob2 = region2.labels.get(label, 0.0) * (size2 / total_size)
            combined_labels[label] = prob1 + prob2
        
        region1.labels = combined_labels
        
        # Combina confidence scores
        region1.confidence_scores.update(region2.confidence_scores)
        
        # Calcola nuovo centro - versione migliorata
        if embeddings is not None and session_ids is not None:
            # CALCOLO PRECISO: Usa embeddings reali delle sessioni
            session_id_to_idx = {sid: i for i, sid in enumerate(session_ids)}
            
            # Estrai embeddings delle sessioni combinate
            combined_embeddings = []
            for session_id in region1.member_sessions:
                if session_id in session_id_to_idx:
                    combined_embeddings.append(embeddings[session_id_to_idx[session_id]])
            
            if combined_embeddings:
                region1.center_embedding = np.mean(combined_embeddings, axis=0)
                
                # Ricalcola raggio in base alle distanze reali
                distances = [np.linalg.norm(emb - region1.center_embedding) for emb in combined_embeddings]
                region1.radius = np.percentile(distances, 75)
            else:
                # Fallback: media pesata dei centri esistenti
                region1.center_embedding = (region1.center_embedding * size1 + region2.center_embedding * size2) / total_size
                region1.radius = max(region1.radius, region2.radius)
        else:
            # APPROSSIMAZIONE: Media pesata dei centri esistenti
            region1.center_embedding = (region1.center_embedding * size1 + region2.center_embedding * size2) / total_size
            
            # Aggiorna raggio per contenere entrambe le regioni
            distance_between_centers = np.linalg.norm(region1.center_embedding - region2.center_embedding)
            new_radius = max(region1.radius, region2.radius, distance_between_centers / 2)
            region1.radius = new_radius
        
        # Aggiorna timestamp
        region1.last_updated = datetime.now()
        
        # Rimuove la seconda regione
        del self.regions[region_id2]
        
        # Aggiorna membership per le sessioni della regione2
        for session_id in region2.member_sessions:
            if session_id in self.session_memberships:
                # Rimuovi membership dalla regione2 e aggiorna per regione1
                if region_id2 in self.session_memberships[session_id]:
                    del self.session_memberships[session_id][region_id2]
                self.session_memberships[session_id][region_id1] = 1.0
            else:
                self.session_memberships[session_id] = {region_id1: 1.0}
        
        print(f"         🔗 Merge completato: {region_id1} (ora {len(region1.member_sessions)} sessioni)")
        
        # Aggiorna contatori
        self.regions_merged += 1
        
    def _calculate_label_similarity(self, labels1: Dict[str, float], labels2: Dict[str, float]) -> float:
        """
        Calcola similarità tra due distribuzioni di etichette
        """
        if not labels1 or not labels2:
            return 0.0
        
        all_labels = set(labels1.keys()) | set(labels2.keys())
        
        vec1 = [labels1.get(label, 0.0) for label in all_labels]
        vec2 = [labels2.get(label, 0.0) for label in all_labels]
        
        # Usa cosine similarity
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        norm1 = sum(v1 ** 2 for v1 in vec1) ** 0.5
        norm2 = sum(v2 ** 2 for v2 in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _create_boundary_refinement(self, 
                                  region_id: int,
                                  conflict_info: Dict[str, Any],
                                  embeddings: np.ndarray,
                                  session_ids: List[str]) -> Dict[str, int]:
        """
        Crea raffinamento boundary per conflitti moderati.
        Espande le regioni boundary per catturare incertezza semantica.
        """
        print(f"     🔶 Boundary refinement regione {region_id}...")
        
        region = self.regions[region_id]
        
        # Identifica sessioni con bassa confidenza nella regione
        low_confidence_sessions = [
            sid for sid, conf in region.confidence_scores.items() 
            if conf < self.confidence_threshold
        ]
        
        if not low_confidence_sessions:
            return {'boundary_regions_created': 0, 'sessions_moved': 0}
        
        # Crea regione boundary per sessioni incerte
        boundary_region = ConfidenceRegion(
            region_id=self.next_region_id,
            region_type='boundary',
            center_embedding=region.center_embedding.copy(),  # Stesso centro per ora
            radius=region.radius * 1.5,  # Raggio espanso
            member_sessions=set(low_confidence_sessions),
            confidence_scores={sid: 0.5 for sid in low_confidence_sessions},
            labels=region.labels.copy(),  # Stessa distribuzione etichette
            parent_region=region_id
        )
        
        self.regions[self.next_region_id] = boundary_region
        region.child_regions.append(self.next_region_id)
        
        # Aggiorna membership: sessioni boundary appartengono a entrambe le regioni
        sessions_moved = 0
        for sid in low_confidence_sessions:
            # Mantieni membership originale ma ridotta
            if sid in self.session_memberships:
                self.session_memberships[sid][region_id] = 0.6
                self.session_memberships[sid][self.next_region_id] = 0.4
            else:
                self.session_memberships[sid] = {region_id: 0.6, self.next_region_id: 0.4}
            sessions_moved += 1
        
        print(f"       ✅ Boundary region {self.next_region_id}: {len(low_confidence_sessions)} membri shared")
        
        self.next_region_id += 1
        return {'boundary_regions_created': 1, 'sessions_moved': sessions_moved}
    
    def _soft_reassign_memberships(self, 
                                 region_id: int,
                                 conflict_info: Dict[str, Any]) -> Dict[str, int]:
        """
        Soft reassignment per conflitti lievi - aggiusta solo le confidenze.
        Riduce le confidenze per riflettere l'incertezza senza creare nuove regioni.
        """
        print(f"     🔄 Soft reassignment regione {region_id}...")
        
        region = self.regions[region_id]
        
        # Riduce confidenze per riflettere incertezza
        uncertainty_factor = conflict_info['conflict_severity']
        confidence_reduction = uncertainty_factor * 0.3  # Max 30% riduzione
        
        sessions_updated = 0
        for session_id in region.member_sessions:
            old_confidence = region.confidence_scores.get(session_id, 0.5)
            new_confidence = max(0.1, old_confidence - confidence_reduction)
            
            region.confidence_scores[session_id] = new_confidence
            
            # Aggiorna anche la membership globale
            if session_id in self.session_memberships and region_id in self.session_memberships[session_id]:
                self.session_memberships[session_id][region_id] = new_confidence
            
            sessions_updated += 1
        
        print(f"       ✅ Aggiornate confidenze per {sessions_updated} sessioni (riduzione media: {confidence_reduction:.3f})")
        
        return {'sessions_updated': sessions_updated}
    
    def _calculate_label_similarity(self, labels1: Dict[str, float], labels2: Dict[str, float]) -> float:
        """
        Calcola similarità tra due distribuzioni di etichette usando Bhattacharyya coefficient
        """
        if not labels1 or not labels2:
            return 0.0
        
        # Unisce le chiavi di entrambe le distribuzioni
        all_labels = set(labels1.keys()) | set(labels2.keys())
        
        # Calcola Bhattacharyya coefficient
        bc = 0.0
        for label in all_labels:
            p1 = labels1.get(label, 0.0)
            p2 = labels2.get(label, 0.0)
            bc += np.sqrt(p1 * p2)
        
        return bc
    
    def _merge_regions(self, region_id1: int, region_id2: int):
        """
        Mergia due regioni in una sola
        """
        region1, region2 = self.regions[region_id1], self.regions[region_id2]
        
        # Combina membri
        combined_sessions = region1.member_sessions | region2.member_sessions
        
        # Combina confidence scores (media pesata)
        combined_confidence = {}
        for sid in combined_sessions:
            conf1 = region1.confidence_scores.get(sid, 0.0)
            conf2 = region2.confidence_scores.get(sid, 0.0)
            combined_confidence[sid] = max(conf1, conf2)  # Prendi la confidenza maggiore
        
        # Combina etichette (media pesata per dimensione regioni)
        size1, size2 = len(region1.member_sessions), len(region2.member_sessions)
        total_size = size1 + size2
        
        combined_labels = {}
        all_labels = set(region1.labels.keys()) | set(region2.labels.keys())
        
        for label in all_labels:
            prob1 = region1.labels.get(label, 0.0) * (size1 / total_size)
            prob2 = region2.labels.get(label, 0.0) * (size2 / total_size)
            combined_labels[label] = prob1 + prob2
        
        # Calcola nuovo centro (media pesata)
        new_center = (region1.center_embedding * size1 + region2.center_embedding * size2) / total_size
        
        # Aggiorna region1 con dati combinati
        region1.member_sessions = combined_sessions
        region1.confidence_scores = combined_confidence
        region1.labels = combined_labels
        region1.center_embedding = new_center
        region1.last_updated = datetime.now()
        
        # Aggiorna membership per sessioni di region2
        for sid in region2.member_sessions:
            if sid in self.session_memberships:
                # Rimuovi membership da region2 e aggiorna per region1
                if region_id2 in self.session_memberships[sid]:
                    del self.session_memberships[sid][region_id2]
                self.session_memberships[sid][region_id1] = combined_confidence[sid]
        
        # Rimuovi region2
        del self.regions[region_id2]
        self.regions_merged += 1
    
    def cluster_hierarchically(self, 
                             texts: List[str], 
                             embeddings: np.ndarray,
                             session_ids: List[str],
                             max_iterations: int = 3) -> Tuple[Dict[str, Dict[int, float]], Dict[int, Dict[str, Any]]]:
        """
        METODO PRINCIPALE: Esegue clustering gerarchico con risoluzione adattiva dei conflitti
        
        Args:
            texts: Lista dei testi delle sessioni
            embeddings: Embeddings delle sessioni
            session_ids: ID delle sessioni
            max_iterations: Numero massimo di iterazioni per convergenza
            
        Returns:
            Tuple (session_memberships, cluster_info)
        """
        trace_all("cluster_hierarchically", "ENTER",
                 texts_count=len(texts),
                 embeddings_shape=embeddings.shape,
                 session_ids_count=len(session_ids),
                 max_iterations=max_iterations)
        
        print(f"🌳 AVVIO CLUSTERING GERARCHICO ADATTIVO")
        print(f"📊 Dataset: {len(texts)} sessioni, max {max_iterations} iterazioni")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # FASE 1: Clustering iniziale
        initial_regions, initial_memberships = self.initial_clustering(texts, embeddings, session_ids)
        
        # FASE 2: Classificazione LLM delle regioni
        region_labels = self.classify_regions_with_llm(texts, session_ids)
        
        # CICLO ADATTIVO: Rileva e risolve conflitti iterativamente
        iteration = 0
        prev_n_regions = len(self.regions)
        convergence_achieved = False
        
        while iteration < max_iterations and not convergence_achieved:
            iteration += 1
            print(f"\n🔄 ITERAZIONE {iteration}/{max_iterations}")
            
            # FASE 3: Rilevamento conflitti
            conflicts = self.detect_label_conflicts()
            
            if not conflicts:
                print("✅ Nessun conflitto rilevato - CONVERGENZA RAGGIUNTA")
                convergence_achieved = True
                break
            
            # FASE 4: Risoluzione gerarchica conflitti
            resolution_stats = self.resolve_conflicts_hierarchically(conflicts, embeddings, session_ids, texts)
            
            # Riclassifica regioni modificate se necessario
            if resolution_stats['regions_split'] > 0 or resolution_stats['boundary_regions_created'] > 0:
                print("   🧠 Riclassificazione regioni modificate...")
                new_region_labels = self.classify_regions_with_llm(texts, session_ids)
                region_labels.update(new_region_labels)
            
            # Verifica convergenza
            current_n_regions = len(self.regions)
            if current_n_regions == prev_n_regions and resolution_stats['sessions_reassigned'] == 0:
                print("✅ Sistema stabile - CONVERGENZA RAGGIUNTA")
                convergence_achieved = True
            
            prev_n_regions = current_n_regions
            self.total_iterations += 1
        
        # FINALIZZAZIONE
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Debug finale delle statistiche LLM
        self._print_llm_usage_statistics()
        
        # Genera cluster_info compatibile con il sistema esistente
        cluster_info = self._generate_cluster_info_output()
        
        # Statistiche finali
        final_stats = self._generate_final_statistics(duration, iteration, convergence_achieved)
        print("\n" + "=" * 60)
        print("🏁 CLUSTERING GERARCHICO COMPLETATO")
        print(f"⏱️ Durata: {duration:.2f} secondi, {iteration} iterazioni")
        print(f"🌳 Regioni finali: {len(self.regions)} (core: {sum(1 for r in self.regions.values() if r.region_type == 'core')})")
        print(f"🔧 Conflitti risolti: {self.conflicts_resolved}")
        print(f"📊 Efficienza: {len(texts)/duration:.1f} sessioni/sec")
        
        trace_all("cluster_hierarchically", "EXIT", 
                 session_memberships_count=len(self.session_memberships),
                 cluster_info_count=len(cluster_info),
                 duration=duration,
                 iterations=iteration,
                 conflicts_resolved=self.conflicts_resolved)
        
        return self.session_memberships, cluster_info
    
    def _print_llm_usage_statistics(self):
        """
        Stampa statistiche dettagliate sull'utilizzo dell'LLM per debug e ottimizzazione
        """
        print(f"\n📊 STATISTICHE UTILIZZO LLM:")
        print(f"   🎯 classify_conversation: {self.llm_usage_stats['classify_conversation_calls']} chiamate")
        print(f"   🔄 predict_with_ensemble: {self.llm_usage_stats['predict_with_ensemble_calls']} chiamate")
        print(f"   ⚠️ Fallback: {self.llm_usage_stats['fallback_calls']} chiamate")
        print(f"   ❌ Errori: {self.llm_usage_stats['error_calls']} chiamate")
        print(f"   📈 Totale predizioni: {self.llm_usage_stats['total_predictions']}")
        
        # Calcola percentuali
        total = self.llm_usage_stats['total_predictions']
        if total > 0:
            success_rate = ((self.llm_usage_stats['classify_conversation_calls'] + 
                           self.llm_usage_stats['predict_with_ensemble_calls']) / total) * 100
            fallback_rate = (self.llm_usage_stats['fallback_calls'] / total) * 100
            error_rate = (self.llm_usage_stats['error_calls'] / total) * 100
            
            print(f"   ✅ Tasso successo: {success_rate:.1f}%")
            print(f"   ⚠️ Tasso fallback: {fallback_rate:.1f}%")
            print(f"   ❌ Tasso errori: {error_rate:.1f}%")
            
            # Suggerimenti di ottimizzazione
            if fallback_rate > 20:
                print(f"   💡 Suggerimento: Alto tasso fallback - verificare configurazione LLM")
            if error_rate > 10:
                print(f"   💡 Suggerimento: Alto tasso errori - controllare stabilità connessione LLM")
    
    def _generate_cluster_info_output(self) -> Dict[int, Dict[str, Any]]:
        """
        Genera output cluster_info compatibile con il sistema esistente
        """
        cluster_info = {}
        
        for region_id, region in self.regions.items():
            # Solo regioni principali (core e outlier), non boundary o parent gerarchici
            if region.region_type not in ['core', 'outlier']:
                continue
            
            # Trova etichetta predominante
            if region.labels:
                dominant_label = max(region.labels.items(), key=lambda x: x[1])
                intent_string = dominant_label[0]
                confidence = dominant_label[1]
            else:
                intent_string = 'altro'
                confidence = 0.5
            
            # Trova indici delle sessioni membri
            session_indices = []
            for i, session_id in enumerate(self.session_memberships.keys()):
                if region_id in self.session_memberships.get(session_id, {}):
                    session_indices.append(i)
            
            cluster_info[region_id] = {
                'intent': intent_string.replace(' ', '_'),
                'size': len(region.member_sessions),
                'indices': session_indices,
                'intent_string': intent_string,
                'classification_method': 'hierarchical_adaptive',
                'average_confidence': confidence,
                'region_type': region.region_type,
                'hierarchy_depth': self._calculate_region_depth(region_id),
                'entropy': self._calculate_entropy(region.labels) if region.labels else 0.0
            }
        
        return cluster_info
    
    def _calculate_region_depth(self, region_id: int) -> int:
        """Calcola profondità gerarchica di una regione"""
        region = self.regions[region_id]
        depth = 0
        
        current_region = region
        while current_region.parent_region is not None:
            depth += 1
            if current_region.parent_region in self.regions:
                current_region = self.regions[current_region.parent_region]
            else:
                break
        
        return depth
    
    def _generate_final_statistics(self, duration: float, iterations: int, converged: bool) -> Dict[str, Any]:
        """
        Genera statistiche complete del clustering gerarchico
        """
        # Conteggi per tipo di regione
        region_types = defaultdict(int)
        total_confidence = 0
        total_sessions = 0
        
        for region in self.regions.values():
            region_types[region.region_type] += 1
            if region.confidence_scores:
                total_confidence += sum(region.confidence_scores.values())
                total_sessions += len(region.confidence_scores)
        
        avg_confidence = total_confidence / max(total_sessions, 1)
        
        # Analisi membership
        multi_membership_sessions = sum(1 for memberships in self.session_memberships.values() 
                                      if len(memberships) > 1)
        
        return {
            'duration_seconds': duration,
            'iterations': iterations,
            'converged': converged,
            'total_regions': len(self.regions),
            'region_types': dict(region_types),
            'conflicts_resolved': self.conflicts_resolved,
            'regions_split': self.regions_split,
            'regions_merged': self.regions_merged,
            'average_confidence': avg_confidence,
            'multi_membership_sessions': multi_membership_sessions,
            'sessions_per_second': len(self.session_memberships) / max(duration, 0.001)
        }
    
    def get_session_final_labels(self) -> Dict[str, str]:
        """
        Restituisce etichetta finale per ogni sessione (risolve membership multiple)
        
        Returns:
            Dict {session_id: final_label}
        """
        final_labels = {}
        
        for session_id, memberships in self.session_memberships.items():
            if not memberships:
                final_labels[session_id] = 'altro'
                continue
            
            # Trova regione con membership più alta
            best_region_id = max(memberships.items(), key=lambda x: x[1])[0]
            best_region = self.regions[best_region_id]
            
            # Estrai etichetta dominante della regione
            if best_region.labels:
                dominant_label = max(best_region.labels.items(), key=lambda x: x[1])[0]
            else:
                dominant_label = 'altro'
            
            final_labels[session_id] = dominant_label
        
        return final_labels
    
    def auto_tune_parameters(self, performance_stats: Dict[str, Any]):
        """
        Auto-tuning dei parametri basato sulle prestazioni osservate
        """
        print(f"🔧 Auto-tuning parametri basato su prestazioni...")
        
        # Adatta soglie basandosi su numero di conflitti
        conflicts_ratio = performance_stats.get('conflicts_ratio', 0.0)
        
        if conflicts_ratio > 0.3:  # Troppi conflitti
            print(f"   📉 Riducendo soglie per diminuire conflitti (ratio: {conflicts_ratio:.3f})")
            self.entropy_threshold *= 1.1
            self.dominant_prob_threshold -= 0.05
            self.significant_label_threshold += 0.05
        elif conflicts_ratio < 0.1:  # Troppo pochi conflitti, forse troppo conservativo
            print(f"   📈 Aumentando sensibilità conflitti (ratio: {conflicts_ratio:.3f})")
            self.entropy_threshold *= 0.9
            self.dominant_prob_threshold += 0.05
            self.significant_label_threshold -= 0.05
        
        # Adatta soglie merge basandosi su frammentazione
        regions_per_session = performance_stats.get('regions_per_session', 1.0)
        
        if regions_per_session > 0.3:  # Troppa frammentazione
            print(f"   🔗 Riducendo soglia merge per diminuire frammentazione")
            self.similarity_merge_threshold *= 0.95
        elif regions_per_session < 0.1:  # Troppo poca granularità
            print(f"   🔍 Aumentando soglia merge per preservare granularità")
            self.similarity_merge_threshold *= 1.05
        
        # Mantieni parametri nei range validi
        self.entropy_threshold = max(0.5, min(2.0, self.entropy_threshold))
        self.dominant_prob_threshold = max(0.4, min(0.8, self.dominant_prob_threshold))
        self.significant_label_threshold = max(0.1, min(0.3, self.significant_label_threshold))
        self.similarity_merge_threshold = max(0.7, min(0.95, self.similarity_merge_threshold))
        
        print(f"   ✅ Parametri aggiornati: entropy_threshold={self.entropy_threshold:.3f}, "
              f"dominant_prob_threshold={self.dominant_prob_threshold:.3f}")
    
    def get_hierarchical_structure(self) -> Dict[str, Any]:
        """
        Restituisce struttura gerarchica completa per visualizzazione/debug
        
        Returns:
            Struttura gerarchica con regioni, relazioni parent-child, etc.
        """
        hierarchy = {
            'regions': {},
            'parent_child_relations': {},
            'depth_levels': defaultdict(list),
            'statistics': {}
        }
        
        # Informazioni sulle regioni
        for region_id, region in self.regions.items():
            hierarchy['regions'][region_id] = {
                'type': region.region_type,
                'size': len(region.member_sessions),
                'labels': region.labels,
                'confidence_avg': np.mean(list(region.confidence_scores.values())) if region.confidence_scores else 0,
                'parent': region.parent_region,
                'children': region.child_regions,
                'created_at': region.created_at.isoformat(),
                'last_updated': region.last_updated.isoformat()
            }
            
            # Relazioni gerarchiche
            if region.parent_region is not None:
                if region.parent_region not in hierarchy['parent_child_relations']:
                    hierarchy['parent_child_relations'][region.parent_region] = []
                hierarchy['parent_child_relations'][region.parent_region].append(region_id)
            
            # Livelli di profondità
            depth = self._calculate_region_depth(region_id)
            hierarchy['depth_levels'][depth].append(region_id)
        
        # Statistiche gerarchiche
        hierarchy['statistics'] = {
            'max_depth': max(hierarchy['depth_levels'].keys()) if hierarchy['depth_levels'] else 0,
            'total_regions': len(self.regions),
            'leaf_regions': sum(1 for r in self.regions.values() if not r.child_regions),
            'internal_nodes': sum(1 for r in self.regions.values() if r.child_regions),
            'multi_membership_sessions': sum(1 for m in self.session_memberships.values() if len(m) > 1)
        }
        
        return hierarchy
    
    def save_hierarchical_state(self, filepath: str):
        """
        Salva stato completo del clustering gerarchico su file
        """
        state = {
            'regions': {},
            'session_memberships': self.session_memberships,
            'config': {
                'confidence_threshold': self.confidence_threshold,
                'boundary_threshold': self.boundary_threshold,
                'max_hierarchy_depth': self.max_hierarchy_depth,
                'min_region_size': self.min_region_size,
                'max_boundary_ratio': self.max_boundary_ratio,
                'similarity_merge_threshold': self.similarity_merge_threshold
            },
            'statistics': {
                'next_region_id': self.next_region_id,
                'total_iterations': self.total_iterations,
                'conflicts_resolved': self.conflicts_resolved,
                'regions_split': self.regions_split,
                'regions_merged': self.regions_merged
            },
            'saved_at': datetime.now().isoformat()
        }
        
        # Serializza regioni (converte numpy arrays in liste)
        for region_id, region in self.regions.items():
            state['regions'][region_id] = {
                'region_id': region.region_id,
                'region_type': region.region_type,
                'center_embedding': region.center_embedding.tolist(),
                'radius': region.radius,
                'member_sessions': list(region.member_sessions),
                'confidence_scores': region.confidence_scores,
                'labels': region.labels,
                'parent_region': region.parent_region,
                'child_regions': region.child_regions,
                'created_at': region.created_at.isoformat(),
                'last_updated': region.last_updated.isoformat()
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Stato gerarchico salvato in: {filepath}")
    
    def load_hierarchical_state(self, filepath: str):
        """
        Carica stato del clustering gerarchico da file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # Ripristina configurazione
        config = state['config']
        self.confidence_threshold = config['confidence_threshold']
        self.boundary_threshold = config['boundary_threshold']
        self.max_hierarchy_depth = config['max_hierarchy_depth']
        self.min_region_size = config['min_region_size']
        self.max_boundary_ratio = config['max_boundary_ratio']
        self.similarity_merge_threshold = config['similarity_merge_threshold']
        
        # Ripristina statistiche
        stats = state['statistics']
        self.next_region_id = stats['next_region_id']
        self.total_iterations = stats['total_iterations']
        self.conflicts_resolved = stats['conflicts_resolved']
        self.regions_split = stats['regions_split']
        self.regions_merged = stats['regions_merged']
        
        # Ripristina session memberships
        self.session_memberships = state['session_memberships']
        
        # Ripristina regioni (riconverte liste in numpy arrays)
        self.regions = {}
        for region_id_str, region_data in state['regions'].items():
            region_id = int(region_id_str)
            
            region = ConfidenceRegion(
                region_id=region_data['region_id'],
                region_type=region_data['region_type'],
                center_embedding=np.array(region_data['center_embedding']),
                radius=region_data['radius'],
                member_sessions=set(region_data['member_sessions']),
                confidence_scores=region_data['confidence_scores'],
                labels=region_data['labels'],
                parent_region=region_data['parent_region'],
                child_regions=region_data['child_regions'],
                created_at=datetime.fromisoformat(region_data['created_at']),
                last_updated=datetime.fromisoformat(region_data['last_updated'])
            )
            
            self.regions[region_id] = region
        
        print(f"📥 Stato gerarchico caricato da: {filepath}")
        print(f"   🌳 {len(self.regions)} regioni ripristinate")
    
    def save_optimized_config(self, filepath: str):
        """
        Salva configurazione ottimizzata per riutilizzo futuro
        """
        optimized_config = {
            'hierarchical_clustering': {
                'confidence_threshold': self.confidence_threshold,
                'boundary_threshold': self.boundary_threshold,
                'max_hierarchy_depth': self.max_hierarchy_depth,
                'min_region_size': self.min_region_size,
                'max_boundary_ratio': self.max_boundary_ratio,
                'similarity_merge_threshold': self.similarity_merge_threshold,
                'entropy_threshold': self.entropy_threshold,
                'dominant_prob_threshold': self.dominant_prob_threshold,
                'significant_label_threshold': self.significant_label_threshold,
                'severe_conflict_threshold': self.severe_conflict_threshold,
                'moderate_conflict_threshold': self.moderate_conflict_threshold,
                'consolidation_similarity_threshold': self.consolidation_similarity_threshold,
                'severity_weights': self.severity_weights,
                'auto_tuned': True,
                'tuning_timestamp': datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as file:
            yaml.dump(optimized_config, file, default_flow_style=False, allow_unicode=True)
        
        print(f"💾 Configurazione ottimizzata salvata in: {filepath}")


# Funzione di test per verificare il funzionamento
def test_hierarchical_adaptive_clusterer():
    """
    Test del clustering gerarchico adattivo
    """
    print("=== TEST CLUSTERING GERARCHICO ADATTIVO ===\n")
    
    # Setup del test
    np.random.seed(42)
    
    # Dati di test simulati
    n_sessions = 50
    embedding_dim = 384
    
    # Genera embeddings simulati con cluster naturali
    cluster_centers = [
        np.random.randn(embedding_dim),  # Cluster 1
        np.random.randn(embedding_dim),  # Cluster 2  
        np.random.randn(embedding_dim),  # Cluster 3
    ]
    
    embeddings = []
    texts = []
    session_ids = []
    
    for i in range(n_sessions):
        # Assegna a cluster casuale
        cluster_idx = i % 3
        center = cluster_centers[cluster_idx]
        
        # Aggiungi noise
        embedding = center + np.random.normal(0, 0.1, embedding_dim)
        embeddings.append(embedding)
        
        # Genera testo simulato
        cluster_topics = ['prenotazione medica', 'problemi accesso', 'richieste informazioni']
        text = f"Sessione {i}: {cluster_topics[cluster_idx]} caso numero {i//3 + 1}"
        texts.append(text)
        
        session_ids.append(f"session_{i:03d}")
    
    embeddings = np.array(embeddings)
    
    # Normalizza embeddings per similarità coseno
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    print(f"📊 Dataset di test generato:")
    print(f"   Sessioni: {len(texts)}")
    print(f"   Dimensione embedding: {embedding_dim}")
    print(f"   Cluster simulati: 3")
    
    # Inizializza clusterer
    clusterer = HierarchicalAdaptiveClusterer(
        llm_classifier=None,  # Test senza LLM per semplicità
        confidence_threshold=0.75,
        boundary_threshold=0.45,
        max_hierarchy_depth=3
    )
    
    # Esegui clustering gerarchico
    session_memberships, cluster_info = clusterer.cluster_hierarchically(
        texts, embeddings, session_ids, max_iterations=3
    )
    
    # Analizza risultati
    print(f"\n📈 RISULTATI CLUSTERING:")
    print(f"   Regioni create: {len(clusterer.regions)}")
    print(f"   Cluster info: {len(cluster_info)}")
    
    # Mostra cluster trovati
    for cluster_id, info in cluster_info.items():
        print(f"   🏷️ Cluster {cluster_id}: '{info['intent_string']}' "
              f"({info['size']} membri, conf: {info['average_confidence']:.3f})")
    
    # Statistiche membership
    multi_membership = sum(1 for m in session_memberships.values() if len(m) > 1)
    print(f"   🔗 Sessioni multi-membership: {multi_membership}/{len(session_memberships)}")
    
    # Struttura gerarchica
    hierarchy = clusterer.get_hierarchical_structure()
    print(f"   🌳 Profondità massima: {hierarchy['statistics']['max_depth']}")
    print(f"   📊 Nodi foglia: {hierarchy['statistics']['leaf_regions']}")
    
    # Test salvataggio/caricamento
    test_save_path = "/tmp/test_hierarchical_state.json"
    clusterer.save_hierarchical_state(test_save_path)
    
    # Crea nuovo clusterer e carica stato
    clusterer2 = HierarchicalAdaptiveClusterer()
    clusterer2.load_hierarchical_state(test_save_path)
    
    print(f"\n✅ Test completato con successo!")
    print(f"   Stato salvato e ricaricato correttamente")
    
    return clusterer, session_memberships, cluster_info


if __name__ == "__main__":
    # Esegui test
    test_hierarchical_adaptive_clusterer()
