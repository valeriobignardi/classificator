#!/usr/bin/env python3
"""
Clustering Intelligente basato su LLM + ML senza pattern predefiniti
Approccio completamente data-driven che usa l'intelligenza artificiale
per identificare automaticamente gli intent dalle conversazioni
"""

import numpy as np
import yaml
import os
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, Counter
# 🔧 [REMOVED] from sklearn.cluster import HDBSCAN  # Ora usiamo HDBSCANClusterer custom
from sklearn.metrics import silhouette_score
import logging
import sys

# Import Tenant per principio universale
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))
from tenant import Tenant

# Import trace_all per debugging
try:
    from Pipeline.end_to_end_pipeline import trace_all
except ImportError:
    # Fallback se non disponibile
    def trace_all(function_name: str, action: str = "ENTER", called_from: str = None, **kwargs):
        pass

# 🔧 [FIX] Import HDBSCANClusterer all'inizio per evitare problemi di contesto
try:
    from .hdbscan_clusterer import HDBSCANClusterer
except ImportError:
    # Fallback per contesti dove import relativo non funziona
    import sys
    sys.path.append(os.path.dirname(__file__))
    from hdbscan_clusterer import HDBSCANClusterer

class IntelligentIntentClusterer:
    """
    Clusterer intelligente che usa LLM per identificare intent automaticamente
    senza bisogno di pattern predefiniti. Approccio completamente ML-driven.
    """
    
    def __init__(self, tenant: Optional[Tenant] = None, config_path: str = None, llm_classifier=None, ensemble_classifier=None):
        """
        Inizializza il clusterer intelligente
        
        PRINCIPIO UNIVERSALE: Accetta oggetto Tenant completo
        
        CORREZIONE 2025-09-04: Aggiunto ensemble_classifier per usare ML+LLM negli outlier
        
        Args:
            tenant: Oggetto Tenant completo (None per compatibilità)
            config_path: Percorso del file di configurazione globale
            llm_classifier: Classificatore LLM per analisi intent (legacy)
            ensemble_classifier: Classificatore ensemble ML+LLM (priorità maggiore)
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        self.tenant = tenant
        self.tenant_id = tenant.tenant_id if tenant else None  # Estrae tenant_id dall'oggetto
        self.config_path = config_path
        
        # 🔧 CORREZIONE: Priorità all'ensemble_classifier se disponibile
        self.ensemble_classifier = ensemble_classifier
        self.llm_classifier = llm_classifier
        
        # Determina quale classificatore usare
        if self.ensemble_classifier is not None:
            self.use_ensemble = True
            print(f"✅ [INTELLIGENT_CLUSTERER] Usando ensemble_classifier (ML+LLM)")
        elif self.llm_classifier is not None:
            self.use_ensemble = False
            print(f"⚠️ [INTELLIGENT_CLUSTERER] Usando solo llm_classifier (fallback)")
        else:
            self.use_ensemble = False
            print(f"❌ [INTELLIGENT_CLUSTERER] Nessun classificatore disponibile")
        self.load_intelligent_config()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_intelligent_config(self):
        """Carica la configurazione per clustering intelligente"""
        try:
            # 🗂️  [NEW] Carica prima config tenant-specific se disponibile
            tenant_clustering_config = {}
            if self.tenant_id:
                tenant_config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tenant_configs')
                tenant_config_file = os.path.join(tenant_config_dir, f'{self.tenant_id}_clustering.yaml')
                
                if os.path.exists(tenant_config_file):
                    print(f"🗂️  [DEBUG] Caricamento config tenant-specific: {tenant_config_file}")
                    with open(tenant_config_file, 'r', encoding='utf-8') as file:
                        tenant_config = yaml.safe_load(file)
                        tenant_clustering_config = tenant_config.get('clustering_parameters', {})
                        print(f"🗂️  [DEBUG] Config tenant caricato: {list(tenant_clustering_config.keys())}")
                else:
                    print(f"🗂️  [DEBUG] Config tenant non trovato: {tenant_config_file}")
            
            # Carica config globale
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)

            intelligent_config = config.get('intelligent_clustering', {})
            self.enabled = intelligent_config.get('enabled', True)
            self.use_llm_primary = intelligent_config.get('use_llm_primary', True)
            self.use_patterns_fallback = intelligent_config.get('use_patterns_fallback', False)
            self.llm_confidence_threshold = intelligent_config.get('llm_confidence_threshold', 0.75)
            self.min_conversations_per_intent = intelligent_config.get('min_conversations_per_intent', 2)
            self.min_average_confidence = intelligent_config.get('min_average_confidence', 0.4)
            
            # Categorie di intent che LLM può usare (come suggerimenti, non limitazioni)
            self.intent_categories = intelligent_config.get('intent_categories', [
                'accesso_problemi', 'prenotazione_esami', 'info_esami_specifici',
                'ritiro_referti', 'richiesta_operatore', 'fatturazione_problemi',
                'modifica_appuntamenti', 'verifica_appuntamenti', 'orari_contatti', 
                'contatti_info', 'indicazioni_sede', 'servizi_supporto',
                'problemi_tecnici', 'info_generali', 'cortesia', 'altro'
            ])
            
            # 🗂️  [NEW] Configurazione clustering: prima tenant-specific, poi globale come fallback
            clustering_config = config.get('clustering', {})
            
            # Usa config tenant se disponibile, altrimenti fallback a globale
            self.min_cluster_size = tenant_clustering_config.get('min_cluster_size', clustering_config.get('min_cluster_size', 2))
            self.min_samples = tenant_clustering_config.get('min_samples', clustering_config.get('min_samples', 1))
            self.cluster_selection_epsilon = tenant_clustering_config.get('cluster_selection_epsilon', clustering_config.get('cluster_selection_epsilon', 0.03))
            self.metric = tenant_clustering_config.get('metric', clustering_config.get('metric', 'euclidean'))
            
            # 🗂️  [NEW] Parametri UMAP per supporto completo
            self.use_umap = tenant_clustering_config.get('use_umap', clustering_config.get('use_umap', True))
            self.umap_n_neighbors = tenant_clustering_config.get('umap_n_neighbors', clustering_config.get('umap_n_neighbors', 30))
            self.umap_n_components = tenant_clustering_config.get('umap_n_components', clustering_config.get('umap_n_components', 52))
            self.umap_min_dist = tenant_clustering_config.get('umap_min_dist', clustering_config.get('umap_min_dist', 0.2))
            self.umap_metric = tenant_clustering_config.get('umap_metric', clustering_config.get('umap_metric', 'cosine'))
            
            print(f"🗂️  [DEBUG] UMAP config finale: use_umap={self.use_umap}, n_components={self.umap_n_components}, min_dist={self.umap_min_dist}")
            
            # Parametri per selezione rappresentanti diversificati
            diverse_config = intelligent_config.get('diverse_representatives', {})
            self.diverse_representatives_enabled = diverse_config.get('enabled', True)
            self.n_representatives = diverse_config.get('max_representatives', 5)
            self.min_representatives = diverse_config.get('min_representatives', 2)
            self.consensus_strategy = diverse_config.get('consensus_strategy', 'majority')
            self.majority_threshold = diverse_config.get('majority_threshold', 0.6)
            self.consensus_confidence_bonus = diverse_config.get('consensus_confidence_bonus', 0.1)
            
            print(f"🧠 Clustering intelligente configurato:")
            print(f"   🎯 Categorie intent suggerite: {len(self.intent_categories)}")
            print(f"   🤖 LLM primario: {self.use_llm_primary}")
            print(f"   📊 Min confidence: {self.llm_confidence_threshold}")
            print(f"   📈 Min conversations per intent: {self.min_conversations_per_intent}")
            print(f"   👥 Rappresentanti diversificati: {self.diverse_representatives_enabled}")
            print(f"   🔢 Max rappresentanti per cluster: {self.n_representatives}")
            print(f"   🤝 Strategia consenso: {self.consensus_strategy}")
            
        except Exception as e:
            self.logger.error(f"Errore caricamento configurazione intelligente: {e}")
            self._load_default_intelligent_config()
    
    def _load_default_intelligent_config(self):
        """Configurazione di default se il config non è disponibile"""
        self.enabled = True
        self.use_llm_primary = True
        self.use_patterns_fallback = False
        self.llm_confidence_threshold = 0.75
        self.min_conversations_per_intent = 2
        self.min_average_confidence = 0.4
        self.intent_categories = [
            'accesso_problemi', 'prenotazione_esami', 'ritiro_referti',
            'modifica_appuntamenti', 'orari_contatti', 'info_generali', 'altro'
        ]
        
        # Parametri rappresentanti diversificati - default
        self.diverse_representatives_enabled = True
        self.n_representatives = 5
        self.min_representatives = 2
        self.consensus_strategy = 'majority'
        self.majority_threshold = 0.6
        self.consensus_confidence_bonus = 0.1
        
        # Parametri clustering default
        self.min_cluster_size = 2
        self.min_samples = 1
        self.cluster_selection_epsilon = 0.03
        self.metric = 'euclidean'
        self.n_representatives = 3
        self.max_representatives = 5
        print("⚠️ Usata configurazione predefinita per clustering intelligente")
    
    def extract_intents_with_llm(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Estrae intent da ogni testo usando LLM in modo completamente automatico
        
        Scopo della funzione:
        - Estrae intent da ogni testo usando ensemble classifier o LLM
        - Usa il metodo batch per efficienza massima invece di loop manuale
        
        Parametri di input e output:
        - texts: List[str] - Lista di testi da analizzare
        
        Valori di ritorno:
        - List[Dict[str, Any]]: Lista di dizionari con intent, confidence e reasoning per ogni testo
        
        Tracciamento aggiornamenti:
        - 2025-09-07: Sostituito loop manuale con classify_batch per efficienza
        
        Args:
            texts: Lista di testi da analizzare
            
        Returns:
            Lista di dizionari con intent, confidence e reasoning per ogni testo
        """
        results = []
        
        print(f"🤖 Analisi LLM di {len(texts)} conversazioni...")
        
        # 🚀 OTTIMIZZAZIONE CRITICA: Usa classify_batch invece di loop manuale
        if self.use_ensemble and self.ensemble_classifier:
            print(f"🧠💻 [ENSEMBLE BATCH] Classificazione di {len(texts)} conversazioni con ensemble...")
            
            # L'ensemble classifier potrebbe non supportare batch, quindi usiamo loop ottimizzato
            for i, text in enumerate(texts):
                try:
                    # Usa ensemble ML+LLM per classificazione completa
                    ensemble_result = self.ensemble_classifier.predict_with_ensemble(
                        text,
                        return_details=True,
                        embedder=getattr(self.ensemble_classifier, 'embedder', None)
                    )
                    
                    intent_info = {
                        'intent': ensemble_result['predicted_label'],
                        'confidence': ensemble_result['confidence'],
                        'reasoning': f"Ensemble {ensemble_result['method']} - conf: {ensemble_result['confidence']:.3f}",
                        'method': f"ensemble_{ensemble_result['method'].lower()}",
                        'text_preview': text[:100] + '...' if len(text) > 100 else text,
                        'ensemble_details': ensemble_result  # 🆕 Salva dettagli completi ensemble
                    }
                    
                    results.append(intent_info)
                    
                    if (i + 1) % 50 == 0 or i == len(texts) - 1:
                        print(f"   📈 Progresso Ensemble: {i+1}/{len(texts)}")
                    
                except Exception as e:
                    self.logger.error(f"Errore analisi Ensemble per testo {i}: {e}")
                    
                    # Fallback per errori
                    results.append({
                        'intent': 'altro',
                        'confidence': 0.05,
                        'reasoning': f'Errore Ensemble: {e}',
                        'method': 'ensemble_error_fallback',
                        'text_preview': text[:100] + '...' if len(text) > 100 else text
                    })
                    
        elif self.llm_classifier and self.llm_classifier.is_available():
            print(f"🤖 [LLM BATCH] Classificazione di {len(texts)} conversazioni con LLM batch...")
            
            # 🚀 USA CLASSIFY_BATCH - MOLTO PIÙ EFFICIENTE!
            try:
                batch_results = self.llm_classifier.classify_batch(texts, show_progress=True)
                
                # Converte i ClassificationResult in formato compatibile
                for i, llm_result in enumerate(batch_results):
                    intent_info = {
                        'intent': llm_result.predicted_label,
                        'confidence': llm_result.confidence,
                        'reasoning': llm_result.motivation,
                        'method': f'llm_batch_{llm_result.method.lower()}',
                        'text_preview': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                        'processing_time': llm_result.processing_time
                    }
                    results.append(intent_info)
                
                print(f"✅ [LLM BATCH] Completata classificazione batch di {len(results)} conversazioni")
                
            except Exception as e:
                self.logger.error(f"Errore classificazione batch LLM: {e}")
                print(f"❌ [LLM BATCH] Errore batch, fallback a loop manuale: {e}")
                
                # Fallback a loop manuale in caso di errore batch
                for i, text in enumerate(texts):
                    try:
                        llm_result = self.llm_classifier.classify_with_motivation(text)
                        
                        intent_info = {
                            'intent': llm_result.predicted_label,
                            'confidence': llm_result.confidence,
                            'reasoning': llm_result.motivation,
                            'method': 'llm_fallback_individual',
                            'text_preview': text[:100] + '...' if len(text) > 100 else text
                        }
                        results.append(intent_info)
                        
                        if (i + 1) % 50 == 0 or i == len(texts) - 1:
                            print(f"   📈 Progresso LLM Fallback: {i+1}/{len(texts)}")
                            
                    except Exception as inner_e:
                        self.logger.error(f"Errore analisi LLM fallback per testo {i}: {inner_e}")
                        
                        results.append({
                            'intent': 'altro',
                            'confidence': 0.05,
                            'reasoning': f'Errore LLM fallback: {inner_e}',
                            'method': 'llm_error_fallback',
                            'text_preview': text[:100] + '...' if len(text) > 100 else text
                        })
        else:
            # Fallback se nessun classificatore disponibile
            print(f"⚠️ [FALLBACK] Nessun classificatore disponibile per {len(texts)} conversazioni")
            for i, text in enumerate(texts):
                results.append({
                    'intent': 'altro',
                    'confidence': 0.1,
                    'reasoning': 'Nessun classificatore disponibile',
                    'method': 'no_classifier_fallback',
                    'text_preview': text[:100] + '...' if len(text) > 100 else text
                })
        
        print(f"🏁 [EXTRACT_INTENTS] Completata analisi di {len(results)} conversazioni")
        return results
    
    def group_by_intelligent_intents(self, texts: List[str], llm_results: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Raggruppa le conversazioni per intent identificati dall'LLM
        
        Args:
            texts: Lista di testi originali
            llm_results: Risultati dell'analisi LLM
            
        Returns:
            Dizionario intent -> lista di indici delle conversazioni
        """
        intent_groups = defaultdict(list)
        
        print(f"📊 Raggruppamento per intent identificati dall'LLM...")
        
        for i, result in enumerate(llm_results):
            intent = result['intent']
            confidence = result['confidence']
            
            # Filtra per confidenza minima
            if confidence >= self.min_average_confidence:
                intent_groups[intent].append(i)
            else:
                # Bassa confidenza -> categorizza come "altro"
                intent_groups['altro'].append(i)
        
        # Filtra gruppi troppo piccoli
        filtered_groups = {}
        for intent, indices in intent_groups.items():
            if len(indices) >= self.min_conversations_per_intent:
                filtered_groups[intent] = indices
            else:
                # Groupe troppo piccoli -> sposta in "altro"
                if 'altro' not in filtered_groups:
                    filtered_groups['altro'] = []
                filtered_groups['altro'].extend(indices)
        
        # Statistiche
        print(f"📊 Gruppi di intent trovati:")
        for intent, indices in sorted(filtered_groups.items(), key=lambda x: len(x[1]), reverse=True):
            avg_confidence = np.mean([llm_results[i]['confidence'] for i in indices])
            print(f"  ✅ {intent}: {len(indices)} conv. (conf. media: {avg_confidence:.3f})")
        
        return filtered_groups
    
    def semantic_sub_clustering(self, intent_groups: Dict[str, List[int]], 
                               embeddings: np.ndarray, texts: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Applica sub-clustering semantico per intent con molte conversazioni
        
        Args:
            intent_groups: Gruppi di intent
            embeddings: Embeddings delle conversazioni
            texts: Testi originali
            
        Returns:
            Dizionario cluster_id -> info cluster
        """
        cluster_info = {}
        cluster_id = 0
        
        print(f"🔍 Applicazione sub-clustering semantico...")
        
        for intent, indices in intent_groups.items():
            if len(indices) >= 10:  # Solo per gruppi grandi
                print(f"  🔍 Sub-clustering per intent '{intent}' ({len(indices)} conv.)")
                
                # Estrai embeddings per questo intent
                intent_embeddings = embeddings[indices]
                
                # Applica HDBSCAN con supporto UMAP (import già fatto all'inizio)
                
                clusterer = HDBSCANClusterer(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    cluster_selection_epsilon=self.cluster_selection_epsilon,
                    metric=self.metric,
                    # 🗂️  Parametri UMAP dal config
                    use_umap=getattr(self, 'use_umap', True),
                    umap_n_neighbors=getattr(self, 'umap_n_neighbors', 30),
                    umap_n_components=getattr(self, 'umap_n_components', 52),
                    umap_min_dist=getattr(self, 'umap_min_dist', 0.2),
                    umap_metric=getattr(self, 'umap_metric', 'cosine')
                )
                
                sub_labels = clusterer.fit_predict(intent_embeddings)
                
                # Processa i sub-cluster
                unique_sub_labels = set(sub_labels)
                for sub_label in unique_sub_labels:
                    if sub_label != -1:  # Escludi outliers
                        # Trova indici originali per questo sub-cluster
                        sub_indices = [indices[i] for i, l in enumerate(sub_labels) if l == sub_label]
                        
                        if len(sub_indices) >= self.min_conversations_per_intent:
                            cluster_info[cluster_id] = {
                                'intent': intent,
                                'sub_cluster': sub_label,
                                'size': len(sub_indices),
                                'indices': sub_indices,
                                'intent_string': f'{intent}_sub_{sub_label}',
                                'classification_method': 'llm_semantic_sub'
                            }
                            cluster_id += 1
                
                # Gestisci outliers del sub-clustering
                outlier_indices = [indices[i] for i, l in enumerate(sub_labels) if l == -1]
                if outlier_indices:
                    cluster_info[cluster_id] = {
                        'intent': intent,
                        'sub_cluster': -1,
                        'size': len(outlier_indices),
                        'indices': outlier_indices,
                        'intent_string': f'{intent}_mixed',
                        'classification_method': 'llm_semantic_main'
                    }
                    cluster_id += 1
            
            else:
                # Gruppi piccoli -> un cluster unico
                cluster_info[cluster_id] = {
                    'intent': intent,
                    'sub_cluster': 0,
                    'size': len(indices),
                    'indices': indices,
                    'intent_string': intent,
                    'classification_method': 'llm_intelligent'
                }
                cluster_id += 1
        
        return cluster_info
    
    def cluster_intelligently(self, texts: List[str], embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
        """
        Clustering intelligente OTTIMIZZATO: HDBSCAN prima, LLM sui rappresentanti dopo
        
        Args:
            texts: Lista di testi da clusterizzare
            embeddings: Embeddings delle conversazioni
            
        Returns:
            Tuple (cluster_labels, cluster_info)
        """
        # 🔍 TRACING ENTER
        trace_all(
            'cluster_intelligently', 
            'ENTER',
            called_from="IntelligentIntentClusterer",
            texts_count=len(texts),
            embeddings_shape=embeddings.shape if embeddings is not None else None,
            tenant_id=self.tenant_id if hasattr(self, 'tenant_id') and self.tenant_id else None,
            use_ensemble=getattr(self, 'use_ensemble', False),
            clustering_config={
                'min_cluster_size': getattr(self, 'min_cluster_size', None),
                'min_samples': getattr(self, 'min_samples', None),
                'use_umap': getattr(self, 'use_umap', None)
            }
        )
        
        print(f"🧠 Avvio clustering intelligente OTTIMIZZATO di {len(texts)} conversazioni...")
        
        # 🆕 Setup debug logging per suggested_labels
        debug_logger = logging.getLogger('suggested_labels_debug')
        debug_logger.setLevel(logging.DEBUG)
        
        # Rimuovi handler esistenti per evitare duplicazioni
        for handler in debug_logger.handlers[:]:
            debug_logger.removeHandler(handler)
        
        # Setup file handler per suggested_labels.log
        log_file_path = '/home/ubuntu/classificatore/suggested_labels.log'
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # Formato del log
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        debug_logger.addHandler(file_handler)
        debug_logger.addHandler(console_handler)
        
        debug_logger.info("=" * 80)
        debug_logger.info("🚀 INIZIO CLUSTERING INTELLIGENTE - ANALISI RAPPRESENTANTI E OUTLIER")
        debug_logger.info(f"📊 Dataset: {len(texts)} conversazioni")
        debug_logger.info("=" * 80)
        
        # FASE 1: Clustering semantico con HDBSCAN (parametri più restrittivi)
        print(f"📊 Fase 1: Clustering semantico con HDBSCAN...")
        
        # 🔍 TRACING FASE 1
        trace_all(
            'cluster_intelligently', 
            'PHASE_1_HDBSCAN_START',
            called_from="cluster_intelligently",
            embeddings_count=len(embeddings),
            min_cluster_size=getattr(self, 'min_cluster_size', None),
            min_samples=getattr(self, 'min_samples', None),
            use_umap=getattr(self, 'use_umap', True)
        )
        
        # 🔧 [FIX] Usa HDBSCANClusterer con supporto UMAP (import già fatto all'inizio)
        
        # Crea clusterer con supporto UMAP
        clusterer = HDBSCANClusterer(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            # 🗂️  Parametri UMAP dal config
            use_umap=getattr(self, 'use_umap', True),
            umap_n_neighbors=getattr(self, 'umap_n_neighbors', 30),
            umap_n_components=getattr(self, 'umap_n_components', 52),
            umap_min_dist=getattr(self, 'umap_min_dist', 0.2),
            umap_metric=getattr(self, 'umap_metric', 'cosine'),
            # 🔧 FIX: Passa oggetto tenant al clusterer
            tenant=self.tenant
        )
        
        # Controllo: HDBSCAN richiede almeno 2 campioni
        if len(embeddings) < 2:
            print(f"   ⚠️ Solo {len(embeddings)} campione/i disponibile/i, salto clustering HDBSCAN")
            # Assegna tutti i campioni al cluster 0
            cluster_labels = np.array([0] * len(embeddings))
            n_clusters = 1 if len(embeddings) > 0 else 0
            n_outliers = 0
        else:
            cluster_labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_outliers = sum(1 for label in cluster_labels if label == -1)
        
        print(f"   📈 Cluster HDBSCAN trovati: {n_clusters}")
        print(f"   🔍 Outliers: {n_outliers}")
        
        # 🔍 TRACING FASE 1 RESULTS
        trace_all(
            'cluster_intelligently', 
            'PHASE_1_HDBSCAN_COMPLETE',
            called_from="cluster_intelligently",
            n_clusters=n_clusters,
            n_outliers=n_outliers,
            total_samples=len(embeddings),
            outlier_ratio=n_outliers / len(embeddings) if len(embeddings) > 0 else 0.0,
            clustering_successful=n_clusters > 0
        )
        
        # FASE 2: Selezione rappresentanti per ogni cluster
        print(f"👥 Fase 2: Selezione rappresentanti per cluster...")
        
        # 🔍 TRACING FASE 2
        trace_all(
            'cluster_intelligently', 
            'PHASE_2_REPRESENTATIVES_START',
            called_from="cluster_intelligently",
            clusters_to_process=n_clusters,
            min_conversations_per_intent=getattr(self, 'min_conversations_per_intent', None),
            diverse_representatives_enabled=getattr(self, 'diverse_representatives_enabled', None)
        )
        cluster_info = {}
        representative_texts = []
        cluster_to_rep_idx = {}
        
        # Per ogni cluster, seleziona rappresentanti diversificati (centro + periferia + diversità)
        for cluster_id in set(cluster_labels):
            if cluster_id >= 0:  # Escludi outliers
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_indices) >= self.min_conversations_per_intent:
                    cluster_embeddings = embeddings[cluster_indices]
                    
                    # Selezione diversificata di rappresentanti
                    selected_reps = self._select_diverse_representatives(
                        cluster_embeddings, cluster_indices, texts
                    )
                    
                    # Aggiungi tutti i rappresentanti alla lista
                    rep_start_idx = len(representative_texts)
                    for rep_idx in selected_reps:
                        representative_texts.append(texts[rep_idx])
                    
                    # Memorizza range di rappresentanti per questo cluster
                    cluster_to_rep_idx[cluster_id] = {
                        'start': rep_start_idx,
                        'count': len(selected_reps),
                        'indices': selected_reps
                    }
                    
                    print(f"   � Cluster {cluster_id}: {len(cluster_indices)} sessioni, {len(selected_reps)} rappresentanti selezionati")
        
        # 🔍 TRACING FASE 2 COMPLETE
        trace_all(
            'cluster_intelligently', 
            'PHASE_2_REPRESENTATIVES_COMPLETE',
            called_from="cluster_intelligently",
            total_representatives=len(representative_texts),
            clusters_with_representatives=len(cluster_to_rep_idx),
            avg_representatives_per_cluster=len(representative_texts) / len(cluster_to_rep_idx) if cluster_to_rep_idx else 0.0,
            cluster_to_rep_mapping=dict(cluster_to_rep_idx)
        )
        
        # FASE 3: Analisi LLM SOLO sui rappresentanti (molto più veloce!)
        print(f"🤖 Fase 3: Analisi LLM di {len(representative_texts)} rappresentanti...")
        debug_logger.info(f"🤖 FASE 3: Analisi LLM di {len(representative_texts)} rappresentanti")
        
        # 🔍 TRACING FASE 3
        trace_all(
            'cluster_intelligently', 
            'PHASE_3_LLM_ANALYSIS_START',
            called_from="cluster_intelligently",
            representative_texts_count=len(representative_texts),
            use_ensemble=getattr(self, 'use_ensemble', False),
            llm_classifier_available=self.llm_classifier is not None,
            ensemble_classifier_available=self.ensemble_classifier is not None
        )
        
        if representative_texts:
            llm_results = self.extract_intents_with_llm(representative_texts)
            
            # 🆕 DEBUG: Log dettagliato per ogni rappresentante
            debug_logger.info("📋 RISULTATI CLASSIFICAZIONE RAPPRESENTANTI:")
            debug_logger.info("-" * 60)
            
            # Mappa risultati ai cluster per debug
            result_idx = 0
            for cluster_id, rep_info in cluster_to_rep_idx.items():
                debug_logger.info(f"🏷️  CLUSTER {cluster_id}:")
                cluster_results = llm_results[rep_info['start']:rep_info['start'] + rep_info['count']]
                
                for i, (rep_idx, result) in enumerate(zip(rep_info['indices'], cluster_results)):
                    debug_logger.info(f"Caso: {rep_idx} - Rappresentante del Cluster n° {cluster_id}: {result['intent']}")
                    print(f"   🔍 Caso {rep_idx} (Cluster {cluster_id}): '{result['intent']}' (conf: {result['confidence']:.3f})")
                    result_idx += 1
                debug_logger.info("")
        else:
            llm_results = []
        
        # 🔍 TRACING FASE 3 COMPLETE
        trace_all(
            'cluster_intelligently', 
            'PHASE_3_LLM_ANALYSIS_COMPLETE',
            called_from="cluster_intelligently",
            llm_results_count=len(llm_results),
            llm_analysis_successful=len(llm_results) > 0,
            clusters_with_results=len([cid for cid, rep_info in cluster_to_rep_idx.items() if rep_info['start'] < len(llm_results)]),
            llm_results_preview=[{
                'intent': r.get('intent', 'unknown'),
                'confidence': r.get('confidence', 0.0)
            } for r in llm_results[:3]]  # Prime 3 per debug
        )
        
        # FASE 4: Creazione cluster_info con etichette da consenso ensemble
        print(f"🔄 Fase 4: Analisi consenso e propagazione etichette...")
        
        # 🔍 TRACING FASE 4
        trace_all(
            'cluster_intelligently', 
            'PHASE_4_CONSENSUS_START',
            called_from="cluster_intelligently",
            clusters_to_process=len(cluster_to_rep_idx),
            consensus_strategy=getattr(self, 'consensus_strategy', 'majority'),
            majority_threshold=getattr(self, 'majority_threshold', 0.6)
        )
        
        for cluster_id, rep_info in cluster_to_rep_idx.items():
            if rep_info['start'] < len(llm_results):
                # Estrai risultati LLM per tutti i rappresentanti di questo cluster
                cluster_llm_results = llm_results[rep_info['start']:rep_info['start'] + rep_info['count']]
                
                # Calcola etichetta di consenso
                consensus_result = self._build_consensus_label(cluster_llm_results)
                
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                cluster_info[cluster_id] = {
                    'intent': consensus_result['final_label'],
                    'sub_cluster': 0,
                    'size': len(cluster_indices),
                    'indices': cluster_indices,
                    'intent_string': consensus_result['final_label'],
                    'classification_method': 'llm_ensemble_consensus',
                    'confidence': consensus_result['confidence'],
                    'reasoning': consensus_result['reasoning'],
                    'consensus_stats': consensus_result['stats'],
                    'representatives_count': rep_info['count']
                }
                
                print(f"   ✅ Cluster {cluster_id}: '{consensus_result['final_label']}' "
                      f"({len(cluster_indices)} sessioni, {rep_info['count']} reps, "
                      f"conf: {consensus_result['confidence']:.3f}, "
                      f"consenso: {consensus_result['stats']['agreement_ratio']:.2f})")
        
        # 🔍 TRACING FASE 4 COMPLETE
        trace_all(
            'cluster_intelligently', 
            'PHASE_4_CONSENSUS_COMPLETE',
            called_from="cluster_intelligently",
            clusters_processed=len(cluster_info),
            avg_cluster_confidence=np.mean([info.get('confidence', 0.0) for info in cluster_info.values()]) if cluster_info else 0.0,
            consensus_results_summary=[{
                'cluster_id': cid,
                'intent': info.get('intent', 'unknown'),
                'confidence': info.get('confidence', 0.0),
                'size': info.get('size', 0)
            } for cid, info in cluster_info.items()]
        )
        
        # FASE 5: Gestione outliers come rappresentanti individuali
        outlier_indices = [i for i, label in enumerate(cluster_labels) if label == -1]
        if outlier_indices:
            print(f"🔍 Fase 5: Analisi individuale di {len(outlier_indices)} outliers come rappresentanti...")
            debug_logger.info(f"🔍 FASE 5: Analisi individuale di {len(outlier_indices)} outliers")
            debug_logger.info("📋 RISULTATI CLASSIFICAZIONE OUTLIER:")
            debug_logger.info("-" * 60)
            
            # 🔍 TRACING FASE 5
            trace_all(
                'cluster_intelligently', 
                'PHASE_5_OUTLIERS_START',
                called_from="cluster_intelligently",
                outlier_count=len(outlier_indices),
                outlier_ratio=len(outlier_indices) / len(texts) if len(texts) > 0 else 0.0,
                outlier_indices=outlier_indices[:10]  # Prime 10 per debug
            )
            
            # 🆕 MODIFICA: Tratta ogni outlier come rappresentante individuale
            outlier_texts = [texts[i] for i in outlier_indices]
            outlier_llm_results = self.extract_intents_with_llm(outlier_texts)
            
            # Debug dettagliato per ogni outlier
            for outlier_idx, llm_result in zip(outlier_indices, outlier_llm_results):
                debug_logger.info(f"Caso: {outlier_idx} - Outlier: {llm_result['intent']}")
                print(f"   🔍 Caso {outlier_idx} (Outlier): '{llm_result['intent']}' (conf: {llm_result['confidence']:.3f})")
            
            debug_logger.info("")
            
            # 🆕 MODIFICA: Raggruppa outliers per intent solo se hanno alta confidenza
            outlier_groups = defaultdict(list)
            for outlier_idx, llm_result in zip(outlier_indices, outlier_llm_results):
                intent = llm_result['intent']
                confidence = llm_result['confidence']
                
                # Solo outliers con alta confidenza vengono raggruppati
                if confidence >= 0.7:  # Soglia ridotta da 0.8 a 0.7
                    outlier_groups[intent].append((outlier_idx, llm_result))
                else:
                    # Outlier a bassa confidenza restano outlier
                    print(f"   ⚠️ Outlier {outlier_idx} a bassa confidenza ({confidence:.3f}) resta classificato come outlier")
            
            # Crea nuovi cluster per outliers raggruppati con stesso intent
            next_cluster_id = max(cluster_labels) + 1 if len(cluster_labels) > 0 else 0
            
            debug_logger.info("🆕 RAGGRUPPAMENTO OUTLIER PER INTENT:")
            debug_logger.info("-" * 60)
            
            for intent, outlier_data in outlier_groups.items():
                if len(outlier_data) >= 2:  # Minimo 2 outliers per formare un cluster
                    indices = [data[0] for data in outlier_data]
                    avg_confidence = np.mean([data[1]['confidence'] for data in outlier_data])
                    
                    # Assegna nuovo cluster ID
                    for idx in indices:
                        cluster_labels[idx] = next_cluster_id
                    
                    cluster_info[next_cluster_id] = {
                        'intent': intent,
                        'sub_cluster': 0,
                        'size': len(indices),
                        'indices': indices,
                        'intent_string': intent,
                        'classification_method': 'llm_outlier_grouped',
                        'confidence': avg_confidence,
                        'reasoning': f'Outliers raggruppati per intent {intent}',
                        'outlier_representatives': [data[1] for data in outlier_data]  # 🆕 Salva tutti i rappresentanti outlier
                    }
                    
                    debug_logger.info(f"🆕 Nuovo cluster {next_cluster_id} da outlier: '{intent}' ({len(indices)} casi, conf: {avg_confidence:.3f})")
                    print(f"   🆕 Nuovo cluster {next_cluster_id}: '{intent}' ({len(indices)} ex-outliers, conf: {avg_confidence:.3f})")
                    
                    next_cluster_id += 1
                else:
                    # Outlier singolo con alta confidenza: mantieni come outlier ma salva classificazione
                    for outlier_idx, llm_result in outlier_data:
                        debug_logger.info(f"🔍 Outlier singolo {outlier_idx}: '{intent}' (conf: {llm_result['confidence']:.3f}) - mantiene status outlier")
            
            # 🆕 SALVA INFO OUTLIER: Crea cluster_info per outlier rimasti
            remaining_outliers = [i for i, label in enumerate(cluster_labels) if label == -1]
            if remaining_outliers:
                # Trova le loro classificazioni
                outlier_classifications = {}
                for outlier_idx, llm_result in zip(outlier_indices, outlier_llm_results):
                    if outlier_idx in remaining_outliers:
                        outlier_classifications[outlier_idx] = llm_result
                
                cluster_info[-1] = {
                    'intent': 'outlier_mixed',
                    'sub_cluster': 0,
                    'size': len(remaining_outliers),
                    'indices': remaining_outliers,
                    'intent_string': 'Outlier (Misti)',
                    'classification_method': 'llm_outlier_individual',
                    'confidence': 0.5,
                    'reasoning': 'Outliers individuali a bassa confidenza o intent unici',
                    'individual_classifications': outlier_classifications  # 🆕 Salva classificazioni individuali
                }
                
                debug_logger.info(f"🔍 Outlier rimanenti: {len(remaining_outliers)} casi mantengono status outlier")
        
        # Statistiche finali
        final_n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        final_n_outliers = sum(1 for label in cluster_labels if label == -1)
        total_llm_calls = len(representative_texts) + len([i for i in outlier_indices if cluster_labels[i] == -1])
        
        # Calcola confidenza media
        all_confidences = []
        for info in cluster_info.values():
            if 'confidence' in info:
                all_confidences.append(info['confidence'])
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        print(f"✅ Clustering intelligente OTTIMIZZATO completato:")
        print(f"   📊 Cluster finali: {final_n_clusters}")
        print(f"   🔍 Outliers rimanenti: {final_n_outliers}")
        print(f"   🤖 Chiamate LLM totali: {total_llm_calls} (vs {len(texts)} dell'approccio naive)")
        print(f"   ⚡ Efficienza: {((len(texts) - total_llm_calls) / len(texts) * 100):.1f}% di riduzione chiamate LLM")
        print(f"   🎯 Confidenza media: {avg_confidence:.3f}")
        
        # 🆕 LOGGING FINALE
        debug_logger.info("=" * 80)
        debug_logger.info("✅ CLUSTERING INTELLIGENTE COMPLETATO")
        debug_logger.info(f"📊 Cluster finali: {final_n_clusters}")
        debug_logger.info(f"🔍 Outliers rimanenti: {final_n_outliers}")
        debug_logger.info(f"🤖 Chiamate LLM totali: {total_llm_calls}")
        debug_logger.info(f"⚡ Efficienza: {((len(texts) - total_llm_calls) / len(texts) * 100):.1f}% riduzione chiamate LLM")
        debug_logger.info(f"🎯 Confidenza media: {avg_confidence:.3f}")
        debug_logger.info("=" * 80)
        
        # Cleanup handlers per evitare memory leak
        for handler in debug_logger.handlers[:]:
            debug_logger.removeHandler(handler)
            handler.close()
        
        # 🔍 TRACING EXIT
        trace_all(
            'cluster_intelligently', 
            'EXIT',
            called_from="IntelligentIntentClusterer",
            final_clusters_count=final_n_clusters,
            final_outliers_count=final_n_outliers,
            total_llm_calls=total_llm_calls,
            efficiency_percentage=((len(texts) - total_llm_calls) / len(texts) * 100) if len(texts) > 0 else 0.0,
            avg_confidence=avg_confidence,
            cluster_info_keys=list(cluster_info.keys()),
            cluster_labels_shape=cluster_labels.shape if cluster_labels is not None else None,
            processing_successful=True
        )
        
        return cluster_labels, cluster_info
    
    def _select_cluster_representatives(self, texts: List[str], embeddings: np.ndarray, cluster_labels: np.ndarray) -> Dict[int, List[Dict]]:
        """
        Seleziona rappresentanti per ogni cluster basandosi sulla posizione centrale
        
        Args:
            texts: Lista di testi
            embeddings: Embeddings
            cluster_labels: Label dei cluster
            
        Returns:
            Dizionario cluster_id -> lista di rappresentanti
        """
        representatives = {}
        
        for cluster_id in set(cluster_labels):
            if cluster_id >= 0:  # Escludi outliers
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_indices) > 0:
                    cluster_embeddings = embeddings[cluster_indices]
                    
                    # Calcola centroide del cluster
                    centroid = np.mean(cluster_embeddings, axis=0)
                    
                    # Trova i rappresentanti più vicini al centroide
                    distances = []
                    for i, emb in enumerate(cluster_embeddings):
                        dist = np.linalg.norm(emb - centroid)
                        distances.append((dist, cluster_indices[i]))
                    
                    # Ordina per distanza e prendi i primi n_representatives
                    distances.sort(key=lambda x: x[0])
                    n_reps = min(self.n_representatives, len(distances))
                    
                    representatives[cluster_id] = []
                    for j in range(n_reps):
                        idx = distances[j][1]
                        representatives[cluster_id].append({
                            'testo_completo': texts[idx],
                            'index': idx,
                            'distance_to_centroid': distances[j][0]
                        })
        
        return representatives
    
    def _select_diverse_representatives(self, cluster_embeddings: np.ndarray, 
                                       cluster_indices: List[int], 
                                       texts: List[str]) -> List[int]:
        """
        Seleziona rappresentanti diversificati per un cluster usando strategia multi-criterio:
        - Centro: rappresentanti più centrali per stabilità
        - Periferia: rappresentanti più lontani per diversità  
        - Diversità: massimizza distanza tra rappresentanti selezionati
        
        Args:
            cluster_embeddings: Embedding del cluster
            cluster_indices: Indici originali nel dataset
            texts: Testi completi per riferimento
            
        Returns:
            Lista di indici globali dei rappresentanti selezionati
        """
        cluster_size = len(cluster_embeddings)
        
        # Numero adattivo di rappresentanti basato sulla dimensione del cluster
        if cluster_size <= 3:
            n_reps = cluster_size  # Prendi tutti se molto piccolo
        elif cluster_size <= 10:
            n_reps = min(3, cluster_size)
        elif cluster_size <= 50:
            n_reps = 4
        else:
            n_reps = 5  # Max 5 rappresentanti anche per cluster grandi
        
        if n_reps <= 1:
            # Fallback: solo medoid se cluster troppo piccolo
            centroid = np.mean(cluster_embeddings, axis=0)
            distances_to_center = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            medoid_local_idx = np.argmin(distances_to_center)
            return [cluster_indices[medoid_local_idx]]
        
        # Calcola centroide e distanze
        centroid = np.mean(cluster_embeddings, axis=0)
        distances_to_center = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        selected_local_indices = []
        
        # 1. CENTRO: Inizia sempre con il medoid (punto più centrale)
        medoid_local_idx = np.argmin(distances_to_center)
        selected_local_indices.append(medoid_local_idx)
        
        # 2. DIVERSITÀ: Seleziona iterativamente punti massimamente distanti dai già selezionati
        for _ in range(n_reps - 1):
            if len(selected_local_indices) >= cluster_size:
                break
                
            max_min_distance = -1
            best_candidate = -1
            
            for candidate_idx in range(cluster_size):
                if candidate_idx in selected_local_indices:
                    continue
                    
                # Calcola distanza minima dai punti già selezionati
                min_distance_to_selected = float('inf')
                for selected_idx in selected_local_indices:
                    distance = np.linalg.norm(
                        cluster_embeddings[candidate_idx] - cluster_embeddings[selected_idx]
                    )
                    min_distance_to_selected = min(min_distance_to_selected, distance)
                
                # Selezione Max-Min: scegli il candidato con la massima distanza minima
                if min_distance_to_selected > max_min_distance:
                    max_min_distance = min_distance_to_selected
                    best_candidate = candidate_idx
            
            if best_candidate != -1:
                selected_local_indices.append(best_candidate)
        
        # Converti indici locali in indici globali
        selected_global_indices = [cluster_indices[local_idx] for local_idx in selected_local_indices]
        
        # Debug info: calcola diversità interna
        if len(selected_local_indices) > 1:
            internal_distances = []
            for i in range(len(selected_local_indices)):
                for j in range(i + 1, len(selected_local_indices)):
                    dist = np.linalg.norm(
                        cluster_embeddings[selected_local_indices[i]] - 
                        cluster_embeddings[selected_local_indices[j]]
                    )
                    internal_distances.append(dist)
            avg_internal_diversity = np.mean(internal_distances)
        else:
            avg_internal_diversity = 0.0
        
        return selected_global_indices
    
    def _build_consensus_label(self, llm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Costruisce etichetta di consenso da multiple classificazioni LLM.
        Implementa algoritmo di consensus voting con fallback intelligenti.
        
        Args:
            llm_results: Lista di risultati LLM per i rappresentanti del cluster
            
        Returns:
            Dizionario con etichetta finale, confidenza e statistiche di consenso
        """
        trace_all(
            'ENTER', 
            '_build_consensus_label', 
            {
                'llm_results_count': len(llm_results) if llm_results else 0,
                'has_results': bool(llm_results),
                'first_result_preview': llm_results[0] if llm_results else None
            }
        )
        
        if not llm_results:
            result = {
                'final_label': 'altro',
                'confidence': 0.1,
                'reasoning': 'Nessun risultato LLM disponibile',
                'stats': {'agreement_ratio': 0.0, 'total_representatives': 0}
            }
            trace_all(
                'EXIT', 
                '_build_consensus_label', 
                {
                    'final_label': result['final_label'],
                    'confidence': result['confidence'],
                    'reason': 'NO_LLM_RESULTS',
                    'stats': result['stats']
                }
            )
            return result
        
        if len(llm_results) == 1:
            # Solo un rappresentante: usa direttamente il risultato
            result = llm_results[0]
            consensus_result = {
                'final_label': result['intent'],
                'confidence': result['confidence'],
                'reasoning': f"Single representative: {result['reasoning']}",
                'stats': {'agreement_ratio': 1.0, 'total_representatives': 1}
            }
            trace_all(
                'EXIT', 
                '_build_consensus_label', 
                {
                    'final_label': consensus_result['final_label'],
                    'confidence': consensus_result['confidence'],
                    'reason': 'SINGLE_REPRESENTATIVE',
                    'stats': consensus_result['stats']
                }
            )
            return consensus_result
        
        # Raccoglie tutte le etichette e le confidenze
        labels = [result['intent'] for result in llm_results]
        confidences = [result['confidence'] for result in llm_results]
        
        # Conta le occorrenze di ogni etichetta
        label_counts = Counter(labels)
        total_reps = len(llm_results)
        
        # Strategia di consenso
        most_common_label, most_common_count = label_counts.most_common(1)[0]
        agreement_ratio = most_common_count / total_reps
        
        # CASO 1: Maggioranza chiara (>60% concordano)
        if agreement_ratio >= 0.6:
            # Usa etichetta di maggioranza con confidenza media pesata
            matching_results = [r for r in llm_results if r['intent'] == most_common_label]
            avg_confidence = np.mean([r['confidence'] for r in matching_results])
            
            # Bonus di confidenza per consenso forte
            consensus_bonus = min(0.1, (agreement_ratio - 0.6) * 0.25)
            final_confidence = min(1.0, avg_confidence + consensus_bonus)
            
            reasoning = f"Consenso maggioranza: {most_common_count}/{total_reps} rappresentanti concordano su '{most_common_label}'"
            
        # CASO 2: Nessuna maggioranza chiara
        else:
            # Strategia: scegli l'etichetta con confidenza più alta
            best_result = max(llm_results, key=lambda x: x['confidence'])
            most_common_label = best_result['intent']
            final_confidence = best_result['confidence'] * 0.8  # Penalizza per mancanza di consenso
            
            reasoning = f"Nessun consenso chiaro. Usata etichetta più confidente: '{most_common_label}' (conf: {best_result['confidence']:.3f})"
        
        # CASO 3: Confidenza troppo bassa -> fallback conservativo
        if final_confidence < 0.5:
            final_confidence = max(0.3, final_confidence)  # Floor minimo
            reasoning += " [Confidenza aumentata per threshold minimo]"
        
        # Statistiche dettagliate per monitoring
        consensus_stats = {
            'agreement_ratio': agreement_ratio,
            'total_representatives': total_reps,
            'label_distribution': dict(label_counts),
            'confidence_range': [min(confidences), max(confidences)],
            'confidence_std': float(np.std(confidences))
        }
        
        final_result = {
            'final_label': most_common_label,
            'confidence': final_confidence,
            'reasoning': reasoning,
            'stats': consensus_stats
        }
        
        trace_all(
            'EXIT', 
            '_build_consensus_label', 
            {
                'final_label': final_result['final_label'],
                'confidence': final_result['confidence'],
                'agreement_ratio': agreement_ratio,
                'total_reps': total_reps,
                'consensus_method': 'MAJORITY' if agreement_ratio >= 0.6 else 'HIGHEST_CONFIDENCE',
                'stats': consensus_stats
            }
        )
        
        return final_result
