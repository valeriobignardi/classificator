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
    
    def __init__(self, config_path: str = None, llm_classifier=None, tenant_id: str = None):
        """
        Inizializza il clusterer intelligente
        
        Args:
            config_path: Percorso del file di configurazione globale
            llm_classifier: Classificatore LLM per analisi intent
            tenant_id: ID del tenant per configurazione specifica
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        self.config_path = config_path
        self.llm_classifier = llm_classifier
        self.tenant_id = tenant_id  # 🗂️  [NEW] Memorizza tenant_id
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
        
        Args:
            texts: Lista di testi da analizzare
            
        Returns:
            Lista di dizionari con intent, confidence e reasoning per ogni testo
        """
        results = []
        
        print(f"🤖 Analisi LLM di {len(texts)} conversazioni...")
        
        for i, text in enumerate(texts):
            try:
                if self.llm_classifier and self.llm_classifier.is_available():
                    # Usa LLM per classificazione intent automatica
                    llm_result = self.llm_classifier.classify_with_motivation(text)
                    
                    # Il risultato è un oggetto ClassificationResult
                    intent_info = {
                        'intent': llm_result.predicted_label,
                        'confidence': llm_result.confidence,
                        'reasoning': llm_result.motivation,
                        'method': 'llm_intelligent',
                        'text_preview': text[:100] + '...' if len(text) > 100 else text
                    }
                    
                else:
                    # Fallback se LLM non disponibile
                    intent_info = {
                        'intent': 'altro',
                        'confidence': 0.1,
                        'reasoning': 'LLM non disponibile',
                        'method': 'fallback',
                        'text_preview': text[:100] + '...' if len(text) > 100 else text
                    }
                
                results.append(intent_info)
                
                # Progress feedback
                if (i + 1) % 50 == 0 or i == len(texts) - 1:
                    print(f"   📈 Progresso LLM: {i+1}/{len(texts)}")
                    
            except Exception as e:
                self.logger.error(f"Errore analisi LLM per testo {i}: {e}")
                
                # Fallback per errori
                results.append({
                    'intent': 'altro',
                    'confidence': 0.05,
                    'reasoning': f'Errore LLM: {e}',
                    'method': 'error_fallback',
                    'text_preview': text[:100] + '...' if len(text) > 100 else text
                })
        
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
        print(f"🧠 Avvio clustering intelligente OTTIMIZZATO di {len(texts)} conversazioni...")
        
        # FASE 1: Clustering semantico con HDBSCAN (parametri più restrittivi)
        print(f"📊 Fase 1: Clustering semantico con HDBSCAN...")
        
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
            umap_metric=getattr(self, 'umap_metric', 'cosine')
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
        
        # FASE 2: Selezione rappresentanti per ogni cluster
        print(f"👥 Fase 2: Selezione rappresentanti per cluster...")
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
        
        # FASE 3: Analisi LLM SOLO sui rappresentanti (molto più veloce!)
        print(f"🤖 Fase 3: Analisi LLM di {len(representative_texts)} rappresentanti...")
        if representative_texts:
            llm_results = self.extract_intents_with_llm(representative_texts)
        else:
            llm_results = []
        
        # FASE 4: Creazione cluster_info con etichette da consenso ensemble
        print(f"🔄 Fase 4: Analisi consenso e propagazione etichette...")
        
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
        
        # FASE 5: Gestione outliers (analisi individuale per alta precisione)
        outlier_indices = [i for i, label in enumerate(cluster_labels) if label == -1]
        if outlier_indices:
            print(f"🔍 Fase 5: Analisi individuale di {len(outlier_indices)} outliers...")
            
            # Analizza outliers individualmente (necessario per alta precisione)
            outlier_texts = [texts[i] for i in outlier_indices]
            outlier_llm_results = self.extract_intents_with_llm(outlier_texts)
            
            # Raggruppa outliers con intent simili e alta confidenza
            outlier_groups = defaultdict(list)
            for i, (outlier_idx, llm_result) in enumerate(zip(outlier_indices, outlier_llm_results)):
                intent = llm_result['intent']
                confidence = llm_result['confidence']
                
                # Solo outliers con alta confidenza vengono raggruppati
                if confidence >= 0.8:  # Soglia alta per outliers
                    outlier_groups[intent].append((outlier_idx, llm_result))
            
            # Crea nuovi cluster per outliers raggruppati
            next_cluster_id = max(cluster_labels) + 1 if len(cluster_labels) > 0 else 0
            
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
                        'reasoning': f'Outliers raggruppati per intent {intent}'
                    }
                    
                    print(f"   🆕 Nuovo cluster {next_cluster_id}: '{intent}' ({len(indices)} ex-outliers, conf: {avg_confidence:.3f})")
                    next_cluster_id += 1
        
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
        if not llm_results:
            return {
                'final_label': 'altro',
                'confidence': 0.1,
                'reasoning': 'Nessun risultato LLM disponibile',
                'stats': {'agreement_ratio': 0.0, 'total_representatives': 0}
            }
        
        if len(llm_results) == 1:
            # Solo un rappresentante: usa direttamente il risultato
            result = llm_results[0]
            return {
                'final_label': result['intent'],
                'confidence': result['confidence'],
                'reasoning': f"Single representative: {result['reasoning']}",
                'stats': {'agreement_ratio': 1.0, 'total_representatives': 1}
            }
        
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
        
        return {
            'final_label': most_common_label,
            'confidence': final_confidence,
            'reasoning': reasoning,
            'stats': consensus_stats
        }
