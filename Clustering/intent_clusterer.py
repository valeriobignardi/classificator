"""
Clustering migliorato con analisi degli intent - configurazione da config.yaml
"""

import numpy as np
import re
import yaml
import os
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter

# Import config_loader per caricare config.yaml con variabili ambiente
from config_loader import load_config


class IntentBasedClusterer:
    """
    Clusterer che prima identifica intent specifici, poi raggruppa semanticamente.
    Tutti i pattern di intent sono caricati da config.yaml.
    """
    
    def __init__(self, config_path: str = None):
        """
        Inizializza il clusterer con pattern da configurazione
        
        Args:
            config_path: Percorso del file di configurazione
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        self.config_path = config_path
        self.load_intent_patterns()
    
    def load_intent_patterns(self):
        """Carica i pattern di intent dal file di configurazione"""
        try:
            config = load_config()
            
            intent_config = config.get('intent_clustering', {})
            self.enabled = intent_config.get('enabled', True)
            self.min_conversations_per_intent = intent_config.get('min_conversations_per_intent', 2)
            
            # Carica pattern di intent
            self.intent_patterns = intent_config.get('intent_patterns', {})
            self.fallback_patterns = intent_config.get('fallback_patterns', {})
            
            # Configurazione per semantic refinement
            semantic_config = intent_config.get('semantic_refinement', {})
            self.semantic_refinement_enabled = semantic_config.get('enabled', True)
            self.min_size_for_refinement = semantic_config.get('min_size_for_refinement', 10)
            self.similarity_threshold = semantic_config.get('similarity_threshold', 0.85)
            
            print(f"✅ Caricati {len(self.intent_patterns)} intent patterns da configurazione")
            print(f"🎯 Intent disponibili: {list(self.intent_patterns.keys())}")
            
        except Exception as e:
            print(f"⚠️ Errore caricamento configurazione intent: {e}")
            # Fallback ai pattern predefiniti
            self._load_default_patterns()
    
    def _load_default_patterns(self):
        """Pattern di fallback se il config non è disponibile"""
        self.enabled = True
        self.min_conversations_per_intent = 2
        self.intent_patterns = {
            'accesso_problemi': [
                'non\\s+riesco.*accedere',
                'errore.*login',
                'problema.*password'
            ],
            'prenotazione_esami': [
                'prenotare.*visita',
                'fissare.*appuntamento'
            ],
            'info_generali': [
                'informazioni.*servizi',
                'cosa\\s+offrite'
            ]
        }
        self.fallback_patterns = {
            'altro': ['.*']
        }
        print("⚠️ Usati pattern predefiniti per intent clustering")
    
    def extract_intents(self, texts: List[str]) -> List[Set[str]]:
        """
        Estrae intent da ogni testo usando i pattern configurati
        
        Returns:
            Lista di set di intent per ogni testo
        """
        results = []
        
        for text in texts:
            text_lower = text.lower()
            detected_intents = set()
            
            # Cerca intent specifici prima
            for intent, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    try:
                        if re.search(pattern, text_lower):
                            detected_intents.add(intent)
                            break  # Un pattern per intent è sufficiente
                    except re.error as e:
                        print(f"⚠️ Errore pattern regex '{pattern}': {e}")
            
            # Se nessun intent specifico trovato, prova i pattern di fallback
            if not detected_intents:
                for category, patterns in self.fallback_patterns.items():
                    for pattern in patterns:
                        try:
                            if re.search(pattern, text_lower):
                                detected_intents.add(category)
                                break
                        except re.error as e:
                            print(f"⚠️ Errore pattern fallback '{pattern}': {e}")
                    if detected_intents:
                        break
            
            # Se ancora nessun intent, aggiungi 'altro'
            if not detected_intents:
                detected_intents.add('altro')
            
            results.append(detected_intents)
        
        return results
    
    def cluster_by_intent_combination(self, texts: List[str], embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Raggruppa per combinazione di intent, poi applica clustering semantico fine se abilitato
        
        Args:
            texts: Lista di testi da clusterizzare
            embeddings: Embeddings corrispondenti ai testi
            
        Returns:
            Tuple (cluster_labels, cluster_info)
        """
        if not self.enabled:
            print("⚠️ Intent clustering disabilitato, using fallback to generic clustering")
            return self._fallback_clustering(embeddings)
        
        print(f"🎯 Estrazione intent da {len(texts)} conversazioni...")
        intents_per_text = self.extract_intents(texts)
        
        # Debug: mostra distribuzione intent
        intent_counter = Counter()
        for intents in intents_per_text:
            for intent in intents:
                intent_counter[intent] += 1
        
        print(f"📊 Distribuzione intent rilevati:")
        for intent, count in intent_counter.most_common():
            print(f"  - {intent}: {count} conversazioni")
        
        # Raggruppa per combinazione di intent
        intent_groups = defaultdict(list)
        for i, intents in enumerate(intents_per_text):
            intent_key = tuple(sorted(intents))
            intent_groups[intent_key].append(i)
        
        # Assegna cluster ID solo ai gruppi che soddisfano la soglia minima
        cluster_labels = np.full(len(texts), -1)
        cluster_info = {}
        cluster_id = 0
        
        print(f"🔍 Analisi gruppi di intent:")
        for intent_combo, indices in intent_groups.items():
            size = len(indices)
            intent_str = ' + '.join(intent_combo) if len(intent_combo) > 1 else intent_combo[0]
            
            if size >= self.min_conversations_per_intent:
                # Assegna cluster ID
                for idx in indices:
                    cluster_labels[idx] = cluster_id
                
                cluster_info[cluster_id] = {
                    'intents': intent_combo,
                    'size': size,
                    'indices': indices,
                    'intent_string': intent_str
                }
                print(f"  ✅ Cluster {cluster_id}: {intent_str} ({size} conv.)")
                cluster_id += 1
            else:
                print(f"  ❌ Scartato: {intent_str} ({size} conv. < {self.min_conversations_per_intent})")
        
        # Applica semantic refinement se abilitato
        if self.semantic_refinement_enabled:
            cluster_labels, cluster_info = self._apply_semantic_refinement(
                cluster_labels, cluster_info, embeddings, texts
            )
        
        return cluster_labels, cluster_info
    
    def _apply_semantic_refinement(self, cluster_labels: np.ndarray, cluster_info: Dict,
                                 embeddings: np.ndarray, texts: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Applica sub-clustering semantico ai cluster grandi
        """
        print(f"🔧 Applicazione semantic refinement...")
        
        new_cluster_labels = cluster_labels.copy()
        new_cluster_info = {}
        next_cluster_id = max(cluster_info.keys()) + 1 if cluster_info else 0
        
        for cluster_id, info in cluster_info.items():
            if info['size'] >= self.min_size_for_refinement:
                print(f"  🔍 Refinement cluster {cluster_id} ({info['size']} conversazioni)")
                
                # Estrai embeddings del cluster
                cluster_indices = info['indices']
                cluster_embeddings = embeddings[cluster_indices]
                
                # Applica clustering semantico fine (es. HDBSCAN con parametri stretti)
                from hdbscan import HDBSCAN
                
                # Normalizza gli embedding per simulare distanza coseno
                cluster_embeddings_norm = cluster_embeddings / np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
                
                fine_clusterer = HDBSCAN(
                    min_cluster_size=max(2, info['size'] // 4),
                    min_samples=1,
                    metric='euclidean',  # Usa euclidean su embedding normalizzati
                    cluster_selection_epsilon=1 - self.similarity_threshold
                )
                
                sub_labels = fine_clusterer.fit_predict(cluster_embeddings_norm)
                
                # Aggiorna cluster labels
                unique_sub_labels = set(sub_labels)
                if len(unique_sub_labels) > 1 and -1 not in unique_sub_labels:
                    # Ha trovato sub-cluster validi
                    for i, sub_label in enumerate(sub_labels):
                        if sub_label != -1:
                            global_idx = cluster_indices[i]
                            new_cluster_labels[global_idx] = next_cluster_id + sub_label
                    
                    # Crea info per i nuovi sub-cluster
                    for sub_label in unique_sub_labels:
                        if sub_label != -1:
                            sub_indices = [cluster_indices[i] for i, sl in enumerate(sub_labels) if sl == sub_label]
                            new_cluster_info[next_cluster_id + sub_label] = {
                                'intents': info['intents'],
                                'size': len(sub_indices),
                                'indices': sub_indices,
                                'intent_string': info['intent_string'] + f" (gruppo {sub_label + 1})",
                                'is_refined': True
                            }
                    
                    next_cluster_id += len(unique_sub_labels)
                    print(f"    ➡️ Suddiviso in {len(unique_sub_labels)} sub-cluster")
                else:
                    # Mantieni cluster originale
                    new_cluster_info[cluster_id] = info
            else:
                # Mantieni cluster originale
                new_cluster_info[cluster_id] = info
        
        return new_cluster_labels, new_cluster_info
    
    def _fallback_clustering(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Clustering di fallback se intent clustering è disabilitato"""
        from hdbscan import HDBSCAN
        
        # Normalizza gli embedding per simulare distanza coseno
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        clusterer = HDBSCAN(min_cluster_size=3, min_samples=2, metric='euclidean')
        cluster_labels = clusterer.fit_predict(embeddings_norm)
        
        # Crea cluster_info di base
        cluster_info = {}
        unique_labels = set(cluster_labels)
        for label in unique_labels:
            if label != -1:
                indices = [i for i, l in enumerate(cluster_labels) if l == label]
                cluster_info[label] = {
                    'intents': ('cluster_generico',),
                    'size': len(indices),
                    'indices': indices,
                    'intent_string': f'Cluster Generico {label}'
                }
        
        return cluster_labels, cluster_info
