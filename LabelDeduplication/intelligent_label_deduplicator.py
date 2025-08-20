"""
Sistema intelligente per la dedupplicazione delle etichette basato su ML e LLM
invece di pattern predefiniti. Utilizza similarità semantica, clustering gerarchico
e validazione LLM per decisioni robuste e scalabili.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import logging
from datetime import datetime
import json

# Aggiungi path per i moduli
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EmbeddingEngine'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LLMClassifier'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SemanticMemory'))

class IntelligentLabelDeduplicator:
    """
    Sistema intelligente per prevenzione duplicati etichette
    basato su ML + LLM invece di pattern predefiniti
    """
    
    def __init__(self, 
                 embedder,
                 llm_classifier,
                 semantic_memory,
                 similarity_threshold: float = 0.85,
                 llm_confidence_threshold: float = 0.7,
                 clustering_threshold: float = 0.15):
        """
        Inizializza il dedupplicatore intelligente
        
        Args:
            embedder: Modello di embedding (LaBSE)
            llm_classifier: Classificatore LLM (Mistral)
            semantic_memory: Gestore memoria semantica
            similarity_threshold: Soglia similarità semantica (0.0-1.0)
            llm_confidence_threshold: Soglia confidenza LLM
            clustering_threshold: Soglia per clustering gerarchico
        """
        self.embedder = embedder
        self.llm_classifier = llm_classifier
        self.semantic_memory = semantic_memory
        self.similarity_threshold = similarity_threshold
        self.llm_confidence_threshold = llm_confidence_threshold
        self.clustering_threshold = clustering_threshold
        
        # Cache per performance
        self.embedding_cache = {}
        self.llm_decision_cache = {}
        
        # Storico decisioni per apprendimento
        self.decision_history = []
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def prevent_duplicate_labels(self, 
                                suggested_labels: Dict[int, str],
                                existing_labels: Optional[List[str]] = None) -> Dict[int, str]:
        """
        Processo intelligente di dedupplicazione a 4 fasi
        
        Args:
            suggested_labels: Dizionario {cluster_id: etichetta_suggerita}
            existing_labels: Lista etichette esistenti (opzionale, recupera dal DB se None)
            
        Returns:
            Dizionario {cluster_id: etichetta_dedupplicata}
        """
        print("🧠 Avvio dedupplicazione intelligente delle etichette...")
        
        # Recupera etichette esistenti se non fornite
        if existing_labels is None:
            existing_labels = self._get_existing_labels()
        
        print(f"📊 Analizzando {len(suggested_labels)} nuove etichette vs {len(existing_labels)} esistenti")
        
        # Fase 1: Analisi semantica con embeddings
        semantic_matches = self._phase1_semantic_analysis(suggested_labels, existing_labels)
        
        # Fase 2: Clustering gerarchico delle etichette
        cluster_groups = self._phase2_hierarchical_clustering(existing_labels)
        
        # Fase 3: Validazione LLM per casi ambigui
        llm_decisions = self._phase3_llm_validation(semantic_matches, cluster_groups)
        
        # Fase 4: Decisione finale con confidence scoring
        final_labels = self._phase4_final_decision(suggested_labels, semantic_matches, llm_decisions)
        
        # Salva decisioni per apprendimento futuro
        self._save_decision_history(suggested_labels, final_labels)
        
        print(f"✅ Dedupplicazione completata: {len(final_labels)} etichette processate")
        return final_labels
    
    def _phase1_semantic_analysis(self, 
                                 suggested_labels: Dict[int, str],
                                 existing_labels: List[str]) -> Dict[int, List[Dict]]:
        """
        Fase 1: Analisi semantica con embeddings
        
        Returns:
            Dict {cluster_id: [candidati_simili]} dove ogni candidato ha:
            {label, similarity_score, match_type}
        """
        print("🔍 Fase 1: Analisi semantica con embeddings...")
        
        semantic_matches = {}
        
        # Se non ci sono etichette esistenti, ritorna matches vuoti
        if not existing_labels:
            for cluster_id in suggested_labels.keys():
                semantic_matches[cluster_id] = []
                print(f"  📋 Cluster {cluster_id}: nessuna etichetta esistente per confronto")
            return semantic_matches
        
        # Genera embeddings per etichette esistenti (con cache)
        existing_embeddings = self._get_embeddings_for_labels(existing_labels)
        
        for cluster_id, suggested_label in suggested_labels.items():
            # Genera embedding per etichetta suggerita
            suggested_embedding = self._get_embedding_for_label(suggested_label)
            
            # Calcola similarità con tutte le etichette esistenti
            similarities = cosine_similarity([suggested_embedding], existing_embeddings)[0]
            
            # Trova candidati simili sopra una soglia più bassa per analisi
            candidates = []
            for i, (existing_label, similarity) in enumerate(zip(existing_labels, similarities)):
                if similarity > 0.5:  # Soglia bassa per catturare tutti i possibili candidati
                    match_type = self._classify_match_type(similarity)
                    candidates.append({
                        'label': existing_label,
                        'similarity_score': float(similarity),
                        'match_type': match_type,
                        'embedding_index': i
                    })
            
            # Ordina per similarità decrescente
            candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
            semantic_matches[cluster_id] = candidates[:5]  # Top 5 candidati
            
            print(f"  📋 Cluster {cluster_id} ('{suggested_label}'): {len(candidates)} candidati simili")
        
        return semantic_matches
    
    def _phase2_hierarchical_clustering(self, existing_labels: List[str]) -> Dict[str, List[str]]:
        """
        Fase 2: Clustering gerarchico delle etichette esistenti
        
        Returns:
            Dict {etichetta_canonica: [etichette_nel_gruppo]}
        """
        print("🌳 Fase 2: Clustering gerarchico delle etichette esistenti...")
        
        if len(existing_labels) < 2:
            if existing_labels:
                return {existing_labels[0]: existing_labels}
            else:
                print("  📭 Nessuna etichetta esistente per clustering")
                return {}
        
        # Genera embeddings per tutte le etichette esistenti
        embeddings = self._get_embeddings_for_labels(existing_labels)
        
        # Clustering gerarchico agglomerativo
        n_clusters = max(1, min(len(existing_labels) // 2, 20))  # Numero adattivo di cluster
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Raggruppa etichette per cluster
        label_groups = defaultdict(list)
        for label, cluster_id in zip(existing_labels, cluster_labels):
            label_groups[cluster_id].append(label)
        
        # Identifica etichette canoniche (più frequenti o centrali)
        cluster_groups = {}
        for cluster_id, labels in label_groups.items():
            # Per ora usa la prima etichetta come canonica
            # In futuro si potrebbe usare la più frequente dal database
            canonical = labels[0]
            cluster_groups[canonical] = labels
            
            if len(labels) > 1:
                print(f"  🏷️  Gruppo {cluster_id}: {canonical} ← {labels[1:]}")
        
        return cluster_groups
    
    def _phase3_llm_validation(self, 
                              semantic_matches: Dict[int, List[Dict]],
                              cluster_groups: Dict[str, List[str]]) -> Dict[int, Dict]:
        """
        Fase 3: Validazione LLM per casi ambigui
        
        Returns:
            Dict {cluster_id: {decision, confidence, motivation}}
        """
        print("🤖 Fase 3: Validazione LLM per casi ambigui...")
        
        llm_decisions = {}
        
        for cluster_id, candidates in semantic_matches.items():
            # Identifica casi che necessitano validazione LLM
            ambiguous_candidates = [
                c for c in candidates 
                if 0.7 <= c['similarity_score'] <= 0.9  # Zona ambigua
            ]
            
            if not ambiguous_candidates:
                continue
            
            # Per ogni candidato ambiguo, chiedi al LLM
            for candidate in ambiguous_candidates[:2]:  # Limita a top 2 per performance
                decision = self._ask_llm_for_label_similarity(
                    label1=f"cluster_{cluster_id}",  # Placeholder, in uso reale avresti il testo
                    label2=candidate['label']
                )
                
                llm_decisions[f"{cluster_id}_{candidate['label']}"] = decision
                
                print(f"  🧠 LLM: cluster_{cluster_id} vs '{candidate['label']}' → {decision['decision']} "
                      f"(conf: {decision['confidence']:.2f})")
        
        return llm_decisions
    
    def _phase4_final_decision(self, 
                              suggested_labels: Dict[int, str],
                              semantic_matches: Dict[int, List[Dict]],
                              llm_decisions: Dict[str, Dict]) -> Dict[int, str]:
        """
        Fase 4: Decisione finale con confidence scoring
        
        Returns:
            Dict {cluster_id: etichetta_finale}
        """
        print("⚖️  Fase 4: Decisione finale con confidence scoring...")
        
        final_labels = {}
        stats = {'reused': 0, 'new': 0, 'normalized': 0}
        
        for cluster_id, suggested_label in suggested_labels.items():
            candidates = semantic_matches.get(cluster_id, [])
            
            if not candidates:
                # Nessun candidato simile → usa etichetta normalizzata
                final_labels[cluster_id] = self._normalize_label(suggested_label)
                stats['new'] += 1
                continue
            
            # Candidato con similarity più alta
            best_candidate = candidates[0]
            similarity_score = best_candidate['similarity_score']
            
            # Decisione basata su similarity + LLM
            llm_key = f"{cluster_id}_{best_candidate['label']}"
            llm_decision = llm_decisions.get(llm_key)
            
            # Calcola score finale combinato
            final_score = self._calculate_combined_score(
                similarity_score, 
                llm_decision
            )
            
            if final_score >= self.similarity_threshold:
                # Riusa etichetta esistente
                final_labels[cluster_id] = best_candidate['label']
                stats['reused'] += 1
                print(f"  ♻️  Cluster {cluster_id}: riuso '{best_candidate['label']}' "
                      f"(score: {final_score:.3f})")
            else:
                # Crea nuova etichetta normalizzata
                final_labels[cluster_id] = self._normalize_label(suggested_label)
                stats['new'] += 1
                print(f"  🆕 Cluster {cluster_id}: nuova '{final_labels[cluster_id]}' "
                      f"(score: {final_score:.3f} < {self.similarity_threshold})")
        
        print(f"📊 Statistiche finali: {stats['reused']} riusate, {stats['new']} nuove")
        return final_labels
    
    def _get_existing_labels(self) -> List[str]:
        """Recupera etichette esistenti dal sistema"""
        try:
            # Usa la memoria semantica se disponibile
            if self.semantic_memory:
                memory_stats = self.semantic_memory.get_memory_stats()
                return list(memory_stats.get('unique_tags', []))
            else:
                # Fallback: lista vuota
                return []
        except Exception as e:
            self.logger.warning(f"Errore recupero etichette esistenti: {e}")
            return []
    
    def _get_embeddings_for_labels(self, labels: List[str]) -> np.ndarray:
        """Genera embeddings per una lista di etichette con cache"""
        embeddings = []
        
        for label in labels:
            if label in self.embedding_cache:
                embeddings.append(self.embedding_cache[label])
            else:
                # Genera embedding per l'etichetta
                embedding = self._get_embedding_for_label(label)
                self.embedding_cache[label] = embedding
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _get_embedding_for_label(self, label: str) -> np.ndarray:
        """Genera embedding per una singola etichetta"""
        if label in self.embedding_cache:
            return self.embedding_cache[label]
        
        # Converte l'etichetta in testo più descrittivo per embedding migliore
        descriptive_text = self._label_to_descriptive_text(label)
        embedding = self.embedder.encode([descriptive_text])[0]
        
        self.embedding_cache[label] = embedding
        return embedding
    
    def _label_to_descriptive_text(self, label: str) -> str:
        """Converte etichetta in testo descrittivo per embedding migliore"""
        # Sostituisce underscore con spazi e migliora leggibilità
        descriptive = label.replace('_', ' ')
        
        # Aggiungi contesto per categorie comuni
        domain_context = {
            'accesso': 'problemi di accesso al portale login password',
            'prenotazione': 'prenotazione visite esami appuntamenti',
            'referti': 'ritiro scarico referti risultati analisi',
            'fatturazione': 'fatture pagamenti costi ricevute',
            'orari': 'orari apertura contatti telefono informazioni',
            'problemi': 'problemi tecnici errori malfunzionamenti'
        }
        
        for keyword, context in domain_context.items():
            if keyword in descriptive.lower():
                descriptive += f" {context}"
                break
        
        return descriptive
    
    def _classify_match_type(self, similarity: float) -> str:
        """Classifica il tipo di match basato sulla similarità"""
        if similarity >= 0.95:
            return 'exact'
        elif similarity >= 0.85:
            return 'high'
        elif similarity >= 0.7:
            return 'medium'
        elif similarity >= 0.5:
            return 'low'
        else:
            return 'none'
    
    def _ask_llm_for_label_similarity(self, label1: str, label2: str) -> Dict[str, Any]:
        """Chiede al LLM di valutare se due etichette sono semanticamente simili"""
        cache_key = f"{label1}||{label2}"
        if cache_key in self.llm_decision_cache:
            return self.llm_decision_cache[cache_key]
        
        if not self.llm_classifier or not hasattr(self.llm_classifier, 'is_available') or not self.llm_classifier.is_available():
            return {'decision': 'uncertain', 'confidence': 0.5, 'motivation': 'LLM non disponibile'}
        
        try:
            prompt = f"""Analizza se queste due etichette si riferiscono allo stesso concetto semantico:

Etichetta 1: {label1}
Etichetta 2: {label2}

Rispondi SOLO con:
- SIMILI se si riferiscono allo stesso concetto
- DIVERSE se sono concetti diversi
- INCERTO se non sei sicuro

Poi spiega brevemente il motivo."""
            
            # Usa il metodo appropriato del classificatore LLM
            if hasattr(self.llm_classifier, 'classify_conversation'):
                response, confidence, motivation = self.llm_classifier.classify_conversation(prompt)
            else:
                # Fallback
                response = "INCERTO"
                confidence = 0.5
                motivation = "Metodo LLM non disponibile"
            
            # Interpreta la risposta
            response_lower = response.lower()
            if 'simili' in response_lower:
                decision = 'similar'
            elif 'diverse' in response_lower:
                decision = 'different'
            else:
                decision = 'uncertain'
            
            result = {
                'decision': decision,
                'confidence': confidence,
                'motivation': motivation
            }
            
            # Cache la decisione
            self.llm_decision_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.warning(f"Errore LLM label similarity: {e}")
            return {'decision': 'uncertain', 'confidence': 0.5, 'motivation': f'Errore: {str(e)}'}
    
    def _calculate_combined_score(self, 
                                 similarity_score: float,
                                 llm_decision: Optional[Dict[str, Any]]) -> float:
        """Calcola score finale combinando similarity + LLM"""
        if not llm_decision:
            return similarity_score
        
        # Pesi per combinazione
        similarity_weight = 0.6
        llm_weight = 0.4
        
        # Score LLM basato su decisione e confidenza
        if llm_decision['decision'] == 'similar':
            llm_score = llm_decision['confidence']
        elif llm_decision['decision'] == 'different':
            llm_score = 1.0 - llm_decision['confidence']
        else:  # uncertain
            llm_score = 0.5
        
        # Combina i punteggi
        combined_score = (similarity_score * similarity_weight) + (llm_score * llm_weight)
        
        return combined_score
    
    def _normalize_label(self, label: str) -> str:
        """Normalizza l'etichetta seguendo convenzioni standard"""
        import re
        
        # Rimuovi caratteri speciali e normalizza spazi
        normalized = re.sub(r'[^\w\s]', '', label)
        normalized = re.sub(r'\s+', '_', normalized.strip())
        normalized = normalized.lower()
        
        # Applica alcune regole di base per coerenza
        basic_rules = {
            'prenotazioni': 'prenotazione',
            'referti': 'ritiro_referti',
            'pagamenti': 'fatturazione',
            'contatti': 'orari_contatti',
            'informazioni': 'info'
        }
        
        for pattern, replacement in basic_rules.items():
            if pattern in normalized:
                normalized = normalized.replace(pattern, replacement)
        
        return normalized
    
    def _save_decision_history(self, 
                              suggested_labels: Dict[int, str],
                              final_labels: Dict[int, str]):
        """Salva lo storico delle decisioni per apprendimento futuro"""
        timestamp = datetime.now().isoformat()
        
        for cluster_id in suggested_labels:
            decision_record = {
                'timestamp': timestamp,
                'cluster_id': cluster_id,
                'suggested_label': suggested_labels[cluster_id],
                'final_label': final_labels[cluster_id],
                'was_reused': suggested_labels[cluster_id] != final_labels[cluster_id]
            }
            self.decision_history.append(decision_record)
        
        # Mantieni solo gli ultimi 1000 record
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Restituisce statistiche del dedupplicatore"""
        if not self.decision_history:
            return {'total_decisions': 0}
        
        total = len(self.decision_history)
        reused = sum(1 for d in self.decision_history if d['was_reused'])
        
        return {
            'total_decisions': total,
            'labels_reused': reused,
            'labels_created': total - reused,
            'reuse_rate': reused / total if total > 0 else 0,
            'cache_size': {
                'embeddings': len(self.embedding_cache),
                'llm_decisions': len(self.llm_decision_cache)
            }
        }
    
    def clear_cache(self):
        """Pulisce le cache per liberare memoria"""
        self.embedding_cache.clear()
        self.llm_decision_cache.clear()
        print("🧹 Cache del dedupplicatore pulite")


def test_intelligent_deduplicator():
    """Test del dedupplicatore intelligente"""
    print("🧪 TEST: Dedupplicatore Intelligente")
    print("=" * 50)
    
    # Mock degli oggetti necessari per il test
    class MockEmbedder:
        def encode(self, texts):
            # Simula embeddings random ma consistenti
            np.random.seed(42)
            return np.random.random((len(texts), 768))
    
    class MockLLM:
        def is_available(self):
            return True
            
        def classify_conversation(self, prompt):
            # Simula risposta del LLM
            if 'accesso' in prompt.lower():
                return "SIMILI", 0.8, "Entrambe riguardano problemi di accesso"
            else:
                return "DIVERSE", 0.9, "Sono concetti diversi"
    
    class MockSemanticMemory:
        def get_memory_stats(self):
            return {
                'unique_tags': [
                    'accesso_portale', 'prenotazione_esami', 'ritiro_referti',
                    'fatturazione_pagamenti', 'orari_contatti'
                ]
            }
    
    # Inizializza dedupplicatore
    deduplicator = IntelligentLabelDeduplicator(
        embedder=MockEmbedder(),
        llm_classifier=MockLLM(),
        semantic_memory=MockSemanticMemory()
    )
    
    # Test con etichette suggerite
    suggested_labels = {
        0: 'problemi_accesso_app',
        1: 'prenotazioni_visite_mediche', 
        2: 'download_risultati_esami',
        3: 'info_pagamenti_fatture',
        4: 'contatto_centralino',
        5: 'assistenza_nuova_categoria'
    }
    
    print("🏷️  Etichette suggerite:")
    for cluster_id, label in suggested_labels.items():
        print(f"  - Cluster {cluster_id}: {label}")
    
    # Esegui dedupplicazione
    final_labels = deduplicator.prevent_duplicate_labels(suggested_labels)
    
    print("\n🎯 Risultati finali:")
    for cluster_id, final_label in final_labels.items():
        original = suggested_labels[cluster_id]
        status = "♻️ RIUSATA" if original != final_label else "🆕 NUOVA"
        print(f"  - Cluster {cluster_id}: {original} → {final_label} {status}")
    
    # Mostra statistiche
    stats = deduplicator.get_statistics()
    print(f"\n📊 Statistiche:")
    print(f"  - Decisioni totali: {stats['total_decisions']}")
    print(f"  - Etichette riusate: {stats['labels_reused']}")
    print(f"  - Nuove etichette: {stats['labels_created']}")
    print(f"  - Tasso di riuso: {stats['reuse_rate']:.2%}")
    
    print("\n✅ Test completato!")


if __name__ == "__main__":
    test_intelligent_deduplicator()
