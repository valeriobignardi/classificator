"""
Pipeline end-to-end per il sistema di classificazione delle conversazioni Humanitas
"""

import sys
import os
import yaml
import json
import glob
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Aggiunge i percorsi per importare gli altri moduli
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LettoreConversazioni'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Preprocessing'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EmbeddingEngine'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Clustering'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Classification'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'TagDatabase'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SemanticMemory'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LLMClassifier'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'HumanReview'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LabelDeduplication'))

from lettore import LettoreConversazioni
from session_aggregator import SessionAggregator
from labse_embedder import LaBSEEmbedder
from hdbscan_clusterer import HDBSCANClusterer
# RIMOSSO: from intent_clusterer import IntentBasedClusterer  # Sistema legacy eliminato
from intelligent_intent_clusterer import IntelligentIntentClusterer
from Clustering.hierarchical_adaptive_clusterer import HierarchicalAdaptiveClusterer  # Nuovo sistema gerarchico
# Aggiungi percorsi per gli import
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MySql'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'TagDatabase'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SemanticMemory'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LLMClassifier'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'HumanReview'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LabelDeduplication'))

from tag_database_connector import TagDatabaseConnector
from semantic_memory_manager import SemanticMemoryManager
from interactive_trainer import InteractiveTrainer
from intelligent_label_deduplicator import IntelligentLabelDeduplicator

# Import del classificatore ensemble avanzato
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Classification'))
from advanced_ensemble_classifier import AdvancedEnsembleClassifier

# Import BERTopic provider
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TopicModeling'))
try:
    from bertopic_feature_provider import BERTopicFeatureProvider
    _BERTopic_AVAILABLE = True
except Exception as _e:
    BERTopicFeatureProvider = None
    _BERTopic_AVAILABLE = False
    print(f"⚠️ BERTopic non disponibile: {_e}")

class EndToEndPipeline:
    """
    Pipeline completa per l'estrazione, clustering, classificazione e salvataggio
    delle sessioni di conversazione
    """
    
    def __init__(self,
                 tenant_slug: str = "humanitas",
                 confidence_threshold: float = None,
                 min_cluster_size: int = None,
                 min_samples: int = None,
                 config_path: str = None,
                 auto_mode: bool = None,
                 shared_embedder=None):
        """
        Inizializza la pipeline
        
        Args:
            tenant_slug: Slug del tenant da processare
            confidence_threshold: Soglia di confidenza (None = legge da config)
            min_cluster_size: Dimensione minima del cluster (None = legge da config)
            min_samples: Numero minimo di campioni (None = legge da config)
            config_path: Percorso del file di configurazione
            auto_mode: Se True, modalità automatica (None = legge da config)
            shared_embedder: Embedder condiviso per evitare CUDA out of memory
        """
        self.tenant_slug = tenant_slug
        
        # Carica configurazione
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Leggi parametri da configurazione con fallback ai valori passati
        pipeline_config = config.get('pipeline', {})
        clustering_config = config.get('clustering', {})
        self.bertopic_config = config.get('bertopic', {
            'enabled': False,
            'top_k': 15,
            'return_one_hot': False,
            'model_subdir': 'bertopic'
        })
        # Imposta parametri usando config o valori passati o default
        self.confidence_threshold = (confidence_threshold if confidence_threshold is not None 
                                    else pipeline_config.get('default_confidence_threshold', 0.7))
        self.auto_mode = (auto_mode if auto_mode is not None 
                         else pipeline_config.get('default_auto_mode', False))
        self.auto_retrain = pipeline_config.get('auto_retrain_on_init', False)
        
        # Inizializza i componenti
        print("🚀 Inizializzazione pipeline...")
        print(f"   🎯 Confidence threshold: {self.confidence_threshold}")
        print(f"   🤖 Auto mode: {self.auto_mode}")
        print(f"   🔄 Auto retrain: {self.auto_retrain}")
        
        self.lettore = LettoreConversazioni()
        self.aggregator = SessionAggregator(schema=tenant_slug)
        
        # Usa embedder condiviso se fornito, altrimenti crea nuovo
        if shared_embedder is not None:
            print("🔄 Utilizzo embedder condiviso per evitare CUDA out of memory")
            self.embedder = shared_embedder
        else:
            print("⚠️ Creazione nuovo embedder (potrebbe causare CUDA out of memory)")
            self.embedder = LaBSEEmbedder()
            
        # Usa parametri di clustering da config o da parametri passati
        cluster_min_size = (min_cluster_size if min_cluster_size is not None 
                           else clustering_config.get('min_cluster_size', 
                                pipeline_config.get('default_min_cluster_size', 5)))
        cluster_min_samples = (min_samples if min_samples is not None 
                              else clustering_config.get('min_samples', 
                                   pipeline_config.get('default_min_samples', 3)))
        
        self.clusterer = HDBSCANClusterer(
            min_cluster_size=cluster_min_size,
            min_samples=cluster_min_samples,
            config_path=config_path
        )
        # Non serve più classifier separato - tutto nell'ensemble
        # self.classifier = rimosso, ora tutto in ensemble_classifier
        self.tag_db = TagDatabaseConnector()
        
        # Inizializza il gestore della memoria semantica
        print("🧠 Inizializzazione memoria semantica...")
        self.semantic_memory = SemanticMemoryManager(
            config_path=config_path,
            embedder=self.embedder
        )
        
        # Inizializza l'ensemble classifier avanzato PRIMA (questo creerà il suo LLM internamente)
        print("🔗 Inizializzazione ensemble classifier avanzato...")
        self.ensemble_classifier = AdvancedEnsembleClassifier(
            llm_classifier=None,  # Creerà il suo IntelligentClassifier internamente
            confidence_threshold=self.confidence_threshold,
            adaptive_weights=True,
            performance_tracking=True,
            client_name=self.tenant_slug  # Passa il nome del tenant come client_name per fine-tuning
        )
        
        # Prova a caricare l'ultimo modello ML salvato
        self._try_load_latest_model()
        
        # Se esiste un provider BERTopic accoppiato al modello caricato, inietta nell'ensemble
        # (gestito dentro _try_load_latest_model)
        
        # Riaddestramento automatico gestito dal QualityGate quando necessario
        if self.auto_retrain:
            print("🔄 Riaddestramento automatico abilitato")
            print("   🎯 Il training verrà gestito dal QualityGate quando necessario")
            print("   📊 Training automatico ogni 5 decisioni umane (config: quality_gate.retraining_threshold)")
        else:
            print("⏸️ Riaddestramento automatico disabilitato (modalità supervisione/API)")
            print("   💡 Per abilitare, usare auto_retrain=True nell'inizializzazione")
        
        # Recupera il classificatore LLM dall'ensemble per gli altri componenti
        llm_classifier = self.ensemble_classifier.llm_classifier
        if llm_classifier and hasattr(llm_classifier, 'is_available') and llm_classifier.is_available():
            print("✅ Classificatore LLM disponibile nell'ensemble")
        else:
            print("⚠️ Classificatore LLM non disponibile, ensemble userà solo ML")
        
        # Inizializza il trainer interattivo
        print("👤 Inizializzazione trainer interattivo...")
        self.interactive_trainer = InteractiveTrainer(llm_classifier=llm_classifier, auto_mode=self.auto_mode)
        
        
        # Inizializza il dedupplicatore intelligente di etichette
        print("🧠 Inizializzazione dedupplicatore intelligente...")
        self.label_deduplicator = IntelligentLabelDeduplicator(
            embedder=self.embedder,
            llm_classifier=llm_classifier,
            semantic_memory=self.semantic_memory,
            similarity_threshold=0.85,
            llm_confidence_threshold=0.7
        )
        
        # Carica la memoria semantica esistente
        if self.semantic_memory.load_semantic_memory():
            stats = self.semantic_memory.get_memory_stats()
            print(f"✅ Memoria semantica caricata: {stats.get('memory_sessions', 0)} campioni, {stats.get('total_tags', 0)} tag")
        else:
            print("⚠️ Memoria semantica inizializzata vuota")
        
        print("✅ Pipeline inizializzata!")
    
    @property
    def intelligent_classifier(self):
        """
        Accesso diretto all'IntelligentClassifier per gestione fine-tuning
        
        Returns:
            IntelligentClassifier instance or None
        """
        if (self.ensemble_classifier and 
            hasattr(self.ensemble_classifier, 'llm_classifier') and
            self.ensemble_classifier.llm_classifier):
            return self.ensemble_classifier.llm_classifier
        return None
    
    def estrai_sessioni(self, 
                       giorni_indietro: int = 30,
                       limit: Optional[int] = None,
                       force_full_extraction: bool = False) -> Dict[str, Dict]:
        """
        Estrae le sessioni dal database remoto
        
        Args:
            giorni_indietro: Numero di giorni indietro da cui estrarre sessioni
            limit: Limite massimo di sessioni da estrarre (ignorato se force_full_extraction=True)
            force_full_extraction: Se True, estrae TUTTE le sessioni ignorando il limit
            
        Returns:
            Dizionario con le sessioni estratte
        """
        print(f"📊 Estrazione sessioni per {self.tenant_slug}...")
        
        # Controlla configurazione training supervisionato
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            supervised_config = config.get('supervised_training', {})
            extraction_config = supervised_config.get('extraction', {})
            
            # Se configurazione prevede estrazione completa, forza estrazione totale
            if extraction_config.get('use_full_dataset', False) or force_full_extraction:
                print(f"🔄 MODALITÀ ESTRAZIONE COMPLETA ATTIVATA")
                print(f"� Ignorando limite sessioni - estrazione di TUTTO il dataset")
                actual_limit = None
                extraction_mode = "COMPLETA"
            else:
                actual_limit = limit
                extraction_mode = "LIMITATA"
                
        except Exception as e:
            print(f"⚠️ Errore lettura config supervised_training: {e}")
            actual_limit = limit
            extraction_mode = "LIMITATA"
        
        print(f"�🔍 Parametri: giorni_indietro={giorni_indietro}, limit_originale={limit}, limit_effettivo={actual_limit}")
        print(f"📊 Modalità estrazione: {extraction_mode}")
        
        # Per ora estraiamo con limit, il filtro temporale può essere aggiunto successivamente
        print(f"🔄 Chiamata aggregator.estrai_sessioni_aggregate...")
        sessioni = self.aggregator.estrai_sessioni_aggregate(limit=actual_limit)
        
        if not sessioni:
            print("❌ Nessuna sessione trovata")
            return {}
        
        print(f"📥 Sessioni grezze estratte: {len(sessioni)}")
        
        # Filtra sessioni vuote
        print(f"🔍 Filtraggio sessioni vuote/irrilevanti...")
        sessioni_filtrate = self.aggregator.filtra_sessioni_vuote(sessioni)
        
        if extraction_mode == "COMPLETA":
            print(f"✅ ESTRAZIONE COMPLETA: {len(sessioni_filtrate)} sessioni totali dal database")
            print(f"🎯 Tutte le discussioni disponibili sono state estratte per clustering completo")
        else:
            print(f"✅ Estrazione limitata: {len(sessioni_filtrate)} sessioni valide")
            
        return sessioni_filtrate
    
    def esegui_clustering(self, sessioni: Dict[str, Dict]) -> tuple:
        """
        Esegue il clustering delle sessioni con approccio intelligente multi-livello:
        1. LLM per comprensione linguaggio naturale (primario)
        2. Pattern regex per fallback veloce (secondario)  
        3. Validazione umana per casi ambigui (terziario)
        
        Args:
            sessioni: Dizionario con le sessioni
            
        Returns:
            Tuple (embeddings, cluster_labels, representatives, suggested_labels)
        """
        print(f"🧩 Clustering intelligente di {len(sessioni)} sessioni...")
        
        # Genera embedding
        print(f"🔍 Encoding {len(sessioni)} testi...")
        testi = [dati['testo_completo'] for dati in sessioni.values()]
        embeddings = self.embedder.encode(testi, show_progress_bar=True)
        print(f"✅ Embedding generati: shape {embeddings.shape}")
        
        # Carica configurazione clustering
        with open(self.clusterer.config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Verifica quale approccio usare (priorità assoluta al sistema intelligente)
        intelligent_config = config.get('intelligent_clustering', {})
        hierarchical_config = config.get('hierarchical_clustering', {})
        
        # NUOVA STRATEGIA: Clustering Gerarchico Adattivo (massima priorità)
        if hierarchical_config.get('enabled', False):
            print(f"🌳 Usando clustering GERARCHICO ADATTIVO (gestione automatica conflitti)")
            
            hierarchical_clusterer = HierarchicalAdaptiveClusterer(
                config_path=self.clusterer.config_path,
                llm_classifier=self.ensemble_classifier.llm_classifier if self.ensemble_classifier else None,
                confidence_threshold=hierarchical_config.get('confidence_threshold', 0.75),
                boundary_threshold=hierarchical_config.get('boundary_threshold', 0.45),
                max_hierarchy_depth=hierarchical_config.get('max_hierarchy_depth', 3)
            )
            
            # Prepara dati per il clustering gerarchico
            session_ids = list(sessioni.keys())
            
            # Esegue clustering gerarchico con risoluzione automatica conflitti
            session_memberships, cluster_info = hierarchical_clusterer.cluster_hierarchically(
                testi, embeddings, session_ids, 
                max_iterations=hierarchical_config.get('max_iterations', 3)
            )
            
            # Debug: Analizza impatto LLM e fallback
            llm_impact = hierarchical_clusterer.get_llm_impact_analysis()
            print(f"\n🔍 ANALISI IMPATTO LLM:")
            print(f"   Tasso successo: {llm_impact['successful_rate']:.1%}")
            print(f"   Tasso fallback: {llm_impact['fallback_rate']:.1%}")
            print(f"   Rischio: {llm_impact['risk_level']}")
            print(f"   {llm_impact['impact_description']}")
            
            if llm_impact['risk_level'] != 'low':
                print(f"\n💡 RACCOMANDAZIONI:")
                for rec in llm_impact['recommendations']:
                    print(f"   - {rec}")
            
            # Analizza impatto specifico del fallback 'altro'
            altro_impact = hierarchical_clusterer.analyze_altro_impact()
            
            # Converte session_memberships in cluster_labels per compatibilità
            cluster_labels = self._convert_hierarchical_to_labels(session_memberships, session_ids)
            
            # Salva stato gerarchico per analisi future
            self._last_hierarchical_clusterer = hierarchical_clusterer
            
        elif intelligent_config.get('enabled', True):  # Default TRUE per sistema intelligente
            print(f"🧠 Usando clustering INTELLIGENTE (LLM + ML senza pattern)")
            
            # USA SOLO IL SISTEMA INTELLIGENTE PURO
            intelligent_clusterer = IntelligentIntentClusterer(
                config_path=self.clusterer.config_path,
                llm_classifier=self.ensemble_classifier.llm_classifier if self.ensemble_classifier else None
            )
            cluster_labels, cluster_info = intelligent_clusterer.cluster_intelligently(testi, embeddings)
            cluster_labels = np.array(cluster_labels)
            
        # RIMOSSO: Intent-based clusterer con pattern rigidi (eliminato per approccio ML+LLM puro)
        # elif intent_config.get('enabled', True): # SISTEMA LEGACY RIMOSSO
            
        else:
            print(f"🔄 Fallback: clustering HDBSCAN geometrico puro")
            # Fallback geometrico senza semantica
            cluster_labels = self.clusterer.fit_predict(embeddings)
            # Fallback finale al clustering HDBSCAN normale
            cluster_labels = self.clusterer.fit_predict(embeddings)
            
            # Crea cluster_info di base per compatibilità
            cluster_info = {}
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                if label != -1:
                    indices = [i for i, l in enumerate(cluster_labels) if l == label]
                    cluster_info[label] = {
                        'intent': 'cluster_hdbscan',
                        'size': len(indices),
                        'indices': indices,
                        'intent_string': f'Cluster HDBSCAN {label}',
                        'classification_method': 'hdbscan'
                    }
        
        # Genera rappresentanti per ogni cluster (selezione intelligente)
        representatives = {}
        suggested_labels = {}
        session_ids = list(sessioni.keys())
        
        for cluster_id, info in cluster_info.items():
            # Trova rappresentanti per questo cluster
            cluster_indices = info['indices']
            cluster_representatives = []
            
            # Selezione intelligente dei rappresentanti (più diversi possibile)
            if len(cluster_indices) <= 3:
                selected_indices = cluster_indices
            else:
                # Seleziona 3 rappresentanti più diversi usando distanza embedding
                cluster_embeddings = embeddings[cluster_indices]
                from sklearn.metrics.pairwise import cosine_distances
                distances = cosine_distances(cluster_embeddings)
                
                # Trova i 3 punti più distanti tra loro
                selected_indices = [cluster_indices[0]]  # Primo punto
                
                for _ in range(min(2, len(cluster_indices) - 1)):
                    max_min_dist = -1
                    best_idx = -1
                    
                    for idx in cluster_indices:
                        if idx not in selected_indices:
                            min_dist_to_selected = min(
                                distances[cluster_indices.index(idx)][cluster_indices.index(sel_idx)]
                                for sel_idx in selected_indices
                            )
                            if min_dist_to_selected > max_min_dist:
                                max_min_dist = min_dist_to_selected
                                best_idx = idx
                    
                    if best_idx != -1:
                        selected_indices.append(best_idx)
            
            for idx in selected_indices:
                session_id = session_ids[idx]
                session_data = sessioni[session_id].copy()
                session_data['session_id'] = session_id
                
                # Aggiungi info sulla classificazione se disponibile
                if 'average_confidence' in info:
                    session_data['classification_confidence'] = info['average_confidence']
                if 'classification_method' in info:
                    session_data['classification_method'] = info['classification_method']
                    
                cluster_representatives.append(session_data)
            
            representatives[cluster_id] = cluster_representatives
            
            # Genera etichetta suggerita basata su intent
            if 'intent_string' in info:
                suggested_labels[cluster_id] = info['intent_string']
            elif 'intent' in info:
                suggested_labels[cluster_id] = info['intent'].replace('_', ' ').title()
            else:
                # Fallback generico
                suggested_labels[cluster_id] = f"Cluster {cluster_id}"
        
        # Statistiche clustering avanzate
        n_clusters = len(cluster_info)
        n_outliers = sum(1 for label in cluster_labels if label == -1)
        
        # Calcola statistiche di confidenza se disponibili
        avg_confidence = 0.0
        high_confidence_clusters = 0
        
        for info in cluster_info.values():
            if 'average_confidence' in info:
                avg_confidence += info['average_confidence']
                if info['average_confidence'] > 0.8:
                    high_confidence_clusters += 1
        
        if n_clusters > 0:
            avg_confidence /= n_clusters
        
        print(f"✅ Clustering intelligente completato!")
        print(f"  📊 Cluster trovati: {n_clusters}")
        print(f"  🔍 Outliers: {n_outliers}")
        print(f"  🏷️  Etichette suggerite: {len(suggested_labels)}")
        print(f"  🎯 Confidenza media: {avg_confidence:.2f}")
        print(f"  🌟 Cluster alta confidenza: {high_confidence_clusters}/{n_clusters}")
        
        # Mostra dettagli dei cluster trovati
        for cluster_id, info in cluster_info.items():
            intent_string = info.get('intent_string', info.get('intent', 'N/A'))
            confidence = info.get('average_confidence', 0.0)
            method = info.get('classification_method', 'unknown')
            print(f"    - Cluster {cluster_id}: {intent_string} ({info['size']} conv., conf: {confidence:.2f}, metodo: {method})")
        
        # Applica dedupplicazione intelligente delle etichette
        if suggested_labels:
            suggested_labels = self.label_deduplicator.prevent_duplicate_labels(suggested_labels)
        
        return embeddings, cluster_labels, representatives, suggested_labels
    
    def allena_classificatore(self,
                            sessioni: Dict[str, Dict],
                            cluster_labels: np.ndarray,
                            representatives: Dict[int, List[Dict]],
                            suggested_labels: Dict[int, str],
                            interactive_mode: bool = True) -> Dict[str, Any]:
        """
        Allena il classificatore supervisionato con review umano interattivo
        
        Args:
            sessioni: Sessioni di training
            cluster_labels: Etichette dei cluster
            representatives: Rappresentanti dei cluster
            suggested_labels: Etichette suggerite
            interactive_mode: Se True, abilita la review umana interattiva
            
        Returns:
            Metriche di training
        """
        print(f"🎓 Training del classificatore...")
        print(f"📊 Sessioni da processare: {len(sessioni)}")
        print(f"🏷️  Etichette suggerite: {len(suggested_labels)}")
        print(f"👤 Modalità interattiva: {interactive_mode}")
        
        # Verifica se ci sono cluster validi
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"📈 Cluster validi trovati: {n_clusters}")
        
        if n_clusters == 0:
            print("⚠️ Nessun cluster trovato. Uso classificazione basata su tag predefiniti...")
            return self._allena_classificatore_fallback(sessioni)
        
        # Se modalità interattiva è abilitata, esegui review umano
        if interactive_mode and n_clusters > 0:
            print(f"\n👤 MODALITÀ INTERATTIVA ABILITATA")
            print(f"📊 Trovati {n_clusters} cluster da revieware")
            print("⏭️ Procedendo automaticamente con review interattiva abilitata")
        
        # Esegui review interattivo se abilitato
        reviewed_labels = suggested_labels.copy()
        print(f"🔍 Labels dopo review: {len(reviewed_labels)}")
        
        if interactive_mode:
            print(f"\n🔍 INIZIO REVIEW INTERATTIVO")
            
            # Review di ogni cluster
            for cluster_id in sorted(suggested_labels.keys()):
                if cluster_id in representatives:
                    suggested_label = suggested_labels[cluster_id]
                    cluster_reps = representatives[cluster_id]
                    
                    # Review umano del cluster
                    final_label, human_confidence = self.interactive_trainer.review_cluster_representatives(
                        cluster_id=cluster_id,
                        representatives=cluster_reps,
                        suggested_label=suggested_label
                    )
                    
                    # Aggiorna l'etichetta se diversa
                    if final_label != suggested_label:
                        reviewed_labels[cluster_id] = final_label
                        print(f"🔄 Cluster {cluster_id}: '{suggested_label}' → '{final_label}'")
            
            print(f"\n✅ Review interattivo completato")
        
        # Prepara dati di training con le etichette reviewed
        # Ora usiamo direttamente embeddings e labels per l'ensemble
        session_ids = list(sessioni.keys())
        session_texts = [sessioni[sid]['testo_completo'] for sid in session_ids]
        train_embeddings = self.embedder.encode(session_texts)
        
        # Crea labels array dai reviewed_labels
        train_labels = []
        for i, session_id in enumerate(session_ids):
            # Trova il cluster di questa sessione
            cluster_id = cluster_labels[i] if i < len(cluster_labels) else -1
            if cluster_id in reviewed_labels:
                train_labels.append(reviewed_labels[cluster_id])
            else:
                train_labels.append('altro')  # fallback
        
        train_labels = np.array(train_labels)
        
        # Verifica che ci siano abbastanza dati per ML training
        unique_train_labels = set(train_labels)
        if len(train_labels) < 2 or len(unique_train_labels) < 2:
            print(f"⚠️ Insufficienti dati per training ML: {len(train_labels)} campioni, {len(unique_train_labels)} classi")
            print(f"🔄 Saltando training ML, usando solo LLM per classificazioni future")
            
            # Salva informazioni cluster per ottimizzazione futura
            self._last_cluster_labels = cluster_labels
            self._last_representatives = representatives
            self._last_reviewed_labels = reviewed_labels
            print(f"💾 Salvate informazioni cluster per ottimizzazione futura")
            
            return {
                'training_accuracy': 0.0,
                'n_samples': len(train_labels),
                'n_features': train_embeddings.shape[1] if len(train_embeddings) > 0 else 0,
                'n_classes': len(unique_train_labels),
                'interactive_review': interactive_mode,
                'skip_reason': 'insufficient_data_for_ml_training',
                'ensemble_mode': 'llm_only'
            }
        
        if len(train_embeddings) < 5:
            print("⚠️ Troppo pochi dati dal clustering. Uso classificazione basata su tag predefiniti...")
            return self._allena_classificatore_fallback(sessioni)
        
        # Augment features con BERTopic se abilitato
        ml_features = train_embeddings
        bertopic_provider = None
        if self.bertopic_config.get('enabled', False):
            if not _BERTopic_AVAILABLE:
                print("⚠️ BERTopic abilitato da config ma dipendenze non installate; proseguo senza augmentation")
            else:
                try:
                    print("🌐 BERTopic attivo: fit+transform per feature augmentation...")
                    bertopic_provider = BERTopicFeatureProvider(
                        use_svd=self.bertopic_config.get('use_svd', False),
                        svd_components=self.bertopic_config.get('svd_components', 32)
                    )
                    bertopic_provider.fit(session_texts, embeddings=train_embeddings)
                    tr = bertopic_provider.transform(
                        session_texts,
                        embeddings=train_embeddings,
                        return_one_hot=self.bertopic_config.get('return_one_hot', False),
                        top_k=self.bertopic_config.get('top_k', None)
                    )
                    topic_probas = tr.get('topic_probas')
                    one_hot = tr.get('one_hot')
                    parts = [train_embeddings]
                    if topic_probas is not None and topic_probas.size > 0:
                        parts.append(topic_probas)
                    if one_hot is not None and one_hot.size > 0:
                        parts.append(one_hot)
                    ml_features = np.concatenate(parts, axis=1)
                    # Inietta provider nell'ensemble per coerenza in inference
                    self.ensemble_classifier.set_bertopic_provider(
                        bertopic_provider,
                        top_k=self.bertopic_config.get('top_k', 15),
                        return_one_hot=self.bertopic_config.get('return_one_hot', False)
                    )
                    print(f"✅ Feature ML con BERTopic: {train_embeddings.shape[1]} -> {ml_features.shape[1]}")
                except Exception as e:
                    print(f"⚠️ Errore BERTopic augmentation: {e}; proseguo con sole embeddings")
                    bertopic_provider = None
        
        # Allena l'ensemble ML con le feature (augmentate o raw)
        print("🎓 Training ensemble ML avanzato...")
        metrics = self.ensemble_classifier.train_ml_ensemble(ml_features, train_labels)
        
        # Salva il modello ensemble e l'eventuale provider BERTopic
        model_name = f"{self.tenant_slug}_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.ensemble_classifier.save_ensemble_model(f"models/{model_name}")
        
        if bertopic_provider is not None:
            try:
                provider_dir = os.path.join("models", f"{model_name}_{self.bertopic_config.get('model_subdir', 'bertopic')}")
                bertopic_provider.save(provider_dir)
                print(f"💾 BERTopic provider salvato in {provider_dir}")
            except Exception as e:
                print(f"⚠️ Errore nel salvataggio del BERTopic provider: {e}")
        
        # Aggiungi statistiche del review interattivo
        if interactive_mode:
            feedback_stats = self.interactive_trainer.get_feedback_summary()
            metrics.update({
                'interactive_review': True,
                'reviewed_clusters': len(reviewed_labels),
                'human_feedback_stats': feedback_stats
            })
        else:
            metrics['interactive_review'] = False
        
        print(f"✅ Classificatore allenato e salvato come '{model_name}'")
        return metrics
    
    def _allena_classificatore_fallback(self, sessioni: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Allena il classificatore usando tag predefiniti come fallback
        """
        print("🔄 Uso approccio fallback con tag predefiniti...")
        
        # Tag predefiniti e parole chiave associate (già normalizzati)
        tag_keywords = {
            'accesso_portale': ['accesso', 'login', 'password', 'entrare', 'portale', 'app'],
            'prenotazione_esami': ['prenota', 'prenotazione', 'visita', 'esame', 'appuntamento'],
            'ritiro_referti': ['referto', 'risultato', 'ritiro', 'ritirar', 'analisi'],
            'problemi_prenotazione': ['problema', 'errore', 'cancella', 'modificar', 'disdire'],
            'fatturazione': ['fattura', 'pagamento', 'ricevuta', 'costo', 'prezzo'],
            'orari_strutture': ['orari', 'orario', 'aperto', 'chiuso', 'quando', 'apertura'],
            'contatti_info': ['telefono', 'contatto', 'numero', 'chiamare', 'email'],
            'altro': []  # Default per tutto il resto
        }
        
        # Assegna etichette basate su parole chiave
        session_texts = []
        session_labels = []
        session_ids = []
        
        for session_id, dati in sessioni.items():
            testo = dati['testo_completo'].lower()
            # Trova il tag più appropriato
            best_tag = 'altro'
            max_matches = 0
            
            for tag, keywords in tag_keywords.items():
                if tag == 'altro':
                    continue
                    
                matches = sum(1 for keyword in keywords if keyword in testo)
                if matches > max_matches:
                    max_matches = matches
                    best_tag = tag
            
            # Se nessun match significativo, usa classificazione avanzata
            if max_matches == 0:
                # Fallback: assegna etichette diverse per forzare diversità
                if len(session_labels) == 0:
                    best_tag = 'info_generali'
                elif len(session_labels) == 1 and session_labels[0] == 'info_generali':
                    best_tag = 'richiesta_operatore'
                else:
                    best_tag = 'altro'
            
            session_texts.append(dati['testo_completo'])
            session_labels.append(best_tag)
            session_ids.append(session_id)
        
        # Genera embedding
        embeddings = self.embedder.encode(session_texts)
        
        # Converte etichette in array
        labels_array = np.array(session_labels)
        
        # Verifica che ci siano almeno 2 classi diverse
        unique_labels = set(session_labels)
        if len(unique_labels) < 2:
            # Se abbiamo solo una sessione, non possiamo fare training ML
            if len(session_labels) == 1:
                print(f"⚠️ Solo 1 sessione disponibile, impossibile training ML")
                print(f"🔄 Saltando training e usando solo LLM per classificazione")
                return {
                    'training_accuracy': 0.0,
                    'n_samples': len(session_labels),
                    'n_features': 0,
                    'n_classes': 1,
                    'fallback_reason': 'insufficient_samples_for_ml'
                }
            
            # Se abbiamo più sessioni ma tutte con la stessa classe, forza diversità
            if 'altro' not in unique_labels:
                session_labels[0] = 'altro'
            if 'informazioni_generali' not in unique_labels and len(session_labels) > 1:
                session_labels[1] = 'informazioni_generali'
            labels_array = np.array(session_labels)
            unique_labels = set(session_labels)
        
        # Se ancora abbiamo solo 1 classe, usa solo LLM
        if len(unique_labels) < 2:
            print(f"⚠️ Impossibile creare diversità di classi, uso solo LLM")
            return {
                'training_accuracy': 0.0,
                'n_samples': len(session_labels),
                'n_features': embeddings.shape[1],
                'n_classes': len(unique_labels),
                'fallback_reason': 'insufficient_class_diversity'
            }
        
        # Allena l'ensemble classifier
        metrics = self.ensemble_classifier.train_ml_ensemble(embeddings, labels_array)
        
        # Salva il modello ensemble
        model_name = f"{self.tenant_slug}_fallback_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.ensemble_classifier.save_ensemble_model(f"models/{model_name}")
        
        print(f"✅ Classificatore fallback allenato e salvato come '{model_name}'")
        print(f"📊 Classi usate: {sorted(unique_labels)}")
        
        return metrics
    
    def classifica_e_salva_sessioni(self,
                                   sessioni: Dict[str, Dict],
                                   batch_size: int = 32,
                                   use_ensemble: bool = True,
                                   optimize_clusters: bool = True) -> Dict[str, Any]:
        """
        Classifica le sessioni usando l'ensemble classifier e salva i risultati nel database TAG
        
        Args:
            sessioni: Sessioni da classificare
            batch_size: Dimensione del batch per la classificazione
            use_ensemble: Se True, usa l'ensemble classifier, altrimenti solo ML
            optimize_clusters: Se True, classifica solo rappresentanti e propaga le etichette
            
        Returns:
            Statistiche della classificazione
        """
        print(f"🏷️  Classificazione e salvataggio di {len(sessioni)} sessioni...")
        print(f"📊 Batch size: {batch_size}, Use ensemble: {use_ensemble}, Optimize clusters: {optimize_clusters}")
        
        if use_ensemble:
            print(f"🔗 Usando ensemble classifier (LLM + ML)")
        else:
            print(f"🤖 Usando solo classificatore ML (parte dell'ensemble)")
            # Ora tutto è gestito dall'ensemble, non serve controllo separato
        
        # Connetti al database TAG
        print(f"💾 Connessione al database TAG...")
        self.tag_db.connetti()
        
        # Prepara dati per classificazione
        session_ids = list(sessioni.keys())
        session_texts = [sessioni[sid]['testo_completo'] for sid in session_ids]
        print(f"📦 Preparati {len(session_texts)} testi per classificazione")
        
        # Classificazione ottimizzata per cluster o tradizionale
        if optimize_clusters and use_ensemble and self.ensemble_classifier:
            # Usa classificazione ottimizzata basata sui cluster
            print(f"🎯 Classificazione ottimizzata per cluster in corso...")
            predictions = self._classifica_ottimizzata_cluster(sessioni, session_ids, session_texts, batch_size)
        elif use_ensemble and self.ensemble_classifier:
            # Usa advanced ensemble classifier con batch prediction tradizionale
            print(f"🔍 Classificazione ensemble avanzata in corso...")
            print(f"📦 Batch prediction di {len(session_texts)} testi...")
            
            try:
                # Usa batch prediction per efficienza con embedder riutilizzabile
                batch_predictions = self.ensemble_classifier.batch_predict(
                    session_texts, 
                    batch_size=batch_size,
                    embedder=self.embedder  # Passa l'embedder esistente per evitare CUDA OOM
                )
                predictions = batch_predictions
                print(f"✅ Batch prediction completata: {len(predictions)} risultati")
                
            except Exception as e:
                print(f"⚠️ Errore nella classificazione ensemble: {e}")
                print(f"🔄 Fallback alla classificazione singola...")
                # Fallback alla singola predizione
                predictions = []
                for i, text in enumerate(session_texts):
                    print(f"🔍 Classificando sessione {i+1}/{len(session_texts)}...")
                    try:
                        prediction = self.ensemble_classifier.predict_with_ensemble(
                            text, 
                            return_details=True, 
                            embedder=self.embedder  # Passa l'embedder esistente
                        )
                        predictions.append(prediction)
                    except Exception as e2:
                        print(f"⚠️ Fallback completo non riuscito: {e2}")
                        # Fallback finale: predizione base
                        ml_prediction = {'predicted_label': 'altro', 'confidence': 0.1}
                        predictions.append({
                            'predicted_label': ml_prediction['predicted_label'],
                            'confidence': ml_prediction['confidence'],
                            'is_high_confidence': False,
                            'method': 'FALLBACK_FINAL',
                            'llm_prediction': None,
                            'ml_prediction': ml_prediction,
                            'ensemble_confidence': ml_prediction['confidence']
                        })
            # RIMOSSO: predictions.extend(batch_predictions)  # batch_predictions non definita in caso di eccezione
        else:
            # Usa ensemble anche per "solo ML" (l'ensemble può disabilitare LLM)
            predictions = []
            for text in session_texts:
                pred = self.ensemble_classifier.predict_with_ensemble(
                    text, 
                    return_details=True,
                    embedder=self.embedder  # Passa l'embedder esistente
                ) 
                pred['method'] = 'ML_ONLY'
                predictions.append(pred)
        
        # Salva classificazioni nel database
        stats = {
            'total_sessions': len(sessioni),
            'high_confidence': 0,
            'low_confidence': 0,
            'saved_successfully': 0,
            'save_errors': 0,
            'classifications_by_tag': {},
            'ensemble_stats': {
                'llm_predictions': 0,
                'ml_predictions': 0,
                'ensemble_agreements': 0,
                'ensemble_disagreements': 0
            } if use_ensemble else None
        }
        
        print(f"💾 Inizio salvataggio di {len(predictions)} classificazioni...")
        for i, (session_id, prediction) in enumerate(zip(session_ids, predictions)):
            if (i + 1) % 10 == 0:  # Debug ogni 10 classificazioni
                print(f"📊 Progresso salvataggio: {i+1}/{len(predictions)} ({((i+1)/len(predictions)*100):.1f}%)")
            
            try:
                # Determina metodo di classificazione
                if use_ensemble:
                    method = prediction.get('method', 'ENSEMBLE')
                    # Accorcia i metodi per il database
                    if method == 'ENSEMBLE':
                        method = 'ENS'
                    elif method == 'LLM':
                        method = 'LLM'
                    elif method == 'ML':
                        method = 'ML'
                    elif method == 'FALLBACK_FINAL':
                        method = 'FALL'
                    
                    confidence = prediction.get('ensemble_confidence', prediction['confidence'])
                    predicted_label = prediction['predicted_label']
                    
                    # Debug delle classificazioni interessanti
                    if i < 5 or (i + 1) % 50 == 0:  # Prime 5 e ogni 50
                        print(f"🏷️  Sessione {i+1}: '{predicted_label}' (conf: {confidence:.3f}, metodo: {method})")
                    
                    # Aggiorna statistiche ensemble
                    if 'llm_prediction' in prediction and prediction['llm_prediction']:
                        stats['ensemble_stats']['llm_predictions'] += 1
                    if 'ml_prediction' in prediction and prediction['ml_prediction']:
                        stats['ensemble_stats']['ml_predictions'] += 1
                    
                    # Verifica accordo tra LLM e ML
                    if (prediction.get('llm_prediction') and prediction.get('ml_prediction') and
                        prediction['llm_prediction']['predicted_label'] == prediction['ml_prediction']['predicted_label']):
                        stats['ensemble_stats']['ensemble_agreements'] += 1
                    elif prediction.get('llm_prediction') and prediction.get('ml_prediction'):
                        stats['ensemble_stats']['ensemble_disagreements'] += 1
                else:
                    method = 'ML_AUTO'
                    confidence = prediction['confidence']
                    predicted_label = prediction['predicted_label']
                
                # Determina se è alta confidenza
                is_high_confidence = confidence >= self.confidence_threshold
                
                # Salva nel database TAG
                success = self.tag_db.classifica_sessione(
                    session_id=session_id,
                    tag_name=predicted_label,
                    tenant_slug=self.tenant_slug,
                    confidence_score=confidence,
                    method=method,
                    classified_by='ens_pipe' if use_ensemble else 'ml_pipe',
                    notes=f"Auto-classificato con confidenza {confidence:.3f}"
                )
                
                if success:
                    stats['saved_successfully'] += 1
                    
                    # Aggiorna statistiche
                    tag_name = prediction['predicted_label']
                    if tag_name not in stats['classifications_by_tag']:
                        stats['classifications_by_tag'][tag_name] = 0
                    stats['classifications_by_tag'][tag_name] += 1
                    
                else:
                    stats['save_errors'] += 1
                
                # Aggiorna contatori confidenza
                if is_high_confidence:
                    stats['high_confidence'] += 1
                else:
                    stats['low_confidence'] += 1
                    
            except Exception as e:
                print(f"❌ Errore nel salvataggio della sessione {session_id}: {e}")
                stats['save_errors'] += 1
        
        self.tag_db.disconnetti()
        
        print(f"✅ Classificazione completata!")
        print(f"  💾 Salvate: {stats['saved_successfully']}/{stats['total_sessions']}")
        print(f"  🎯 Alta confidenza: {stats['high_confidence']}")
        print(f"  ⚠️  Bassa confidenza: {stats['low_confidence']}")
        print(f"  ❌ Errori: {stats['save_errors']}")
        
        if use_ensemble and stats['ensemble_stats']:
            ens = stats['ensemble_stats']
            print(f"  🔗 Ensemble: {ens['llm_predictions']} LLM + {ens['ml_predictions']} ML")
            print(f"  🤝 Accordi: {ens['ensemble_agreements']}, Disaccordi: {ens['ensemble_disagreements']}")
        
        return stats
    
    def esegui_pipeline_completa(self,
                               giorni_indietro: int = 7,
                               limit: Optional[int] = 200,
                               batch_size: int = 32,
                               interactive_mode: bool = True,
                               use_ensemble: bool = True,
                               force_full_extraction: bool = False) -> Dict[str, Any]:
        """
        Esegue la pipeline completa end-to-end con review interattivo e ensemble classifier
        
        Args:
            giorni_indietro: Giorni di dati da processare
            limit: Limite massimo di sessioni per estrazione (ignorato se force_full_extraction=True)
            batch_size: Dimensione batch per classificazione
            interactive_mode: Se True, abilita la review umana interattiva
            use_ensemble: Se True, usa l'ensemble classifier (LLM + ML)
            force_full_extraction: Se True, estrae tutto il dataset ignorando il limit
            
        Returns:
            Risultati completi della pipeline
        """
        start_time = datetime.now()
        print(f"🚀 AVVIO PIPELINE END-TO-END")
        print(f"📅 Periodo: ultimi {giorni_indietro} giorni")
        print(f"🏥 Tenant: {self.tenant_slug}")
        
        # Determina modalità estrazione
        extraction_mode = "COMPLETA" if force_full_extraction else "LIMITATA"
        effective_limit = None if force_full_extraction else limit
        
        print(f"🔢 Limite sessioni: {limit or 'Nessuno'}")
        print(f"📊 Modalità estrazione: {extraction_mode}")
        print(f"👤 Review interattivo: {'Abilitato' if interactive_mode else 'Disabilitato'}")
        print(f"🔗 Ensemble classifier: {'Abilitato' if use_ensemble else 'Disabilitato'}")
        print("-" * 50)
        
        try:
            # 1. Estrazione sessioni
            sessioni = self.estrai_sessioni(giorni_indietro=giorni_indietro, 
                                           limit=effective_limit,
                                           force_full_extraction=force_full_extraction)
            
            if len(sessioni) < 3:
                raise ValueError(f"Troppo poche sessioni ({len(sessioni)}) per qualsiasi analisi")
            elif len(sessioni) < 10:
                print(f"⚠️ Attenzione: Solo {len(sessioni)} sessioni trovate. Risultati potrebbero essere limitati.")
            
            # 2. Clustering
            embeddings, cluster_labels, representatives, suggested_labels = self.esegui_clustering(sessioni)
            
            # 3. Training classificatore con review interattivo
            training_metrics = self.allena_classificatore(
                sessioni, cluster_labels, representatives, suggested_labels, 
                interactive_mode=interactive_mode
            )
            
            # 4. Classificazione usando ensemble o ML singolo
            if use_ensemble:
                classification_stats = self.classifica_e_salva_sessioni(
                    sessioni, batch_size=batch_size, use_ensemble=True
                )
            else:
                classification_stats = self.classifica_e_salva_sessioni(
                    sessioni, batch_size=batch_size, use_ensemble=False
                )
            
            # 5. Aggiorna memoria semantica con nuove classificazioni
            print(f"🧠 Aggiornamento memoria semantica...")
            memory_update_stats = self._update_semantic_memory_with_classifications(
                sessioni, classification_stats
            )
            
            # 6. Statistiche finali
            end_time = datetime.now()
            duration = end_time - start_time
            
            results = {
                'pipeline_info': {
                    'tenant_slug': self.tenant_slug,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_seconds': duration.total_seconds(),
                    'giorni_processati': giorni_indietro,
                    'interactive_mode': interactive_mode,
                    'ensemble_mode': use_ensemble
                },
                'extraction': {
                    'total_sessions': len(sessioni)
                },
                'clustering': {
                    'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                    'n_outliers': sum(1 for label in cluster_labels if label == -1),
                    'suggested_labels': len(suggested_labels)
                },
                'training_metrics': training_metrics,
                'classification_stats': classification_stats,
                'memory_update_stats': memory_update_stats
            }
            
            print("-" * 50)
            print(f"✅ PIPELINE COMPLETATA CON SUCCESSO!")
            print(f"⏱️  Durata: {duration.total_seconds():.1f} secondi")
            print(f"📊 Sessioni processate: {len(sessioni)}")
            print(f"🎯 Accuracy classificatore: {training_metrics.get('accuracy', 0):.3f}")
            print(f"💾 Classificazioni salvate: {classification_stats['saved_successfully']}")
            
            if interactive_mode and 'human_feedback_stats' in training_metrics:
                feedback = training_metrics['human_feedback_stats']
                print(f"👤 Feedback umano: {feedback.get('total_reviews', 0)} review")
                
            if use_ensemble and classification_stats.get('ensemble_stats'):
                ens = classification_stats['ensemble_stats']
                print(f"🔗 Predizioni ensemble: {ens['llm_predictions']} LLM + {ens['ml_predictions']} ML")
            
            return results
            
        except Exception as e:
            print(f"❌ ERRORE NELLA PIPELINE: {e}")
            raise
        
        finally:
            # Cleanup
            try:
                self.aggregator.chiudi_connessione()
                if hasattr(self.tag_db, 'connection') and self.tag_db.connection:
                    self.tag_db.disconnetti()
            except:
                pass
    
    def _update_semantic_memory_with_classifications(self, 
                                                     sessioni: Dict[str, Dict],
                                                     classification_stats: Dict[str, Any]) -> Dict[str, int]:
        """
        Aggiorna la memoria semantica con le nuove classificazioni ad alta confidenza
        """
        if not self.semantic_memory:
            return {'error': 'Semantic memory not initialized'}
        
        added_to_memory = 0
        for session_id, session_data in sessioni.items():
            # Trova la classificazione per questa sessione
            # Questa è una semplificazione, in un caso reale si dovrebbe avere un mapping diretto
            # tra session_id e la sua predizione nel dizionario classification_stats
            # Per ora, assumiamo che l'etichetta sia reperibile in qualche modo.
            # TODO: Migliorare il passaggio dei dati di classificazione
            
            # Simuliamo il recupero dell'etichetta classificata
            found = False
            for tag, count in classification_stats.get('classifications_by_tag', {}).items():
                # Questo è un modo imperfetto di trovare l'etichetta, ma sufficiente per ora
                # In un refactor futuro, `classifica_e_salva_sessioni` dovrebbe ritornare
                # un dizionario {session_id: {predicted_label, confidence}}
                pass # Logica da implementare

        # Per ora, aggiungiamo tutte le sessioni classificate con successo
        # alla memoria semantica. In futuro, si potrebbe filtrare per confidenza.
        classificazioni_salvate = self.tag_db.get_classificazioni_by_session_ids(list(sessioni.keys()))

        for classificazione in classificazioni_salvate:
            session_id = classificazione['session_id']
            if session_id in sessioni:
                self.semantic_memory.add_session_to_memory(
                    session_id=session_id,
                    conversation_text=sessioni[session_id]['testo_completo'],
                    tag=classificazione['tag_name'],
                    classified_by='pipeline_update',
                    confidence=classificazione['confidence_score']
                )
                added_to_memory += 1

        if added_to_memory > 0:
            self.semantic_memory.save_semantic_memory()
            print(f"🧠 Memoria semantica aggiornata con {added_to_memory} nuove sessioni.")

        return {'added_to_memory': added_to_memory}
    
    def get_statistiche_database(self) -> Dict[str, Any]:
        """
        Recupera statistiche dal database TAG
        
        Returns:
            Statistiche delle classificazioni salvate
        """
        self.tag_db.connetti()
        try:
            stats = self.tag_db.get_statistiche_classificazioni()
            return stats
        finally:
            self.tag_db.disconnetti()
    
    def analizza_e_consolida_etichette(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Analizza le etichette esistenti e propone/applica consolidamenti per eliminare duplicati semantici
        
        Args:
            dry_run: Se True, mostra solo le proposte senza applicarle
            
        Returns:
            Risultati dell'analisi e consolidamento
        """
        print(f"🔍 Analisi etichette esistenti per consolidamento...")
        
        # 1. Recupera tutte le etichette esistenti
        self.tag_db.connetti()
        try:
            query = """
            SELECT tag_name, COUNT(*) as count
            FROM session_classifications 
            GROUP BY tag_name
            ORDER BY count DESC
            """
            etichette_stats = self.tag_db.esegui_query(query)
            
            if not etichette_stats:
                print("❌ Nessuna etichetta trovata")
                return {}
            
            etichette_list = [(tag, count) for tag, count in etichette_stats]
            print(f"📊 Trovate {len(etichette_list)} etichette uniche")
            
        finally:
            self.tag_db.disconnetti()
        
        # 2. Definisci regole di consolidamento semantico
        consolidation_rules = {
            # Accesso - tutte le varianti di accesso
            'accesso_portale': [
                'accesso_app_personale',
                'accesso_ospedale',  # Potrebbe essere diverso, da valutare
                'accesso_sistema'
            ],
            
            # Fatturazione - tutte le varianti di pagamenti
            'fatturazione_pagamenti': [
                'pagamenti_parcheggio',  # Specifico ma sempre fatturazione
                'fatturazione',
                'pagamenti'
            ],
            
            # Referti - tutti i tipi di ritiro referti
            'ritiro_referti': [
                'ritiro_referti_istologici',
                'ritiro_esami',
                'download_referti'
            ],
            
            # Prenotazioni - tutte le varianti
            'prenotazione_esami': [
                'prenotazione_privata',
                'prenotazione_visite',
                'prenota_appuntamento'
            ],
            
            # Contatti e orari - informazioni generali
            'orari_contatti': [
                'contatti_info',
                'informazioni_orari',
                'info_struttura'
            ],
            
            # Trasporti e alloggi - servizi di supporto
            'servizi_supporto': [
                'prenotazione_trasporti',
                'alloggio_accompagnatore',
                'servizi_accompagnamento'
            ],
            
            # Assistenza medica specializzata
            'assistenza_medica_specializzata': [
                'riabilitazione_neurologica',
                'trattamento_cistite_chronic',
                'medicamenti_farmaci',
                'informazioni_esami_gravidanza_aborti_spontanei',
                'invito_medico_procedura_chirurgica'
            ]
        }
        
        # 3. Analizza consolidamenti possibili
        consolidation_plan = {}
        total_consolidations = 0
        
        for target_label, source_labels in consolidation_rules.items():
            # Trova etichette esistenti che matchano le regole
            found_sources = []
            target_count = 0
            
            for etichetta, count in etichette_list:
                if etichetta == target_label:
                    target_count = count
                elif etichetta in source_labels:
                    found_sources.append((etichetta, count))
            
            if found_sources:
                total_source_count = sum(count for _, count in found_sources)
                consolidation_plan[target_label] = {
                    'target_existing_count': target_count,
                    'sources_to_merge': found_sources,
                    'total_source_count': total_source_count,
                    'final_count': target_count + total_source_count
                }
                total_consolidations += len(found_sources)
        
        # 4. Mostra piano di consolidamento
        print(f"\n📋 PIANO DI CONSOLIDAMENTO:")
        print(f"🔧 {total_consolidations} etichette da consolidare")
        print("-" * 60)
        
        for target, plan in consolidation_plan.items():
            print(f"\n🎯 Target: {target}")
            print(f"  📊 Esistenti: {plan['target_existing_count']} classificazioni")
            print(f"  🔄 Da consolidare:")
            for source, count in plan['sources_to_merge']:
                print(f"    - {source}: {count} classificazioni")
            print(f"  ✅ Totale finale: {plan['final_count']} classificazioni")
        
        # 5. Applica consolidamento se non è dry_run
        if not dry_run:
            print(f"\n🚀 APPLICAZIONE CONSOLIDAMENTO...")
            applied_consolidations = 0
            
            self.tag_db.connetti()
            try:
                for target, plan in consolidation_plan.items():
                    for source_label, count in plan['sources_to_merge']:
                        print(f"  🔄 Consolidando {source_label} → {target}...")
                        
                        # Update query per cambiare tutte le occorrenze
                        update_query = """
                        UPDATE session_classifications 
                        SET tag_name = %s, 
                            notes = CONCAT(COALESCE(notes, ''), ' [Consolidato da: ', %s, ']'),
                            updated_at = NOW()
                        WHERE tag_name = %s
                        """
                        
                        risultato = self.tag_db.esegui_update(
                            update_query, 
                            (target, source_label, source_label)
                        )
                        
                        if risultato:
                            applied_consolidations += 1
                            print(f"    ✅ {count} classificazioni aggiornate")
                        else:
                            print(f"    ❌ Errore nell'aggiornamento")
                
            finally:
                self.tag_db.disconnetti()
            
            print(f"\n✅ CONSOLIDAMENTO COMPLETATO!")
            print(f"🔧 {applied_consolidations} etichette consolidate con successo")
            
            # Aggiorna memoria semantica per riflettere i cambiamenti
            print(f"🧠 Aggiornamento memoria semantica...")
            self.semantic_memory.load_semantic_memory()
            
        else:
            print(f"\n💡 MODALITÀ PREVIEW - Nessun cambiamento applicato")
            print(f"📝 Usa dry_run=False per applicare le modifiche")
        
        # 6. Statistiche finali
        results = {
            'total_labels_analyzed': len(etichette_list),
            'consolidation_plan': consolidation_plan,
            'total_consolidations_possible': total_consolidations,
            'applied': not dry_run,
            'applied_consolidations': applied_consolidations if not dry_run else 0
        }
        
        return results
    
    def ottimizza_normalizzazione_etichette(self) -> None:
        """
        Aggiorna la mappa di normalizzazione basata sui consolidamenti
        """
        print(f"🔧 Ottimizzazione mappa di normalizzazione...")
        
        # Aggiorna il metodo _normalize_label con le nuove regole
        enhanced_normalization_map = {
            # Accesso - tutte le varianti
            'accesso portale': 'accesso_portale',
            'accesso_portale': 'accesso_portale',
            'accesso app personale': 'accesso_portale',
            'accesso_app_personale': 'accesso_portale',
            'accesso ospedale': 'accesso_portale',  # Consolidato
            'accesso_ospedale': 'accesso_portale',  # Consolidato
            'accesso sistema': 'accesso_portale',
            'accesso_sistema': 'accesso_portale',
            
            # Fatturazione - tutte le varianti
            'fatturazione': 'fatturazione_pagamenti',
            'fatturazione_pagamenti': 'fatturazione_pagamenti',
            'fatturazione pagamenti': 'fatturazione_pagamenti',
            'pagamenti': 'fatturazione_pagamenti',
            'pagamenti parcheggio': 'fatturazione_pagamenti',
            'pagamenti_parcheggio': 'fatturazione_pagamenti',
            
            # Referti - tutte le varianti
            'ritiro referti': 'ritiro_referti',
            'ritiro_referti': 'ritiro_referti',
            'ritiro referti istologici': 'ritiro_referti',
            'ritiro_referti_istologici': 'ritiro_referti',
            'ritiro esami': 'ritiro_referti',
            'ritiro_esami': 'ritiro_referti',
            'download referti': 'ritiro_referti',
            'download_referti': 'ritiro_referti',
            
            # Prenotazioni - tutte le varianti
            'prenotazione esami': 'prenotazione_esami',
            'prenotazione_esami': 'prenotazione_esami',
            'prenotazione privata': 'prenotazione_esami',
            'prenotazione_privata': 'prenotazione_esami',
            'prenotazione visite': 'prenotazione_esami',
            'prenotazione_visite': 'prenotazione_esami',
            
            # Orari e contatti
            'orari strutture': 'orari_contatti',
            'orari_strutture': 'orari_contatti',
            'contatti info': 'orari_contatti',
            'contatti_info': 'orari_contatti',
            'orari contatti': 'orari_contatti',
            'orari_contatti': 'orari_contatti',
            
            # Servizi di supporto
            'prenotazione trasporti': 'servizi_supporto',
            'prenotazione_trasporti': 'servizi_supporto',
            'alloggio accompagnatore': 'servizi_supporto',
            'alloggio_accompagnatore': 'servizi_supporto',
            'servizi accompagnamento': 'servizi_supporto',
            'servizi_accompagnamento': 'servizi_supporto',
            'servizi supporto': 'servizi_supporto',
            'servizi_supporto': 'servizi_supporto',
            
            # Assistenza medica specializzata
            'riabilitazione neurologica': 'assistenza_medica_specializzata',
            'riabilitazione_neurologica': 'assistenza_medica_specializzata',
            'trattamento cistite chronic': 'assistenza_medica_specializzata',
            'trattamento_cistite_chronic': 'assistenza_medica_specializzata',
            'medicamenti farmaci': 'assistenza_medica_specializzata',
            'medicamenti_farmaci': 'assistenza_medica_specializzata',
            'assistenza tecnica internazionali': 'assistenza_medica_specializzata',
            'assistenza_tecnica_internazionali': 'assistenza_medica_specializzata',
            'informazioni esami gravidanza aborti spontanei': 'assistenza_medica_specializzata',
            'informazioni_esami_gravidanza_aborti_spontanei': 'assistenza_medica_specializzata',
            'invito medico procedura chirurgica': 'assistenza_medica_specializzata',
            'invito_medico_procedura_chirurgica': 'assistenza_medica_specializzata',
            
            # Preparazione esami
            'preparazione esami': 'preparazione_esami',
            'preparazione_esami': 'preparazione_esami',
            
            # Altri
            'altro': 'altro',
            'informazioni generali': 'altro',  # Troppo generico
            'informazioni_generali': 'altro'   # Troppo generico
        }
        
        # Sovrascrivi il metodo _normalize_label temporaneamente
        # (In un vero refactor, questo andrebbe nel file di configurazione)
        self._enhanced_normalization_map = enhanced_normalization_map
        print(f"✅ Mappa di normalizzazione aggiornata con {len(enhanced_normalization_map)} regole")

    def test_consolidamento_etichette(self) -> None:
        """
        Testa il sistema di consolidamento delle etichette
        """
        print("=== TEST SISTEMA CONSOLIDAMENTO ETICHETTE ===\n")
        
        # 1. Mostra situazione attuale
        print("📊 SITUAZIONE ATTUALE:")
        stats_prima = self.get_statistiche_database()
        print(f"  Totale classificazioni: {stats_prima['total_classificazioni']}")
        print("  Distribuzione etichette:")
        for item in stats_prima['per_tag'][:10]:  # Top 10
            print(f"    {item['tag']}: {item['count']}")
        
        # 2. Analizza consolidamento (dry run)
        print(f"\n🔍 ANALISI CONSOLIDAMENTO (PREVIEW):")
        risultati_analisi = self.analizza_e_consolida_etichette(dry_run=True)
        
        # 3. Chiede conferma per applicare
        if risultati_analisi['total_consolidations_possible'] > 0:
            print(f"\n❓ Vuoi applicare i consolidamenti? (Questo modificherà il database)")
            print(f"📝 Per applicare, chiama: pipeline.analizza_e_consolida_etichette(dry_run=False)")
        else:
            print(f"\n✅ Nessun consolidamento necessario!")
        
        return risultati_analisi
    
    def esegui_training_interattivo(self,
                                  giorni_indietro: int = 7,
                                  limit: Optional[int] = 100,
                                  max_human_review_sessions: Optional[int] = None,
                                  confidence_threshold: float = 0.7,
                                  force_review: bool = False,
                                  disagreement_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Esegue solo la fase di training con review interattivo senza classificazione
        NUOVA LOGICA: 
        - Estrae SEMPRE tutte le discussioni dal database (ignora limit per clustering)
        - Il limit diventa il numero massimo di sessioni rappresentative da sottoporre all'umano
        
        Args:
            giorni_indietro: Giorni di dati da processare (per compatibilità, non utilizzato in estrazione completa)
            limit: DEPRECATO - ora indica max sessioni per review umana (default dalla config)
            max_human_review_sessions: Numero massimo sessioni rappresentative per review umana
            confidence_threshold: Soglia confidenza per selezione sessioni (default: 0.7)
            force_review: Forza revisione casi già revisionati (default: False)
            disagreement_threshold: Soglia disagreement ensemble per priorità (default: 0.3)
            
        Returns:
            Risultati del training interattivo con statistiche complete
        """
        start_time = datetime.now()
        
        # Determina limite review umana dalla configurazione
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            supervised_config = config.get('supervised_training', {})
            human_review_config = supervised_config.get('human_review', {})
            
            # Limite sessioni per review umana (non per estrazione!)
            if max_human_review_sessions is not None:
                human_limit = max_human_review_sessions
            elif limit is not None:
                human_limit = limit  # Retrocompatibilità
                print(f"⚠️ ATTENZIONE: Parametro 'limit' è deprecato. Ora indica max sessioni per review umana.")
            else:
                human_limit = human_review_config.get('max_total_sessions', 500)
                
        except Exception as e:
            print(f"⚠️ Errore lettura config: {e}")
            human_limit = limit or 500
        
        print(f"🎓 TRAINING SUPERVISIONATO AVANZATO")
        print(f"� NUOVA LOGICA:")
        print(f"  🔄 Estrazione: TUTTE le discussioni dal database")
        print(f"  🧩 Clustering: Su tutto il dataset completo")
        print(f"  👤 Review umana: Massimo {human_limit} sessioni rappresentative")
        print("-" * 50)
        
        try:
            # 1. Estrazione COMPLETA (ignora qualsiasi limite)
            print(f"📊 FASE 1: ESTRAZIONE COMPLETA DATASET")
            sessioni = self.estrai_sessioni(giorni_indietro=giorni_indietro, 
                                           limit=None,  # FORZATO a None per estrazione completa
                                           force_full_extraction=True)
            
            if len(sessioni) < 5:
                raise ValueError(f"Troppo poche sessioni ({len(sessioni)}) per training significativo")
            
            print(f"✅ Dataset completo: {len(sessioni)} sessioni totali")
            
            # 2. Clustering COMPLETO su tutto il dataset
            print(f"\n📊 FASE 2: CLUSTERING COMPLETO")
            embeddings, cluster_labels, representatives, suggested_labels = self.esegui_clustering(sessioni)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_outliers = sum(1 for label in cluster_labels if label == -1)
            print(f"✅ Clustering completo: {n_clusters} cluster, {n_outliers} outlier")
            
            # 3. Selezione intelligente rappresentanti per review umana
            print(f"\n📊 FASE 3: SELEZIONE RAPPRESENTANTI PER REVIEW UMANA")
            limited_representatives, selection_stats = self._select_representatives_for_human_review(
                representatives, suggested_labels, human_limit, sessioni,
                confidence_threshold=confidence_threshold,
                force_review=force_review,
                disagreement_threshold=disagreement_threshold
            )
            
            print(f"✅ Selezione completata:")
            print(f"  📋 Cluster originali: {len(representatives)}")
            print(f"  👤 Cluster per review: {len(limited_representatives)}")
            print(f"  📝 Sessioni per review: {selection_stats['total_sessions_for_review']}")
            print(f"  🚫 Cluster esclusi: {selection_stats['excluded_clusters']}")
            
            # 4. Training interattivo con rappresentanti selezionati
            print(f"\n📊 FASE 4: TRAINING SUPERVISIONATO")
            training_metrics = self.allena_classificatore(
                sessioni, cluster_labels, limited_representatives, suggested_labels, 
                interactive_mode=True
            )
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            results = {
                'pipeline_info': {
                    'mode': 'supervised_training_advanced',
                    'tenant_slug': self.tenant_slug,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_seconds': duration.total_seconds()
                },
                'extraction_stats': {
                    'total_sessions_extracted': len(sessioni),
                    'extraction_mode': 'FULL_DATASET',
                    'ignored_original_limit': limit
                },
                'clustering_stats': {
                    'total_sessions_clustered': len(sessioni),
                    'n_clusters': n_clusters,
                    'n_outliers': n_outliers,
                    'clustering_mode': 'COMPLETE'
                },
                'human_review_stats': {
                    'max_sessions_for_review': human_limit,
                    'actual_sessions_for_review': selection_stats['total_sessions_for_review'],
                    'clusters_reviewed': len(limited_representatives),
                    'clusters_excluded': selection_stats['excluded_clusters'],
                    'selection_strategy': selection_stats.get('strategy', 'unknown')
                },
                'training_metrics': training_metrics
            }
            
            print("-" * 50)
            print(f"✅ TRAINING SUPERVISIONATO COMPLETATO!")
            print(f"⏱️  Durata: {duration.total_seconds():.1f} secondi")
            print(f"📊 Dataset: {len(sessioni)} sessioni totali")
            print(f"🧩 Clustering: {n_clusters} cluster su dataset completo")
            print(f"👤 Review: {selection_stats['total_sessions_for_review']} sessioni riviste dall'umano")
            
            if 'human_feedback_stats' in training_metrics:
                feedback = training_metrics['human_feedback_stats']
                print(f"👤 Review umane: {feedback.get('total_reviews', 0)}")
                print(f"✅ Etichette approvate: {feedback.get('approved_labels', 0)}")
                print(f"📝 Nuove etichette: {feedback.get('new_labels', 0)}")
            
            return results
            
        except Exception as e:
            print(f"❌ ERRORE NEL TRAINING: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _select_representatives_for_human_review(self,
                                               representatives: Dict[int, List[Dict]],
                                               suggested_labels: Dict[int, str],
                                               max_sessions: int,
                                               all_sessions: Dict[str, Dict],
                                               confidence_threshold: float = 0.7,
                                               force_review: bool = False,
                                               disagreement_threshold: float = 0.3) -> Tuple[Dict[int, List[Dict]], Dict[str, Any]]:
        """
        Seleziona intelligentemente i rappresentanti dei cluster per la review umana
        rispettando il limite massimo di sessioni.
        
        Args:
            representatives: Dizionario cluster_id -> lista rappresentanti
            suggested_labels: Etichette suggerite per cluster
            max_sessions: Numero massimo di sessioni da sottoporre all'umano
            all_sessions: Tutte le sessioni per calcolare statistiche
            confidence_threshold: Soglia confidenza per selezione (default: 0.7)
            force_review: Forza revisione casi già revisionati (default: False)
            disagreement_threshold: Soglia disagreement ensemble per priorità (default: 0.3)
            
        Returns:
            Tuple (limited_representatives, selection_stats)
        """
        print(f"🔍 Selezione intelligente rappresentanti per review umana...")
        
        # Carica configurazione
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            supervised_config = config.get('supervised_training', {})
            human_review_config = supervised_config.get('human_review', {})
            
            # Parametri di selezione
            min_reps_per_cluster = human_review_config.get('min_representatives_per_cluster', 1)
            max_reps_per_cluster = human_review_config.get('max_representatives_per_cluster', 5)
            default_reps_per_cluster = human_review_config.get('representatives_per_cluster', 3)
            selection_strategy = human_review_config.get('selection_strategy', 'prioritize_by_size')
            min_cluster_size = human_review_config.get('min_cluster_size_for_review', 2)
            
        except Exception as e:
            print(f"⚠️ Errore config, uso valori default: {e}")
            min_reps_per_cluster = 1
            max_reps_per_cluster = 5
            default_reps_per_cluster = 3
            selection_strategy = 'prioritize_by_size'
            min_cluster_size = 2
        
        # Calcola dimensioni cluster
        cluster_sizes = {}
        for cluster_id, reps in representatives.items():
            cluster_sizes[cluster_id] = len(reps) if reps else 0
        
        # Filtra cluster troppo piccoli
        eligible_clusters = {
            cluster_id: reps for cluster_id, reps in representatives.items()
            if cluster_sizes.get(cluster_id, 0) >= min_cluster_size
        }
        
        excluded_small_clusters = len(representatives) - len(eligible_clusters)
        
        print(f"📊 Analisi cluster:")
        print(f"  📋 Cluster totali: {len(representatives)}")
        print(f"  ✅ Cluster eleggibili: {len(eligible_clusters)}")
        print(f"  🚫 Cluster troppo piccoli (< {min_cluster_size}): {excluded_small_clusters}")
        
        # Se non abbiamo cluster eleggibili, ritorna tutto disponibile
        if not eligible_clusters:
            print(f"⚠️ Nessun cluster eleggibile, ritorno cluster disponibili")
            return representatives, {
                'total_sessions_for_review': sum(len(reps) for reps in representatives.values()),
                'excluded_clusters': 0,
                'strategy': 'fallback_all'
            }
        
        # Calcola sessioni totali se prendiamo tutti i rappresentanti default
        total_sessions_with_default = sum(
            min(len(reps), default_reps_per_cluster) 
            for reps in eligible_clusters.values()
        )
        
        print(f"📊 Calcolo sessioni:")
        print(f"  🎯 Limite massimo: {max_sessions}")
        print(f"  📝 Con {default_reps_per_cluster} reps/cluster: {total_sessions_with_default}")
        
        # Se stiamo sotto il limite, usa default
        if total_sessions_with_default <= max_sessions:
            print(f"✅ Possiamo usare configurazione standard ({default_reps_per_cluster} reps/cluster)")
            
            limited_representatives = {}
            total_selected_sessions = 0
            
            for cluster_id, reps in eligible_clusters.items():
                selected_reps = reps[:default_reps_per_cluster]
                limited_representatives[cluster_id] = selected_reps
                total_selected_sessions += len(selected_reps)
            
            return limited_representatives, {
                'total_sessions_for_review': total_selected_sessions,
                'excluded_clusters': excluded_small_clusters,
                'strategy': f'standard_{default_reps_per_cluster}_per_cluster'
            }
        
        # Dobbiamo applicare selezione intelligente
        print(f"⚡ Applicazione selezione intelligente (strategia: {selection_strategy})")
        
        if selection_strategy == 'prioritize_by_size':
            # Ordina cluster per dimensione (più grandi prima)
            sorted_clusters = sorted(eligible_clusters.keys(), 
                                   key=lambda cid: cluster_sizes[cid], 
                                   reverse=True)
            
        elif selection_strategy == 'prioritize_by_confidence':
            # Ordina per confidenza (più bassi prima = hanno più bisogno di review)
            def get_cluster_confidence(cluster_id):
                reps = eligible_clusters[cluster_id]
                confidences = [
                    rep.get('classification_confidence', 0.5) 
                    for rep in reps if rep.get('classification_confidence') is not None
                ]
                return sum(confidences) / len(confidences) if confidences else 0.5
            
            sorted_clusters = sorted(eligible_clusters.keys(), 
                                   key=get_cluster_confidence)
                                   
        else:  # balanced
            # Strategia bilanciata: alterna grandi e a bassa confidenza
            sorted_clusters = list(eligible_clusters.keys())
        
        # Assegna rappresentanti rispettando il limite
        limited_representatives = {}
        total_selected_sessions = 0
        remaining_budget = max_sessions
        
        print(f"🎯 Inizio selezione cluster ordinati...")
        
        for cluster_id in sorted_clusters:
            reps = eligible_clusters[cluster_id]
            cluster_size = cluster_sizes[cluster_id]
            
            # Calcola quanti rappresentanti assegnare a questo cluster
            if remaining_budget <= 0:
                break
                
            # Proporzionale alla dimensione, ma rispetta min/max
            base_reps = max(min_reps_per_cluster, 
                           min(max_reps_per_cluster, 
                               int(cluster_size / 10) + 1))  # +1 rep ogni 10 sessioni
            
            # Non superare il budget rimanente
            actual_reps = min(base_reps, remaining_budget, len(reps))
            
            if actual_reps > 0:
                limited_representatives[cluster_id] = reps[:actual_reps]
                total_selected_sessions += actual_reps
                remaining_budget -= actual_reps
                
                print(f"  ✅ Cluster {cluster_id}: {actual_reps}/{len(reps)} reps (size: {cluster_size})")
            else:
                print(f"  🚫 Cluster {cluster_id}: escluso (budget esaurito)")
        
        excluded_clusters = len(eligible_clusters) - len(limited_representatives) + excluded_small_clusters
        
        print(f"✅ Selezione completata:")
        print(f"  📝 Sessioni selezionate: {total_selected_sessions}/{max_sessions}")
        print(f"  👤 Cluster per review: {len(limited_representatives)}")
        print(f"  🚫 Cluster esclusi totali: {excluded_clusters}")
        
        return limited_representatives, {
            'total_sessions_for_review': total_selected_sessions,
            'excluded_clusters': excluded_clusters,
            'strategy': selection_strategy,
            'budget_used': total_selected_sessions,
            'budget_available': max_sessions
        }
    
    def classifica_sessioni_esistenti(self,
                                    session_ids: List[str],
                                    use_ensemble: bool = True,
                                    review_uncertain: bool = True) -> Dict[str, Any]:
        """
        Classifica un set specifico di sessioni esistenti
        
        Args:
            session_ids: Lista di ID delle sessioni da classificare
            use_ensemble: Se True, usa l'ensemble classifier
            review_uncertain: Se True, permette review manuale per casi incerti
            
        Returns:
            Risultati della classificazione
        """
        print(f"🏷️  Classificazione di {len(session_ids)} sessioni specifiche...")
        
        # Recupera i dati delle sessioni
        sessioni = {}
        for session_id in session_ids:
            try:
                # In una implementazione reale, recupereresti i dati dal database
                # Per ora simulo il recupero
                session_data = self.aggregator.get_session_by_id(session_id)
                if session_data:
                    sessioni[session_id] = session_data
            except Exception as e:
                print(f"⚠️ Impossibile recuperare sessione {session_id}: {e}")
        
        if not sessioni:
            print("❌ Nessuna sessione valida trovata")
            return {}
        
        print(f"📊 Recuperate {len(sessioni)} sessioni valide")
        
        # Classifica le sessioni
        classification_stats = self.classifica_e_salva_sessioni(
            sessioni, use_ensemble=use_ensemble
        )
        
        # Review manuale per casi incerti se richiesto
        if review_uncertain and classification_stats.get('low_confidence', 0) > 0:
            print(f"\n🔍 REVIEW CASI INCERTI")
            print(f"Trovate {classification_stats['low_confidence']} classificazioni a bassa confidenza")
            print("⏭️ Procedendo automaticamente con review dei casi incerti")
            uncertain_stats = self._review_uncertain_classifications(sessioni, classification_stats)
            classification_stats.update(uncertain_stats)
        
        return classification_stats
    
    def _review_uncertain_classifications(self, 
                                        sessioni: Dict[str, Dict], 
                                        classification_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Permette review manuale delle classificazioni incerte
        """
        print(f"👤 Inizio review manuale classificazioni incerte...")
        
        review_stats = {
            'reviewed_sessions': 0,
            'corrections_made': 0,
            'confirmations': 0
        }
        
        # Per ora implementazione base - in futuro si potrebbe migliorare
        # recuperando dal database le classificazioni a bassa confidenza
        print(f"💡 Funzionalità di review completa sarà implementata nelle prossime iterazioni")
        print(f"📝 Per ora usa l'interfaccia del database per correzioni manuali")
        
        return review_stats
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """
        Recupera statistiche dettagliate sull'ensemble classifier
        
        Returns:
            Statistiche dell'ensemble classifier
        """
        if not self.ensemble_classifier:
            return {'error': 'Ensemble classifier non inizializzato'}
        
        # Usa il metodo integrato dell'advanced ensemble classifier
        return self.ensemble_classifier.get_ensemble_statistics()
    
    def adjust_ensemble_weights(self, llm_weight: float, ml_weight: float) -> None:
        """
        Regola manualmente i pesi dell'ensemble classifier
        
        Args:
            llm_weight: Nuovo peso per LLM (0.0 - 1.0)
            ml_weight: Nuovo peso per ML (0.0 - 1.0)
        """
        if not self.ensemble_classifier:
            print("❌ Ensemble classifier non disponibile")
            return
        
        # Normalizza i pesi
        total = llm_weight + ml_weight
        if total > 0:
            llm_weight = llm_weight / total
            ml_weight = ml_weight / total
        
        # Aggiorna i pesi nell'advanced ensemble classifier
        self.ensemble_classifier.weights['llm'] = llm_weight
        self.ensemble_classifier.weights['ml_ensemble'] = ml_weight
        
        print(f"✅ Pesi ensemble aggiornati: LLM={llm_weight:.3f}, ML={ml_weight:.3f}")
    
    def set_auto_retrain(self, enable: bool) -> None:
        """
        Sovrascrive temporaneamente il setting auto_retrain da config
        
        Args:
            enable: Se True, abilita riaddestramento per questa istanza
        """
        self.auto_retrain = enable
        if enable:
            print("🔄 Auto-retrain abilitato manualmente per questa istanza")
        else:
            print("⏸️ Auto-retrain disabilitato manualmente per questa istanza")

    def _prevent_duplicate_labels(self, suggested_labels: Dict[int, str]) -> Dict[int, str]:
        """
        Sistema intelligente per prevenzione duplicati etichette
        basato su ML + LLM invece di pattern predefiniti
        
        Args:
            suggested_labels: Etichette suggerite dal clustering
            
        Returns:
            Etichette deduplicate e normalizzate
        """
        print("🧠 Sistema intelligente di dedupplicazione etichette...")
        
        try:
            # Usa il nuovo sistema intelligente
            deduplicated_labels = self.label_deduplicator.prevent_duplicate_labels(suggested_labels)
            
            # Mostra statistiche
            stats = self.label_deduplicator.get_statistics()
            if stats['total_decisions'] > 0:
                print(f"� Statistiche dedupplicazione:")
                print(f"  - Etichette riusate: {stats['labels_reused']}")
                print(f"  - Nuove etichette: {stats['labels_created']}")
                print(f"  - Tasso di riuso: {stats['reuse_rate']:.2%}")
            
            return deduplicated_labels
            
        except Exception as e:
            print(f"⚠️ Errore nel sistema intelligente: {e}")
            print(f"🚫 NESSUN FALLBACK PATTERN-BASED disponibile - sistema puro ML+LLM")
            # Ritorna le etichette originali senza modifiche se il sistema intelligente fallisce
            # Meglio etichette potenzialmente duplicate che pattern rigidi
            return suggested_labels
    
    def _normalize_label_name(self, label: str) -> str:
        """
        Normalizza SOLO il formato dell'etichetta - nessuna mappatura semantica
        """
        import re
        
        # Rimuovi caratteri speciali e normalizza spazi
        normalized = re.sub(r'[^\w\s]', '', label)
        normalized = re.sub(r'\s+', '_', normalized.strip())
        normalized = normalized.lower()
        
        # NESSUNA REGOLA SEMANTICA - solo formattazione
        return normalized
    
    def _try_load_latest_model(self):
        """
        Prova a caricare l'ultimo modello ML salvato dal disco
        """
        try:
            import os
            import glob            
            models_dir = "models"
            if not os.path.exists(models_dir):
                print("⚠️ Directory models/ non trovata, nessun modello da caricare")
                return
            
            config_files = glob.glob(os.path.join(models_dir, "*_config.json"))
            if not config_files:
                print("⚠️ Nessun modello trovato nella directory models/")
                return
            
            config_files.sort()
            latest_config = config_files[-1]
            model_base_name = latest_config.replace("_config.json", "")
            ml_file = f"{model_base_name}_ml_ensemble.pkl"
            if not os.path.exists(ml_file):
                print(f"⚠️ File ML non trovato: {ml_file}")
                return
            
            print(f"📥 Caricamento ultimo modello: {os.path.basename(model_base_name)}")
            self.ensemble_classifier.load_ensemble_model(model_base_name)
            print("✅ Modello caricato con successo")

            # Carica eventuale provider BERTopic accoppiato
            subdir = self.bertopic_config.get('model_subdir', 'bertopic')
            provider_dir = f"{model_base_name}_{subdir}"
            if self.bertopic_config.get('enabled', False) and _BERTopic_AVAILABLE and os.path.isdir(provider_dir):
                try:
                    provider = BERTopicFeatureProvider().load(provider_dir)
                    self.ensemble_classifier.set_bertopic_provider(
                        provider,
                        top_k=self.bertopic_config.get('top_k', 15),
                        return_one_hot=self.bertopic_config.get('return_one_hot', False)
                    )
                    print(f"🔗 BERTopic provider caricato da {provider_dir}")
                except Exception as e:
                    print(f"⚠️ Errore caricamento BERTopic provider: {e}")
            else:
                if self.bertopic_config.get('enabled', False) and not _BERTopic_AVAILABLE:
                    print("⚠️ BERTopic abilitato ma non disponibile in runtime; proseguo senza provider")
        except Exception as e:
            print(f"⚠️ Errore nel caricamento del modello: {e}")
            print("   Il sistema continuerà con modelli non addestrati")

    def _recluster_outliers_with_history(self, 
                                         nuove_sessioni: Dict[str, Dict],
                                         cluster_labels: np.ndarray,
                                         embeddings: np.ndarray) -> Tuple[Dict[str, Dict], np.ndarray]:
        """
        Re-clustering intelligente che include gli outlier storici
        
        Args:
            nuove_sessioni: Nuove sessioni da processare
            cluster_labels: Labels dal clustering corrente
            embeddings: Embeddings delle nuove sessioni
            
        Returns:
            Tuple (sessioni_aggiornate, labels_aggiornati)
        """
        print("🔄 Re-clustering outlier con storico...")
        
        # 1. Identifica outlier nelle nuove sessioni
        nuovi_outlier_indices = [i for i, label in enumerate(cluster_labels) if label == -1]
        session_ids = list(nuove_sessioni.keys())
        nuovi_outlier_ids = [session_ids[i] for i in nuovi_outlier_indices]
        
        if not nuovi_outlier_ids:
            print("  ✅ Nessun nuovo outlier da processare")
            return nuove_sessioni, cluster_labels
        
        print(f"  🔍 Trovati {len(nuovi_outlier_ids)} nuovi outlier")
        
        # 2. Recupera outlier storici dal database
        self.tag_db.connetti()
        try:
            # Query per trovare sessioni non classificate o con bassa confidenza
            query = """
            SELECT session_id, confidence_score 
            FROM session_classifications 
            WHERE tenant_name = %s 
            AND (confidence_score < 0.5 OR tag_name = 'outlier' OR tag_name = 'altro')
            ORDER BY confidence_score ASC
            LIMIT 50
            """
            
            outlier_storico = self.tag_db.esegui_query(query, (self.tenant_slug,))
            outlier_storico_ids = [row[0] for row in outlier_storico] if outlier_storico else []
            
        finally:
            self.tag_db.disconnetti()
        
        print(f"  📚 Trovati {len(outlier_storico_ids)} outlier storici")
        
        # 3. Se abbiamo abbastanza outlier totali, tenta re-clustering
        total_outliers = len(nuovi_outlier_ids) + len(outlier_storico_ids)
        
        if total_outliers < 5:
            print(f"  ⚠️ Troppo pochi outlier totali ({total_outliers}) per re-clustering")
            return nuove_sessioni, cluster_labels
        
        # 4. Recupera dati degli outlier storici
        sessioni_outlier_storici = {}
        for outlier_id in outlier_storico_ids[:20]:  # Limita a 20 per performance
            try:
                # Recupera dati sessione dal database originale
                session_data = self.aggregator.get_session_by_id(outlier_id)
                if session_data:
                    sessioni_outlier_storici[outlier_id] = session_data
            except Exception as e:
                print(f"    ⚠️ Errore recupero outlier {outlier_id}: {e}")
        
        print(f"  📊 Recuperati dati per {len(sessioni_outlier_storici)} outlier storici")
        
        # 5. Combina tutti gli outlier per re-clustering
        tutte_sessioni_outlier = {}
        
        # Aggiungi nuovi outlier
        for outlier_id in nuovi_outlier_ids:
            tutte_sessioni_outlier[outlier_id] = nuove_sessioni[outlier_id]
        
        # Aggiungi outlier storici
        tutte_sessioni_outlier.update(sessioni_outlier_storici)
        
        # 6. Re-clustering solo degli outlier
        print(f"  🧩 Re-clustering di {len(tutte_sessioni_outlier)} outlier...")
        
        outlier_texts = [dati['testo_completo'] for dati in tutte_sessioni_outlier.values()]
        outlier_embeddings = self.embedder.encode(outlier_texts)
        
        # Usa parametri più permissivi per outlier
        from Clustering.hdbscan_clusterer import HDBSCANClusterer
        outlier_clusterer = HDBSCANClusterer(
            min_cluster_size=2,  # Molto più basso
            min_samples=1,       # Molto più basso
            cluster_selection_epsilon=0.1  # Più permissivo
        )
        
        outlier_cluster_labels = outlier_clusterer.fit_predict(outlier_embeddings)
        
        # 7. Analizza risultati re-clustering
        nuovi_cluster_trovati = len(set(outlier_cluster_labels)) - (1 if -1 in outlier_cluster_labels else 0)
        outlier_rimanenti = sum(1 for label in outlier_cluster_labels if label == -1)
        
        print(f"  ✅ Re-clustering completato:")
        print(f"    🆕 Nuovi cluster: {nuovi_cluster_trovati}")
        print(f"    🔍 Outlier rimanenti: {outlier_rimanenti}")
        
        # 8. Aggiorna le labels originali
        updated_labels = cluster_labels.copy()
        max_existing_cluster = max(cluster_labels) if len(cluster_labels) > 0 else -1
        
        # Mappa nuovi cluster agli ID originali
        for i, (session_id, new_label) in enumerate(zip(tutte_sessioni_outlier.keys(), outlier_cluster_labels)):
            if session_id in nuove_sessioni:
                # Trova l'indice originale nella lista nuove_sessioni
                original_index = session_ids.index(session_id)
                
                if new_label != -1:
                    # Assegna nuovo cluster ID
                    updated_labels[original_index] = max_existing_cluster + 1 + new_label
                    print(f"    🎯 {session_id}: outlier → cluster {updated_labels[original_index]}")
        
        # 9. Gestisci outlier storici che hanno formato nuovi cluster
        for i, (session_id, new_label) in enumerate(zip(tutte_sessioni_outlier.keys(), outlier_cluster_labels)):
            if session_id not in nuove_sessioni and new_label != -1:
                # Outlier storico che ora fa parte di un cluster
                new_cluster_id = max_existing_cluster + 1 + new_label
                print(f"    📚 Outlier storico {session_id} ora in cluster {new_cluster_id}")
                
                # Aggiorna nel database (marca per re-classificazione)
                self.tag_db.connetti()
                try:
                    update_query = """
                    UPDATE session_classifications 
                    SET tag_name = 'recluster_needed', 
                        notes = CONCAT(COALESCE(notes, ''), ' [Re-cluster candidate]'),
                        confidence_score = 0.5
                    WHERE session_id = %s AND tenant_name = %s
                    """
                    self.tag_db.esegui_update(update_query, (session_id, self.tenant_slug))
                finally:
                    self.tag_db.disconnetti()
        
        return nuove_sessioni, updated_labels


