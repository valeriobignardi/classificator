"""
Backup di Pipeline/end_to_end_pipeline.py
Creato: 2025-08-07 10:17:00
"""

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
    
    # ...resto del file invariato...
