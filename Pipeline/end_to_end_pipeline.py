"""
Pipeline end-to-end per il sistema di classificazione delle conversazioni Humanitas
"""

import sys
import os
import yaml
import json
import glob
import traceback
import hashlib
import time
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
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))

from lettore import LettoreConversazioni
from session_aggregator import SessionAggregator
# RIMOSSO: from labse_embedder import LaBSEEmbedder - Ora usa simple_embedding_manager
from hdbscan_clusterer import HDBSCANClusterer
# RIMOSSO: from intent_clusterer import IntentBasedClusterer  # Sistema legacy eliminato
from intelligent_intent_clusterer import IntelligentIntentClusterer
from Clustering.hierarchical_adaptive_clusterer import HierarchicalAdaptiveClusterer  # Nuovo sistema gerarchico

# Import per architettura UUID centralizzata
from Utils.tenant import Tenant
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
# Importa funzione di pulizia tag per prevenire caratteri speciali
from Classification.intelligent_classifier import clean_label_text

# Import del classificatore ensemble avanzato
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Classification'))
from advanced_ensemble_classifier import AdvancedEnsembleClassifier

# Import per gestione tenant-aware naming
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mongo_classification_reader import MongoClassificationReader

# 🆕 Import per gestione parametri tenant UMAP
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))
from tenant_config_helper import TenantConfigHelper
from tenant import Tenant

# Import BERTopic provider
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TopicModeling'))
try:
    from bertopic_feature_provider import BERTopicFeatureProvider
    _BERTopic_AVAILABLE = True
except Exception as _e:
    BERTopicFeatureProvider = None
    _BERTopic_AVAILABLE = False
    print(f"⚠️ BERTopic non disponibile: {_e}")


def trace_all(function_name: str, action: str = "ENTER", called_from: str = None, **kwargs):
    """
    Sistema di tracing completo per tracciare il flusso della pipeline
    
    Scopo della funzione: Tracciare ingresso, uscita ed errori di tutte le funzioni
    Parametri di input: function_name, action, called_from, **kwargs (parametri da tracciare)
    Parametri di output: None (scrive su file)
    Valori di ritorno: None
    Tracciamento aggiornamenti: 2025-09-06 - Valerio Bignardi - Sistema tracing pipeline
    
    Args:
        function_name (str): Nome della funzione da tracciare
        action (str): "ENTER", "EXIT", "ERROR"
        called_from (str): Nome della funzione chiamante (per tracciare chiamate annidate)
        **kwargs: Parametri da tracciare (input, return_value, exception, etc.)
        
    Autore: Valerio Bignardi
    Data: 2025-09-06
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
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
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


def get_supervised_training_params_from_db(tenant_id: str) -> Dict[str, Any]:
    """
    Recupera i parametri di training supervisionato dalla tabella MySQL 'soglie'
    
    Scopo: Leggere configurazione training supervisionato dal database invece che da config.yaml
    
    Parametri:
        tenant_id (str): ID del tenant
        
    Returns:
        dict: Parametri di training supervisionato o valori default
        
    Tracciamento aggiornamenti:
        - 28/01/2025: Creazione funzione per leggere parametri training supervisionato
    """
    trace_all("get_supervised_training_params_from_db", "ENTER", tenant_id=tenant_id)
    
    import mysql.connector
    from mysql.connector import Error
    
    # Valori default
    default_params = {
        'confidence_threshold_priority': 0.7,
        'max_representatives_per_cluster': 5,
        'max_total_sessions': 500,
        'min_representatives_per_cluster': 1,
        'overflow_handling': 'proportional',
        'representatives_per_cluster': 3,
        'selection_strategy': 'prioritize_by_size'
    }
    
    try:
        # Leggi configurazione database da config.yaml
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        db_config = config['tag_database']
        
        # Connessione al database
        print(f"🔌 [TRAINING DB] Connessione al database MySQL per tenant {tenant_id}")
        connection = mysql.connector.connect(
            host=db_config['host'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password'],
            port=db_config.get('port', 3306)
        )
        
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            
            # Query per recuperare parametri training supervisionato
            query = """
            SELECT confidence_threshold_priority, max_representatives_per_cluster, 
                   max_total_sessions, min_representatives_per_cluster,
                   overflow_handling, representatives_per_cluster, selection_strategy
            FROM soglie 
            WHERE tenant_id = %s 
            ORDER BY last_updated DESC 
            LIMIT 1
            """
            
            cursor.execute(query, (tenant_id,))
            result = cursor.fetchone()
            
            if result:
                print(f"✅ [TRAINING DB] Parametri trovati per tenant {tenant_id}")
                result_params = {
                    'confidence_threshold_priority': float(result['confidence_threshold_priority']),
                    'max_representatives_per_cluster': result['max_representatives_per_cluster'],
                    'max_total_sessions': result['max_total_sessions'],
                    'min_representatives_per_cluster': result['min_representatives_per_cluster'],
                    'overflow_handling': result['overflow_handling'],
                    'representatives_per_cluster': result['representatives_per_cluster'],
                    'selection_strategy': result['selection_strategy']
                }
                trace_all("get_supervised_training_params_from_db", "EXIT", return_value=result_params)
                return result_params
            else:
                print(f"⚠️ [TRAINING DB] Nessun record trovato per tenant {tenant_id}, uso valori default")
                trace_all("get_supervised_training_params_from_db", "EXIT", return_value=default_params)
                return default_params
                
    except Error as e:
        print(f"❌ [TRAINING DB] Errore MySQL per tenant {tenant_id}: {e}")
        print(f"📋 [TRAINING DB] Uso valori default")
        trace_all("get_supervised_training_params_from_db", "ERROR", exception=e)
        trace_all("get_supervised_training_params_from_db", "EXIT", return_value=default_params)
        return default_params
    except Exception as e:
        print(f"❌ [TRAINING DB] Errore generico: {e}")
        print(f"📋 [TRAINING DB] Uso valori default")
        trace_all("get_supervised_training_params_from_db", "ERROR", exception=e)
        trace_all("get_supervised_training_params_from_db", "EXIT", return_value=default_params)
        return default_params
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()


class EndToEndPipeline:
    """
    Pipeline completa per l'estrazione, clustering, classificazione e salvataggio
    delle sessioni di conversazione
    """
    
    def __init__(self,# Nuovo sistema di gestione tenant UUID centralizzato
                 tenant: Tenant = None, # Nuovo - oggetto Tenant
                 tenant_slug: str = None,  # Retrocompatibilità - DEPRECATO
                 confidence_threshold: float = None, # Soglia di confidenza
                 min_cluster_size: int = None, # Dimensione minima del cluster
                 min_samples: int = None, # Numero minimo di campioni
                 config_path: str = None, # Percorso del file di configurazione     
                 auto_mode: bool = None, # Modalità automatica
                 shared_embedder=None): # Evita CUDA out of memory
        """
        Inizializza la pipeline
        
        Args:
            tenant: Oggetto Tenant (NUOVO - architettura UUID)
            tenant_slug: Slug del tenant (DEPRECATO - solo retrocompatibilità)
            confidence_threshold: Soglia di confidenza (None = legge da config)
            min_cluster_size: Dimensione minima del cluster (None = legge da config)
            min_samples: Numero minimo di campioni (None = legge da config)
            config_path: Percorso del file di configurazione
            auto_mode: Se True, modalità automatica (None = legge da config)
            shared_embedder: Embedder condiviso per evitare CUDA out of memory
        """
        trace_all("__init__", "ENTER", 
                 tenant=tenant, tenant_slug=tenant_slug, 
                 confidence_threshold=confidence_threshold,
                 min_cluster_size=min_cluster_size, min_samples=min_samples,
                 config_path=config_path, auto_mode=auto_mode)
        
        # 🎯 NUOVO SISTEMA: Crea oggetto Tenant UNA VOLTA con TUTTE le info
        from Utils.tenant import Tenant
        
        # GESTIONE TENANT UUID CENTRALIZZATA
        if tenant is not None:
            # Architettura moderna - usa oggetto Tenant diretto
            self.tenant = tenant
            print(f"🎯 [PIPELINE] Inizializzazione con oggetto Tenant: {tenant.tenant_name} ({tenant.tenant_id})")
        elif tenant_slug is not None:
            # Retrocompatibilità - converte parametro a oggetto Tenant
            print(f"⚠️ [PIPELINE] DEPRECATO: uso di tenant_slug '{tenant_slug}' - convertendo a oggetto Tenant")
            # Determina se il parametro è UUID o slug e crea oggetto Tenant
            if Tenant._is_valid_uuid(tenant_slug):
                self.tenant = Tenant.from_uuid(tenant_slug)
            else:
                self.tenant = Tenant.from_slug(tenant_slug)
            print(f"🔄 [PIPELINE] Conversione completata: {self.tenant.tenant_name} ({self.tenant.tenant_id})")
        else:
            # Fallback a humanitas default
            print(f"⚠️ [PIPELINE] Nessun tenant specificato - usando Humanitas default")
            self.tenant = Tenant.from_slug("humanitas")
            print(f"🏥 [PIPELINE] Fallback completato: {self.tenant.tenant_name} ({self.tenant.tenant_id})")
        
        # Mantieni retrocompatibilità per codice esistente
        self.tenant_id = self.tenant.tenant_id # UUID 
        self.tenant_slug = self.tenant.tenant_slug
        
        # Ogni operazione creerà istanza tenant-specifica quando necessario
        self.mongo_reader = None  # Placeholder - istanze create dinamicamente
        
        # 🆕 Inizializza helper per parametri tenant UMAP
        self.config_helper = TenantConfigHelper()
        
        # Carica configurazione
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        # Salva il percorso del config come attributo dell'istanza
        self.config_path = config_path
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Leggi parametri da configurazione con fallback ai valori passati
        pipeline_config = config.get('pipeline', {})
        clustering_config = config.get('clustering', {})
        
        # 🆕 CARICA PARAMETRI PERSONALIZZATI TENANT (se esistono)
        if hasattr(self, 'tenant_id') and self.tenant_id:
            tenant_config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tenant_configs')
            tenant_config_file = os.path.join(tenant_config_dir, f'{self.tenant_id}_clustering.yaml')
            
            if os.path.exists(tenant_config_file):
                try:
                    with open(tenant_config_file, 'r', encoding='utf-8') as f:
                        tenant_config = yaml.safe_load(f)
                        tenant_clustering_params = tenant_config.get('clustering_parameters', {})
                        
                        if tenant_clustering_params:
                            print(f"🎯 [PIPELINE CLUSTERING CONFIG] Caricati parametri personalizzati per tenant {self.tenant_id}:")
                            for param, value in tenant_clustering_params.items():
                                old_value = clustering_config.get(param, 'non_definito')
                                clustering_config[param] = value
                                print(f"   {param}: {old_value} -> {value}")
                        else:
                            print(f"📋 [PIPELINE CLUSTERING CONFIG] File config personalizzata vuoto per tenant {self.tenant_id}")
                            print(f"🔄 [PIPELINE CLUSTERING CONFIG] Usando configurazione default")
                            
                except Exception as e:
                    print(f"⚠️ [PIPELINE CLUSTERING CONFIG] Errore caricamento config personalizzata tenant {self.tenant_id}: {e}")
                    print("🔄 [PIPELINE CLUSTERING CONFIG] Fallback alla configurazione default da config.yaml")
                    # Non solleva eccezione - continua con parametri default
            else:
                print(f"� [PIPELINE CLUSTERING CONFIG] Nessun file personalizzato per tenant {self.tenant_id}")
                print(f"�🔄 [PIPELINE CLUSTERING CONFIG] Usando configurazione default da config.yaml")
        
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
        start_time = time.time()
        print(f"\n🚀 [FASE 1: INIZIALIZZAZIONE] Avvio pipeline...")
        print(f"🏥 [FASE 1: INIZIALIZZAZIONE] Tenant: {self.tenant_slug}")
        print(f"🎯 [FASE 1: INIZIALIZZAZIONE] Configurazione:")
        print(f"   📊 Confidence threshold: {self.confidence_threshold}")
        print(f"   🤖 Auto mode: {self.auto_mode}")
        print(f"   🔄 Auto retrain: {self.auto_retrain}")
        
        # Inizializza componenti base
        print(f"� [FASE 1: INIZIALIZZAZIONE] Inizializzazione lettore conversazioni...")
        self.lettore = LettoreConversazioni(tenant=self.tenant)
        
        print(f"� [FASE 1: INIZIALIZZAZIONE] Inizializzazione aggregator...")
        print(f"   🔍 Schema: '{self.tenant_slug}'")
        print(f"   🆔 Tenant ID: '{self.tenant_id}'")
        
        self.aggregator = SessionAggregator(tenant=self.tenant)
        
        # Gestione embedder
        if shared_embedder is not None:
            print("🔄 [FASE 1: INIZIALIZZAZIONE] Utilizzo embedder condiviso")
            self.embedder = shared_embedder
        else:
            print("🧠 [FASE 1: INIZIALIZZAZIONE] Sistema embedder dinamico (lazy loading)")
            self.embedder = None  # Sarà caricato quando serve tramite _get_embedder()
            
        # Configurazione clustering
        cluster_min_size = (min_cluster_size if min_cluster_size is not None 
                           else clustering_config.get('min_cluster_size', 
                                pipeline_config.get('default_min_cluster_size', 5)))
        cluster_min_samples = (min_samples if min_samples is not None 
                              else clustering_config.get('min_samples', 
                                   pipeline_config.get('default_min_samples', 3)))
        
        print(f"🔧 [FASE 1: INIZIALIZZAZIONE] Parametri clustering:")
        print(f"   📊 Min cluster size: {cluster_min_size}")
        print(f"   📊 Min samples: {cluster_min_samples}")
        
        # 🔧 [FIX] Passa TUTTI i parametri tenant-specific all'HDBSCANClusterer
        print(f"🐛 DEBUG CLUSTER_ALPHA - PRIMA:")
        print(f"   📋 clustering_config.get('alpha', 1.0): {clustering_config.get('alpha', 1.0)}")
        print(f"   📋 Type: {type(clustering_config.get('alpha', 1.0))}")
        
        cluster_alpha = float(clustering_config.get('alpha', 1.0))
        
        print(f"🐛 DEBUG CLUSTER_ALPHA - DOPO CONVERSIONE:")
        print(f"   📋 cluster_alpha: {cluster_alpha}")
        print(f"   📋 Type: {type(cluster_alpha)}")
        
        if cluster_alpha <= 0:
            print(f"⚠️ Alpha non valido ({cluster_alpha}), uso default 1.0")
            cluster_alpha = 1.0
            
        print(f"🐛 DEBUG CLUSTER_ALPHA - FINALE:")
        print(f"   📋 cluster_alpha finale: {cluster_alpha}")
        print(f"   📋 Type finale: {type(cluster_alpha)}")
        print(f"   📋 Alpha > 0: {cluster_alpha > 0}")
        cluster_selection_method = clustering_config.get('cluster_selection_method', 'eom')
        cluster_selection_epsilon = float(clustering_config.get('cluster_selection_epsilon', 0.05))
        cluster_metric = clustering_config.get('metric', 'cosine')
        cluster_allow_single = clustering_config.get('allow_single_cluster', False)
        # 🔧 FIX CRITICO: max_cluster_size=None causa TypeError in HDBSCAN
        # Protezione robusta: None o 0 -> 0 (unlimited), altrimenti valore intero
        max_cluster_raw = clustering_config.get('max_cluster_size', 0)
        # CAMBIO: None viene convertito a 0 invece che mantenuto None per evitare errori HDBSCAN
        if max_cluster_raw is None or max_cluster_raw == 0:
            cluster_max_size = 0  # 0 = unlimited in HDBSCAN (comportamento equivalente a None ma senza errori)
        else:
            cluster_max_size = int(max_cluster_raw)  # Valore esplicito
        
        # Debug della correzione
        if max_cluster_raw is None:
            print(f"🔧 [FIX] max_cluster_size: None -> 0 (protezione anti-errore HDBSCAN)")
        
        # � PARAMETRI UNIFICATI da MySQL - tutti i parametri in una sola chiamata
        print(f"📊 [PIPELINE] Caricamento parametri unificati per tenant: {self.tenant_id}")
        from Utils.tenant_config_helper import get_all_clustering_parameters_for_tenant
        unified_params = get_all_clustering_parameters_for_tenant(self.tenant_id)
        
        # 🔍 Debug parametri caricati
        print(f"   ✅ Parametri HDBSCAN: {len([k for k in unified_params.keys() if not k.startswith('umap_') and not k.endswith('_threshold') and k not in ['enable_smart_review', 'max_pending_per_batch', 'minimum_consensus_threshold']])}")
        print(f"   ✅ Parametri UMAP: {len([k for k in unified_params.keys() if k.startswith('umap_') or k == 'use_umap'])}")
        print(f"   ✅ Parametri Review Queue: {len([k for k in unified_params.keys() if k.endswith('_threshold') or k in ['enable_smart_review', 'max_pending_per_batch', 'minimum_consensus_threshold']])}")
        
        # 🎯 COSTRUZIONE DIZIONARI PARAMETRI PER BERTOPIC
        # Uso i parametri dal database MySQL unificato - override dei parametri da config.yaml
        bertopic_hdbscan_params = {
            'min_cluster_size': unified_params.get('min_cluster_size', cluster_min_size),
            'min_samples': unified_params.get('min_samples', cluster_min_samples),
            'alpha': unified_params.get('alpha', cluster_alpha),
            'cluster_selection_method': unified_params.get('cluster_selection_method', cluster_selection_method),
            'cluster_selection_epsilon': unified_params.get('cluster_selection_epsilon', cluster_selection_epsilon),
            'metric': unified_params.get('metric', cluster_metric),
            'allow_single_cluster': unified_params.get('allow_single_cluster', cluster_allow_single),
            'max_cluster_size': unified_params.get('max_cluster_size', cluster_max_size),
            # Parametri specifici per BERTopic
            'prediction_data': True,  # Necessario per BERTopic
            'match_reference_implementation': True  # Compatibilità
        }
        
        # Parametri UMAP per BERTopic (solo se UMAP è abilitato)
        bertopic_umap_params = None
        if unified_params.get('use_umap', False):
            bertopic_umap_params = {
                'n_neighbors': unified_params.get('n_neighbors', 10),
                'min_dist': unified_params.get('min_dist', 0.05),
                'n_components': unified_params.get('n_components', 3),
                'metric': unified_params.get('umap_metric', 'euclidean'),  # ✅ CORRETTO: usa metrica UMAP dal database
                'random_state': unified_params.get('random_state', 42)  # ✅ CORRETTO: usa random_state dal database
            }
            print(f"🗂️  [UMAP] Parametri caricati dal database MySQL: {bertopic_umap_params}")
        else:
            print(f"🗂️  [UMAP] Disabilitato per tenant {self.tenant_id}")
        
        print(f"🔧 [FIX DEBUG] Parametri tenant passati a HDBSCANClusterer:")
        print(f"   min_cluster_size: {cluster_min_size}")
        print(f"   min_samples: {cluster_min_samples}")
        print(f"   alpha: {cluster_alpha}")
        print(f"   cluster_selection_method: {cluster_selection_method}")
        print(f"   cluster_selection_epsilon: {cluster_selection_epsilon}")
        print(f"   metric: {cluster_metric}")
        print(f"   allow_single_cluster: {cluster_allow_single}")
        print(f"   max_cluster_size: {cluster_max_size}")
        print(f"   🗂️  use_umap: {unified_params.get('use_umap', False)}")
        if unified_params.get('use_umap', False):
            print(f"   🗂️  umap_n_neighbors: {unified_params.get('n_neighbors', 10)}")
            print(f"   🗂️  umap_min_dist: {unified_params.get('min_dist', 0.05)}")
            print(f"   🗂️  umap_n_components: {unified_params.get('n_components', 3)}")
        
        print(f"🎯 [BERTOPIC] Parametri consistenti configurati:")
        print(f"   📊 HDBSCAN: {len(bertopic_hdbscan_params)} parametri")
        if bertopic_umap_params:
            print(f"   📊 UMAP: {len(bertopic_umap_params)} parametri")
        else:
            print(f"   📊 UMAP: Disabilitato (usa embeddings pre-computati)")
        
        # 💾 MEMORIZZA PARAMETRI COME ATTRIBUTI DELLA CLASSE
        # Per poterli usare nella creazione del BERTopicFeatureProvider
        self.bertopic_hdbscan_params = bertopic_hdbscan_params
        self.bertopic_umap_params = bertopic_umap_params
        # 🎯 MEMORIZZA PARAMETRI UNIFICATI per uso globale nella pipeline
        self.unified_params = unified_params
        
        self.clusterer = HDBSCANClusterer(
            min_cluster_size=unified_params.get('min_cluster_size', cluster_min_size),
            min_samples=unified_params.get('min_samples', cluster_min_samples),
            alpha=unified_params.get('alpha', cluster_alpha),
            cluster_selection_method=unified_params.get('cluster_selection_method', cluster_selection_method),
            cluster_selection_epsilon=unified_params.get('cluster_selection_epsilon', cluster_selection_epsilon),
            metric=unified_params.get('metric', cluster_metric),
            allow_single_cluster=unified_params.get('allow_single_cluster', cluster_allow_single),
            max_cluster_size=unified_params.get('max_cluster_size', cluster_max_size),
            # 🆕 PARAMETRI UMAP dal database MySQL unificato
            use_umap=unified_params.get('use_umap', False),
            umap_n_neighbors=unified_params.get('n_neighbors', 10),
            umap_min_dist=unified_params.get('min_dist', 0.05),
            umap_metric=unified_params.get('umap_metric', 'euclidean'),
            umap_n_components=unified_params.get('n_components', 3),
            umap_random_state=unified_params.get('random_state', 42),
            config_path=config_path,
            tenant=self.tenant  # 🔧 PASSA OGGETTO TENANT AL CLUSTERER
        )
        # Non serve più classifier separato - tutto nell'ensemble
        # self.classifier = rimosso, ora tutto in ensemble_classifier
        self.tag_db = TagDatabaseConnector(tenant=self.tenant)
        
        # Inizializza il gestore della memoria semantica
        print("🧠 Inizializzazione memoria semantica...")
        self.semantic_memory = SemanticMemoryManager(
            tenant=self.tenant,  # Passa l'oggetto Tenant completo
            config_path=config_path,
            embedder=self.embedder
        )
        
        # Inizializza attributi per BERTopic pre-addestrato
        self._bertopic_provider_trained = None
        
        # 🚀 OTTIMIZZAZIONE: Cache per features ML pre-calcolate (evita ricalcolo BERTopic)
        self._ml_features_cache = {}  # session_id -> ml_features numpy array
        self._cache_valid_timestamp = None  # Timestamp validità cache
        print("🏗️ Cache features ML inizializzata per ottimizzazione BERTopic")
        
        # Inizializza l'ensemble classifier avanzato PRIMA (questo creerà il suo LLM internamente)
        print("🔗 Inizializzazione ensemble classifier avanzato...")
        self.ensemble_classifier = AdvancedEnsembleClassifier(
            llm_classifier=None,  # Creerà il suo IntelligentClassifier internamente
            confidence_threshold=self.confidence_threshold,
            adaptive_weights=True,
            performance_tracking=True,
            client_name=self.tenant_slug  # CORRETTO: Passa il tenant_slug per MongoDB collections
        )
        
        # Assegna client_name per retrocompatibilità
        self.client_name = self.tenant_slug
        
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
        self.interactive_trainer = InteractiveTrainer(
            tenant=self.tenant,  # Passa oggetto Tenant completo
            llm_classifier=llm_classifier, 
            auto_mode=self.auto_mode,
            bertopic_model=None  # Sarà inizializzato dopo il clustering
        )
        
        
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
        print(f"💾 [FASE 1: INIZIALIZZAZIONE] Caricamento memoria semantica...")
        if self.semantic_memory.load_semantic_memory():
            stats = self.semantic_memory.get_memory_stats()
            print(f"   📊 Campioni: {stats.get('memory_sessions', 0)}")
            print(f"   🏷️ Tag: {stats.get('total_tags', 0)}")
        else:
            print("   ⚠️ Memoria vuota (prima esecuzione)")
        
        initialization_time = time.time() - start_time
        print(f"✅ [FASE 1: INIZIALIZZAZIONE] Completata in {initialization_time:.2f}s")
        print(f"🎯 [FASE 1: INIZIALIZZAZIONE] Pipeline pronta per l'uso!")
        
        trace_all("__init__", "EXIT", initialization_time=initialization_time)
    
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
        trace_all("estrai_sessioni", "ENTER",
                 giorni_indietro=giorni_indietro,
                 limit=limit,
                 force_full_extraction=force_full_extraction)
        
        start_time = time.time()
        print(f"\n� [FASE 2: ESTRAZIONE] Avvio estrazione sessioni...")
        print(f"🏥 [FASE 2: ESTRAZIONE] Tenant: {self.tenant_slug}")
        print(f"📅 [FASE 2: ESTRAZIONE] Giorni indietro: {giorni_indietro}")
        
        # Controlla configurazione training supervisionato
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            supervised_config = config.get('supervised_training', {})
            extraction_config = supervised_config.get('extraction', {})
            
            # Se configurazione prevede estrazione completa, forza estrazione totale
            if extraction_config.get('use_full_dataset', False) or force_full_extraction:
                print(f"🔄 [FASE 2: ESTRAZIONE] Modalità COMPLETA attivata")
                print(f"🎯 [FASE 2: ESTRAZIONE] Ignorando limite - estrazione totale dataset")
                actual_limit = None
                extraction_mode = "COMPLETA"
            else:
                actual_limit = limit
                extraction_mode = "LIMITATA"
                print(f"🔢 [FASE 2: ESTRAZIONE] Limite sessioni: {limit}")
                
        except Exception as e:
            print(f"⚠️ [FASE 2: ESTRAZIONE] Errore config: {e}")
            actual_limit = limit
            extraction_mode = "LIMITATA"
        
        print(f"� [FASE 2: ESTRAZIONE] Modalità: {extraction_mode}")
        
        # Estrazione dal database
        print(f"� [FASE 2: ESTRAZIONE] Connessione database...")
        sessioni = self.aggregator.estrai_sessioni_aggregate(limit=actual_limit)
        
        if not sessioni:
            print("❌ [FASE 2: ESTRAZIONE] ERRORE: Nessuna sessione trovata")
            return {}
        
        print(f"📥 [FASE 2: ESTRAZIONE] Sessioni grezze: {len(sessioni)}")
        
        # Filtra sessioni vuote
        print(f"🔍 [FASE 2: ESTRAZIONE] Filtraggio sessioni...")
        sessioni_filtrate = self.aggregator.filtra_sessioni_vuote(sessioni)
        
        # Calcola statistiche filtri
        filtered_out = len(sessioni) - len(sessioni_filtrate)
        elapsed_time = time.time() - start_time
        
        if extraction_mode == "COMPLETA":
            print(f"✅ [FASE 2: ESTRAZIONE] Completata in {elapsed_time:.2f}s")
            print(f"📊 [FASE 2: ESTRAZIONE] Dataset completo: {len(sessioni_filtrate)} sessioni")
            print(f"🗑️ [FASE 2: ESTRAZIONE] Filtrate: {filtered_out} sessioni vuote/irrilevanti")
            print(f"🎯 [FASE 2: ESTRAZIONE] Pronte per clustering completo")
        else:
            print(f"✅ [FASE 2: ESTRAZIONE] Completata in {elapsed_time:.2f}s")
            print(f"📊 [FASE 2: ESTRAZIONE] Dataset limitato: {len(sessioni_filtrate)} sessioni")
            print(f"🗑️ [FASE 2: ESTRAZIONE] Filtrate: {filtered_out} sessioni vuote/irrilevanti")
            
        trace_all("estrai_sessioni", "EXIT", return_value_count=len(sessioni_filtrate))
        return sessioni_filtrate
    
    def _generate_cluster_info_from_labels(self, cluster_labels: np.ndarray, session_texts: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Genera cluster_info basico dai cluster labels per visualizzazione
        
        Args:
            cluster_labels: Array delle etichette cluster
            session_texts: Lista dei testi corrispondenti
            
        Returns:
            Dizionario cluster_info compatibile
        """
        cluster_info = {}
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label != -1:  # Esclude outlier
                indices = [i for i, l in enumerate(cluster_labels) if l == label]
                cluster_info[label] = {
                    'intent': f'cluster_hdbscan_{label}',
                    'size': len(indices),
                    'indices': indices,
                    'intent_string': f'Cluster HDBSCAN {label}',
                    'classification_method': 'hdbscan_optimized',
                    'average_confidence': 0.7  # Default per clustering ottimizzato
                }
        
        return cluster_info
    
    def _get_embedder(self):
        """
        Ottiene embedder per pipeline tramite sistema dinamico (lazy loading)
        
        Scopo: Carica embedder solo quando necessario, basato sulla configurazione
        tenant specifica, evitando caricamento all'avvio del server
        
        Returns:
            Embedder configurato per il tenant della pipeline
            
        Data ultima modifica: 2025-08-25
        """
        if self.embedder is None:
            print(f"🔄 LAZY LOADING: Caricamento embedder dinamico per tenant '{self.tenant_slug}'")
            
            # Import dinamico per evitare circular dependencies
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'EmbeddingEngine'))
                from simple_embedding_manager import simple_embedding_manager
                
                # NUOVO SISTEMA: SimpleEmbeddingManager con reset automatico
                # Passa l'oggetto Tenant corretto invece della stringa tenant_slug
                self.embedder = simple_embedding_manager.get_embedder_for_tenant(self.tenant)
                print(f"✅ Embedder caricato per tenant {self.tenant.tenant_name} ({self.tenant.tenant_id}): {type(self.embedder).__name__}")
                
            except ImportError as e:
                print(f"⚠️ Fallback: Impossibile importare simple_embedding_manager: {e}")
                print(f"🔄 Uso fallback LaBSE remoto/locale")
                
                # AGGIORNAMENTO 2025-08-29: Fallback coerente con servizio Docker
                try:
                    from labse_remote_client import LaBSERemoteClient
                    self.embedder = LaBSERemoteClient(service_url="http://localhost:8081")
                    print(f"✅ Fallback remoto configurato")
                except Exception as remote_error:
                    print(f"❌ CRITICO: Servizio Docker LaBSE non disponibile: {remote_error}")
                    raise RuntimeError(f"Servizio Docker LaBSE richiesto ma non disponibile: {remote_error}")
                
                
        return self.embedder
    
    def _get_embedder_name(self):
        """
        Ottiene il nome del modello di embedding attuale per il salvataggio in MongoDB.
        Include informazioni sulla riduzione dimensionale UMAP se applicata.
        
        Returns:
            str: Nome del modello/embedder utilizzato con info UMAP
        """
        if self.embedder is None:
            return "unknown_embedder"
        
        # Determina il nome basato sul tipo di embedder
        embedder_type = type(self.embedder).__name__
        
        base_name = ""
        if hasattr(self.embedder, 'model_name'):
            # Per modelli con attributo model_name (es. OpenAI, BGE-M3)
            base_name = f"{embedder_type}_{self.embedder.model_name}"
        elif hasattr(self.embedder, 'model_path'):
            # Per modelli con attributo model_path (es. LaBSE)
            model_path = str(self.embedder.model_path)
            if '/' in model_path:
                model_name = model_path.split('/')[-1]
            else:
                model_name = model_path
            base_name = f"{embedder_type}_{model_name}"
        else:
            # Fallback: solo il tipo
            base_name = embedder_type
        
        # 🆕 Aggiungi informazioni UMAP se disponibili
        if hasattr(self.clusterer, 'umap_info') and self.clusterer.umap_info.get('applied'):
            umap_params = self.clusterer.umap_info.get('parameters', {})
            n_components = umap_params.get('n_components', 'unknown')
            base_name += f"_UMAP{n_components}D"
            print(f"📏 [DEBUG EMBED] Embedding con UMAP: {base_name}")
        else:
            print(f"📏 [DEBUG EMBED] Embedding senza UMAP: {base_name}")
        
        return base_name

    def _analyze_and_show_problematic_conversations(self, sessioni: Dict[str, Dict], 
                                                   testi: List[str], 
                                                   session_ids: List[str], 
                                                   error_msg: str) -> None:
        """
        Analizza e mostra per intero le conversazioni troppo lunghe che causano errori di embedding.
        
        Scopo: Quando il modello di embedding fallisce per testo troppo lungo, mostra la conversazione
               completa in console per debugging e analisi.
        
        Args:
            sessioni: Dizionario con i dati delle sessioni
            testi: Lista dei testi completi delle conversazioni  
            session_ids: Lista degli ID delle sessioni corrispondenti ai testi
            error_msg: Messaggio di errore originale
        
        Data ultima modifica: 25/08/2025 - Implementazione iniziale
        """
        print(f"🔍 ANALISI CONVERSAZIONI PROBLEMATICHE PER LUNGHEZZA TESTO")
        print(f"=" * 80)
        print(f"📋 Errore originale: {error_msg}")
        print(f"📊 Numero totale conversazioni: {len(testi)}")
        print(f"=" * 80)
        
        # Calcola statistiche di lunghezza per identificare le conversazioni problematiche
        lunghezze = [(i, len(testo)) for i, testo in enumerate(testi)]
        lunghezze_ordinate = sorted(lunghezze, key=lambda x: x[1], reverse=True)
        
        print(f"📈 STATISTICHE LUNGHEZZA TESTI:")
        print(f"   Testo più lungo: {lunghezze_ordinate[0][1]} caratteri")
        print(f"   Testo più corto: {lunghezze_ordinate[-1][1]} caratteri")
        
        media_lunghezza = sum(len(testo) for testo in testi) / len(testi)
        print(f"   Lunghezza media: {media_lunghezza:.0f} caratteri")
        print(f"=" * 80)
        
        # Mostra le conversazioni più lunghe (potenziali problematiche)
        print(f"📋 CONVERSAZIONI PIÙ LUNGHE (POTENZIALI CAUSE DELL'ERRORE):")
        print(f"=" * 80)
        
        # Mostra le prime 3 conversazioni più lunghe per intero
        for rank, (indice, lunghezza) in enumerate(lunghezze_ordinate[:3], 1):
            session_id = session_ids[indice]
            testo_completo = testi[indice]
            dati_sessione = sessioni[session_id]
            
            print(f"\n🚨 CONVERSAZIONE #{rank} - LUNGHEZZA: {lunghezza} CARATTERI")
            print(f"🆔 Session ID: {session_id}")
            print(f"🤖 Agent: {dati_sessione.get('agent_name', 'N/A')}")
            print(f"💬 Numero messaggi: {dati_sessione.get('num_messaggi_totali', 0)}")
            print(f"👤 Messaggi USER: {dati_sessione.get('num_messaggi_user', 0)}")
            print(f"🤖 Messaggi AGENT: {dati_sessione.get('num_messaggi_agent', 0)}")
            print(f"⏰ Primo messaggio: {dati_sessione.get('primo_messaggio', 'N/A')}")
            print(f"⏰ Ultimo messaggio: {dati_sessione.get('ultimo_messaggio', 'N/A')}")
            print(f"📝 TESTO COMPLETO:")
            print(f"-" * 60)
            
            # MOSTRA IL TESTO COMPLETO SENZA OMETTERE NULLA
            print(testo_completo)
            
            print(f"-" * 60)
            print(f"📊 FINE CONVERSAZIONE #{rank}")
            print(f"=" * 80)
        
        # Suggerimenti per la risoluzione
        print(f"\n💡 SUGGERIMENTI PER RISOLVERE IL PROBLEMA:")
        print(f"   1. Configurare troncamento automatico del testo nel modello di embedding")
        print(f"   2. Implementare pre-processing per dividere conversazioni molto lunghe")  
        print(f"   3. Considerare modelli di embedding con context length maggiore")
        print(f"   4. Filtrare conversazioni anomalmente lunghe prima del clustering")
        print(f"   5. Verificare la configurazione 'only_user' per ridurre la lunghezza")
        print(f"=" * 80)
        
        # Log aggiuntivo per debug tecnico
        print(f"\n🔧 DEBUG TECNICO:")
        print(f"   Embedder attuale: {type(self._get_embedder()).__name__}")
        print(f"   Device embedder: {getattr(self._get_embedder(), 'device', 'N/A')}")
        
        if hasattr(self._get_embedder(), 'model'):
            model = self._get_embedder().model
            max_seq_length = getattr(model, 'max_seq_length', 'N/A')
            print(f"   Max sequence length: {max_seq_length}")
        
        print(f"=" * 80)
    
    def _addestra_bertopic_anticipato(self, sessioni: Dict[str, Dict], embeddings: np.ndarray) -> Optional[Any]:
        """
        Addestra BERTopic sui testi completi PRIMA del clustering per ottimizzare le features.
        
        Args:
            sessioni: Dizionario con le sessioni complete
            embeddings: Embedding LaBSE pre-generati
            
        Returns:
            BERTopicFeatureProvider addestrato o None se non abilitato/disponibile
            
        Autore: Valerio Bignardi
        Data creazione: 2025-09-06
        Ultima modifica: 2025-09-06 - Aggiunta del tracing completo
        """
        trace_all("_addestra_bertopic_anticipato", "ENTER", 
                 sessioni_count=len(sessioni),
                 embeddings_shape=str(embeddings.shape),
                 bertopic_enabled=self.bertopic_config.get('enabled', False))
        
        print(f"🤖 DEBUG: INIZIO TRAINING BERTOPIC ANTICIPATO")
        print(f"   📊 Sessioni ricevute: {len(sessioni)}")
        print(f"   📊 Embeddings shape: {embeddings.shape}")
        print(f"   🔧 BERTopic config enabled: {self.bertopic_config.get('enabled', False)}")
        print(f"   🔧 BERTopic disponibile: {_BERTopic_AVAILABLE}")
        
        if not self.bertopic_config.get('enabled', False):
            print("❌ BERTopic non abilitato nella configurazione, salto training anticipato")
            trace_all("_addestra_bertopic_anticipato", "EXIT", 
                     return_value=None, reason="BERTopic non abilitato")
            return None
            
        if not _BERTopic_AVAILABLE:
            print("❌ BERTopic SALTATO: Dipendenze non installate")
            print("   💡 Installare: pip install bertopic umap hdbscan")
            trace_all("_addestra_bertopic_anticipato", "EXIT", 
                     return_value=None, reason="BERTopic dipendenze non disponibili")
            return None
            
        n_samples = len(sessioni)
        
        # 🛠️ CONTROLLO DATASET SIZE: se troppo piccolo, salta BERTopic
        if n_samples < 25:  # Soglia minima per BERTopic affidabile
            print(f"⚠️ Dataset troppo piccolo per BERTopic ({n_samples} < 25 campioni)")
            print("   🔄 Salto training BERTopic - il sistema userà solo clustering LLM")
            
            # 🆕 SALVA WARNING PER INTERFACCIA UTENTE
            warning_info = {
                'type': 'dataset_too_small_for_bertopic',
                'current_size': n_samples,
                'minimum_required': 25,
                'recommended_size': 50,
                'message': f'Dataset troppo piccolo ({n_samples} campioni) per BERTopic clustering. ' +
                          f'Minimo: 25, raccomandato: 50+ campioni. Clustering semplificato in uso.',
                'impact': 'BERTopic clustering disabilitato, features ridotte per ML ensemble'
            }
            
            # Salva warning in un attributo della classe per il retrieval
            if not hasattr(self, 'training_warnings'):
                self.training_warnings = []
            self.training_warnings.append(warning_info)
            
            return None
            
        try:
            testi = [dati['testo_completo'] for dati in sessioni.values()]
            
            print(f"\n🚀 TRAINING BERTOPIC ANTICIPATO (NUOVO FLUSSO OTTIMIZZATO):")
            print(f"   📊 Dataset completo: {len(testi)} sessioni")
            print(f"   📊 Embeddings shape: {embeddings.shape}")
            print(f"   🎯 Addestramento su TUTTO il dataset per features ottimali")
            
            bertopic_provider = BERTopicFeatureProvider(
                use_svd=self.bertopic_config.get('use_svd', False),
                svd_components=self.bertopic_config.get('svd_components', 32),
                embedder=self.embedder,  # ✅ AGGIUNTO: passa embedder configurato
                hdbscan_params=self.bertopic_hdbscan_params,  # ✅ NUOVO: parametri HDBSCAN consistenti
                umap_params=self.bertopic_umap_params  # ✅ NUOVO: parametri UMAP consistenti
            )
            
            print("   🔥 Esecuzione bertopic_provider.fit() su dataset completo...")
            start_bertopic = time.time()
            # 🆕 NUOVA STRATEGIA: Lascia che BERTopic gestisca embeddings internamente
            # con l'embedder scelto dall'utente dall'interfaccia
            bertopic_provider.fit(testi)  # Non passa embeddings - BERTopic li calcola internamente
            fit_time = time.time() - start_bertopic
            print(f"   ✅ BERTopic FIT completato in {fit_time:.2f} secondi")
            
            print("   🔄 Esecuzione bertopic_provider.transform() per features extraction...")
            start_transform = time.time()
            tr = bertopic_provider.transform(
                testi,
                # Non passa embeddings - BERTopic usa embedder interno personalizzato
                return_one_hot=self.bertopic_config.get('return_one_hot', False),
                top_k=self.bertopic_config.get('top_k', None)
            )
            transform_time = time.time() - start_transform
            print(f"   ✅ BERTopic TRANSFORM completato in {transform_time:.2f} secondi")
            
            topic_probas = tr.get('topic_probas')
            one_hot = tr.get('one_hot')
            
            print(f"   📊 Topic probabilities shape: {topic_probas.shape if topic_probas is not None else 'None'}")
            print(f"   📊 One-hot shape: {one_hot.shape if one_hot is not None else 'None'}")
            print(f"   ✅ BERTopic provider addestrato con successo su {len(testi)} sessioni")
            
            trace_all("_addestra_bertopic_anticipato", "EXIT", 
                     return_value="BERTopicProvider", success=True,
                     processed_texts=len(testi))
            
            return bertopic_provider
            
        except Exception as e:
            print(f"❌ ERRORE Training BERTopic anticipato: {e}")
            print(f"   🔍 Traceback: {traceback.format_exc()}")
            
            trace_all("_addestra_bertopic_anticipato", "ERROR", 
                     error_message=str(e), return_value=None)
            
            return None

    def esegui_clustering(self, sessioni: Dict[str, Dict], force_reprocess: bool = False) -> tuple:
        """
        Esegue il clustering delle sessioni con approccio intelligente multi-livello:
        
        NUOVO: Supporta clustering incrementale vs completo
        - force_reprocess=False: Usa clustering incrementale se modello disponibile
        - force_reprocess=True: Clustering completo da zero
        
        1. BERTopic training anticipato su dataset completo (NUOVO)
        2. LLM per comprensione linguaggio naturale (primario)
        3. Pattern regex per fallback veloce (secondario)  
        4. Validazione umana per casi ambigui (terziario)
        
        Args:
            sessioni: Dizionario con le sessioni
            force_reprocess: Se True, forza clustering completo
            
        Returns:
            Tuple (embeddings, cluster_labels, representatives, suggested_labels)
        """
        trace_all("esegui_clustering", "ENTER", 
                 num_sessioni=len(sessioni), force_reprocess=force_reprocess)
        
        start_time = time.time()
        print(f"\n🚀 Avvio clustering intelligente...")
        print(f"📊 Dataset: {len(sessioni)} sessioni")
        print(f"🎯 Modalità: {'COMPLETO' if force_reprocess else 'INTELLIGENTE'}")
        
        # Assicurati che la directory dei modelli esista
        import os
        os.makedirs("models", exist_ok=True)
        
        if force_reprocess:
            print(f"🔄 [FASE 4: CLUSTERING] Clustering completo da zero...")
            result = self._esegui_clustering_completo(sessioni)
        else:
            print(f"🧠 [FASE 4: CLUSTERING] Clustering incrementale (se possibile)...")
            result = self._esegui_clustering_completo(sessioni)  # Per ora sempre completo
        
        # Calcola statistiche finali
        embeddings, cluster_labels, representatives, suggested_labels = result
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_outliers = list(cluster_labels).count(-1)
        n_representatives = sum(len(reps) for reps in representatives.values())
        
        # 🆕 DEBUG DETTAGLIATO RISULTATI CLUSTERING
        total_sessions = len(cluster_labels)
        unique_clusters = set(cluster_labels)
        
        print(f"\n🔍 [FASE 4: DEBUG] ANALISI DETTAGLIATA CLUSTERING:")
        print(f"   📊 Sessioni totali processate: {total_sessions}")
        print(f"   🎯 Cluster identificati: {unique_clusters}")
        print(f"   📈 Cluster validi (>= 0): {n_clusters}")
        print(f"   🔍 Outliers (-1): {n_outliers}")
        print(f"   📋 Representatives generati: {n_representatives}")
        print(f"   🏷️ Suggested labels: {len(suggested_labels)}")
        
        # Debug distribuzione cluster
        if total_sessions > 0:
            outlier_percentage = (n_outliers / total_sessions * 100)
            clustered_percentage = ((total_sessions - n_outliers) / total_sessions * 100)
            
            print(f"   📊 Distribuzione clustering:")
            print(f"     ✅ Clusterizzate: {total_sessions - n_outliers} ({clustered_percentage:.1f}%)")
            print(f"     🔍 Outliers: {n_outliers} ({outlier_percentage:.1f}%)")
            
            # Analisi per cluster specifico
            if n_clusters > 0:
                cluster_sizes = {}
                for label in cluster_labels:
                    if label != -1:
                        cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
                
                print(f"   📈 Dimensioni cluster:")
                for cluster_id, size in sorted(cluster_sizes.items()):
                    print(f"     🎯 Cluster {cluster_id}: {size} sessioni")
            
            # 🚨 ANALISI QUALITÀ CLUSTERING
            if outlier_percentage > 80:
                print(f"   ⚠️ WARNING: {outlier_percentage:.1f}% outliers - clustering potrebbe fallire!")
            elif outlier_percentage > 60:
                print(f"   ⚠️ ATTENZIONE: {outlier_percentage:.1f}% outliers - qualità clustering bassa")
            else:
                print(f"   ✅ BUONO: {outlier_percentage:.1f}% outliers - clustering accettabile")
        
        elapsed_time = time.time() - start_time
        print(f"✅ Clustering completato in {elapsed_time:.2f}s")
        print(f"📈 Risultati finali del clustering:")
        print(f"   🎯 Cluster trovati: {n_clusters}")
        print(f"   🔍 Outliers: {n_outliers}")
        print(f"   👥 Rappresentanti: {n_representatives}")
        print(f"   🏷️ Etichette generate: {len(suggested_labels)}")
        
        # 🚨 EARLY WARNING se clustering sembra fragile
        if n_clusters == 0:
            print(f"\n❌ [WARNING] CLUSTERING POTENZIALMENTE FALLITO!")
            print(f"   🔍 Tutti i {total_sessions} punti sono outlier")
            print(f"   💡 Il training supervisionato verrà interrotto")
        elif n_clusters < 2:
            print(f"\n⚠️ [WARNING] CLUSTERING DEBOLE!")
            print(f"   🎯 Solo {n_clusters} cluster trovato")
            print(f"   💡 Diversità limitata per training ML")
        
        trace_all("esegui_clustering", "EXIT", return_value=result)
        return result
    
    def _esegui_clustering_completo(self, sessioni: Dict[str, Dict]) -> tuple:
        """
        Esegue il clustering delle sessioni con approccio intelligente multi-livello:
        1. BERTopic training anticipato su dataset completo (NUOVO)
        2. LLM per comprensione linguaggio naturale (primario)
        3. Pattern regex per fallback veloce (secondario)  
        4. Validazione umana per casi ambigui (terziario)
        
        Args:
            sessioni: Dizionario con le sessioni
            
        Returns:
            Tuple (embeddings, cluster_labels, representatives, suggested_labels)
        """
        trace_all("_esegui_clustering_completo", "ENTER", sessioni_count=len(sessioni))
        
        # FASE 3: GENERAZIONE EMBEDDINGS
        start_time = time.time()
        print(f"\n🚀 [FASE 3: EMBEDDINGS] Avvio generazione embeddings...")
        print(f"� [FASE 3: EMBEDDINGS] Dataset: {len(sessioni)} sessioni")
        
        testi = [dati['testo_completo'] for dati in sessioni.values()]
        session_ids = list(sessioni.keys())
        
        # Analizza caratteristiche dataset
        lunghezze = [len(testo) for testo in testi]
        avg_length = sum(lunghezze) / len(lunghezze)
        max_length = max(lunghezze)
        min_length = min(lunghezze)
        
        print(f"📊 [FASE 3: EMBEDDINGS] Caratteristiche testo:")
        print(f"   📏 Lunghezza media: {avg_length:.0f} caratteri")
        print(f"   📏 Lunghezza massima: {max_length} caratteri")
        print(f"   📏 Lunghezza minima: {min_length} caratteri")
        
        try:
            print(f"🧠 [FASE 3: EMBEDDINGS] Generazione embeddings...")
            embeddings = self._get_embedder().encode(testi, show_progress_bar=True, session_ids=session_ids)
            
            embedding_time = time.time() - start_time
            print(f"✅ [FASE 3: EMBEDDINGS] Completata in {embedding_time:.2f}s")
            print(f"📈 [FASE 3: EMBEDDINGS] Shape: {embeddings.shape}")
            print(f"⚡ [FASE 3: EMBEDDINGS] Throughput: {len(testi)/embedding_time:.1f} testi/secondo")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ [FASE 3: EMBEDDINGS] ERRORE: {error_msg}")
            print(f"❌ ERRORE durante generazione embeddings: {error_msg}")
            
            # Controlla se è un errore di lunghezza del testo/token limit
            if any(keyword in error_msg.lower() for keyword in [
                'context length', 'token limit', 'too long', 'maximum context', 
                'sequence length', 'input too long', 'context size'
            ]):
                print(f"\n🚨 ERRORE DI LUNGHEZZA TESTO RILEVATO!")
                print(f"🔍 Il sistema ha trovato testo troppo lungo per il modello di embedding.")
                print(f"📋 Analizzando le conversazioni per identificare quella problematica...\n")
                
                # Trova e mostra la conversazione più lunga che ha causato l'errore
                self._analyze_and_show_problematic_conversations(sessioni, testi, session_ids, error_msg)
            
            # Re-raise l'errore dopo aver mostrato le informazioni
            raise e
        
        # 🆕 NUOVO: Training BERTopic anticipato su dataset completo
        print(f"\n📊 [FASE 2.2: TRAINING BERTOPIC (FEATURE AUGMENTATION)]")
        print(f"🔍 DEBUG: Avvio training BERTopic anticipato...")
        print(f"   📊 Sessioni per training: {len(sessioni)}")
        print(f"   📊 Embeddings shape: {embeddings.shape}")
        print(f"   🔧 Config BERTopic enabled: {self.bertopic_config.get('enabled', False)}")
        
        self._bertopic_provider_trained = self._addestra_bertopic_anticipato(sessioni, embeddings)
        
        print(f"🔍 DEBUG: Risultato training BERTopic:")
        print(f"   📋 Provider creato: {self._bertopic_provider_trained is not None}")
        if self._bertopic_provider_trained:
            print(f"   ✅ BERTopic provider disponibile per augmentation features")
            print(f"   📊 Tipo provider: {type(self._bertopic_provider_trained)}")
            if hasattr(self._bertopic_provider_trained, 'model'):
                print(f"   📊 Modello interno: {type(self._bertopic_provider_trained.model)}")
                print(f"   📊 Numero topic: {len(self._bertopic_provider_trained.model.get_topics()) if hasattr(self._bertopic_provider_trained.model, 'get_topics') else 'N/A'}")
            # Assegna il modello BERTopic al trainer interattivo per validazione "altro"
            if hasattr(self._bertopic_provider_trained, 'model'):
                self.interactive_trainer.bertopic_model = self._bertopic_provider_trained.model
                print(f"   🔗 BERTopic model assegnato al trainer per validazione ALTRO")
        else:
            print(f"   ❌ BERTopic provider NON CREATO!")
            print(f"   ⚠️ BERTopic provider non disponibile, proseguo con sole embeddings")
        
        print(f"\n📊 [FASE 2.3: ESECUZIONE CLUSTERING (HDBSCAN/INTELLIGENT)]")
        
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
            print(f"🧠 Usando clustering INTELLIGENTE (ML+LLM ensemble senza pattern)")
            
            # 🔧 CORREZIONE 2025-09-04: Passa ensemble_classifier completo per usare ML+LLM negli outlier
            intelligent_clusterer = IntelligentIntentClusterer(
                tenant=self.tenant,  # Passa oggetto Tenant completo
                config_path=self.clusterer.config_path,
                llm_classifier=self.ensemble_classifier.llm_classifier if self.ensemble_classifier else None,
                ensemble_classifier=self.ensemble_classifier  # 🆕 Passa ensemble completo
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
        
        # ✅ OUTLIER GESTITI AUTOMATICAMENTE
        # Gli outlier non hanno bisogno di "rappresentanti" artificiali perché:
        # 1. Vengono classificati individualmente con ensemble ML+LLM in fase runtime
        # 2. Non beneficiano della propagazione (sono rappresentanti di se stessi)
        # 3. Il review umano si concentra sui cluster veri per il training supervisionato
        outlier_indices = [i for i, label in enumerate(cluster_labels) if label == -1]
        n_outliers = len(outlier_indices)
        
        if n_outliers > 0:
            print(f"🔍 Trovati {n_outliers} outlier - verranno classificati individualmente con ensemble ML+LLM")
        
        # Statistiche clustering avanzate
        n_clusters = len(cluster_info)
        
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
        
        # Salva i dati dell'ultimo clustering per le visualizzazioni
        self._last_embeddings = embeddings
        self._last_cluster_labels = cluster_labels
        
        # Genera cluster info usando i session texts
        session_texts = [sessioni[sid].get('testo_completo', '') for sid in list(sessioni.keys())]
        self._last_cluster_info = self._generate_cluster_info_from_labels(cluster_labels, session_texts)
        
        # NUOVA FUNZIONALITÀ: Visualizzazione grafica cluster
        try:
            from Utils.cluster_visualization import ClusterVisualizationManager
            
            visualizer = ClusterVisualizationManager()
            session_texts = [sessioni[sid].get('testo_completo', '') for sid in list(sessioni.keys())]
            
            # Visualizzazione per PARAMETRI CLUSTERING (senza etichette finali)
            print("\n🎨 GENERAZIONE VISUALIZZAZIONI PARAMETRI CLUSTERING...")
            visualization_results = visualizer.visualize_clustering_parameters(
                embeddings=embeddings,
                cluster_labels=cluster_labels,
                cluster_info=cluster_info,
                session_texts=session_texts,
                save_html=True,
                show_console=True
            )
            
        except ImportError:
            print("⚠️ Sistema visualizzazione non disponibile - installare plotly")
        except Exception as e:
            print(f"⚠️ Errore nella visualizzazione cluster: {e}")
        
        # 🆕 SALVA MODELLO PER PREDIZIONI FUTURE
        model_path = f"models/hdbscan_{self.tenant_id}.pkl"
        if hasattr(self.clusterer, 'save_model_for_incremental_prediction'):
            saved = self.clusterer.save_model_for_incremental_prediction(model_path, self.tenant_id)
        if saved:
            print(f"💾 Modello HDBSCAN salvato per predizioni incrementali: {model_path}")
        else:
            print(f"⚠️ Impossibile salvare modello HDBSCAN")
        
        result = (embeddings, cluster_labels, representatives, suggested_labels)
        trace_all("_esegui_clustering_completo", "EXIT", 
                 embeddings_shape=embeddings.shape if hasattr(embeddings, 'shape') else len(embeddings),
                 cluster_labels_count=len(cluster_labels),
                 representatives_clusters=len(representatives),
                 suggested_labels_count=len(suggested_labels))
        return result    # ❌ METODO NON UTILIZZATO - COMMENTATO 2025-09-06
    # Questo metodo è stato sostituito da _classify_and_save_representatives_post_training()
    # che viene chiamato DOPO il training ML per salvare rappresentanti con predizioni complete
    # Il vecchio metodo salvava solo etichette dal clustering PRIMA del training ML
    # causando il problema di rappresentanti con ml_prediction/llm_prediction = N/A
    # def _save_representatives_for_review(self, 
    #                                    sessioni: Dict[str, Dict], 
    #                                    representatives: Dict[int, List[Dict]], 
    #                                    suggested_labels: Dict[int, str],
    #                                    cluster_labels: np.ndarray) -> bool:
    #     """
    #     Salva i rappresentanti in MongoDB come "pending review" PRIMA della review umana
    #     
    #     Scopo della funzione: Popolare la review queue con rappresentanti, outlier e propagati
    #     Parametri di input: sessioni, representatives, suggested_labels, cluster_labels
    #     Parametri di output: success flag
    #     Valori di ritorno: True se salvato con successo
    #     Tracciamento aggiornamenti: 2025-08-28 - Fix review queue mancante
    #     
    #     Args:
    #         sessioni: Tutte le sessioni del dataset
    #         representatives: Dict {cluster_id: [rappresentanti]}
    #         suggested_labels: Dict {cluster_id: etichetta_suggerita}
    #         cluster_labels: Array delle etichette cluster per tutte le sessioni
    #         
    #     Returns:
    #         bool: True se salvato con successo
    #         
    #     Autore: Valerio Bignardi
    #     Data: 2025-08-28
    #     """
    #     start_time = time.time()
    #     print(f"\n� [FASE 7: SALVATAGGIO] Avvio salvataggio rappresentanti...")
    #     
    #     try:
    #         # 🆕 Crea istanza MongoClassificationReader per salvataggio
    #         from mongo_classification_reader import MongoClassificationReader
    #         mongo_reader = MongoClassificationReader(tenant=self.tenant)
    #         print("✅ [FASE 7: SALVATAGGIO] MongoDB reader creato per tenant")
    #         
    #         saved_count = 0
    #         failed_count = 0
    #         total_to_save = sum(len(reps) for reps in representatives.values())
    #         
    #         print(f"📊 [FASE 7: SALVATAGGIO] Target: {total_to_save} rappresentanti")
    #         print(f"🏷️ [FASE 7: SALVATAGGIO] Cluster: {list(representatives.keys())}")
    #         
    #         # Salva rappresentanti per ogni cluster
    #         for cluster_id, cluster_reps in representatives.items():
    #             suggested_label = suggested_labels.get(cluster_id, f"Cluster {cluster_id}")
    #             
    #             print(f"📋 [FASE 7: SALVATAGGIO] Cluster {cluster_id}: {len(cluster_reps)} rappresentanti")
    #             print(f"   🏷️ Etichetta: '{suggested_label}'")
    #             
    #             for rep_data in cluster_reps:
    #                 session_id = rep_data.get('session_id')
    #                 conversation_text = rep_data.get('testo_completo', '')
    #                 
    #                 # Prepara metadati cluster per distinguere tipi di sessioni
    #                 cluster_metadata = {
    #                     'cluster_id': cluster_id,
    #                     'is_representative': True,  # ✅ È un rappresentante
    #                     'cluster_size': len([1 for label in cluster_labels if label == cluster_id]),
    #                     'suggested_label': suggested_label,
    #                     'selection_reason': 'cluster_representative'
    #                 }
    #                 
    #                 # Metadati speciali per outlier
    #                 if cluster_id == -1:
    #                     cluster_metadata['selection_reason'] = 'outlier_representative'
    #                     cluster_metadata['is_outlier'] = True
    #                 
    #                 # 🧹 PULIZIA CRITICA: Applica pulizia caratteri speciali a suggested_label
    #                 clean_suggested_label = clean_label_text(suggested_label)
    #                 if clean_suggested_label != suggested_label:
    #                     print(f"🧹 Suggested label pulita: '{suggested_label}' → '{clean_suggested_label}'")
    #                     suggested_label = clean_suggested_label
    #                 
    #                 # Prepara decision finale per rappresentanti
    #                 final_decision = {
    #                     'predicted_label': suggested_label,  # Ora è pulita
    #                     'confidence': 0.7,  # Confidenza media per clustering
    #                     'method': 'supervised_training_clustering',
    #                     'reasoning': f'Rappresentante del cluster {cluster_id} selezionato per review umana'
    #                 }
    #                 
    #                 # Salva in MongoDB come "pending review"
    #                 success = mongo_reader.save_classification_result(
    #                     session_id=session_id,
    #                     client_name=self.tenant.tenant_slug,  # 🔧 FIX: usa tenant_slug non tenant_id
    #                     final_decision=final_decision,
    #                     conversation_text=conversation_text,
    #                     needs_review=True,  # ✅ FONDAMENTALE: marca per review
    #                     review_reason='supervised_training_representative',
    #                     classified_by='supervised_training_pipeline',
    #                     notes=f'Rappresentante cluster {cluster_id} per training supervisionato',
    #                     cluster_metadata=cluster_metadata
    #                 )
    #                 
    #                 if success:
    #                     saved_count += 1
    #                 else:
    #                     failed_count += 1
    #                     print(f"   ❌ ERRORE salvando {session_id}")
    #         
    #         # 🆕 SALVA ANCHE LE SESSIONI PROPAGATE (non rappresentanti)
    #         print(f"📋 [FASE 7: SALVATAGGIO] Salvataggio sessioni propagate...")
    #         propagated_count = self._save_propagated_sessions_metadata(
    #             sessioni, representatives, cluster_labels, suggested_labels
    #         )
    #         
    #         elapsed_time = time.time() - start_time
    #         print(f"✅ [FASE 7: SALVATAGGIO] Completata in {elapsed_time:.2f}s")
    #         print(f"� [FASE 7: SALVATAGGIO] Risultati:")
    #         print(f"   ✅ Rappresentanti salvati: {saved_count}/{total_to_save}")
    #         print(f"   📋 Sessioni propagate: {propagated_count}")
    #         print(f"   ❌ Errori: {failed_count}")
    #         print(f"   🎯 Review queue popolata: {saved_count + propagated_count} sessioni totali")
    #         
    #         return saved_count > 0
    #         
    #     except Exception as e:
    #         elapsed_time = time.time() - start_time
    #         print(f"❌ [FASE 7: SALVATAGGIO] ERRORE dopo {elapsed_time:.2f}s: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return False
    
    def _determine_propagated_status(self, 
                                   cluster_representatives: List[Dict],
                                   consensus_threshold: float = 0.7) -> Dict:
        """
        Determina lo status dei propagated basandosi sui rappresentanti del cluster
        
        Scopo della funzione: I propagati vengono SEMPRE auto-classificati, mai review automatico
        Parametri di input: cluster_representatives, consensus_threshold
        Parametri di output: Dict con status e label da propagare
        Valori di ritorno: needs_review (sempre False), propagated_label, reason
        Tracciamento aggiornamenti: 2025-09-05 - CORREZIONE: Propagati mai in review automatico
        
        Args:
            cluster_representatives: Lista rappresentanti del cluster
            consensus_threshold: Soglia di consenso (default 70%)
        
        Returns:
            Dict con status e label da propagare
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        
        # 1. Conta rappresentanti reviewed vs non-reviewed
        reviewed_reps = [r for r in cluster_representatives 
                        if r.get('human_reviewed', False)]
        
        # 2. Se nessuno è stato reviewed → TUTTI in review
        if len(reviewed_reps) == 0:
            return {
                'needs_review': True,
                'propagated_label': None,
                'reason': 'no_reviewed_representatives'
            }
        
        # 3. Analizza le classificazioni dei reviewed
        reviewed_labels = [r.get('classification') for r in reviewed_reps]
        label_counts = {}
        for label in reviewed_labels:
            if label:  # Esclude None e valori vuoti
                label_counts[label] = label_counts.get(label, 0) + 1
        
        if not label_counts:
            # Nessuna label valida nei reviewed
            return {
                'needs_review': True,
                'propagated_label': None,
                'reason': 'no_valid_labels'
            }
        
        # 4. Trova la label più votata
        most_voted_label = max(label_counts.keys(), key=lambda k: label_counts[k])
        consensus_ratio = label_counts[most_voted_label] / len(reviewed_labels)
        
        # 5. Decisione basata su consenso (soglia 0.7 = 70%)
        if consensus_ratio >= consensus_threshold:
            # CONSENSO FORTE → Auto-classifica propagated
            return {
                'needs_review': False,
                'propagated_label': most_voted_label,
                'reason': f'consensus_{int(consensus_ratio*100)}%'
            }
        elif consensus_ratio == 0.5 and len(reviewed_labels) == 2:
            # 🚫 CORREZIONE: Anche caso 50-50 viene auto-classificato
            # Non più review obbligatoria - usa label più votata
            return {
                'needs_review': False,  # 🚫 FORZA: Mai review automatico
                'propagated_label': most_voted_label,
                'reason': 'auto_classified_despite_50_50_split'
            }
        else:
            # 🚫 CORREZIONE: PROPAGATED NON vanno MAI automaticamente in review
            # Anche con disaccordo, vengono auto-classificati con label più votata
            return {
                'needs_review': False,  # 🚫 FORZA: Mai review automatico per propagati
                'propagated_label': most_voted_label,
                'reason': f'auto_classified_despite_disagreement_{int(consensus_ratio*100)}%'
            }
    

    def _save_propagated_sessions_metadata(self, 
                                         sessioni: Dict[str, Dict],
                                         representatives: Dict[int, List[Dict]], 
                                         cluster_labels: np.ndarray,
                                         suggested_labels: Dict[int, str]) -> int:
        """
        Salva metadati per le sessioni non-rappresentanti (propagate)
        
        Scopo della funzione: Marcare sessioni propagate per filtri interfaccia
        Parametri di input: sessioni, representatives, cluster_labels, suggested_labels  
        Parametri di output: numero sessioni salvate
        Valori di ritorno: conteggio sessioni processate
        Tracciamento aggiornamenti: 2025-08-28 - Nuovo per gestione filtri
        
        Args:
            sessioni: Tutte le sessioni
            representatives: Representatives per cluster
            cluster_labels: Labels per ogni sessione
            suggested_labels: Labels suggerite per cluster
            
        Returns:
            int: Numero di sessioni propagate salvate
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        print(f"   🔄 Salvando metadati sessioni propagate...")
        
        # 🆕 Crea istanza MongoClassificationReader per salvataggio propagated
        from mongo_classification_reader import MongoClassificationReader
        mongo_reader = MongoClassificationReader(tenant=self.tenant)
        
        # Crea set di session_id rappresentanti per escluderli
        representative_session_ids = set()
        for cluster_reps in representatives.values():
            for rep in cluster_reps:
                representative_session_ids.add(rep.get('session_id'))
        
        session_ids = list(sessioni.keys())
        saved_count = 0
        
        # Processa tutte le sessioni non-rappresentanti
        for i, session_id in enumerate(session_ids):
            if session_id not in representative_session_ids:
                cluster_id = cluster_labels[i] if i < len(cluster_labels) else -1
                suggested_label = suggested_labels.get(cluster_id, 'altro')
                
                # Prepara metadati per sessione
                cluster_metadata = {
                    'cluster_id': cluster_id,
                    'is_representative': False,  # ✅ NON è rappresentante
                    'suggested_label': suggested_label,
                }
                
                # 🚨 CORREZIONE CRITICA: Gli OUTLIER non devono essere propagati!
                if cluster_id == -1:
                    # OUTLIER - deve essere classificato individualmente
                    cluster_metadata['selection_reason'] = 'outlier_individual_classification'
                    # NON aggiungiamo 'propagated_from' per outlier
                else:
                    # MEMBRO DI CLUSTER - può essere propagato
                    cluster_metadata['propagated_from'] = f'cluster_{cluster_id}'
                    cluster_metadata['selection_reason'] = 'cluster_propagated'
                    cluster_metadata['is_outlier'] = True
                
                # 🧹 PULIZIA CRITICA: Applica pulizia caratteri speciali a suggested_label
                clean_suggested_label = clean_label_text(suggested_label)
                if clean_suggested_label != suggested_label:
                    print(f"🧹 Propagated label pulita: '{suggested_label}' → '{clean_suggested_label}'")
                    suggested_label = clean_suggested_label
                
                final_decision = {
                    'predicted_label': suggested_label,  # Ora è pulita
                    'confidence': 0.6,  # Confidenza più bassa per propagate
                    'method': 'supervised_training_propagated',
                    'reasoning': f'Sessione propagata dal cluster {cluster_id} durante training supervisionato'
                }
                
                # Salva come pending review (TUTTE le sessioni propagate devono essere reviewabili)
                success = mongo_reader.save_classification_result(
                    session_id=session_id,
                    client_name=self.tenant.tenant_slug,  # 🔧 FIX: usa tenant_slug non tenant_id
                    final_decision=final_decision,
                    conversation_text=sessioni[session_id].get('testo_completo', ''),
                    needs_review=True,  # ✅ ANCHE LE PROPAGATE devono essere pending per filtri
                    review_reason='supervised_training_propagated',
                    classified_by='supervised_training_pipeline',
                    notes=f'Sessione propagata da cluster {cluster_id}',
                    cluster_metadata=cluster_metadata
                )
                
                if success:
                    saved_count += 1
        
        return saved_count
    
    def update_propagated_after_review(self, cluster_id: int, tenant_uuid: str):
        """
        Aggiorna status propagated dopo review umana di un rappresentante
        
        Scopo della funzione: Trigger automatico dopo review umana rappresentanti
        Parametri di input: cluster_id del cluster reviewato, tenant_uuid del tenant
        Parametri di output: numero sessioni propagate aggiornate
        Valori di ritorno: conteggio aggiornamenti effettuati
        Tracciamento aggiornamenti: 2025-08-29 - Modificato per oggetto Tenant
        
        Args:
            cluster_id: ID del cluster i cui rappresentanti sono stati reviewed
            tenant_uuid: UUID del tenant per creare l'oggetto Tenant
            
        Returns:
            int: Numero di sessioni propagate aggiornate
            
        Nota: Chiamato automaticamente ogni volta che un rappresentante viene reviewed
        
        Autore: Valerio Bignardi  
        Data: 2025-08-29
        """
        
        try:
            # Crea oggetto Tenant
            tenant = Tenant.from_uuid(tenant_uuid)
            
            from mongo_classification_reader import MongoClassificationReader
            mongo_reader = MongoClassificationReader(tenant=tenant)
            mongo_reader.connect()  # Stabilisce connessione database
            
            # 1. Ottieni tutti i rappresentanti del cluster
            representatives = mongo_reader.get_cluster_representatives(
                self.tenant.tenant_slug, cluster_id
            )
            
            if not representatives:
                print(f"⚠️ Nessun rappresentante trovato per cluster {cluster_id}")
                return 0
            
            # 2. Determina status propagated usando logica intelligente
            propagated_status = self._determine_propagated_status(representatives)
            
            print(f"🔄 Cluster {cluster_id}: {propagated_status['reason']}")
            
            # 3. Se c'è consenso, aggiorna tutti i propagated del cluster
            if not propagated_status['needs_review']:
                updated_count = mongo_reader.update_cluster_propagated(
                    client_name=self.tenant.tenant_slug,
                    cluster_id=cluster_id,
                    final_label=propagated_status['propagated_label'],
                    review_status='auto_classified',
                    classified_by='consensus_' + propagated_status['reason'],
                    notes=f"Auto-classificati per consenso: {propagated_status['reason']}"
                )
                
                print(f"✅ {updated_count} sessioni propagate auto-classificate per consenso")
                return updated_count
            else:
                print(f"👤 Sessioni propagate rimangono in review: {propagated_status['reason']}")
                return 0
                
        except Exception as e:
            print(f"❌ Errore aggiornamento propagated per cluster {cluster_id}: {e}")
            import traceback
            traceback.print_exc()
            return 0


    def _classify_and_save_representatives_post_training(self,
                                                       sessioni: Dict[str, Dict],
                                                       representatives: Dict[int, List[Dict]],
                                                       suggested_labels: Dict[int, str],
                                                       cluster_labels: np.ndarray,
                                                       reviewed_labels: Dict[int, str]) -> bool:
        """
        Classifica e salva i rappresentanti dei cluster DOPO il training ML
        con predizioni complete ML+LLM invece di sole etichette suggerite
        
        Scopo della funzione: Popolare review queue con rappresentanti classificati
        Parametri di input: sessioni, representatives, suggested_labels, cluster_labels, reviewed_labels
        Parametri di output: success flag
        Valori di ritorno: True se classificato e salvato con successo
        Tracciamento aggiornamenti: 2025-09-06 - Valerio Bignardi - Fix N/A predictions
        
        Args:
            sessioni: Tutte le sessioni del dataset
            representatives: Dict {cluster_id: [rappresentanti]}
            suggested_labels: Dict {cluster_id: etichetta_suggerita}
            cluster_labels: Array delle etichette cluster per tutte le sessioni
            reviewed_labels: Dict {cluster_id: etichetta_finale_da_review_umano}
            
        Returns:
            bool: True se classificato e salvato con successo
            
        Autore: Valerio Bignardi
        Data: 2025-09-06
        """
        trace_all("_classify_and_save_representatives_post_training", "ENTER",
                 sessioni_count=len(sessioni),
                 representatives_clusters=len(representatives),
                 suggested_labels_count=len(suggested_labels),
                 cluster_labels_count=len(cluster_labels),
                 reviewed_labels_count=len(reviewed_labels))
        
        start_time = time.time()
        print(f"🎯 [CLASSIFICAZIONE RAPPRESENTANTI] Avvio classificazione completa...")
        
        try:
            # Crea istanza MongoClassificationReader per salvataggio
            from mongo_classification_reader import MongoClassificationReader
            mongo_reader = MongoClassificationReader(tenant=self.tenant)
            print("✅ [CLASSIFICAZIONE RAPPRESENTANTI] MongoDB reader creato per tenant")
            
            saved_count = 0
            failed_count = 0
            total_to_classify = sum(len(reps) for reps in representatives.values())
            
            print(f"📊 [CLASSIFICAZIONE RAPPRESENTANTI] Target: {total_to_classify} rappresentanti")
            print(f"🏷️ [CLASSIFICAZIONE RAPPRESENTANTI] Cluster: {list(representatives.keys())}")
            
            # Classifica e salva rappresentanti per ogni cluster
            for cluster_id, cluster_reps in representatives.items():
                # Determina etichetta finale (reviewed ha priorità su suggested)
                if cluster_id in reviewed_labels:
                    final_label = reviewed_labels[cluster_id]
                    label_source = "human_reviewed"
                else:
                    final_label = suggested_labels.get(cluster_id, 'altro')
                    label_source = "clustering_suggested"
                
                print(f"🎯 [CLUSTER {cluster_id}] Classificazione {len(cluster_reps)} rappresentanti")
                print(f"   🏷️ Etichetta finale: '{final_label}' (fonte: {label_source})")
                
                for rep_data in cluster_reps:
                    session_id = rep_data.get('session_id')
                    conversation_text = rep_data.get('testo_completo', '')
                    
                    try:
                        # 🎯 CLASSIFICAZIONE COMPLETA CON ENSEMBLE ML+LLM
                        print(f"   🧠 Classificando rappresentante {session_id}...")
                        
                        # Usa cache ML features se disponibili
                        cached_ml_features = self._get_cached_ml_features(session_id)
                        
                        # Classificazione ensemble con features pre-calcolate
                        prediction_result = self.ensemble_classifier.predict_with_ensemble(
                            conversation_text,
                            return_details=True,
                            embedder=self._get_embedder(),
                            ml_features_precalculated=cached_ml_features
                        )
                        
                        # Estrai predizioni ML e LLM separate
                        ml_result = prediction_result.get('ml_prediction', {})
                        llm_result = prediction_result.get('llm_prediction', {})
                        
                        # Crea final_decision con etichetta reviewed/suggested
                        final_decision = {
                            'predicted_label': final_label,
                            'confidence': 0.8,  # Confidenza alta per post-training
                            'method': f'supervised_training_post_training_{label_source}',
                            'reasoning': f'Rappresentante del cluster {cluster_id} classificato dopo training supervisionato'
                        }
                        
                        # Prepara metadati cluster
                        cluster_metadata = {
                            'cluster_id': cluster_id,
                            'is_representative': True,
                            'cluster_size': len([1 for label in cluster_labels if label == cluster_id]),
                            'suggested_label': suggested_labels.get(cluster_id, 'altro'),
                            'final_label': final_label,
                            'label_source': label_source,
                            'selection_reason': 'cluster_representative_post_training'
                        }
                        
                        # Metadati speciali per outlier
                        if cluster_id == -1:
                            cluster_metadata['selection_reason'] = 'outlier_representative_post_training'
                            cluster_metadata['is_outlier'] = True
                        
                        # 🎯 SALVATAGGIO COMPLETO CON ML+LLM PREDICTIONS
                        success = mongo_reader.save_classification_result(
                            session_id=session_id,
                            client_name=self.tenant.tenant_slug,
                            ml_result=ml_result,  # ✅ INCLUSO: Predizione ML
                            llm_result=llm_result,  # ✅ INCLUSO: Predizione LLM
                            final_decision=final_decision,
                            conversation_text=conversation_text,
                            needs_review=True,  # ✅ FONDAMENTALE: marca per review
                            review_reason='supervised_training_representative_post_training',
                            classified_by='supervised_training_pipeline_post_training',
                            notes=f'Rappresentante cluster {cluster_id} classificato DOPO training supervisionato',
                            cluster_metadata=cluster_metadata
                        )
                        
                        if success:
                            saved_count += 1
                            print(f"   ✅ {session_id}: ML={ml_result.get('predicted_label', 'N/A')} LLM={llm_result.get('predicted_label', 'N/A')} Final={final_label}")
                        else:
                            failed_count += 1
                            print(f"   ❌ ERRORE salvando {session_id}")
                            
                    except Exception as rep_error:
                        failed_count += 1
                        print(f"   ❌ ERRORE classificando {session_id}: {rep_error}")
                        continue
            
            elapsed_time = time.time() - start_time
            print(f"✅ [CLASSIFICAZIONE RAPPRESENTANTI] Completata in {elapsed_time:.2f}s")
            print(f"📊 [CLASSIFICAZIONE RAPPRESENTANTI] Risultati:")
            print(f"   ✅ Rappresentanti classificati e salvati: {saved_count}/{total_to_classify}")
            print(f"   ❌ Errori: {failed_count}")
            print(f"   🎯 Review queue popolata con predizioni ML+LLM complete")
            
            result = saved_count > 0
            trace_all("_classify_and_save_representatives_post_training", "EXIT", 
                     return_value=result, saved_count=saved_count, total_to_classify=total_to_classify)
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ [CLASSIFICAZIONE RAPPRESENTANTI] ERRORE dopo {elapsed_time:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            trace_all("_classify_and_save_representatives_post_training", "ERROR", 
                     error=str(e), error_type=type(e).__name__)
            return False
    
    # RIMOSSA: _allena_classificatore_fallback() 
    # Il training supervisionato ora richiede clustering riuscito per funzionare.
    # In caso di clustering fallito, il processo si interrompe con errore esplicativo.
    
    def _create_ml_features_cache(self, session_ids: List[str], session_texts: List[str], 
                                ml_features: np.ndarray) -> None:
        """
        Crea cache delle features ML per evitare ricalcolo BERTopic durante le predizioni
        
        Args:
            session_ids: Lista degli ID delle sessioni
            session_texts: Lista dei testi delle sessioni  
            ml_features: Features ML già calcolate (embeddings + BERTopic)
            
        Autore: Valerio Bignardi
        Data creazione: 2025-09-06
        Ultima modifica: 2025-09-06 - Ottimizzazione BERTopic + tracing
        """
        trace_all("_create_ml_features_cache", "ENTER", 
                 session_count=len(session_ids),
                 ml_features_shape=str(ml_features.shape))
        
        try:
            # Pulisci cache precedente
            self._ml_features_cache.clear()
            
            # Popola cache con nuove features
            for i, session_id in enumerate(session_ids):
                if i < len(ml_features):
                    self._ml_features_cache[session_id] = ml_features[i:i+1]  # Mantieni 2D
                    
            # Aggiorna timestamp validità
            from datetime import datetime
            self._cache_valid_timestamp = datetime.now()
            
            print(f"🏗️ Cache ML features aggiornata:")
            print(f"   📊 {len(self._ml_features_cache)} sessioni cached")
            print(f"   🔧 Feature shape: {next(iter(self._ml_features_cache.values())).shape}")
            print(f"   ⏰ Timestamp: {self._cache_valid_timestamp}")
            
            trace_all("_create_ml_features_cache", "EXIT", 
                     cached_sessions=len(self._ml_features_cache),
                     success=True)
            
        except Exception as e:
            print(f"⚠️ Errore creazione cache ML features: {e}")
            self._ml_features_cache.clear()
            
            trace_all("_create_ml_features_cache", "ERROR", 
                     error_message=str(e), success=False)
    
    def _get_cached_ml_features(self, session_id: str) -> Optional[np.ndarray]:
        """
        Recupera features ML cached per una sessione
        
        Args:
            session_id: ID della sessione
            
        Returns:
            Features ML cached o None se non disponibili
            
        Autore: Valerio Bignardi
        Data creazione: 2025-09-06
        Ultima modifica: 2025-09-06 - Ottimizzazione BERTopic
        """
        return self._ml_features_cache.get(session_id)
    
    def classifica_e_salva_sessioni(self,
                                   sessioni: Dict[str, Dict],
                                   batch_size: int = 32,
                                   use_ensemble: bool = True,
                                   optimize_clusters: bool = True,
                                   force_review: bool = False,
                                   embeddings: Optional[np.ndarray] = None,
                                   cluster_labels: Optional[np.ndarray] = None,
                                   cluster_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Classifica le sessioni usando l'ensemble classifier e salva i risultati nel database MongoDB
        
        LOGICA UNIFICATA POST-TRAINING:
        - Se force_review=True: Cancella collection MongoDB e riclassifica tutto da zero
        - Se force_review=False: Usa logica progressiva intelligente (20%/7 giorni)
        - SEMPRE: Ensemble LLM+ML con clustering ottimizzato
        - SEMPRE: altro_tag_validation abilitata
        - SEMPRE: Auto-classificazione completa (mai human review)
        
        Args:
            sessioni: Sessioni da classificare
            batch_size: Dimensione del batch per la classificazione
            use_ensemble: Se True, usa l'ensemble classifier (SEMPRE True in produzione)
            optimize_clusters: Se True, usa clustering ottimizzato (SEMPRE True in produzione)
            force_review: Se True, forza riclassificazione completa
            embeddings: Embeddings precomputati (opzionale, evita duplicazione clustering)
            cluster_labels: Labels cluster precomputati (opzionale, evita duplicazione clustering)
            cluster_info: Info cluster precomputate (opzionale, evita duplicazione clustering)  
            force_review: Se True, cancella MongoDB e riprocessa tutto da capo
            
        Returns:
            Statistiche della classificazione
            
        Autore: Valerio Bignardi
        Data: 2025-08-29
        """
        trace_all("classifica_e_salva_sessioni", "ENTER",
                 sessioni_count=len(sessioni),
                 batch_size=batch_size,
                 use_ensemble=use_ensemble, 
                 optimize_clusters=optimize_clusters,
                 force_review=force_review)
        
        # 🔍 DEBUG: Import e trace della funzione
        from Pipeline.debug_pipeline import debug_pipeline, debug_flow
        
        debug_pipeline("classifica_e_salva_sessioni", "ENTRY - Avvio classificazione e salvataggio", {
            "num_sessioni": len(sessioni),
            "batch_size": batch_size,
            "use_ensemble": use_ensemble,
            "optimize_clusters": optimize_clusters,
            "force_review": force_review,
            "tenant": self.tenant_slug
        }, "ENTRY")
        
        print(f"🏷️  CLASSIFICAZIONE POST-TRAINING di {len(sessioni)} sessioni...")
        print(f"📊 Batch size: {batch_size}")
        print(f"🔄 Force review: {force_review}")
        print(f"🎯 Optimize clusters: {optimize_clusters}")
        print(f"🔗 Use ensemble: {use_ensemble}")
        
        # 🆕 GESTIONE FORCE_REVIEW: Pulizia MongoDB se richiesta  
        if force_review:
            print(f"🧹 FORCE REVIEW: Cancellazione collection MongoDB per tenant '{self.tenant_slug}'")
            try:
                # Aggiungi percorso root per import
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                from mongo_classification_reader import MongoClassificationReader
                
                # CAMBIO RADICALE: Usa oggetto Tenant
                mongo_reader = MongoClassificationReader(tenant=self.tenant)
                
                # Cancella tutte le classificazioni del tenant
                clear_result = mongo_reader.clear_tenant_collection(self.tenant_slug)
                
                if clear_result['success']:
                    deleted_count = clear_result['deleted_count']
                    print(f"✅ Cancellate {deleted_count} classificazioni esistenti")
                else:
                    print(f"⚠️ Errore nella cancellazione: {clear_result['error']}")
                
                # Force clustering completo
                print(f"🔄 Force review attivato → Clustering completo forzato")
                
            except Exception as e:
                print(f"⚠️ Errore nella cancellazione MongoDB: {e}")
                print(f"🔄 Continuando con la classificazione...")
        
        # Forza sempre ensemble e clustering ottimizzato per classificazione post-training
        use_ensemble = True
        optimize_clusters = True
        
        # 🔍 CONTROLLO STATO ML ENSEMBLE (definizione anticipata per evitare UnboundLocalError)
        ml_ensemble_trained = (
            hasattr(self.ensemble_classifier, 'ml_ensemble') and 
            self.ensemble_classifier.ml_ensemble is not None and
            hasattr(self.ensemble_classifier.ml_ensemble, 'classes_')
        )
        
        print(f"🎯 MODALITÀ UNIFICATA: {'Ensemble LLM+ML' if ml_ensemble_trained else 'LLM-only'} + Clustering Ottimizzato")
        
        # 🆕 GESTIONE PRIMO AVVIO: Se ML non è allenato, non dovrebbe arrivare qui
        # La logica LLM-only dovrebbe essere gestita nel training supervisionato
        if not ml_ensemble_trained:
            print(f"⚠️ AVVERTIMENTO: Classificazione chiamata senza ML allenato")
            print(f"🎯 Suggerimento: Eseguire prima training supervisionato con review umana")
            # Continua comunque con solo LLM per compatibilità
            print(f"🚀 Fallback: Classificazione LLM-only")
        
        # 🚨 DEBUG PARAMETRI: Verifica configurazione pre-classificazione
        print(f"")
        print(f"🔍 [PRE-CLASSIFICATION DEBUG]")
        print(f"   📊 Sessioni da classificare: {len(sessioni)}")
        print(f"   🎯 optimize_clusters: {optimize_clusters} (FORZA SEMPRE)")
        print(f"   🔧 use_ensemble: {use_ensemble} (FORZA SEMPRE)")
        print(f"   📦 batch_size: {batch_size}")
        print(f"   🔄 force_review: {force_review}")
        if len(sessioni) < 10:
            print(f"   ⚠️  ATTENZIONE: Dataset piccolo ({len(sessioni)} < 10 sessioni)")
            print(f"   ⚠️  Potrebbero esserci problemi di clustering")
        print(f"🔍 [/PRE-CLASSIFICATION DEBUG]")
        print(f"")
        
        # Connetti al database TAG (legacy, potrebbe non essere più necessario)
        print(f"💾 Connessione al database TAG...")
        try:
            self.tag_db.connetti()
        except Exception as e:
            print(f"⚠️ Errore connessione TAG DB (ignorabile): {e}")
        
        # 🆕 CONTROLLO E TRAINING ML ENSEMBLE SE NECESSARIO
        print(f"\n🔍 VERIFICA STATO ML ENSEMBLE...")
        
        print(f"📊 ML Ensemble già allenato: {ml_ensemble_trained}")
        
        if not ml_ensemble_trained:
            print(f"🚨 ML ENSEMBLE NON ALLENATO - Modalità LLM-only con preparazione training")
            print(f"📋 Sessioni disponibili: {len(sessioni)}")
            
            # 🎯 NUOVA SEQUENZA CORRETTA: 
            # 1. Classifica PRIMA con LLM-only (senza training ML)
            # 2. Salva i dati per training futuro
            # 3. Training ML DOPO review umana
            
            print(f"🔄 PRIMO AVVIO - Sequenza corretta:")
            print(f"   1️⃣ Classificazione LLM-only dei rappresentanti")
            print(f"   2️⃣ Salvataggio dati per training futuro")
            print(f"   3️⃣ Training ML dopo review umana")
            
            # IMPORTANTE: NON fare training automatico qui
            # Il training ML avverrà quando ci saranno dati rivisti dall'umano
        else:
            print(f"✅ ML Ensemble già allenato - procedo con classificazione completa")
            
        print(f"🔍 [/VERIFICA ML ENSEMBLE]\n")
        
        # Prepara dati per classificazione
        session_ids = list(sessioni.keys())
        session_texts = [sessioni[sid]['testo_completo'] for sid in session_ids]
        print(f"📦 Preparati {len(session_texts)} testi per classificazione")
        
        # 🎯 LOGICA UNIFICATA: SEMPRE Classificazione ottimizzata + ensemble
        # La logica intelligente (20%/7 giorni) è gestita automaticamente nel clustering upstream
        print(f"🚀 Classificazione ottimizzata con ensemble LLM+ML in corso...")
        
        try:
            debug_pipeline("classifica_e_salva_sessioni", "TENTATIVO - Chiamata _classifica_ottimizzata_cluster", {
                "num_sessioni": len(sessioni), 
                "num_session_ids": len(session_ids), # Controllo incrociato
                "num_session_texts": len(session_texts), # Controllo incrociato
                "batch_size": batch_size, # Parametro critico che serve per debug
                "optimize_clusters": optimize_clusters, # FORZATO True rappresenta la logica unificata post-training 
                "use_ensemble": use_ensemble
            }, "INFO")
            
            print(f"🎯 [CLUSTERING DEBUG] Iniziando classificazione ottimizzata...")
            print(f"🎯 [CLUSTERING DEBUG] Sessioni: {len(sessioni)}, optimize_clusters: {optimize_clusters}")
            
            # 🚨 DEBUG SUPER CRITICO: Traccia chiamata cluster ottimizzato
            print(f"🚨 [SUPER DEBUG] CHIAMATA _classifica_ottimizzata_cluster INIZIATA")
            print(f"   📊 Input sessioni: {len(sessioni)}")
            print(f"   📊 Input session_ids: {len(session_ids)}")
            print(f"   📊 Input session_texts: {len(session_texts)}")
            
            predictions = self._classifica_ottimizzata_cluster(
                sessioni, session_ids, session_texts,
                embeddings=embeddings, cluster_labels=cluster_labels, cluster_info=cluster_info, batch_size=batch_size
            )
            
            # 🚨 DEBUG SUPER CRITICO: Verifica risultato
            print(f"🚨 [SUPER DEBUG] _classifica_ottimizzata_cluster COMPLETATA!")
            print(f"   📊 Output predictions: {len(predictions)}")
            print(f"   ✅ Con cluster_metadata: {sum(1 for p in predictions if p.get('cluster_metadata'))}")
            print(f"   ❌ Senza cluster_metadata: {sum(1 for p in predictions if not p.get('cluster_metadata'))}")
            
            debug_pipeline("classifica_e_salva_sessioni", "SUCCESS - _classifica_ottimizzata_cluster completata", {
                "num_predictions": len(predictions),
                "predictions_with_cluster_metadata": sum(1 for p in predictions if p.get('cluster_metadata')),
                "predictions_without_cluster_metadata": sum(1 for p in predictions if not p.get('cluster_metadata'))
            }, "SUCCESS")
            
            print(f"✅ Classificazione ottimizzata completata: {len(predictions)} risultati")
            
        except Exception as e:
            from Pipeline.debug_pipeline import debug_exception
            
            # 🚨 DEBUG FALLBACK: Cattura l'errore esatto
            print(f"")
            print(f"🚨🚨🚨 [FALLBACK TRIGGER] 🚨🚨🚨")
            print(f"🚨 ERRORE NELLA CLASSIFICAZIONE OTTIMIZZATA!")
            print(f"🚨 Tipo errore: {type(e).__name__}")
            print(f"🚨 Messaggio: {str(e)}")
            print(f"🚨 Parametri attuali:")
            print(f"   📊 Sessioni: {len(sessioni)}")
            print(f"   🎯 optimize_clusters: {optimize_clusters}")
            print(f"   🔧 use_ensemble: {use_ensemble}")
            print(f"   📦 batch_size: {batch_size}")
            print(f"🚨 ATTIVANDO FALLBACK LLM_STRUCTURED")
            print(f"🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨")
            print(f"")
            
            debug_exception("classifica_e_salva_sessioni", e, {
                "num_sessioni": len(sessioni),
                "batch_size": batch_size,
                "optimize_clusters": optimize_clusters,
                "use_ensemble": use_ensemble,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "FALLBACK_REASON": "CLUSTERING_OPTIMIZATION_FAILED"
            })
            
            print(f"❌ ERRORE nella classificazione ottimizzata: {e}")
            print(f"🔄 Fallback alla classificazione ensemble tradizionale...")
            
            debug_pipeline("classifica_e_salva_sessioni", "FALLBACK - Tentativo batch_predict", {
                "num_texts": len(session_texts),
                "batch_size": batch_size,
                "fallback_trigger": "clustering_optimization_failed",
                "original_error": str(e)
            }, "WARNING")
            
            # Fallback: classificazione ensemble tradizionale
            try:
                batch_predictions = self.ensemble_classifier.batch_predict(
                    session_texts, 
                    batch_size=batch_size,
                    embedder=self.embedder
                )
                predictions = batch_predictions
                
                debug_pipeline("classifica_e_salva_sessioni", "FALLBACK SUCCESS - batch_predict completato", {
                    "num_predictions": len(predictions),
                    "predictions_type": "batch_predict_fallback"
                }, "WARNING")
                
                print(f"✅ Fallback completato: {len(predictions)} risultati")
                
            except Exception as e2:
                debug_exception("classifica_e_salva_sessioni_fallback", e2)
                
                print(f"❌ ERRORE anche nel fallback: {e2}")
                # Fallback finale: predizioni singole
                predictions = []
                for i, text in enumerate(session_texts):
                    try:
                        prediction = self.ensemble_classifier.predict_with_ensemble(
                            text, 
                            return_details=True, 
                            embedder=self.embedder
                        )
                        predictions.append(prediction)
                    except Exception as e3:
                        # Fallback assoluto
                        predictions.append({
                            'predicted_label': 'altro',
                            'confidence': 0.1,
                            'is_high_confidence': False,
                            'method': 'FALLBACK_FINAL',
                            'llm_prediction': None,
                            'ml_prediction': {'predicted_label': 'altro', 'confidence': 0.1},
                            'ensemble_confidence': 0.1
                        })
                print(f"⚠️ Fallback finale completato: {len(predictions)} risultati")
        
        # 🆕 CONTATORE DEBUG per RAPPRESENTANTI e OUTLIERS
        # Conta solo i casi che vengono classificati individualmente (esclude PROPAGATI)
        classification_counter = 0
        total_individual_cases = 0
        
        # Pre-conta i casi individuali per il totale
        for prediction in predictions:
            method = prediction.get('method', '')
            if method.startswith('REPRESENTATIVE') or method.startswith('OUTLIER'):
                total_individual_cases += 1
        
        # Salva classificazioni nel database
        stats = {
            'total_sessions': len(sessioni),
            'high_confidence': 0,
            'low_confidence': 0,
            'saved_successfully': 0,
            'save_errors': 0,
            'classifications_by_tag': {},
            'individual_cases_classified': 0,  # Solo rappresentanti e outliers
            'propagated_cases': 0,  # Casi ereditati
            'ensemble_stats': {
                'llm_predictions': 0,
                'ml_predictions': 0,
                'ensemble_agreements': 0,
                'ensemble_disagreements': 0
            } if use_ensemble else None
        }
        
        print(f"💾 Inizio salvataggio di {len(predictions)} classificazioni...")
        print(f"📊 Casi individuali da classificare: {total_individual_cases} (rappresentanti + outliers)")
        
        for i, (session_id, prediction) in enumerate(zip(session_ids, predictions)):
            # 🆕 DEBUG CONTATORE per casi classificati individualmente
            method = prediction.get('method', '')
            session_type = None
            
            if method.startswith('REPRESENTATIVE'):
                classification_counter += 1
                session_type = "RAPPRESENTANTE"
                stats['individual_cases_classified'] += 1
                print(f"📋 caso n° {classification_counter:02d} / {total_individual_cases:03d} {session_type}")
                
            elif method.startswith('OUTLIER'):
                classification_counter += 1 
                session_type = "OUTLIER"
                stats['individual_cases_classified'] += 1
                print(f"📋 caso n° {classification_counter:02d} / {total_individual_cases:03d} {session_type}")
                
            elif 'PROPAGATED' in method or 'CLUSTER_PROPAGATED' in method:
                stats['propagated_cases'] += 1
                # I propagati non entrano nel contatore come richiesto
            
            # Debug standard ogni 10 classificazioni per tutti i tipi
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
                    
                    # 🧹 PULIZIA CRITICA: Applica pulizia caratteri speciali SUBITO dopo estrazione
                    clean_predicted_label = clean_label_text(predicted_label)
                    if clean_predicted_label != predicted_label:
                        print(f"🧹 Label pulita (ensemble): '{predicted_label}' → '{clean_predicted_label}'")
                        predicted_label = clean_predicted_label
                        # 🔧 FIX CRITICO: Aggiorna anche il dizionario prediction
                        prediction['predicted_label'] = clean_predicted_label
                    
                    # 🆕 VALIDAZIONE "ALTRO" CON LLM + BERTopic + SIMILARITÀ
                    print (f"   🔍 Verifica necessità validazione 'altro' per sessione {i+1}...")
                    if predicted_label == 'altro' and hasattr(self, 'interactive_trainer') and self.interactive_trainer.altro_validator: # Abilita validazione altro se predict altro e validator attivo 
                        print (f"   🔍 Predicted label è 'altro', avvio validazione...")
                        try:
                            conversation_text = sessioni[session_id].get('testo_completo', '')
                            if conversation_text:
                                # Esegui validazione del tag "altro"
                                validated_label, validated_confidence, validation_info = self.interactive_trainer.handle_altro_classification(
                                    conversation_text=conversation_text,
                                    force_human_decision=False  # Automatico durante training
                                )
                                
                                # Usa il risultato della validazione se diverso da "altro"
                                if validated_label != 'altro':
                                    # 🧹 PULIZIA CRITICA: Applica pulizia al validated_label (primo blocco)
                                    clean_validated_label = clean_label_text(validated_label)
                                    if clean_validated_label != validated_label:
                                        print(f"🧹 Validated label pulita (ensemble): '{validated_label}' → '{clean_validated_label}'")
                                    
                                    predicted_label = clean_validated_label
                                    confidence = validated_confidence
                                    method = f"{method}_ALTRO_VAL"  # Marca che è stato validato
                                    
                                    # 🔧 FIX CRITICO: Aggiorna anche il dizionario prediction
                                    prediction['predicted_label'] = clean_validated_label
                                    prediction['confidence'] = validated_confidence
                                    # Non aggiornare method nel prediction per mantenere coerenza
                                    
                                    if i < 10:  # Debug per le prime 10
                                        print(f"🔍 Sessione {i+1}: ALTRO→'{clean_validated_label}' (path: {validation_info.get('validation_path', 'unknown')})")
                        
                        except Exception as e:
                            print(f"⚠️ Errore validazione ALTRO per sessione {i+1}: {e}")
                            # Continua con predicted_label = 'altro' originale
                    
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
                    
                    # 🧹 PULIZIA CRITICA: Applica pulizia caratteri speciali per ML_AUTO
                    clean_predicted_label = clean_label_text(predicted_label)
                    if clean_predicted_label != predicted_label:
                        print(f"🧹 Label pulita (ML_AUTO): '{predicted_label}' → '{clean_predicted_label}'")
                        predicted_label = clean_predicted_label
                        # 🔧 FIX CRITICO: Aggiorna anche il dizionario prediction
                        prediction['predicted_label'] = clean_predicted_label
                    
                    # 🆕 VALIDAZIONE "ALTRO" ANCHE PER ML_AUTO 
                    if predicted_label == 'altro' and hasattr(self, 'interactive_trainer') and self.interactive_trainer.altro_validator:
                        try:
                            conversation_text = sessioni[session_id].get('testo_completo', '')
                            if conversation_text:
                                # Esegui validazione del tag "altro"
                                validated_label, validated_confidence, validation_info = self.interactive_trainer.handle_altro_classification(
                                    conversation_text=conversation_text,
                                    force_human_decision=False  # Automatico durante training
                                )
                                
                                # Usa il risultato della validazione se diverso da "altro"
                                if validated_label != 'altro':
                                    # 🧹 PULIZIA CRITICA: Applica pulizia al validated_label (secondo blocco)
                                    clean_validated_label = clean_label_text(validated_label)
                                    if clean_validated_label != validated_label:
                                        print(f"🧹 Validated label pulita (ML_AUTO): '{validated_label}' → '{clean_validated_label}'")
                                    
                                    predicted_label = clean_validated_label
                                    confidence = validated_confidence
                                    method = f"{method}_ALTRO_VAL"  # Marca che è stato validato
                                    
                                    # 🔧 FIX CRITICO: Aggiorna anche il dizionario prediction
                                    prediction['predicted_label'] = clean_validated_label
                                    prediction['confidence'] = validated_confidence
                                    # Non aggiornare method nel prediction per mantenere coerenza
                                    
                                    if i < 10:  # Debug per le prime 10
                                        print(f"🔍 Sessione {i+1}: ALTRO→'{clean_validated_label}' (path: {validation_info.get('validation_path', 'unknown')})")
                        
                        except Exception as e:
                            print(f"⚠️ Errore validazione ALTRO per sessione {i+1}: {e}")
                            # Continua con predicted_label = 'altro' originale
                
                # Determina se è alta confidenza
                is_high_confidence = confidence >= self.confidence_threshold
                
                # 🆕 SALVA USANDO SOLO MONGODB (SISTEMA UNIFICATO)
                from mongo_classification_reader import MongoClassificationReader
                
                # CAMBIO RADICALE: Usa oggetto Tenant
                mongo_reader = MongoClassificationReader(tenant=self.tenant)
                
                # Estrai dati ensemble se disponibili
                ml_result = None
                llm_result = None
                has_disagreement = False
                disagreement_score = 0.0
                
                if prediction:
                    # 🔧 FIX CRITICO: Estrazione corretta ML/LLM predictions
                    # Distingui tra metodo ensemble (per final_decision) e metodo cluster (per metadata)
                    ensemble_method = prediction.get('method', '')  # ENSEMBLE, LLM, ML
                    cluster_method = prediction.get('cluster_method', ensemble_method)  # REPRESENTATIVE, OUTLIER, CLUSTER_PROPAGATED
                    
                    print(f"   🔧 [EXTRACTION] Session {session_id}: ensemble_method={ensemble_method}, cluster_method={cluster_method}")
                    
                    # ✅ SEMPRE estrai da ensemble predictions se disponibili
                    if ensemble_method in ['ENSEMBLE', 'LLM', 'ML'] and prediction.get('ml_prediction') is not None:
                        # Predizione ensemble completa - estrai predizioni separate
                        ml_prediction_data = prediction.get('ml_prediction')
                        llm_prediction_data = prediction.get('llm_prediction')
                        
                        # ✅ Estrai ML prediction se disponibile
                        if ml_prediction_data is not None:
                            ml_result = ml_prediction_data
                            print(f"   ✅ [EXTRACTION] ML result estratto: {ml_result.get('predicted_label', 'N/A')}")
                        
                        # ✅ Estrai LLM prediction se disponibile  
                        if llm_prediction_data is not None:
                            llm_result = llm_prediction_data
                            print(f"   ✅ [EXTRACTION] LLM result estratto: {llm_result.get('predicted_label', 'N/A')}")
                            
                    elif prediction.get('method') == 'LLM' and 'ml_prediction' not in prediction:
                        # 🔄 FALLBACK: Training supervisionato - solo LLM disponibile  
                        llm_result = {
                            'predicted_label': prediction.get('predicted_label'),
                            'confidence': prediction.get('confidence', 0.0),
                            'motivation': prediction.get('motivation', 'LLM training supervisionato'),
                            'method': 'LLM'
                        }
                        ml_result = None  # ML non disponibile durante training
                        print(f"   🔄 [EXTRACTION] Fallback LLM-only per training supervisionato")
                    else:
                        print(f"   ⚠️ [EXTRACTION] Nessuna prediction ensemble trovata per {session_id}")
                        print(f"      ensemble_method={ensemble_method}, has_ml_prediction={prediction.get('ml_prediction') is not None}")
                    
                    # Calcola disagreement se entrambi disponibili
                    if ml_result and llm_result:
                        ml_label = ml_result.get('predicted_label', '')
                        llm_label = llm_result.get('predicted_label', '')
                        ml_conf = ml_result.get('confidence', 0.0)
                        llm_conf = llm_result.get('confidence', 0.0)
                        
                        has_disagreement = (ml_label != llm_label)
                        if has_disagreement:
                            disagreement_score = 1.0
                        else:
                            disagreement_score = abs(ml_conf - llm_conf)
                        
                        print(f"   📊 [DISAGREEMENT] ML={ml_label}({ml_conf:.2f}) vs LLM={llm_label}({llm_conf:.2f}) → disagreement={has_disagreement}")
                else:
                    print(f"   ❌ [EXTRACTION] Nessuna prediction disponibile per {session_id}")
                
                # 🆕 VALUTAZIONE INTELLIGENTE PER REVIEW QUEUE
                # Determina se serve review in base al tipo e confidenza
                needs_review = False
                review_reason = "auto_classified_post_training"
                
                # 🎯 USA SOGLIE UNIFICATI GIÀ CARICATI dal database MySQL
                # Non serve ricaricare - usiamo unified_params già disponibile
                outlier_threshold = self.unified_params.get('outlier_confidence_threshold', 0.7)
                propagated_threshold = self.unified_params.get('propagated_confidence_threshold', 0.8)  
                representative_threshold = self.unified_params.get('representative_confidence_threshold', 0.9)
                enable_smart_review = self.unified_params.get('enable_smart_review', True)
                
                print(f"   🎯 [REVIEW-THRESHOLDS UNIFIED] Outlier: {outlier_threshold}, Propagated: {propagated_threshold}, Representative: {representative_threshold}")
                
                # 🎯 FIX: OUTLIERS con bassa confidenza vanno in review
                if prediction and 'OUTLIER' in prediction.get('method', ''):
                    if enable_smart_review and confidence < outlier_threshold:
                        needs_review = True
                        review_reason = f"outlier_low_confidence_{confidence:.3f}_threshold_{outlier_threshold}"
                        print(f"   🎯 OUTLIER {session_id}: confidenza {confidence:.3f} < {outlier_threshold} → PENDING REVIEW")
                    else:
                        review_reason = f"outlier_high_confidence_{confidence:.3f}_threshold_{outlier_threshold}"
                        print(f"   ✅ OUTLIER {session_id}: confidenza {confidence:.3f} ≥ {outlier_threshold} → AUTO_CLASSIFIED")
                
                # 🎯 FIX: RAPPRESENTANTI con bassa confidenza vanno in review
                elif prediction and 'REPRESENTATIVE' in prediction.get('method', ''):
                    if enable_smart_review and confidence < representative_threshold:
                        needs_review = True
                        review_reason = f"representative_low_confidence_{confidence:.3f}_threshold_{representative_threshold}"
                        print(f"   🎯 RAPPRESENTANTE {session_id}: confidenza {confidence:.3f} < {representative_threshold} → PENDING REVIEW")
                    else:
                        review_reason = f"representative_high_confidence_{confidence:.3f}_threshold_{representative_threshold}"
                
                # 🚫 CORREZIONE: PROPAGATED NON vanno MAI automaticamente in review
                elif prediction and 'PROPAGATED' in prediction.get('method', ''):
                    # I propagati sono SEMPRE auto-classificati, mai in review automaticamente
                    # Vanno in review SOLO quando aggiunti manualmente dall'utente via UI
                    needs_review = False  # 🚫 FORZA: Nessun review automatico per propagati
                    review_reason = f"propagated_auto_classified_conf_{confidence:.3f}"
                    print(f"   ✅ PROPAGATED {session_id}: auto-classificato (conf: {confidence:.3f}) - NO REVIEW automatico")
                
                # Debug: mostra solo casi che vanno in review
                if needs_review:
                    print(f"   👤 REVIEW QUEUE: {session_id} - {review_reason}")
                elif has_disagreement and disagreement_score > 0.3:
                    if i < 5:  # Debug prime 5
                        print(f"   📊 Sessione {i+1}: Disaccordo {disagreement_score:.2f} ma auto-approvata")
                
                # Ottieni dati sessione
                session_data = sessioni[session_id]
                
                # 🆕 COSTRUISCI CLUSTER METADATA per classificazione ottimizzata
                cluster_metadata = None
                if optimize_clusters and prediction:
                    # 🔧 FIX CRITICO: Distingui metodo ensemble da metodo cluster
                    ensemble_method = prediction.get('method', '')  # ENSEMBLE, LLM, ML
                    cluster_method = prediction.get('cluster_method', '')  # REPRESENTATIVE, OUTLIER, CLUSTER_PROPAGATED
                    cluster_id = prediction.get('cluster_id', -1)
                    
                    print(f"   🔧 [METADATA] Session {session_id}: ensemble={ensemble_method}, cluster={cluster_method}, cluster_id={cluster_id}")
                    
                    # ✅ USA CLUSTER_METHOD per metadata (non ensemble_method)
                    if 'REPRESENTATIVE' in cluster_method:
                        cluster_metadata = {
                            'cluster_id': cluster_id,
                            'is_representative': True,
                            'cluster_size': None,  # Potremmo calcolarlo se necessario
                            'confidence': confidence,
                            'method': cluster_method,  # ✅ Usa cluster method per metadata
                            'ensemble_method': ensemble_method  # ✅ Traccia anche ensemble method
                        }
                    elif 'CLUSTER_PROPAGATED' in cluster_method or cluster_method == 'CLUSTER_PROPAGATED':
                        # 🔧 FIX: usa 'in' invece di '==' e prendi il vero source_representative
                        source_rep = prediction.get('source_representative', 'cluster_propagation')
                        cluster_metadata = {
                            'cluster_id': cluster_id,
                            'is_representative': False,
                            'propagated_from': source_rep,  # Usa il vero session_id del rappresentante
                            'propagation_confidence': confidence,
                            'method': cluster_method,  # ✅ Usa cluster method per metadata
                            'ensemble_method': ensemble_method  # ✅ Traccia anche ensemble method
                        }
                    elif 'OUTLIER' in cluster_method:
                        cluster_metadata = {
                            'cluster_id': -1,
                            'is_representative': False,
                            'outlier_score': 1.0 - confidence,  # Outlier score inversamente correlato alla confidenza
                            'method': cluster_method,  # ✅ Usa cluster method per metadata
                            'ensemble_method': ensemble_method  # ✅ Traccia anche ensemble method
                        }
                else:
                    # 🆕 FIX: Se optimize_clusters=False o prediction senza cluster info,
                    # NON costruire cluster_metadata per evitare auto-assegnazione a outlier_X
                    # Il sistema di salvataggio gestirà correttamente i casi senza cluster info
                    debug_pipeline("classifica_e_salva_sessioni", f"CLUSTER_METADATA=None per {session_id}", {
                        "optimize_clusters": optimize_clusters,
                        "prediction_method": prediction.get('method') if prediction else 'No prediction',
                        "prediction_has_cluster_info": bool(prediction and prediction.get('cluster_id'))
                    }, "WARNING")
                    
                    print(f"   ⚠️ Sessione {session_id}: nessun cluster metadata (optimize_clusters={optimize_clusters})")
                    cluster_metadata = None
                
                # 🔍 DEBUG: Trace prima del salvataggio
                debug_pipeline("classifica_e_salva_sessioni", f"PRE-SAVE session {session_id}", {
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "method": method,
                    "has_cluster_metadata": bool(cluster_metadata),
                    "cluster_metadata": cluster_metadata,
                    "classified_by": 'post_training_pipeline'
                }, "INFO")
                
                # � FIX CRITICO: USA ENSEMBLE METHOD per final_decision
                final_decision_method = prediction.get('method', 'UNKNOWN') if prediction else 'UNKNOWN'
                
                print(f"� [FINAL-DECISION] Session {session_id}:")
                print(f"   📋 final_decision method: '{final_decision_method}' (DOVREBBE essere ENSEMBLE/LLM/ML)")
                print(f"   📋 cluster_method: '{prediction.get('cluster_method', 'N/A') if prediction else 'N/A'}' (per metadata)")
                
                success = mongo_reader.save_classification_result(
                    session_id=session_id,
                    client_name=self.tenant_slug,
                    ml_result=ml_result,
                    llm_result=llm_result,
                    final_decision={
                        'predicted_label': predicted_label,
                        'confidence': confidence,
                        'method': final_decision_method,  # ✅ USA ENSEMBLE METHOD 
                        'reasoning': f"Auto-classificato post-training con confidenza {confidence:.3f}"
                    },
                    conversation_text=session_data['testo_completo'],
                    needs_review=needs_review,  # Sempre False in fase post-training
                    review_reason=review_reason,  # "auto_classified_post_training"
                    classified_by='post_training_pipeline',  # Specifica fase
                    notes=f"Classificazione post-training automatica (confidenza {confidence:.3f})",
                    cluster_metadata=cluster_metadata,  # Metadata cluster per filtri UI
                    embedding=None,  # ✅ TEMPORAL FIX: Disabled embedding per ora
                    embedding_model=None  # ✅ TEMPORAL FIX: Disabled embedding per ora
                )
                
                if success:
                    stats['saved_successfully'] += 1
                    stats['mongo_saves'] = stats.get('mongo_saves', 0) + 1
                    
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
        print(f"  📋 Classificati individualmente: {stats['individual_cases_classified']} (rappresentanti + outliers)")
        print(f"  🔄 Casi propagati: {stats['propagated_cases']} (ereditano etichetta)")
        print(f"  🎯 Alta confidenza: {stats['high_confidence']}")
        print(f"  ⚠️  Bassa confidenza: {stats['low_confidence']}")
        print(f"  ❌ Errori: {stats['save_errors']}")
        
        if use_ensemble and stats['ensemble_stats']:
            ens = stats['ensemble_stats']
            print(f"  🔗 Ensemble: {ens['llm_predictions']} LLM + {ens['ml_predictions']} ML")
            print(f"  🤝 Accordi: {ens['ensemble_agreements']}, Disaccordi: {ens['ensemble_disagreements']}")
        
        # Verifica integrità conteggi
        expected_total = stats['individual_cases_classified'] + stats['propagated_cases']
        if expected_total != stats['total_sessions']:
            print(f"⚠️ ATTENZIONE: Conteggio inconsistente! Individ.({stats['individual_cases_classified']}) + Propag.({stats['propagated_cases']}) != Tot.({stats['total_sessions']})")
        else:
            print(f"✅ Integrità conteggi verificata: {expected_total} casi processati")
        
        # NUOVA FUNZIONALITÀ: Visualizzazione grafica STATISTICHE COMPLETE
        # (con etichette finali dopo classificazione)
        try:
            from Utils.cluster_visualization import ClusterVisualizationManager
            
            visualizer = ClusterVisualizationManager()
            session_texts = [sessioni[sid].get('testo_completo', '') for sid in session_ids]
            
            # Verifica se abbiamo i dati del clustering precedente
            if (hasattr(self, '_last_embeddings') and 
                hasattr(self, '_last_cluster_labels') and 
                hasattr(self, '_last_cluster_info')):
                
                print("\n🎨 GENERAZIONE VISUALIZZAZIONI STATISTICHE COMPLETE...")
                # Visualizzazione per STATISTICHE (con etichette finali)
                visualization_results = visualizer.visualize_classification_statistics(
                    embeddings=self._last_embeddings,  # Embeddings dell'ultimo clustering
                    cluster_labels=self._last_cluster_labels,  # Cluster labels dell'ultimo clustering
                    final_predictions=predictions,  # Predizioni finali con etichette
                    cluster_info=self._last_cluster_info,  # Info cluster
                    session_texts=session_texts,
                    save_html=True,
                    show_console=True
                )
            else:
                print("ℹ️  Visualizzazione avanzata richiede esecuzione clustering prima della classificazione")
                
        except ImportError:
            print("⚠️ Sistema visualizzazione avanzato non disponibile - installare plotly")
        except Exception as e:
            print(f"⚠️ Errore nella visualizzazione statistiche avanzate: {e}")
            import traceback
            traceback.print_exc()
        
        trace_all("classifica_e_salva_sessioni", "EXIT", return_value=stats)
        return stats
    


















    def esegui_pipeline_completa(self,
                               giorni_indietro: int = 7, #mai utilizzato
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
        trace_all("esegui_pipeline_completa", "ENTER",
                 giorni_indietro=giorni_indietro,
                 limit=limit,
                 batch_size=batch_size,
                 interactive_mode=interactive_mode,
                 use_ensemble=use_ensemble,
                 force_full_extraction=force_full_extraction)
        
        # 🔍 DEBUG: Import utility debug
        from Pipeline.debug_pipeline import debug_pipeline, debug_flow
        
        debug_pipeline("esegui_pipeline_completa", "ENTRY - Pipeline completa avviata", {
            "giorni_indietro": giorni_indietro,
            "limit": limit,
            "batch_size": batch_size, 
            "interactive_mode": interactive_mode,
            "use_ensemble": use_ensemble,
            "force_full_extraction": force_full_extraction,
            "tenant": self.tenant_slug
        }, "ENTRY")
        
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
            
            # 3. Verifica se ML è già allenato per decidere il flusso
            ml_ensemble_trained = (
                hasattr(self.ensemble_classifier, 'ml_ensemble') and 
                self.ensemble_classifier.ml_ensemble is not None and
                hasattr(self.ensemble_classifier.ml_ensemble, 'classes_')
            )
            
            print(f"📊 ML Ensemble già allenato: {ml_ensemble_trained}")
            
            if not ml_ensemble_trained:
                # PRIMO AVVIO: Classificazione LLM-only + preparazione training
                print(f"🚀 PRIMO AVVIO: Classificazione LLM-only con preparazione training")
                llm_classification_result = self._classifica_llm_only_e_prepara_training(
                    sessioni, cluster_labels, representatives, suggested_labels
                )
                
                print(f"✅ Classificazione LLM-only completata")
                print(f"👤 Prossimo step: Review umana → Training ML")
                print(f"📋 Sessioni in review queue: {llm_classification_result.get('saved_for_review', 0)}")
                
                # Prepara risultato per PRIMO AVVIO (senza training ML ancora)
                return {
                    'phase': 'llm_only_primo_avvio',
                    'classification_result': llm_classification_result,
                    'ml_training_needed': True,
                    'human_review_needed': True,
                    'total_sessions': len(sessioni),
                    'clusters_found': len(representatives),
                    'representatives_classified': llm_classification_result.get('representatives_classified', 0),
                    'outliers_classified': llm_classification_result.get('outliers_classified', 0),
                    'next_action': 'human_review_then_ml_training'
                }
            else:
                # AVVII SUCCESSIVI: Training normale con ensemble ML+LLM
                print(f"🔄 AVVIO NORMALE: ML già allenato, procedendo con training ensemble")
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
            
            trace_all("esegui_pipeline_completa", "EXIT", return_value=results)
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
            
        Autore: Valerio Bignardi
        Data creazione: 2025-09-06
        Ultima modifica: 2025-09-06 - Aggiunta del tracing completo
        """
        trace_all("analizza_e_consolida_etichette", "ENTER", dry_run=dry_run)
        
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
        
        trace_all("analizza_e_consolida_etichette", "EXIT", 
                 total_labels_analyzed=len(etichette_list),
                 total_consolidations_possible=total_consolidations,
                 applied=not dry_run,
                 applied_consolidations=applied_consolidations if not dry_run else 0)
        
        return results

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
        trace_all("esegui_training_interattivo", "ENTER", 
                 giorni_indietro=giorni_indietro, limit=limit,
                 max_human_review_sessions=max_human_review_sessions,
                 confidence_threshold=confidence_threshold,
                 force_review=force_review, disagreement_threshold=disagreement_threshold)
        
        start_time = datetime.now()
        
        # 🔍 DEBUG: Setup logging dettagliato
        import logging
        debug_logger = logging.getLogger('training_debug')
        debug_logger.setLevel(logging.DEBUG)
        if not debug_logger.handlers:
            fh = logging.FileHandler('/home/ubuntu/classificatore/logging.log') # Percorso fisso per debug avanzato
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            debug_logger.addHandler(fh)
        
        debug_logger.info("=" * 80)
        debug_logger.info("🚀 INIZIO esegui_training_interattivo()")
        debug_logger.info(f"   📋 Parametri ricevuti:")
        debug_logger.info(f"      giorni_indietro: {giorni_indietro}")
        debug_logger.info(f"      limit: {limit}")
        debug_logger.info(f"      max_human_review_sessions: {max_human_review_sessions}")
        debug_logger.info(f"      confidence_threshold: {confidence_threshold}")
        debug_logger.info(f"      force_review: {force_review}")
        debug_logger.info(f"      disagreement_threshold: {disagreement_threshold}")
        debug_logger.info(f"   🏢 Tenant: {self.tenant.tenant_name if self.tenant else 'N/A'}")
        debug_logger.info("=" * 80)
        
        # 🔧 CORREZIONE: Aggiorna il confidence threshold con il valore passato
        self.confidence_threshold = confidence_threshold
        print(f"🎯 Confidence threshold aggiornato a: {self.confidence_threshold}")
        
        # Determina limite review umana dalla configurazione DATABASE
        try:
            # NUOVO: Leggi parametri dal database MySQL
            if hasattr(self, 'tenant') and self.tenant:
                training_params = get_supervised_training_params_from_db(self.tenant.tenant_id)
                print(f"✅ Parametri training da database MySQL")
            else:
                print(f"⚠️ Tenant non disponibile, leggo da config.yaml")
                training_params = None
            
            if training_params:
                # Usa parametri dal database
                human_limit = training_params.get('max_total_sessions', max_human_review_sessions)
                confidence_threshold_priority = training_params.get('confidence_threshold_priority', confidence_threshold)
                print(f"📊 [DB MYSQL] max_total_sessions: {human_limit}")
                print(f"📊 [DB MYSQL] confidence_threshold_priority: {confidence_threshold_priority}")
            else:
                # Fallback a config.yaml
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                
                supervised_config = config.get('supervised_training', {}) # Sezione supervisione avanzata get config 
                human_review_config = supervised_config.get('human_review', {}) # Sezione review umana get config 
                
                human_limit = human_review_config.get('max_total_sessions', max_human_review_sessions)
                confidence_threshold_priority = human_review_config.get('confidence_threshold_priority',confidence_threshold_priority)
                print(f"📊 [CONFIG YAML] max_total_sessions: {human_limit}")
                print(f"📊 [CONFIG YAML] confidence_threshold_priority: {confidence_threshold_priority}")
            
            # Gestione parametri legacy
            if max_human_review_sessions is not None:
                human_limit = max_human_review_sessions
            elif limit is not None:
                human_limit = limit  # Retrocompatibilità
                
        except Exception as e:
            print(f"⚠️ Errore lettura config: {e}")
            human_limit = limit or 500
        
        debug_logger.info(f"📊 PARAMETRI DETERMINATI:")
        debug_logger.info(f"   human_limit: {human_limit}")
        debug_logger.info(f"   confidence_threshold: {confidence_threshold}")
        debug_logger.info(f"   force_review: {force_review}")
        debug_logger.info(f"   disagreement_threshold: {disagreement_threshold}")

        print(f"🎓 TRAINING SUPERVISIONATO AVANZATO")
        print(f"🚨 [DEBUG] esegui_training_interattivo() CHIAMATA CONFERMATA")
        print(f"🚨 [DEBUG] Funzione: esegui_training_interattivo")
        print(f"🚨 [DEBUG] Parametri ricevuti:")
        print(f"   - max_human_review_sessions: {max_human_review_sessions}")
        print(f"   - confidence_threshold: {confidence_threshold}")
        print(f"   - force_review: {force_review}")
        print(f"   - disagreement_threshold: {disagreement_threshold}")
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
            print(f"🚨 [DEBUG] FASE 1 COMPLETATA - Estrazione sessioni")
            print(f"🚨 [DEBUG] Sessioni estratte: {len(sessioni)}")
            
            debug_logger.info(f"📊 FASE 1 COMPLETATA - Estrazione sessioni")
            debug_logger.info(f"   Sessioni estratte: {len(sessioni)}")

            # 2. Clustering COMPLETO su tutto il dataset
            print(f"\n📊 FASE 2: CLUSTERING COMPLETO")
            print(f"🚨 [DEBUG] AVVIO CLUSTERING nella funzione esegui_training_interattivo")
            
            debug_logger.info(f"📊 AVVIO FASE 2 - Clustering completo")
            embeddings, cluster_labels, representatives, suggested_labels = self.esegui_clustering(sessioni)
            
            # Genera cluster_info dai cluster_labels per evitare duplicazione
            session_texts = [sessioni[sid]['testo_completo'] for sid in sessioni.keys()]
            cluster_info = self._generate_cluster_info_from_labels(cluster_labels, session_texts)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_outliers = sum(1 for label in cluster_labels if label == -1)
            print(f"✅ Clustering completo: {n_clusters} cluster, {n_outliers} outlier")
            print(f"🚨 [DEBUG] FASE 2 COMPLETATA - Clustering")
            print(f"🚨 [DEBUG] Cluster trovati: {n_clusters}, Outlier: {n_outliers}")
            print(f"🚨 [DEBUG] Rappresentanti totali: {len(representatives)}")
            print(f"🚨 [DEBUG] Cluster info generato: {len(cluster_info)} cluster")
            
            debug_logger.info(f"📊 FASE 2 COMPLETATA - Clustering")
            debug_logger.info(f"   Cluster trovati: {n_clusters}, Outlier: {n_outliers}")
            debug_logger.info(f"   Rappresentanti totali: {len(representatives)}")

            # 3. Selezione intelligente rappresentanti per review umana
            print(f"\n📊 FASE 3: SELEZIONE RAPPRESENTANTI PER REVIEW UMANA")
            print(f"🚨 [DEBUG] AVVIO SELEZIONE RAPPRESENTANTI - funzione esegui_training_interattivo")
            limited_representatives, selection_stats = self._select_representatives_for_human_review(
                representatives, suggested_labels, human_limit, sessioni,
                confidence_threshold=confidence_threshold,
                force_review=force_review,
                disagreement_threshold=disagreement_threshold
            )
            trace_all("I rappresentanti selezionati sono: ", "DEBUG", selected_keys=list(limited_representatives.keys()))
            
            print(f"✅ Selezione completata:")
            print(f"  📋 Cluster originali: {len(representatives)}")
            print(f"  👤 Cluster per review: {len(limited_representatives)}")
            print(f"  📝 Sessioni per review: {selection_stats['total_sessions_for_review']}")
            print(f"  🚫 Cluster esclusi: {selection_stats['excluded_clusters']}")
            print(f"🚨 [DEBUG] FASE 3 COMPLETATA - Selezione rappresentanti")
            print(f"🚨 [DEBUG] Limited representatives keys: {list(limited_representatives.keys())}")
            
            debug_logger.info(f"📊 FASE 3 COMPLETATA - Selezione rappresentanti")
            debug_logger.info(f"   Cluster originali: {len(representatives)}")
            debug_logger.info(f"   Cluster per review: {len(limited_representatives)}")
            debug_logger.info(f"   Sessioni per review: {selection_stats['total_sessions_for_review']}")
            debug_logger.info(f"   Cluster esclusi: {selection_stats['excluded_clusters']}")

            # 4. Classificazione completa con ensemble ML+LLM
            print(f"\n📊 FASE 4: CLASSIFICAZIONE COMPLETA CON ENSEMBLE ML+LLM")
            print(f"🚨 [DEBUG] AVVIO CLASSIFICAZIONE ENSEMBLE COMPLETA")
            
            debug_logger.info(f"📊 AVVIO FASE 4 - Classificazione ensemble completa")
            
            # Usa la funzione esistente che fa tutto: ensemble, salvataggio, outlier, propagazione
            classification_results = self.classifica_e_salva_sessioni(
                sessioni=sessioni,
                batch_size=32,
                use_ensemble=True,
                optimize_clusters=True,
                force_review=False,
                embeddings=embeddings,
                cluster_labels=cluster_labels,
                cluster_info=cluster_info
            )
            
            print(f"✅ Classificazione ensemble completa:")
            print(f"  � Sessioni processate: {classification_results.get('total_sessions', 0)}")
            print(f"  💾 Salvate con successo: {classification_results.get('saved_successfully', 0)}")
            print(f"  🎯 Alta confidenza: {classification_results.get('high_confidence', 0)}")
            print(f"  ⚠️ Bassa confidenza: {classification_results.get('low_confidence', 0)}")
            print(f"🚨 [DEBUG] FASE 4 COMPLETATA - Classificazione completa")
            
            debug_logger.info(f"📊 FASE 4 COMPLETATA - Classificazione completa")
            debug_logger.info(f"   Sessioni processate: {classification_results.get('total_sessions', 0)}")
            debug_logger.info(f"   Salvate con successo: {classification_results.get('saved_successfully', 0)}")
            
            # Aggiorna le metriche di training per compatibilità
            training_metrics = {
                'classification_stats': classification_results,
                'ensemble_used': True,
                'interactive_training': False,
                'phase': 'complete_classification_with_ensemble'
            }
            
            print(f"🚨🚨🚨 [DEBUG CRITICO] allena_classificatore() COMPLETATO!")
            print(f"🚨🚨🚨 [DEBUG CRITICO] Training metrics ricevuti: {type(training_metrics)}")
            print(f"🚨🚨🚨 [DEBUG CRITICO] Training metrics keys: {list(training_metrics.keys()) if isinstance(training_metrics, dict) else 'Not a dict'}")
            
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
                'training_metrics': training_metrics,
                
                # 🆕 INCLUDI WARNING PER INTERFACCIA UTENTE  
                'warnings': getattr(self, 'training_warnings', [])
            }
            
            print("-" * 50)
            print(f"✅ TRAINING SUPERVISIONATO COMPLETATO!")
            print(f"⏱️  Durata: {duration.total_seconds():.1f} secondi")
            print(f"📊 Dataset: {len(sessioni)} sessioni totali")
            print(f"🧩 Clustering: {n_clusters} cluster su dataset completo")
            print(f"� Sessioni selezionate per review: {selection_stats['total_sessions_for_review']}")
            
            if 'human_feedback_stats' in training_metrics:
                feedback = training_metrics['human_feedback_stats']
                print(f"👤 Review umane effettuate: {feedback.get('total_reviews', 0)}")
                print(f"✅ Etichette approvate dall'umano: {feedback.get('approved_labels', 0)}")
                print(f"📝 Nuove etichette create dall'umano: {feedback.get('new_labels', 0)}")
            
            trace_all("esegui_training_interattivo", "EXIT", return_value=results)
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
        
        # Carica configurazione DAL DATABASE
        try:
            # NUOVO: Leggi parametri dal database MySQL
            if hasattr(self, 'tenant') and self.tenant:
                training_params = get_supervised_training_params_from_db(self.tenant.tenant_id)
                print(f"✅ Parametri selezione rappresentanti da database MySQL")
            else:
                print(f"⚠️ Tenant non disponibile, leggo da config.yaml")
                training_params = None
            
            if training_params:
                # Usa parametri dal database
                min_reps_per_cluster = training_params.get('min_representatives_per_cluster', 1)
                max_reps_per_cluster = training_params.get('max_representatives_per_cluster', 5)
                default_reps_per_cluster = training_params.get('representatives_per_cluster', 3)
                selection_strategy = training_params.get('selection_strategy', 'prioritize_by_size')
                confidence_threshold_priority = training_params.get('confidence_threshold_priority', 0.7)
                print(f"📊 [DB MYSQL] min_representatives_per_cluster: {min_reps_per_cluster}")
                print(f"📊 [DB MYSQL] max_representatives_per_cluster: {max_reps_per_cluster}")
                print(f"📊 [DB MYSQL] representatives_per_cluster: {default_reps_per_cluster}")
                print(f"📊 [DB MYSQL] selection_strategy: {selection_strategy}")
                print(f"📊 [DB MYSQL] confidence_threshold_priority: {confidence_threshold_priority}")
                min_cluster_size = 2  # Fisso, non configurabile
            else:
                # Fallback a config.yaml
                print (f"⚠️ Parametri selezione rappresentanti non trovati nel DB, leggo da config.yaml")
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
                print(f"📊 [CONFIG YAML] Parametri da config.yaml")
            
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
            
        Autore: Valerio Bignardi
        Data creazione: 2025-09-06
        Ultima modifica: 2025-09-06 - Aggiunta del tracing completo
        """
        trace_all("get_ensemble_statistics", "ENTER", 
                 ensemble_available=self.ensemble_classifier is not None)
        
        if not self.ensemble_classifier:
            result = {'error': 'Ensemble classifier non inizializzato'}
            trace_all("get_ensemble_statistics", "EXIT", 
                     return_value=result, success=False)
            return result
        
        # Usa il metodo integrato dell'advanced ensemble classifier
        result = self.ensemble_classifier.get_ensemble_statistics()
        trace_all("get_ensemble_statistics", "EXIT", 
                 return_value=result, success=True)
        return result
    
    def adjust_ensemble_weights(self, llm_weight: float, ml_weight: float) -> None:
        """
        Regola manualmente i pesi dell'ensemble classifier
        
        Args:
            llm_weight: Nuovo peso per LLM (0.0 - 1.0)
            ml_weight: Nuovo peso per ML (0.0 - 1.0)
            
        Autore: Valerio Bignardi
        Data creazione: 2025-09-06
        Ultima modifica: 2025-09-06 - Aggiunta del tracing completo
        """
        trace_all("adjust_ensemble_weights", "ENTER", 
                 llm_weight=llm_weight, ml_weight=ml_weight,
                 ensemble_available=self.ensemble_classifier is not None)
        
        if not self.ensemble_classifier:
            print("❌ Ensemble classifier non disponibile")
            trace_all("adjust_ensemble_weights", "EXIT", 
                     success=False, reason="Ensemble classifier non disponibile")
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
        
        trace_all("adjust_ensemble_weights", "EXIT", 
                 success=True, final_llm_weight=llm_weight, final_ml_weight=ml_weight)
    
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
        start_time = time.time()
        print(f"\n🚀 [FASE 8: DEDUPPLICAZIONE] Avvio scoperta e normalizzazione tag...")
        print(f"📊 [FASE 8: DEDUPPLICAZIONE] Etichette candidate: {len(suggested_labels)}")
        
        try:
            # Usa il nuovo sistema intelligente
            deduplicated_labels = self.label_deduplicator.prevent_duplicate_labels(suggested_labels)
            
            # Analizza risultati
            stats = self.label_deduplicator.get_statistics()
            deduplication_time = time.time() - start_time
            
            print(f"✅ [FASE 8: DEDUPPLICAZIONE] Completata in {deduplication_time:.2f}s")
            if stats['total_decisions'] > 0:
                print(f"📊 [FASE 8: DEDUPPLICAZIONE] Risultati:")
                print(f"   🔄 Etichette riusate: {stats['labels_reused']}")
                print(f"   🆕 Nuove etichette: {stats['labels_created']}")
                print(f"   📈 Tasso riuso: {stats['reuse_rate']:.1%}")
                print(f"   🏷️ Database tag aggiornato: {stats['labels_reused'] + stats['labels_created']} totali")
            else:
                print(f"⚠️ [FASE 8: DEDUPPLICAZIONE] Nessuna dedupplicazione necessaria")
            
            return deduplicated_labels
            
        except Exception as e:
            deduplication_time = time.time() - start_time
            print(f"❌ [FASE 8: DEDUPPLICAZIONE] ERRORE dopo {deduplication_time:.2f}s: {e}")
            print(f"� [FASE 8: DEDUPPLICAZIONE] Fallback: etichette originali")
            # Ritorna le etichette originali senza modifiche se il sistema intelligente fallisce
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
        Prova a caricare il modello ML più recente specifico per il tenant corrente
        """
        try:
            import os
            import glob            
            models_dir = "models"
            if not os.path.exists(models_dir):
                print("⚠️ Directory models/ non trovata, nessun modello da caricare")
                return
            
            # Cerca modelli specifici per questo tenant
            tenant_pattern = os.path.join(models_dir, f"{self.tenant_slug}_*_config.json")
            config_files = glob.glob(tenant_pattern)
            
            if not config_files:
                print(f"⚠️ Nessun modello trovato per tenant '{self.tenant_slug}' nella directory models/")
                print(f"🚀 ATTIVAZIONE AUTO-TRAINING: Tentativo di addestramento automatico...")
                
                # Verifica se auto-training è possibile
                if self._should_enable_auto_training():
                    print(f"✅ Auto-training abilitato - Esecuzione training automatico")
                    auto_training_result = self._execute_auto_training()
                    if auto_training_result:
                        print(f"✅ Auto-training completato con successo!")
                        # Ri-tenta il caricamento del modello appena creato
                        config_files = glob.glob(tenant_pattern)
                        if config_files:
                            print(f"🔄 Ricaricamento modello auto-addestrato...")
                            # Continua con il normale flusso di caricamento
                        else:
                            print(f"⚠️ Auto-training completato ma nessun modello trovato")
                            return
                    else:
                        print(f"❌ Auto-training fallito - Continuando senza modelli")
                        return
                else:
                    print(f"⚠️ Auto-training non possibile - Continuando senza modelli")
                    return
            
            # Ordina per data (più recente per ultimo)
            config_files.sort()
            latest_config = config_files[-1]
            model_base_name = latest_config.replace("_config.json", "")
            ml_file = f"{model_base_name}_ml_ensemble.pkl"
            if not os.path.exists(ml_file):
                print(f"⚠️ File ML non trovato: {ml_file}")
                return
            
            print(f"📥 Caricamento ultimo modello per {self.tenant_slug}: {os.path.basename(model_base_name)}")
            self.ensemble_classifier.load_ensemble_model(model_base_name)
            print("✅ Modello caricato con successo")

            # Carica eventuale provider BERTopic accoppiato
            subdir = self.bertopic_config.get('model_subdir', 'bertopic')
            provider_dir = f"{model_base_name}_{subdir}"
            
            print(f"\n🔍 VERIFICA BERTOPIC LOADING:")
            print(f"   📋 BERTopic enabled: {self.bertopic_config.get('enabled', False)}")
            print(f"   📋 BERTopic available: {_BERTopic_AVAILABLE}")
            print(f"   📁 Provider directory: {provider_dir}")
            print(f"   📁 Directory exists: {os.path.isdir(provider_dir)}")
            
            if os.path.isdir(provider_dir):
                provider_files = [f for f in os.listdir(provider_dir) if os.path.isfile(os.path.join(provider_dir, f))]
                print(f"   � File trovati: {len(provider_files)} -> {provider_files}")
            
            if self.bertopic_config.get('enabled', False) and _BERTopic_AVAILABLE and os.path.isdir(provider_dir):
                try:
                    print(f"\n🔄 CARICAMENTO BERTOPIC:")
                    print(f"   📁 Path completo: {os.path.abspath(provider_dir)}")
                    
                    start_load = time.time()
                    provider = BERTopicFeatureProvider().load(provider_dir)
                    load_time = time.time() - start_load
                    
                    print(f"   ⏱️ Caricamento completato in {load_time:.2f} secondi")
                    
                    # Verifica se il provider è stato caricato correttamente
                    if provider is not None:
                        print(f"   ✅ BERTopic provider caricato con successo")
                        self.ensemble_classifier.set_bertopic_provider(
                            provider,
                            top_k=self.bertopic_config.get('top_k', 15),
                            return_one_hot=self.bertopic_config.get('return_one_hot', False)
                        )
                        print(f"   ✅ BERTopic provider configurato nell'ensemble")
                    else:
                        print(f"   ⚠️ BERTopic provider restituito None - modello corrotto o incompatibile")
                        print(f"   💡 Suggerimento: eseguire training per ricreare il modello")
                        
                except Exception as e:
                    print(f"❌ ERRORE CARICAMENTO BERTOPIC: {e}")
                    print(f"   🔍 Tipo errore: {type(e).__name__}")
                    print(f"   🔍 Stack trace: {traceback.format_exc()}")
                    print("   💡 Continuando senza BERTopic provider...")
            else:
                if self.bertopic_config.get('enabled', False) and not _BERTopic_AVAILABLE:
                    print("⚠️ BERTopic abilitato ma non disponibile in runtime; proseguo senza provider")
                elif self.bertopic_config.get('enabled', False):
                    print(f"⚠️ Directory BERTopic provider non trovata: {provider_dir}")
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
        outlier_session_ids = list(tutte_sessioni_outlier.keys())
        outlier_embeddings = self._get_embedder().encode(outlier_texts, session_ids=outlier_session_ids)
        
        # Usa parametri più permissivi per outlier
        from Clustering.hdbscan_clusterer import HDBSCANClusterer
        outlier_clusterer = HDBSCANClusterer(
            min_cluster_size=2,  # Molto più basso
            min_samples=1,       # Molto più basso
            cluster_selection_epsilon=0.1,  # Più permissivo
            tenant=self.tenant  # 🔧 PASSA OGGETTO TENANT
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

    def _classifica_ottimizzata_cluster(self, 
                                      sessioni: Dict[str, Dict], 
                                      session_ids: List[str],
                                      session_texts: List[str],
                                      embeddings: Optional[np.ndarray] = None,
                                      cluster_labels: Optional[np.ndarray] = None,
                                      cluster_info: Optional[Dict] = None,
                                      batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Classificazione ottimizzata basata su cluster:
        1. USA clustering già fatto (parametri passati) 
        2. Seleziona rappresentanti per ogni cluster
        3. Classifica SOLO i rappresentanti con ML+LLM 
        4. Propaga automaticamente etichette a tutte le sessioni del cluster
        5. Gestisce outlier separatamente
        
        Args:
            sessioni: Dizionario delle sessioni {session_id: session_data}
            session_ids: Lista degli ID sessioni in ordine
            session_texts: Lista dei testi corrispondenti
            embeddings: Embeddings già calcolati (RICHIESTO)
            cluster_labels: Labels clustering già calcolati (RICHIESTO)
            cluster_info: Info cluster già calcolate (RICHIESTO)
            batch_size: Dimensione batch per ottimizzazione
            
        Returns:
            Lista predizioni per tutte le sessioni (stesso ordine di session_ids)
        """
        trace_all("_classifica_ottimizzata_cluster", "ENTER",
                 sessioni_count=len(sessioni),
                 session_ids_count=len(session_ids),
                 session_texts_count=len(session_texts),
                 batch_size=batch_size)
        
        # 🔍 DEBUG: Trace dell'ingresso nel metodo ottimizzato
        from Pipeline.debug_pipeline import debug_pipeline, debug_flow, debug_exception
        
        debug_pipeline("_classifica_ottimizzata_cluster", "ENTRY - Avvio classificazione ottimizzata", {
            "num_sessioni": len(sessioni),
            "num_session_ids": len(session_ids),
            "num_session_texts": len(session_texts),
            "batch_size": batch_size,
            "tenant": self.tenant_slug
        }, "ENTRY")
        
        print(f"🎯 CLASSIFICAZIONE OTTIMIZZATA PER CLUSTER")
        print(f"   📊 Sessioni totali: {len(sessioni)}")
        print(f"   🧩 Clustering già fatto: {embeddings is not None and cluster_labels is not None}")
        
        # Valida parametri obbligatori
        if embeddings is None or cluster_labels is None or cluster_info is None:
            raise ValueError("_classifica_ottimizzata_cluster richiede embeddings, cluster_labels e cluster_info")
        
        print(f"✅ USO CLUSTERING ESISTENTE (evita duplicazione)")
        print(f"   📊 Embeddings shape: {embeddings.shape}")
        print(f"   🏷️  Labels count: {len(cluster_labels)}")
        print(f"   🧩 Cluster info: {len(cluster_info)} cluster")
        
        try:
            # Usa clustering fornito come parametri
            n_clusters = len([l for l in cluster_labels if l != -1])
            n_outliers = sum(1 for l in cluster_labels if l == -1)
            
            print(f"   📈 Cluster trovati: {n_clusters}")
            print(f"   🔍 Outliers: {n_outliers}")
            
            # STEP 2: Selezione rappresentanti per ogni cluster 
            print(f"👥 STEP 2: Selezione rappresentanti per classificazione...")
            representatives = {}
            suggested_labels = {}
            
            # Raggruppa sessioni per cluster
            cluster_sessions = {}
            for i, (session_id, cluster_id) in enumerate(zip(session_ids, cluster_labels)):
                if cluster_id not in cluster_sessions:
                    cluster_sessions[cluster_id] = []
                cluster_sessions[cluster_id].append({
                    'session_id': session_id,
                    'index': i,
                    'testo_completo': session_texts[i],
                    **sessioni[session_id]
                })
            
            # Seleziona rappresentanti per cluster validi (non outlier)
            for cluster_id, sessions in cluster_sessions.items():
                if cluster_id == -1:  # Salta outlier per ora
                    continue
                    
                # Seleziona max 3 rappresentanti per cluster
                max_reps = min(3, len(sessions))
                representatives[cluster_id] = sessions[:max_reps]
                suggested_labels[cluster_id] = cluster_info.get(cluster_id, {}).get('intent_string', f'cluster_{cluster_id}')
                
                print(f"   🏷️  Cluster {cluster_id}: {len(sessions)} sessioni, {max_reps} rappresentanti")
            
            # STEP 3: Classificazione dei soli rappresentanti
            start_time = time.time()
            print(f"\n🚀 [FASE 5: CLASSIFICAZIONE] Avvio classificazione rappresentanti...")
            
            representative_predictions = {}
            total_representatives = sum(len(reps) for reps in representatives.values())
            
            print(f"📊 [FASE 5: CLASSIFICAZIONE] Target: {total_representatives} rappresentanti")
            print(f"🎯 [FASE 5: CLASSIFICAZIONE] Ottimizzazione: {total_representatives} invece di {len(sessioni)} sessioni totali")
            
            rep_count = 0
            success_count = 0
            error_count = 0
            
            for cluster_id, reps in representatives.items():
                cluster_predictions = []
                
                print(f"📋 [FASE 5: CLASSIFICAZIONE] Cluster {cluster_id}: {len(reps)} rappresentanti")
                
                for rep in reps:
                    rep_count += 1
                    rep_text = rep['testo_completo']
                    
                    # Classifica il rappresentante con ensemble ML+LLM
                    try:
                        # 🚀 OTTIMIZZAZIONE: Usa features ML cached se disponibili
                        cached_features = self._get_cached_ml_features(rep['session_id'])
                        if cached_features is not None:
                            print(f"   ✅ Usando features cached per rappresentante {rep['session_id']}")
                        
                        prediction = self.ensemble_classifier.predict_with_ensemble(
                            rep_text,
                            return_details=True,
                            embedder=self.embedder,
                            ml_features_precalculated=cached_features
                        )
                        prediction['representative_session_id'] = rep['session_id']
                        prediction['cluster_id'] = cluster_id
                        cluster_predictions.append(prediction)
                        success_count += 1
                        
                        if rep_count % 10 == 0 or rep_count == total_representatives:  # Progress ogni 10 reps
                            percent = (rep_count / total_representatives) * 100
                            print(f"⚡ [FASE 5: CLASSIFICAZIONE] Progress: {rep_count}/{total_representatives} ({percent:.1f}%)")
                        
                    except Exception as e:
                        error_count += 1
                        print(f"⚠️ [FASE 5: CLASSIFICAZIONE] Errore rep {rep['session_id']}: {e}")
                        # Fallback con bassa confidenza
                        cluster_predictions.append({
                            'predicted_label': 'altro',
                            'confidence': 0.3,
                            'ensemble_confidence': 0.3,
                            'method': 'REP_FALLBACK',
                            'representative_session_id': rep['session_id'],
                            'cluster_id': cluster_id,
                            'llm_prediction': None,
                            'ml_prediction': {'predicted_label': 'altro', 'confidence': 0.3}
                        })
                
                representative_predictions[cluster_id] = cluster_predictions
                
            classification_time = time.time() - start_time
            print(f"✅ [FASE 5: CLASSIFICAZIONE] Completata in {classification_time:.2f}s")
            print(f"📊 [FASE 5: CLASSIFICAZIONE] Risultati:")
            print(f"   ✅ Successi: {success_count}/{total_representatives}")
            print(f"   ❌ Errori: {error_count}")
            print(f"   ⚡ Throughput: {success_count/classification_time:.1f} classificazioni/secondo")
            
            # STEP 4: Propagazione etichette ai cluster
            start_time = time.time()
            print(f"\n� [FASE 6: PROPAGAZIONE] Avvio logica consenso...")
            print(f"📊 [FASE 6: PROPAGAZIONE] Cluster da processare: {len(representative_predictions)}")
            
            cluster_final_labels = {}
            auto_classified = 0
            needs_review = 0
            
            for cluster_id, predictions in representative_predictions.items():
                if not predictions:
                    continue
                    
                # Conta le etichette per trovare consenso
                label_counts = {}
                for pred in predictions:
                    label = pred.get('predicted_label', 'altro')
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                # Trova etichetta più votata
                most_common_label = max(label_counts.keys(), key=lambda k: label_counts[k])
                consensus_votes = label_counts[most_common_label]
                total_votes = len(predictions)
                consensus_ratio = consensus_votes / total_votes
                
                # Logica consenso (70% threshold)
                if consensus_ratio >= 0.7:
                    auto_classified += 1
                    review_needed = False
                    reason = f"consenso_{consensus_ratio:.0%}"
                elif consensus_ratio == 0.5 and total_votes == 2:
                    needs_review += 1
                    review_needed = True
                    reason = "pareggio_50_50"
                else:
                    needs_review += 1
                    review_needed = True
                    reason = f"consenso_basso_{consensus_ratio:.0%}"
                
                # Scegli migliore predizione per calcolo confidence
                best_prediction = max(predictions, key=lambda p: p.get('confidence', 0))
                
                cluster_final_labels[cluster_id] = {
                    'label': most_common_label,
                    'confidence': best_prediction.get('confidence', 0.5),
                    'consensus_ratio': consensus_ratio,
                    'total_representatives': total_votes,
                    'needs_review': review_needed,
                    'reason': reason,
                    'method': 'CLUSTER_PROPAGATED',
                    'source_representative': best_prediction['representative_session_id']
                }
                
                status_symbol = "🎯" if not review_needed else "👤"
                print(f"   {status_symbol} Cluster {cluster_id}: '{most_common_label}' ({consensus_ratio:.0%} consenso)")
            
            propagation_time = time.time() - start_time
            print(f"✅ [FASE 6: PROPAGAZIONE] Completata in {propagation_time:.2f}s")
            print(f"📊 [FASE 6: PROPAGAZIONE] Risultati:")
            print(f"   🎯 Auto-classificati: {auto_classified} cluster (≥70% consenso)")
            print(f"   👤 Richiedono review: {needs_review} cluster (<70% consenso)")
            
            # STEP 5: Costruzione predizioni finali per tutte le sessioni
            # ✅ CORREZIONE BUG 2025-08-29: Aggiunta cluster_metadata a tutte le predizioni
            # per garantire corretta classificazione tipo (RAPPRESENTANTE/PROPAGATO/OUTLIER vs NORMALE)
            print(f"🏗️  STEP 5: Costruzione predizioni finali...")
            all_predictions = []
            
            for i, (session_id, cluster_id) in enumerate(zip(session_ids, cluster_labels)):
                
                if cluster_id != -1 and cluster_id in cluster_final_labels:
                    # Sessione in cluster: usa etichetta propagata
                    cluster_label_info = cluster_final_labels[cluster_id]
                    
                    # Verifica se questa sessione è un rappresentante
                    is_representative = any(
                        rep['session_id'] == session_id 
                        for reps in representatives.get(cluster_id, []) 
                        for rep in ([reps] if isinstance(reps, dict) else reps)
                    )
                    
                    if is_representative:
                        # Trova la predizione originale del rappresentante
                        original_pred = None
                        for pred in representative_predictions.get(cluster_id, []):
                            if pred.get('representative_session_id') == session_id:
                                original_pred = pred
                                break
                        
                        if original_pred:
                            # 🔧 FIX CRITICO: Preserva le predictions ensemble originali
                            prediction = original_pred.copy()
                            
                            # ✅ PRESERVA metodo ensemble originale per final_decision
                            ensemble_method = prediction.get('method', 'ENSEMBLE')  # ENSEMBLE, LLM, ML
                            
                            # ✅ SALVA metodo cluster separato per metadata
                            prediction['cluster_method'] = 'REPRESENTATIVE'
                            
                            # ✅ MANTIENI metodo ensemble originale 
                            prediction['method'] = ensemble_method
                            
                            print(f"   ✅ RAPPRESENTANTE {session_id}: ensemble_method={ensemble_method}, cluster_method=REPRESENTATIVE")
                            
                        else:
                            # 🚨 ERRORE GRAVE: Un rappresentante non ha predizione originale!
                            # Questo non dovrebbe mai accadere se il codice funziona correttamente
                            print(f"❌ ERRORE CRITICO: Rappresentante {session_id} del cluster {cluster_id} non trovato in representative_predictions!")
                            print(f"   Available representatives: {[p.get('representative_session_id') for p in representative_predictions.get(cluster_id, [])]}")
                            raise Exception(f"Bug nel matching rappresentanti: {session_id} non trovato")
                        
                        # ✅ CORREZIONE BUG: Aggiungi cluster_metadata per RAPPRESENTANTE
                        prediction['cluster_metadata'] = {
                            'cluster_id': cluster_id,
                            'selection_reason': 'rappresentante',
                            'is_representative': True,
                            'cluster_size': len(cluster_sessions.get(cluster_id, [])),
                            'consensus_ratio': cluster_label_info.get('consensus_ratio', 1.0),
                            'total_representatives': cluster_label_info.get('total_representatives', 1)
                        }
                        
                    else:
                        # Sessione normale: usa etichetta propagata
                        prediction = {
                            'predicted_label': cluster_label_info['label'],
                            'confidence': cluster_label_info['confidence'],
                            'ensemble_confidence': cluster_label_info['confidence'],
                            'method': 'CLUSTER_PROPAGATED',  # ✅ Mantieni metodo cluster per propagati
                            'cluster_method': 'CLUSTER_PROPAGATED',  # ✅ Aggiungi anche cluster_method 
                            'cluster_id': cluster_id,
                            'source_representative': cluster_label_info['source_representative'],
                            'llm_prediction': None,
                            'ml_prediction': {'predicted_label': cluster_label_info['label'], 'confidence': cluster_label_info['confidence']}
                        }
                        
                        # ✅ CORREZIONE BUG: Aggiungi cluster_metadata per PROPAGATO
                        prediction['cluster_metadata'] = {
                            'cluster_id': cluster_id,
                            'selection_reason': 'propagato',
                            'is_representative': False,
                            'propagated_from': cluster_label_info['source_representative'],
                            'cluster_size': len(cluster_sessions.get(cluster_id, [])),
                            'consensus_ratio': cluster_label_info.get('consensus_ratio', 1.0),
                            'needs_review': cluster_label_info.get('needs_review', False)
                        }
                
                else:
                    # 🎯 OUTLIER: Classifica con ensemble ML+LLM (NON con IntelligentIntentClusterer)
                    print(f"   🎯 Outlier {session_id}: classificazione diretta con ensemble ML+LLM...")
                    
                    # 🔧 FIX CRITICO: OUTLIER DEVONO USARE ENSEMBLE ESATTAMENTE COME I RAPPRESENTANTI
                    try:
                        # 🚀 OTTIMIZZAZIONE: Usa features ML cached se disponibili per outlier
                        cached_features = self._get_cached_ml_features(session_id)
                        if cached_features is not None:
                            print(f"   ✅ Usando features cached per outlier {session_id}")
                        
                        # ✅ CLASSIFICAZIONE ENSEMBLE DIRETTA per outlier
                        prediction = self.ensemble_classifier.predict_with_ensemble(
                            session_texts[i],
                            return_details=True,
                            embedder=self.embedder,
                            ml_features_precalculated=cached_features
                        )
                        
                        # ✅ PRESERVA metodo ensemble originale per final_decision
                        ensemble_method = prediction.get('method', 'ENSEMBLE')  # ENSEMBLE, LLM, ML
                        
                        # ✅ SALVA metodo cluster separato per metadata
                        prediction['cluster_method'] = 'OUTLIER'
                        prediction['cluster_id'] = -1
                        
                        # ✅ MANTIENI metodo ensemble originale 
                        prediction['method'] = ensemble_method
                        
                        print(f"   ✅ OUTLIER {session_id}: ensemble_method={ensemble_method}, cluster_method=OUTLIER")
                        print(f"   📊 ML prediction: {prediction.get('ml_prediction', {}).get('predicted_label', 'N/A')}")
                        print(f"   📊 LLM prediction: {prediction.get('llm_prediction', {}).get('predicted_label', 'N/A')}")
                        
                    except Exception as e:
                        print(f"❌ ERRORE: Classificazione ensemble outlier fallita per {session_id}: {e}")
                        # Fallback con etichetta di default
                        prediction = {
                            'predicted_label': 'altro',
                            'confidence': 0.3,
                            'ensemble_confidence': 0.3,
                            'method': 'OUTLIER_FALLBACK',
                            'cluster_method': 'OUTLIER_FALLBACK',
                            'cluster_id': -1,
                            'llm_prediction': {'predicted_label': 'altro', 'confidence': 0.3},
                            'ml_prediction': {'predicted_label': 'altro', 'confidence': 0.3}
                        }
                    
                    # ✅ Aggiungi cluster_metadata per OUTLIER
                    prediction['cluster_metadata'] = {
                        'cluster_id': -1,
                        'selection_reason': 'outlier_ensemble_classification',  # ✅ Aggiornato
                        'is_outlier': True,
                        'is_representative': True,  # 🎯 CORREZIONE: outlier = rappresentante di se stesso
                        'classified_as_representative': True,
                        'ensemble_method': prediction.get('method', 'ENSEMBLE')  # ✅ Traccia metodo ensemble
                    }
                
                all_predictions.append(prediction)
            
            # 🔍 DEBUG: Analisi delle predizioni generate
            metadata_stats = {
                "total_predictions": len(all_predictions),
                "with_cluster_metadata": sum(1 for p in all_predictions if p.get('cluster_metadata')),
                "without_cluster_metadata": sum(1 for p in all_predictions if not p.get('cluster_metadata')),
                "representative_count": sum(1 for p in all_predictions if 'REPRESENTATIVE' in p.get('method', '')),
                "propagated_count": sum(1 for p in all_predictions if p.get('method') == 'CLUSTER_PROPAGATED'),
                "outlier_count": sum(1 for p in all_predictions if 'OUTLIER' in p.get('method', ''))
            }
            
            # 🚨 DEBUG CRITICO: Verifica predizioni prima del return
            print(f"🚨 [DEBUG CRITICO] ANALISI PREDIZIONI PRE-RETURN:")
            print(f"   📊 Total predictions: {metadata_stats['total_predictions']}")
            print(f"   ✅ With cluster_metadata: {metadata_stats['with_cluster_metadata']}")
            print(f"   ❌ Without cluster_metadata: {metadata_stats['without_cluster_metadata']}")
            print(f"   🎯 Representatives: {metadata_stats['representative_count']}")
            print(f"   📡 Propagated: {metadata_stats['propagated_count']}")
            print(f"   🔍 Outliers: {metadata_stats['outlier_count']}")
            
            # 🚨 DEBUG SUPER CRITICO: Controllo prime 3 predizioni
            print(f"🚨 [DEBUG SUPER CRITICO] CAMPIONE PRIME 3 PREDIZIONI:")
            for i, pred in enumerate(all_predictions[:3]):
                session_id = session_ids[i]
                has_metadata = pred.get('cluster_metadata') is not None
                method = pred.get('method', 'UNKNOWN')
                print(f"   #{i+1} Session {session_id}:")
                print(f"      method: {method}")
                print(f"      has_cluster_metadata: {has_metadata}")
                if has_metadata:
                    cluster_id = pred['cluster_metadata'].get('cluster_id', 'N/A')
                    reason = pred['cluster_metadata'].get('selection_reason', 'N/A')
                    print(f"      cluster_id: {cluster_id}, reason: {reason}")
                else:
                    print(f"      ❌ NESSUN CLUSTER_METADATA!")
            
            debug_pipeline("_classifica_ottimizzata_cluster", "SUCCESS - Predizioni generate con metadata", metadata_stats, "SUCCESS")
            
            # Statistiche finali
            propagated_count = sum(1 for p in all_predictions if p.get('method') == 'CLUSTER_PROPAGATED')
            representative_count = sum(1 for p in all_predictions if 'REPRESENTATIVE' in p.get('method', ''))
            outlier_count = sum(1 for p in all_predictions if 'OUTLIER' in p.get('method', ''))
            
            print(f"✅ CLASSIFICAZIONE OTTIMIZZATA COMPLETATA:")
            print(f"   🎯 Rappresentanti classificati: {representative_count}")
            print(f"   📡 Propagate da cluster: {propagated_count}")
            print(f"   🔍 Outlier classificati: {outlier_count}")
            print(f"   🚀 Efficienza: {representative_count + outlier_count}/{len(sessioni)} classificazioni ML+LLM effettive")
            print(f"      (risparmio: {len(sessioni) - (representative_count + outlier_count)} classificazioni)")
            
            trace_all("_classifica_ottimizzata_cluster", "EXIT", 
                     return_value_count=len(all_predictions),
                     representative_count=representative_count,
                     propagated_count=propagated_count, 
                     outlier_count=outlier_count,
                     method="SUCCESS")
            return all_predictions
            
        except Exception as e:
            debug_exception("_classifica_ottimizzata_cluster", e, {
                "num_sessioni": len(sessioni),
                "tenant": self.tenant_slug if hasattr(self, 'tenant_slug') else 'N/A'
            })
            
            print(f"❌ ERRORE in classificazione ottimizzata: {e}")
            print(f"🔄 Fallback alla classificazione standard...")
            
            debug_pipeline("_classifica_ottimizzata_cluster", "FALLBACK - Generazione predizioni senza cluster_metadata", {
                "num_texts": len(session_texts)
            }, "WARNING")
            
            # Fallback: classificazione standard di tutte le sessioni
            predictions = []
            for i, text in enumerate(session_texts):
                try:
                    # 🚀 OTTIMIZZAZIONE: Usa features cached anche nel fallback se disponibili  
                    session_id = session_ids[i] if i < len(session_ids) else f"fallback_{i}"
                    cached_features = self._get_cached_ml_features(session_id)
                    
                    pred = self.ensemble_classifier.predict_with_ensemble(
                        text, 
                        return_details=True,
                        embedder=self.embedder,
                        ml_features_precalculated=cached_features
                    )
                    pred['method'] = 'OPTIMIZE_FALLBACK'
                    pred['classified_by'] = 'fallback_ensemble'  # ✅ FIX: aggiungo classified_by
                    # ❌ ATTENZIONE: Questo fallback NON genera cluster_metadata!
                    predictions.append(pred)
                except Exception as e2:
                    predictions.append({
                        'predicted_label': 'altro',
                        'confidence': 0.1,
                        'ensemble_confidence': 0.1,
                        'method': 'OPTIMIZE_FALLBACK_ERROR',
                        'classified_by': 'fallback_error',  # ✅ FIX: aggiungo classified_by
                        'llm_prediction': None,
                        'ml_prediction': {'predicted_label': 'altro', 'confidence': 0.1}
                    })
            
            debug_pipeline("_classifica_ottimizzata_cluster", "FALLBACK COMPLETE - Predizioni senza metadata", {
                "num_predictions": len(predictions),
                "all_missing_cluster_metadata": True
            }, "WARNING")
            
            trace_all("_classifica_ottimizzata_cluster", "EXIT", 
                     return_value_count=len(predictions), method="FALLBACK")
            return predictions

    def _generate_cluster_info_from_labels(self, cluster_labels: np.ndarray, session_texts: List[str]) -> Dict[int, Dict]:
        """
        Genera informazioni cluster dai soli labels HDBSCAN (senza BERTopic riaddestramento)
        
        Args:
            cluster_labels: Labels cluster da HDBSCAN
            session_texts: Testi delle sessioni
            
        Returns:
            Dizionario con informazioni cluster
        """
        cluster_info = {}
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip outlier
                continue
                
            # Trova sessioni nel cluster
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_texts = [session_texts[i] for i in cluster_indices]
            
            # Genera nome cluster semplice
            cluster_info[cluster_id] = {
                'size': len(cluster_indices),
                'intent': f'cluster_{cluster_id}',
                'intent_string': f'Cluster {cluster_id}',
                'confidence': 0.8,  # Confidenza media
                'method': 'HDBSCAN_SIMPLE'
            }
            
        return cluster_info

    def _propagate_labels_to_sessions(self, 
                                    sessioni: Dict[str, Dict],
                                    cluster_labels: np.ndarray,
                                    reviewed_labels: Dict[int, str],
                                    representatives: Dict[int, List[Dict]] = None) -> Dict[str, Any]:
        """
        Propaga le etichette dai rappresentanti di cluster SOLO alle sessioni propagate.
        
        CORREZIONE CRITICA: NON tocca rappresentanti né outliers già processati!
        
        LOGICA CORRETTA:
        - RAPPRESENTANTI: GIÀ salvati con classified_by='supervised_training_pipeline' - SKIP
        - OUTLIERS: GIÀ salvati con classified_by='supervised_training_pipeline' - SKIP  
        - PROPAGATI: Solo questi vengono toccati con classified_by='cluster_propagation'
        
        Args:
            sessioni: Dizionario delle sessioni {session_id: session_data}
            cluster_labels: Array delle etichette cluster per ogni sessione
            reviewed_labels: Dizionario {cluster_id: final_label} dalle review umane
            representatives: Dict dei rappresentanti per identificare chi NON toccare
                           (deve includere -1 per outlier se ci sono stati)
            
        Returns:
            Statistiche della propagazione
        """
        print(f"🔄 PROPAGAZIONE ETICHETTE SOLO AI PROPAGATI (NON rappresentanti/outliers)")
        print(f"   📊 Sessioni totali: {len(sessioni)}")
        print(f"   🏷️  Etichette da propagare: {len(reviewed_labels)}")
        
        # 🆕 CREA SET DI RAPPRESENTANTI DA NON TOCCARE
        representative_session_ids = set()
        if representatives:
            for cluster_reps in representatives.values():
                for rep in cluster_reps:
                    if isinstance(rep, dict):
                        rep_id = rep.get('session_id')
                        if rep_id:
                            representative_session_ids.add(rep_id)
                    else:
                        representative_session_ids.add(str(rep))
        
        print(f"   👑 Rappresentanti da SALTARE: {len(representative_session_ids)}")
        
        stats = {
            'total_sessions': len(sessioni),
            'labeled_sessions': 0,
            'unlabeled_sessions': 0,
            'skipped_representatives': 0,
            'skipped_outliers': 0,
            'propagated_sessions': 0,
            'propagated_by_cluster': {},
            'confidence_distribution': {},
            'save_errors': 0,
            'save_successes': 0,
            'mongo_saves': 0
        }
        
        # Connetti al database TAG per il salvataggio
        self.tag_db.connetti()
        
        try:
            # Connetti a MongoDB per salvataggio
            from mongo_classification_reader import MongoClassificationReader
            mongo_reader = MongoClassificationReader(tenant=self.tenant)
            if not mongo_reader.connect():
                print(f"❌ Errore connessione MongoDB per propagazione")
                return stats
            
            # Itera su tutte le sessioni
            session_ids = list(sessioni.keys())
            
            for i, session_id in enumerate(session_ids):
                session_data = sessioni[session_id]
                
                # 🚨 CONTROLLO CRITICO: SALTA RAPPRESENTANTI
                if session_id in representative_session_ids:
                    print(f"👑 SALTATO RAPPRESENTANTE: {session_id}")
                    stats['skipped_representatives'] += 1
                    continue
                
                # 🆕 CONTROLLO AGGIUNTIVO: VERIFICA SE GIÀ SALVATO COME RAPPRESENTANTE IN DB
                try:
                    collection = mongo_reader.db[mongo_reader.get_collection_name()]
                    existing_doc = collection.find_one(
                        {"session_id": session_id, "tenant_id": self.tenant.tenant_id}
                    )
                    if existing_doc and existing_doc.get("classification_type") == "RAPPRESENTANTE":
                        print(f"👑 SALTATO RAPPRESENTANTE DA DB: {session_id} (già salvato come RAPPRESENTANTE)")
                        stats['skipped_representatives'] += 1
                        continue
                    elif existing_doc and existing_doc.get("metadata", {}).get("representative", False):
                        print(f"👑 SALTATO RAPPRESENTANTE DA METADATA: {session_id} (metadata.representative=True)")
                        stats['skipped_representatives'] += 1
                        continue
                except Exception as e:
                    print(f"⚠️ Errore controllo rappresentante per {session_id}: {e}")
                    # Continua con il processo se il controllo fallisce
                
                # Trova il cluster di questa sessione
                cluster_id = cluster_labels[i] if i < len(cluster_labels) else -1
                
                # 🚨 CONTROLLO CRITICO: SALTA OUTLIERS (cluster_id = -1)
                if cluster_id == -1:
                    print(f"🔴 SALTATO OUTLIER: {session_id} (cluster {cluster_id})")
                    stats['skipped_outliers'] += 1
                    continue
                
                # 🆕 PROCESSA SOLO PROPAGATI (membri di cluster, NON rappresentanti)
                if cluster_id in reviewed_labels:
                    # Usa l'etichetta dal cluster con pulizia caratteri speciali
                    raw_label = reviewed_labels[cluster_id]
                    final_label = clean_label_text(raw_label)
                    if final_label != raw_label:
                        print(f"🧹 Label cluster pulita: '{raw_label}' → '{final_label}'")
                    
                    confidence = 0.85  # Alta confidenza per propagazione da cluster
                    method = 'CLUSTER_PROPAGATION'
                    notes = f"Propagata da cluster {cluster_id}"
                    stats['labeled_sessions'] += 1
                    stats['propagated_sessions'] += 1
                    
                    print(f"🔗 PROPAGAZIONE: {session_id} -> {final_label} (da cluster {cluster_id})")
                    
                    # Aggiorna statistiche per cluster
                    if cluster_id not in stats['propagated_by_cluster']:
                        stats['propagated_by_cluster'][cluster_id] = {
                            'label': final_label,
                            'count': 0
                        }
                    stats['propagated_by_cluster'][cluster_id]['count'] += 1
                    
                else:
                    # Cluster senza etichetta assegnata - skip
                    print(f"⚠️ CLUSTER SENZA ETICHETTA: {session_id} (cluster {cluster_id}) - SALTATO")
                    stats['unlabeled_sessions'] += 1
                    continue
                
                # Trova il cluster di questa sessione
                cluster_id = cluster_labels[i] if i < len(cluster_labels) else -1
                
                # Determina l'etichetta da assegnare
                if cluster_id in reviewed_labels:
                    # Usa l'etichetta dal cluster con pulizia caratteri speciali
                    raw_label = reviewed_labels[cluster_id]
                    final_label = clean_label_text(raw_label)
                    if final_label != raw_label:
                        print(f"🧹 Label cluster pulita (propagation): '{raw_label}' → '{final_label}'")
                    confidence = 0.85  # Alta confidenza per propagazione da cluster
                    method = 'CLUSTER_PROPAGATION'
                    notes = f"Propagata da cluster {cluster_id}"
                    stats['labeled_sessions'] += 1
                    
                    # Aggiorna statistiche per cluster
                    if cluster_id not in stats['propagated_by_cluster']:
                        stats['propagated_by_cluster'][cluster_id] = {
                            'label': final_label,
                            'count': 0
                        }
                    stats['propagated_by_cluster'][cluster_id]['count'] += 1
                    
                else:
                    # Cluster senza etichetta assegnata - skip
                    print(f"⚠️ CLUSTER SENZA ETICHETTA: {session_id} (cluster {cluster_id}) - SALTATO")
                    stats['unlabeled_sessions'] += 1
                    continue
                
                # 🆕 SALVATAGGIO SOLO PER SESSIONI PROPAGATE
                # Aggiorna distribuzione confidenze
                conf_range = f"{int(confidence*10)*10}%-{int(confidence*10)*10+10}%"
                stats['confidence_distribution'][conf_range] = stats['confidence_distribution'].get(conf_range, 0) + 1
                
                # Salva SOLO sessioni propagate con classified_by='cluster_propagation'
                try:
                    conversation_text = session_data.get('testo_completo', '')
                    ml_result = None
                    llm_result = None
                    
                    # 🔄 Usa logica generale per sessioni (sia cluster che outlier)
                    if hasattr(self, 'ensemble') and self.ensemble and conversation_text:
                        try:
                            # Esegui classificazione con ensemble
                            ensemble_result = self.ensemble.classify_text(conversation_text)
                            
                            if ensemble_result:
                                # Estrai ml_result se disponibile
                                if 'ml_result' in ensemble_result and ensemble_result['ml_result']:
                                    ml_pred = ensemble_result['ml_result']
                                    ml_result = {
                                        'predicted_label': ml_pred.get('predicted_label', 'unknown'),
                                        'confidence': ml_pred.get('confidence', 0.0),
                                        'method': 'ml_ensemble',
                                        'probabilities': ml_pred.get('probabilities', {})
                                    }
                                
                                # Estrai llm_result se disponibile
                                if 'llm_result' in ensemble_result and ensemble_result['llm_result']:
                                    llm_pred = ensemble_result['llm_result']
                                    llm_result = {
                                        'predicted_label': llm_pred.get('predicted_label', 'unknown'),
                                        'confidence': llm_pred.get('confidence', 0.0),
                                        'method': 'llm_ensemble',
                                        'reasoning': llm_pred.get('reasoning', '')
                                    }
                            
                            print(f"🔍 CLASSIFICAZIONE GENERALE {session_id}: ML={ml_result['predicted_label'] if ml_result else 'N/A'}, LLM={llm_result['predicted_label'] if llm_result else 'N/A'}")
                            
                        except Exception as e:
                            print(f"⚠️ Errore classificazione ensemble per {session_id}: {e}")
                            # Continua con ml_result=None, llm_result=None come fallback
                    
                    success = mongo_reader.save_classification_result(
                        session_id=session_id,
                        client_name=self.tenant.tenant_slug,  # 🔧 FIX: usa tenant_slug non tenant_id
                        # 🆕 USA RISULTATI REALI DELL'ENSEMBLE invece di None/simulazioni
                        ml_result=ml_result,  # Risultato reale ML
                        llm_result=llm_result if llm_result else {
                            'predicted_label': final_label,
                            'confidence': confidence,
                            'method': method,
                            'reasoning': notes
                        },  # Usa risultato ensemble se disponibile, altrimenti fallback
                        final_decision={
                            'predicted_label': final_label,
                            'confidence': confidence,
                            'method': method,
                            'reasoning': notes
                        },
                        conversation_text=session_data['testo_completo'],
                        needs_review=False,  # Propagazione automatica
                        classified_by='cluster_propagation',
                        notes=notes,
                        # 🆕 METADATI CLUSTER per nuova UI
                        cluster_metadata={
                            'cluster_id': cluster_id,
                            'is_representative': False,  # Sessione propagata
                            'propagated_from': 'cluster_propagation',
                            'propagation_confidence': confidence
                        }
                    )
                    
                    if success:
                        stats['save_successes'] += 1
                        stats['mongo_saves'] += 1
                    else:
                        stats['save_errors'] += 1
                        
                    # Log progresso ogni 100 sessioni
                    if (i + 1) % 100 == 0 or (i + 1) == len(session_ids):
                        print(f"   💾 Propagazione: {i+1}/{len(session_ids)} ({((i+1)/len(session_ids)*100):.1f}%)")
                        
                except Exception as e:
                    print(f"   ❌ Errore salvataggio sessione {session_id}: {e}")
                    stats['save_errors'] += 1
                except Exception as e:
                    print(f"   ⚠️ Warning salvataggio MongoDB per {session_id}: {e}")
        
        finally:
            self.tag_db.disconnetti()
        
        # Mostra statistiche finali
        print(f"✅ PROPAGAZIONE COMPLETATA!")
        print(f"   💾 Salvate: {stats['save_successes']}/{stats['total_sessions']} sessioni")
        print(f"   🔗 SOLO propagati toccati: {stats['propagated_sessions']}")
        print(f"   👑 Rappresentanti SALTATI: {stats['skipped_representatives']}")
        print(f"   � Outliers SALTATI: {stats['skipped_outliers']}")
        print(f"   🏷️  Etichettate da cluster: {stats['labeled_sessions']}")
        print(f"   ❌ Errori: {stats['save_errors']}")
        
        # Mostra distribuzione per cluster
        print(f"   📊 Distribuzione per cluster:")
        for cluster_id, cluster_info in stats['propagated_by_cluster'].items():
            print(f"      Cluster {cluster_id}: {cluster_info['count']} sessioni → '{cluster_info['label']}'")
            
        return stats
        """
        Ricarica configurazione LLM per il tenant corrente della pipeline
        
        FUNZIONE CRITICA: Permette di aggiornare il modello LLM
        senza riavviare il server quando l'utente cambia configurazione da React UI.
        
        Returns:
            Risultato del reload con dettagli
            
        Ultima modifica: 26 Agosto 2025
        """
        try:
            print(f"🔄 RELOAD LLM CONFIGURATION per pipeline tenant {self.tenant_slug}")
            
            # Usa ensemble classifier per reload
            if hasattr(self, 'ensemble_classifier') and self.ensemble_classifier:
                result = self.ensemble_classifier.reload_llm_configuration(self.tenant_slug)
                
                if result.get('success'):
                    print(f"✅ Pipeline LLM ricaricato: {result.get('old_model')} -> {result.get('new_model')}")
                else:
                    print(f"❌ Errore reload pipeline LLM: {result.get('error')}")
                
                return result
            else:
                return {
                    'success': False,
                    'error': 'Ensemble classifier non disponibile nella pipeline'
                }
                
        except Exception as e:
            print(f"❌ Errore reload LLM configuration pipeline: {e}")
            return {
                'success': False,
                'error': f'Errore pipeline reload: {str(e)}'
            }
    
    def get_current_llm_info(self) -> Dict[str, Any]:
        """
        Ottiene informazioni sul modello LLM corrente della pipeline
        
        Returns:
            Informazioni dettagliate su configurazione LLM
        """
        try:
            if hasattr(self, 'ensemble_classifier') and self.ensemble_classifier:
                llm_info = self.ensemble_classifier.get_current_llm_info()
                llm_info['tenant_slug'] = self.tenant_slug
                llm_info['pipeline_ready'] = True
                return llm_info
            else:
                return {
                    'pipeline_ready': False,
                    'error': 'Ensemble classifier non disponibile',
                    'tenant_slug': self.tenant_slug
                }
        except Exception as e:
            return {
                'pipeline_ready': False,
                'error': f'Errore info LLM: {str(e)}',
                'tenant_slug': self.tenant_slug
            }
    
    def _esegui_clustering_incrementale(self, sessioni: Dict[str, Dict]) -> tuple:
        """
        Esegue clustering incrementale usando modello esistente se disponibile
        
        Args:
            sessioni: Dizionario con le sessioni (solo nuove sessioni)
            
        Returns:
            Tuple (embeddings, cluster_labels, representatives, suggested_labels)
        """
        print(f"🎯 CLUSTERING INCREMENTALE - {len(sessioni)} nuove sessioni...")
        
        # Controlla se esiste modello HDBSCAN salvato
        model_path = f"models/hdbscan_{self.tenant_id}.pkl"
        
        if self._ha_modello_hdbscan_salvato(model_path):
            print(f"📂 Modello HDBSCAN esistente trovato: {model_path}")
            
            # Carica modello esistente
            if hasattr(self.clusterer, 'load_model_for_incremental_prediction'):
                loaded = self.clusterer.load_model_for_incremental_prediction(model_path)
                
                if loaded:
                    print(f"✅ Modello HDBSCAN caricato con successo")
                    
                    # Controlla se vale la pena usare predizione incrementale
                    if self._dovrebbe_usare_clustering_incrementale(len(sessioni)):
                        try:
                            # Genera embeddings solo per nuove sessioni
                            print(f"🔍 Encoding {len(sessioni)} nuovi testi...")
                            testi = [dati['testo_completo'] for dati in sessioni.values()]
                            session_ids = list(sessioni.keys())
                            
                            embeddings = self._get_embedder().encode(testi, show_progress_bar=True, session_ids=session_ids)
                            print(f"✅ Nuovi embedding generati: shape {embeddings.shape}")
                            
                            # Predizione incrementale
                            print(f"🔮 Predizione incrementale sui nuovi punti...")
                            new_labels, prediction_strengths = self.clusterer.predict_new_points(
                                embeddings, fit_umap=False
                            )
                            
                            # Genera rappresentanti e etichette per i nuovi cluster
                            representatives = self._generate_cluster_representatives(embeddings, new_labels, sessioni)
                            suggested_labels = self._generate_suggested_labels(representatives, new_labels)
                            
                            print(f"✅ CLUSTERING INCREMENTALE COMPLETATO")
                            print(f"   🎯 Nuovi punti assegnati a cluster esistenti")
                            print(f"   💪 Strength predizione media: {prediction_strengths.mean():.3f}")
                            
                            return embeddings, new_labels, representatives, suggested_labels
                            
                        except Exception as e:
                            print(f"❌ Errore durante predizione incrementale: {str(e)}")
                            print(f"🔄 Fallback a clustering completo...")
                    else:
                        print(f"🔄 Troppi nuovi punti, fallback a clustering completo...")
                else:
                    print(f"❌ Impossibile caricare modello, fallback a clustering completo...")
        else:
            print(f"📂 Nessun modello HDBSCAN esistente trovato")
        
        # Fallback: clustering completo
        print(f"🔄 Esecuzione clustering completo come fallback...")
        return self._esegui_clustering_completo(sessioni)
    
    def _ha_modello_hdbscan_salvato(self, model_path: str) -> bool:
        """
        Controlla se esiste un modello HDBSCAN salvato
        """
        import os
        return os.path.exists(model_path)
    
    def _dovrebbe_usare_clustering_incrementale(self, nuove_sessioni: int) -> bool:
        """
        Decide se usare clustering incrementale o completo
        
        Criteri:
        - Troppe nuove sessioni (>20% rispetto al training) → completo
        - Modello troppo vecchio (>7 giorni) → completo
        - Altrimenti → incrementale
        """
        import os
        from datetime import datetime, timedelta
        
        # Controllo numero sessioni (se troppe → clustering completo)
        MAX_RATIO_INCREMENTALE = 0.2  # 20% massimo nuove sessioni
        
        # Stima sessioni totali dal modello (se disponibile)
        model_path = f"models/hdbscan_{self.tenant_id}.pkl"
        try:
            if os.path.exists(model_path):
                import pickle
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Controlla età del modello
                timestamp_str = model_data.get('timestamp')
                if timestamp_str:
                    model_timestamp = datetime.fromisoformat(timestamp_str)
                    age_days = (datetime.now() - model_timestamp).days
                    
                    if age_days > 7:  # Modello più vecchio di 7 giorni
                        print(f"⏰ Modello troppo vecchio ({age_days} giorni), necessario retraining")
                        return False
                
                # Stima dimensioni dataset originale
                embeddings_shape = model_data.get('embeddings_shape')
                if embeddings_shape:
                    original_sessions = embeddings_shape[0]
                    ratio = nuove_sessioni / max(original_sessions, 1)
                    
                    if ratio > MAX_RATIO_INCREMENTALE:
                        print(f"📊 Troppe nuove sessioni ({ratio:.1%} del dataset), necessario retraining")
                        return False
                
                print(f"✅ Clustering incrementale appropriato")
                return True
        
        except Exception as e:
            print(f"⚠️ Errore valutazione clustering incrementale: {str(e)}")
        
        return False

    def _should_enable_auto_training(self) -> bool:
        """
        Determina se l'auto-training dovrebbe essere abilitato
        
        Criteri:
        - Deve esserci abbastanza dati per training
        - Auto-training deve essere abilitato in configurazione
        - Non deve essere in modalità solo-LLM
        
        Returns:
            True se auto-training è possibile, False altrimenti
            
        Autore: Valerio Bignardi
        Data: 2025-09-05
        """
        try:
            print(f"🔍 Valutazione auto-training per tenant '{self.tenant_slug}'...")
            
            # 1. Controlla configurazione auto_retrain
            if not getattr(self, 'auto_retrain', False):
                print(f"   ❌ Auto-retrain disabilitato in configurazione")
                return False
            
            # 2. Estrai campione dati per valutare dimensione dataset
            try:
                test_sessions = self.estrai_sessioni(limit=100)  # Campione piccolo per test
                if not test_sessions or len(test_sessions) < 20:
                    print(f"   ❌ Dataset insufficiente: {len(test_sessions) if test_sessions else 0} sessioni")
                    return False
                
                print(f"   ✅ Dataset sufficiente: {len(test_sessions)}+ sessioni disponibili")
                
            except Exception as e:
                print(f"   ❌ Errore estrazione dati test: {e}")
                return False
            
            # 3. Verifica disponibilità componenti necessari
            if not hasattr(self, 'ensemble_classifier') or not self.ensemble_classifier:
                print(f"   ❌ Ensemble classifier non disponibile")
                return False
            
            print(f"   ✅ Ensemble classifier disponibile")
            
            # 4. Controlla se BERTopic è disponibile per clustering
            try:
                from TopicModeling.bertopic_feature_provider import BERTopicFeatureProvider
                if not BERTopicFeatureProvider().is_available():
                    print(f"   ⚠️ BERTopic non disponibile, auto-training limitato")
                    # Continua comunque - può fare training solo ML
                else:
                    print(f"   ✅ BERTopic disponibile")
            except Exception as e:
                print(f"   ⚠️ Controllo BERTopic fallito: {e}")
            
            print(f"✅ Auto-training abilitato per '{self.tenant_slug}'")
            return True
            
        except Exception as e:
            print(f"❌ Errore valutazione auto-training: {e}")
            return False
    
    def _execute_auto_training(self) -> bool:
        """
        Esegue l'auto-training quando non esistono modelli
        
        Flusso:
        1. Estrae dataset per training
        2. Esegue clustering per generare etichette
        3. Addestra ML ensemble
        4. Salva modelli
        
        Returns:
            True se training completato con successo, False altrimenti
            
        Autore: Valerio Bignardi
        Data: 2025-09-05
        """
        try:
            print(f"🚀 ESECUZIONE AUTO-TRAINING per tenant '{self.tenant_slug}'")
            start_time = datetime.now()
            
            # 1. Estrazione dataset per training
            print(f"📥 FASE 1: Estrazione dataset...")
            training_sessions = self.estrai_sessioni(limit=None)  # Estrai tutto per training
            
            if not training_sessions or len(training_sessions) < 50:
                print(f"❌ Dataset insufficiente per training: {len(training_sessions) if training_sessions else 0} sessioni")
                return False
            
            print(f"✅ Dataset estratto: {len(training_sessions)} sessioni")
            
            # 2. Clustering per generare etichette automatiche
            print(f"🧩 FASE 2: Clustering automatico...")
            embeddings, cluster_labels, representatives, suggested_labels = self.esegui_clustering(training_sessions)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_outliers = sum(1 for label in cluster_labels if label == -1)
            
            if n_clusters < 3:
                print(f"❌ Troppi pochi cluster per training affidabile: {n_clusters}")
                return False
            
            print(f"✅ Clustering completato: {n_clusters} cluster, {n_outliers} outlier")
            
            # 3. Training automatico ML ensemble (non interattivo)
            print(f"🎓 FASE 3: Training ML ensemble...")
            training_metrics = self.allena_classificatore(
                training_sessions, 
                cluster_labels, 
                representatives, 
                suggested_labels, 
                interactive_mode=False  # IMPORTANTE: Modalità non interattiva
            )
            
            if not training_metrics or not training_metrics.get('training_success', False):
                print(f"❌ Training ML ensemble fallito")
                return False
            
            training_accuracy = training_metrics.get('training_accuracy', 0.0)
            print(f"✅ Training completato - Accuracy: {training_accuracy:.3f}")
            
            # 4. Verifica che i modelli siano stati salvati
            print(f"🔍 FASE 4: Verifica salvataggio modelli...")
            
            import os
            import glob
            models_dir = "models"
            tenant_pattern = os.path.join(models_dir, f"{self.tenant_slug}_*_config.json")
            saved_models = glob.glob(tenant_pattern)
            
            if not saved_models:
                print(f"❌ Nessun modello salvato dopo training")
                return False
            
            print(f"✅ Modelli salvati: {len(saved_models)} file trovati")
            
            # 5. Statistiche finali
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"")
            print(f"🎉 AUTO-TRAINING COMPLETATO PER '{self.tenant_slug}'!")
            print(f"   ⏱️ Durata: {duration:.1f} secondi")
            print(f"   📊 Sessioni processate: {len(training_sessions)}")
            print(f"   🧩 Cluster generati: {n_clusters}")
            print(f"   🎯 Accuracy ML: {training_accuracy:.3f}")
            print(f"   💾 Modelli salvati: {len(saved_models)}")
            print(f"")
            
            return True
            
        except Exception as e:
            print(f"❌ ERRORE AUTO-TRAINING: {e}")
            import traceback
            print(f"Stack trace: {traceback.format_exc()}")
            return False

    def _classifica_llm_only_e_prepara_training(self, 
                                              sessioni: Dict[str, Dict],
                                              cluster_labels: np.ndarray,
                                              representatives: Dict[int, List[Dict]],
                                              suggested_labels: Dict[int, str]) -> Dict[str, Any]:
        """
        Classifica usando SOLO LLM quando ML non è allenato e prepara dati per training futuro.
        
        RICEVE I DATI DEL CLUSTERING GIÀ FATTI - NON RICALCOLA
        
        Sequenza corretta per PRIMO AVVIO:
        1. Classificazione LLM-only dei rappresentanti (GIÀ identificati dal clustering)
        2. Classificazione LLM-only degli outlier (GIÀ identificati dal clustering)  
        3. Salvataggio immediato per training futuro e review umana
        
        Scopo della funzione: Gestire primo avvio quando ML ensemble non è disponibile
        Parametri di input: sessioni, cluster_labels, representatives, suggested_labels (GIÀ CALCOLATI)
        Parametri di output: risultati classificazione con dati preparati per training
        Valori di ritorno: Dict con statistiche e dati preparati
        Tracciamento aggiornamenti: 2025-09-06 - Valerio Bignardi - Corretto per non ricalcolare clustering
        
        Args:
            sessioni: Dizionario con le sessioni da classificare
            cluster_labels: Array con etichette cluster (GIÀ CALCOLATE)
            representatives: Dict con rappresentanti per cluster (GIÀ SELEZIONATI)  
            suggested_labels: Dict con etichette suggerite per cluster (GIÀ CALCOLATE)
            
        Returns:
            Dict con risultati classificazione e dati preparati per training
            
        Autore: Valerio Bignardi
        Data: 2025-09-06
        """
        trace_all("_classifica_llm_only_e_prepara_training", "ENTER", 
                 sessioni_count=len(sessioni),
                 cluster_labels_count=len(cluster_labels),
                 representatives_count=len(representatives),
                 suggested_labels_count=len(suggested_labels))
        
        start_time = time.time()
        print(f"\n🚀 [LLM-ONLY PRIMO AVVIO] Avvio classificazione preparatoria...")
        print(f"📊 [LLM-ONLY PRIMO AVVIO] Sessioni da processare: {len(sessioni)}")
        print(f"📊 [LLM-ONLY PRIMO AVVIO] Rappresentanti ricevuti: {sum(len(reps) for reps in representatives.values())}")
        print(f"📊 [LLM-ONLY PRIMO AVVIO] Cluster trovati: {len(representatives)}")
        print(f"🎯 [LLM-ONLY PRIMO AVVIO] Modalità: Classificazione + Preparazione Training")
        
        try:
            # Verifica che LLM sia disponibile
            if not self.ensemble_classifier.llm_classifier or not self.ensemble_classifier.llm_classifier.is_available():
                raise RuntimeError("LLM classifier non disponibile per classificazione primo avvio")
            
            n_outliers = list(cluster_labels).count(-1)
            total_representatives = sum(len(reps) for reps in representatives.values())
            
            print(f"   ✅ Dati clustering ricevuti:")
            print(f"   📊 Cluster: {len(representatives)}")
            print(f"   🔍 Outliers nel dataset: {n_outliers}")
            print(f"   👥 Rappresentanti totali: {total_representatives}")
            
            # 1. CLASSIFICAZIONE LLM-ONLY DEI RAPPRESENTANTI
            print(f"\n� FASE 1: CLASSIFICAZIONE LLM-ONLY RAPPRESENTANTI")
            training_data_list = []
            saved_count = 0
            
            # Istanza MongoClassificationReader per salvataggio
            mongo_reader = self._get_mongo_reader()
            
            for cluster_id, cluster_reps in representatives.items():
                print(f"   📋 Cluster {cluster_id}: {len(cluster_reps)} rappresentanti")
                
                for rep in cluster_reps:
                    session_id = rep['session_id']
                    conversation_text = rep['testo_completo']
                    
                    # Classificazione LLM-only
                    llm_result = self.ensemble_classifier.predict_with_llm_only(
                        conversation_text, 
                        return_details=True
                    )
                    
                    # 🔧 FORZARE CONFIDENZA 0.5 per primo avvio (tutto in review)
                    forced_confidence = 0.5
                    
                    # Prepara dati per training futuro
                    embeddings = self._get_embedder().encode([conversation_text])[0]
                    
                    training_data = {
                        'session_id': session_id,
                        'text': conversation_text,
                        'embeddings': embeddings,
                        'llm_label': llm_result['predicted_label'],
                        'llm_confidence': forced_confidence,  # 🔧 Forzata a 0.5
                        'cluster_id': cluster_id,
                        'is_representative': True,
                        'human_reviewed': False,  # Sarà aggiornato dopo review
                        'final_label': llm_result['predicted_label'],  # Inizialmente = LLM
                        'training_ready': True,
                        'source': 'llm_only_primo_avvio',
                        'needs_review': True  # 🔧 Tutto in review per primo avvio
                    }
                    
                    training_data_list.append(training_data)
                    
                    # Salva in MongoDB per review
                    success = mongo_reader.save_classification_result(
                        session_id=session_id,
                        client_name=self.tenant.tenant_slug,
                        final_decision={
                            'predicted_label': llm_result['predicted_label'],
                            'confidence': llm_result['confidence'],
                            'method': 'llm_only_primo_avvio_rappresentante'
                        },
                        conversation_text=conversation_text,
                        needs_review=True,  # Tutti i rappresentanti necessitano review
                        review_reason='primo_avvio_rappresentante',
                        classified_by='pipeline_llm_only',
                        cluster_metadata={
                            'cluster_id': cluster_id,
                            'is_representative': True,
                            'training_ready': True
                        }
                    )
                    
                    if success:
                        saved_count += 1
            
            # 2. CLASSIFICAZIONE LLM-ONLY DEGLI OUTLIER
            print(f"\n� FASE 2: CLASSIFICAZIONE LLM-ONLY OUTLIERS")
            outlier_sessions = []
            
            # Trova sessioni outlier dai cluster_labels
            session_ids = list(sessioni.keys())
            for i, session_id in enumerate(session_ids):
                if i < len(cluster_labels) and cluster_labels[i] == -1:
                    outlier_sessions.append(session_id)
            
            print(f"   📊 Outliers da classificare: {len(outlier_sessions)}")
            
            for session_id in outlier_sessions:
                conversation_text = sessioni[session_id]['testo_completo']
                
                # Classificazione LLM-only
                llm_result = self.ensemble_classifier.predict_with_llm_only(
                    conversation_text,
                    return_details=True
                )
                
                # 🔧 FORZARE CONFIDENZA 0.5 per primo avvio (tutto in review)
                forced_confidence = 0.5
                
                # Prepara dati per training futuro
                embeddings = self._get_embedder().encode([conversation_text])[0]
                
                training_data = {
                    'session_id': session_id,
                    'text': conversation_text,
                    'embeddings': embeddings,
                    'llm_label': llm_result['predicted_label'],
                    'llm_confidence': forced_confidence,  # 🔧 Forzata a 0.5
                    'cluster_id': -1,
                    'is_representative': False,
                    'human_reviewed': False,
                    'final_label': llm_result['predicted_label'],
                    'training_ready': True,
                    'source': 'llm_only_primo_avvio',
                    'needs_review': True  # 🔧 Tutto in review per primo avvio
                }
                
                training_data_list.append(training_data)
                
                # Salva in MongoDB per review
                success = mongo_reader.save_classification_result(
                    session_id=session_id,
                    client_name=self.tenant.tenant_slug,
                    final_decision={
                        'predicted_label': llm_result['predicted_label'],
                        'confidence': llm_result['confidence'],
                        'method': 'llm_only_primo_avvio_outlier'
                    },
                    conversation_text=conversation_text,
                    needs_review=True,  # Tutti gli outlier necessitano review
                    review_reason='primo_avvio_outlier',
                    classified_by='pipeline_llm_only',
                    cluster_metadata={
                        'cluster_id': -1,
                        'is_representative': False,
                        'training_ready': True
                    }
                )
                
                if success:
                    saved_count += 1
            
            elapsed_time = time.time() - start_time
            
            # Salva dati per training futuro come attributi della classe
            self._llm_training_data = training_data_list
            self._last_cluster_labels = cluster_labels
            self._last_representatives = representatives
            self._last_suggested_labels = suggested_labels
            
            # 🆕 FASE 1.5: Crea file training per correzioni umane future
            print(f"\n📁 FASE 1.5: CREAZIONE FILE TRAINING")
            training_file_path = self.create_ml_training_file(training_data_list)
            
            # Prepara risultato finale
            result = {
                'status': 'llm_only_success',
                'total_sessions': len(sessioni),
                'representatives_classified': total_representatives,
                'outliers_classified': len(outlier_sessions),
                'total_classified': len(training_data_list),
                'saved_for_training': len(training_data_list),
                'saved_for_review': saved_count,
                'training_data_ready': True,
                'training_file_path': training_file_path,  # 🆕 Path del file training
                'needs_human_review': True,
                'next_step': 'human_review_then_ml_training',
                'processing_time': elapsed_time
            }
            
            print(f"✅ [LLM-ONLY PRIMO AVVIO] Completata in {elapsed_time:.2f}s")
            print(f"📊 [LLM-ONLY PRIMO AVVIO] Risultati:")
            print(f"   🎯 Classificazioni totali: {result['total_classified']}")
            print(f"   👥 Rappresentanti: {result['representatives_classified']}")
            print(f"   🔍 Outlier: {result['outliers_classified']}")
            print(f"   💾 Dati training preparati: {result['saved_for_training']}")
            print(f"   📋 Salvati per review: {result['saved_for_review']}")
            print(f"   👤 Necessitano review umana: SÌ")
            print(f"   🎯 Prossimo step: Review umana → Training ML")
            
            trace_all("_classifica_llm_only_e_prepara_training", "EXIT", return_value=result)
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ [LLM-ONLY PRIMO AVVIO] ERRORE dopo {elapsed_time:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            
            trace_all("_classifica_llm_only_e_prepara_training", "ERROR", error=str(e))
            
            # Ritorna errore strutturato
            return {
                'status': 'error',
                'error': str(e),
                'training_data_ready': False,
                'processing_time': elapsed_time
            }

    def create_ml_training_file(self, training_data_list: List[Dict[str, Any]]) -> str:
        """
        Crea file per addestramento ML con identificatori univoci per aggiornamenti umani.
        
        FASE 1.5: Creazione file training per future correzioni umane
        
        Scopo della funzione: Creare file ML training con ID univoci per update
        Parametri di input: training_data_list con dati di classificazione
        Parametri di output: path del file di training creato
        Valori di ritorno: path del file training creato
        Tracciamento aggiornamenti: 2025-09-07 - Valerio Bignardi - Nuovo per flusso FASE 1.5
        
        FLUSSO:
        1. Crea file JSON con struttura training-ready
        2. Ogni record ha unique_id per permettere update da review umana
        3. Include conversation_text, predicted_label, confidence, cluster_info
        4. File sarà aggiornato quando utente corregge etichette
        
        Args:
            training_data_list: Lista di dict con dati classificazione
            
        Returns:
            str: Path del file di training creato
            
        Autore: Valerio Bignardi  
        Data: 2025-09-07
        """
        trace_all("create_ml_training_file", "ENTER", data_count=len(training_data_list))
        
        import json
        import os
        from datetime import datetime
        
        try:
            print(f"📁 [TRAINING FILE] Creazione file training per ML...")
            print(f"   📊 Records da salvare: {len(training_data_list)}")
            
            # Crea directory se non esiste
            training_dir = "training_files"
            os.makedirs(training_dir, exist_ok=True)
            
            # Nome file con timestamp e tenant
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tenant_slug = self.tenant.tenant_slug if self.tenant else "unknown"
            filename = f"ml_training_{tenant_slug}_{timestamp}.json"
            filepath = os.path.join(training_dir, filename)
            
            # Prepara struttura training file
            training_file_data = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'tenant_slug': tenant_slug,
                    'tenant_name': self.tenant.tenant_name if self.tenant else "Unknown",
                    'total_records': len(training_data_list),
                    'source': 'llm_only_primo_avvio',
                    'confidence_forced': 0.5,
                    'needs_human_review': True,
                    'version': '1.0'
                },
                'training_records': []
            }
            
            # Processa ogni record
            for i, data in enumerate(training_data_list):
                training_record = {
                    'unique_id': f"{tenant_slug}_{timestamp}_{i:06d}",  # ID univoco per update
                    'session_id': data.get('session_id'),
                    'conversation_text': data.get('text', ''),
                    'predicted_label': data.get('llm_label'),
                    'confidence': data.get('llm_confidence', 0.5),
                    'cluster_id': data.get('cluster_id', -1),
                    'is_representative': data.get('is_representative', False),
                    'human_reviewed': False,  # Sarà aggiornato dopo review
                    'human_corrected_label': None,  # Sarà popolato da review umana
                    'needs_review': data.get('needs_review', True),
                    'embeddings_included': True,  # Embeddings salvati separatamente
                    'created_at': datetime.now().isoformat(),
                    'last_updated': None
                }
                
                training_file_data['training_records'].append(training_record)
            
            # Salva file JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(training_file_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ File training creato: {filepath}")
            print(f"   📊 Records salvati: {len(training_file_data['training_records'])}")
            print(f"   🔑 ID univoci generati per future correzioni umane")
            
            trace_all("create_ml_training_file", "EXIT", filepath=filepath)
            return filepath
            
        except Exception as e:
            error_msg = f"Errore creazione file training: {str(e)}"
            print(f"❌ [TRAINING FILE] {error_msg}")
            trace_all("create_ml_training_file", "ERROR", error=error_msg)
            return None

    def update_training_file_with_human_corrections(self, 
                                                   training_file_path: str,
                                                   session_id: str, 
                                                   corrected_label: str) -> bool:
        """
        Aggiorna file training con correzioni umane dalla review.
        
        FASE 2.5: Update file training con correzioni umane
        
        Scopo della funzione: Aggiornare file training con correzioni dalla review umana
        Parametri di input: path file, session_id, etichetta corretta
        Parametri di output: success flag
        Valori di ritorno: True se aggiornamento riuscito
        Tracciamento aggiornamenti: 2025-09-07 - Valerio Bignardi - Update training file
        
        Args:
            training_file_path: Path del file training da aggiornare
            session_id: ID sessione da correggere
            corrected_label: Etichetta corretta dall'umano
            
        Returns:
            bool: True se aggiornamento riuscito
            
        Autore: Valerio Bignardi
        Data: 2025-09-07
        """
        trace_all("update_training_file_with_human_corrections", "ENTER",
                 file_path=training_file_path, session_id=session_id, 
                 corrected_label=corrected_label)
        
        import json
        from datetime import datetime
        
        try:
            print(f"🔄 [TRAINING UPDATE] Aggiornamento file con correzione umana...")
            print(f"   📁 File: {training_file_path}")
            print(f"   🆔 Session: {session_id}")
            print(f"   🏷️ Nuova label: {corrected_label}")
            
            # Carica file esistente
            if not os.path.exists(training_file_path):
                print(f"❌ [TRAINING UPDATE] File non trovato: {training_file_path}")
                return False
            
            with open(training_file_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            
            # Trova e aggiorna record
            updated = False
            for record in training_data['training_records']:
                if record['session_id'] == session_id:
                    record['human_corrected_label'] = corrected_label
                    record['human_reviewed'] = True
                    record['last_updated'] = datetime.now().isoformat()
                    updated = True
                    break
            
            if not updated:
                print(f"⚠️ [TRAINING UPDATE] Session {session_id} non trovata nel file")
                return False
            
            # Aggiorna metadata
            training_data['metadata']['last_human_update'] = datetime.now().isoformat()
            
            # Salva file aggiornato
            with open(training_file_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ File aggiornato con correzione umana")
            
            trace_all("update_training_file_with_human_corrections", "EXIT", 
                     updated=True)
            return True
            
        except Exception as e:
            error_msg = f"Errore aggiornamento file training: {str(e)}"
            print(f"❌ [TRAINING UPDATE] {error_msg}")
            trace_all("update_training_file_with_human_corrections", "ERROR", 
                     error=error_msg)
            return False

    def _get_mongo_reader(self):
        """
        Ottiene istanza MongoClassificationReader per salvataggio classificazioni.
        
        Scopo della funzione: Creare istanza MongoDB reader tenant-aware
        Parametri di input: None (usa self.tenant)
        Parametri di output: MongoClassificationReader configurato
        Valori di ritorno: Istanza MongoClassificationReader
        Tracciamento aggiornamenti: 2025-09-06 - Valerio Bignardi - Helper per MongoDB
        
        Returns:
            MongoClassificationReader: Istanza configurata per il tenant
            
        Autore: Valerio Bignardi
        Data: 2025-09-06
        """
        from mongo_classification_reader import MongoClassificationReader
        return MongoClassificationReader(tenant=self.tenant)

    def aggiungi_caso_come_esempio_llm(self, 
                                      session_id: str,
                                      conversation_text: str,
                                      etichetta_corretta: str,
                                      categoria: str = None,
                                      note_utente: str = None) -> Dict[str, Any]:
        """
        Aggiunge un caso di review umana come esempio LLM nel database MySQL.
        
        Scopo della funzione: Salvare conversazioni corrette dall'umano come esempi 
                              di training per migliorare le prestazioni LLM
        Parametri di input: session_id, conversation_text, etichetta_corretta, categoria, note_utente
        Parametri di output: risultato operazione con dettagli
        Valori di ritorno: Dict con success, esempio_id, messaggio
        Tracciamento aggiornamenti: 2025-09-07 - Valerio Bignardi - Aggiunto supporto note_utente
        
        Args:
            session_id: ID univoco della sessione 
            conversation_text: Testo completo della conversazione
            etichetta_corretta: Etichetta corretta assegnata dall'umano
            categoria: Categoria opzionale per raggruppamento (default: etichetta_corretta)
            note_utente: Note scritte dall'utente nell'interfaccia React per spiegare la decisione
            
        Returns:
            Dict con risultato dell'operazione:
            {
                'success': bool,
                'esempio_id': int,
                'message': str,
                'details': {...}
            }
            
        Autore: Valerio Bignardi
        Data: 2025-09-07
        """
        trace_all("aggiungi_caso_come_esempio_llm", "ENTER",
                 session_id=session_id,
                 etichetta_corretta=etichetta_corretta,
                 categoria=categoria,
                 note_utente=note_utente,
                 text_length=len(conversation_text))
        
        start_time = time.time()
        
        try:
            print(f"📚 [ESEMPIO LLM] Aggiunta caso come esempio...")
            print(f"   🆔 Session ID: {session_id}")
            print(f"   🏷️ Etichetta corretta: {etichetta_corretta}")
            print(f"   📝 Lunghezza testo: {len(conversation_text)} caratteri")
            if note_utente:
                print(f"   📋 Note utente: {note_utente[:100]}{'...' if len(note_utente) > 100 else ''}")
            
            # Validazione input
            if not session_id or not conversation_text or not etichetta_corretta:
                error_msg = "Parametri mancanti: session_id, conversation_text e etichetta_corretta sono obbligatori"
                print(f"❌ [ESEMPIO LLM] {error_msg}")
                return {
                    'success': False,
                    'esempio_id': None,
                    'message': error_msg,
                    'details': {'validation_error': True}
                }
            
            # Verifica tenant
            if not self.tenant:
                error_msg = "Tenant non configurato nella pipeline"
                print(f"❌ [ESEMPIO LLM] {error_msg}")
                return {
                    'success': False,
                    'esempio_id': None,
                    'message': error_msg,
                    'details': {'tenant_error': True}
                }
            
            # Usa categoria = etichetta se non specificata
            if not categoria:
                categoria = etichetta_corretta
            
            # Formatta il contenuto come conversazione UTENTE/ASSISTENTE
            esempio_content = f"UTENTE: {conversation_text}\nASSISTENTE: Classificazione: {etichetta_corretta}"
            
            # Genera nome esempio basato su timestamp e session_id
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            esempio_name = f"review_caso_{session_id}_{timestamp_str}"
            
            # Descrizione dettagliata con note utente se disponibili
            base_description = f"Esempio da review umana - Session: {session_id}, Corretto in: {etichetta_corretta}"
            if note_utente and note_utente.strip():
                description = f"{base_description}. Note: {note_utente.strip()}"
            else:
                description = base_description
            
            # Crea istanza PromptManager per accedere alla funzione create_example
            from Utils.prompt_manager import PromptManager
            prompt_manager = PromptManager(config_path=self.config_path)
            
            print(f"   📝 Creazione esempio nel database MySQL...")
            print(f"   📛 Nome esempio: {esempio_name}")
            print(f"   🗂️ Categoria: {categoria}")
            
            # Crea esempio nel database
            esempio_id = prompt_manager.create_example(
                tenant=self.tenant,
                esempio_name=esempio_name,
                esempio_content=esempio_content,
                engine='LLM',  # Esempio specifico per LLM
                esempio_type='CONVERSATION',  # Tipo conversazione
                description=description,
                categoria=categoria,
                livello_difficolta='MEDIO'  # Default medio
            )
            
            if esempio_id:
                elapsed_time = time.time() - start_time
                success_msg = f"Esempio LLM creato con successo: ID {esempio_id}"
                
                print(f"✅ [ESEMPIO LLM] {success_msg}")
                print(f"   ⏱️ Completato in {elapsed_time:.2f}s")
                print(f"   🆔 ID esempio: {esempio_id}")
                print(f"   📂 Salvato in database: TAG.esempi")
                
                result = {
                    'success': True,
                    'esempio_id': esempio_id,
                    'message': success_msg,
                    'details': {
                        'esempio_name': esempio_name,
                        'categoria': categoria,
                        'engine': 'LLM',
                        'esempio_type': 'CONVERSATION',
                        'processing_time': elapsed_time,
                        'tenant_id': self.tenant.tenant_id,
                        'tenant_name': self.tenant.tenant_name
                    }
                }
                
                trace_all("aggiungi_caso_come_esempio_llm", "EXIT", 
                         return_value=result)
                return result
                
            else:
                error_msg = "Errore nella creazione dell'esempio (esempio_id = None)"
                print(f"❌ [ESEMPIO LLM] {error_msg}")
                
                result = {
                    'success': False,
                    'esempio_id': None,
                    'message': error_msg,
                    'details': {'database_error': True}
                }
                
                trace_all("aggiungi_caso_come_esempio_llm", "ERROR", 
                         error_message=error_msg)
                return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"Errore aggiunta esempio LLM: {str(e)}"
            
            print(f"❌ [ESEMPIO LLM] {error_msg}")
            print(f"   ⏱️ Fallito dopo {elapsed_time:.2f}s")
            import traceback
            traceback.print_exc()
            
            result = {
                'success': False,
                'esempio_id': None,
                'message': error_msg,
                'details': {
                    'exception': str(e),
                    'processing_time': elapsed_time
                }
            }
            
            trace_all("aggiungi_caso_come_esempio_llm", "ERROR", 
                     error_message=error_msg, exception=str(e))
            return result

    def esegui_training_supervisionato_fase1(self, 
                                           sessioni: Dict[str, Dict],
                                           cluster_labels: np.ndarray,
                                           representatives: Dict[int, List[Dict]],
                                           suggested_labels: Dict[int, str]) -> Dict[str, Any]:
        """
        FASE 1: TRAINING SUPERVISIONATO - Preparazione + Classificazione Iniziale
        
        Scopo della funzione: Eseguire clustering, classificazione con ML+LLM e salvataggio
        Parametri di input: sessioni, cluster_labels, representatives, suggested_labels
        Parametri di output: risultato con statistiche di classificazione
        Valori di ritorno: Dict con success, total_classified, review_queue_populated
        Tracciamento aggiornamenti: 2025-09-07 - Valerio Bignardi - Nuovo flusso a 3 fasi
        
        FLUSSO:
        1. Estrai sessioni ✅ (già fatto prima di chiamare questo metodo)
        2. Esegui clustering ✅ (già fatto prima di chiamare questo metodo)  
        3. Classifica TUTTE le sessioni con ML+LLM:
           - Se ML modello disponibile → ensemble ML+LLM
           - Se ML modello NON disponibile → solo LLM (confidenza 0.5)
        4. Salva TUTTO in MongoDB ✅
        5. Review queue popolata automaticamente:
           - Rappresentanti con bassa confidenza → review umana
           - Outlier con bassa confidenza → review umana
           - Primo avvio → TUTTO ha confidenza 0.5 → TUTTO in review
        
        Args:
            sessioni: Dict sessioni estratte
            cluster_labels: Array cluster ID per ogni sessione
            representatives: Dict rappresentanti per cluster
            suggested_labels: Dict etichette suggerite per cluster
            
        Returns:
            Dict con risultato della fase 1:
            {
                'success': bool,
                'total_sessions': int,
                'total_classified': int,
                'review_queue_populated': int,
                'ml_model_available': bool,
                'classification_method': str,
                'cluster_stats': {...}
            }
            
        Autore: Valerio Bignardi
        Data: 2025-09-07
        """
        trace_all("esegui_training_supervisionato_fase1", "ENTER",
                 sessioni_count=len(sessioni),
                 cluster_labels_count=len(cluster_labels),
                 representatives_count=len(representatives),
                 suggested_labels_count=len(suggested_labels))
        
        start_time = time.time()
        
        try:
            print(f"🚀 [FASE 1 TRAINING] Avvio classificazione iniziale...")
            print(f"   📊 Sessioni da classificare: {len(sessioni)}")
            print(f"   🏷️ Cluster con etichette: {len(suggested_labels)}")
            print(f"   👥 Cluster con rappresentanti: {len(representatives)}")
            
            # Statistiche clustering
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_outliers = list(cluster_labels).count(-1)
            total_sessions = len(cluster_labels)
            
            print(f"   🎯 Cluster validi: {n_clusters}")
            print(f"   🔍 Outliers: {n_outliers}")
            print(f"   📈 Clusterizzate: {((total_sessions - n_outliers) / total_sessions * 100):.1f}%")
            
            # 3. CLASSIFICA TUTTE LE SESSIONI con ML+LLM
            print(f"\n🤖 [FASE 1] Classificazione di tutte le sessioni...")
            
            # Verifica disponibilità modello ML
            ml_model_available = self.ensemble_classifier.has_trained_ml_model() if hasattr(self.ensemble_classifier, 'has_trained_ml_model') else False
            
            if ml_model_available:
                classification_method = "ENSEMBLE_ML_LLM"
                base_confidence = 0.7  # Confidence più alta con ML disponibile
                print(f"   ✅ Modello ML disponibile - Usando ensemble ML+LLM")
            else:
                classification_method = "SOLO_LLM"
                base_confidence = 0.5  # Confidence default per primo avvio
                print(f"   ⚠️ Modello ML non disponibile - Usando solo LLM (primo avvio)")
            
            classifications = []
            review_queue_count = 0
            
            # Prepara lista sessioni per classificazione batch
            session_texts = []
            session_ids = []
            session_cluster_ids = []
            
            for i, (session_id, session_data) in enumerate(sessioni.items()):
                conversation_text = session_data.get('conversation_text', '')
                cluster_id = cluster_labels[i] if i < len(cluster_labels) else -1
                
                session_texts.append(conversation_text)
                session_ids.append(session_id)
                session_cluster_ids.append(cluster_id)
            
            print(f"   📝 Preparate {len(session_texts)} sessioni per classificazione batch")
            
            # Esegui classificazione batch
            if ml_model_available:
                # Ensemble ML+LLM
                print(f"   🔄 Esecuzione classificazione ensemble...")
                batch_results = self._classify_sessions_batch_ensemble(session_texts, session_cluster_ids, suggested_labels)
            else:
                # Solo LLM (primo avvio)
                print(f"   🔄 Esecuzione classificazione solo LLM...")
                batch_results = self._classify_sessions_batch_llm_only(session_texts, session_cluster_ids, suggested_labels, base_confidence)
            
            # 4. SALVA TUTTO IN MONGODB
            print(f"\n💾 [FASE 1] Salvataggio classificazioni in MongoDB...")
            
            mongo_reader = self._get_mongo_reader()
            
            for i, session_id in enumerate(session_ids):
                session_data = sessioni[session_id]
                cluster_id = session_cluster_ids[i]
                classification_result = batch_results[i]
                
                # Determina se va in review queue
                confidence = classification_result.get('confidence', base_confidence)
                classification = classification_result.get('classification', 'ALTRO')
                
                # Logica review queue:
                # - Primo avvio (ML non disponibile): TUTTO in review
                # - Rappresentanti con bassa confidenza: in review
                # - Outlier con bassa confidenza: in review
                needs_review = False
                if not ml_model_available:
                    needs_review = True  # Primo avvio - tutto in review
                elif cluster_id == -1:
                    needs_review = True  # Outlier sempre in review
                elif cluster_id in representatives and confidence < 0.7:
                    needs_review = True  # Rappresentanti con bassa confidenza
                
                if needs_review:
                    review_queue_count += 1
                
                # Prepara record per MongoDB
                classification_record = {
                    'session_id': session_id,
                    'conversation_text': session_data.get('conversation_text', ''),
                    'classification': classification,
                    'confidence': confidence,
                    'cluster_id': cluster_id,
                    'is_representative': cluster_id in representatives and session_id in [r['session_id'] for r in representatives[cluster_id]],
                    'is_outlier': cluster_id == -1,
                    'review_status': 'pending' if needs_review else 'auto_approved',
                    'classification_method': classification_method,
                    'timestamp': datetime.now(),
                    'tenant_id': self.tenant.tenant_id,
                    'tenant_name': self.tenant.tenant_name,
                    'metadata': {
                        'clustering_algorithm': 'hdbscan',
                        'embedding_model': getattr(self.embedder, 'model_name', 'unknown'),
                        'llm_model': getattr(self.ensemble_classifier, 'llm_model', 'unknown'),
                        'ml_model_available': ml_model_available
                    }
                }
                
                classifications.append(classification_record)
            
            # Salvataggio batch in MongoDB
            print(f"   📊 Salvando {len(classifications)} classificazioni...")
            try:
                # Usa il metodo di salvataggio esistente o implementa batch save
                for record in classifications:
                    mongo_reader.save_classification(record)
                
                print(f"   ✅ Salvate {len(classifications)} classificazioni in MongoDB")
                
            except Exception as save_error:
                print(f"   ❌ Errore salvataggio MongoDB: {save_error}")
                raise
            
            # 5. STATISTICHE FINALI
            elapsed_time = time.time() - start_time
            
            result = {
                'success': True,
                'total_sessions': total_sessions,
                'total_classified': len(classifications),
                'review_queue_populated': review_queue_count,
                'ml_model_available': ml_model_available,
                'classification_method': classification_method,
                'cluster_stats': {
                    'valid_clusters': n_clusters,
                    'outliers': n_outliers,
                    'cluster_percentage': ((total_sessions - n_outliers) / total_sessions * 100) if total_sessions > 0 else 0
                },
                'processing_time': elapsed_time
            }
            
            print(f"\n✅ [FASE 1] Completata con successo!")
            print(f"   📊 Sessioni classificate: {len(classifications)}")
            print(f"   📋 In review queue: {review_queue_count}")
            print(f"   🤖 Metodo: {classification_method}")
            print(f"   ⏱️ Tempo elaborazione: {elapsed_time:.2f}s")
            print(f"   🎯 Pronto per review umana (Fase 2)")
            
            trace_all("esegui_training_supervisionato_fase1", "EXIT", return_value=result)
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"Errore nella Fase 1 training supervisionato: {str(e)}"
            
            print(f"❌ [FASE 1] {error_msg}")
            print(f"   ⏱️ Fallito dopo {elapsed_time:.2f}s")
            import traceback
            traceback.print_exc()
            
            result = {
                'success': False,
                'error': error_msg,
                'total_sessions': 0,
                'total_classified': 0,
                'review_queue_populated': 0,
                'processing_time': elapsed_time
            }
            
            trace_all("esegui_training_supervisionato_fase1", "ERROR", 
                     error_message=error_msg, exception=str(e))
            return result

    def _classify_sessions_batch_ensemble(self, session_texts: List[str], 
                                        cluster_ids: List[int],
                                        suggested_labels: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Classifica sessioni in batch usando ensemble ML+LLM
        
        Scopo: Classificazione batch con modello ML disponibile
        Input: session_texts, cluster_ids, suggested_labels
        Output: Lista risultati classificazione
        Data ultima modifica: 2025-09-07
        """
        results = []
        
        print(f"   🔄 Classificazione ensemble per {len(session_texts)} sessioni...")
        
        # Usa il metodo predict_with_ensemble se disponibile
        try:
            for i, text in enumerate(session_texts):
                cluster_id = cluster_ids[i]
                
                # Usa suggested label come baseline se disponibile
                suggested_label = suggested_labels.get(cluster_id, 'ALTRO')
                
                # Classifica con ensemble
                ensemble_result = self.ensemble_classifier.predict_with_ensemble([text])
                
                if ensemble_result and len(ensemble_result) > 0:
                    classification = ensemble_result[0].get('classification', suggested_label)
                    confidence = ensemble_result[0].get('confidence', 0.7)
                else:
                    # Fallback al suggested label
                    classification = suggested_label
                    confidence = 0.6
                
                results.append({
                    'classification': classification,
                    'confidence': confidence,
                    'method': 'ensemble_ml_llm'
                })
                
        except Exception as e:
            print(f"   ⚠️ Errore ensemble, fallback a solo LLM: {e}")
            # Fallback a solo LLM
            return self._classify_sessions_batch_llm_only(session_texts, cluster_ids, suggested_labels, 0.6)
        
        return results

    def _classify_sessions_batch_llm_only(self, session_texts: List[str],
                                        cluster_ids: List[int], 
                                        suggested_labels: Dict[int, str],
                                        base_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Classifica sessioni in batch usando LLM con batch processing ottimizzato
        
        Scopo: Classificazione batch usando LLM reale con batch processing per OpenAI
        Input: session_texts, cluster_ids, suggested_labels, base_confidence
        Output: Lista risultati classificazione
        Data ultima modifica: 2025-01-31
        """
        results = []
        
        print(f"   🔄 Classificazione LLM ottimizzata per {len(session_texts)} sessioni...")
        
        # 🔧 NUOVO: Usa LLM reale con batch processing ottimizzato
        if (self.ensemble_classifier and 
            self.ensemble_classifier.llm_classifier and 
            hasattr(self.ensemble_classifier.llm_classifier, 'classify_multiple_conversations_optimized')):
            
            try:
                print(f"   🚀 Avvio classificazione LLM batch ottimizzata...")
                
                # Usa il nuovo metodo di batch processing ottimizzato
                llm_results = self.ensemble_classifier.llm_classifier.classify_multiple_conversations_optimized(
                    conversations=session_texts,
                    context=None
                )
                
                print(f"   ✅ Completata classificazione LLM: {len(llm_results)} risultati")
                
                # Converte risultati ClassificationResult in formato dict
                for i, llm_result in enumerate(llm_results):
                    cluster_id = cluster_ids[i] if i < len(cluster_ids) else None
                    
                    results.append({
                        'classification': llm_result.predicted_label,
                        'confidence': llm_result.confidence,
                        'method': f'llm_batch_{llm_result.method}',
                        'motivation': getattr(llm_result, 'motivation', ''),
                        'cluster_id': cluster_id,
                        'suggested_label': suggested_labels.get(cluster_id, 'ALTRO') if cluster_id else 'ALTRO'
                    })
                
                return results
                
            except Exception as e:
                print(f"   ❌ Errore classificazione LLM batch, fallback a metodo precedente: {e}")
                # Fallback al metodo precedente in caso di errore
        
        # 🔄 FALLBACK: Metodo precedente (suggested labels)
        print(f"   🔄 Fallback a suggested labels per {len(session_texts)} sessioni...")
        
        for i, text in enumerate(session_texts):
            cluster_id = cluster_ids[i]
            
            # Usa suggested label come classification di base
            suggested_label = suggested_labels.get(cluster_id, 'ALTRO')
            
            # Per primo avvio o fallback, usa sempre suggested label con confidence bassa
            classification = suggested_label
            confidence = base_confidence
            
            results.append({
                'classification': classification,
                'confidence': confidence,
                'method': 'llm_fallback_suggested_labels',
                'cluster_id': cluster_id,
                'suggested_label': suggested_label
            })
        
        return results

    def load_training_data_from_file(self, training_file_path: str = None) -> List[Dict[str, Any]]:
        """
        Carica dati di training dal file JSON per il riaddestramento ML.
        
        FASE 3: Caricamento dati training dal file per riaddestramento
        
        Scopo della funzione: Leggere file training JSON con correzioni umane
        Parametri di input: path del file (opzionale, cerca automaticamente)
        Parametri di output: lista dati training-ready
        Valori di ritorno: Lista di dict con dati corretti per training ML
        Tracciamento aggiornamenti: 2025-09-07 - Valerio Bignardi - FASE 3
        
        Args:
            training_file_path: Path specifico del file (se None, cerca l'ultimo)
            
        Returns:
            List[Dict]: Lista dati training con correzioni umane
            
        Autore: Valerio Bignardi
        Data: 2025-09-07
        """
        trace_all("load_training_data_from_file", "ENTER", 
                 training_file_path=training_file_path)
        
        import json
        import os
        import glob
        
        try:
            print(f"📂 [TRAINING LOAD] Caricamento dati training dal file...")
            
            # Se nessun path specifico, cerca l'ultimo file per questo tenant
            if not training_file_path:
                training_dir = "training_files"
                if not os.path.exists(training_dir):
                    print(f"❌ [TRAINING LOAD] Directory training_files non trovata")
                    return []
                
                tenant_slug = self.tenant.tenant_slug if self.tenant else "unknown"
                pattern = os.path.join(training_dir, f"ml_training_{tenant_slug}_*.json")
                files = glob.glob(pattern)
                
                if not files:
                    print(f"❌ [TRAINING LOAD] Nessun file training trovato per {tenant_slug}")
                    return []
                
                # Prendi il file più recente
                training_file_path = max(files, key=os.path.getctime)
                print(f"   📁 File automaticamente selezionato: {training_file_path}")
            
            # Carica file JSON
            if not os.path.exists(training_file_path):
                print(f"❌ [TRAINING LOAD] File non trovato: {training_file_path}")
                return []
            
            with open(training_file_path, 'r', encoding='utf-8') as f:
                training_file_data = json.load(f)
            
            # Estrai records di training
            training_records = training_file_data.get('training_records', [])
            
            # Filtra solo records pronti per training (con correzioni umane o conferme)
            ready_for_training = []
            human_corrected = 0
            llm_confirmed = 0
            
            for record in training_records:
                # Record è pronto se ha correzione umana O è stato confermato nella review
                if record.get('human_corrected_label'):
                    # Usa etichetta corretta dall'umano
                    training_record = {
                        'session_id': record['session_id'],
                        'conversation_text': record['conversation_text'],
                        'predicted_label': record['human_corrected_label'],  # Label corretta
                        'confidence': 1.0,  # Alta confidenza per correzioni umane
                        'cluster_id': record.get('cluster_id', -1),
                        'is_representative': record.get('is_representative', False),
                        'source': 'human_corrected'
                    }
                    ready_for_training.append(training_record)
                    human_corrected += 1
                    
                elif record.get('human_reviewed') and not record.get('needs_review', True):
                    # Record confermato come corretto (nessuna correzione necessaria)
                    training_record = {
                        'session_id': record['session_id'],
                        'conversation_text': record['conversation_text'],
                        'predicted_label': record['predicted_label'],  # Label originale confermata
                        'confidence': 0.8,  # Alta confidenza per conferme umane
                        'cluster_id': record.get('cluster_id', -1),
                        'is_representative': record.get('is_representative', False),
                        'source': 'human_confirmed'
                    }
                    ready_for_training.append(training_record)
                    llm_confirmed += 1
            
            print(f"   ✅ File caricato: {len(training_records)} records totali")
            print(f"   📊 Pronti per training: {len(ready_for_training)}")
            print(f"   🔧 Corretti dall'umano: {human_corrected}")
            print(f"   ✅ Confermati LLM: {llm_confirmed}")
            
            # Salva path del file per riferimenti futuri
            self._last_training_file_path = training_file_path
            
            trace_all("load_training_data_from_file", "EXIT", 
                     records_loaded=len(ready_for_training))
            return ready_for_training
            
        except Exception as e:
            error_msg = f"Errore caricamento file training: {str(e)}"
            print(f"❌ [TRAINING LOAD] {error_msg}")
            trace_all("load_training_data_from_file", "ERROR", error=error_msg)
            return []

    def manual_retrain_model(self, force: bool = False) -> Dict[str, Any]:
        """
        Riaddestra il modello ML usando i dati corretti dalla review umana.
        CHIAMATO SOLO quando l'utente preme "RIADDESTRA MODELLO" dall'interfaccia React.
        
        Scopo della funzione: Riaddestramento ML usando pipeline esistente
        Parametri di input: force (ignora controlli di sicurezza se True)
        Parametri di output: risultato del riaddestramento con metriche
        Valori di ritorno: Dict con success, accuracy, dettagli
        Tracciamento aggiornamenti: 2025-09-07 - Valerio Bignardi - Usa pipeline esistente
        
        Args:
            force: Se True, ignora controlli di sicurezza e forza riaddestramento
            
        Returns:
            Dict con risultato del riaddestramento:
            {
                'success': bool,
                'accuracy': float,
                'message': str,
                'training_stats': {...}
            }
            
        Autore: Valerio Bignardi
        Data: 2025-09-07
        """
        trace_all("manual_retrain_model", "ENTER", force=force)
        
        start_time = time.time()
        
        try:
            print(f"🔄 [RIADDESTRAMENTO MANUALE] Avvio riaddestramento modello ML...")
            print(f"   🏢 Tenant: {self.tenant.tenant_name if self.tenant else 'N/A'}")
            print(f"   ⚡ Force mode: {force}")
            
            # 1. Verifica prerequisiti
            if not self.tenant:
                error_msg = "Tenant non configurato"
                return {
                    'success': False,
                    'accuracy': 0.0,
                    'message': error_msg,
                    'training_stats': {'error': 'no_tenant'}
                }
            
            # 2. Carica dati di training corretti dal file di training
            print(f"   📚 Caricamento dati di training dal file JSON...")
            
            training_data = self.load_training_data_from_file()
            
            if not training_data or len(training_data) < 5:
                min_required = 5
                found = len(training_data) if training_data else 0
                
                if not force:
                    error_msg = f"Dati insufficienti per training: {found}/{min_required} richiesti"
                    print(f"❌ [RIADDESTRAMENTO] {error_msg}")
                    return {
                        'success': False,
                        'accuracy': 0.0,
                        'message': error_msg,
                        'training_stats': {
                            'data_found': found,
                            'minimum_required': min_required,
                            'insufficient_data': True
                        }
                    }
                else:
                    print(f"⚠️ [RIADDESTRAMENTO] Force mode: procedo con {found} campioni")
            
            print(f"   ✅ Caricati {len(training_data)} esempi di training dal file")
            
            # 3. Prepara features e labels dal file JSON
            session_texts = [data['conversation_text'] for data in training_data]
            # Usa label corretta dall'umano se disponibile, altrimenti label originale
            train_labels = [
                data.get('human_corrected_label') or data.get('predicted_label') 
                for data in training_data
            ]
            
            # Genera embeddings
            print(f"   🧠 Generazione embeddings...")
            embedder = self._get_embedder()
            train_embeddings = embedder.encode(session_texts)
            
            # 4. Riaddestra ensemble ML
            print(f"   🎓 Riaddestramento ensemble ML...")
            print(f"     📊 Samples: {len(train_labels)}")
            print(f"     🏷️ Unique labels: {len(set(train_labels))}")
            
            # Usa BERTopic se disponibile per feature augmentation
            ml_features = train_embeddings
            bertopic_provider = getattr(self, '_bertopic_provider_trained', None)
            
            if bertopic_provider is not None:
                try:
                    print(f"     🤖 Applicazione feature augmentation BERTopic...")
                    tr = bertopic_provider.transform(
                        session_texts,
                        embeddings=train_embeddings,
                        return_one_hot=self.bertopic_config.get('return_one_hot', False),
                        top_k=self.bertopic_config.get('top_k', None)
                    )
                    
                    # Concatena features aggiuntive
                    parts = [train_embeddings]
                    if tr.get('topic_probas') is not None:
                        parts.append(tr['topic_probas'])
                    if tr.get('one_hot') is not None:
                        parts.append(tr['one_hot'])
                    
                    ml_features = np.concatenate(parts, axis=1)
                    print(f"     ✅ Features augmented: {train_embeddings.shape[1]} → {ml_features.shape[1]}")
                    
                except Exception as e:
                    print(f"     ⚠️ Errore BERTopic augmentation: {e}")
                    print(f"     🔄 Proseguo con sole embeddings")
            
            # Training effettivo
            training_metrics = self.ensemble_classifier.train_ml_ensemble(
                ml_features, 
                np.array(train_labels)
            )
            
            if not training_metrics or not training_metrics.get('accuracy'):
                error_msg = "Training ML fallito - nessuna metrica restituita"
                print(f"❌ [RIADDESTRAMENTO] {error_msg}")
                return {
                    'success': False,
                    'accuracy': 0.0,
                    'message': error_msg,
                    'training_stats': {'training_failed': True}
                }
            
            # 5. Salva modello aggiornato
            print(f"   💾 Salvataggio modello aggiornato...")
            model_name = f"retrained_{self.tenant.tenant_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = f"models/{model_name}"
            
            try:
                self.ensemble_classifier.save_ensemble_model(model_path)
                print(f"   ✅ Modello salvato: {model_path}")
            except Exception as save_error:
                print(f"   ⚠️ Errore salvataggio modello: {save_error}")
                # Non è fatale, il modello è in memoria
            
            # 6. Risultati finali
            elapsed_time = time.time() - start_time
            accuracy = training_metrics.get('accuracy', 0.0)
            
            success_msg = f"Riaddestramento completato - Accuracy: {accuracy:.3f}"
            
            print(f"✅ [RIADDESTRAMENTO MANUALE] {success_msg}")
            print(f"   ⏱️ Completato in {elapsed_time:.2f}s")
            print(f"   📚 Training samples: {len(training_data)}")
            print(f"   🏷️ Classi uniche: {len(set(train_labels))}")
            
            result = {
                'success': True,
                'accuracy': accuracy,
                'message': success_msg,
                'training_stats': {
                    **training_metrics,
                    'training_samples': len(training_data),
                    'unique_classes': len(set(train_labels)),
                    'processing_time': elapsed_time,
                    'force_mode': force
                }
            }
            
            trace_all("manual_retrain_model", "EXIT", return_value=result)
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"Errore durante riaddestramento: {str(e)}"
            
            print(f"❌ [RIADDESTRAMENTO MANUALE] {error_msg}")
            print(f"   ⏱️ Fallito dopo {elapsed_time:.2f}s")
            import traceback
            traceback.print_exc()
            
            result = {
                'success': False,
                'accuracy': 0.0,
                'message': error_msg,
                'training_stats': {
                    'exception': str(e),
                    'processing_time': elapsed_time,
                    'force_mode': force
                }
            }
            
            trace_all("manual_retrain_model", "ERROR", 
                     error_message=error_msg, exception=str(e))
            return result
