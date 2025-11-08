#!/usr/bin/env python3
"""
File: clustering_test_service.py
Autore: Sistema di Classificazione
Data: 2025-08-25

Descrizione: Servizio per test e validazione algoritmi di clustering HDBSCAN
Supporta test con campioni configurabili e genera visualizzazioni interattive

Storia aggiornamenti:
- 2025-08-25: Creazione servizio test clustering
- 2025-08-26: Aggiunto supporto GPU clustering e parametri avanzati
- 2025-08-26: Aggiunto sistema suggerimenti ottimizzazione outliers
"""

import os
import sys
import time
import yaml
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Add project root to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__)))  # Add Clustering directory

from hdbscan_clusterer import HDBSCANClusterer
from hdbscan_tuning_guide import HDBSCANTuningGuide

# Importa database risultati clustering per salvataggio automatico
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Database'))
from clustering_results_db import ClusteringResultsDB

# Importazione classe Tenant per eliminare conversioni ridondanti
try:
    from Utils.tenant import Tenant
    TENANT_AVAILABLE = True
except ImportError:
    TENANT_AVAILABLE = False
    print("⚠️ CLUSTERING TEST: Classe Tenant non disponibile, uso retrocompatibilità")

import sys
import os
import yaml
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import numpy as np
import logging
import gc

# Importazioni per pulizia memoria GPU
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Aggiunta percorsi per moduli
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Pipeline'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EmbeddingEngine'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Clustering'))

from end_to_end_pipeline import EndToEndPipeline
from hdbscan_clusterer import HDBSCANClusterer

# Import config_loader per caricare config.yaml con variabili ambiente
from config_loader import load_config



class ClusteringTestService:
    """
    Servizio per test rapidi di clustering HDBSCAN senza coinvolgimento LLM
    
    Scopo: Permettere agli utenti di testare rapidamente parametri di clustering
    utilizzando la pipeline esistente senza dover eseguire l'intera pipeline di classificazione.
    
    Args:
        config_path: Percorso del file di configurazione dei tenant
        
    Methods:
        run_clustering_test: Esegue test clustering e restituisce metriche di qualità
        
    Data ultima modifica: 2025-08-25
    """
    
    def __init__(self, config_path: str = None):
        """
        Inizializza il servizio di test clustering utilizzando la pipeline esistente
        
        Args:
            config_path: Percorso del file config.yaml (opzionale)
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'config.yaml'
        )
        self.min_conversations_required = 50
        self._setup_logging()
        
        # Pipeline cache per evitare reinizializzazioni multiple
        self.pipeline_cache = {}
        
        # Inizializza database per salvataggio risultati clustering
        try:
            self.results_db = ClusteringResultsDB(config_path)
            if hasattr(self, 'logger'):
                self.logger.info("✅ Database risultati clustering inizializzato")
            else:
                print("✅ Database risultati clustering inizializzato")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ Errore inizializzazione database risultati: {e}")
            else:
                print(f"❌ Errore inizializzazione database risultati: {e}")
            self.results_db = None
        
    def _setup_logging(self):
        """
        Configura logging per il servizio
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _get_pipeline(self, tenant: 'Tenant') -> EndToEndPipeline:
        """
        Ottiene pipeline per il tenant specificato con embedder dinamico configurato
        
        AGGIORNAMENTO 2025-08-31: Solo oggetto Tenant centralizzato - NESSUNA retrocompatibilità
        
        Args:
            tenant: Oggetto Tenant OBBLIGATORIO
            
        Returns:
            Istanza EndToEndPipeline configurata per tenant
            
        Autore: Valerio Bignardi
        Data: 2025-08-31
        Ultimo aggiornamento: 2025-08-31 - Eliminata retrocompatibilità stringhe
        """
        if not tenant or not hasattr(tenant, 'tenant_id'):
            raise ValueError("❌ ERRORE: Deve essere passato un oggetto Tenant valido, non stringa!")
        
        # Usa direttamente i dati del tenant
        tenant_id = tenant.tenant_id
        tenant_slug = tenant.tenant_slug
        tenant_display = f"{tenant.tenant_name} ({tenant_id})"
            
        if tenant_id not in self.pipeline_cache:
            try:
                # 🔧 CREA OGGETTO TENANT per architettura moderna
                if TENANT_AVAILABLE:
                    # Crea oggetto Tenant da UUID
                    tenant_obj = Tenant.from_uuid(tenant_id)
                    tenant_slug = tenant_obj.tenant_slug
                    print(f"🔄 Risoluzione tenant: UUID {tenant_id} -> slug '{tenant_slug}' -> Tenant object")
                else:
                    # Fallback: risolvi solo UUID -> slug 
                    tenant_obj = None
                    tenant_slug = self._resolve_tenant_slug_from_uuid(tenant_id)
                    print(f"🔄 Risoluzione tenant: UUID {tenant_id} -> slug '{tenant_slug}' (no Tenant obj)")
                
                # NUOVO SISTEMA: SimpleEmbeddingManager con reset automatico
                from EmbeddingEngine.simple_embedding_manager import simple_embedding_manager
                
                # Usa oggetto Tenant se disponibile, altrimenti fallback
                if TENANT_AVAILABLE and tenant_obj:
                    shared_embedder = simple_embedding_manager.get_embedder_for_tenant(tenant_obj)
                else:
                    # Fallback - crea oggetto Tenant minimo o errore
                    try:
                        temp_tenant = Tenant.from_uuid(tenant_id)
                        shared_embedder = simple_embedding_manager.get_embedder_for_tenant(temp_tenant)
                    except:
                        raise ValueError(f"Impossibile creare embedder per tenant {tenant_id} - oggetto Tenant richiesto")
                
                # 🆕 ARCHITETTURA MODERNA: passa oggetto Tenant completo
                if TENANT_AVAILABLE and tenant_obj:
                    pipeline = EndToEndPipeline(
                        tenant=tenant_obj,  # 🆕 PASSA OGGETTO TENANT COMPLETO
                        confidence_threshold=0.7,
                        auto_mode=True,
                        shared_embedder=shared_embedder  # Passa embedder con UUID
                    )
                else:
                    # Fallback retrocompatibilità
                    pipeline = EndToEndPipeline(
                        tenant_slug=tenant_slug,  # 🔧 FIX: passa SLUG non UUID!
                        confidence_threshold=0.7,
                        auto_mode=True,
                        shared_embedder=shared_embedder  # Passa embedder con UUID
                    )
                self.pipeline_cache[tenant_id] = pipeline  # Cache con UUID come key
                logging.info(f"✅ Pipeline {tenant_id} inizializzata con slug '{tenant_slug}' e cached")
            except Exception as e:
                logging.error(f"❌ Errore inizializzazione pipeline per UUID {tenant_id}: {e}")
                raise RuntimeError(f"Impossibile inizializzare pipeline per UUID {tenant_id}: {e}")
        
        # 🆕 VERIFICA VALIDITÀ EMBEDDER E AUTO-RELOAD SE NECESSARIO
        pipeline = self.pipeline_cache[tenant_id]
        
        try:
            # Verifica che l'embedder sia valido e abbia il modello caricato
            if hasattr(pipeline, 'embedder') and pipeline.embedder:
                # Controlla se il modello è presente e valido
                if not hasattr(pipeline.embedder, 'model') or pipeline.embedder.model is None:
                    print(f"⚠️  Embedder per {tenant_id} non ha modello valido, ricarico...")
                    
                    # Ricarica il modello embedder
                    if hasattr(pipeline.embedder, 'load_model'):
                        pipeline.embedder.load_model()
                        print(f"✅ Modello embedder ricaricato per {tenant_id}")
                    else:
                        # Se non ha load_model, ottieni un nuovo embedder
                        from EmbeddingEngine.simple_embedding_manager import simple_embedding_manager
                        # Converte tenant_id in oggetto Tenant
                        tenant_obj = Tenant.from_uuid(tenant_id)
                        new_embedder = simple_embedding_manager.get_embedder_for_tenant(tenant_obj)
                        pipeline.embedder = new_embedder
                        print(f"✅ Nuovo embedder ottenuto per {tenant_id}")
                        
                # Test rapido per verificare che l'embedder funzioni
                try:
                    test_embedding = pipeline.embedder.encode(["test"], show_progress_bar=False)
                    print(f"✅ Embedder per {tenant_id} verificato e funzionante")
                except Exception as test_error:
                    print(f"❌ Test embedder fallito per {tenant_id}: {test_error}")
                    # Forza ricaricamento completo
                    from EmbeddingEngine.simple_embedding_manager import simple_embedding_manager
                    # Converte tenant_id in oggetto Tenant
                    tenant_obj = Tenant.from_uuid(tenant_id)
                    pipeline.embedder = simple_embedding_manager.get_embedder_for_tenant(tenant_obj)
                    print(f"🔄 Embedder sostituito completamente per {tenant_id}")
            
        except Exception as embedder_check_error:
            print(f"⚠️  Errore verifica embedder per {tenant_id}: {embedder_check_error}")
            # In caso di errore, continua con la pipeline esistente (meglio che fallire)
        
        return pipeline
    
    def _resolve_tenant_slug_from_uuid(self, tenant_uuid: str) -> str:
        """
        Risolve tenant UUID in slug usando database
        
        Args:
            tenant_uuid: UUID del tenant
            
        Returns:
            Slug del tenant o UUID se non trovato
        """
        try:
            from TagDatabase.tag_database_connector import TagDatabaseConnector
            
            tag_connector = TagDatabaseConnector.create_for_tenant_resolution()
            tag_connector.connetti()
            
            query = "SELECT tenant_slug FROM tenants WHERE tenant_id = %s"
            result = tag_connector.esegui_query(query, (tenant_uuid,))
            
            if result and len(result) > 0:
                tenant_slug = result[0][0]
                tag_connector.disconnetti()
                return tenant_slug
            
            tag_connector.disconnetti()
            return tenant_uuid  # fallback
        except Exception as e:
            print(f"⚠️ Errore risoluzione tenant slug per UUID {tenant_uuid}: {e}")
            return tenant_uuid

    def _resolve_uuid_from_slug(self, tenant_slug: str) -> str:
        """
        Risolve tenant slug in UUID usando database TAG LOCALE
        
        Args:
            tenant_slug: Slug del tenant (es: 'golvalerio')
            
        Returns:
            UUID del tenant o errore se non trovato
        """
        try:
            from TagDatabase.tag_database_connector import TagDatabaseConnector
            
            tag_connector = TagDatabaseConnector.create_for_tenant_resolution()
            tag_connector.connetti()
            
            query = """
            SELECT tenant_id, tenant_name 
            FROM tenants 
            WHERE tenant_slug = %s AND is_active = 1
            """
            
            result = tag_connector.esegui_query(query, (tenant_slug,))
            tag_connector.disconnetti()
            
            if result and len(result) > 0:
                tenant_uuid, tenant_name = result[0]
                print(f"✅ UUID risolto da slug (DB TAG locale): {tenant_name} ({tenant_slug}) -> {tenant_uuid}")
                return tenant_uuid
            else:
                error_msg = f"❌ ERRORE: Tenant slug '{tenant_slug}' non trovato nel database TAG locale"
                print(error_msg)
                raise ValueError(error_msg)
                
        except Exception as e:
            error_msg = f"❌ ERRORE risoluzione UUID per slug '{tenant_slug}': {e}"
            print(error_msg)
            raise RuntimeError(f"Impossibile risolvere tenant slug '{tenant_slug}': {e}")
    
    def load_tenant_clustering_config(self, tenant_id: str) -> Dict[str, Any]:
        """
        Carica la configurazione clustering specifica del tenant
        
        Args:
            tenant_id: ID o slug del tenant
            
        Returns:
            Dizionario con parametri clustering del tenant
        """
        try:
            # Cerca configurazione tenant-specifica
            tenant_config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tenant_configs')
            
            # 🔧 [FIX] Prova prima con UUID del tenant se sembra essere un UUID
            potential_uuid_file = None
            if '-' in tenant_id and len(tenant_id) == 36:  # Formato UUID
                potential_uuid_file = os.path.join(tenant_config_dir, f'{tenant_id}_clustering.yaml')
                print(f"📁 [DEBUG] Tentativo caricamento config UUID: {potential_uuid_file}")
                
                if os.path.exists(potential_uuid_file):
                    print(f"✅ [DEBUG] Trovato file config UUID: {potential_uuid_file}")
                    with open(potential_uuid_file, 'r', encoding='utf-8') as file:
                        tenant_config = load_config()
                        config_params = tenant_config.get('clustering_parameters', {})
                        print(f"📋 [DEBUG] Parametri caricati da UUID: {list(config_params.keys())}")
                        print(f"�️  [DEBUG] UMAP nel config UUID: use_umap = {config_params.get('use_umap', 'NON_TROVATO')}")
                        return config_params
                else:
                    print(f"❌ [DEBUG] File config UUID non trovato: {potential_uuid_file}")
            
            # Fallback: slug del tenant
            tenant_slug_file = os.path.join(tenant_config_dir, f'{tenant_id}_clustering.yaml')
            print(f"📁 [DEBUG] Fallback a config slug: {tenant_slug_file}")
            
            if os.path.exists(tenant_slug_file):
                print(f"✅ [DEBUG] Trovato file config slug: {tenant_slug_file}")
                with open(tenant_slug_file, 'r', encoding='utf-8') as file:
                    tenant_config = load_config()
                    config_params = tenant_config.get('clustering_parameters', {})
                    print(f"📋 [DEBUG] Parametri caricati da slug: {list(config_params.keys())}")
                    print(f"🗂️  [DEBUG] UMAP nel config slug: use_umap = {config_params.get('use_umap', 'NON_TROVATO')}")
                    return config_params
            
            # Fallback: configurazione globale
            print(f"📁 [DEBUG] Fallback a config globale per tenant {tenant_id}")
            config = load_config()
            global_params = config.get('clustering', {})
            print(f"📋 [DEBUG] Parametri globali: {list(global_params.keys())}")
            return global_params
                
        except Exception as e:
            print(f"⚠️ Errore caricamento config clustering: {e}")
            # Parametri di default se tutto fallisce
            return {
                'min_cluster_size': 3,
                'min_samples': 2,
                'cluster_selection_epsilon': 0.15,
                'metric': 'cosine'
            }

    def get_sample_conversations(self, tenant: 'Tenant', limit: int = 1000) -> Dict[str, Any]:
        """
        Recupera un campione di conversazioni dal database per il tenant utilizzando pipeline
        
        Scopo: Estrarre conversazioni per il test di clustering
        Parametri di input:
            - tenant: Oggetto Tenant OBBLIGATORIO
            - limit (int): Numero massimo di conversazioni da recuperare
        Valori di ritorno:
            - Dict[str, Any]: Dizionario con sessioni per il clustering
        Data ultima modifica: 2025-08-31 - Eliminata retrocompatibilità
        
        Autore: Valerio Bignardi
        Data: 2025-08-31
        Ultimo aggiornamento: 2025-08-31
        """
        try:
            if not tenant or not hasattr(tenant, 'tenant_id'):
                raise ValueError("❌ ERRORE: Deve essere passato un oggetto Tenant valido!")
            
            # Definisci tenant_display subito per evitare errori nell'except
            tenant_display = f"{tenant.tenant_name}" if hasattr(tenant, 'tenant_name') else str(tenant)
            logging.info(f"📊 Estrazione {limit} conversazioni per tenant '{tenant_display}'")
            pipeline = self._get_pipeline(tenant)  # Passa oggetto Tenant
            
            # Estrai sessioni usando la funzione della pipeline
            sessioni = pipeline.estrai_sessioni(limit=limit)
            
            if not sessioni:
                logging.warning(f"❌ Nessuna sessione trovata per tenant '{tenant_display}'")
                return {}
            
            logging.info(f"✅ Estratte {len(sessioni)} sessioni valide per '{tenant_display}'")
            return sessioni
            
        except Exception as e:
            # tenant_display è ora sempre definita
            logging.error(f"❌ Errore estrazione conversazioni per {tenant_display}: {e}")
            return {}
    
    def get_parameter_suggestions(self) -> Dict[str, Any]:
        """
        Ottieni descrizioni parametri e preset ottimizzati
        
        Returns:
            Dizionario con descrizioni e preset
        """
        return {
            'parameter_descriptions': HDBSCANTuningGuide.get_parameter_descriptions(),
            'quick_fix_presets': HDBSCANTuningGuide.get_quick_fix_presets(),
            'current_defaults': {
                'min_cluster_size': 15,
                'min_samples': 3,
                'cluster_selection_epsilon': 0.08,
                'metric': 'cosine',
                'cluster_selection_method': 'eom',
                'alpha': 1.0,
                'max_cluster_size': 0,
                'allow_single_cluster': False
            }
        }
    
    def _cleanup_gpu_memory(self):
        """
        Pulisce la memoria GPU per evitare accumulo di memoria tra test consecutivi
        
        Scopo: Libera la memoria GPU utilizzata dai modelli per evitare errori CUDA OOM
        
        Data ultima modifica: 2025-08-25
        """
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # Forza la pulizia della cache CUDA
                torch.cuda.empty_cache()
                
                # Garbage collection per liberare oggetti Python
                gc.collect()
                
                # Log memoria GPU libera dopo pulizia
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                    memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
                    print(f"🧹 Memoria GPU pulita - Allocata: {memory_allocated:.2f}GB, Riservata: {memory_reserved:.2f}GB")
                    
        except Exception as e:
            print(f"⚠️ Errore durante pulizia memoria GPU: {str(e)}")

    def cleanup_all_pipelines(self):
        """
        Pulisce tutte le pipeline dalla cache preservando gli embedder
        
        Scopo:
        Libera memoria da tutte le pipeline cache senza distruggere embedder condivisi
        
        AGGIORNAMENTO 2025-08-26: Preservazione embedder per stabilità
        
        Data ultima modifica: 2025-08-26
        """
        try:
            # Libera solo risorse non-critiche delle pipeline
            for tenant_slug, pipeline in self.pipeline_cache.items():
                try:
                    # Pulizia risorse pipeline senza toccare l'embedder
                    if hasattr(pipeline, 'classification_service'):
                        pipeline.classification_service = None
                        
                    if hasattr(pipeline, 'clustering_service'):
                        pipeline.clustering_service = None
                        
                    # ✅ NON TOCCARE L'EMBEDDER - è gestito da SimpleEmbeddingManager
                    print(f"🧹 Pipeline {tenant_slug} pulita (embedder preservato)")
                except Exception as e:
                    print(f"⚠️ Errore pulizia pipeline per {tenant_slug}: {e}")
            
            # Pulisce completamente la cache
            self.pipeline_cache.clear()
            print(f"🧹 Cache pipeline pulita completamente")
            
            # Garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"⚠️ Errore durante pulizia cache pipeline: {str(e)}")

    def clear_pipeline_for_tenant(self, tenant_id: str):
        """
        Pulisce la pipeline specifica per un tenant dalla cache
        
        Scopo:
        Rimuove la pipeline cachata SENZA distruggere l'embedder condiviso
        
        AGGIORNAMENTO 2025-08-26: Preservazione embedder per evitare errori
        
        Args:
            tenant_id: UUID del tenant da rimuovere dalla cache
            
        Data ultima modifica: 2025-08-26
        """
        try:
            if tenant_id in self.pipeline_cache:
                pipeline = self.pipeline_cache[tenant_id]
                
                # 🔧 NON DISTRUGGERE L'EMBEDDER - è gestito da SimpleEmbeddingManager
                # OLD PROBLEMATIC CODE:
                # pipeline.embedder.model = None  ← CAUSA IL BUG
                
                # Pulizia solo risorse non-critiche
                try:
                    if hasattr(pipeline, 'classification_service'):
                        pipeline.classification_service = None
                        
                    if hasattr(pipeline, 'clustering_service'):
                        pipeline.clustering_service = None
                        
                    # ✅ MANTIENI L'EMBEDDER INTATTO - è condiviso tra operazioni
                    print(f"🧹 Pipeline per tenant {tenant_id} pulita (embedder preservato)")
                except Exception as e:
                    print(f"⚠️ Errore cleanup pipeline: {e}")
                
                # Rimuove dalla cache
                del self.pipeline_cache[tenant_id]
                print(f"✅ Pipeline per tenant {tenant_id} rimossa dalla cache")
                
                # Garbage collection localizzato
                gc.collect()
                
            else:
                print(f"ℹ️ Nessuna pipeline cachata per tenant {tenant_id}")
                
        except Exception as e:
            print(f"❌ Errore pulizia pipeline per tenant {tenant_id}: {e}")

    def run_clustering_test(self, 
                          tenant,  # 🎯 OGGETTO TENANT COMPLETO
                          custom_parameters: Optional[Dict] = None,
                          sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Esegue test clustering completo utilizzando la pipeline esistente
        
        Args:
            tenant: Oggetto tenant completo (Utils.tenant.Tenant) 
            custom_parameters: Parametri clustering personalizzati (opzionale)
            sample_size: Dimensione campione (opzionale, default 100)
            
        Returns:
            Risultati completi del test clustering
            
        Autore: Valerio Bignardi
        Data: 2025-08-31
        Ultimo aggiornamento: 2025-08-31 - Refactoring per oggetto tenant
        """
        start_time = time.time()
        print(f"🚀 Avvio test clustering per tenant {tenant.tenant_slug} ({tenant.tenant_name})...")
        
        if sample_size is None:
            sample_size = 100
        
        try:
            # 1. Carica configurazione clustering usando l'oggetto tenant
            base_clustering_config = self.load_tenant_clustering_config(tenant.tenant_id)
            print(f"📋 [DEBUG] Configurazione base tenant {tenant.tenant_slug}: {base_clustering_config}")
            
            # 2. Usa direttamente lo slug dal tenant (niente risoluzione!)
            print(f"🎯 [DEBUG] Tenant: {tenant.tenant_slug} ({tenant.tenant_name}) - ID: {tenant.tenant_id}")
            
            if custom_parameters:
                # Unisci parametri personalizzati con configurazione tenant
                clustering_config = base_clustering_config.copy()
                clustering_config.update(custom_parameters)
                print(f"🎛️ [DEBUG] Parametri personalizzati ricevuti: {custom_parameters}")
                print(f"🔧 [DEBUG] Configurazione finale (base + custom): {clustering_config}")
                print(f"🗂️  [DEBUG] UMAP preservato: use_umap = {clustering_config.get('use_umap', 'NON_TROVATO')}")
            else:
                clustering_config = base_clustering_config
                print(f"🎛️ Uso parametri tenant: {clustering_config}")
            
            # 3. Recupera conversazioni campione usando l'oggetto tenant completo
            sessioni = self.get_sample_conversations(tenant, sample_size)
            
            if len(sessioni) < self.min_conversations_required:
                return {
                    'success': False,
                    'error': f'Troppe poche conversazioni trovate ({len(sessioni)}). Minimo richiesto: {self.min_conversations_required}',
                    'tenant_id': tenant.tenant_id,
                    'tenant_slug': tenant.tenant_slug,
                    'execution_time': time.time() - start_time
                }
            
            # 3. Prepara testi per embedding
            texts = []
            session_ids = []
            for session_id, session_data in sessioni.items():
                if 'testo_completo' in session_data and session_data['testo_completo']:
                    texts.append(session_data['testo_completo'])
                    session_ids.append(session_id)
            
            if len(texts) < self.min_conversations_required:
                return {
                    'success': False,
                    'error': f'Troppe poche conversazioni valide trovate ({len(texts)}). Minimo richiesto: {self.min_conversations_required}',
                    'tenant_id': tenant.tenant_id,
                    'tenant_slug': tenant.tenant_slug,
                    'execution_time': time.time() - start_time
                }
            
            # 4. Genera embeddings
            print(f"🔍 Generazione embeddings per {len(texts)} conversazioni...")
            try:
                # Usa l'oggetto tenant per la pipeline  
                pipeline = self._get_pipeline(tenant)
                embeddings = pipeline.embedder.encode(texts, show_progress_bar=True)
                print(f"✅ Embeddings generati: {embeddings.shape}")
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Errore generazione embeddings: {str(e)}',
                    'tenant_id': tenant.tenant_id,
                    'tenant_slug': tenant.tenant_slug,
                    'execution_time': time.time() - start_time
                }
            
            # 5. Esegue clustering HDBSCAN
            print(f"🔗 Avvio clustering HDBSCAN...")
            
            # 🔍 DEBUG: Controlla valore alpha prima di passarlo
            alpha_value = clustering_config.get('alpha', 1.0)
            print(f"🔍 [DEBUG ALPHA] Valore alpha ricevuto: {alpha_value} (tipo: {type(alpha_value)})")
            
            # Validazione alpha prima di passarlo al clusterer
            if not isinstance(alpha_value, (int, float)) or alpha_value <= 0:
                print(f"⚠️  [ALPHA FIX] Alpha non valido: {alpha_value}, uso default 1.0")
                alpha_value = 1.0
            
            try:
                # 🆕 Includere parametri UMAP dal clustering config
                clusterer = HDBSCANClusterer(
                    min_cluster_size=clustering_config.get('min_cluster_size', 3),
                    min_samples=clustering_config.get('min_samples', 2),
                    cluster_selection_epsilon=clustering_config.get('cluster_selection_epsilon', 0.15),
                    metric=clustering_config.get('metric', 'cosine'),
                    # Parametri HDBSCAN avanzati
                    cluster_selection_method=clustering_config.get('cluster_selection_method', 'eom'),
                    alpha=alpha_value,  # USA ALPHA VALIDATO
                    max_cluster_size=clustering_config.get('max_cluster_size', 0),
                    allow_single_cluster=clustering_config.get('allow_single_cluster', False),
                    
                    # 🆕 PARAMETRI UMAP - Aggiunto supporto completo
                    use_umap=clustering_config.get('use_umap', False),
                    umap_n_neighbors=clustering_config.get('umap_n_neighbors', 15),
                    umap_min_dist=clustering_config.get('umap_min_dist', 0.1),
                    umap_metric=clustering_config.get('umap_metric', 'cosine'),
                    umap_n_components=clustering_config.get('umap_n_components', 50),
                    umap_random_state=clustering_config.get('umap_random_state', 42),
                    
                    # 🔧 FIX: Passa oggetto tenant al clusterer
                    tenant=tenant
                )
                
                cluster_labels = clusterer.fit_predict(embeddings)
                
                # 🆕 DEBUG: Stampa informazioni dettagliate UMAP
                print(f"\n🔬 [DEBUG CLUSTERING TEST] Verifica applicazione UMAP...")
                if hasattr(clusterer, 'print_umap_debug_summary'):
                    clusterer.print_umap_debug_summary()
                else:
                    print(f"⚠️  [DEBUG] Metodo print_umap_debug_summary non disponibile nel clusterer")
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Errore clustering HDBSCAN: {str(e)}',
                    'tenant_id': tenant.tenant_id,
                    'tenant_slug': tenant.tenant_slug,
                    'execution_time': time.time() - start_time
                }
            
            # 6. Analizza risultati
            unique_labels = set(cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # -1 = outliers
            n_outliers = int(np.sum(cluster_labels == -1))  # Converti numpy.int64 -> int
            n_clustered = len(texts) - n_outliers
            
            # 7. Calcola metriche di qualità
            quality_metrics = self._calculate_quality_metrics(embeddings, cluster_labels)
            
            # 8. Genera suggerimenti per ottimizzazione
            tuning_analysis = None
            if n_outliers > 0:
                tuning_analysis = HDBSCANTuningGuide.analyze_outlier_problem(
                    n_points=len(embeddings),
                    n_outliers=n_outliers, 
                    current_params=clustering_config
                )
            
            # 9. Genera dati per visualizzazioni interattive
            print("🎨 Generazione dati per visualizzazioni frontend...")
            visualization_data = self._generate_visualization_data(embeddings, cluster_labels, texts, session_ids)
            
            # 9. Costruisce cluster dettagliati
            detailed_clusters = self._build_detailed_clusters(texts, session_ids, cluster_labels)
            
            # 10. Analizza outliers
            outlier_analysis = self._analyze_outliers(texts, session_ids, cluster_labels, embeddings)
            
            execution_time = time.time() - start_time
            
            print(f"✅ Clustering completato in {execution_time:.2f}s")
            print(f"📊 Risultati: {n_clusters} clusters, {n_outliers} outliers, {n_clustered} conversazioni clusterizzate")
            
            # Costruisci risultato finale
            result_data = {
                'success': True,
                'tenant_id': tenant.tenant_id,
                'tenant_slug': tenant.tenant_slug,
                'execution_time': execution_time,
                'statistics': {
                    'total_conversations': len(texts),
                    'n_clusters': n_clusters,
                    'n_outliers': n_outliers,
                    'n_clustered': n_clustered,
                    'clustering_ratio': round(n_clustered / len(texts), 3),
                    'parameters_used': clustering_config
                },
                'quality_metrics': quality_metrics,
                'detailed_clusters': detailed_clusters,
                'outlier_analysis': outlier_analysis,
                'visualization_data': visualization_data,  # ✨ NUOVO: Dati per grafici frontend
                'tuning_suggestions': tuning_analysis,  # 🎯 NUOVO: Suggerimenti ottimizzazione
                'recommendations': self._generate_recommendations(
                    len(texts), n_clusters, n_outliers, quality_metrics
                )
            }
            
            # 🆕 SALVATAGGIO AUTOMATICO NEL DATABASE
            if self.results_db:
                try:
                    saved_result = self.results_db.save_clustering_result(
                        tenant_id=tenant.tenant_id,
                        results_data=result_data,
                        parameters_data=clustering_config,
                        execution_time=execution_time
                    )
                    
                    if saved_result and saved_result.get('record_id'):
                        print(f"💾 Risultati salvati nel database con ID: {saved_result['record_id']}")
                        result_data['saved_version_id'] = saved_result['record_id']
                        result_data['version_number'] = saved_result['version_number']
                        result_data['tenant_id'] = tenant.tenant_id
                    else:
                        print("⚠️ Errore nel salvataggio risultati nel database")
                        
                except Exception as e:
                    print(f"⚠️ Errore salvataggio database: {e}")
            else:
                print("⚠️ Database risultati non disponibile - salvataggio saltato")
            
            # 🧹 IMPORTANTE: Pulizia completa memoria dopo il test
            self.cleanup_all_pipelines()  # Pulisce la cache delle pipeline
            self._cleanup_gpu_memory()    # Poi pulisce la memoria GPU
            
            return result_data
            
        except Exception as e:
            # 🧹 Pulizia memoria anche in caso di errore
            self.cleanup_all_pipelines()
            self._cleanup_gpu_memory()
            
            return {
                'success': False,
                'error': f'Errore generale nel test clustering: {str(e)}',
                'tenant_id': tenant.tenant_id,
                'tenant_slug': tenant.tenant_slug,
                'execution_time': time.time() - start_time
            }
    
    def _calculate_quality_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calcola metriche di qualità del clustering
        
        Args:
            embeddings: Array numpy con gli embeddings
            labels: Array numpy con le etichette dei cluster
            
        Returns:
            Dizionario con metriche di qualità
        """
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            # Filtra outliers per metriche (non tutte supportano -1)
            mask = labels != -1
            outliers_count = np.sum(labels == -1)
            valid_points = np.sum(mask)
            
            if valid_points < 2:  # Serve almeno 2 punti non-outlier
                return {
                    'silhouette_score': 0.0,
                    'calinski_harabasz_score': 0.0,
                    'davies_bouldin_score': 0.0,
                    'note': 'Troppi pochi punti non-outlier per calcolare metriche'
                }
            
            filtered_embeddings = embeddings[mask]
            filtered_labels = labels[mask]
            unique_clusters = np.unique(filtered_labels)
            
            metrics = {}
            
            # Silhouette Score (più alto = meglio, range [-1, 1])
            if len(unique_clusters) > 1:
                silhouette = float(silhouette_score(filtered_embeddings, filtered_labels))
                metrics['silhouette_score'] = silhouette
            else:
                metrics['silhouette_score'] = 0.0
                
            # Calinski-Harabasz Index (più alto = meglio)
            if len(unique_clusters) > 1:
                calinski = float(calinski_harabasz_score(filtered_embeddings, filtered_labels))
                metrics['calinski_harabasz_score'] = calinski
            else:
                metrics['calinski_harabasz_score'] = 0.0
                
            # Davies-Bouldin Index (più basso = meglio)
            if len(unique_clusters) > 1:
                davies = float(davies_bouldin_score(filtered_embeddings, filtered_labels))
                metrics['davies_bouldin_score'] = davies
            else:
                metrics['davies_bouldin_score'] = 0.0
            
            return metrics
            
        except ImportError:
            return {
                'note': 'sklearn non disponibile per calcolo metriche di qualità'
            }
        except Exception as e:
            return {
                'error': f'Errore calcolo metriche: {str(e)}'
            }
    
    def _generate_visualization_data(self, 
                                   embeddings: np.ndarray, 
                                   cluster_labels: np.ndarray,
                                   texts: List[str],
                                   session_ids: List[str]) -> Dict[str, Any]:
        """
        Genera dati per le visualizzazioni interattive del clustering
        
        Args:
            embeddings: Array numpy con gli embeddings originali
            cluster_labels: Array numpy con le etichette dei cluster  
            texts: Lista testi delle conversazioni
            session_ids: Lista ID sessioni
            
        Returns:
            Dizionario con dati di visualizzazione per il frontend
            
        Data ultima modifica: 2025-08-26
        """
        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            import numpy as np
            
            visualization_data = {}
            
            # 1. Riduzione dimensionale 2D con t-SNE
            print("🔍 Generazione coordinate t-SNE 2D...")
            tsne_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            tsne_coords = tsne_2d.fit_transform(embeddings)
            
            # 2. Riduzione dimensionale 2D con PCA  
            print("🔍 Generazione coordinate PCA 2D...")
            pca_2d = PCA(n_components=2, random_state=42)
            pca_coords = pca_2d.fit_transform(embeddings)
            
            # 3. Riduzione dimensionale 3D con PCA
            print("🔍 Generazione coordinate PCA 3D...")
            if embeddings.shape[1] >= 3:
                pca_3d = PCA(n_components=3, random_state=42)
                pca_3d_coords = pca_3d.fit_transform(embeddings)
            else:
                # Fallback per embeddings con meno di 3 dimensioni
                pca_3d_coords = np.column_stack([pca_coords, np.zeros(len(pca_coords))])
            
            # 4. Prepara i dati per ogni punto
            points_data = []
            for i, (session_id, text, label) in enumerate(zip(session_ids, texts, cluster_labels)):
                point_data = {
                    'session_id': session_id,
                    'cluster_id': int(label) if label != -1 else -1,  # Frontend expects cluster_id
                    'cluster_label': f'Cluster {int(label)}' if label != -1 else 'Outliers',  # Frontend expects string
                    'text_preview': text[:150] + "..." if len(text) > 150 else text,
                    'text_length': len(text),
                    'is_outlier': label == -1,
                    # Frontend expects x,y,z - aggiungiamo coordinate multiple
                    'x': float(tsne_coords[i, 0]),      # Default to t-SNE coordinates
                    'y': float(tsne_coords[i, 1]),
                    'z': float(pca_3d_coords[i, 2]),    # 3D coordinate
                    # Manteniamo anche coordinate specifiche per backend
                    'tsne_x': float(tsne_coords[i, 0]),
                    'tsne_y': float(tsne_coords[i, 1]),
                    'pca_x': float(pca_coords[i, 0]),
                    'pca_y': float(pca_coords[i, 1]),
                    'pca_3d_x': float(pca_3d_coords[i, 0]),
                    'pca_3d_y': float(pca_3d_coords[i, 1]),
                    'pca_3d_z': float(pca_3d_coords[i, 2])
                }
                points_data.append(point_data)
            
            # 5. Statistiche sui cluster per colorazione
            cluster_stats = {}
            cluster_colors = {}  # Frontend expects cluster_colors
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                if label == -1:  # Outliers
                    cluster_stats[-1] = {
                        'label': 'Outliers',
                        'count': int(np.sum(cluster_labels == -1)),
                        'color': '#666666'
                    }
                    cluster_colors[-1] = '#666666'
                else:
                    color = f'hsl({(int(label) * 137.5) % 360}, 70%, 50%)'
                    cluster_stats[int(label)] = {
                        'label': f'Cluster {int(label)}',
                        'count': int(np.sum(cluster_labels == label)),
                        'color': color
                    }
                    cluster_colors[int(label)] = color
            
            visualization_data = {
                'points': points_data,
                'cluster_info': cluster_stats,
                'cluster_colors': cluster_colors,  # Add cluster_colors for frontend
                'dimensions': {
                    'original': embeddings.shape[1],
                    'tsne_2d': 2,
                    'pca_2d': 2,
                    'pca_3d': 3
                },
                'explained_variance_ratio': {
                    'pca_2d': pca_2d.explained_variance_ratio_.tolist(),
                    'pca_3d': pca_3d.explained_variance_ratio_.tolist() if embeddings.shape[1] >= 3 else [0.0, 0.0, 0.0]
                },
                'total_points': len(points_data),
                'n_clusters': len([l for l in unique_labels if l != -1]),
                'n_outliers': int(np.sum(cluster_labels == -1))
            }
            
            print(f"✅ Dati visualizzazione generati: {len(points_data)} punti, {visualization_data['n_clusters']} cluster")
            
            return visualization_data
            
        except ImportError as e:
            return {
                'error': f'Librerie visualizzazione non disponibili: {str(e)}',
                'points': [],
                'cluster_info': {},
                'dimensions': {'original': 0}
            }
        except Exception as e:
            print(f"❌ Errore generazione dati visualizzazione: {e}")
            return {
                'error': f'Errore nella generazione dati visualizzazione: {str(e)}',
                'points': [],
                'cluster_info': {},
                'dimensions': {'original': 0}
            }
    
    def _build_detailed_clusters(self, texts: List[str], session_ids: List[str], labels: np.ndarray) -> Dict[str, Any]:
        """
        Costruisce informazioni dettagliate sui cluster
        
        Args:
            texts: Lista dei testi
            session_ids: Lista degli ID sessione  
            labels: Array numpy con le etichette dei cluster
            
        Returns:
            Dizionario con dettagli dei cluster
        """
        cluster_groups = defaultdict(list)
        
        for i, label in enumerate(labels):
            if label != -1:  # Ignora outliers per i cluster
                cluster_groups[int(label)].append({
                    'session_id': session_ids[i],
                    'text': texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i],
                    'text_length': len(texts[i])
                })
        
        # Ordina cluster per dimensione (decrescente)
        sorted_clusters = []
        for cluster_id, conversations in sorted(cluster_groups.items(), key=lambda x: len(x[1]), reverse=True):
            sorted_clusters.append({
                'cluster_id': cluster_id,
                'size': len(conversations),
                'conversations': conversations[:10],  # Mostra solo i primi 10 per brevità
                'total_conversations': len(conversations)
            })
        
        return {
            'clusters': sorted_clusters,
            'total_clusters': len(cluster_groups)
        }
    
    def _analyze_outliers(self, texts: List[str], session_ids: List[str], 
                         labels: np.ndarray, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Analizza le conversazioni classificate come outliers
        
        Args:
            texts: Lista dei testi
            session_ids: Lista degli ID sessione
            labels: Array numpy con le etichette dei cluster
            embeddings: Array numpy con gli embeddings
            
        Returns:
            Dizionario con analisi degli outliers
        """
        outlier_indices = np.where(labels == -1)[0]
        
        if len(outlier_indices) == 0:
            return {
                'count': 0,
                'percentage': 0.0,
                'samples': [],
                'note': 'Nessun outlier trovato'
            }
        
        outliers = []
        for i in outlier_indices[:10]:  # Mostra solo i primi 10
            outliers.append({
                'session_id': session_ids[i],
                'text': texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i],
                'text_length': len(texts[i])
            })
        
        return {
            'count': len(outlier_indices),
            'percentage': round((len(outlier_indices) / len(texts)) * 100, 2),
            'samples': outliers,
            'total_outliers': len(outlier_indices)
        }
    
    def _generate_recommendations(self, n_conversations: int, n_clusters: int, 
                                n_outliers: int, quality_metrics: Dict) -> List[str]:
        """
        Genera raccomandazioni basate sui risultati del clustering
        
        Args:
            n_conversations: Numero totale conversazioni
            n_clusters: Numero di cluster trovati
            n_outliers: Numero di outliers
            quality_metrics: Metriche di qualità
            
        Returns:
            Lista di raccomandazioni testuali
        """
        recommendations = []
        
        # Analisi numero cluster
        cluster_ratio = n_clusters / n_conversations if n_conversations > 0 else 0
        if cluster_ratio > 0.5:
            recommendations.append("⚠️ Troppi cluster piccoli. Considera di aumentare min_cluster_size.")
        elif cluster_ratio < 0.05:
            recommendations.append("⚠️ Troppi pochi cluster. Considera di diminuire min_cluster_size.")
        else:
            recommendations.append("✅ Numero di cluster appropriato per il dataset.")
        
        # Analisi outliers
        outlier_ratio = n_outliers / n_conversations if n_conversations > 0 else 0
        if outlier_ratio > 0.3:
            recommendations.append("⚠️ Molti outliers (>30%). Considera di diminuire min_cluster_size o min_samples.")
        elif outlier_ratio < 0.05:
            recommendations.append("⚠️ Pochi outliers (<5%). Potresti avere cluster troppo inclusivi.")
        else:
            recommendations.append("✅ Percentuale di outliers bilanciata.")
        
        # Analisi qualità (se disponibili)
        if 'silhouette_score' in quality_metrics:
            silhouette = quality_metrics['silhouette_score']
            if silhouette > 0.5:
                recommendations.append("✅ Buona separazione dei cluster (silhouette > 0.5).")
            elif silhouette < 0.2:
                recommendations.append("⚠️ Cluster poco separati (silhouette < 0.2). Rivedi i parametri.")
            else:
                recommendations.append("⏸️ Qualità cluster moderata. Potresti ottimizzare i parametri.")
        
        return recommendations
