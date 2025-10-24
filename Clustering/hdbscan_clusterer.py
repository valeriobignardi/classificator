"""
Clustering HDBSCAN per scoperta automatica di categorie con supporto GPU

Autore: GitHub Copilot
Data creazione: 26 Agosto 2025
Aggiornamenti:
- 26/08/2025: Aggiunto supporto GPU clustering con cuML HDBSCAN
- 26/08/2025: Fallback automatico su CPU se GPU non disponibile
- 26/08/2025: Configurazione GPU tramite config.yaml
- 07/09/2025: Aggiunto sistema di tracing completo
"""

import sys
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import time
import yaml

def trace_all(component: str = None, action: str = "ENTER", function: str = None, message: str = None, details: Dict = None, **kwargs):
    """
    Sistema di tracing completo per tracciare il flusso del clustering HDBSCAN
    
    Scopo della funzione: Tracciare ingresso, uscita ed errori di tutte le funzioni
    Parametri di input: component, action, function, message, details, **kwargs
    Parametri di output: None (scrive su file)
    Valori di ritorno: None
    Tracciamento aggiornamenti: 2025-09-07 - Valerio Bignardi - Sistema tracing HDBSCAN
    
    Args:
        component (str): Nome del componente (es. "HDBSCANClusterer")
        action (str): "ENTER", "EXIT", "ERROR"
        function (str): Nome della funzione
        message (str): Messaggio descrittivo
        details (Dict): Dettagli da tracciare
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
        
        # Costruisci messaggio di tracing
        function_name = function or "unknown"
        component_name = component or "Unknown"
        
        message_parts = [f"[{timestamp}]", f"{action:>5}", "->", f"{component_name}::{function_name}"]
        
        # Aggiungi messaggio se fornito
        if message:
            message_parts.append(f"- {message}")
        
        # Aggiungi dettagli se richiesto
        if action == "ENTER" and include_parameters and details:
            params_str = []
            for key, value in details.items():
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
        
        # Aggiungi dettagli anche per EXIT se disponibili
        elif action == "EXIT" and include_parameters and details:
            details_str = []
            for key, value in details.items():
                try:
                    if isinstance(value, (dict, list)):
                        if len(str(value)) > 100:
                            value_str = f"{type(value).__name__}(size={len(value)})"
                        else:
                            value_str = json.dumps(value, default=str, ensure_ascii=False)[:100]
                    else:
                        value_str = str(value)[:100]
                    details_str.append(f"{key}={value_str}")
                except Exception:
                    details_str.append(f"{key}=<{type(value).__name__}>")
            
            if details_str:
                message_parts.append(f"({', '.join(details_str)})")
        
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

# Import standard CPU clustering
import hdbscan
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter

# 🆕 UMAP import
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
    print("🚀 UMAP disponibile per riduzione dimensionale!")
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP non disponibile - installare con: pip install umap-learn")

# GPU clustering imports con fallback
try:
    import os
    # Setup CUDA environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    import cuml
    import cupy as cp
    from cuml.cluster import HDBSCAN as cumlHDBSCAN
    
    # Test GPU funzionante
    _ = cp.cuda.Device(0).compute_capability
    GPU_AVAILABLE = True
    print("🚀 cuML GPU clustering disponibile!")
except Exception as e:
    GPU_AVAILABLE = False
    print(f"⚠️  cuML non disponibile ({str(e)[:50]}...) - solo clustering CPU")

# Aggiunge i percorsi per importare gli altri moduli
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EmbeddingEngine'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Preprocessing'))

# RIMOSSO: from labse_embedder import LaBSEEmbedder - Ora usa solo Docker service
from session_aggregator import SessionAggregator

class HDBSCANClusterer:
    """
    Clustering delle sessioni usando HDBSCAN per scoperta automatica di categorie
    
    Supporta clustering sia su CPU che su GPU (cuML) con fallback automatico.
    Configurazione tramite config.yaml: gpu_enabled, gpu_fallback_cpu, gpu_memory_limit
    
    Input: Array di embeddings (n_samples, embedding_dim)
    Output: Array di etichette cluster (-1 per outlier)
    
    Ultima modifica: 26 Agosto 2025 - Aggiunto supporto GPU
    """
    
    def __init__(self, 
                 min_cluster_size: Optional[int] = None,
                 min_samples: Optional[int] = None,
                 cluster_selection_epsilon: Optional[float] = None,
                 metric: Optional[str] = None,
                 cluster_selection_method: Optional[str] = None,
                 alpha: Optional[float] = None,
                 max_cluster_size: Optional[int] = None,
                 allow_single_cluster: Optional[bool] = None,
                 # 🆕 PARAMETRI UMAP
                 use_umap: Optional[bool] = None,
                 umap_n_neighbors: Optional[int] = None,
                 umap_min_dist: Optional[float] = None,
                 umap_metric: Optional[str] = None,
                 umap_n_components: Optional[int] = None,
                 umap_random_state: Optional[int] = None,
                 config_path: Optional[str] = None,
                 tenant: Optional[object] = None):  # 🆕 PARAMETRO TENANT per config React
        """
        Inizializza il clusterer HDBSCAN con parametri da configurazione tenant React
        
        Args:
            min_cluster_size: Dimensione minima dei cluster (sovrascrive config)
            min_samples: Numero minimo di campioni (sovrascrive config)
            cluster_selection_epsilon: Distanza massima per cluster (sovrascrive config)
            metric: Metrica di distanza (sovrascrive config)
            tenant: Oggetto Tenant per caricamento configurazione React
            config_path: Percorso del file di configurazione (fallback)
            
        Ultima modifica: 02 Settembre 2025 - Supporto parametri React
        """
        trace_all(
            component="HDBSCANClusterer",
            action="ENTER",
            function="__init__",
            message="Inizializzazione HDBSCANClusterer",
            details={
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_epsilon": cluster_selection_epsilon,
                "metric": metric,
                "use_umap": use_umap,
                "tenant": type(tenant).__name__ if tenant else None
            }
        )
        
        # Salva riferimento tenant per caricamento config React
        self.tenant = tenant
        
        # 🔧 ESTRAI tenant_id dall'oggetto Tenant
        print(f"🔍 [DEBUG TENANT] Tipo oggetto tenant: {type(tenant)}")
        if tenant:
            print(f"🔍 [DEBUG TENANT] Attributi tenant: {dir(tenant)}")
            if hasattr(tenant, 'tenant_id'):
                self.tenant_id = tenant.tenant_id
                print(f"✅ [TENANT] tenant_id estratto: {self.tenant_id}")
            elif hasattr(tenant, 'id'):
                self.tenant_id = tenant.id  # Fallback se attributo diverso
                print(f"✅ [TENANT] tenant_id estratto da .id: {self.tenant_id}")
            else:
                self.tenant_id = None
                print(f"❌ [TENANT] Nessun tenant_id trovato negli attributi: {[attr for attr in dir(tenant) if not attr.startswith('_')]}")
        else:
            self.tenant_id = None
            print("⚠️ [TENANT] Oggetto tenant è None")
        
        # Carica configurazione
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # 🆕 PRIORITÀ CONFIGURAZIONE: tenant_configs > config.yaml > defaults
        clustering_config = self._load_tenant_clustering_config()
        
        self.min_cluster_size = min_cluster_size or clustering_config.get('min_cluster_size', 5)
        self.min_samples = min_samples or clustering_config.get('min_samples', 3)
        self.cluster_selection_epsilon = cluster_selection_epsilon or clustering_config.get('cluster_selection_epsilon', 0.05)
        self.metric = metric or clustering_config.get('metric', 'cosine')
        
        # Parametri avanzati da config o input - ESPANSI PER FRONTEND
        # 🔧 BUGFIX: Usa controllo esplicito per evitare sovrascrittura con default
        self.cluster_selection_method = cluster_selection_method if cluster_selection_method is not None else clustering_config.get('cluster_selection_method', 'eom')
        self.alpha = alpha if alpha is not None else clustering_config.get('alpha', 1.0)  # Controllo noise/outlier - BUGFIX
        # 🔧 BUGFIX: Assicura che alpha sia sempre un float per HDBSCAN
        self.alpha = float(self.alpha) if self.alpha is not None else 1.0
        self.max_cluster_size = max_cluster_size if max_cluster_size is not None else clustering_config.get('max_cluster_size', 0)  # 0 = unlimited 
        self.allow_single_cluster = allow_single_cluster if allow_single_cluster is not None else clustering_config.get('allow_single_cluster', False)
        
        # Parametri performance
        self.leaf_size = clustering_config.get('leaf_size', 40)
        self.n_representatives = clustering_config.get('n_representatives', 3)
        self.min_silhouette_score = clustering_config.get('min_silhouette_score', 0.2)
        self.max_outlier_ratio = clustering_config.get('max_outlier_ratio', 0.7)
        
        # Configurazione GPU - NOVITÀ
        self.gpu_enabled = clustering_config.get('gpu_enabled', False) and GPU_AVAILABLE
        self.gpu_fallback_cpu = clustering_config.get('gpu_fallback_cpu', True)
        self.gpu_memory_limit = clustering_config.get('gpu_memory_limit', 0.8)
        
        # 🆕 CONFIGURAZIONE UMAP
        umap_config = clustering_config.get('umap', {})
        self.use_umap = use_umap if use_umap is not None else umap_config.get('use_umap', False)
        self.umap_n_neighbors = umap_n_neighbors or umap_config.get('n_neighbors', 30)
        self.umap_min_dist = umap_min_dist if umap_min_dist is not None else umap_config.get('min_dist', 0.1)
        self.umap_metric = umap_metric or umap_config.get('metric', 'cosine')
        self.umap_n_components = umap_n_components or umap_config.get('n_components', 50)  # Riduzione dimensionale prima di HDBSCAN
        self.umap_random_state = umap_random_state or umap_config.get('random_state', 42)
        
        # Valida disponibilità UMAP
        if self.use_umap and not UMAP_AVAILABLE:
            print("⚠️  UMAP richiesto ma non disponibile - disabilitato")
            self.use_umap = False
        
        # Stato clustering
        self.clusterer = None
        self.cluster_labels = None
        self.cluster_probabilities = None
        self.outlier_scores = None
        self.gpu_used = False  # Track se GPU è stato effettivamente usato
        
        # Log configurazione
        print(f"🔧 HDBSCANClusterer inizializzato:")
        print(f"   min_cluster_size: {self.min_cluster_size}")
        print(f"   min_samples: {self.min_samples}")
        print(f"   metric: {self.metric}")
        print(f"   cluster_selection_epsilon: {self.cluster_selection_epsilon}")
        
        # 🆕 LOG CONFIGURAZIONE UMAP
        if self.use_umap:
            print(f"   🗂️  UMAP enabled: True")
            print(f"     📏 n_neighbors: {self.umap_n_neighbors}")
            print(f"     📐 min_dist: {self.umap_min_dist}")
            print(f"     📊 n_components: {self.umap_n_components}")
            print(f"     🎯 metric: {self.umap_metric}")
        else:
            print(f"   🗂️  UMAP enabled: False")
        
        print(f"   🚀 GPU enabled: {self.gpu_enabled}")
        if self.gpu_enabled:
            print(f"   💾 GPU memory limit: {self.gpu_memory_limit*100:.0f}%")
            print(f"   🔄 CPU fallback: {self.gpu_fallback_cpu}")
            
            # ⚠️  AVVISO LIMITAZIONI GPU
            print(f"   ⚠️  [GPU MODE] Limitazioni cuML HDBSCAN:")
            if self.max_cluster_size and self.max_cluster_size > 0:
                print(f"     🚫 max_cluster_size={self.max_cluster_size} sarà IGNORATO su GPU")
            if hasattr(self, 'leaf_size'):
                print(f"     🚫 leaf_size={self.leaf_size} sarà IGNORATO su GPU")
            print(f"     ✅ Parametri supportati: cluster_selection_method, alpha, allow_single_cluster")
        
        # 🆕 DEBUG PARAMETRI AVANZATI PER REACT
        print(f"🎯 [DEBUG REACT] Parametri configurati:")
        print(f"   cluster_selection_method: {self.cluster_selection_method}")
        print(f"   alpha: {self.alpha}")
        print(f"   allow_single_cluster: {self.allow_single_cluster}")
        
        trace_all(
            component="HDBSCANClusterer",
            action="EXIT",
            function="__init__",
            message="HDBSCANClusterer inizializzato con successo",
            details={
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
                "umap_enabled": self.use_umap,
                "gpu_enabled": self.gpu_enabled
            }
        )
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Carica la configurazione dal file YAML
        
        Returns:
            Dizionario di configurazione
            
        Ultima modifica: 26 Agosto 2025
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"⚠️ Errore nel caricamento config da {config_path}: {e}")
            print("📝 Uso parametri predefiniti")
            return {}
    
    def _load_tenant_clustering_config(self) -> Dict[str, Any]:
        """
        Carica i parametri di clustering personalizzati per il tenant dall'interfaccia React
        
        Scopo:
        - Legge i parametri HDBSCAN e UMAP salvati dall'interfaccia React
        - Supporta override completo dei parametri di default
        - Gestisce fallback sicuro alla configurazione default da config.yaml
        
        Autore: Valerio Bignardi
        Data ultima modifica: 2025-09-02
        
        Returns:
            Dict con i parametri personalizzati merged con quelli di default
        """
        # Carica configurazione base da config.yaml
        base_clustering_config = self.config.get('clustering', {})
        
        # Se non c'è tenant_id, usa solo configurazione base
        if not self.tenant_id:
            print(f"📋 [TENANT CONFIG] Nessun tenant_id fornito, uso config.yaml")
            return base_clustering_config
            
        try:
            # Costruisci path al file di configurazione tenant
            tenant_config_dir = os.path.join(os.path.dirname(__file__), '..', 'tenant_configs')
            tenant_config_file = os.path.join(tenant_config_dir, f'{self.tenant_id}_clustering.yaml')
            
            if os.path.exists(tenant_config_file):
                with open(tenant_config_file, 'r', encoding='utf-8') as f:
                    tenant_config = yaml.safe_load(f)
                    tenant_clustering_params = tenant_config.get('clustering_parameters', {})
                    
                    if tenant_clustering_params:
                        print(f"🎯 [TENANT CONFIG] Parametri personalizzati trovati per tenant {self.tenant_id}")
                        print(f"   📊 Parametri personalizzati: {list(tenant_clustering_params.keys())}")
                        
                        # Merge: tenant_params sovrascrivono base_params
                        merged_config = base_clustering_config.copy()
                        merged_config.update(tenant_clustering_params)
                        
                        # Log delle sovrascritture
                        for param, value in tenant_clustering_params.items():
                            base_value = base_clustering_config.get(param, 'non_definito')
                            print(f"   🔄 {param}: {base_value} → {value}")
                        
                        return merged_config
                    else:
                        print(f"📋 [TENANT CONFIG] File config tenant {self.tenant_id} vuoto, uso config.yaml")
                        return base_clustering_config
            else:
                print(f"📋 [TENANT CONFIG] Nessun file config per tenant {self.tenant_id}, uso config.yaml")
                return base_clustering_config
                
        except Exception as e:
            print(f"⚠️ [TENANT CONFIG] Errore caricamento config tenant {self.tenant_id}: {e}")
            print("🔄 [TENANT CONFIG] Fallback a config.yaml")
            return base_clustering_config
    
    def _apply_umap_reduction(self, embeddings: np.ndarray, fit_new: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Applica riduzione dimensionale con UMAP prima del clustering HDBSCAN
        
        Args:
            embeddings: Array di embeddings originali (n_samples, orig_dim)
            fit_new: Se True, fit nuovo reducer. Se False, usa reducer esistente (per predizioni incrementali)
            
        Returns:
            Tuple[embeddings ridotti (n_samples, umap_n_components), info UMAP]
            
        Data creazione: 27 Agosto 2025
        Ultimo aggiornamento: 28 Agosto 2025 - Aggiunto supporto predizioni incrementali
        """
        trace_all(
            component="HDBSCANClusterer",
            action="ENTER",
            function="_apply_umap_reduction",
            message="Inizio riduzione dimensionale UMAP",
            details={
                "input_shape": embeddings.shape,
                "fit_new": fit_new,
                "umap_enabled": self.use_umap,
                "umap_available": UMAP_AVAILABLE,
                "target_components": self.umap_n_components
            }
        )
        
        print(f"🔍 [DEBUG UMAP] Controllo condizioni applicazione UMAP...")
        print(f"   ✅ self.use_umap = {self.use_umap}")
        print(f"   ✅ UMAP_AVAILABLE = {UMAP_AVAILABLE}")
        
        if not self.use_umap:
            print(f"❌ [DEBUG UMAP] UMAP disabilitato da configurazione (use_umap=False)")
            return embeddings, {'applied': False, 'reason': 'UMAP disabled by configuration'}
        
        if not UMAP_AVAILABLE:
            print(f"❌ [DEBUG UMAP] UMAP non disponibile (libreria non installata)")
            return embeddings, {'applied': False, 'reason': 'UMAP library not available'}
        
        print(f"� [DEBUG UMAP] UMAP ABILITATO - Iniziando riduzione dimensionale...")
        print(f"   📏 Dimensioni input: {embeddings.shape}")
        print(f"   🔢 Tipo dati input: {embeddings.dtype}")
        print(f"   📊 Range valori input: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
        print(f"   🎯 Parametri UMAP configurati:")
        print(f"     📐 n_neighbors: {self.umap_n_neighbors}")
        print(f"     📏 min_dist: {self.umap_min_dist}")
        print(f"     🔢 n_components: {self.umap_n_components}")
        print(f"     📈 metric: {self.umap_metric}")
        print(f"     🎲 random_state: {self.umap_random_state}")
        
        # 🆕 DEBUG: Verifica che gli embedding siano validi
        if np.any(np.isnan(embeddings)):
            print(f"⚠️  [DEBUG UMAP] ATTENZIONE: Trovati NaN negli embeddings!")
            nan_count = np.sum(np.isnan(embeddings))
            print(f"   🔢 NaN trovati: {nan_count}")
        
        if np.any(np.isinf(embeddings)):
            print(f"⚠️  [DEBUG UMAP] ATTENZIONE: Trovati valori infiniti negli embeddings!")
            inf_count = np.sum(np.isinf(embeddings))
            print(f"   🔢 Infiniti trovati: {inf_count}")
        
        start_time = time.time()
        
        try:
            print(f"🔧 [DEBUG UMAP] Inizializzazione riduttore UMAP...")
            
            if fit_new:
                # TRAINING: Inizializza e salva nuovo reducer
                print(f"🆕 [DEBUG UMAP] Modalità TRAINING - fit nuovo reducer")
                
                # Inizializza UMAP
                self.umap_reducer = umap.UMAP(
                    n_neighbors=self.umap_n_neighbors,
                    min_dist=self.umap_min_dist,
                    n_components=self.umap_n_components,
                    metric=self.umap_metric,
                    random_state=self.umap_random_state,
                    verbose=True
                )
                
                print(f"✅ [DEBUG UMAP] Riduttore UMAP inizializzato con successo")
                print(f"⏳ [DEBUG UMAP] Applicazione fit_transform agli embeddings...")
                
                # Applica riduzione dimensionale
                embeddings_reduced = self.umap_reducer.fit_transform(embeddings)
                
            else:
                # PREDIZIONE: Usa reducer esistente
                print(f"🔮 [DEBUG UMAP] Modalità PREDIZIONE - usa reducer esistente")
                
                if not hasattr(self, 'umap_reducer') or self.umap_reducer is None:
                    print(f"❌ [DEBUG UMAP] ERRORE: Nessun reducer UMAP disponibile per predizione!")
                    return embeddings, {'applied': False, 'reason': 'No UMAP reducer available for prediction'}
                
                print(f"⏳ [DEBUG UMAP] Applicazione transform agli embeddings...")
                
                # Applica riduzione con reducer esistente
                embeddings_reduced = self.umap_reducer.transform(embeddings)
            
            reduction_time = time.time() - start_time
            
            # 🆕 DEBUG: Verifica risultato riduzione
            print(f"✅ [DEBUG UMAP] UMAP COMPLETATO con SUCCESSO in {reduction_time:.2f}s")
            print(f"   📐 Dimensioni output: {embeddings_reduced.shape}")
            print(f"   � Tipo dati output: {embeddings_reduced.dtype}")
            print(f"   �📊 Range valori output: [{embeddings_reduced.min():.4f}, {embeddings_reduced.max():.4f}]")
            print(f"   📈 Riduzione dimensionale: {embeddings.shape[1]} → {embeddings_reduced.shape[1]} dimensioni")
            print(f"   📉 Fattore riduzione: {embeddings.shape[1] / embeddings_reduced.shape[1]:.1f}x")
            
            # 🆕 DEBUG: Verifica che la riduzione sia realmente avvenuta
            if embeddings.shape[1] == embeddings_reduced.shape[1]:
                print(f"⚠️  [DEBUG UMAP] ATTENZIONE: Le dimensioni non sono cambiate!")
            else:
                print(f"✅ [DEBUG UMAP] RIDUZIONE DIMENSIONALE CONFERMATA")
            
            # 🆕 DEBUG: Controllo qualità riduzione
            if np.any(np.isnan(embeddings_reduced)):
                print(f"❌ [DEBUG UMAP] ERRORE: NaN nel risultato UMAP!")
            else:
                print(f"✅ [DEBUG UMAP] Nessun NaN nel risultato UMAP")
                
            if np.any(np.isinf(embeddings_reduced)):
                print(f"❌ [DEBUG UMAP] ERRORE: Valori infiniti nel risultato UMAP!")
            else:
                print(f"✅ [DEBUG UMAP] Nessun valore infinito nel risultato UMAP")
            
            # 🆕 DEBUG: Verifica embedding ridotti non siano tutti uguali
            unique_rows = len(np.unique(embeddings_reduced, axis=0))
            print(f"📊 [DEBUG UMAP] Diversità embeddings ridotti: {unique_rows}/{embeddings_reduced.shape[0]} righe uniche")
            if unique_rows == 1:
                print(f"❌ [DEBUG UMAP] ERRORE: Tutti gli embeddings ridotti sono identici!")
            elif unique_rows < embeddings_reduced.shape[0] * 0.5:
                print(f"⚠️  [DEBUG UMAP] ATTENZIONE: Bassa diversità negli embeddings ridotti")
            else:
                print(f"✅ [DEBUG UMAP] Buona diversità negli embeddings ridotti")
            
            umap_info = {
                'applied': True,
                'input_shape': embeddings.shape,
                'output_shape': embeddings_reduced.shape,
                'reduction_time': reduction_time,
                'reduction_factor': embeddings.shape[1] / embeddings_reduced.shape[1],
                'unique_embeddings': unique_rows,
                'parameters': {
                    'n_neighbors': self.umap_n_neighbors,
                    'min_dist': self.umap_min_dist,
                    'n_components': self.umap_n_components,
                    'metric': self.umap_metric,
                    'random_state': self.umap_random_state
                }
            }
            
            print(f"📋 [DEBUG UMAP] Info riduzione salvate nel clusterer")
            
            trace_all(
                component="HDBSCANClusterer",
                action="EXIT",
                function="_apply_umap_reduction",
                message="Riduzione UMAP completata con successo",
                details={
                    "applied": True,
                    "original_shape": embeddings.shape,
                    "reduced_shape": embeddings_reduced.shape,
                    "reduction_time": umap_info['reduction_time'],
                    "reduction_factor": umap_info['reduction_factor']
                },
                return_value=f"Reduced embeddings {embeddings_reduced.shape}, UMAP info"
            )
            
            return embeddings_reduced, umap_info
            
        except Exception as e:
            print(f"❌ [DEBUG UMAP] ERRORE durante riduzione UMAP: {e}")
            print(f"🔄 [DEBUG UMAP] Fallback: uso embeddings originali")
            
            fallback_info = {
                'applied': False, 
                'reason': f'UMAP failed: {str(e)}',
                'fallback': True
            }
            
            trace_all(
                component="HDBSCANClusterer",
                action="EXIT",
                function="_apply_umap_reduction",
                message="Riduzione UMAP fallita - fallback a embeddings originali",
                details={
                    "applied": False,
                    "error": str(e),
                    "fallback": True,
                    "original_shape": embeddings.shape
                },
                return_value=f"Original embeddings {embeddings.shape}, fallback info"
            )
            
            return embeddings, fallback_info
    
    def _check_gpu_memory(self, embeddings_size_mb: float) -> bool:
        """
        Verifica se c'è abbastanza memoria GPU per il clustering
        
        Args:
            embeddings_size_mb: Dimensione stimata degli embeddings in MB
            
        Returns:
            True se c'è abbastanza memoria GPU
            
        Ultima modifica: 26 Agosto 2025
        """
        if not self.gpu_enabled or not GPU_AVAILABLE:
            return False
            
        try:
            import torch
            if not torch.cuda.is_available():
                return False
                
            # Memoria GPU disponibile
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            gpu_mem_available = gpu_mem_total * self.gpu_memory_limit
            
            # Stima memoria necessaria (embeddings + overhead clustering)
            estimated_memory_gb = (embeddings_size_mb * 3) / 1024  # 3x overhead per clustering
            
            print(f"💾 Memoria GPU: {gpu_mem_available:.1f}GB disponibili, {estimated_memory_gb:.1f}GB necessari")
            
            return estimated_memory_gb < gpu_mem_available
            
        except Exception as e:
            print(f"⚠️ Errore controllo memoria GPU: {e}")
            return False
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Esegue il clustering sugli embedding con supporto GPU
        
        Args:
            embeddings: Array di embedding (n_samples, embedding_dim)
            
        Returns:
            Array delle etichette dei cluster (-1 per outlier)
            
        Ultima modifica: 26 Agosto 2025 - Aggiunto supporto GPU
        """
        trace_all(
            component="HDBSCANClusterer",
            action="ENTER",
            function="fit_predict",
            message="Inizio clustering HDBSCAN",
            details={
                "n_samples": embeddings.shape[0],
                "embedding_dim": embeddings.shape[1],
                "embeddings_size_mb": embeddings.nbytes / (1024**2),
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
                "gpu_enabled": self.gpu_enabled,
                "umap_enabled": self.use_umap
            }
        )
        
        n_samples, embedding_dim = embeddings.shape
        embeddings_size_mb = (embeddings.nbytes / (1024**2))
        
        print(f"🔍 Clustering HDBSCAN su {n_samples} embedding ({embeddings_size_mb:.1f}MB)...")
        print(f"⚙️  Parametri: min_cluster_size={self.min_cluster_size}, "
              f"min_samples={self.min_samples}, metric={self.metric}")
        
        # 🆕 DEBUG: Stato iniziale embeddings
        print(f"📊 [DEBUG FIT_PREDICT] Embedding input:")
        print(f"   📏 Shape: {embeddings.shape}")
        print(f"   🔢 Dtype: {embeddings.dtype}")
        print(f"   📈 Range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
        print(f"   💾 Memory: {embeddings_size_mb:.1f}MB")
        
        # 🆕 STEP 1: Applica UMAP se abilitato
        print(f"\n🗂️  [DEBUG FIT_PREDICT] STEP 1: Controllo applicazione UMAP...")
        embeddings_for_clustering, umap_info = self._apply_umap_reduction(embeddings)
        self.umap_info = umap_info  # Salva info per debugging
        
        print(f"📋 [DEBUG FIT_PREDICT] Risultato UMAP:")
        print(f"   ✅ UMAP applicato: {umap_info.get('applied', False)}")
        if umap_info.get('applied'):
            print(f"   📏 Shape post-UMAP: {embeddings_for_clustering.shape}")
            print(f"   📈 Range post-UMAP: [{embeddings_for_clustering.min():.4f}, {embeddings_for_clustering.max():.4f}]")
            print(f"   ⏱️  Tempo riduzione: {umap_info.get('reduction_time', 0):.2f}s")
        else:
            print(f"   ❌ Motivo non applicazione: {umap_info.get('reason', 'Unknown')}")
        
        # STEP 2: Normalizza gli embedding se necessario
        print(f"\n🔧 [DEBUG FIT_PREDICT] STEP 2: Normalizzazione per metrica {self.metric}...")
        if self.metric == 'cosine':
            # Per coseno, usa euclidean su embedding normalizzati
            print(f"   🎯 Normalizzazione per metrica cosine -> euclidean")
            embeddings_norm = embeddings_for_clustering / np.linalg.norm(embeddings_for_clustering, axis=1, keepdims=True)
            metric_for_clustering = 'euclidean'
            print(f"   📊 Range embeddings normalizzati: [{embeddings_norm.min():.4f}, {embeddings_norm.max():.4f}]")
        else:
            print(f"   🎯 Nessuna normalizzazione necessaria per metrica {self.metric}")
            embeddings_norm = embeddings_for_clustering
            metric_for_clustering = self.metric
        
        print(f"   📏 Shape finale per clustering: {embeddings_norm.shape}")
        print(f"   📈 Range finale: [{embeddings_norm.min():.4f}, {embeddings_norm.max():.4f}]")
        print(f"   🎯 Metrica effettiva clustering: {metric_for_clustering}")
        
        # Determina se usare GPU
        use_gpu = self._should_use_gpu(embeddings_size_mb)
        
        start_time = time.time()
        print(f"\n🚀 [DEBUG FIT_PREDICT] STEP 3: Avvio clustering...")
        print(f"   🖥️  Modalità: {'GPU' if use_gpu else 'CPU'}")
        
        if use_gpu:
            cluster_labels = self._fit_predict_gpu(embeddings_norm, metric_for_clustering)
        else:
            cluster_labels = self._fit_predict_cpu(embeddings_norm, metric_for_clustering)
        
        clustering_time = time.time() - start_time
        
        # Salva risultati
        self.cluster_labels = cluster_labels
        
        # 🆕 Salva forma embeddings per compatibilità predizioni future
        self.last_embeddings_shape = embeddings.shape
        
        # Statistiche clustering
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_outliers = list(cluster_labels).count(-1)
        
        device_used = "🚀 GPU" if self.gpu_used else "🖥️  CPU"
        print(f"\n✅ [DEBUG FIT_PREDICT] CLUSTERING COMPLETATO in {clustering_time:.2f}s con {device_used}!")
        print(f"📊 Risultati finali:")
        print(f"   🎯 Cluster trovati: {n_clusters}")
        print(f"   🔍 Outlier: {n_outliers} ({n_outliers/len(cluster_labels)*100:.1f}%)")
        print(f"   📈 Silhouette score: {self._calculate_silhouette_score(embeddings_norm):.3f}")
        
        # 🆕 DEBUG: Riepilogo trasformazioni applicate
        print(f"\n📋 [DEBUG FIT_PREDICT] RIEPILOGO TRASFORMAZIONI:")
        print(f"   🔤 Input originale: {embeddings.shape}")
        if umap_info.get('applied'):
            print(f"   🗂️  Dopo UMAP: {embeddings_for_clustering.shape}")
        print(f"   🔧 Dopo normalizzazione: {embeddings_norm.shape}")
        print(f"   🎯 Utilizzato per clustering: {embeddings_norm.shape} con metrica {metric_for_clustering}")
        
        # 🆕 SALVA EMBEDDING FINALI per accesso dalla pipeline (Question 4)
        self.final_embeddings = embeddings_for_clustering  # Salva embeddings dopo UMAP (se applicato)
        self.final_embeddings_normalized = embeddings_norm  # Salva embeddings dopo normalizzazione
        print(f"💾 [DEBUG FIT_PREDICT] Embedding finali salvati per accesso pipeline:")
        print(f"   📦 final_embeddings (post-UMAP): {self.final_embeddings.shape}")
        print(f"   📦 final_embeddings_normalized: {self.final_embeddings_normalized.shape}")
        
        trace_all(
            component="HDBSCANClusterer",
            action="EXIT",
            function="fit_predict",
            message="Clustering HDBSCAN completato",
            details={
                "clustering_time": clustering_time,
                "n_clusters": n_clusters,
                "n_outliers": n_outliers,
                "outlier_percentage": n_outliers/len(cluster_labels)*100,
                "device_used": "GPU" if self.gpu_used else "CPU",
                "umap_applied": umap_info.get('applied', False),
                "final_shape": embeddings_norm.shape
            },
            return_value=f"cluster_labels array with {len(cluster_labels)} labels, {n_clusters} clusters"
        )
        
        return cluster_labels
    
    def _should_use_gpu(self, embeddings_size_mb: float) -> bool:
        """
        Determina se utilizzare GPU per il clustering
        
        Args:
            embeddings_size_mb: Dimensione embeddings in MB
            
        Returns:
            True se GPU dovrebbe essere usato
            
        Ultima modifica: 26 Agosto 2025
        """
        if not self.gpu_enabled:
            print("🖥️  GPU disabilitato da configurazione - uso CPU")
            return False
            
        if not GPU_AVAILABLE:
            print("❌ cuML non disponibile - uso CPU")
            return False
        
        if not self._check_gpu_memory(embeddings_size_mb):
            if self.gpu_fallback_cpu:
                print("💾 Memoria GPU insufficiente - fallback su CPU")
                return False
            else:
                raise RuntimeError("Memoria GPU insufficiente e fallback CPU disabilitato")
        
        print("🚀 Utilizzo GPU per clustering accelerato")
        return True
    
    def _fit_predict_gpu(self, embeddings_norm: np.ndarray, metric: str) -> np.ndarray:
        """
        Clustering GPU con cuML HDBSCAN
        
        Args:
            embeddings_norm: Embeddings normalizzati
            metric: Metrica di distanza
            
        Returns:
            Array di etichette cluster
            
        Ultima modifica: 26 Agosto 2025
        """
        try:
            # ⚠️  AVVISO PARAMETRI NON SUPPORTATI DA cuML GPU
            unsupported_params = []
            if self.max_cluster_size and self.max_cluster_size > 0:
                unsupported_params.append(f"max_cluster_size={self.max_cluster_size}")
            if hasattr(self, 'leaf_size') and self.leaf_size != 40:  # 40 è il default
                unsupported_params.append(f"leaf_size={self.leaf_size}")
            
            if unsupported_params:
                print(f"⚠️  [GPU CLUSTERING] Parametri NON supportati da cuML (saranno ignorati):")
                for param in unsupported_params:
                    print(f"   🚫 {param} - solo disponibile su CPU")
                print(f"✅ [GPU CLUSTERING] Parametri supportati attivi:")
                print(f"   🎯 cluster_selection_method={self.cluster_selection_method}")
                print(f"   🎛️  alpha={self.alpha}")
                print(f"   🔘 allow_single_cluster={self.allow_single_cluster}")
            
            # Converti a CuPy array per GPU
            gpu_embeddings = cp.asarray(embeddings_norm, dtype=cp.float32)
            
            # Inizializza clusterer GPU - SOLO PARAMETRI SUPPORTATI
            gpu_params = {
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'cluster_selection_epsilon': self.cluster_selection_epsilon,
                'metric': metric,
                'cluster_selection_method': self.cluster_selection_method,
                'allow_single_cluster': self.allow_single_cluster,
                'alpha': self.alpha  # Controllo noise - SUPPORTATO
            }
            
            print(f"� [GPU CLUSTERING] Inizializzazione cuML HDBSCAN con parametri supportati...")
            self.clusterer = cumlHDBSCAN(**gpu_params)
            
            # Clustering su GPU
            cluster_labels = self.clusterer.fit_predict(gpu_embeddings)
            
            # Converti risultati back to CPU
            if hasattr(cluster_labels, 'get'):
                cluster_labels = cluster_labels.get()  # CuPy to NumPy
            
            # Ottieni probabilità se disponibili
            if hasattr(self.clusterer, 'probabilities_'):
                self.cluster_probabilities = self.clusterer.probabilities_
                if hasattr(self.cluster_probabilities, 'get'):
                    self.cluster_probabilities = self.cluster_probabilities.get()
            else:
                self.cluster_probabilities = np.ones(len(cluster_labels))
            
            # Outlier scores (se disponibili)
            if hasattr(self.clusterer, 'outlier_scores_'):
                self.outlier_scores = self.clusterer.outlier_scores_
                if hasattr(self.outlier_scores, 'get'):
                    self.outlier_scores = self.outlier_scores.get()
            else:
                self.outlier_scores = np.zeros(len(cluster_labels))
            
            self.gpu_used = True
            
            # Pulizia memoria GPU
            del gpu_embeddings
            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
            
            return cluster_labels
            
        except Exception as e:
            print(f"❌ Errore clustering GPU: {e}")
            if self.gpu_fallback_cpu:
                print("🔄 Fallback automatico su CPU...")
                return self._fit_predict_cpu(embeddings_norm, metric)
            else:
                raise
    
    def _fit_predict_cpu(self, embeddings_norm: np.ndarray, metric: str) -> np.ndarray:
        """
        Clustering CPU standard con hdbscan
        
        Args:
            embeddings_norm: Embeddings normalizzati
            metric: Metrica di distanza
            
        Returns:
            Array di etichette cluster
            
        Ultima modifica: 26 Agosto 2025
        """
        # Costruisci parametri dinamicamente per HDBSCAN
        hdbscan_params = {
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'cluster_selection_epsilon': self.cluster_selection_epsilon,
            'metric': metric,
            'cluster_selection_method': self.cluster_selection_method,
            'allow_single_cluster': self.allow_single_cluster,
            'alpha': self.alpha,  # Controllo noise - VALIDATO
            'leaf_size': self.leaf_size,  # NUOVO
            'prediction_data': True  # 🆕 ABILITATO per predizioni incrementali
        }
        
        # FIX: Aggiungi max_cluster_size SOLO se > 0 (per evitare errore NoneType)
        if self.max_cluster_size and self.max_cluster_size > 0:
            hdbscan_params['max_cluster_size'] = self.max_cluster_size
            
        print(f"🔍 [HDBSCAN DEBUG] Parametri finali: alpha={hdbscan_params['alpha']}, min_cluster_size={hdbscan_params['min_cluster_size']}")
            
        # Inizializza clusterer CPU standard
        self.clusterer = hdbscan.HDBSCAN(**hdbscan_params)
        
        cluster_labels = self.clusterer.fit_predict(embeddings_norm)
        
        # Salva risultati clustering
        self.cluster_probabilities = self.clusterer.probabilities_
        self.outlier_scores = self.clusterer.outlier_scores_
        self.gpu_used = False
        
        return cluster_labels
    
    def predict_new_points(self, new_embeddings: np.ndarray, fit_umap: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scopo: Predice cluster per nuovi punti usando modello HDBSCAN esistente
        
        Parametri input:
            - new_embeddings: Nuovi embeddings da classificare (n_samples, n_features)
            - fit_umap: Se True, applica UMAP ai nuovi embeddings (deve essere False per predizione incrementale)
            
        Output:
            - Tuple (labels, strengths): etichette cluster predette e confidenze
            
        Ultimo aggiornamento: 2025-08-28
        """
        trace_all(
            component="HDBSCANClusterer",
            action="ENTER",
            function="predict_new_points",
            message="Inizio predizione incrementale",
            details={
                "n_new_points": len(new_embeddings),
                "input_shape": new_embeddings.shape,
                "fit_umap": fit_umap,
                "has_trained_model": self.clusterer is not None,
                "umap_enabled": self.use_umap
            }
        )
        
        if self.clusterer is None:
            raise ValueError("Nessun modello HDBSCAN trained disponibile. Eseguire prima fit_predict().")
        
        if not hasattr(self.clusterer, 'prediction_data_'):
            raise ValueError("Modello HDBSCAN non ha prediction_data. Abilitare prediction_data=True durante il training.")
        
        print(f"🎯 PREDIZIONE INCREMENTALE - {len(new_embeddings)} nuovi punti...")
        print(f"   📏 Shape embeddings: {new_embeddings.shape}")
        print(f"   📊 Range embeddings: [{new_embeddings.min():.4f}, {new_embeddings.max():.4f}]")
        
        try:
            # STEP 1: Applica UMAP se necessario (con reducer esistente)
            embeddings_for_prediction = new_embeddings
            if self.use_umap and hasattr(self, 'umap_reducer') and self.umap_reducer is not None:
                if fit_umap:
                    # ATTENZIONE: questo dovrebbe essere usato solo in casi speciali
                    print(f"⚠️ ATTENZIONE: fit_umap=True può causare inconsistenze!")
                    embeddings_for_prediction, _ = self._apply_umap_reduction(new_embeddings, fit_new=True)
                else:
                    # Usa reducer esistente (modalità corretta per predizione incrementale)
                    print(f"🗂️ Applicazione UMAP con reducer esistente...")
                    embeddings_for_prediction = self.umap_reducer.transform(new_embeddings)
                    print(f"   📏 Shape post-UMAP: {embeddings_for_prediction.shape}")
            
            # STEP 2: Normalizzazione coerente con training
            embeddings_norm = embeddings_for_prediction
            metric_for_prediction = self.metric
            
            if self.metric == 'cosine':
                # Normalizza per metrica cosine (come nel training)
                embeddings_norm = embeddings_for_prediction / np.linalg.norm(embeddings_for_prediction, axis=1, keepdims=True)
                metric_for_prediction = 'euclidean'
                print(f"   🎯 Normalizzazione cosine applicata")
            
            # STEP 3: Predizione usando HDBSCAN approximate_predict
            print(f"🔮 Avvio predizione incrementale...")
            
            # Import della funzione prediction da hdbscan
            import hdbscan.prediction
            
            # Usa approximate_predict per assegnare nuovi punti ai cluster esistenti
            predicted_labels, prediction_strengths = hdbscan.prediction.approximate_predict(
                self.clusterer, embeddings_norm
            )
            
            print(f"✅ Predizione completata!")
            print(f"   🎯 Labels predette: {len(set(predicted_labels))} cluster unici")
            print(f"   📊 Outlier: {sum(1 for l in predicted_labels if l == -1)} punti")
            print(f"   💪 Strength media: {prediction_strengths.mean():.3f}")
            
            trace_all(
                component="HDBSCANClusterer",
                action="EXIT",
                function="predict_new_points",
                message="Predizione incrementale completata",
                details={
                    "n_predicted": len(predicted_labels),
                    "unique_clusters": len(set(predicted_labels)),
                    "n_outliers": sum(1 for l in predicted_labels if l == -1),
                    "mean_strength": float(prediction_strengths.mean()),
                    "umap_applied": hasattr(self, 'umap_reducer') and self.umap_reducer is not None
                },
                return_value=f"Predicted labels array with {len(predicted_labels)} predictions"
            )
            
            return predicted_labels, prediction_strengths
            
        except Exception as e:
            error_msg = f"Errore durante predizione incrementale: {str(e)}"
            print(f"❌ {error_msg}")
            
            trace_all(
                component="HDBSCANClusterer",
                action="EXIT",
                function="predict_new_points",
                message="Predizione incrementale fallita",
                details={
                    "error": str(e),
                    "n_new_points": len(new_embeddings)
                },
                exception=e
            )
            
            raise RuntimeError(error_msg)
    
    def save_model_for_incremental_prediction(self, model_path: str, tenant_id: str) -> bool:
        """
        Scopo: Salva modello HDBSCAN trained per riuso futuro nelle predizioni incrementali
        
        Parametri input:
            - model_path: Percorso dove salvare il modello
            - tenant_id: ID del tenant per identificazione modello
            
        Output:
            - True se salvato con successo, False altrimenti
            
        Ultimo aggiornamento: 2025-08-28
        """
        try:
            import pickle
            from datetime import datetime
            import os
            
            if self.clusterer is None:
                print("❌ Nessun modello da salvare")
                return False
            
            # Crea directory se non esistente
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Dati del modello da salvare
            model_data = {
                'clusterer': self.clusterer,  # HDBSCAN con prediction_data
                'umap_reducer': getattr(self, 'umap_reducer', None),  # Reducer UMAP se disponibile
                'parameters': {
                    'hdbscan': {
                        'min_cluster_size': self.min_cluster_size,
                        'min_samples': self.min_samples,
                        'metric': self.metric,
                        'cluster_selection_method': self.cluster_selection_method,
                        'alpha': self.alpha,
                        'cluster_selection_epsilon': self.cluster_selection_epsilon,
                        'allow_single_cluster': self.allow_single_cluster,
                        'max_cluster_size': self.max_cluster_size
                    },
                    'umap': {
                        'use_umap': self.use_umap,
                        'n_neighbors': self.umap_n_neighbors,
                        'min_dist': self.umap_min_dist,
                        'n_components': self.umap_n_components,
                        'metric': self.umap_metric,
                        'random_state': self.umap_random_state
                    } if self.use_umap else None
                },
                'embeddings_shape': getattr(self, 'last_embeddings_shape', None),
                'tenant_id': tenant_id,
                'timestamp': datetime.now().isoformat(),
                'gpu_used': getattr(self, 'gpu_used', False)
            }
            
            # Salva con pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ Modello HDBSCAN salvato: {model_path}")
            print(f"   🏷️ Tenant: {tenant_id}")
            print(f"   📅 Timestamp: {model_data['timestamp']}")
            print(f"   🗂️ UMAP incluso: {self.use_umap}")
            
            return True
            
        except Exception as e:
            print(f"❌ Errore salvataggio modello: {str(e)}")
            return False
    
    def load_model_for_incremental_prediction(self, model_path: str) -> bool:
        """
        Scopo: Carica modello HDBSCAN esistente per predizioni incrementali
        
        Parametri input:
            - model_path: Percorso del modello salvato
            
        Output:
            - True se caricato con successo, False altrimenti
            
        Ultimo aggiornamento: 2025-08-28
        """
        try:
            import pickle
            import os
            
            if not os.path.exists(model_path):
                print(f"❌ File modello non trovato: {model_path}")
                return False
            
            # Carica modello
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Ripristina clusterer
            self.clusterer = model_data['clusterer']
            
            # Ripristina reducer UMAP se disponibile
            if model_data.get('umap_reducer'):
                self.umap_reducer = model_data['umap_reducer']
            
            # Verifica compatibilità parametri
            saved_params = model_data['parameters']
            if not self._verify_parameter_compatibility(saved_params):
                print("⚠️ Parametri non compatibili, necessario retraining")
                return False
            
            print(f"✅ Modello HDBSCAN caricato: {model_path}")
            print(f"   🏷️ Tenant: {model_data.get('tenant_id', 'Unknown')}")
            print(f"   📅 Salvato: {model_data.get('timestamp', 'Unknown')}")
            print(f"   🗂️ UMAP: {bool(model_data.get('umap_reducer'))}")
            print(f"   🎯 GPU utilizzato: {model_data.get('gpu_used', False)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Errore caricamento modello: {str(e)}")
            return False
    
    def _verify_parameter_compatibility(self, saved_params: Dict) -> bool:
        """
        Scopo: Verifica se parametri attuali sono compatibili con modello salvato
        
        Parametri input:
            - saved_params: Parametri salvati nel modello
            
        Output:
            - True se compatibili, False se necessario retraining
            
        Ultimo aggiornamento: 2025-08-28
        """
        try:
            # Parametri HDBSCAN critici che richiedono retraining se cambiati
            critical_hdbscan_params = [
                'min_cluster_size', 'min_samples', 'metric', 
                'cluster_selection_method', 'alpha', 'cluster_selection_epsilon'
            ]
            
            saved_hdbscan = saved_params.get('hdbscan', {})
            
            for param in critical_hdbscan_params:
                current_value = getattr(self, param, None)
                saved_value = saved_hdbscan.get(param, None)
                
                if current_value != saved_value:
                    print(f"⚠️ Parametro HDBSCAN '{param}' cambiato: {saved_value} → {current_value}")
                    return False
            
            # Verifica parametri UMAP se utilizzato
            if self.use_umap and saved_params.get('umap'):
                critical_umap_params = ['n_neighbors', 'min_dist', 'n_components', 'metric']
                saved_umap = saved_params['umap']
                
                for param in critical_umap_params:
                    current_value = getattr(self, f'umap_{param}', None)
                    saved_value = saved_umap.get(param, None)
                    
                    if current_value != saved_value:
                        print(f"⚠️ Parametro UMAP '{param}' cambiato: {saved_value} → {current_value}")
                        return False
            
            print("✅ Parametri compatibili con modello salvato")
            return True
            
        except Exception as e:
            print(f"❌ Errore verifica compatibilità: {str(e)}")
            return False

    def _calculate_silhouette_score(self, embeddings: np.ndarray) -> float:
        """Calcola il silhouette score per valutare la qualità del clustering"""
        try:
            from sklearn.metrics import silhouette_score
            if len(set(self.cluster_labels)) > 1:
                # Esclude outlier dal calcolo
                mask = self.cluster_labels != -1
                if mask.sum() > 1:
                    return silhouette_score(embeddings[mask], self.cluster_labels[mask])
            return 0.0
        except:
            return 0.0
    
    def get_umap_debug_info(self) -> Dict:
        """
        Restituisce informazioni di debug dettagliate su UMAP
        
        Returns:
            Dizionario con informazioni di debug UMAP
            
        Data creazione: 27 Agosto 2025
        """
        if not hasattr(self, 'umap_info'):
            return {
                'error': 'Clustering non ancora eseguito - nessuna info UMAP disponibile',
                'available': False
            }
        
        debug_info = {
            'available': True,
            'configuration': {
                'use_umap': self.use_umap,
                'umap_available': UMAP_AVAILABLE,
                'parameters': {
                    'n_neighbors': self.umap_n_neighbors,
                    'min_dist': self.umap_min_dist,
                    'n_components': self.umap_n_components,
                    'metric': self.umap_metric,
                    'random_state': self.umap_random_state
                }
            },
            'execution': self.umap_info.copy()
        }
        
        # Calcola statistiche aggiuntive se UMAP è stato applicato
        if self.umap_info.get('applied'):
            input_shape = self.umap_info.get('input_shape')
            output_shape = self.umap_info.get('output_shape')
            if input_shape and output_shape:
                debug_info['statistics'] = {
                    'input_dimensions': input_shape[1],
                    'output_dimensions': output_shape[1],
                    'reduction_factor': input_shape[1] / output_shape[1],
                    'samples_processed': input_shape[0],
                    'dimensionality_reduction_percentage': (1 - output_shape[1] / input_shape[1]) * 100
                }
        
        return debug_info
    
    def print_umap_debug_summary(self) -> None:
        """
        Stampa un riepilogo completo delle informazioni di debug UMAP
        
        Data creazione: 27 Agosto 2025
        """
        print(f"\n" + "="*80)
        print(f"📋 RIEPILOGO DEBUG UMAP")
        print(f"="*80)
        
        debug_info = self.get_umap_debug_info()
        
        if not debug_info.get('available'):
            print(f"❌ {debug_info.get('error')}")
            return
        
        config = debug_info['configuration']
        execution = debug_info['execution']
        
        # Configurazione
        print(f"🔧 CONFIGURAZIONE:")
        print(f"   🔘 UMAP abilitato: {config['use_umap']}")
        print(f"   📚 Libreria disponibile: {config['umap_available']}")
        print(f"   ⚙️  Parametri:")
        for param, value in config['parameters'].items():
            print(f"     • {param}: {value}")
        
        # Esecuzione
        print(f"\n🚀 ESECUZIONE:")
        print(f"   ✅ UMAP applicato: {execution.get('applied', False)}")
        
        if execution.get('applied'):
            print(f"   ⏱️  Tempo esecuzione: {execution.get('reduction_time', 0):.2f}s")
            print(f"   📏 Shape input: {execution.get('input_shape')}")
            print(f"   📐 Shape output: {execution.get('output_shape')}")
            
            if 'statistics' in debug_info:
                stats = debug_info['statistics']
                print(f"   📊 Riduzione dimensionale: {stats['input_dimensions']} → {stats['output_dimensions']} "
                      f"({stats['dimensionality_reduction_percentage']:.1f}% riduzione)")
                print(f"   📈 Fattore riduzione: {stats['reduction_factor']:.1f}x")
                print(f"   🔢 Campioni processati: {stats['samples_processed']}")
        else:
            reason = execution.get('reason', 'Motivo non specificato')
            print(f"   ❌ Motivo non applicazione: {reason}")
            if execution.get('fallback'):
                print(f"   🔄 Utilizzati embeddings originali come fallback")
        
        print(f"="*80)
    
    def get_cluster_statistics(self) -> Dict:
        """
        Restituisce statistiche dettagliate sui cluster
        
        Returns:
            Dizionario con statistiche sui cluster
        """
        if self.cluster_labels is None:
            return {}
        
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        
        stats = {
            'n_clusters': len(unique_labels) - (1 if -1 in unique_labels else 0),
            'n_outliers': counts[unique_labels == -1][0] if -1 in unique_labels else 0,
            'cluster_sizes': {},
            'cluster_probabilities_avg': {},
            'total_samples': len(self.cluster_labels)
        }
        
        for label, count in zip(unique_labels, counts):
            if label != -1:  # Esclude outlier
                stats['cluster_sizes'][int(label)] = int(count)
                
                # Probabilità media per questo cluster
                mask = self.cluster_labels == label
                avg_prob = np.mean(self.cluster_probabilities[mask])
                stats['cluster_probabilities_avg'][int(label)] = float(avg_prob)
        
        return stats
    
    def get_gpu_parameter_support(self) -> Dict[str, bool]:
        """
        Restituisce informazioni sui parametri supportati in modalità GPU
        
        Returns:
            Dizionario con nome_parametro -> is_supported_on_gpu
            
        Ultima modifica: 26 Agosto 2025
        """
        gpu_support = {
            'min_cluster_size': True,
            'min_samples': True,
            'cluster_selection_epsilon': True,
            'metric': True,
            'cluster_selection_method': True,
            'alpha': True,
            'allow_single_cluster': True,
            # Parametri NON supportati da cuML
            'max_cluster_size': False,
            'leaf_size': False
        }
        
        return gpu_support
    
    def get_effective_parameters(self) -> Dict[str, any]:
        """
        Restituisce i parametri che saranno effettivamente utilizzati
        considerando le limitazioni GPU
        
        Returns:
            Dizionario con parametri effettivi
            
        Ultima modifica: 26 Agosto 2025
        """
        effective_params = {
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'cluster_selection_epsilon': self.cluster_selection_epsilon,
            'metric': self.metric,
            'cluster_selection_method': self.cluster_selection_method,
            'alpha': self.alpha,
            'allow_single_cluster': self.allow_single_cluster,
            'gpu_enabled': self.gpu_enabled
        }
        
        # Parametri che dipendono da GPU/CPU
        if self.gpu_enabled and GPU_AVAILABLE:
            # Su GPU questi parametri sono ignorati
            effective_params['max_cluster_size'] = None  # Ignorato su GPU
            effective_params['leaf_size'] = None  # Ignorato su GPU
            effective_params['note'] = 'GPU mode: max_cluster_size e leaf_size ignorati'
        else:
            # Su CPU tutti i parametri sono supportati
            effective_params['max_cluster_size'] = self.max_cluster_size if (self.max_cluster_size and self.max_cluster_size > 0) else None
            effective_params['leaf_size'] = self.leaf_size
            effective_params['note'] = 'CPU mode: tutti i parametri supportati'
        
        return effective_params
    
    def get_cluster_representatives(self, 
                                 embeddings: np.ndarray, 
                                 session_data: Dict[str, Dict],
                                 n_representatives: Optional[int] = None) -> Dict[int, List[Dict]]:
        """
        Trova i rappresentanti più centrali per ogni cluster
        
        Args:
            embeddings: Array di embedding
            session_data: Dati delle sessioni con session_id come chiave
            n_representatives: Numero di rappresentanti per cluster (usa config se None)
            
        Returns:
            Dizionario con cluster_id -> lista di rappresentanti
        """
        if self.cluster_labels is None:
            return {}
        
        # Usa parametro da config se non specificato
        if n_representatives is None:
            n_representatives = self.n_representatives
        
        print(f"🎯 Ricerca rappresentanti per ogni cluster...")
        
        representatives = {}
        session_ids = list(session_data.keys())
        
        unique_clusters = set(self.cluster_labels)
        unique_clusters.discard(-1)  # Rimuove outlier
        
        for cluster_id in unique_clusters:
            # Trova tutti i punti di questo cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_embeddings) == 0:
                continue
            
            # Calcola il centroide del cluster
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Trova i punti più vicini al centroide
            distances_to_centroid = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_indices = np.argsort(distances_to_centroid)[:n_representatives]
            
            # Recupera i dati delle sessioni rappresentative
            cluster_representatives = []
            for idx in closest_indices:
                original_idx = cluster_indices[idx]
                session_id = session_ids[original_idx]
                session_info = session_data[session_id].copy()
                session_info['distance_to_centroid'] = float(distances_to_centroid[idx])
                session_info['cluster_probability'] = float(self.cluster_probabilities[original_idx])
                cluster_representatives.append(session_info)
            
            representatives[int(cluster_id)] = cluster_representatives
        
        print(f"✅ Trovati rappresentanti per {len(representatives)} cluster")
        return representatives
    
    def visualize_clusters(self, 
                          embeddings: np.ndarray, 
                          session_data: Dict[str, Dict],
                          method: str = 'tsne',
                          save_path: Optional[str] = None) -> None:
        """
        Visualizza i cluster in 2D
        
        Args:
            embeddings: Array di embedding
            session_data: Dati delle sessioni
            method: Metodo di riduzione dimensionale ('tsne' o 'pca')
            save_path: Percorso per salvare il grafico
        """
        if self.cluster_labels is None:
            print("❌ Eseguire prima il clustering")
            return
        
        print(f"📊 Visualizzazione cluster con {method.upper()}...")
        
        # Riduzione dimensionale
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)
        
        coords_2d = reducer.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        unique_labels = set(self.cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Outlier in nero
                mask = self.cluster_labels == label
                plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                          c='black', marker='x', s=50, alpha=0.6, label='Outlier')
            else:
                mask = self.cluster_labels == label
                plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                          c=[color], s=60, alpha=0.8, label=f'Cluster {label}')
        
        plt.title(f'Visualizzazione Cluster ({method.upper()})')
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Grafico salvato in: {save_path}")
        
        plt.show()
    
    def suggest_cluster_labels(self, representatives: Dict[int, List[Dict]]) -> Dict[int, str]:
        """
        Suggerisce etichette per i cluster basandosi sui rappresentanti
        
        Args:
            representatives: Rappresentanti per ogni cluster
            
        Returns:
            Dizionario con cluster_id -> etichetta suggerita
        """
        print(f"🏷️  Generazione etichette suggerite per i cluster...")
        
        suggested_labels = {}
        
        # Pattern per riconoscere categorie comuni
        patterns = {
            'accesso_portale': [
                'accesso', 'login', 'password', 'portale', 'app', 'non riesco', 'errore'
            ],
            'prenotazione_esami': [
                'prenotare', 'prenotazione', 'esame', 'visita', 'appuntamento'
            ],
            'ritiro_referti': [
                'referto', 'referti', 'ritirare', 'risultati', 'esito'
            ],
            'problemi_tecnici': [
                'errore', 'bug', 'non funziona', 'problema tecnico', 'malfunzionamento'
            ],
            'fatturazione': [
                'fattura', 'pagamento', 'costo', 'prezzo', 'ricevuta'
            ],
            'informazioni_generali': [
                'informazioni', 'orari', 'dove', 'come', 'quando'
            ]
        }
        
        for cluster_id, cluster_reps in representatives.items():
            # Combina tutti i testi dei rappresentanti
            combined_text = ' '.join([rep['testo_completo'].lower() for rep in cluster_reps])
            
            # Conta le occorrenze per ogni categoria
            category_scores = {}
            for category, keywords in patterns.items():
                score = sum(combined_text.count(keyword) for keyword in keywords)
                if score > 0:
                    category_scores[category] = score
            
            # Scegli la categoria con il punteggio più alto
            if category_scores:
                best_category = max(category_scores, key=category_scores.get)
                suggested_labels[cluster_id] = best_category.replace('_', ' ').title()
            else:
                # Fallback: usa parole più frequenti
                words = combined_text.split()
                word_freq = Counter([w for w in words if len(w) > 3])
                if word_freq:
                    top_words = [word for word, _ in word_freq.most_common(3)]
                    suggested_labels[cluster_id] = f"Argomento: {', '.join(top_words)}"
                else:
                    suggested_labels[cluster_id] = f"Cluster {cluster_id}"
        
        print(f"✅ Etichette suggerite generate per {len(suggested_labels)} cluster")
        return suggested_labels

# Test del clustering
if __name__ == "__main__":
    print("=== TEST HDBSCAN CLUSTERING ===\n")
    
    # Inizializza componenti
    print("🔧 Caricamento embedder per test - SOLO Docker")
    from EmbeddingEngine.labse_remote_client import LaBSERemoteClient
    embedder = LaBSERemoteClient(
        service_url="http://localhost:8081",
        fallback_local=False  # 🚫 NESSUN FALLBACK LOCALE
    )
    print("✅ Embedder Docker remoto caricato per test")
    
    aggregator = SessionAggregator(schema='humanitas')
    clusterer = HDBSCANClusterer(min_cluster_size=3, min_samples=2)
    
    try:
        # Estrai sessioni
        print("📊 Estrazione sessioni per clustering...")
        sessioni = aggregator.estrai_sessioni_aggregate(limit=200)  # Più dati per clustering
        sessioni_filtrate = aggregator.filtra_sessioni_vuote(sessioni)
        
        if len(sessioni_filtrate) < 10:
            print("⚠️  Troppo poche sessioni per clustering significativo")
            exit()
        
        # Genera embedding
        testi = [dati['testo_completo'] for dati in sessioni_filtrate.values()]
        print(f"🔍 Generazione embedding per {len(testi)} sessioni...")
        embeddings = embedder.encode(testi, show_progress_bar=True)
        
        # Clustering
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Statistiche
        stats = clusterer.get_cluster_statistics()
        print(f"\n📊 STATISTICHE CLUSTERING:")
        print(f"  Cluster trovati: {stats['n_clusters']}")
        print(f"  Outlier: {stats['n_outliers']}")
        print(f"  Dimensioni cluster: {stats['cluster_sizes']}")
        
        # Rappresentanti
        representatives = clusterer.get_cluster_representatives(
            embeddings, sessioni_filtrate, n_representatives=2
        )
        
        # Etichette suggerite
        suggested_labels = clusterer.suggest_cluster_labels(representatives)
        
        # Mostra risultati
        print(f"\n🎯 CLUSTER E RAPPRESENTANTI:")
        print("=" * 80)
        
        for cluster_id, reps in representatives.items():
            label = suggested_labels.get(cluster_id, f"Cluster {cluster_id}")
            print(f"\n🏷️  CLUSTER {cluster_id}: {label}")
            print(f"📊 Dimensione: {stats['cluster_sizes'][cluster_id]} sessioni")
            print(f"🎯 Probabilità media: {stats['cluster_probabilities_avg'][cluster_id]:.3f}")
            
            for i, rep in enumerate(reps, 1):
                print(f"\n  📝 Rappresentante {i}:")
                print(f"    Sessione: {rep['session_id']}")
                print(f"    Distanza dal centro: {rep['distance_to_centroid']:.3f}")
                print(f"    Testo: {rep['testo_completo'][:150]}...")
                print(f"    {'─' * 40}")
        
    finally:
        aggregator.chiudi_connessione()
