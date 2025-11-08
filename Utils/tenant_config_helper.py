#!/usr/bin/env python3
"""
File: tenant_config_helper.py
Autore: GitHub Copilot & Valerio Bignardi
Data creazione: 2025-08-26
Ultima modifica: 2025-08-26

Helper per gestire parametri di configurazione per tenant,
incluso il parametro only_user per la lettura delle conversazioni.

Scopo:
- Centralizzare la lettura dei parametri per tenant
- Gestire fallback ai valori di default da config.yaml
- Supportare parametri personalizzati salvati nel database/file tenant
- Fornire interfaccia unificata per accesso ai parametri
"""

import os
import sys
import yaml
from typing import Any, Dict, Optional, Union
from datetime import datetime

# Import config_loader per caricare config.yaml con variabili ambiente
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config



class TenantConfigHelper:
    """
    Helper per gestire parametri di configurazione specifici per tenant.
    
    Gestisce:
    - Parametri di clustering personalizzati per tenant
    - Parametro only_user per la lettura conversazioni
    - Fallback ai valori default da config.yaml
    - Cache dei parametri per performance
    """
    
    def __init__(self):
        """
        Inizializza l'helper caricando la configurazione base
        """
        self.base_config = self._load_base_config()
        self.tenant_configs_cache = {}
        
    def _load_base_config(self) -> Dict[str, Any]:
        """
        Carica la configurazione base da config.yaml
        
        Returns:
            Dict con la configurazione base
        """
        try:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', 'config.yaml'
            )
            
            config = load_config()
                
            print(f"✅ [CONFIG] Configurazione base caricata da {config_path}")
            return config
            
        except Exception as e:
            print(f"❌ [CONFIG] Errore caricamento config.yaml: {e}")
            return {}
    
    def _load_tenant_config(self, tenant_id: str) -> Dict[str, Any]:
        """
        Carica la configurazione personalizzata per un tenant
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Dict con la configurazione del tenant
        """
        if tenant_id in self.tenant_configs_cache:
            return self.tenant_configs_cache[tenant_id]
        
        try:
            tenant_config_dir = os.path.join(
                os.path.dirname(__file__), '..', 'tenant_configs'
            )
            tenant_config_file = os.path.join(
                tenant_config_dir, f'{tenant_id}_clustering.yaml'
            )
            
            if os.path.exists(tenant_config_file):
                tenant_config = load_config()
                    
                clustering_params = tenant_config.get('clustering_parameters', {})
                
                print(f"✅ [CONFIG] Config personalizzata tenant {tenant_id}: "
                      f"{len(clustering_params)} parametri")
                
                # Cache per performance
                self.tenant_configs_cache[tenant_id] = clustering_params
                return clustering_params
                
            else:
                print(f"📋 [CONFIG] Nessuna config personalizzata per tenant {tenant_id}")
                return {}
                
        except Exception as e:
            print(f"⚠️ [CONFIG] Errore caricamento config tenant {tenant_id}: {e}")
            return {}
    
    def get_only_user_setting(self, tenant_id: str) -> bool:
        """
        Recupera il setting only_user per un tenant specifico.
        
        Ordine di priorità:
        1. Configurazione personalizzata del tenant (clustering parameters)
        2. Valore di default da config.yaml (False)
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            bool: True se only_user è attivo, False altrimenti
            
        Ultima modifica: 2025-08-26
        """
        try:
            # 1. Prova a leggere dalla config personalizzata del tenant
            tenant_config = self._load_tenant_config(tenant_id)
            
            if 'only_user' in tenant_config:
                value = tenant_config['only_user']
                print(f"🎯 [ONLY_USER] Tenant {tenant_id}: {value} (da config personalizzata)")
                return bool(value)
            
            # 2. Fallback al valore di default (False per retrocompatibilità)
            default_value = False
            print(f"📋 [ONLY_USER] Tenant {tenant_id}: {default_value} (default - nessuna config personalizzata)")
            return default_value
            
        except Exception as e:
            print(f"⚠️ [ONLY_USER] Errore per tenant {tenant_id}: {e}")
            print(f"🔄 [ONLY_USER] Usando valore default: False")
            return False
    
    def get_clustering_parameter(
        self, 
        tenant_id: str, 
        parameter_name: str, 
        default_value: Any = None
    ) -> Any:
        """
        Recupera un parametro di clustering per un tenant.
        
        Ordine di priorità:
        1. Configurazione personalizzata del tenant
        2. Configurazione base da config.yaml
        3. Valore di default fornito
        
        Args:
            tenant_id: ID del tenant
            parameter_name: Nome del parametro
            default_value: Valore di default se non trovato
            
        Returns:
            Any: Valore del parametro
            
        Ultima modifica: 2025-08-26
        """
        try:
            # 1. Prova config personalizzata tenant
            tenant_config = self._load_tenant_config(tenant_id)
            
            if parameter_name in tenant_config:
                value = tenant_config[parameter_name]
                print(f"🎯 [PARAM] {parameter_name} per tenant {tenant_id}: {value} (personalizzato)")
                return value
            
            # 2. Prova config base clustering
            base_clustering = self.base_config.get('clustering', {})
            if parameter_name in base_clustering:
                value = base_clustering[parameter_name]
                print(f"📋 [PARAM] {parameter_name} per tenant {tenant_id}: {value} (da config.yaml)")
                return value
            
            # 3. Usa default fornito
            print(f"🔄 [PARAM] {parameter_name} per tenant {tenant_id}: {default_value} (default)")
            return default_value
            
        except Exception as e:
            print(f"⚠️ [PARAM] Errore parametro {parameter_name} per tenant {tenant_id}: {e}")
            return default_value
    
    def get_umap_parameters(self, tenant_id: str) -> Dict[str, Any]:
        """
        Recupera tutti i parametri UMAP per un tenant dal database MySQL.
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Dict con i parametri UMAP: use_umap, n_neighbors, min_dist, etc.
            
        Data ultima modifica: 03/09/2025 - Valerio Bignardi
        """
        try:
            import mysql.connector
            from mysql.connector import Error
            
            db_config = self.base_config.get('tag_database', {})
            
            if not db_config:
                print(f"⚠️ [UMAP] Configurazione tag_database mancante, uso parametri default per {tenant_id}")
                return self._get_default_umap_parameters()
            
            # Connessione al database
            connection = mysql.connector.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database'],
                autocommit=True
            )
            
            cursor = connection.cursor(dictionary=True)
            
            # Query per recuperare parametri UMAP dall'ultimo record
            query = """
            SELECT 
                use_umap,
                umap_n_neighbors,
                umap_min_dist,
                umap_metric,
                umap_n_components,
                umap_random_state
            FROM soglie 
            WHERE tenant_id = %s 
            ORDER BY id DESC 
            LIMIT 1
            """
            
            cursor.execute(query, (tenant_id,))
            db_result = cursor.fetchone()
            
            if db_result:
                # Parametri UMAP dal database
                umap_params = {
                    'use_umap': bool(db_result['use_umap']),
                    'n_neighbors': db_result['umap_n_neighbors'],
                    'min_dist': float(db_result['umap_min_dist']),
                    'metric': db_result['umap_metric'],
                    'n_components': db_result['umap_n_components'],
                    'random_state': db_result['umap_random_state']
                }
                
                print(f"🗂️  [UMAP DB] Parametri personalizzati per tenant {tenant_id}:")
                print(f"   🔍 [DEBUG] Fonte: DATABASE MySQL")
                
            else:
                # Fallback a parametri default
                umap_params = self._get_default_umap_parameters()
                print(f"🗂️  [UMAP DB] Parametri default per tenant {tenant_id} (nessun record DB)")
                print(f"   🔍 [DEBUG] Fonte: CONFIG.YAML fallback")
            
            cursor.close()
            connection.close()
            
            # Debug logging completo
            if umap_params['use_umap']:
                print(f"   ✅ UMAP ABILITATO:")
                print(f"      n_neighbors: {umap_params['n_neighbors']}")
                print(f"      min_dist: {umap_params['min_dist']}")
                print(f"      n_components: {umap_params['n_components']}")
                print(f"      metric: {umap_params['metric']}")
                print(f"      random_state: {umap_params['random_state']}")
            else:
                print(f"   ❌ UMAP DISABILITATO per tenant {tenant_id}")
                
            return umap_params
            
        except Error as db_error:
            print(f"❌ [UMAP DB] Errore database per tenant {tenant_id}: {db_error}")
            print(f"   🔍 [DEBUG] Fallback a config.yaml")
            return self._get_default_umap_parameters()
            
        except Exception as e:
            print(f"❌ [UMAP DB] Errore generico per tenant {tenant_id}: {e}")
            print(f"   🔍 [DEBUG] Fallback a config.yaml")
            return self._get_default_umap_parameters()
            
    def _get_default_umap_parameters(self) -> Dict[str, Any]:
        """
        Restituisce parametri UMAP di default da config.yaml
        
        Returns:
            Dict con parametri UMAP default
        """
        bertopic_config = self.base_config.get('bertopic', {})
        umap_config = bertopic_config.get('umap_params', {})
        
        return {
            'use_umap': False,
            'n_neighbors': umap_config.get('n_neighbors', 15),
            'min_dist': umap_config.get('min_dist', 0.1),
            'metric': umap_config.get('metric', 'cosine'),
            'n_components': umap_config.get('n_components', 50),
            'random_state': 42
        }
    
    def invalidate_cache(self, tenant_id: str = None):
        """
        Invalida la cache dei parametri
        
        Args:
            tenant_id: ID del tenant specifico, None per invalidare tutto
            
        Ultima modifica: 2025-08-26
        """
        if tenant_id:
            if tenant_id in self.tenant_configs_cache:
                del self.tenant_configs_cache[tenant_id]
                print(f"🔄 [CACHE] Cache invalidata per tenant {tenant_id}")
        else:
            self.tenant_configs_cache.clear()
            print(f"🔄 [CACHE] Cache completamente invalidata")

    def get_hdbscan_parameters(self, tenant_id: str) -> Dict[str, Any]:
        """
        Recupera tutti i parametri HDBSCAN per un tenant dal database MySQL.
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Dict con i parametri HDBSCAN completi
            
        Data ultima modifica: 03/09/2025 - Valerio Bignardi
        """
        try:
            import mysql.connector
            from mysql.connector import Error
            
            db_config = self.base_config.get('tag_database', {})
            
            if not db_config:
                print(f"⚠️ [HDBSCAN] Configurazione tag_database mancante, uso parametri default per {tenant_id}")
                return self._get_default_hdbscan_parameters()
            
            # Connessione al database
            connection = mysql.connector.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database'],
                autocommit=True
            )
            
            cursor = connection.cursor(dictionary=True)
            
            # Query per recuperare parametri HDBSCAN dall'ultimo record
            query = """
            SELECT 
                min_cluster_size,
                min_samples,
                cluster_selection_epsilon,
                metric,
                cluster_selection_method,
                alpha,
                max_cluster_size,
                allow_single_cluster,
                only_user
            FROM soglie 
            WHERE tenant_id = %s 
            ORDER BY id DESC 
            LIMIT 1
            """
            
            cursor.execute(query, (tenant_id,))
            db_result = cursor.fetchone()
            
            if db_result:
                # Parametri HDBSCAN dal database
                hdbscan_params = {
                    'min_cluster_size': db_result['min_cluster_size'],
                    'min_samples': db_result['min_samples'],
                    'cluster_selection_epsilon': float(db_result['cluster_selection_epsilon']),
                    'metric': db_result['metric'],
                    'cluster_selection_method': db_result['cluster_selection_method'],
                    'alpha': float(db_result['alpha']),
                    'max_cluster_size': db_result['max_cluster_size'],
                    'allow_single_cluster': bool(db_result['allow_single_cluster']),
                    'only_user': bool(db_result['only_user'])
                }
                
                print(f"⚙️ [HDBSCAN DB] Parametri personalizzati per tenant {tenant_id}:")
                print(f"   🔍 [DEBUG] Fonte: DATABASE MySQL")
                
            else:
                # Fallback a parametri default
                hdbscan_params = self._get_default_hdbscan_parameters()
                print(f"⚙️ [HDBSCAN DB] Parametri default per tenant {tenant_id} (nessun record DB)")
                print(f"   🔍 [DEBUG] Fonte: CONFIG.YAML fallback")
            
            cursor.close()
            connection.close()
            
            # Debug logging completo
            print(f"   📊 PARAMETRI HDBSCAN:")
            print(f"      min_cluster_size: {hdbscan_params['min_cluster_size']}")
            print(f"      min_samples: {hdbscan_params['min_samples']}")
            print(f"      cluster_selection_epsilon: {hdbscan_params['cluster_selection_epsilon']}")
            print(f"      metric: {hdbscan_params['metric']}")
            print(f"      cluster_selection_method: {hdbscan_params['cluster_selection_method']}")
            print(f"      alpha: {hdbscan_params['alpha']}")
            print(f"      only_user: {hdbscan_params['only_user']}")
                
            return hdbscan_params
            
        except Error as db_error:
            print(f"❌ [HDBSCAN DB] Errore database per tenant {tenant_id}: {db_error}")
            print(f"   🔍 [DEBUG] Fallback a config.yaml")
            return self._get_default_hdbscan_parameters()
            
        except Exception as e:
            print(f"❌ [HDBSCAN DB] Errore generico per tenant {tenant_id}: {e}")
            print(f"   🔍 [DEBUG] Fallback a config.yaml")
            return self._get_default_hdbscan_parameters()
            
    def _get_default_hdbscan_parameters(self) -> Dict[str, Any]:
        """
        Restituisce parametri HDBSCAN di default da config.yaml
        
        Returns:
            Dict con parametri HDBSCAN default
        """
        clustering_config = self.base_config.get('clustering', {})
        bertopic_hdbscan = self.base_config.get('bertopic', {}).get('hdbscan_params', {})
        
        return {
            'min_cluster_size': clustering_config.get('min_cluster_size', 5),
            'min_samples': clustering_config.get('min_samples', 3),
            'cluster_selection_epsilon': 0.12,
            'metric': bertopic_hdbscan.get('metric', 'cosine'),
            'cluster_selection_method': 'leaf',
            'alpha': 0.8,
            'max_cluster_size': 0,
            'allow_single_cluster': False,
            'only_user': True
        }


# Istanza globale singleton per evitare reload ripetuti
_tenant_config_helper = None

def get_tenant_config_helper() -> TenantConfigHelper:
    """
    Restituisce l'istanza singleton del TenantConfigHelper
    
    Returns:
        TenantConfigHelper: Istanza singleton
    """
    global _tenant_config_helper
    if _tenant_config_helper is None:
        _tenant_config_helper = TenantConfigHelper()
    return _tenant_config_helper


# Funzioni di convenienza per accesso diretto
def get_only_user_for_tenant(tenant_id: str) -> bool:
    """
    Funzione di convenienza per ottenere il setting only_user per un tenant
    
    Args:
        tenant_id: ID del tenant
        
    Returns:
        bool: True se only_user è attivo
    """
    helper = get_tenant_config_helper()
    return helper.get_only_user_setting(tenant_id)


def get_clustering_param_for_tenant(
    tenant_id: str, 
    parameter_name: str, 
    default_value: Any = None
) -> Any:
    """
    Funzione di convenienza per ottenere un parametro di clustering per un tenant
    
    Args:
        tenant_id: ID del tenant
        parameter_name: Nome del parametro
        default_value: Valore di default
        
    Returns:
        Any: Valore del parametro
    """
    helper = get_tenant_config_helper()
    return helper.get_clustering_parameter(tenant_id, parameter_name, default_value)


def get_review_queue_thresholds_for_tenant(tenant_id: str) -> Dict[str, Any]:
    """
    Funzione di convenienza per ottenere le soglie Review Queue per un tenant dal database MySQL
    
    Args:
        tenant_id: ID del tenant
        
    Returns:
        Dict: Soglie review queue dal database o default da config.yaml come fallback
        
    Data ultima modifica: 2025-09-03 - Valerio Bignardi
    """
    try:
        import mysql.connector
        from mysql.connector import Error
        
        # Carica configurazione database
        helper = get_tenant_config_helper()
        config = helper.base_config  # Usa la configurazione già caricata
        db_config = config.get('tag_database', {})
        
        if not db_config:
            print(f"⚠️ [TENANT-CONFIG] Configurazione tag_database mancante, uso soglie default per {tenant_id}")
            return _get_default_review_thresholds()
        
        # Connessione al database
        connection = mysql.connector.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            autocommit=True
        )
        
        cursor = connection.cursor(dictionary=True)
        
        # Query per recuperare l'ultimo record per il tenant
        query = """
        SELECT 
            enable_smart_review,
            max_pending_per_batch,
            minimum_consensus_threshold,
            outlier_confidence_threshold,
            propagated_confidence_threshold,
            representative_confidence_threshold
        FROM soglie 
        WHERE tenant_id = %s 
        ORDER BY id DESC 
        LIMIT 1
        """
        
        cursor.execute(query, (tenant_id,))
        db_result = cursor.fetchone()
        
        if db_result:
            # Soglie dal database
            thresholds = {
                'outlier_confidence_threshold': float(db_result['outlier_confidence_threshold']),
                'propagated_confidence_threshold': float(db_result['propagated_confidence_threshold']),
                'representative_confidence_threshold': float(db_result['representative_confidence_threshold']),
                'minimum_consensus_threshold': db_result['minimum_consensus_threshold'],
                'enable_smart_review': bool(db_result['enable_smart_review']),
                'max_pending_per_batch': db_result['max_pending_per_batch']
            }
            
            print(f"📊 [TENANT-CONFIG] Soglie Review Queue personalizzate dal DB per tenant {tenant_id}")
            
            cursor.close()
            connection.close()
            return thresholds
        else:
            # Nessun record personalizzato trovato
            print(f"📊 [TENANT-CONFIG] Nessuna soglia personalizzata trovata per {tenant_id}, uso default")
            
            cursor.close()
            connection.close()
            return _get_default_review_thresholds()
            
    except Error as db_error:
        print(f"❌ [TENANT-CONFIG] Errore database per soglie tenant {tenant_id}: {db_error}")
        return _get_default_review_thresholds()
    except Exception as e:
        print(f"❌ [TENANT-CONFIG] Errore generico soglie tenant {tenant_id}: {e}")
        return _get_default_review_thresholds()


def _get_default_review_thresholds() -> Dict[str, Any]:
    """
    Restituisce le soglie Review Queue di default
    
    Returns:
        Dict: Soglie default
    """
    return {
        'outlier_confidence_threshold': 0.6,
        'propagated_confidence_threshold': 0.75,
        'representative_confidence_threshold': 0.85,
        'minimum_consensus_threshold': 2,
        'enable_smart_review': True,
        'max_pending_per_batch': 150
    }


# ============================================================
# NUOVE FUNZIONI DI CONVENIENZA PER PARAMETRI UNIFICATI
# ============================================================

def get_hdbscan_parameters_for_tenant(tenant_id: str) -> Dict[str, Any]:
    """
    Funzione di convenienza per ottenere parametri HDBSCAN per un tenant
    
    Args:
        tenant_id: ID del tenant
        
    Returns:
        Dict: Parametri HDBSCAN dal database o default da config.yaml
        
    Data ultima modifica: 2025-09-03 - Valerio Bignardi
    """
    helper = get_tenant_config_helper()
    return helper.get_hdbscan_parameters(tenant_id)


def get_umap_parameters_for_tenant(tenant_id: str) -> Dict[str, Any]:
    """
    Funzione di convenienza per ottenere parametri UMAP per un tenant
    
    Args:
        tenant_id: ID del tenant
        
    Returns:
        Dict: Parametri UMAP dal database o default da config.yaml
        
    Data ultima modifica: 2025-09-03 - Valerio Bignardi
    """
    helper = get_tenant_config_helper()
    return helper.get_umap_parameters(tenant_id)


def get_all_clustering_parameters_for_tenant(tenant_id: str) -> Dict[str, Any]:
    """
    Funzione di convenienza per ottenere TUTTI i parametri clustering per un tenant
    Include parametri HDBSCAN + UMAP + soglie Review Queue
    
    Args:
        tenant_id: ID del tenant
        
    Returns:
        Dict: Parametri completi dal database o default da config.yaml
        
    Data ultima modifica: 2025-09-03 - Valerio Bignardi
    """
    try:
        import mysql.connector
        from mysql.connector import Error
        
        helper = get_tenant_config_helper()
        db_config = helper.base_config.get('tag_database', {})
        
        if not db_config:
            print(f"⚠️ [CLUSTERING-ALL] Configurazione tag_database mancante per {tenant_id}")
            # Fallback separato
            hdbscan_params = helper.get_hdbscan_parameters(tenant_id)
            umap_params = helper.get_umap_parameters(tenant_id)
            review_thresholds = get_review_queue_thresholds_for_tenant(tenant_id)
            
            return {
                **hdbscan_params,
                **umap_params,
                **review_thresholds
            }
        
        # Query unificata per tutti i parametri
        connection = mysql.connector.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            autocommit=True
        )
        
        cursor = connection.cursor(dictionary=True)
        
        query = """
        SELECT *
        FROM soglie 
        WHERE tenant_id = %s 
        ORDER BY id DESC 
        LIMIT 1
        """
        
        cursor.execute(query, (tenant_id,))
        db_result = cursor.fetchone()
        
        if db_result:
            # Tutti i parametri dal database
            all_params = {
                # HDBSCAN
                'min_cluster_size': db_result['min_cluster_size'],
                'min_samples': db_result['min_samples'],
                'cluster_selection_epsilon': float(db_result['cluster_selection_epsilon']),
                'metric': db_result['metric'],
                'cluster_selection_method': db_result['cluster_selection_method'],
                'alpha': float(db_result['alpha']),
                'max_cluster_size': db_result['max_cluster_size'],
                'allow_single_cluster': bool(db_result['allow_single_cluster']),
                'only_user': bool(db_result['only_user']),
                
                # UMAP
                'use_umap': bool(db_result['use_umap']),
                'n_neighbors': db_result['umap_n_neighbors'],
                'min_dist': float(db_result['umap_min_dist']),
                'umap_metric': db_result['umap_metric'],
                'n_components': db_result['umap_n_components'],
                'random_state': db_result['umap_random_state'],
                
                # REVIEW QUEUE
                'outlier_confidence_threshold': float(db_result['outlier_confidence_threshold']),
                'propagated_confidence_threshold': float(db_result['propagated_confidence_threshold']),
                'representative_confidence_threshold': float(db_result['representative_confidence_threshold']),
                'minimum_consensus_threshold': db_result['minimum_consensus_threshold'],
                'enable_smart_review': bool(db_result['enable_smart_review']),
                'max_pending_per_batch': db_result['max_pending_per_batch']
            }
            
            print(f"🎯 [CLUSTERING-ALL DB] Parametri COMPLETI per tenant {tenant_id} dal database (record ID {db_result['id']})")
            print(f"   🔍 [DEBUG] Fonte: DATABASE MySQL")
            
        else:
            # Fallback completo a config.yaml
            hdbscan_params = helper.get_hdbscan_parameters(tenant_id)
            umap_params = helper.get_umap_parameters(tenant_id)
            review_thresholds = get_review_queue_thresholds_for_tenant(tenant_id)
            
            all_params = {
                **hdbscan_params,
                **umap_params,
                **review_thresholds
            }
            
            print(f"🎯 [CLUSTERING-ALL] Parametri COMPLETI per tenant {tenant_id} da config.yaml (fallback)")
            print(f"   🔍 [DEBUG] Fonte: CONFIG.YAML fallback")
        
        cursor.close()
        connection.close()
        
        return all_params
        
    except Error as db_error:
        print(f"❌ [CLUSTERING-ALL] Errore database per tenant {tenant_id}: {db_error}")
        # Fallback separato
        helper = get_tenant_config_helper()
        hdbscan_params = helper.get_hdbscan_parameters(tenant_id)
        umap_params = helper.get_umap_parameters(tenant_id)
        review_thresholds = get_review_queue_thresholds_for_tenant(tenant_id)
        
        return {
            **hdbscan_params,
            **umap_params,
            **review_thresholds
        }
        
    except Exception as e:
        print(f"❌ [CLUSTERING-ALL] Errore generico per tenant {tenant_id}: {e}")
        # Fallback separato
        helper = get_tenant_config_helper()
        hdbscan_params = helper.get_hdbscan_parameters(tenant_id)
        umap_params = helper.get_umap_parameters(tenant_id)
        review_thresholds = get_review_queue_thresholds_for_tenant(tenant_id)
        
        return {
            **hdbscan_params,
            **umap_params,
            **review_thresholds
        }
