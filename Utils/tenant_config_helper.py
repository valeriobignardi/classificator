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
            
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                
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
                with open(tenant_config_file, 'r', encoding='utf-8') as f:
                    tenant_config = yaml.safe_load(f)
                    
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
        Recupera tutti i parametri UMAP per un tenant.
        
        Returns:
            Dict con i parametri UMAP: use_umap, n_neighbors, min_dist, etc.
            
        Data creazione: 27 Agosto 2025
        """
        try:
            # Carica config personalizzata tenant
            tenant_config = self._load_tenant_config(tenant_id)
            
            # Parametri UMAP con defaults ottimizzati per clustering
            umap_params = {
                'use_umap': tenant_config.get('use_umap', False),
                'n_neighbors': tenant_config.get('umap_n_neighbors', 30),
                'min_dist': tenant_config.get('umap_min_dist', 0.1),
                'metric': tenant_config.get('umap_metric', 'cosine'),
                'n_components': tenant_config.get('umap_n_components', 50),
                'random_state': tenant_config.get('umap_random_state', 42)
            }
            
            if umap_params['use_umap']:
                print(f"🗂️  [UMAP] Parametri per tenant {tenant_id}:")
                print(f"   use_umap: {umap_params['use_umap']}")
                print(f"   n_neighbors: {umap_params['n_neighbors']}")
                print(f"   min_dist: {umap_params['min_dist']}")
                print(f"   n_components: {umap_params['n_components']}")
                print(f"   metric: {umap_params['metric']}")
            else:
                print(f"🗂️  [UMAP] Disabilitato per tenant {tenant_id}")
                
            return umap_params
            
        except Exception as e:
            print(f"⚠️ [UMAP] Errore parametri per tenant {tenant_id}: {e}")
            return {
                'use_umap': False,
                'n_neighbors': 30,
                'min_dist': 0.1,
                'metric': 'cosine',
                'n_components': 50,
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
