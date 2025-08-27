#!/usr/bin/env python3
"""
File: embedding_engine_factory.py
Autore: GitHub Copilot
Data creazione: 2025-08-25
Descrizione: Factory per creazione dinamica degli embedding engines

Storia aggiornamenti:
2025-08-25 - Creazione factory per gestione dinamica embedding engines
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Union
from threading import Lock

# Aggiunta percorsi per imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'EmbeddingEngine'))
sys.path.append(os.path.join(current_dir, '..', 'AIConfiguration'))

from base_embedder import BaseEmbedder
from ai_configuration_service import AIConfigurationService


class EmbeddingEngineFactory:
    """
    Factory per creazione dinamica degli embedding engines
    
    Scopo: Centralizzare la creazione degli embedding engines basandosi
    sulla configurazione AI per ogni tenant, eliminando hardcode di LaBSE
    
    Funzionalità:
    - Creazione dinamica embedder per tenant
    - Cache degli embedder per performance
    - Memory management automatico
    - Supporto per tutti gli engine (LaBSE, BGE-M3, OpenAI)
    
    Data ultima modifica: 2025-08-25
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Singleton pattern per factory condivisa"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EmbeddingEngineFactory, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Inizializza factory con cache embedder"""
        if hasattr(self, '_initialized'):
            return
            
        self.logger = logging.getLogger(__name__)
        self.ai_config_service = AIConfigurationService()
        self._embedder_cache: Dict[str, BaseEmbedder] = {}
        self._config_cache: Dict[str, Dict] = {}  # Cache configurazioni per invalidation
        self._cache_lock = Lock()
        self._initialized = True
        
        print(f"🏭 EmbeddingEngineFactory inizializzata")
    
    def get_embedder_for_tenant(self, tenant_id: str, force_reload: bool = False) -> BaseEmbedder:
        """
        Ottiene embedder configurato per il tenant
        
        Args:
            tenant_id: ID del tenant
            force_reload: Forza ricaricamento embedder
            
        Returns:
            Istanza BaseEmbedder configurata
            
        Raises:
            RuntimeError: Se embedder non disponibile
        """
        cache_key = f"{tenant_id}"
        
        with self._cache_lock:
            # CONFIGURAZIONE FORCE RELOAD: Bypassa cache se richiesto
            if force_reload:
                print(f"🔧 FACTORY: Force reload richiesto per tenant {tenant_id} - bypasso COMPLETAMENTE la cache")
                print(f"🔍 FACTORY DEBUG: Chiamata ai_config_service.get_tenant_configuration({tenant_id}, force_no_cache=True)...")
                try:
                    config = self.ai_config_service.get_tenant_configuration(tenant_id, force_no_cache=True)
                    print(f"✅ FACTORY DEBUG: Config ottenuta con force_no_cache=True")
                    if config and config.get('embedding_engine'):
                        engine_current = config['embedding_engine'].get('current', 'NONE')
                        print(f"🎯 FACTORY DEBUG: Engine dal DB con force_no_cache = '{engine_current}'")
                    else:
                        print(f"❌ FACTORY DEBUG: Config o embedding_engine mancanti!")
                except Exception as e:
                    print(f"❌ FACTORY DEBUG: ERRORE get_tenant_configuration con force_no_cache: {e}")
                    raise
            else:
                print(f"🔍 FACTORY DEBUG: Chiamata ai_config_service.get_tenant_configuration({tenant_id}) normale...")
                config = self.ai_config_service.get_tenant_configuration(tenant_id)
                if config and config.get('embedding_engine'):
                    engine_current = config['embedding_engine'].get('current', 'NONE')
                    print(f"🎯 FACTORY DEBUG: Engine normale = '{engine_current}'")
                else:
                    print(f"❌ FACTORY DEBUG: Config normale o embedding_engine mancanti!")
                
            if not config or not config.get('embedding_engine'):
                raise RuntimeError(f"Configurazione embedding engine per tenant {tenant_id} non disponibile")

            embedding_config = config['embedding_engine']
            engine_type = embedding_config['current']
            
            # CONTROLLO CACHE INVALIDATION: Verifica se configurazione è cambiata
            should_reload = force_reload
            
            # DEBUG: Mostra configurazione letta dal database
            if force_reload:
                print(f"🔍 FACTORY DEBUG: Configurazione letta dal database per {tenant_id}:")
                print(f"   Engine corrente: {engine_type}")
                print(f"   Config: {embedding_config.get('config', {})}")
            
            if cache_key in self._config_cache:
                cached_config = self._config_cache[cache_key]
                current_config = embedding_config.copy()
                
                # Confronta configurazioni per rilevare cambiamenti
                if (cached_config.get('current') != current_config.get('current') or 
                    cached_config.get('config') != current_config.get('config')):
                    print(f"🔄 Configurazione embedder cambiata per tenant {tenant_id}")
                    print(f"   Precedente: {cached_config.get('current')}")
                    print(f"   Corrente: {current_config.get('current')}")
                    should_reload = True
            else:
                # Prima volta per questo tenant
                should_reload = True
            
            # Usa cache se disponibile e configurazione non cambiata
            if not should_reload and cache_key in self._embedder_cache:
                cached_embedder = self._embedder_cache[cache_key]
                
                # Verifica se embedder è stato marcato come invalidato
                if hasattr(cached_embedder, '_cache_invalidated') and cached_embedder._cache_invalidated:
                    print(f"🔄 Embedder marked as stale per tenant {tenant_id}, reload necessario")
                    should_reload = True
                else:
                    print(f"♻️  Uso embedder cached per tenant {tenant_id}: {engine_type}")
                    return cached_embedder
            
            print(f"🔧 Creazione embedder {engine_type} per tenant {tenant_id}")
            
            # Rimuovi old embedder se presente
            if cache_key in self._embedder_cache:
                print(f"🗑️ Rimozione embedder precedente per {tenant_id}")
                old_embedder = self._embedder_cache[cache_key]
                old_embedder_type = type(old_embedder).__name__
                print(f"🗑️ FACTORY DEBUG: Rimuovo embedder precedente tipo '{old_embedder_type}'")
                self._cleanup_embedder(old_embedder)
                del self._embedder_cache[cache_key]
            
            # Crea nuovo embedder
            print(f"🚀 FACTORY DEBUG: Avvio creazione nuovo embedder tipo '{engine_type}'")
            embedder = self._create_embedder(engine_type, embedding_config.get('config', {}))
            new_embedder_type = type(embedder).__name__
            print(f"✅ FACTORY DEBUG: Nuovo embedder creato: {new_embedder_type}")
            
            # Assicurati che il nuovo embedder non sia marcato come invalidato
            if hasattr(embedder, '_cache_invalidated'):
                embedder._cache_invalidated = False
            
            # Cache embedder e configurazione
            self._embedder_cache[cache_key] = embedder
            self._config_cache[cache_key] = embedding_config.copy()
            
            print(f"✅ Embedder {engine_type} ({new_embedder_type}) creato e cached per tenant {tenant_id}")
            return embedder
    
    def _create_embedder(self, engine_type: str, config: Dict[str, Any]) -> BaseEmbedder:
        """
        Crea embedder specifico per tipo
        
        Args:
            engine_type: Tipo di engine (labse, bge_m3, openai_large, openai_small)
            config: Configurazione specifica per l'engine
            
        Returns:
            Istanza BaseEmbedder
        """
        print(f"🚀 FACTORY DEBUG: Creazione embedder tipo '{engine_type}' con config: {config}")
        try:
            if engine_type == 'labse':
                from labse_embedder import LaBSEEmbedder
                print(f"✅ FACTORY DEBUG: Importo LaBSEEmbedder...")
                embedder = LaBSEEmbedder(**config)
                print(f"✅ FACTORY DEBUG: LaBSEEmbedder creato con successo")
                return embedder
                
            elif engine_type == 'bge_m3':
                from bge_m3_embedder import BGE_M3_Embedder
                print(f"✅ FACTORY DEBUG: Importo BGE_M3_Embedder...")
                embedder = BGE_M3_Embedder(**config)
                print(f"✅ FACTORY DEBUG: BGE_M3_Embedder creato con successo")
                return embedder
                
            elif engine_type in ['openai_large', 'openai_small']:
                from openai_embedder import OpenAIEmbedder
                print(f"✅ FACTORY DEBUG: Importo OpenAIEmbedder...")
                # Ottieni chiave API
                api_key = self.ai_config_service.openai_api_key
                if not api_key:
                    raise RuntimeError("API key OpenAI non configurata")
                
                model_name = 'text-embedding-3-large' if engine_type == 'openai_large' else 'text-embedding-3-small'
                print(f"🎯 FACTORY DEBUG: Creazione OpenAI embedder con model='{model_name}'")
                embedder = OpenAIEmbedder(api_key=api_key, model_name=model_name, **config)
                print(f"✅ FACTORY DEBUG: OpenAIEmbedder creato con successo")
                return embedder
                
            else:
                print(f"❌ FACTORY DEBUG: Engine type '{engine_type}' non supportato")
                raise ValueError(f"Engine type {engine_type} non supportato")
                
        except Exception as e:
            print(f"❌ FACTORY DEBUG: ERRORE creazione embedder {engine_type}: {e}")
            raise RuntimeError(f"Errore creazione embedder {engine_type}: {e}")
    
    def _cleanup_embedder(self, embedder: BaseEmbedder):
        """
        Cleanup risorse embedder (GPU/RAM)
        
        Args:
            embedder: Embedder da pulire
        """
        try:
            # Cleanup specifico per LaBSE
            if hasattr(embedder, 'model') and embedder.model is not None:
                if hasattr(embedder.model, 'to'):
                    embedder.model.to('cpu')  # Sposta su CPU
                del embedder.model
                embedder.model = None
                print(f"🧹 Cleanup memoria GPU per embedder")
            
            # Cleanup generico
            if hasattr(embedder, 'cleanup'):
                embedder.cleanup()
                
        except Exception as e:
            self.logger.warning(f"Errore cleanup embedder: {e}")
    
    def clear_cache(self, tenant_id: Optional[str] = None):
        """
        Pulisce cache embedder
        
        Args:
            tenant_id: ID tenant specifico, None per tutti
        """
        with self._cache_lock:
            if tenant_id:
                cache_key = f"{tenant_id}"
                if cache_key in self._embedder_cache:
                    self._cleanup_embedder(self._embedder_cache[cache_key])
                    del self._embedder_cache[cache_key]
                if cache_key in self._config_cache:
                    del self._config_cache[cache_key]
                print(f"🧹 Cache embedder pulita per tenant {tenant_id}")
            else:
                for cache_key, embedder in self._embedder_cache.items():
                    self._cleanup_embedder(embedder)
                self._embedder_cache.clear()
                self._config_cache.clear()
                print(f"🧹 Cache embedder completamente pulita")
    
    def get_default_embedder(self) -> BaseEmbedder:
        """
        Ottiene embedder di default (LaBSE) per compatibilità
        
        Returns:
            Embedder di default
        """
        from labse_embedder import LaBSEEmbedder
        return LaBSEEmbedder()
    
    def reload_tenant_embedder(self, tenant_id: str) -> BaseEmbedder:
        """
        Ricarica embedder per tenant (utile dopo cambio configurazione)
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Nuovo embedder configurato
        """
        print(f"🔄 Ricaricamento embedder per tenant {tenant_id}")
        return self.get_embedder_for_tenant(tenant_id, force_reload=True)
    
    def invalidate_tenant_cache(self, tenant_id: str):
        """
        Invalida cache per tenant specifico (da chiamare quando configurazione cambia)
        IMPORTANTE: Non distrugge l'embedder se ancora in uso dall'EmbeddingManager
        
        Args:
            tenant_id: ID del tenant
        """
        with self._cache_lock:
            cache_key = f"{tenant_id}"
            
            # Invalida sempre la cache configurazione per forzare reload
            if cache_key in self._config_cache:
                del self._config_cache[cache_key]
                print(f"🗑️ Cache configurazione invalidata per tenant {tenant_id}")
            
            # Per l'embedder, solo marca come "invalidato" ma non distruggere
            # Sarà l'EmbeddingManager a gestire il cleanup quando necessario
            if cache_key in self._embedder_cache:
                # Marca l'embedder come "stale" per forzare reload al prossimo get
                embedder = self._embedder_cache[cache_key]
                if hasattr(embedder, '_cache_invalidated'):
                    embedder._cache_invalidated = True
                print(f"🏷️ Embedder marcato per reload per tenant {tenant_id}")
                # NON eliminiamo dalla cache per evitare problemi con EmbeddingManager
            else:
                print(f"ℹ️  Nessuna cache da invalidare per tenant {tenant_id}")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        Stato della cache embedder
        
        Returns:
            Informazioni cache
        """
        with self._cache_lock:
            return {
                'cached_tenants': list(self._embedder_cache.keys()),
                'cache_size': len(self._embedder_cache),
                'embedders': {
                    key: type(embedder).__name__ 
                    for key, embedder in self._embedder_cache.items()
                },
                'cached_configs': {
                    key: config.get('current', 'unknown')
                    for key, config in self._config_cache.items()
                }
            }

    def _get_current_engine_type(self, tenant_id: str) -> Optional[str]:
        """
        Ottiene tipo engine corrente configurato per tenant dal database
        
        Args:
            tenant_id: UUID tenant
            
        Returns:
            Nome engine corrente o None se errore
            
        Data ultima modifica: 2025-08-26
        """
        try:
            config = self._ai_config_service.get_tenant_configuration(tenant_id, force_no_cache=True)
            if config and 'embedding' in config:
                engine_type = config['embedding'].get('engine', 'labse')
                print(f"🔍 FACTORY DEBUG: Engine corrente per {tenant_id} = '{engine_type}'")
                return engine_type
        except Exception as e:
            print(f"⚠️ FACTORY ERROR: Errore lettura engine per {tenant_id}: {e}")
        
        return None


# Istanza globale singleton
embedding_factory = EmbeddingEngineFactory()
