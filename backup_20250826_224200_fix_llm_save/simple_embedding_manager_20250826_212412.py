#!/usr/bin/env python3
"""
File: simple_embedding_manager.py
Autore: GitHub Copilot
Data creazione: 2025-08-26
Descrizione: EmbeddingManager SEMPLIFICATO con reset completo

FILOSOFIA: "Quando in dubbio, distruggi tutto e riparti"

Storia aggiornamenti:
2025-08-26 - Creazione manager semplificato con destroyer integration
"""

import os
import sys
import logging
from typing import Optional
from threading import Lock

# Aggiunta percorsi per imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from base_embedder import BaseEmbedder
from embedding_destroyer import embedding_destroyer


class SimpleEmbeddingManager:
    """
    EmbeddingManager SEMPLIFICATO
    
    PRINCIPI:
    - Un solo embedder attivo alla volta
    - Quando cambia configurazione -> DISTRUGGI TUTTO
    - Nessuna cache complessa
    - Logica lineare e semplice
    
    BENEFICI:
    - Zero inconsistenze possibili  
    - GPU memory sempre pulita
    - Debug facile
    - Codice comprensibile
    
    Data ultima modifica: 2025-08-26
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SimpleEmbeddingManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Inizializza manager semplificato"""
        if hasattr(self, '_initialized'):
            return
            
        self.logger = logging.getLogger(__name__)
        self._current_embedder: Optional[BaseEmbedder] = None
        self._current_tenant_id: Optional[str] = None
        self._current_engine_type: Optional[str] = None
        self._manager_lock = Lock()
        self._initialized = True
        
        print(f"🎯 SimpleEmbeddingManager inizializzato")
    
    def get_embedder_for_tenant(self, tenant_id: str) -> BaseEmbedder:
        """
        Ottiene embedder per tenant - LOGICA SEMPLIFICATA
        
        ALGORITMO:
        1. Se non ho embedder -> Crealo
        2. Se ho embedder per stesso tenant -> Controlla se config è cambiata
        3. Se config cambiata -> NUCLEAR RESET + Ricrea
        4. Altrimenti riusa esistente
        
        Args:
            tenant_id: ID tenant (slug o UUID)
            
        Returns:
            Embedder configurato per tenant
        """
        # Normalizza tenant ID
        normalized_tenant_id = self._normalize_tenant_id(tenant_id)
        
        with self._manager_lock:
            print(f"🎯 SIMPLE MANAGER: Richiesta embedder per tenant {normalized_tenant_id}")
            
            # Caso 1: Nessun embedder corrente
            if self._current_embedder is None:
                print(f"📥 Primo embedder - creazione per tenant {normalized_tenant_id}")
                return self._create_fresh_embedder(normalized_tenant_id)
            
            # Caso 2: Stesso tenant
            if self._current_tenant_id == normalized_tenant_id:
                # Controlla se configurazione è cambiata
                current_engine = self._get_current_engine_for_tenant(normalized_tenant_id)
                
                if current_engine == self._current_engine_type:
                    print(f"♻️  Riuso embedder esistente per {normalized_tenant_id}: {self._current_engine_type}")
                    return self._current_embedder
                else:
                    print(f"🔄 CONFIG CHANGED: {self._current_engine_type} -> {current_engine}")
                    # NUCLEAR RESET quando config cambia
                    embedding_destroyer.smart_reset_for_tenant(
                        normalized_tenant_id, 
                        self._current_engine_type or 'unknown', 
                        current_engine or 'unknown'
                    )
                    return self._create_fresh_embedder(normalized_tenant_id)
            
            # Caso 3: Tenant diverso
            else:
                print(f"🔄 TENANT SWITCH: {self._current_tenant_id} -> {normalized_tenant_id}")
                # Per sicurezza, nuclear reset quando cambio tenant
                # (evita mixed state tra tenant diversi)
                embedding_destroyer.nuclear_reset(f"Tenant switch to {normalized_tenant_id}")
                return self._create_fresh_embedder(normalized_tenant_id)
    
    def _create_fresh_embedder(self, tenant_id: str) -> BaseEmbedder:
        """
        Crea embedder completamente nuovo per tenant
        
        Args:
            tenant_id: UUID tenant
            
        Returns:
            Nuovo embedder
        """
        print(f"🚀 CREATING FRESH EMBEDDER per tenant {tenant_id}")
        
        try:
            # Reset stato interno
            self._current_embedder = None
            self._current_tenant_id = None
            self._current_engine_type = None
            
            # Ottieni configurazione corrente
            engine_type = self._get_current_engine_for_tenant(tenant_id)
            if not engine_type:
                engine_type = 'labse'  # Default fallback
            
            print(f"🎯 Engine configurato: {engine_type}")
            
            # Crea embedder usando factory (che ora è stato resettato)
            embedder = self._create_embedder_directly(engine_type)
            
            # Aggiorna stato
            self._current_embedder = embedder
            self._current_tenant_id = tenant_id
            self._current_engine_type = engine_type
            
            print(f"✅ FRESH EMBEDDER creato: {type(embedder).__name__} per tenant {tenant_id}")
            return embedder
            
        except Exception as e:
            print(f"❌ ERRORE creazione fresh embedder: {e}")
            # Fallback a LaBSE
            print(f"🔄 Fallback a LaBSE default")
            embedder = self._create_labse_default()
            self._current_embedder = embedder
            self._current_tenant_id = tenant_id
            self._current_engine_type = 'labse'
            return embedder
    
    def _create_embedder_directly(self, engine_type: str) -> BaseEmbedder:
        """
        Crea embedder direttamente (bypassa factory cache)
        
        Args:
            engine_type: Tipo engine
            
        Returns:
            Nuovo embedder
        """
        print(f"🔧 DIRECT CREATE: {engine_type}")
        
        if engine_type == 'labse':
            from labse_embedder import LaBSEEmbedder
            return LaBSEEmbedder()
            
        elif engine_type == 'bge_m3':
            from bge_m3_embedder import BGE_M3_Embedder
            return BGE_M3_Embedder()
            
        elif engine_type in ['openai_small', 'openai_large']:
            from openai_embedder import OpenAIEmbedder
            
            # Ottieni API key
            api_key = self._get_openai_api_key()
            if not api_key:
                raise RuntimeError("OpenAI API key non configurata")
            
            model_name = 'text-embedding-3-large' if engine_type == 'openai_large' else 'text-embedding-3-small'
            return OpenAIEmbedder(api_key=api_key, model_name=model_name)
            
        else:
            raise ValueError(f"Engine type {engine_type} non supportato")
    
    def _create_labse_default(self) -> BaseEmbedder:
        """Crea LaBSE default per fallback"""
        from labse_embedder import LaBSEEmbedder
        return LaBSEEmbedder()
    
    def _normalize_tenant_id(self, tenant_identifier: str) -> str:
        """Normalizza tenant ID (come prima)"""
        try:
            from Database.database_ai_config_service import DatabaseAIConfigService
            db_service = DatabaseAIConfigService()
            normalized_id = db_service._resolve_tenant_id(tenant_identifier)
            
            if normalized_id != tenant_identifier:
                print(f"🔄 SIMPLE MANAGER: Normalizzato '{tenant_identifier}' -> '{normalized_id}'")
            
            return normalized_id
            
        except Exception as e:
            print(f"⚠️ SIMPLE MANAGER: Errore normalizzazione '{tenant_identifier}': {e}")
            return tenant_identifier
    
    def _get_current_engine_for_tenant(self, tenant_id: str) -> Optional[str]:
        """Ottiene engine corrente dal database"""
        try:
            from AIConfiguration.ai_configuration_service import AIConfigurationService
            ai_service = AIConfigurationService()
            config = ai_service.get_tenant_configuration(tenant_id, force_no_cache=True)
            
            if config and config.get('embedding_engine'):
                engine = config['embedding_engine'].get('current', 'labse')
                print(f"🔍 SIMPLE MANAGER: Engine da DB per {tenant_id} = '{engine}'")
                return engine
                
        except Exception as e:
            print(f"⚠️ SIMPLE MANAGER: Errore lettura config per {tenant_id}: {e}")
        
        return None
    
    def _get_openai_api_key(self) -> Optional[str]:
        """Ottiene API key OpenAI"""
        try:
            from AIConfiguration.ai_configuration_service import AIConfigurationService
            ai_service = AIConfigurationService()
            return ai_service.openai_api_key
        except Exception as e:
            print(f"⚠️ Errore ottenimento OpenAI API key: {e}")
            return None
    
    def force_reset(self, reason: str = "Manual reset"):
        """
        Forza reset completo del sistema
        
        Args:
            reason: Motivo del reset
        """
        print(f"💥 FORCE RESET richiesto: {reason}")
        
        with self._manager_lock:
            # Reset stato interno
            self._current_embedder = None
            self._current_tenant_id = None
            self._current_engine_type = None
            
            # Nuclear reset
            embedding_destroyer.nuclear_reset(reason)
            
            print(f"✅ FORCE RESET completato")
    
    def get_status(self) -> dict:
        """Status corrente manager"""
        with self._manager_lock:
            return {
                'current_tenant': self._current_tenant_id,
                'current_engine': self._current_engine_type,
                'embedder_type': type(self._current_embedder).__name__ if self._current_embedder else None,
                'has_embedder': self._current_embedder is not None
            }


# Istanza globale
simple_embedding_manager = SimpleEmbeddingManager()
