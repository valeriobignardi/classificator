#!/usr/bin/env python3
"""
File: embedding_manager.py
Autore: GitHub Copilot
Data creazione: 2025-08-25
Descrizione: Manager centralizzato per embedder condivisi con memory management

Storia aggiornamenti:
2025-08-25 - Creazione manager per embedder condivisi sistema
"""

import os
import sys
import gc
import logging
from typing import Dict, Optional
from threading import Lock

# Aggiunta percorsi per imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'EmbeddingEngine'))

from base_embedder import BaseEmbedder
from embedding_engine_factory import embedding_factory


class EmbeddingManager:
    """
    Manager centralizzato per embedder condivisi
    
    Scopo: Sostituire get_shared_embedder() hardcodato con gestione dinamica
    basata su configurazione tenant, con memory management ottimizzato
    
    Funzionalità:
    - Embedder condiviso per applicazione
    - Memory management GPU/RAM
    - Switch dinamico embedder
    - Cleanup automatico
    
    Data ultima modifica: 2025-08-25
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EmbeddingManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Inizializza manager"""
        if hasattr(self, '_initialized'):
            return
            
        self.logger = logging.getLogger(__name__)
        self._current_embedder: Optional[BaseEmbedder] = None
        self._current_tenant_id: Optional[str] = None
        self._manager_lock = Lock()
        self._initialized = True
        
        print(f"🎯 EmbeddingManager inizializzato")
    
    def _normalize_tenant_id(self, tenant_identifier: str) -> str:
        """
        Normalizza tenant identifier (slug o UUID) in UUID
        
        Args:
            tenant_identifier: slug (es. 'wopta') o UUID
            
        Returns:
            UUID del tenant o identifier originale se non trovato
            
        Ultima modifica: 2025-08-25
        """
        try:
            from Database.database_ai_config_service import DatabaseAIConfigService
            db_service = DatabaseAIConfigService()
            normalized_id = db_service._resolve_tenant_id(tenant_identifier)
            
            if normalized_id != tenant_identifier:
                print(f"🔄 EMBEDDING MANAGER: Normalizzato '{tenant_identifier}' -> '{normalized_id}'")
            
            return normalized_id
            
        except Exception as e:
            print(f"⚠️ EMBEDDING MANAGER: Errore normalizzazione '{tenant_identifier}': {e}")
            return tenant_identifier
    
    def get_shared_embedder(self, tenant_id: str = "default") -> BaseEmbedder:
        """
        Ottiene embedder condiviso per applicazione
        
        Args:
            tenant_id: ID tenant per configurazione embedding engine (slug o UUID)
            
        Returns:
            Embedder condiviso configurato per tenant
        """
        import traceback
        stack_trace = ''.join(traceback.format_stack()[-3:-1])
        print(f"🔍 CHIAMATA get_shared_embedder(tenant_id='{tenant_id}') da:")
        print(f"   {stack_trace.strip()}")
        
        # NORMALIZZA SEMPRE A UUID
        normalized_tenant_id = self._normalize_tenant_id(tenant_id)
        
        with self._manager_lock:
            # Se stesso tenant, restituisce embedder esistente
            if (self._current_embedder is not None and 
                self._current_tenant_id == normalized_tenant_id):
                print(f"♻️  Riuso embedder esistente per tenant {normalized_tenant_id}: {type(self._current_embedder).__name__}")
                return self._current_embedder
            
            print(f"🔄 Switch embedder condiviso da {self._current_tenant_id} a {normalized_tenant_id}")
            
            # Cleanup embedder precedente se diverso tenant
            if (self._current_embedder is not None and 
                self._current_tenant_id != normalized_tenant_id):
                self._cleanup_current_embedder()
            
            # Ottieni nuovo embedder
            try:
                self._current_embedder = embedding_factory.get_embedder_for_tenant(normalized_tenant_id)
                self._current_tenant_id = normalized_tenant_id
                
                print(f"✅ Embedder condiviso aggiornato per tenant {normalized_tenant_id}: {type(self._current_embedder).__name__}")
                return self._current_embedder
                
            except Exception as e:
                self.logger.error(f"Errore ottenimento embedder per tenant {normalized_tenant_id}: {e}")
                # Fallback a default LaBSE
                print(f"🔄 Fallback a embedder default LaBSE")
                self._current_embedder = embedding_factory.get_default_embedder()
                self._current_tenant_id = "default"
                return self._current_embedder
    
    def _cleanup_current_embedder(self):
        """Cleanup embedder corrente"""
        if self._current_embedder is not None:
            try:
                print(f"🧹 Cleanup embedder condiviso precedente")
                
                # Memory cleanup specifico
                if hasattr(self._current_embedder, 'model') and self._current_embedder.model is not None:
                    if hasattr(self._current_embedder.model, 'to'):
                        self._current_embedder.model.to('cpu')
                    del self._current_embedder.model
                    self._current_embedder.model = None
                
                # Cleanup generico
                if hasattr(self._current_embedder, 'cleanup'):
                    self._current_embedder.cleanup()
                
                # Force garbage collection
                gc.collect()
                
                # Clear GPU cache se disponibile
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print(f"🧹 GPU cache pulita")
                except ImportError:
                    pass
                    
            except Exception as e:
                self.logger.warning(f"Errore cleanup embedder: {e}")
            finally:
                self._current_embedder = None
                self._current_tenant_id = None
    
    def switch_tenant_embedder(self, tenant_id: str, force_reload: bool = False) -> BaseEmbedder:
        """
        Forza switch a embedder specifico per tenant
        
        Args:
            tenant_id: Nuovo tenant ID (slug o UUID)
            force_reload: Se True, forza reload anche per stesso tenant
            
        Returns:
            Nuovo embedder configurato
        """
        # NORMALIZZA SEMPRE A UUID
        normalized_tenant_id = self._normalize_tenant_id(tenant_id)
        
        with self._manager_lock:
            print(f"🔄 Switch forzato embedder a tenant {tenant_id} -> {normalized_tenant_id} (force_reload={force_reload})")
            
            # Se force_reload=True, SEMPRE cleanup e ricarica
            if force_reload:
                print(f"🔄 Force reload: cleanup embedder corrente e ricarica configurazione")
                self._cleanup_current_embedder()
                
                # FORZA ANCHE LA FACTORY A RICARICARE BYPASSANDO LA SUA CACHE
                try:
                    print(f"🔧 FORCE RELOAD: ordino alla factory di bypassare completamente la cache")
                    self._current_embedder = embedding_factory.get_embedder_for_tenant(normalized_tenant_id, force_reload=True)
                    self._current_tenant_id = normalized_tenant_id
                    
                    print(f"✅ Embedder FORZATAMENTE ricaricato per tenant {normalized_tenant_id}: {type(self._current_embedder).__name__}")
                    return self._current_embedder
                    
                except Exception as e:
                    self.logger.error(f"Errore force reload embedder per tenant {normalized_tenant_id}: {e}")
                    # Fallback a default
                    print(f"🔄 Fallback a embedder default LaBSE dopo errore force reload")
                    self._current_embedder = embedding_factory.get_default_embedder()
                    self._current_tenant_id = "default"
                    return self._current_embedder
            
            # Cleanup current solo se tenant diverso
            elif self._current_tenant_id != normalized_tenant_id:
                self._cleanup_current_embedder()
                
            # Per stesso tenant senza force_reload, usa logica normale
            # Il factory gestirà il reload usando la logica di cache invalidation
            
            # Ottieni nuovo embedder (il factory userà configurazione aggiornata)
            return self.get_shared_embedder(normalized_tenant_id)
    
    def reload_embedder(self) -> BaseEmbedder:
        """
        Ricarica embedder corrente (utile dopo cambio configurazione)
        
        Returns:
            Embedder ricaricato
        """
        if self._current_tenant_id is not None:
            return self.switch_tenant_embedder(self._current_tenant_id)
        else:
            return self.get_shared_embedder()
    
    def get_current_status(self) -> Dict[str, any]:
        """
        Status corrente manager
        
        Returns:
            Informazioni status
        """
        with self._manager_lock:
            return {
                'current_tenant': self._current_tenant_id,
                'embedder_type': type(self._current_embedder).__name__ if self._current_embedder else None,
                'has_active_embedder': self._current_embedder is not None,
                'factory_cache': embedding_factory.get_cache_status()
            }
    
    def cleanup_all(self):
        """Cleanup completo manager"""
        with self._manager_lock:
            print(f"🧹 Cleanup completo EmbeddingManager")
            self._cleanup_current_embedder()
            embedding_factory.clear_cache()


# Istanza globale singleton
embedding_manager = EmbeddingManager()
