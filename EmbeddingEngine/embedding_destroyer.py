#!/usr/bin/env python3
"""
File: embedding_destroyer.py
Autore: GitHub Copilot
Data creazione: 2025-08-26
Descrizione: Destroyer per reset completo embedder system

FILOSOFIA: "Distruggere tutto e ricreare è più sicuro che patchare"

Storia:
2025-08-26 - Creazione destroyer per reset completo sistema embedding
"""

import gc
import logging
import torch
import sys
import os
from typing import Optional, Dict, Any
from threading import Lock

# Aggiunta percorsi per imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

class EmbeddingDestroyer:
    """
    Destroyer per reset completo del sistema embedding
    
    SCOPO: Quando l'engine cambia, DISTRUGGI TUTTO e riparti da zero
    - Pulisce GPU memory completamente
    - Reset totale cache
    - Kill di tutti gli embedder attivi
    - Restart "simulato" del sistema embedding
    
    BENEFICI:
    - Nessuna inconsistenza possibile
    - GPU memory sempre pulita
    - Logica semplice e robusta
    - Debug più facile
    
    Data ultima modifica: 2025-08-26
    """
    
    def __init__(self):
        """Inizializza destroyer"""
        self.logger = logging.getLogger(__name__)
        self._lock = Lock()
        
        print("💥 EmbeddingDestroyer inizializzato - Ready to destroy and rebuild!")
    
    def nuclear_reset(self, reason: str = "System reset"):
        """
        RESET NUCLEARE: Distrugge TUTTO il sistema embedding
        
        Equivale a riavviare il server ma solo per la parte embedding
        
        Args:
            reason: Motivo del reset (per logging)
        """
        with self._lock:
            print(f"💥 NUCLEAR RESET AVVIATO: {reason}")
            
            # Step 1: Destroy all singleton instances
            self._destroy_singletons()
            
            # Step 2: Clear GPU memory aggressively
            self._nuclear_gpu_cleanup()
            
            # Step 3: Reset Python imports cache
            self._reset_imports_cache()
            
            # Step 4: Force garbage collection
            self._force_garbage_collection()
            
            print("✅ NUCLEAR RESET COMPLETATO - Sistema embedding completamente pulito")
    
    def _destroy_singletons(self):
        """Distrugge tutte le istanze singleton"""
        print("🗑️  Distruggendo singleton instances...")
        
        try:
            # Reset EmbeddingManager
            from embedding_manager import embedding_manager
            if hasattr(embedding_manager, '_current_embedder'):
                self._brutal_embedder_kill(embedding_manager._current_embedder)
                embedding_manager._current_embedder = None
                embedding_manager._current_tenant_id = None
            
            # Reset istanza singleton
            from embedding_manager import EmbeddingManager
            if hasattr(EmbeddingManager, '_instance'):
                EmbeddingManager._instance = None
                
            print("🗑️  EmbeddingManager distrutto")
        except Exception as e:
            print(f"⚠️  Errore distruzione EmbeddingManager: {e}")
        
        try:
            # Reset EmbeddingEngineFactory
            from embedding_engine_factory import embedding_factory
            
            # Kill tutti gli embedder cached
            for tenant_id, embedder in embedding_factory._embedder_cache.items():
                print(f"🗑️  Uccidendo embedder cached per {tenant_id}: {type(embedder).__name__}")
                self._brutal_embedder_kill(embedder)
            
            # Clear cache
            embedding_factory._embedder_cache.clear()
            embedding_factory._config_cache.clear()
            
            # Reset istanza singleton
            from embedding_engine_factory import EmbeddingEngineFactory
            if hasattr(EmbeddingEngineFactory, '_instance'):
                EmbeddingEngineFactory._instance = None
                
            print("🗑️  EmbeddingEngineFactory distrutto")
        except Exception as e:
            print(f"⚠️  Errore distruzione EmbeddingEngineFactory: {e}")
    
    def _brutal_embedder_kill(self, embedder):
        """Uccide brutalmente un embedder rilasciando tutta la memoria"""
        if embedder is None:
            return
            
        embedder_type = type(embedder).__name__
        print(f"💀 BRUTAL KILL: {embedder_type}")
        
        try:
            # Kill model se presente
            if hasattr(embedder, 'model') and embedder.model is not None:
                print(f"💀 Killing model...")
                
                # Se è un modello torch, sposta su CPU prima di killare
                if hasattr(embedder.model, 'to'):
                    embedder.model.to('cpu')
                    print(f"💀 Spostato su CPU")
                
                # Delete model
                del embedder.model
                embedder.model = None
                print(f"💀 Model deleted")
            
            # Kill tokenizer se presente
            if hasattr(embedder, 'tokenizer') and embedder.tokenizer is not None:
                del embedder.tokenizer
                embedder.tokenizer = None
                print(f"💀 Tokenizer deleted")
            
            # Kill device reference
            if hasattr(embedder, 'device'):
                embedder.device = None
                print(f"💀 Device reference deleted")
            
            # Generic cleanup se disponibile
            if hasattr(embedder, 'cleanup'):
                embedder.cleanup()
                print(f"💀 Cleanup method called")
                
        except Exception as e:
            print(f"⚠️  Errore brutal kill {embedder_type}: {e}")
            # Continue anyway
    
    def _nuclear_gpu_cleanup(self):
        """Pulizia NUCLEARE della GPU"""
        print("🧹 GPU NUCLEAR CLEANUP...")
        
        try:
            # Clear CUDA cache multiple times
            if torch.cuda.is_available():
                print(f"🧹 GPU Memory prima cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
                
                # Multiple aggressive cleanups
                for i in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Wait for all operations to complete
                    print(f"🧹 GPU Cleanup pass {i+1}")
                
                print(f"🧹 GPU Memory dopo cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            else:
                print("🧹 CUDA non disponibile, skip GPU cleanup")
                
        except Exception as e:
            print(f"⚠️  Errore GPU cleanup: {e}")
    
    def _reset_imports_cache(self):
        """Reset della cache imports Python per forzare reimport"""
        print("🔄 Resetting imports cache...")
        
        # Moduli da forzare al reimport
        modules_to_reset = [
            'labse_embedder',
            'openai_embedder', 
            'bge_m3_embedder',
            'embedding_manager',
            'embedding_engine_factory'
        ]
        
        for module_name in modules_to_reset:
            if module_name in sys.modules:
                print(f"🔄 Removing {module_name} from sys.modules")
                del sys.modules[module_name]
    
    def _force_garbage_collection(self):
        """Garbage collection aggressivo"""
        print("🧹 AGGRESSIVE GARBAGE COLLECTION...")
        
        # Multiple GC passes
        for i in range(5):
            collected = gc.collect()
            print(f"🧹 GC Pass {i+1}: collected {collected} objects")
    
    def smart_reset_for_tenant(self, tenant_id: str, old_engine: str, new_engine: str):
        """
        Reset intelligente quando cambia engine per un tenant
        
        Se gli engine sono molto diversi (es. LaBSE -> OpenAI), fa nuclear reset
        Se sono simili (es. OpenAI small -> large), fa soft reset
        
        Args:
            tenant_id: ID tenant
            old_engine: Engine precedente
            new_engine: Nuovo engine
        """
        print(f"🎯 SMART RESET: {tenant_id} da '{old_engine}' a '{new_engine}'")
        
        # Determina se serve nuclear reset
        needs_nuclear = self._should_use_nuclear_reset(old_engine, new_engine)
        
        if needs_nuclear:
            print(f"💥 Engine change troppo drastico: NUCLEAR RESET necessario")
            self.nuclear_reset(f"Engine change {old_engine} -> {new_engine}")
        else:
            print(f"🔄 Engine change compatibile: SOFT RESET")
            self._soft_reset_for_tenant(tenant_id)
    
    def _should_use_nuclear_reset(self, old_engine: str, new_engine: str) -> bool:
        """Determina se serve nuclear reset basandosi sui tipi di engine"""
        
        # Gruppi di engine compatibili
        torch_models = {'labse', 'bge_m3'}
        api_models = {'openai_small', 'openai_large'}
        
        old_group = None
        new_group = None
        
        if old_engine in torch_models:
            old_group = 'torch'
        elif old_engine in api_models:
            old_group = 'api'
        
        if new_engine in torch_models:
            new_group = 'torch'
        elif new_engine in api_models:
            new_group = 'api'
        
        # Se cambio gruppo (torch <-> api), serve nuclear reset
        nuclear_needed = old_group != new_group
        
        print(f"🔍 Engine analysis: {old_engine}({old_group}) -> {new_engine}({new_group}), nuclear={nuclear_needed}")
        return nuclear_needed
    
    def _soft_reset_for_tenant(self, tenant_id: str):
        """Soft reset per tenant specifico (quando engine compatibili)"""
        print(f"🔄 SOFT RESET per tenant {tenant_id}")
        
        try:
            # Invalida solo cache per questo tenant
            from embedding_manager import embedding_manager
            embedding_manager.invalidate_cache_for_tenant(tenant_id)
            
            from embedding_engine_factory import embedding_factory
            embedding_factory.clear_cache(tenant_id)
            
            # Light GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"✅ SOFT RESET completato per tenant {tenant_id}")
            
        except Exception as e:
            print(f"⚠️  Errore soft reset, fallback a nuclear: {e}")
            self.nuclear_reset(f"Soft reset failed for {tenant_id}")


# Istanza globale
embedding_destroyer = EmbeddingDestroyer()
