"""
LLM Factory - Factory per gestione dinamica modelli LLM per tenant
Implementa lo stesso pattern di EmbeddingEngineFactory per garantire
reload automatico quando la configurazione del tenant cambia.

Autore: GitHub Copilot
Data creazione: 26 Agosto 2025
Aggiornamenti:
- 26/08/2025: Implementazione factory pattern per LLM con cache invalidation
"""

import sys
import os
import threading
from typing import Dict, Any, Optional
import yaml
import subprocess
import requests
import json
import logging

# Import per AI Configuration Service
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AIConfiguration'))
from ai_configuration_service import AIConfigurationService

# Import per IntelligentClassifier
from Classification.intelligent_classifier import IntelligentClassifier


class LLMFactory:
    """
    Factory per creazione e gestione dinamica classificatori LLM per tenant
    
    Implementa lo stesso pattern di EmbeddingEngineFactory:
    1. Cache per tenant con invalidazione intelligente
    2. Reload automatico quando configurazione cambia
    3. Gestione tenant-specific dei modelli LLM
    4. Fallback a modelli default in caso di errore
    
    Scopo: Risolvere il bug di cambio modello LLM applicando
    la stessa logica funzionante del sistema embedding.
    
    Ultima modifica: 26 Agosto 2025
    """
    
    def __init__(self):
        """
        Inizializza LLM Factory con cache e servizi
        
        Ultima modifica: 26 Agosto 2025
        """
        # Cache per classificatori LLM (tenant_id -> IntelligentClassifier)
        self._llm_cache = {}
        
        # Cache per configurazioni (tenant_id -> config)
        self._config_cache = {}
        
        # Lock per thread safety
        self._cache_lock = threading.Lock()
        
        try:
            # Carica servizio di configurazione
            self.ai_config_service = AIConfigurationService()
        except Exception as e:
            self.logger.error(f"❌ Errore inizializzazione AIConfigurationService: {e}")
            raise
        
        print("🏭 LLM Factory inizializzato con cache invalidation intelligente")
    
    def get_llm_for_tenant(self, tenant_id: str, force_reload: bool = False) -> IntelligentClassifier:
        """
        Ottiene classificatore LLM configurato per il tenant
        
        ALGORITMO (identico a EmbeddingEngineFactory):
        1. Se force_reload=True, bypassa completamente la cache
        2. Ottieni configurazione corrente dal database
        3. Confronta con cache per rilevare cambiamenti
        4. Se configurazione cambiata, invalida e ricarica
        5. Altrimenti riusa classifier cached
        
        Args:
            tenant_id: ID del tenant
            force_reload: Forza ricaricamento classifier
            
        Returns:
            Istanza IntelligentClassifier configurata
            
        Raises:
            RuntimeError: Se classifier non disponibile
        """
        cache_key = f"{tenant_id}"
        
        with self._cache_lock:
            # CONFIGURAZIONE FORCE RELOAD: Bypassa cache se richiesto
            if force_reload:
                print(f"🔧 LLM FACTORY: Force reload richiesto per tenant {tenant_id} - bypasso COMPLETAMENTE la cache")
                print(f"🔍 LLM FACTORY DEBUG: Chiamata ai_config_service.get_tenant_configuration({tenant_id}, force_no_cache=True)...")
                try:
                    config = self.ai_config_service.get_tenant_configuration(tenant_id, force_no_cache=True)
                    print(f"✅ LLM FACTORY DEBUG: Config ottenuta con force_no_cache=True")
                    if config and config.get('llm_model'):
                        model_current = config['llm_model'].get('current', 'NONE')
                        print(f"🎯 LLM FACTORY DEBUG: Modello dal DB con force_no_cache = '{model_current}'")
                    else:
                        print(f"❌ LLM FACTORY DEBUG: Config o llm_model mancanti!")
                except Exception as e:
                    print(f"❌ LLM FACTORY DEBUG: ERRORE get_tenant_configuration con force_no_cache: {e}")
                    raise
            else:
                print(f"🔍 LLM FACTORY DEBUG: Chiamata ai_config_service.get_tenant_configuration({tenant_id}) normale...")
                config = self.ai_config_service.get_tenant_configuration(tenant_id)
                
                if config and config.get('llm_model'):
                    model_current = config['llm_model'].get('current', 'NONE')
                    print(f"🎯 LLM FACTORY DEBUG: Modello normale = '{model_current}'")
                else:
                    print(f"❌ LLM FACTORY DEBUG: Config normale o llm_model mancanti!")
                
            if not config or not config.get('llm_model'):
                raise RuntimeError(f"Configurazione modello LLM per tenant {tenant_id} non disponibile")

            llm_config = config['llm_model']
            model_name = llm_config['current']
            
            # CONTROLLO CACHE INVALIDATION: Verifica se configurazione è cambiata
            should_reload = force_reload
            
            # DEBUG: Mostra configurazione letta dal database
            if force_reload:
                print(f"🔍 LLM FACTORY DEBUG: Configurazione letta dal database per {tenant_id}:")
                print(f"   Modello corrente: {model_name}")
                print(f"   Config: {llm_config.get('config', {})}")
            
            if cache_key in self._config_cache:
                cached_config = self._config_cache[cache_key]
                current_config = llm_config.copy()
                
                # Confronta configurazioni per rilevare cambiamenti
                if (cached_config.get('current') != current_config.get('current') or 
                    cached_config.get('config') != current_config.get('config')):
                    print(f"🔄 Configurazione LLM cambiata per tenant {tenant_id}")
                    print(f"   Precedente: {cached_config.get('current')}")
                    print(f"   Corrente: {current_config.get('current')}")
                    should_reload = True
            else:
                # Prima volta per questo tenant
                should_reload = True
            
            # Usa cache se disponibile e configurazione non cambiata
            if not should_reload and cache_key in self._llm_cache:
                cached_classifier = self._llm_cache[cache_key]
                
                # Verifica se classifier è stato marcato come invalidato
                if hasattr(cached_classifier, '_cache_invalidated') and cached_classifier._cache_invalidated:
                    print(f"🔄 LLM Classifier marked as stale per tenant {tenant_id}, reload necessario")
                    should_reload = True
                else:
                    print(f"♻️  Uso LLM classifier cached per tenant {tenant_id}: {model_name}")
                    return cached_classifier
            
            print(f"🔧 Creazione LLM classifier {model_name} per tenant {tenant_id}")
            
            # GESTIONE INTELLIGENTE MEMORIA OLLAMA
            # Ottimizza memoria liberando il modello precedente
            old_model_name = None
            if cache_key in self._llm_cache:
                print(f"🗑️ Rimozione LLM classifier precedente per {tenant_id}")
                old_classifier = self._llm_cache[cache_key]
                old_model_name = getattr(old_classifier, 'model_name', 'unknown')
                print(f"🗑️ LLM FACTORY DEBUG: Rimuovo classifier precedente modello '{old_model_name}'")
                
                # OTTIMIZZAZIONE: Libera memoria Ollama del modello precedente
                if old_model_name and old_model_name != 'unknown':
                    self._optimize_ollama_memory(old_model_name, model_name)
                
                self._cleanup_classifier(old_classifier)
                del self._llm_cache[cache_key]
            
            # Crea nuovo classifier
            print(f"🚀 LLM FACTORY DEBUG: Avvio creazione nuovo classifier modello '{model_name}'")
            classifier = self._create_classifier(model_name, tenant_id, llm_config.get('config', {}))
            new_model_name = getattr(classifier, 'model_name', 'unknown')
            print(f"✅ LLM FACTORY DEBUG: Nuovo classifier creato: {new_model_name}")
            
            # Assicurati che il nuovo classifier non sia marcato come invalidato
            if hasattr(classifier, '_cache_invalidated'):
                classifier._cache_invalidated = False
            
            # Cache classifier e configurazione
            self._llm_cache[cache_key] = classifier
            self._config_cache[cache_key] = llm_config.copy()
            
            print(f"✅ LLM Classifier {model_name} ({new_model_name}) creato e cached per tenant {tenant_id}")
            return classifier
    
    def _create_classifier(self, model_name: str, tenant_id: str, config: Dict[str, Any]) -> IntelligentClassifier:
        """
        Crea nuovo IntelligentClassifier con modello specificato
        
        Args:
            model_name: Nome del modello LLM
            tenant_id: ID del tenant
            config: Configurazione aggiuntiva
            
        Returns:
            Istanza IntelligentClassifier configurata
        """
        try:
            # Determina il client_name dal tenant_id
            client_name = tenant_id
            
            # Crea classifier con modello specificato
            classifier = IntelligentClassifier(
                model_name=model_name,
                client_name=client_name,
                enable_cache=True,
                enable_logging=False  # Evita log eccessivi
            )
            
            return classifier
            
        except Exception as e:
            self.logger.error(f"❌ Errore creazione LLM classifier {model_name}: {e}")
            raise RuntimeError(f"Impossibile creare classifier {model_name}: {e}")
    
    def _optimize_ollama_memory(self, old_model: str, new_model: str):
        """
        Ottimizza memoria Ollama fermando il modello precedente e caricando quello nuovo
        
        STRATEGIA OTTIMIZZAZIONE:
        1. Se old_model è Ollama: ollama stop <old_model> per liberare memoria
        2. Se new_model è Ollama: ollama run <new_model> per pre-caricarlo
        
        Args:
            old_model: Nome del modello precedente
            new_model: Nome del nuovo modello
        """
        try:
            if old_model == new_model:
                print(f"🔄 Stesso modello ({old_model}), skip ottimizzazione")
                return
                
            # Usa il nuovo metodo di gestione Ollama
            optimization_success = self._manage_ollama_model_switch(old_model, new_model)
            
            if optimization_success:
                print(f"✅ Ottimizzazione Ollama completata: {old_model} -> {new_model}")
            else:
                print(f"⚠️ Ottimizzazione Ollama parziale o fallita")
                
        except Exception as e:
            print(f"❌ Errore ottimizzazione Ollama: {e}")
    
    def _cleanup_classifier(self, classifier: IntelligentClassifier):
        """
        Cleanup di un classifier non più utilizzato
        
        Args:
            classifier: Classifier da pulire
        """
        try:
            # Pulizia cache interna del classifier
            if hasattr(classifier, 'prediction_cache'):
                classifier.prediction_cache.clear()
            
            # Pulizia connessioni se presenti
            if hasattr(classifier, 'mysql_connector') and classifier.mysql_connector:
                try:
                    classifier.mysql_connector.disconnetti()
                except:
                    pass
            
            if hasattr(classifier, 'mongo_reader') and classifier.mongo_reader:
                try:
                    if hasattr(classifier.mongo_reader, 'close_connection'):
                        classifier.mongo_reader.close_connection()
                except:
                    pass
            
        except Exception as e:
            print(f"⚠️ Errore cleanup LLM classifier: {e}")
    
    def clear_cache(self, tenant_id: Optional[str] = None):
        """
        Pulisce cache LLM classifier
        
        Args:
            tenant_id: ID tenant specifico, None per tutti
        """
        with self._cache_lock:
            if tenant_id:
                cache_key = f"{tenant_id}"
                if cache_key in self._llm_cache:
                    self._cleanup_classifier(self._llm_cache[cache_key])
                    del self._llm_cache[cache_key]
                if cache_key in self._config_cache:
                    del self._config_cache[cache_key]
                print(f"🧹 Cache LLM classifier pulita per tenant {tenant_id}")
            else:
                for cache_key, classifier in self._llm_cache.items():
                    self._cleanup_classifier(classifier)
                self._llm_cache.clear()
                self._config_cache.clear()
                print(f"🧹 Cache LLM classifier completamente pulita")
    
    def reload_tenant_llm(self, tenant_id: str) -> IntelligentClassifier:
        """
        Ricarica classifier LLM per tenant (utile dopo cambio configurazione)
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Nuovo classifier configurato
        """
        print(f"🔄 Ricaricamento LLM classifier per tenant {tenant_id}")
        return self.get_llm_for_tenant(tenant_id, force_reload=True)
    
    def invalidate_tenant_cache(self, tenant_id: str):
        """
        Invalida cache per tenant specifico (da chiamare quando configurazione cambia)
        IMPORTANTE: Non distrugge il classifier se ancora in uso
        
        Args:
            tenant_id: ID del tenant
        """
        with self._cache_lock:
            cache_key = f"{tenant_id}"
            
            # Invalida sempre la cache configurazione per forzare reload
            if cache_key in self._config_cache:
                del self._config_cache[cache_key]
                print(f"🗑️ Cache configurazione LLM invalidata per tenant {tenant_id}")
            
            # Per il classifier, solo marca come "invalidato" ma non distruggere
            # Sarà il LLMManager a gestire il cleanup quando necessario
            if cache_key in self._llm_cache:
                # Marca il classifier come "stale" per forzare reload al prossimo get
                classifier = self._llm_cache[cache_key]
                if hasattr(classifier, '_cache_invalidated'):
                    classifier._cache_invalidated = True
                print(f"🏷️ LLM Classifier marcato per reload per tenant {tenant_id}")
            else:
                print(f"ℹ️  Nessuna cache da invalidare per tenant {tenant_id}")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        Ottiene stato corrente della cache
        
        Returns:
            Stato cache con dettagli
        """
        with self._cache_lock:
            cache_status = {}
            
            for tenant_id, classifier in self._llm_cache.items():
                model_name = getattr(classifier, 'model_name', 'unknown')
                is_invalidated = getattr(classifier, '_cache_invalidated', False)
                
                cache_status[tenant_id] = {
                    'model_name': model_name,
                    'is_available': classifier.is_available() if hasattr(classifier, 'is_available') else False,
                    'is_invalidated': is_invalidated,
                    'cache_key': f"{tenant_id}"
                }
            
            return {
                'cached_tenants': list(self._llm_cache.keys()),
                'cache_details': cache_status,
                'total_cached': len(self._llm_cache)
            }

    def _is_ollama_model(self, model_name: str) -> bool:
        """
        Verifica se un modello è di tipo Ollama
        
        Args:
            model_name: Nome del modello da verificare
            
        Returns:
            True se è un modello Ollama
        """
        # Lista modelli Ollama noti (aggiornata dinamicamente)
        ollama_models = ['mistral:7b', 'gpt-oss:20b', 'llama3.3:70b-instruct-q2_K']
        return model_name in ollama_models or ':' in model_name

    def _get_available_ollama_models(self) -> Dict[str, str]:
        """
        Ottiene lista modelli Ollama disponibili
        
        Returns:
            Dizionario con modelli disponibili
        """
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                models = response.json()
                available = {}
                for model in models.get('models', []):
                    name = model.get('name', 'unknown')
                    size = model.get('size', 0)
                    available[name] = {
                        'name': name,
                        'size_gb': round(size / 1_000_000_000, 1),
                        'size_bytes': size
                    }
                return available
            else:
                print(f"⚠️ Errore API Ollama: {response.status_code}")
                return {}
        except Exception as e:
            print(f"⚠️ Errore connessione Ollama: {e}")
            return {}

    def _manage_ollama_model_switch(self, old_model: str, new_model: str) -> bool:
        """
        Gestisce il cambio ottimizzato di modelli Ollama
        
        OTTIMIZZAZIONE IMPORTANTE:
        - Esegue 'ollama stop <old_model>' per liberare memoria
        - Esegue 'ollama run <new_model>' per caricare il nuovo modello
        - Solo se entrambi sono modelli Ollama
        
        Args:
            old_model: Modello precedente
            new_model: Nuovo modello da caricare
            
        Returns:
            True se ottimizzazione eseguita con successo
        """
        if not self._is_ollama_model(old_model) or not self._is_ollama_model(new_model):
            print(f"🔄 Modelli non Ollama, skip ottimizzazione: {old_model} -> {new_model}")
            return False

        success = True
        
        try:
            # 1. Stop del modello precedente per liberare memoria
            if old_model != new_model:  # Solo se diverso
                print(f"🛑 Ollama: Stop modello precedente '{old_model}'...")
                result = subprocess.run(['ollama', 'stop', old_model], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print(f"✅ Modello '{old_model}' fermato con successo")
                else:
                    print(f"⚠️ Warning stop modello '{old_model}': {result.stderr}")
                    success = False

            # 2. Avvio del nuovo modello
            print(f"🚀 Ollama: Avvio nuovo modello '{new_model}'...")
            # Uso 'ollama run' con un prompt minimale per pre-caricare il modello
            result = subprocess.run(['ollama', 'run', new_model, 'test'], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"✅ Modello '{new_model}' caricato e pronto")
            else:
                print(f"❌ Errore caricamento modello '{new_model}': {result.stderr}")
                success = False
                
        except subprocess.TimeoutExpired:
            print(f"⏱️ Timeout during Ollama optimization")
            success = False
        except Exception as e:
            print(f"❌ Errore ottimizzazione Ollama: {e}")
            success = False
            
        return success


# Istanza globale factory (singleton pattern)
llm_factory = LLMFactory()
