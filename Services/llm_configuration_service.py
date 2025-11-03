#!/usr/bin/env python3
"""
============================================================================
LLM Configuration Service
============================================================================

Autore: Valerio Bignardi
Data creazione: 2025-01-31
Ultima modifica: 2025-01-31

Descrizione:
    Servizio per la gestione centralizzata della configurazione LLM per tenant.
    Fornisce funzionalità di validazione, aggiornamento e gestione parametri
    specifici per modello con supporto hot-reload.

Funzionalità principali:
    - Gestione parametri LLM per tenant
    - Validazione parametri specifici per modello  
    - Hot-reload configurazione senza restart server
    - Cache intelligente per performance
    - Logging dettagliato operazioni

============================================================================
"""

import os
import yaml
import time
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import threading
from Classification.intelligent_classifier import IntelligentClassifier


class LLMConfigurationService:
    """
    Servizio centralizzato per gestione configurazione LLM tenant
    
    Scopo:
        Fornisce interfaccia unificata per gestione parametri LLM per tenant
        con validazione, caching e hot-reload
        
    Parametri input:
        config_path: Percorso file configurazione YAML
        
    Parametri output:
        Istanza servizio configurato
        
    Data ultima modifica: 2025-01-31
    """
    
    def __init__(self, config_path: str = None):
        """
        Inizializza servizio configurazione LLM
        
        Args:
            config_path: Percorso file config.yaml (default: auto-detect)
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'config.yaml'
        )
        self.config_cache = {}
        self.cache_timestamp = 0
        self.cache_lock = threading.Lock()
        self.reload_enabled = True
        
        # Cache per modelli disponibili (evita chiamate ripetitive)
        self.models_cache = {}
        self.models_cache_timestamp = 0
        self.models_cache_ttl = 300  # 5 minuti TTL
        
        # Carica configurazione iniziale
        self._reload_config()
        
        print(f"🔧 [LLMConfigService] Servizio inizializzato con config: {self.config_path}")
    
    
    def _reload_config(self) -> bool:
        """
        Ricarica configurazione da file YAML
        
        Returns:
            True se ricaricamento riuscito, False altrimenti
            
        Data ultima modifica: 2025-01-31
        """
        try:
            with self.cache_lock:
                if not os.path.exists(self.config_path):
                    print(f"⚠️ [LLMConfigService] Config file non trovato: {self.config_path}")
                    return False
                
                file_timestamp = os.path.getmtime(self.config_path)
                
                # Ricarica solo se file modificato
                if file_timestamp > self.cache_timestamp:
                    with open(self.config_path, 'r', encoding='utf-8') as file:
                        self.config_cache = yaml.safe_load(file)
                    
                    self.cache_timestamp = file_timestamp
                    print(f"🔄 [LLMConfigService] Configurazione ricaricata")
                    return True
                
                return True  # Cache valida
                
        except Exception as e:
            print(f"❌ [LLMConfigService] Errore ricaricamento config: {e}")
            return False
    
    
    def get_available_models(self, tenant_id: str = None) -> List[Dict[str, Any]]:
        """
        Recupera lista modelli LLM disponibili con informazioni complete
        CON CACHE OTTIMIZZATA per ridurre chiamate ripetitive
        
        Args:
            tenant_id: ID tenant (per future estensioni)
            
        Returns:
            Lista dizionari con info modelli
            
        Data ultima modifica: 2025-01-31
        """
        try:
            current_time = time.time()
            cache_key = f"models_{tenant_id or 'global'}"
            
            # Controlla cache (TTL 5 minuti)
            with self.cache_lock:
                if (cache_key in self.models_cache and 
                    (current_time - self.models_cache_timestamp) < self.models_cache_ttl):
                    cached_models = self.models_cache[cache_key]
                    if len(cached_models) > 0:  # Cache hit con dati
                        print(f"♻️  [LLMConfigService] Uso cache modelli: {len(cached_models)} modelli (età: {int(current_time - self.models_cache_timestamp)}s)")
                        return cached_models
            
            # Cache miss: ricarica da configurazione
            if self.reload_enabled:
                self._reload_config()
            
            llm_config = self.config_cache.get('llm', {})
            models_config = llm_config.get('models', {})
            available_models = models_config.get('available', [])
            
            models_info = []
            for model in available_models:
                if isinstance(model, dict):
                    # Modello con configurazione completa
                    model_info = {
                        'name': model.get('name'),
                        'display_name': model.get('display_name', model.get('name')),
                        'provider': model.get('provider', 'ollama'),
                        'max_input_tokens': model.get('max_input_tokens', 8000),
                        'max_output_tokens': model.get('max_output_tokens', 4000),
                        'context_limit': model.get('context_limit', 8192),
                        'requires_raw_mode': model.get('requires_raw_mode', False),
                        'parallel_calls_max': model.get('parallel_calls_max'),
                        'rate_limit_per_minute': model.get('rate_limit_per_minute'),
                        'rate_limit_per_day': model.get('rate_limit_per_day')
                    }
                    models_info.append(model_info)
                else:
                    # Backward compatibility - modello come stringa
                    models_info.append({
                        'name': model,
                        'display_name': model,
                        'provider': 'ollama',
                        'max_input_tokens': 8000,
                        'max_output_tokens': 4000,
                        'context_limit': 8192,
                        'requires_raw_mode': True if 'mistral:7b' in model else False
                    })
            
            # Salva in cache
            with self.cache_lock:
                self.models_cache[cache_key] = models_info
                self.models_cache_timestamp = current_time
            
            print(f"📋 [LLMConfigService] Modelli ricaricati e cached: {len(models_info)}")
            return models_info
            
        except Exception as e:
            print(f"❌ [LLMConfigService] Errore recupero modelli: {e}")
            return []
    
    
    def get_tenant_parameters(self, tenant_id: str) -> Dict[str, Any]:
        """
        Recupera parametri LLM per un tenant specifico
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Dizionario con parametri LLM e metadati
            
        Data ultima modifica: 2025-01-31
        """
        try:
            if self.reload_enabled:
                self._reload_config()
            
            # Usa IntelligentClassifier per caricamento consistente
            classifier = IntelligentClassifier(
                client_name=tenant_id,
                enable_logging=False
            )
            
            tenant_config = classifier.load_tenant_llm_config(tenant_id)
            
            return {
                'tenant_id': tenant_id,
                'parameters': tenant_config,
                'source': tenant_config.get('source', 'default'),
                'last_modified': self._get_tenant_last_modified(tenant_id),
                'current_model': classifier.model_name
            }
            
        except Exception as e:
            print(f"❌ [LLMConfigService] Errore recupero parametri tenant {tenant_id}: {e}")
            return {
                'tenant_id': tenant_id,
                'parameters': {},
                'source': 'error',
                'error': str(e)
            }
    
    
    def update_tenant_parameters(
        self, 
        tenant_id: str, 
        parameters: Dict[str, Any],
        model_name: str = None
    ) -> Dict[str, Any]:
        """
        Aggiorna parametri LLM per un tenant
        
        Args:
            tenant_id: ID del tenant
            parameters: Nuovi parametri LLM
            model_name: Nome modello per validazione
            
        Returns:
            Dizionario con risultato operazione
            
        Data ultima modifica: 2025-01-31
        """
        try:
            # Validazione parametri
            validation_result = self.validate_parameters(parameters, model_name)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': f"Parametri non validi: {validation_result['errors']}",
                    'tenant_id': tenant_id
                }
            
            # Backup configurazione corrente
            self._create_config_backup()
            
            # Aggiorna configurazione
            with self.cache_lock:
                if 'tenant_configs' not in self.config_cache:
                    self.config_cache['tenant_configs'] = {}
                
                if tenant_id not in self.config_cache['tenant_configs']:
                    self.config_cache['tenant_configs'][tenant_id] = {}
                
                self.config_cache['tenant_configs'][tenant_id]['llm_parameters'] = parameters
                self.config_cache['tenant_configs'][tenant_id]['last_modified'] = datetime.now().isoformat()
                
                # Salva su file
                with open(self.config_path, 'w', encoding='utf-8') as file:
                    yaml.dump(self.config_cache, file, default_flow_style=False, sort_keys=False)
                
                self.cache_timestamp = os.path.getmtime(self.config_path)
            
            print(f"💾 [LLMConfigService] Parametri aggiornati per tenant {tenant_id}")
            
            return {
                'success': True,
                'tenant_id': tenant_id,
                'message': 'Parametri aggiornati con successo',
                'parameters': parameters,
                'validation': validation_result
            }
            
        except Exception as e:
            error_msg = f'Errore aggiornamento parametri: {str(e)}'
            print(f"❌ [LLMConfigService] {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'tenant_id': tenant_id
            }
    
    
    def reset_tenant_parameters(self, tenant_id: str) -> Dict[str, Any]:
        """
        Ripristina parametri default per un tenant
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Dizionario con risultato operazione
            
        Data ultima modifica: 2025-01-31
        """
        try:
            # Backup configurazione corrente
            self._create_config_backup()
            
            with self.cache_lock:
                # Rimuovi configurazione tenant specifica
                if 'tenant_configs' in self.config_cache and tenant_id in self.config_cache['tenant_configs']:
                    if 'llm_parameters' in self.config_cache['tenant_configs'][tenant_id]:
                        del self.config_cache['tenant_configs'][tenant_id]['llm_parameters']
                    
                    # Se tenant config è vuoto, rimuovi completamente
                    if not self.config_cache['tenant_configs'][tenant_id]:
                        del self.config_cache['tenant_configs'][tenant_id]
                
                # Salva configurazione
                with open(self.config_path, 'w', encoding='utf-8') as file:
                    yaml.dump(self.config_cache, file, default_flow_style=False, sort_keys=False)
                
                self.cache_timestamp = os.path.getmtime(self.config_path)
            
            # Recupera parametri default
            default_params = self.get_tenant_parameters(tenant_id)
            
            print(f"🔄 [LLMConfigService] Parametri resettati per tenant {tenant_id}")
            
            return {
                'success': True,
                'tenant_id': tenant_id,
                'message': 'Parametri resettati ai valori di default',
                'default_parameters': default_params['parameters']
            }
            
        except Exception as e:
            error_msg = f'Errore reset parametri: {str(e)}'
            print(f"❌ [LLMConfigService] {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'tenant_id': tenant_id
            }
    
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Recupera informazioni specifiche per un modello
        Cerca sia nei modelli configurati che in quelli Ollama disponibili
        
        Args:
            model_name: Nome del modello
            
        Returns:
            Dizionario con info modello o None se non trovato
            
        Data ultima modifica: 2025-09-01
        """
        try:
            print(f"🔍 [DEBUG] get_model_info cercando: {model_name}")
            
            # Prima cerca nei modelli configurati
            available_models = self.get_available_models()
            print(f"🔍 [DEBUG] Modelli configurati: {[m.get('name', 'NO_NAME') for m in available_models]}")
            
            for model in available_models:
                if model.get('name') == model_name:
                    print(f"✅ [DEBUG] Modello {model_name} trovato nei configurati")
                    return model
            
            # Se non trovato, cerca nei modelli Ollama via AIConfigurationService
            print(f"🔍 [DEBUG] Modello {model_name} non trovato nei configurati, cercando in Ollama...")
            try:
                # Chiama direttamente l'endpoint esistente che funziona
                import requests
                dummy_tenant = "015007d9-d413-11ef-86a5-96000228e7fe"
                response = requests.get(f"http://localhost:5000/api/ai-config/{dummy_tenant}/llm-models", timeout=10)
                
                if response.status_code == 200:
                    models_data = response.json()
                    print(f"🔍 [DEBUG] Risposta endpoint: {models_data.get('success', False)}")
                    
                    if models_data.get('success') and 'models' in models_data:
                        ollama_models = models_data['models'].get('models', {}).get('ollama_available', [])
                        print(f"🔍 [DEBUG] Modelli Ollama trovati: {[m.get('name', 'NO_NAME') for m in ollama_models]}")
                        
                        for model in ollama_models:
                            if model.get('name') == model_name:
                                print(f"✅ [DEBUG] Modello {model_name} trovato in Ollama")
                                # Standardizza la struttura per compatibilità
                                return {
                                    'name': model['name'],
                                    'display_name': model['name'],
                                    'description': model.get('description', f'Modello Ollama: {model["name"]}'),
                                    'max_input_tokens': 8000,  # Default per modelli Ollama
                                    'max_output_tokens': 4000,
                                    'context_limit': 8192,
                                    'requires_raw_mode': 'mistral:7b' in model['name'],
                                    'installed': model.get('installed', False),
                                    'size': model.get('size', 'Unknown'),
                                    'category': model.get('category', 'unknown'),
                                    'source': 'ollama'
                                }
                else:
                    print(f"❌ [DEBUG] Errore endpoint: {response.status_code}")
            except Exception as ollama_error:
                print(f"⚠️ [LLMConfigService] Errore ricerca Ollama per {model_name}: {ollama_error}")
            
            print(f"⚠️ [LLMConfigService] Modello {model_name} non trovato né in config né in Ollama")
            return None
            
        except Exception as e:
            print(f"❌ [LLMConfigService] Errore recupero info modello {model_name}: {e}")
            return None
    
    
    def validate_parameters(
        self, 
        parameters: Dict[str, Any], 
        model_name: str = None
    ) -> Dict[str, Any]:
        """
        Valida parametri LLM per un modello specifico
        
        Args:
            parameters: Parametri da validare
            model_name: Nome modello per vincoli specifici
            
        Returns:
            Dizionario con risultato validazione
            
        Data ultima modifica: 2025-01-31
        """
        errors = []
        warnings = []
        
        try:
            # Recupera vincoli modello
            model_constraints = None
            if model_name:
                model_constraints = self.get_model_info(model_name)
            
            # Validazione tokenization
            if 'tokenization' in parameters:
                tokenization = parameters['tokenization']
                max_tokens = tokenization.get('max_tokens')
                
                if max_tokens is not None:
                    if not isinstance(max_tokens, int) or max_tokens < 100:
                        errors.append("max_tokens deve essere un intero >= 100")
                    elif model_constraints and max_tokens > model_constraints.get('max_input_tokens', 8000):
                        max_allowed = model_constraints.get('max_input_tokens', 8000)
                        errors.append(f"max_tokens ({max_tokens}) supera il limite del modello {model_name} ({max_allowed})")
                    elif max_tokens > 8000:
                        warnings.append(f"max_tokens ({max_tokens}) molto alto, potrebbe impattare le performance")
            
            # Validazione generation
            if 'generation' in parameters:
                generation = parameters['generation']
                
                # 🆕 GPT-5: Ignora parametri non supportati
                is_gpt5 = model_name and model_name.lower() == 'gpt-5'
                if is_gpt5:
                    unsupported_params = ['temperature', 'top_p', 'top_k', 'repeat_penalty']
                    for param in unsupported_params:
                        if param in generation:
                            warnings.append(f"⚠️ GPT-5: parametro '{param}' non supportato, verrà ignorato")
                
                # Temperature (skip per GPT-5)
                temp = generation.get('temperature')
                if temp is not None and not is_gpt5:
                    if not isinstance(temp, (int, float)) or temp < 0.0 or temp > 2.0:
                        errors.append("temperature deve essere tra 0.0 e 2.0")
                    elif temp > 1.5:
                        warnings.append(f"temperature ({temp}) molto alta, potrebbe generare risposte incoerenti")
                
                # Top K (skip per GPT-5)
                top_k = generation.get('top_k')
                if top_k is not None and not is_gpt5:
                    if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
                        errors.append("top_k deve essere tra 1 e 100")
                
                # Top P (skip per GPT-5)
                top_p = generation.get('top_p')
                if top_p is not None and not is_gpt5:
                    if not isinstance(top_p, (int, float)) or top_p < 0.1 or top_p > 1.0:
                        errors.append("top_p deve essere tra 0.1 e 1.0")
                
                # Repeat Penalty (skip per GPT-5)
                repeat_penalty = generation.get('repeat_penalty')
                if repeat_penalty is not None and not is_gpt5:
                    if not isinstance(repeat_penalty, (int, float)) or repeat_penalty < 0.8 or repeat_penalty > 1.5:
                        errors.append("repeat_penalty deve essere tra 0.8 e 1.5")
                
                # Max Tokens Output
                max_tokens_out = generation.get('max_tokens')
                if max_tokens_out is not None:
                    if not isinstance(max_tokens_out, int) or max_tokens_out < 50:
                        errors.append("generation.max_tokens deve essere >= 50")
                    elif model_constraints and max_tokens_out > model_constraints.get('max_output_tokens', 4000):
                        max_allowed = model_constraints.get('max_output_tokens', 4000)
                        errors.append(f"generation.max_tokens ({max_tokens_out}) supera il limite del modello ({max_allowed})")
            
            # Validazione connection
            if 'connection' in parameters:
                connection = parameters['connection']
                timeout = connection.get('timeout')
                if timeout is not None:
                    if not isinstance(timeout, int) or timeout < 30 or timeout > 600:
                        errors.append("timeout deve essere tra 30 e 600 secondi")
                    elif timeout > 300:
                        warnings.append(f"timeout ({timeout}s) molto alto")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'model_constraints': model_constraints
            }
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f'Errore validazione: {str(e)}'],
                'warnings': [],
                'model_constraints': None
            }
    
    
    def test_model_configuration(
        self, 
        tenant_id: str, 
        model_name: str, 
        parameters: Dict[str, Any],
        test_prompt: str = "Test di connessione. Rispondi con 'OK' se funziona."
    ) -> Dict[str, Any]:
        """
        Testa modello LLM con parametri specifici
        
        Args:
            tenant_id: ID del tenant
            model_name: Nome del modello da testare
            parameters: Parametri di test
            test_prompt: Prompt di test
            
        Returns:
            Dizionario con risultato test
            
        Data ultima modifica: 2025-01-31
        """
        try:
            # Validazione parametri
            validation_result = self.validate_parameters(parameters, model_name)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': f"Parametri non validi: {validation_result['errors']}",
                    'tenant_id': tenant_id,
                    'model_name': model_name
                }
            
            # Istanza temporanea classifier per test
            test_classifier = IntelligentClassifier(
                client_name=f"{tenant_id}_test",
                enable_logging=False,
                model_name=model_name
            )
            
            # Override parametri temporaneamente
            original_params = {}
            if 'tokenization' in parameters:
                tokenization = parameters['tokenization']
                original_params['max_tokens'] = test_classifier.max_tokens
                test_classifier.max_tokens = tokenization.get('max_tokens', test_classifier.max_tokens)
            
            if 'generation' in parameters:
                generation = parameters['generation']
                original_params['temperature'] = test_classifier.temperature
                original_params['top_k'] = test_classifier.top_k
                original_params['top_p'] = test_classifier.top_p
                
                test_classifier.temperature = generation.get('temperature', test_classifier.temperature)
                test_classifier.top_k = generation.get('top_k', test_classifier.top_k)
                test_classifier.top_p = generation.get('top_p', test_classifier.top_p)
            
            # Esegui test con timeout
            start_time = time.time()
            
            # Chiamata HTTP diretta all'API Ollama (invece di usare ollama_client inesistente)
            ollama_url = test_classifier.ollama_url or "http://localhost:11434"
            
            # Costruisci payload per Ollama API
            payload = {
                "model": model_name,
                "prompt": test_prompt,
                "stream": False,
                "options": {
                    'temperature': test_classifier.temperature,
                    'top_k': test_classifier.top_k,
                    'top_p': test_classifier.top_p,
                    'num_predict': 100  # Limitato per test rapido
                }
            }
            
            # Effettua chiamata HTTP diretta
            response = requests.post(
                f"{ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result_data = response.json()
            
            test_duration = time.time() - start_time
            response_text = result_data.get('response', '')
            
            print(f"🧪 [LLMConfigService] Test {model_name} per {tenant_id} completato ({test_duration:.2f}s)")
            
            return {
                'success': True,
                'tenant_id': tenant_id,
                'model_name': model_name,
                'test_duration': round(test_duration, 2),
                'response_preview': response_text[:200] + '...' if len(response_text) > 200 else response_text,
                'response_length': len(response_text),
                'parameters_used': parameters,
                'validation': validation_result
            }
            
        except Exception as e:
            error_msg = f'Errore test modello: {str(e)}'
            print(f"❌ [LLMConfigService] {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'tenant_id': tenant_id,
                'model_name': model_name
            }
    
    
    def get_tenant_list(self) -> List[str]:
        """
        Recupera lista tenant con configurazioni LLM personalizzate
        
        Returns:
            Lista ID tenant
            
        Data ultima modifica: 2025-01-31
        """
        try:
            if self.reload_enabled:
                self._reload_config()
            
            tenant_configs = self.config_cache.get('tenant_configs', {})
            tenants_with_llm = []
            
            for tenant_id, config in tenant_configs.items():
                if 'llm_parameters' in config:
                    tenants_with_llm.append(tenant_id)
            
            return tenants_with_llm
            
        except Exception as e:
            print(f"❌ [LLMConfigService] Errore recupero lista tenant: {e}")
            return []
    
    
    def _get_tenant_last_modified(self, tenant_id: str) -> Optional[str]:
        """
        Recupera timestamp ultima modifica per tenant
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Timestamp ISO o None
            
        Data ultima modifica: 2025-01-31
        """
        try:
            tenant_configs = self.config_cache.get('tenant_configs', {})
            tenant_config = tenant_configs.get(tenant_id, {})
            return tenant_config.get('last_modified')
            
        except Exception as e:
            print(f"❌ [LLMConfigService] Errore recupero timestamp {tenant_id}: {e}")
            return None
    
    
    def _create_config_backup(self) -> str:
        """
        Crea backup configurazione corrente
        
        Returns:
            Percorso file backup
            
        Data ultima modifica: 2025-01-31
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(os.path.dirname(self.config_path), 'backup')
            
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            backup_path = os.path.join(backup_dir, f'config_{timestamp}.yaml')
            
            with open(self.config_path, 'r', encoding='utf-8') as source:
                with open(backup_path, 'w', encoding='utf-8') as backup:
                    backup.write(source.read())
            
            print(f"💾 [LLMConfigService] Backup creato: {backup_path}")
            return backup_path
            
        except Exception as e:
            print(f"❌ [LLMConfigService] Errore creazione backup: {e}")
            return ""
    
    
    def disable_hot_reload(self):
        """
        Disabilita hot-reload per performance critiche
        
        Data ultima modifica: 2025-01-31
        """
        self.reload_enabled = False
        print("⏸️ [LLMConfigService] Hot-reload disabilitato")
    
    
    def enable_hot_reload(self):
        """
        Riabilita hot-reload
        
        Data ultima modifica: 2025-01-31
        """
        self.reload_enabled = True
        self._reload_config()
        print("▶️ [LLMConfigService] Hot-reload abilitato")
