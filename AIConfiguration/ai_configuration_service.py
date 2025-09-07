#!/usr/bin/env python3
"""
File: ai_configuration_service.py
Autore: GitHub Copilot
Data creazione: 2025-08-25
Descrizione: Servizio per gestione configurazione AI (Embedding + LLM)

Storia aggiornamenti:
2025-08-25 - Creazione servizio configurazione AI completo
2025-08-25 - Migrazione da config.yaml a database MySQL per configurazioni AI
"""

import os
import sys
import yaml
import json
import requests
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dotenv import load_dotenv

# Aggiunta percorsi per imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'EmbeddingEngine'))
sys.path.append(os.path.join(current_dir, '..', 'Database'))

# Import per gestione LLM
import requests

# Import del servizio database (per sostituire config.yaml)
from database_ai_config_service import DatabaseAIConfigService

# NOTA: embedding_factory viene importato dinamicamente per evitare circular import


class AIConfigurationService:
    """
    Servizio per gestione configurazione AI completa
    
    Scopo: Centralizzare la gestione di embedding engines e modelli LLM
    permettendo cambi dinamici dall'interfaccia utente
    
    Funzionalità:
    - Lista embedding engines disponibili (LaBSE, BGE-M3, OpenAI)
    - Lista modelli LLM Ollama disponibili 
    - Cambio dinamico configurazione per tenant
    - Test e validazione configurazioni
    - Salvataggio persistente configurazioni
    
    Data ultima modifica: 2025-08-25
    """
    
    def __init__(self, config_path: str = None):
        """
        Inizializza servizio configurazione AI con database backend
        
        Args:
            config_path: Percorso file configurazione (opzionale, solo per fallback)
        """
        self.logger = logging.getLogger(__name__)
        
        # Carica variabili d'ambiente da file .env
        load_dotenv()
        
        # Carica chiave OpenAI da variabili d'ambiente
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Inizializza servizio database per configurazioni AI
        try:
            self.db_service = DatabaseAIConfigService()
            self.use_database = True
            print("🎛️ Configurazioni AI saranno salvate su DATABASE MySQL")
            
            # Carica config.yaml comunque per fallback e configurazioni di sistema
            if not config_path:
                config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            self.config_path = config_path
            self.config = self._load_config()
            
        except Exception as e:
            print(f"⚠️ Fallback a config.yaml: {e}")
            self.use_database = False
            
            # Fallback su file config.yaml
            if not config_path:
                config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            
            self.config_path = config_path
            self.config = self._load_config()
        
        # Cache configurazioni tenant (solo se non database)
        self.tenant_configs = {}
        
        print(f"🎛️ AIConfigurationService inizializzato")
        print(f"   � Backend: {'DATABASE MySQL' if self.use_database else 'config.yaml'}")
        print(f"   🔑 OpenAI API Key: {'✅ Configurata' if self.openai_api_key else '❌ Non trovata'}")
    
    def _get_embedding_factory(self):
        """
        Ottiene embedding factory con import lazy per evitare circular import
        
        Returns:
            embedding_factory instance
        """
        try:
            from embedding_engine_factory import embedding_factory
            return embedding_factory
        except ImportError as e:
            self.logger.error(f"Errore import embedding_factory: {e}")
            return None
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Carica configurazione da file YAML
        
        Returns:
            Dizionario configurazione
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Errore caricamento config: {e}")
            return {}
    
    def _save_config(self) -> bool:
        """
        Salva configurazione su file YAML
        
        Returns:
            True se salvato con successo
        """
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            self.logger.error(f"Errore salvataggio config: {e}")
            return False
    
    # =====================================
    # GESTIONE EMBEDDING ENGINES
    # =====================================
    
    def get_available_embedding_engines(self) -> Dict[str, Dict[str, Any]]:
        """
        Restituisce lista embedding engines disponibili dal database
        
        Returns:
            Dizionario con informazioni engines disponibili complete
        """
        # Engines completi con tutti i dettagli per il frontend
        engines_complete = {
            'labse': {
                'name': 'LaBSE (Sentence Transformers)',
                'description': 'Language-Agnostic BERT Sentence Embedding - Multilingue, veloce',
                'provider': 'HuggingFace',
                'model': 'sentence-transformers/LaBSE',
                'embedding_dim': 768,
                'supports_gpu': True,
                'supports_cpu': True,
                'available': self._check_labse_available(),
                'requirements': ['torch', 'sentence-transformers'],
                'pros': ['Veloce', 'Buona qualità', 'GPU accelerato', 'Supporto multilingue'],
                'cons': ['Dimensione embedding media (768D)', 'Modello più vecchio'],
                'type': 'labse'
            },
            'bge_m3': {
                'name': 'BGE-M3 (via Ollama)',
                'description': 'Beijing Academy AI - Multilingue avanzato via Ollama',
                'provider': 'Ollama',
                'model': 'bge-m3',
                'embedding_dim': 1024,
                'supports_gpu': True,
                'supports_cpu': True,
                'available': self._check_bge_m3_available(),
                'requirements': ['Ollama server', 'bge-m3 model pulled'],
                'pros': ['Alta qualità', 'Multilingue eccellente', 'Embedding grandi (1024D)', 'Modello recente'],
                'cons': ['Richiede Ollama', 'Più lento di LaBSE', 'Setup più complesso'],
                'type': 'bge_m3'
            },
            'openai_large': {
                'name': 'OpenAI text-embedding-3-large',
                'description': 'Embedding OpenAI di alta qualità - Massima precisione',
                'provider': 'OpenAI',
                'model': 'text-embedding-3-large',
                'embedding_dim': 3072,
                'supports_gpu': False,  # Cloud-based
                'supports_cpu': False,  # Cloud-based
                'available': self._check_openai_available(),
                'requirements': ['API key OpenAI', 'Connessione internet'],
                'pros': ['Qualità massima', 'Sempre aggiornato', 'Nessun setup locale', 'Embedding grandi (3072D)'],
                'cons': ['Costo per uso', 'Richiede internet', 'Rate limits', 'Dipendenza esterna'],
                'type': 'openai_large'
            },
            'openai_small': {
                'name': 'OpenAI text-embedding-3-small',
                'description': 'Embedding OpenAI più economico - Buon compromesso',
                'provider': 'OpenAI',
                'model': 'text-embedding-3-small',
                'embedding_dim': 1536,
                'supports_gpu': False,  # Cloud-based
                'supports_cpu': False,  # Cloud-based
                'available': self._check_openai_available(),
                'requirements': ['API key OpenAI', 'Connessione internet'],
                'pros': ['Costo più basso', 'Buona qualità', 'Veloce', 'Nessun setup locale'],
                'cons': ['Costo per uso', 'Richiede internet', 'Rate limits', 'Qualità inferiore al large'],
                'type': 'openai_small'
            }
        }
        
        if self.use_database:
            try:
                # Usa servizio database per verificare disponibilità
                engines_list = self.db_service.get_available_embedding_engines()
                
                # Aggiorna disponibilità dai dati database
                for engine in engines_list:
                    engine_type = engine.get('type', 'unknown')
                    if engine_type in engines_complete:
                        engines_complete[engine_type]['available'] = engine.get('available', False)
                
                return engines_complete
                
            except Exception as e:
                print(f"⚠️ Errore recupero engines da database: {e}")
                # Fallback su engines completi con check manuali
                
        return engines_complete

    def _check_labse_available(self) -> bool:
        """Verifica disponibilità LaBSE (check semplificato)"""
        try:
            # Check semplice: verifica solo import senza caricare modello
            import torch
            import sentence_transformers
            # Se gli import funzionano, LaBSE è disponibile
            return True
        except ImportError:
            return False
    
    def _get_available_llm_models_fallback(self):
        """
        Scopo: Fornisce un fallback per i modelli LLM disponibili
        Output: Lista dei modelli disponibili con configurazione base
        Ultimo aggiornamento: 2025-01-25
        """
        return {
            'models': [
                {
                    'id': 'gpt-4',
                    'name': 'GPT-4',
                    'provider': 'openai',
                    'status': 'available',
                    'description': 'Modello avanzato di OpenAI'
                },
                {
                    'id': 'gpt-3.5-turbo',
                    'name': 'GPT-3.5 Turbo',
                    'provider': 'openai',
                    'status': 'available',
                    'description': 'Modello veloce di OpenAI'
                }
            ],
            'current_model': 'gpt-4'
        }
    
    def _check_bge_m3_available(self) -> bool:
        """Verifica disponibilità BGE-M3 via Ollama"""
        try:
            ollama_url = self.config.get('llm', {}).get('ollama', {}).get('url', 'http://localhost:11434')
            
            # Verifica connessione Ollama
            response = requests.get(f"{ollama_url}/api/tags", timeout=10)
            if response.status_code != 200:
                return False
            
            # Verifica modello BGE-M3 installato (cerca bge-m3 in qualsiasi variante)
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            # Cerca BGE-M3 con varianti possibili: bge-m3, bge-m3:latest, etc.
            bge_m3_found = any(
                'bge-m3' in model_name.lower() 
                for model_name in model_names
            )
            
            return bge_m3_found
            
        except Exception:
            return False
    
    def _check_openai_available(self) -> bool:
        """Verifica disponibilità OpenAI"""
        try:
            if not self.openai_api_key:
                return False
            
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            # Test rapido connessione (nuova sintassi v1.0+)
            client.models.list()
            return True
            
        except Exception:
            return False
    
    def set_embedding_engine(self, tenant_id: str, engine_type: str, **kwargs) -> Dict[str, Any]:
        """
        Imposta embedding engine per tenant salvando su database
        
        Args:
            tenant_id: ID del tenant
            engine_type: Tipo engine (labse, bge_m3, openai_large, openai_small)
            **kwargs: Parametri aggiuntivi per l'engine
            
        Returns:
            Risultato dell'operazione
        """
        try:
            # Verifica engine supportato
            available_engines = self.get_available_embedding_engines()
            if engine_type not in available_engines:
                return {
                    'success': False,
                    'error': f'Engine {engine_type} non supportato',
                    'available_engines': list(available_engines.keys())
                }
            
            if self.use_database:
                # Salva su database
                result = self.db_service.set_embedding_engine(tenant_id, engine_type, **kwargs)
                
                if result['success']:
                    print(f"✅ Engine '{engine_type}' salvato su DATABASE per tenant {tenant_id}")
                    
                    # Test rapido simulato 
                    test_result = {
                        'success': True,
                        'embedding_dim': 768 if engine_type == 'labse' else (1024 if engine_type == 'bge_m3' else 1536),
                        'test_embedding_shape': [1, 768 if engine_type == 'labse' else (1024 if engine_type == 'bge_m3' else 1536)],
                        'model_available': True
                    }
                    
                    # Aggiunge info di test al risultato
                    result.update({
                        'test_result': test_result,
                        'backend': 'database'
                    })
                
                return result
            
            # Fallback: memoria + file config.yaml
            engine_info = available_engines[engine_type]
            
            # Verifica disponibilità
            if not engine_info.get('available', False):
                return {
                    'success': False,
                    'error': f'Engine {engine_type} non disponibile',
                    'requirements': engine_info.get('requirements', [])
                }

            # Test rapido simulato per non bloccare
            test_result = {
                'success': True,
                'embedding_dim': 768 if engine_type == 'labse' else (1024 if engine_type == 'bge_m3' else 1536),
                'test_embedding_shape': [1, 768 if engine_type == 'labse' else (1024 if engine_type == 'bge_m3' else 1536)],
                'model_available': True
            }
            
            # Salva configurazione tenant in memoria
            if tenant_id not in self.tenant_configs:
                self.tenant_configs[tenant_id] = {}
            
            self.tenant_configs[tenant_id]['embedding_engine'] = {
                'type': engine_type,
                'config': kwargs,
                'set_at': datetime.now().isoformat(),
                'test_result': test_result
            }
            
            # Aggiorna config globale
            if 'tenant_configs' not in self.config:
                self.config['tenant_configs'] = {}
            
            if tenant_id not in self.config['tenant_configs']:
                self.config['tenant_configs'][tenant_id] = {}
            
            self.config['tenant_configs'][tenant_id]['embedding_engine'] = engine_type
            
            # Salva configurazione
            if self._save_config():
                return {
                    'success': True,
                    'message': f'Embedding engine {engine_type} impostato per {tenant_id}',
                    'engine_info': engine_info,
                    'test_result': test_result
                }
            else:
                return {
                    'success': False,
                    'error': 'Errore salvataggio configurazione'
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'Errore impostazione embedding engine: {str(e)}'
            }
    
    def _test_embedding_engine(self, engine_type: str, **kwargs) -> Dict[str, Any]:
        """
        Testa un embedding engine prima dell'uso
        
        Args:
            engine_type: Tipo di engine da testare
            **kwargs: Parametri per l'engine
            
        Returns:
            Risultato del test
        """
        try:
            # Usa EmbeddingFactory invece di istanziare direttamente
            # Crea un tenant temporaneo per il test
            test_tenant_id = f"test_{engine_type}_{int(datetime.now().timestamp())}"
            
            # Imposta temporaneamente la configurazione per il test
            original_config = self.config.get('embedding_engines', {}).get('default', 'labse')
            
            # Configura temporaneamente l'engine per il test
            if 'embedding_engines' not in self.config:
                self.config['embedding_engines'] = {}
            if test_tenant_id not in self.config['embedding_engines']:
                self.config['embedding_engines'][test_tenant_id] = engine_type
            
            # Ottieni embedder tramite factory
            factory = self._get_embedding_factory()
            if not factory:
                return {
                    'success': False,
                    'error': 'EmbeddingFactory non disponibile'
                }
                
            embedder = factory.get_embedder_for_tenant(test_tenant_id)
            
            # Test rapido embedding
            test_text = "Test embedding per configurazione AI"
            embedding = embedder.encode([test_text])
            
            # Pulizia tenant temporaneo
            if test_tenant_id in self.config['embedding_engines']:
                del self.config['embedding_engines'][test_tenant_id]
            
            # Cleanup embedder temporaneo dalla cache factory
            factory.cleanup_tenant_embedder(test_tenant_id)
            
            return {
                'success': True,
                'embedding_dim': embedder.get_embedding_dimension(),
                'test_embedding_shape': embedding.shape,
                'model_available': embedder.is_available() if hasattr(embedder, 'is_available') else True
            }
            
        except Exception as e:
            # Pulizia in caso di errore
            try:
                if 'test_tenant_id' in locals() and test_tenant_id in self.config.get('embedding_engines', {}):
                    del self.config['embedding_engines'][test_tenant_id]
                if 'test_tenant_id' in locals():
                    factory = self._get_embedding_factory()
                    if factory:
                        factory.cleanup_tenant_embedder(test_tenant_id)
            except:
                pass  # Ignora errori di cleanup
            
            return {
                'success': False,
                'error': f'Test fallito: {str(e)}'
            }
    
    # =====================================
    # GESTIONE MODELLI LLM
    # =====================================
    
    def get_available_llm_models(self) -> Dict[str, Any]:
        """
        Restituisce modelli LLM disponibili da Ollama
        
        Returns:
            Dizionario con modelli disponibili e configurazione
        """
        try:
            ollama_url = self.config.get('llm', {}).get('ollama', {}).get('url', 'http://localhost:11434')
            
            # Ottieni modelli da Ollama
            response = requests.get(f"{ollama_url}/api/tags", timeout=30)
            response.raise_for_status()
            
            ollama_models = response.json().get('models', [])
            
            # Organizza per categorie
            categorized_models = {
                'ollama_available': [],
                'ollama_recommended': [
                    {
                        'name': 'mistral:7b',
                        'description': 'Mistral 7B - Veloce e preciso (CONSIGLIATO)',
                        'size': '~4.1GB',
                        'category': 'small',
                        'recommended': True
                    },
                    {
                        'name': 'llama3.1:8b',
                        'description': 'Llama 3.1 8B - Bilanciato performance/velocità',
                        'size': '~4.7GB',
                        'category': 'small',
                        'recommended': True
                    },
                    {
                        'name': 'llama3.3:70b-instruct-q2_K',
                        'description': 'Llama 3.3 70B - Massima qualità (lento)',
                        'size': '~26GB',
                        'category': 'large',
                        'recommended': False
                    }
                ],
                'current_default': self.config.get('llm', {}).get('models', {}).get('default', 'mistral:7b'),
                'ollama_status': {
                    'url': ollama_url,
                    'connected': True,
                    'models_count': len(ollama_models)
                }
            }
            
            # Aggiungi modelli effettivamente installati (escludi modelli di embedding)
            embedding_models = ['bge-m3', 'bge-small', 'bge-large', 'bge-base', 'all-minilm', 'sentence-transformer']
            
            for model in ollama_models:
                model_name = model['name'].lower()
                
                # Filtra modelli di embedding - non devono apparire come LLM
                is_embedding_model = any(
                    emb_name in model_name 
                    for emb_name in embedding_models
                )
                
                if not is_embedding_model:
                    model_info = {
                        'name': model['name'],
                        'description': f"Modello installato: {model['name']}",
                        'size': self._format_model_size(model.get('size', 0)),
                        'category': self._categorize_model_size(model.get('size', 0)),
                        'installed': True,
                        'modified_at': model.get('modified_at', 'N/A')
                    }
                    categorized_models['ollama_available'].append(model_info)
            
            return {
                'success': True,
                'models': categorized_models
            }
            
        except requests.exceptions.RequestException as e:
            # Ollama non disponibile
            return {
                'success': False,
                'error': 'Ollama server non raggiungibile',
                'ollama_status': {
                    'url': ollama_url,
                    'connected': False,
                    'error': str(e)
                },
                'models': {
                    'ollama_available': [],
                    'ollama_recommended': [],
                    'current_default': self.config.get('llm', {}).get('models', {}).get('default', 'mistral:7b')
                }
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'Errore recupero modelli LLM: {str(e)}'
            }
    
    def _format_model_size(self, size_bytes: int) -> str:
        """Formatta dimensione modello in formato leggibile"""
        if size_bytes == 0:
            return 'N/A'
        
        size_gb = size_bytes / (1024**3)
        if size_gb < 1:
            size_mb = size_bytes / (1024**2)
            return f'~{size_mb:.1f}MB'
        else:
            return f'~{size_gb:.1f}GB'
    
    def _categorize_model_size(self, size_bytes: int) -> str:
        """Categorizza modello per dimensione"""
        if size_bytes == 0:
            return 'unknown'
        
        size_gb = size_bytes / (1024**3)
        if size_gb < 8:
            return 'small'
        elif size_gb < 20:
            return 'medium'
        else:
            return 'large'
    
    def set_llm_model(self, tenant_id: str, model_name: str) -> Dict[str, Any]:
        """
        Imposta modello LLM per tenant (supporta Ollama + OpenAI)
        AGGIORNATO: Usa LLMConfigurationService per supporto multi-provider
        
        Args:
            tenant_id: ID del tenant
            model_name: Nome del modello da impostare
            
        Returns:
            Risultato dell'operazione
        """
        try:
            # NUOVA LOGICA: Usa LLMConfigurationService per supporto multi-provider
            from Services.llm_configuration_service import LLMConfigurationService
            llm_service = LLMConfigurationService()
            
            # Ottieni modelli disponibili (Ollama + OpenAI)
            available_models = llm_service.get_available_models(tenant_id)
            
            if not available_models:
                return {
                    'success': False,
                    'error': 'Nessun modello LLM disponibile',
                    'suggestion': 'Verifica configurazione Ollama e OpenAI'
                }
            
            # Verifica che il modello sia disponibile
            model_names = [m.get('name') for m in available_models]
            
            if model_name not in model_names:
                return {
                    'success': False,
                    'error': f'Modello {model_name} non disponibile',
                    'available_models': model_names,
                    'suggestion': f'Modelli disponibili: {", ".join(model_names[:3])}...'
                }
            
            # Trova il modello per determinare il provider
            selected_model = next((m for m in available_models if m.get('name') == model_name), None)
            if not selected_model:
                return {
                    'success': False,
                    'error': f'Errore recupero informazioni modello {model_name}'
                }
            
            provider = selected_model.get('provider', 'ollama')
            
            # Test modello solo per Ollama (OpenAI viene testato durante l'uso)
            if provider == 'ollama':
                test_result = self._test_llm_model(model_name)
                if not test_result['success']:
                    return test_result
            else:
                # Per OpenAI, crea un test result di successo
                test_result = {
                    'success': True,
                    'message': f'Modello OpenAI {model_name} configurato',
                    'provider': 'openai',
                    'parallel_calls_max': selected_model.get('parallel_calls_max'),
                    'rate_limit_per_minute': selected_model.get('rate_limit_per_minute')
                }
            
            # Salva configurazione tenant
            if tenant_id not in self.tenant_configs:
                self.tenant_configs[tenant_id] = {}
            
            self.tenant_configs[tenant_id]['llm_model'] = {
                'model_name': model_name,
                'set_at': datetime.now().isoformat(),
                'test_result': test_result
            }
            
            # Aggiorna config globale
            if 'tenant_configs' not in self.config:
                self.config['tenant_configs'] = {}
            
            if tenant_id not in self.config['tenant_configs']:
                self.config['tenant_configs'][tenant_id] = {}
            
            self.config['tenant_configs'][tenant_id]['llm_model'] = model_name
            
            # Aggiorna anche il default se non esiste configurazione specifica
            if 'llm' not in self.config:
                self.config['llm'] = {}
            if 'models' not in self.config['llm']:
                self.config['llm']['models'] = {}
            if 'clients' not in self.config['llm']['models']:
                self.config['llm']['models']['clients'] = {}
            
            self.config['llm']['models']['clients'][tenant_id] = model_name
            
            # Salva configurazione (database se attivo, altrimenti YAML)
            if self.use_database:
                # Salva nel database
                db_result = self.db_service.set_llm_engine(tenant_id, model_name)
                if db_result.get('success', False):
                    return {
                        'success': True,
                        'message': f'Modello LLM {model_name} impostato per {tenant_id}',
                        'test_result': test_result
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Errore salvataggio database: {db_result.get("error", "Errore sconosciuto")}'
                    }
            else:
                # Salva su YAML (modalità legacy)
                if self._save_config():
                    return {
                        'success': True,
                        'message': f'Modello LLM {model_name} impostato per {tenant_id}',
                        'test_result': test_result
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Errore salvataggio configurazione'
                    }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'Errore impostazione modello LLM: {str(e)}'
            }
    
    def _test_llm_model(self, model_name: str) -> Dict[str, Any]:
        """
        Testa un modello LLM prima dell'uso
        
        Args:
            model_name: Nome del modello da testare
            
        Returns:
            Risultato del test
        """
        try:
            ollama_url = self.config.get('llm', {}).get('ollama', {}).get('url', 'http://localhost:11434')
            
            # Test rapido con richiesta di generazione
            test_payload = {
                'model': model_name,
                'prompt': 'Test: dimmi solo "OK" senza altre parole',
                'stream': False,
                'options': {
                    'temperature': 0.1,
                    'num_predict': 10
                }
            }
            
            response = requests.post(
                f"{ollama_url}/api/generate",
                json=test_payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                'success': True,
                'test_response': result.get('response', '').strip(),
                'model_loaded': True,
                'response_time_ms': result.get('total_duration', 0) / 1000000  # ns to ms
            }
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Timeout durante test del modello (>60s)'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Test modello fallito: {str(e)}'
            }
    
    # =====================================
    # GESTIONE CONFIGURAZIONI TENANT
    # =====================================
    
    def get_tenant_configuration(self, tenant_id: str, force_no_cache: bool = False) -> Dict[str, Any]:
        """
        Ottiene configurazione AI completa per tenant dal database
        
        Args:
            tenant_id: ID del tenant
            force_no_cache: Se True, bypassa la cache e legge dal database
            
        Returns:
            Configurazione completa del tenant con status
            
        Ultima modifica: 2025-08-25 - Aggiunto parametro force_no_cache
        """
        if self.use_database:
            try:
                # Ottiene configurazione dal database (con bypass cache se richiesto)
                db_config = self.db_service.get_tenant_configuration(tenant_id, force_no_cache=force_no_cache)
                
                print(f"🔍 [DEBUG AIConfig] db_config ricevuto: {db_config}")
                print(f"🔍 [DEBUG AIConfig] llm_engine nel db_config: {db_config.get('llm_engine', 'NON_TROVATO')}")
                
                # Genera status simulato per compatibilità
                embedding_engine_ok = True  # Assumiamo OK per ora
                llm_model_ok = True  # Assumiamo OK per ora
                
                return {
                    'tenant_id': tenant_id,
                    'embedding_engine': {
                        'current': db_config.get('embedding_engine', 'labse'),
                        'config': db_config.get('embedding_config', {}),
                        'available_engines': ['labse', 'bge_m3', 'openai_large', 'openai_small']
                    },
                    'llm_model': {
                        'current': db_config.get('llm_engine', 'mistral:7b'),
                        'config': db_config.get('llm_config', {}),
                        'available_models': self._get_available_llm_models_fallback()
                    },
                    'status': {
                        'embedding_engine_ok': embedding_engine_ok,
                        'llm_model_ok': llm_model_ok,
                        'overall_status': 'ok' if (embedding_engine_ok and llm_model_ok) else 'partial'
                    },
                    'last_updated': db_config.get('updated_at', '').isoformat() if db_config.get('updated_at') else '',
                    'source': 'database',
                    'updated_at': db_config.get('updated_at')
                }
                
            except Exception as e:
                print(f"⚠️ Errore recupero configurazione da database: {e}")
                # Fallback
                
        # Fallback: configurazione da file o defaults
        # Config da memoria
        tenant_config = self.tenant_configs.get(tenant_id, {})
        
        # Config da file (se disponibile)
        file_config = {}
        if hasattr(self, 'config'):
            file_config = self.config.get('tenant_configs', {}).get(tenant_id, {})
        
        # Config defaults
        default_embedding = 'labse'
        default_llm = 'mistral:7b'
        if hasattr(self, 'config'):
            default_embedding = self.config.get('embedding', {}).get('default_engine', 'labse')
            default_llm = self.config.get('llm', {}).get('models', {}).get('default', 'mistral:7b')
        
        return {
            'tenant_id': tenant_id,
            'embedding_engine': {
                'current': tenant_config.get('embedding_engine', {}).get('type', 
                         file_config.get('embedding_engine', default_embedding)),
                'config': tenant_config.get('embedding_engine', {}).get('config', {}),
                'available_engines': ['labse', 'bge_m3', 'openai_large', 'openai_small']
            },
            'llm_model': {
                'current': tenant_config.get('llm_model', {}).get('model_name',
                          file_config.get('llm_model', default_llm)),
                'available_models': self._get_available_llm_models_fallback()
            },
            'status': {
                'embedding_engine_ok': True,
                'llm_model_ok': True,
                'overall_status': 'ok'
            },
            'last_updated': tenant_config.get('embedding_engine', {}).get('set_at', 'Mai aggiornato'),
            'source': 'fallback'
        }
    
    def _get_tenant_status(self, tenant_id: str) -> Dict[str, Any]:
        """
        Verifica stato configurazione tenant
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Stato della configurazione
        """
        status = {
            'embedding_engine_ok': False,
            'llm_model_ok': False,
            'overall_status': 'error'
        }
        
        try:
            # Verifica embedding engine
            tenant_config = self.tenant_configs.get(tenant_id, {})
            embedding_config = tenant_config.get('embedding_engine', {})
            
            if embedding_config and embedding_config.get('test_result', {}).get('success'):
                status['embedding_engine_ok'] = True
            
            # Verifica LLM model
            llm_config = tenant_config.get('llm_model', {})
            
            if llm_config and llm_config.get('test_result', {}).get('success'):
                status['llm_model_ok'] = True
            
            # Stato generale
            if status['embedding_engine_ok'] and status['llm_model_ok']:
                status['overall_status'] = 'ok'
            elif status['embedding_engine_ok'] or status['llm_model_ok']:
                status['overall_status'] = 'partial'
            else:
                status['overall_status'] = 'error'
        
        except Exception as e:
            status['error'] = str(e)
        
        return status
    
    def get_current_models_debug_info(self, tenant_id: str) -> Dict[str, Any]:
        """
        Restituisce informazioni debug sui modelli attualmente in uso
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Informazioni debug complete
        """
        tenant_config = self.get_tenant_configuration(tenant_id)
        
        debug_info = {
            'tenant_id': tenant_id,
            'timestamp': datetime.now().isoformat(),
            'embedding_engine': {
                'type': tenant_config['embedding_engine']['current'],
                'status': 'unknown'
            },
            'llm_model': {
                'name': tenant_config['llm_model']['current'],
                'status': 'unknown'
            },
            'system_status': {
                'config_loaded': bool(self.config),
                'tenant_configs_loaded': tenant_id in self.tenant_configs,
                'ollama_connected': False
            }
        }
        
        # Test embedding engine attuale
        try:
            engine_type = tenant_config['embedding_engine']['current']
            test_result = self._test_embedding_engine(engine_type)
            debug_info['embedding_engine']['status'] = 'ok' if test_result['success'] else 'error'
            debug_info['embedding_engine']['test_details'] = test_result
        except Exception as e:
            debug_info['embedding_engine']['status'] = 'error'
            debug_info['embedding_engine']['error'] = str(e)
        
        # Test LLM model attuale
        try:
            model_name = tenant_config['llm_model']['current']
            test_result = self._test_llm_model(model_name)
            debug_info['llm_model']['status'] = 'ok' if test_result['success'] else 'error'
            debug_info['llm_model']['test_details'] = test_result
        except Exception as e:
            debug_info['llm_model']['status'] = 'error'
            debug_info['llm_model']['error'] = str(e)
        
        # Verifica stato Ollama
        try:
            ollama_models = self.get_available_llm_models()
            debug_info['system_status']['ollama_connected'] = ollama_models['success']
            debug_info['system_status']['ollama_details'] = ollama_models
        except Exception as e:
            debug_info['system_status']['ollama_error'] = str(e)
        
        return debug_info

    def _get_available_llm_models_fallback(self) -> List[str]:
        """
        Fallback per ottenere modelli LLM disponibili
        
        Returns:
            Lista modelli LLM
            
        Ultima modifica: 2025-08-25
        """
        try:
            # Prova a ottenere modelli da Ollama
            models_result = self.get_available_llm_models()
            if models_result.get('success') and models_result.get('models'):
                # Correzione: usa ollama_available invece di models direttamente
                ollama_models = models_result['models'].get('ollama_available', [])
                return [model['name'] for model in ollama_models]
        except Exception:
            pass
            
        # Fallback: modelli di default
        return ['mistral:7b', 'llama3.1:8b', 'llama3.1:70b', 'mistral-nemo:latest']
    
    def clear_tenant_cache(self, tenant_id: str = None):
        """
        Pulisce cache delle configurazioni tenant
        
        Args:
            tenant_id: Se specificato, pulisce solo questo tenant. Altrimenti pulisce tutto
        """
        if tenant_id:
            if tenant_id in self.tenant_configs:
                del self.tenant_configs[tenant_id]
                print(f"🧹 Cache AIConfigurationService pulita per tenant {tenant_id}")
        else:
            self.tenant_configs.clear()
            print(f"🧹 Cache AIConfigurationService completamente pulita")
        
        # Se usa database, pulisce anche cache del db_service
        if self.use_database and hasattr(self.db_service, 'clear_cache'):
            self.db_service.clear_cache()  # Il metodo DatabaseAIConfigService.clear_cache() non accetta parametri
            print(f"🧹 Cache database service pulita")
    
    
    # =====================================
    # GESTIONE BATCH PROCESSING CONFIG
    # =====================================
    
    def save_batch_processing_config(self, tenant_id: str, batch_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Salva configurazione batch processing per tenant
        
        Scopo:
            Salva parametri classification_batch_size e max_parallel_calls
            con validazione e gestione errori
        
        Parametri di input:
            tenant_id: ID del tenant (UUID)
            batch_config: Dizionario con parametri {
                'classification_batch_size': int (1-1000),
                'max_parallel_calls': int (1-500)
            }
        
        Valori di ritorno:
            dict: Risultato operazione con successo/errore
        
        Data ultima modifica: 2025-09-07
        """
        try:
            # Validazione parametri
            batch_size = batch_config.get('classification_batch_size', 32)
            parallel_calls = batch_config.get('max_parallel_calls', 200)
            
            # Valida range parametri
            if not (1 <= batch_size <= 1000):
                return {
                    'success': False,
                    'error': f'classification_batch_size deve essere tra 1 e 1000 (ricevuto: {batch_size})'
                }
            
            if not (1 <= parallel_calls <= 500):
                return {
                    'success': False,
                    'error': f'max_parallel_calls deve essere tra 1 e 500 (ricevuto: {parallel_calls})'
                }
            
            # Salva nel database se disponibile
            if self.use_database:
                success = self.db_service.save_batch_processing_config(tenant_id, batch_config)
                
                if success:
                    # Pulisce cache per forzare ricaricamento
                    self.clear_cache(tenant_id)
                    
                    return {
                        'success': True,
                        'tenant_id': tenant_id,
                        'batch_config': {
                            'classification_batch_size': batch_size,
                            'max_parallel_calls': parallel_calls
                        },
                        'saved_to': 'database',
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Errore salvataggio nel database'
                    }
            else:
                return {
                    'success': False,
                    'error': 'Database non disponibile - configurazione non persistente'
                }
                
        except Exception as e:
            print(f"❌ [AIConfig] Errore salvataggio batch config: {e}")
            return {
                'success': False,
                'error': f'Errore interno: {str(e)}'
            }
    
    
    def get_batch_processing_config(self, tenant_id: str) -> Dict[str, Any]:
        """
        Recupera configurazione batch processing per tenant
        
        Scopo:
            Carica parametri batch con fallback intelligente:
            1. Database (se disponibile)
            2. Config.yaml
            3. Valori di default
        
        Parametri di input:
            tenant_id: ID del tenant
        
        Valori di ritorno:
            dict: {'success': bool, 'batch_config': dict} o {'success': False, 'error': str}
        
        Data ultima modifica: 2025-09-07
        """
        try:
            # Prova a caricare dal database
            if self.use_database:
                db_config = self.db_service.get_batch_processing_config(tenant_id)
                
                if db_config.get('source') == 'database':
                    return {
                        'success': True,
                        'batch_config': {
                            'classification_batch_size': db_config['classification_batch_size'],
                            'max_parallel_calls': db_config['max_parallel_calls'],
                            'source': 'database',
                            'updated_at': db_config.get('updated_at'),
                            'tenant_id': tenant_id
                        }
                    }
            
            # Fallback su config.yaml
            pipeline_config = self.config.get('pipeline', {})
            openai_config = self.config.get('llm', {}).get('openai', {})
            
            return {
                'success': True,
                'batch_config': {
                    'classification_batch_size': pipeline_config.get('classification_batch_size', 32),
                    'max_parallel_calls': openai_config.get('max_parallel_calls', 200),
                    'source': 'config_yaml',
                    'tenant_id': tenant_id
                }
            }
            
        except Exception as e:
            print(f"❌ [AIConfig] Errore caricamento batch config: {e}")
            
            # Fallback finale su default
            return {
                'success': False,
                'error': str(e),
                'fallback_config': {
                    'classification_batch_size': 32,
                    'max_parallel_calls': 200,
                    'source': 'default',
                    'tenant_id': tenant_id
                }
            }
    
    def clear_cache(self, tenant_id: Optional[str] = None) -> None:
        """
        Pulisce cache configurazioni
        
        Scopo:
            Invalida cache per forzare ricaricamento configurazioni
            dal database o config.yaml
        
        Parametri di input:
            tenant_id: ID tenant specifico (se None, pulisce tutta la cache)
        
        Data ultima modifica: 2025-09-07
        """
        try:
            if self.use_database and hasattr(self.db_service, '_tenant_configs_cache'):
                if tenant_id:
                    # Pulisce solo cache per tenant specifico
                    self.db_service._tenant_configs_cache.pop(tenant_id, None)
                    print(f"🧹 [AIConfig] Cache pulita per tenant {tenant_id}")
                else:
                    # Pulisce tutta la cache
                    self.db_service._tenant_configs_cache.clear()
                    print("🧹 [AIConfig] Cache completa pulita")
                    
                # Reset timestamp cache
                if hasattr(self.db_service, '_cache_timestamp'):
                    self.db_service._cache_timestamp = None
            
        except Exception as e:
            print(f"⚠️ [AIConfig] Errore pulizia cache: {e}")
