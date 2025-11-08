"""
IntelligentClassifier - Classificatore LLM robusto per intent e tag di conversazioni

Questo modulo implementa un classificatore basato su Large Language Model (LLM) 
che utilizza Ollama/Mistral per la classificazione semantica avanzata di conversazioni
con capacità di fallback, caching e robustezza.

Caratteristiche principali:
- Integrazione con Ollama (Mistral 7B)
- Sistema di prompt avanzato con esempi e contesto
- Parsing robusto delle risposte JSON
- Fallback multipli per garantire sempre una risposta
- Caching delle predizioni per efficienza
- Validazione delle etichette generate
- Logging dettagliato per debugging
- Gestione timeout e retry automatici

Autore: Pipeline Humanitas
Data: 26 Giugno 2025
"""

import json
import re
import time
import logging
import os
import sys
import yaml
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hashlib
import threading
from dataclasses import dataclass, asdict

# Import per debugging LLM
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Debug'))
try:
    from llm_debugger import LLMDebugger
    LLM_DEBUGGER_AVAILABLE = True
except ImportError:
    LLMDebugger = None
    LLM_DEBUGGER_AVAILABLE = False

# Import per fine-tuning
import sys
import os

# Aggiungi path per ToolManager
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from tool_manager import ToolManager
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'FineTuning'))

# Import PromptManager per gestione prompt da database
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
try:
    from prompt_manager import PromptManager
    PROMPT_MANAGER_AVAILABLE = True
except ImportError:
    PromptManager = None
    PROMPT_MANAGER_AVAILABLE = False

# Import Tenant class per gestione centralizzata tenant
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from Utils.tenant import Tenant
    TENANT_AVAILABLE = True
except ImportError:
    Tenant = None
    TENANT_AVAILABLE = False

# Import OpenAI Service per supporto GPT-4o
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Services'))
try:
    from openai_service import OpenAIService, sync_chat_completion
    OPENAI_SERVICE_AVAILABLE = True
except ImportError:
    OpenAIService = None
    sync_chat_completion = None
    OPENAI_SERVICE_AVAILABLE = False
try:
    from tokenization_utils import TokenizationManager
    TOKENIZATION_AVAILABLE = True
except ImportError:
    TokenizationManager = None
    TOKENIZATION_AVAILABLE = False
try:
    from mistral_finetuning_manager import MistralFineTuningManager
    FINETUNING_AVAILABLE = True
except ImportError:
    MistralFineTuningManager = None
    FINETUNING_AVAILABLE = False

# Import per MySQL
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MySql'))
    from connettore import MySqlConnettore
    MYSQL_AVAILABLE = True
except ImportError:
    MySqlConnettore = None
    MYSQL_AVAILABLE = False

# Import per Database AI Configuration Service
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Database'))
    from database_ai_config_service import DatabaseAIConfigService
    DATABASE_AI_CONFIG_AVAILABLE = True
except ImportError:
    DatabaseAIConfigService = None
    DATABASE_AI_CONFIG_AVAILABLE = False

# Import per MongoDB - USA IL CONNETTORE GIUSTO
try:
    sys.path.append(os.path.dirname(__file__))
    from mongo_classification_reader import MongoClassificationReader
    MONGODB_AVAILABLE = True
    # DISABILITATO IL VECCHIO CONNETTORE CHE GENERA NOMI SBAGLIATI:
    # from connettore_mongo import MongoDBConnector
    MongoDBConnector = None  # Force disabling
except ImportError:
    MongoClassificationReader = None
    MONGODB_AVAILABLE = False

# Import per BERTopic provider
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TopicModeling'))
    from bertopic_feature_provider import BERTopicFeatureProvider
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTopicFeatureProvider = None
    BERTOPIC_AVAILABLE = False

# 🔍 Import tracing con fallback per evitare crash
try:
    from Utils.tracing import trace_all
except ImportError:
    def trace_all(function_name: str, action: str = "ENTER", called_from: str = None, **kwargs):
        """Fallback tracing function se il modulo principale non è disponibile"""
        pass


def clean_label_text(label: str) -> str:
    """
    Pulisce l'etichetta da caratteri speciali che possono causare problemi nel salvataggio
    
    Scopo della funzione: Rimuove caratteri speciali problematici dalle etichette
    Parametri di input: label (stringa etichetta da pulire)
    Parametri di output: Etichetta pulita
    Valori di ritorno: Stringa depurata da caratteri speciali
    Tracciamento aggiornamenti: 2025-01-27 - Aggiunta pulizia backslash e caratteri speciali
    
    Args:
        label: Etichetta da pulire
        
    Returns:
        Etichetta depurata da caratteri speciali
        
    Autore: Valerio Bignardi
    Data: 2025-09-04
    """
    if not label or not isinstance(label, str):
        return label or ""
    
    cleaned = label.strip()
    if not cleaned:
        return ""
    
    # STEP 1: Rimuovi virgolette esterne (doppie e singole)
    if (cleaned.startswith('"') and cleaned.endswith('"')) or \
       (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1]
    
    # STEP 2: Converti backslash problematici in spazi (per preservare separazione)
    cleaned = cleaned.replace('\\', ' ')
    
    # STEP 3: Rimuovi backslash di escape residui
    cleaned = cleaned.replace('\\"', '"').replace("\\'", "'")
    
    # STEP 4: Sostituisci separatori comuni con underscore
    separators = ['-', '.', '/', '@', '#', '$', '%', '^', '&', '*', '+', '=', '|', ':', ';', '<', '>', '?', '!']
    for sep in separators:
        cleaned = cleaned.replace(sep, '_')
    
    # STEP 5: Rimuovi altri caratteri speciali (ma non spazi e underscore)
    import re
    cleaned = re.sub(r'[^a-zA-Z0-9_\s]', '', cleaned)
    
    # STEP 6: Normalizza spazi multipli e converti in underscore
    cleaned = re.sub(r'\s+', '_', cleaned)
    
    # STEP 7: Rimuovi underscore multipli e ai bordi
    cleaned = re.sub(r'_+', '_', cleaned)
    cleaned = cleaned.strip('_')
    
    # STEP 8: Converti in uppercase per consistenza (opzionale)
    cleaned = cleaned.upper()
    
    return cleaned


@dataclass
class ClassificationResult:
    """Risultato della classificazione LLM"""
    predicted_label: str
    confidence: float
    motivation: str
    method: str = "LLM"
    raw_response: Optional[str] = None
    processing_time: float = 0.0
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte in dizionario per compatibilità"""
        return asdict(self)


class IntelligentClassifier:
    """
    Classificatore LLM intelligente per conversazioni Humanitas
    
    Utilizza Ollama con Mistral 7B per classificazione semantica avanzata
    con sistema di prompt ottimizzato per il dominio ospedaliero.
    """
    
    def __init__(self,
                 ollama_url: str = None,
                 model_name: str = None,
                 temperature: float = None,
                 max_tokens: int = None,
                 timeout: int = None,
                 enable_cache: bool = True,
                 cache_ttl_hours: int = 24,
                 enable_logging: bool = True,
                 embedder=None,
                 semantic_memory=None,
                 config_path: str = None,
                 client_name: str = None,
                 tenant=None,
                 enable_finetuning: bool = True):
        """
        Inizializza il classificatore intelligente
        
        Args:
            ollama_url: URL del server Ollama (se None, legge da config)
            model_name: Nome del modello base da utilizzare (se None, legge da config)
            temperature: Temperatura per la generazione (se None, legge da config)
            max_tokens: Numero massimo di token da generare (se None, legge da config)
            timeout: Timeout per le richieste in secondi (se None, legge da config)
            enable_cache: Se abilitare il caching delle risposte
            cache_ttl_hours: Durata cache in ore
            enable_logging: Se abilitare il logging dettagliato
            embedder: Embedder per calcoli semantici (opzionale)
            semantic_memory: Gestore memoria semantica (opzionale)
            config_path: Percorso file configurazione (opzionale)
            client_name: [DEPRECATED] Nome del cliente - usare tenant invece
            tenant: Oggetto Tenant centralizzato con tutte le informazioni tenant
            enable_finetuning: Se abilitare il fine-tuning automatico
            
        UPGRADE: Preferire l'uso del parametro 'tenant' invece di 'client_name'
        """
        
        # 🏗️ GESTIONE TENANT CENTRALIZZATA
        # Priorità: tenant object > client_name (backwards compatibility)
        if tenant and TENANT_AVAILABLE:
            self.tenant = tenant
            self.client_name = tenant.tenant_slug  # Backwards compatibility
            self.tenant_id = tenant.tenant_id
            if enable_logging:
                print(f"🎯 IntelligentClassifier: Uso tenant centralizzato {tenant}")
        elif client_name and TENANT_AVAILABLE:
            # GESTIONE TENANT DI TEST
            if client_name.endswith('_test'):
                # Client di test: estrae tenant reale e crea tenant temporaneo
                base_tenant_id = client_name.replace('_test', '')
                if enable_logging:
                    print(f"🧪 Rilevato tenant di test: {client_name}, base tenant: {base_tenant_id}")
                try:
                    # Cerca di ottenere tenant reale per copiare configurazione
                    if Tenant._is_valid_uuid(base_tenant_id):
                        base_tenant = Tenant.from_uuid(base_tenant_id)
                    else:
                        base_tenant = Tenant.from_slug(base_tenant_id)
                    
                    # Crea tenant temporaneo per test con configurazione base
                    self.tenant = None  # Nessun tenant reale per i test
                    self.client_name = client_name
                    self.tenant_id = base_tenant_id  # Usa configurazione del tenant base
                    if enable_logging:
                        print(f"🧪 Tenant di test configurato: {client_name} -> config da {base_tenant.tenant_name}")
                except Exception as e:
                    if enable_logging:
                        print(f"⚠️ Errore setup tenant test {client_name}: {e}, uso fallback")
                    self.tenant = None
                    self.client_name = client_name
                    self.tenant_id = "humanitas"  # Fallback sicuro
            else:
                # Legacy mode: crea Tenant da client_name
                try:
                    # Prova prima come slug, poi come UUID
                    if Tenant._is_valid_uuid(client_name):
                        self.tenant = Tenant.from_uuid(client_name)
                    else:
                        self.tenant = Tenant.from_slug(client_name)
                    self.client_name = client_name
                    self.tenant_id = self.tenant.tenant_id
                    if enable_logging:
                        print(f"🔄 IntelligentClassifier: Convertito client_name -> Tenant {self.tenant}")
                except Exception as e:
                    if enable_logging:
                        print(f"⚠️ Fallback: impossibile creare Tenant da {client_name}: {e}")
                    self.tenant = None
                    self.client_name = client_name
                    self.tenant_id = client_name or "humanitas"
        else:
            # Fallback completo: prova a creare Tenant da tenant_id di default
            default_tenant_id = client_name or "humanitas"
            try:
                # Prova a creare un oggetto Tenant da slug "humanitas" per default
                if TENANT_AVAILABLE:
                    self.tenant = Tenant.from_slug(default_tenant_id)
                    self.client_name = self.tenant.tenant_slug
                    self.tenant_id = self.tenant.tenant_id
                    if enable_logging:
                        print(f"🔧 Tenant auto-creato da slug {default_tenant_id}: {self.tenant}")
                else:
                    raise ImportError("Tenant non disponibile")
            except Exception as e:
                if enable_logging:
                    print(f"⚠️ Impossibile auto-creare Tenant da {default_tenant_id}: {e}")
                    print("🔧 Uso fallback senza oggetto Tenant")
                self.tenant = None
                self.client_name = client_name
                self.tenant_id = default_tenant_id
        # Carica configurazione PRIMA di tutto
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        with open(self.config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # Estrai configurazione LLM
        llm_config = self.config.get('llm', {})
        ollama_config = llm_config.get('ollama', {})
        models_config = llm_config.get('models', {})
        generation_config = llm_config.get('generation', {})
        
        # Configurazione client_name: mantieni quello impostato dall'oggetto tenant
        if not hasattr(self, 'client_name') or self.client_name is None:
            self.client_name = client_name
        
        # Carica configurazione tenant-specific LLM
        tenant_llm_config = self.load_tenant_llm_config(self.tenant_id)
        
        # Imposta parametri da config tenant o globale
        self.ollama_url = ollama_url or tenant_llm_config['connection']['url']
        self.timeout = timeout or tenant_llm_config['connection']['timeout']
        self.temperature = temperature or tenant_llm_config['generation']['temperature']
        self.max_tokens = max_tokens or tenant_llm_config['generation']['max_tokens']
        
        # Parametri aggiuntivi da configurazione tenant
        self.top_k = tenant_llm_config['generation']['top_k']
        self.top_p = tenant_llm_config['generation']['top_p']
        self.repeat_penalty = tenant_llm_config['generation']['repeat_penalty']
        
        # Parametri tokenizzazione per LLM (separati da embedding)
        self.llm_tokenization = tenant_llm_config['tokenization']
        
        print(f"🎯 Config LLM caricata per {self.tenant_id} (fonte: {tenant_llm_config['source']})")
        print(f"   📝 Max output tokens: {self.max_tokens}")
        print(f"   📊 Max input tokens: {self.llm_tokenization['max_tokens']}")
        print(f"   🌡️  Temperature: {self.temperature}")
        print(f"   🔢 Top K: {self.top_k}, Top P: {self.top_p}")
        
        # Determina il modello da usare in base al cliente
        # 🔥 PRIORITÀ 1: Parametro esplicito
        if model_name:
            self.model_name = model_name
            print(f"🎯 Uso modello esplicito: {self.model_name}")
        else:
            # 🔥 PRIORITÀ 2: Database (AIConfigurationService)
            database_model = self._load_model_from_database()
            if database_model:
                self.model_name = database_model
                print(f"🎯 Uso modello dal DATABASE per {client_name}: {self.model_name}")
            elif client_name and client_name in models_config.get('clients', {}):
                # 🔥 PRIORITÀ 3: Config.yaml legacy
                self.model_name = models_config['clients'][client_name]
                print(f"🎯 Uso modello specifico LEGACY per {client_name}: {self.model_name}")
            else:
                # 🔥 PRIORITÀ 4: Default fallback
                self.model_name = models_config.get('default', 'mistral:7b')
                print(f"🎯 Uso modello default: {self.model_name}")
        
        # Salva il modello base originale per confronti
        self.base_model_name = self.model_name
        
        # Altri parametri di inizializzazione
        self.enable_cache = enable_cache
        self.cache_ttl_hours = cache_ttl_hours
        self.enable_finetuning = enable_finetuning
        self.enable_logging = enable_logging  # Fix: attributo mancante
        
        # Inizializza LLM Debugger
        self.llm_debugger = None
        if LLM_DEBUGGER_AVAILABLE:
            try:
                self.llm_debugger = LLMDebugger(config_path=config_path)
                if enable_logging and self.llm_debugger.enabled:
                    print(f"🔍 LLM Debugger attivato per IntelligentClassifier")
            except Exception as e:
                if enable_logging:
                    print(f"⚠️ LLM Debugger non disponibile: {e}")
                self.llm_debugger = None
        
        # Fine-tuning manager (se abilitato e disponibile)
        self.finetuning_manager = None
        if self.enable_finetuning and FINETUNING_AVAILABLE:
            try:
                if self.tenant:
                    self.finetuning_manager = MistralFineTuningManager(
                        tenant=self.tenant,
                        config_path=config_path,
                        ollama_url=ollama_url
                    )
                else:
                    print("⚠️ Finetuning richiede oggetto Tenant - disabilitato")
                    self.finetuning_manager = None
                
                # Auto-switch al modello fine-tuned se disponibile per questo cliente
                if self.client_name:
                    finetuned_model = self.finetuning_manager.get_client_model(self.client_name)
                    if finetuned_model:
                        self.model_name = finetuned_model
                        if enable_logging:
                            print(f"🎯 Uso modello fine-tuned per {self.client_name}: {finetuned_model}")
                        
            except Exception as e:
                if enable_logging:
                    print(f"⚠️ Fine-tuning manager non disponibile: {e}")
                self.finetuning_manager = None
        
        # MySQL connector per gestione TAG.tags
        self.mysql_connector = None
        if MYSQL_AVAILABLE:
            try:
                self.mysql_connector = MySqlConnettore()
                if enable_logging:
                    print(f"🗄️ MySQL connector inizializzato per TAG.tags")
            except Exception as e:
                if enable_logging:
                    print(f"⚠️ MySQL connector non disponibile: {e}")
                self.mysql_connector = None

        # ToolManager per recupero tools dal database
        self.tool_manager = None
        try:
            self.tool_manager = ToolManager()
            if enable_logging:
                print(f"🛠️ ToolManager inizializzato per recupero tools dal database")
        except Exception as e:
            if enable_logging:
                print(f"⚠️ ToolManager non disponibile: {e}")
            self.tool_manager = None

        # MongoDB connector (se disponibile) - USA OGGETTO TENANT
        self.mongo_reader = None
        if MONGODB_AVAILABLE:
            try:
                # Verifica che abbiamo un tenant valido per MongoDB
                if self.tenant:
                    self.mongo_reader = MongoClassificationReader(tenant=self.tenant)
                    if enable_logging:
                        print(f"🗄️ MongoDB reader inizializzato per tenant: {self.tenant.tenant_name}")
                else:
                    if enable_logging:
                        print(f"⚠️ MongoDB reader non inizializzato: tenant mancante")
                    self.mongo_reader = None
            except Exception as e:
                if enable_logging:
                    print(f"⚠️ MongoDB reader non disponibile: {e}")
                self.mongo_reader = None
        
        # Carica configurazione
        self.config = self._load_config(config_path)
        
        # Componenti embedding opzionali
        self.embedder = embedder
        self.semantic_memory = semantic_memory
        
        # Configurazioni embedding da config.yaml
        self.enable_embeddings = self.config.get('pipeline', {}).get('intelligent_classifier_embedding', False)
        self.enable_embedding_validation = self.config.get('pipeline', {}).get('embedding_validation', True)
        self.enable_semantic_fallback = self.config.get('pipeline', {}).get('semantic_fallback', True)
        self.enable_new_category_detection = self.config.get('pipeline', {}).get('new_category_detection', True)
        self.embedding_similarity_threshold = self.config.get('pipeline', {}).get('embedding_similarity_threshold', 0.85)
        
        # Configurazioni sistema di fallback BERTopic
        self.bertopic_fallback_threshold = self.config.get('pipeline', {}).get('bertopic_fallback_threshold', 0.70)
        self.new_tag_similarity_threshold = self.config.get('pipeline', {}).get('new_tag_similarity_threshold', 0.75)  
        self.auto_tag_creation = self.config.get('pipeline', {}).get('auto_tag_creation', True)
        self.llm_confidence_threshold = self.config.get('pipeline', {}).get('llm_confidence_threshold', 0.85)
        
        # 🆕 Configurazione debug prompt da config.yaml
        # Parametri input: debug_prompt - Se True mostra prompt LLM, se False nasconde
        # Ultimo aggiornamento: 2025-08-29
        self.debug_prompt = self.config.get('debug', {}).get('debug_prompt', True)
        
        # Inizializzazione BERTopic provider (se disponibile ed abilitato)
        self.bertopic_provider = None
        if (BERTOPIC_AVAILABLE and self.enable_semantic_fallback and 
            self.enable_new_category_detection):
            try:
                self.bertopic_provider = BERTopicFeatureProvider()
                if enable_logging:
                    print(f"🎯 BERTopic provider inizializzato per sistema di fallback intelligente")
            except Exception as e:
                if enable_logging:
                    print(f"⚠️ BERTopic provider non disponibile: {e}")
                self.bertopic_provider = None
        
        # Setup logging
        if enable_logging:
            self.logger = self._setup_logger()
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())
        
        # INTEGRAZIONE MYSQL: Inizializzazione connector database con supporto multi-tenant
        # Usa le informazioni tenant centralizzate se disponibili
        self.mysql_connector = None
        try:
            from TagDatabase.tag_database_connector import TagDatabaseConnector
            if self.tenant:
                # Usa oggetto Tenant completo (PRINCIPIO UNIVERSALE)
                self.mysql_connector = TagDatabaseConnector(tenant=self.tenant)
            else:
                # ERRORE: Non dovrebbe mai succedere con principio universale
                raise ValueError("ERRORE PRINCIPIO UNIVERSALE: self.tenant è None!")
            if enable_logging:
                self.logger.info(f"✅ TagDatabaseConnector inizializzato per tenant {self.tenant_id}")
        except Exception as e:
            if enable_logging:
                self.logger.error(f"❌ Errore inizializzazione TagDatabaseConnector: {e}")
        
        # INIZIALIZZAZIONE PROMPT MANAGER: Sistema database-driven per prompt
        self.prompt_manager = None
        try:
            self.prompt_manager = PromptManager(config_path=config_path)
            if enable_logging:
                self.logger.info(f"✅ PromptManager inizializzato per tenant: {self.tenant_id}")
        except Exception as e:
            if enable_logging:
                self.logger.error(f"❌ Errore inizializzazione PromptManager: {e}")
                self.logger.error("❌ PromptManager obbligatorio - sistema non può funzionare senza configurazione prompt")
        
        # INIZIALIZZAZIONE DATABASE AI CONFIG SERVICE: Configurazioni AI da database
        self.ai_config_service = None
        try:
            if DATABASE_AI_CONFIG_AVAILABLE:
                self.ai_config_service = DatabaseAIConfigService()
                if enable_logging:
                    self.logger.info(f"✅ DatabaseAIConfigService inizializzato per configurazioni AI da database")
            else:
                if enable_logging:
                    self.logger.warning(f"⚠️ DatabaseAIConfigService non disponibile - usando solo config.yaml")
        except Exception as e:
            if enable_logging:
                self.logger.error(f"❌ Errore inizializzazione DatabaseAIConfigService: {e}")
                self.logger.warning("⚠️ Fallback a config.yaml per configurazioni batch processing")
            self.prompt_manager = None
        
        # VALIDAZIONE PROMPT OBBLIGATORI per il tenant
        self._validate_required_prompts()
        
        # Cache per le predizioni
        self._prediction_cache = {}
        self._cache_lock = threading.Lock()
        
        # Cache semantica per embedding (se abilitata)
        self._semantic_cache = []
        self._semantic_cache_lock = threading.Lock()
        
        # Cache embedding per esempi curati (per performance)
        self._example_embeddings_cache = None
        
        # Session HTTP con retry automatici
        self.session = self._setup_http_session()
        
        # Stato di disponibilità
        self._is_available = None
        self._last_availability_check = None
        self._availability_cache_duration = 300  # 5 minuti
        
        # Statistiche
        self.stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'semantic_cache_hits': 0,
            'embedding_fallback_used': 0,
            'llm_corrections': 0,
            'new_categories_detected': 0,
            'errors': 0,
            'avg_response_time': 0.0,
            'response_times': []
        }
        
        # Carica etichette dinamicamente dal database TAG.tags
        self.domain_labels = []
        self.label_descriptions = {}  # Mappa tag_name -> tag_description
        self._load_domain_labels_from_database()
        
        # Fallback etichette di base se DB vuoto (prima esecuzione)
        if not self.domain_labels:
            self.domain_labels = ['altro']  # Solo etichetta base
            if enable_logging:
                print(f"⚠️ Nessuna etichetta trovata in TAG.tags - Prima esecuzione")
        else:
            if enable_logging:
                print(f"✅ Caricate {len(self.domain_labels)} etichette da TAG.tags")

        # Inizializzazione TokenizationManager per gestione conversazioni lunghe
        self.tokenizer = None
        if TOKENIZATION_AVAILABLE:
            try:
                self.tokenizer = TokenizationManager()
                if enable_logging:
                    print(f"✅ TokenizationManager integrato per gestione conversazioni lunghe")
            except Exception as e:
                if enable_logging:
                    print(f"⚠️ Errore inizializzazione TokenizationManager: {e}")
                self.tokenizer = None
        else:
            if enable_logging:
                print(f"⚠️ TokenizationManager non disponibile")























        # Esempi curati per few-shot learning Wopta - Assicurazione Vita (senza duplicati/inconsistenze)
        self.curated_examples = [
            {
                "text": "Come faccio a disdire la polizza vita?",
                "label": "disdetta_polizza",
                "motivation": "L'utente vuole disdire la sua polizza vita"
            },
            {
                "text": "Quanti anni può durare al massimo la polizza vita?",
                "label": "info_durata_polizza",
                "motivation": "Richiesta di informazioni sulla durata della polizza vita"
            },
            {
                "text": "Il beneficiario della polizza deve essere mio figlio che deve ancora nascere, come posso fare?",
                "label": "info_beneficiario",
                "motivation": "Domanda su come impostare un beneficiario non ancora nato"
            },
            {
                "text": "Devo cambiare la mail con cui mi sono registrato al portale",
                "label": "cambio_anagrafica",
                "motivation": "Cambio delle informazioni anagrafiche richiesto"
            },
            {
                "text": "Quando inserisco il CF dice che non è valido",
                "label": "problema_accesso_portale",
                "motivation": "Problema tecnico di accesso al portale online"
            },
            {
                "text": "Non riesco ad entrare nel mio account del portale Wopta",
                "label": "problema_accesso_portale",
                "motivation": "Difficoltà di login al portale assicurazione"
            },
            {
                "text": "Cosa copre esattamente la polizza vita Wopta?",
                "label": "info_condizioni_polizza",
                "motivation": "Richiesta di informazioni sulle condizioni di polizza"
            },
            {
                "text": "In caso di morte, come funziona l'indennizzo?",
                "label": "info_caso_morte",
                "motivation": "Richiesta informazioni su come funziona la copertura in caso di morte"
            },
            {
                "text": "La polizza copre anche viaggi all'estero?",
                "label": "info_copertura_geografica_assicurazione",
                "motivation": "Richiesta informazioni sulla copertura geografica"
            },
            {
                "text": "Devo fare visite mediche per attivare la polizza?",
                "label": "info_questionario_medico",
                "motivation": "Domande su questionari o visite mediche richieste"
            },
            {
                "text": "Come posso contattare l'assistenza Wopta?",
                "label": "info_contatti",
                "motivation": "Domanda sui contatti dell'assicurazione"
            },
            {
                "text": "Mi dai l'indirizzo della sede di Wopta?",
                "label": "info_sede_aziendale",
                "motivation": "Richiesta informazioni sulla sede aziendale"
            },
            {
                "text": "Su che sito posso aprire la polizza?",
                "label": "info_sottoscrizione_online",
                "motivation": "Informazioni su come sottoscrivere la polizza online"
            },
            {
                "text": "Non ho ricevuto la documentazione della polizza",
                "label": "problema_amministrativo",
                "motivation": "Problemi con documenti o questioni amministrative"
            },
            {
                "text": "Posso modificare l'importo assicurato dopo la sottoscrizione?",
                "label": "modifica_condizioni_polizza",
                "motivation": "Richiesta di modifica delle condizioni contrattuali"
            },
            {
                "text": "Quanto costa la polizza vita per una persona di 35 anni?",
                "label": "info_costi_premio",
                "motivation": "Richiesta informazioni sui costi e premi assicurativi"
            },
            {
                "text": "Come funziona il riscatto della polizza?",
                "label": "info_riscatto_polizza",
                "motivation": "Richiesta informazioni sul riscatto della polizza"
            },
            {
                "text": "Come faccio a raggiungervi? Parto da Milano",
                "label": "indicazioni_stradali",
                "motivation": "Informazioni su come raggiungere la sede Wopta"
            }
        ]
        
        self.logger.info(f"IntelligentClassifier inizializzato - Model: {model_name}, URL: {ollama_url}")
        self.logger.info(f"Embedding features: {self.enable_embeddings}, Validation: {self.enable_embedding_validation}")
        
        # 🤖 NUOVO: Inizializzazione servizio OpenAI per modelli GPT
        self.openai_service = None
        if OPENAI_SERVICE_AVAILABLE:
            try:
                # Determina se il modello corrente è OpenAI
                self.is_openai_model = self._is_openai_model(self.model_name)
                
                if self.is_openai_model:
                    # Inizializza servizio OpenAI (legge configurazione da .env per Azure)
                    # OpenAIService ora gestisce automaticamente Azure OpenAI vs OpenAI standard
                    openai_config = self.config.get('llm', {}).get('openai', {})
                    self.openai_service = OpenAIService(
                        max_parallel_calls=openai_config.get('max_parallel_calls', 200),
                        rate_limit_per_minute=openai_config.get('rate_limit_per_minute', 10000)
                    )
                    if enable_logging:
                        print(f"🤖 [OpenAI] Servizio inizializzato per modello: {self.model_name}")
                        print(f"🤖 [OpenAI] Max parallel calls: {openai_config.get('max_parallel_calls', 200)}")
                else:
                    if enable_logging:
                        print(f"🔧 [Ollama] Modello rilevato: {self.model_name}")
                        
            except Exception as e:
                if enable_logging:
                    print(f"⚠️ [OpenAI] Servizio non disponibile: {e}")
                self.openai_service = None
                self.is_openai_model = False
        else:
            self.is_openai_model = False
            if enable_logging:
                print("⚠️ [OpenAI] Service non disponibile - usando solo Ollama")
    
    def _is_openai_model(self, model_name: str) -> bool:
        """
        Determina se il modello specificato è un modello OpenAI
        
        Args:
            model_name: Nome del modello da verificare
            
        Returns:
            True se è un modello OpenAI, False altrimenti
            
        Data ultima modifica: 2025-01-31
        """
        # 🔍 TRACING ENTER
        trace_all("_is_openai_model", "ENTER", 
                 called_from="__init__",
                 model_name=model_name if model_name else "None",
                 tenant_id=getattr(self, 'tenant_id', 'unknown'))
        
        if not model_name:
            trace_all("_is_openai_model", "EXIT", 
                     called_from="__init__",
                     result=False,
                     exit_reason="EMPTY_MODEL_NAME")
            return False
            
        # Lista modelli OpenAI supportati
        openai_models = ['gpt-4o', 'gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo']
        
        # Verifica diretta nel nome
        if model_name in openai_models:
            trace_all("_is_openai_model", "EXIT", 
                     called_from="__init__",
                     result=True,
                     exit_reason="DIRECT_MATCH",
                     matched_model=model_name)
            return True
            
        # Verifica anche nella configurazione modelli
        try:
            available_models = self.config.get('llm', {}).get('models', {}).get('available', [])
            for model_config in available_models:
                if isinstance(model_config, dict):
                    if (model_config.get('name') == model_name and 
                        model_config.get('provider') == 'openai'):
                        trace_all("_is_openai_model", "EXIT", 
                                 called_from="__init__",
                                 result=True,
                                 exit_reason="CONFIG_MATCH",
                                 provider="openai")
                        return True
                elif isinstance(model_config, str) and model_config == model_name:
                    # Fallback per modelli senza provider specificato
                    is_gpt = model_name.startswith('gpt-')
                    if is_gpt:
                        trace_all("_is_openai_model", "EXIT", 
                                 called_from="__init__",
                                 result=True,
                                 exit_reason="GPT_PREFIX_MATCH")
                        return True
        except Exception as e:
            trace_all("_is_openai_model", "ERROR", 
                     called_from="__init__",
                     error_type=type(e).__name__,
                     error_message=str(e))
            print(f"⚠️ Errore verifica modello OpenAI: {e}")
            
        # Fallback: verifica se il nome inizia con 'gpt-'
        is_gpt_fallback = model_name.startswith('gpt-')
        trace_all("_is_openai_model", "EXIT", 
                 called_from="__init__",
                 result=is_gpt_fallback,
                 exit_reason="GPT_PREFIX_FALLBACK")
        return is_gpt_fallback
    
    def _setup_logger(self) -> logging.Logger:
        """Setup del logger con configurazione appropriata"""
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_http_session(self) -> requests.Session:
        """Setup session HTTP con retry automatici"""
        session = requests.Session()
        
        # Strategia di retry
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def load_tenant_llm_config(self, tenant_id: str = None) -> Dict[str, Any]:
        """
        Carica configurazione LLM specifica per tenant
        
        Args:
            tenant_id: ID del tenant (se None, usa self.tenant_id)
            
        Returns:
            Dizionario con parametri LLM per il tenant
            
        Data ultima modifica: 2025-08-31
        """
        tenant_id = tenant_id or self.tenant_id
        
        try:
            # Configurazione LLM globale come base
            llm_config = self.config.get('llm', {})
            generation_config = llm_config.get('generation', {})
            tokenization_config = llm_config.get('tokenization', {})
            connection_config = llm_config.get('ollama', {})
            openai_config = llm_config.get('openai', {})
            
            # Configurazione tenant-specific
            tenant_configs = self.config.get('tenant_configs', {})
            tenant_config = tenant_configs.get(tenant_id, {})
            tenant_llm_params = tenant_config.get('llm_parameters', {})
            
            if tenant_llm_params:
                print(f"📋 Caricamento parametri LLM specifici per tenant: {tenant_id}")
                
                # Merge configurazioni con priorità a tenant
                tenant_generation = tenant_llm_params.get('generation', {})
                tenant_tokenization = tenant_llm_params.get('tokenization', {})
                tenant_connection = tenant_llm_params.get('connection', {})
                tenant_openai = tenant_llm_params.get('openai', {})
                
                merged_config = {
                    'generation': {
                        'max_tokens': tenant_generation.get('max_tokens', generation_config.get('max_tokens', 150)),
                        'temperature': tenant_generation.get('temperature', generation_config.get('temperature', 0.1)),
                        'top_k': tenant_generation.get('top_k', generation_config.get('top_k', 40)),
                        'top_p': tenant_generation.get('top_p', generation_config.get('top_p', 0.9)),
                        'repeat_penalty': tenant_generation.get('repeat_penalty', generation_config.get('repeat_penalty', 1.1))
                    },
                    'tokenization': {
                        'max_tokens': tenant_tokenization.get('max_tokens', tokenization_config.get('max_tokens', 8000)),
                        'model_name': tokenization_config.get('model_name', 'cl100k_base'),
                        'truncation_strategy': tokenization_config.get('truncation_strategy', 'start')
                    },
                    'connection': {
                        'timeout': tenant_connection.get('timeout', connection_config.get('timeout', 300)),
                        'url': connection_config.get('url', 'http://localhost:11434')
                    },
                    'openai': {
                        'parallel_calls_max': tenant_openai.get('parallel_calls_max', openai_config.get('max_parallel_calls', 200))
                    },
                    'source': 'tenant_specific'
                }
                
                return merged_config
            
            else:
                print(f"📋 Caricamento parametri LLM globali per tenant: {tenant_id}")
                return {
                    'generation': {
                        'max_tokens': generation_config.get('max_tokens', 150),
                        'temperature': generation_config.get('temperature', 0.1),
                        'top_k': generation_config.get('top_k', 40),
                        'top_p': generation_config.get('top_p', 0.9),
                        'repeat_penalty': generation_config.get('repeat_penalty', 1.1)
                    },
                    'tokenization': {
                        'max_tokens': tokenization_config.get('max_tokens', 8000),
                        'model_name': tokenization_config.get('model_name', 'cl100k_base'),
                        'truncation_strategy': tokenization_config.get('truncation_strategy', 'start')
                    },
                    'connection': {
                        'timeout': connection_config.get('timeout', 300),
                        'url': connection_config.get('url', 'http://localhost:11434')
                    },
                    'openai': {
                        'parallel_calls_max': openai_config.get('max_parallel_calls', 200)
                    },
                    'source': 'global'
                }
                
        except Exception as e:
            print(f"⚠️  Errore caricamento config LLM per tenant {tenant_id}: {e}")
            return {
                'generation': {'max_tokens': 150, 'temperature': 0.1, 'top_k': 40, 'top_p': 0.9, 'repeat_penalty': 1.1},
                'tokenization': {'max_tokens': 8000, 'model_name': 'cl100k_base', 'truncation_strategy': 'start'},
                'connection': {'timeout': 300, 'url': 'http://localhost:11434'},
                'openai': {'parallel_calls_max': 200},
                'source': 'fallback'
            }
    
    def is_available(self) -> bool:
        """
        Verifica se il servizio LLM è disponibile (Ollama o OpenAI)
        
        Scopo della funzione:
        - Verificare la disponibilità del servizio LLM (Ollama o OpenAI)
        - Validare che il modello specifico sia disponibile
        - Implementare cache per ottimizzare le verifiche ripetute
        
        Parametri di input e output:
        - self: istanza corrente del classificatore
        
        Valori di ritorno:
        - bool: True se il servizio LLM e il modello sono disponibili, False altrimenti
        
        Tracciamento aggiornamenti:
        - 2025-09-06: Aggiunto tracing completo ENTER/EXIT/ERROR
        - 2025-09-07: Aggiunto supporto OpenAI per modelli GPT-4o
        
        Returns:
            True se il servizio è disponibile, False altrimenti
        """
        
        # 🔍 TRACING ENTER
        trace_all("is_available", "ENTER", 
                 called_from="external_api_check",
                 ollama_url=self.ollama_url,
                 model_name=self.model_name,
                 is_openai_model=getattr(self, 'is_openai_model', False),
                 has_cache=bool(self._last_availability_check),
                 cache_duration=self._availability_cache_duration)
        
        # Cache del controllo di disponibilità
        now = datetime.now()
        if (self._last_availability_check and 
            (now - self._last_availability_check).total_seconds() < self._availability_cache_duration):
            
            # 🔍 TRACING EXIT - Cache Hit
            trace_all("is_available", "EXIT", 
                     called_from="external_api_check",
                     result=self._is_available,
                     source="CACHE",
                     seconds_since_last_check=(now - self._last_availability_check).total_seconds(),
                     exit_reason="CACHE_HIT")
            
            return self._is_available
        
        try:
            # 🤖 NUOVO: Gestione OpenAI vs Ollama
            if getattr(self, 'is_openai_model', False):
                # Verifica disponibilità OpenAI
                if hasattr(self, 'openai_service') and self.openai_service:
                    # Per OpenAI, assumiamo sempre disponibile se il servizio è inizializzato
                    # (la verifica reale avviene al momento della chiamata API)
                    self._is_available = True
                    self.logger.debug(f"🤖 Servizio OpenAI disponibile per modello {self.model_name}")
                    
                    # 🔍 TRACING EXIT - OpenAI Success
                    trace_all("is_available", "EXIT", 
                             called_from="external_api_check",
                             result=True,
                             source="OPENAI_SERVICE",
                             model_name=self.model_name,
                             exit_reason="OPENAI_AVAILABLE")
                else:
                    self._is_available = False
                    self.logger.warning(f"🤖 Servizio OpenAI non inizializzato per modello {self.model_name}")
                    
                    # 🔍 TRACING EXIT - OpenAI Fail
                    trace_all("is_available", "EXIT", 
                             called_from="external_api_check",
                             result=False,
                             source="OPENAI_SERVICE",
                             exit_reason="OPENAI_SERVICE_MISSING")
            else:
                # Logica originale per Ollama
                response = self.session.get(
                    f"{self.ollama_url}/api/tags",
                    timeout=5
                )
                
                if response.status_code == 200:
                    # Verifica che il modello sia disponibile
                    tags_data = response.json()
                    available_models = [model['name'] for model in tags_data.get('models', [])]
                    
                    self._is_available = self.model_name in available_models
                    
                    # Se il modello non viene trovato, prova con :latest
                    if not self._is_available and not self.model_name.endswith(':latest'):
                        model_with_latest = f"{self.model_name}:latest"
                        if model_with_latest in available_models:
                            self.logger.info(f"🔄 Modello trovato con suffisso :latest: {model_with_latest}")
                            self.model_name = model_with_latest
                            self._is_available = True
                    
                    if not self._is_available:
                        self.logger.warning(f"Modello {self.model_name} non trovato. Disponibili: {available_models}")
                    else:
                        self.logger.debug(f"Servizio Ollama disponibile con modello {self.model_name}")
                        
                    # 🔍 TRACING EXIT - Success
                    trace_all("is_available", "EXIT", 
                             called_from="external_api_check",
                             result=self._is_available,
                             source="OLLAMA_API",
                             status_code=response.status_code,
                             available_models_count=len(available_models),
                             model_found=self._is_available,
                             exit_reason="SUCCESS")
                    
                else:
                    self._is_available = False
                    self.logger.warning(f"Ollama non disponibile - Status: {response.status_code}")
                    
                    # 🔍 TRACING EXIT - Bad Status
                    trace_all("is_available", "EXIT", 
                             called_from="external_api_check",
                             result=False,
                             source="OLLAMA_API",
                             status_code=response.status_code,
                             exit_reason="BAD_STATUS_CODE")
        
        except Exception as e:
            self._is_available = False
            self.logger.warning(f"Errore nella verifica disponibilità LLM: {e}")
            
            # 🔍 TRACING ERROR
            trace_all("is_available", "ERROR", 
                     called_from="external_api_check",
                     error_type=type(e).__name__,
                     error_message=str(e))
            
            # 🔍 TRACING EXIT - Error
            trace_all("is_available", "EXIT", 
                     called_from="external_api_check",
                     result=False,
                     source="ERROR",
                     exit_reason="EXCEPTION")
        
        self._last_availability_check = now
        return self._is_available
    
    def _generate_cache_key(self, text: str, context: Optional[str] = None) -> str:
        """Genera chiave cache per il testo"""
        # 🔍 TRACING ENTER
        trace_all("_generate_cache_key", "ENTER", 
                 called_from="classify_with_motivation",
                 text_length=len(text) if text else 0,
                 context_provided=context is not None,
                 model_name=self.model_name)
        
        content = f"{text}|{context or ''}|{self.model_name}|{self.temperature}"
        cache_key = hashlib.md5(content.encode()).hexdigest()
        
        # 🔍 TRACING EXIT
        trace_all("_generate_cache_key", "EXIT", 
                 called_from="classify_with_motivation",
                 cache_key_prefix=cache_key[:8],
                 content_hash_length=len(cache_key))
        
        return cache_key
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[ClassificationResult]:
        """Recupera predizione dalla cache se valida"""
        # 🔍 TRACING ENTER
        trace_all("_get_cached_prediction", "ENTER", 
                 called_from="classify_with_motivation",
                 cache_key_prefix=cache_key[:8] if cache_key else "None",
                 cache_enabled=self.enable_cache,
                 cache_size=len(self._prediction_cache))
        
        if not self.enable_cache:
            trace_all("_get_cached_prediction", "EXIT", 
                     called_from="classify_with_motivation",
                     exit_reason="CACHE_DISABLED",
                     result=None)
            return None
        
        with self._cache_lock:
            if cache_key in self._prediction_cache:
                cached_result, timestamp = self._prediction_cache[cache_key]
                
                # Verifica se la cache è ancora valida
                if datetime.now() - timestamp < timedelta(hours=self.cache_ttl_hours):
                    self.stats['cache_hits'] += 1
                    self.logger.debug(f"Cache hit per chiave {cache_key[:8]}...")
                    
                    trace_all("_get_cached_prediction", "EXIT", 
                             called_from="classify_with_motivation",
                             exit_reason="CACHE_HIT",
                             cached_label=cached_result.predicted_label,
                             cache_age_hours=(datetime.now() - timestamp).total_seconds() / 3600)
                    return cached_result
                else:
                    # Rimuovi cache scaduta
                    del self._prediction_cache[cache_key]
        
        trace_all("_get_cached_prediction", "EXIT", 
                 called_from="classify_with_motivation",
                 exit_reason="CACHE_MISS",
                 result=None)
        return None
    
    def _cache_prediction(self, cache_key: str, result: ClassificationResult) -> None:
        """Salva predizione nella cache"""
        if not self.enable_cache:
            return
        
        with self._cache_lock:
            self._prediction_cache[cache_key] = (result, datetime.now())
            
            # Limita la dimensione della cache (max 1000 elementi)
            if len(self._prediction_cache) > 1000:
                # Rimuovi gli elementi più vecchi
                sorted_items = sorted(
                    self._prediction_cache.items(),
                    key=lambda x: x[1][1]  # Ordina per timestamp
                )
                # Mantieni solo i 800 più recenti
                self._prediction_cache = dict(sorted_items[-800:])
    
    def _validate_required_prompts(self) -> None:
        """
        Log dei prompt disponibili - non blocca più le operazioni
        """
        try:
            if self.prompt_manager:
                self.logger.info(f"ℹ️  PromptManager disponibile per tenant {self.tenant_id}")
            else:
                self.logger.info(f"ℹ️  PromptManager non disponibile per tenant {self.tenant_id} - usando fallback ML")
        except Exception as e:
            self.logger.info(f"ℹ️  Info prompt per tenant {self.tenant_id}: {e}")
    
    def add_new_label_to_database(self, label_name: str, label_description: str = None) -> bool:
        """
        Aggiunge una nuova etichetta al database TAG se non esiste già
        
        Args:
            label_name: Nome della nuova etichetta
            label_description: Descrizione opzionale dell'etichetta
            
        Returns:
            True se l'etichetta è stata aggiunta con successo o esiste già
        """
        try:
            from TagDatabase.tag_database_connector import TagDatabaseConnector
            
            # Usa oggetto Tenant completo (PRINCIPIO UNIVERSALE)
            if not self.tenant:
                raise ValueError("ERRORE PRINCIPIO UNIVERSALE: self.tenant è None!")
            
            # Normalizza in forma canonica MAIUSCOLA
            cleaned_label = clean_label_text(label_name)
            if cleaned_label != label_name:
                self.logger.info(f"🧹 Etichetta pulita prima dell'inserimento: '{label_name}' → '{cleaned_label}'")

            description = label_description or f"Etichetta generata automaticamente: {cleaned_label}"
            tag_db = TagDatabaseConnector(tenant=self.tenant)
            success = tag_db.add_tag_if_not_exists(tag_name=cleaned_label, tag_description=description)
            return bool(success)
                    
        except Exception as e:
            self.logger.error(f"Errore nell'aggiunta dell'etichetta al database: {e}")
            return False
    
    def _get_available_labels(self) -> str:
        """
        Recupera le etichette disponibili dal database TAG o usa quelle hardcoded come fallback
        
        Scopo della funzione:
        - Recuperare tutte le etichette disponibili per il tenant dal database TAG
        - Fallback a etichette hardcoded se database non disponibile
        
        Parametri di input e output:
        - self: istanza corrente del classificatore
        
        Valori di ritorno:
        - str: String formattata con etichette separate da " | "
        
        Tracciamento aggiornamenti:
        - 2025-09-06: Aggiunto tracing completo ENTER/EXIT/ERROR
        
        Returns:
            String formattata con le etichette disponibili per il system message
        """
        
        # 🔍 TRACING ENTER
        trace_all("_get_available_labels", "ENTER", 
                 called_from="classify_with_motivation",
                 tenant_id=str(self.tenant_id),
                 has_tenant_object=bool(self.tenant),
                 hardcoded_labels_count=len(self.domain_labels) if self.domain_labels else 0)
        
        try:
            # Tenta di caricare le etichette dal database TAG
            from TagDatabase.tag_database_connector import TagDatabaseConnector
            
            # Usa oggetto Tenant completo (PRINCIPIO UNIVERSALE)
            if not self.tenant:
                raise ValueError("ERRORE PRINCIPIO UNIVERSALE: self.tenant è None!")
            
            tag_db = TagDatabaseConnector(tenant=self.tenant)
            if tag_db.connetti():
                # Query per ottenere tutte le etichette attive dal database per il tenant
                query = "SELECT tag_name FROM tags WHERE tenant_id = %s ORDER BY tag_name"
                result = tag_db.esegui_query(query, (self.tenant_id,))
                
                if result and len(result) > 0:
                    # Estrae i nomi delle etichette dal risultato della query
                    db_labels = [row[0] for row in result]
                    labels_string = " | ".join(db_labels)
                    
                    self.logger.info(f"Etichette caricate dal database TAG: {len(db_labels)} etichette trovate")
                    tag_db.disconnetti()
                    
                    # 🔍 TRACING EXIT - Database Success
                    trace_all("_get_available_labels", "EXIT", 
                             called_from="classify_with_motivation",
                             source="DATABASE_TAG",
                             labels_count=len(db_labels),
                             labels_preview=labels_string[:100] + "..." if len(labels_string) > 100 else labels_string,
                             exit_reason="DATABASE_SUCCESS")
                    
                    return labels_string
                
                tag_db.disconnetti()
        
        except Exception as e:
            self.logger.warning(f"Errore nel caricamento etichette dal database TAG: {e}")
            
            # 🔍 TRACING ERROR
            trace_all("_get_available_labels", "ERROR", 
                     called_from="classify_with_motivation",
                     error_type=type(e).__name__,
                     error_message=str(e))
        
        # Fallback alle etichette hardcoded se il database non è disponibile
        hardcoded_labels = " | ".join(self.domain_labels)
        self.logger.info("Utilizzando etichette hardcoded come fallback")
        
        # 🔍 TRACING EXIT - Fallback
        trace_all("_get_available_labels", "EXIT", 
                 called_from="classify_with_motivation",
                 source="HARDCODED_FALLBACK",
                 labels_count=len(self.domain_labels) if self.domain_labels else 0,
                 labels_preview=hardcoded_labels[:100] + "..." if len(hardcoded_labels) > 100 else hardcoded_labels,
                 exit_reason="FALLBACK")
        
        return hardcoded_labels

    def _get_priority_labels_hint(self) -> str:
        """
        Genera hint sulle etichette più frequenti per guidare meglio il LLM
        
        Returns:
            String con hint delle etichette prioritarie
        """
        try:
            # Tenta di recuperare statistiche dal database per identificare pattern
            from MySql.connettore import ConnettoreDB
            
            connector = ConnettoreDB()
            if connector.connetti():
                # Query per le etichette più frequenti negli ultimi 30 giorni
                query = """
                SELECT predicted_label, COUNT(*) as freq 
                FROM session_classifications 
                WHERE prediction_timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                    AND predicted_label != 'altro'
                GROUP BY predicted_label 
                ORDER BY freq DESC 
                LIMIT 5
                """
                
                result = connector.esegui_query(query)
                if result and len(result) > 0:
                    top_labels = [row[0] for row in result[:3]]  # Top 3
                    
                    hint = f"""
ETICHETTE FREQUENTI (ultimi 30gg): {' | '.join(top_labels)}
- Considera queste prime se il testo è ambiguo"""
                    
                    connector.disconnetti()
                    return hint
                
                connector.disconnetti()
        
        except Exception as e:
            self.logger.debug(f"Impossibile recuperare statistiche etichette: {e}")
        
        # Fallback con hint statico basato sulla conoscenza del dominio
        return """
            ETICHETTE COMUNI: ritiro_cartella_clinica_referti | prenotazione_esami | info_contatti
            - La maggior parte delle richieste riguarda questi 3 intenti principali
        """

    def _build_system_message(self, conversation_context: Optional[str] = None) -> str:
        """
        Costruisce il system message intelligente per guidare il comportamento del LLM
        Utilizza PromptManager per caricare prompt dal database con supporto multi-tenant
        
        Args:
            conversation_context: Contesto aggiuntivo per specializzare il prompt
        
        Returns:
            System message ottimizzato e context-aware
        """
        # Tenta di utilizzare il PromptManager per caricare il prompt dal database
        if self.prompt_manager:
            try:
                # Variabili dinamiche per il sistema LLM
                variables = {
                    'available_labels': self._get_available_labels(),
                    'priority_labels': self._get_priority_labels_hint(),
                    'context_guidance': f"\nCONTESTO SPECIFICO: {conversation_context}" if conversation_context else ""
                }
                
                # Carica prompt dal database con validazione STRICT
                try:
                    system_prompt = self.prompt_manager.get_prompt_strict(
                        self.tenant,  # 🔧 FIX: Passa oggetto Tenant invece di tenant_id
                        engine="LLM",
                        prompt_type="SYSTEM", 
                        prompt_name="intelligent_classifier_system",
                        variables=variables
                    )
                    
                    self.logger.info(f"✅ Prompt SYSTEM caricato da database per tenant {self.tenant_id}")
                    
                    # 🔍 STAMPA DETTAGLIATA PROMPT SYSTEM (solo se debug_prompt=True)
                    if self.debug_prompt:
                        print("\n" + "="*80)
                        print("🤖 DEBUG PROMPT SYSTEM - DATABASE")
                        print("="*80)
                        print(f"📋 Prompt Name: LLM/SYSTEM/intelligent_classifier_system")
                        print(f"🏢 Tenant ID: {self.tenant_id}")
                        print(f"📝 Variables Used: {list(variables.keys())}")
                        print("-"*80)
                        print("📄 SYSTEM PROMPT CONTENT (dopo sostituzione placeholder):")
                        print("-"*80)
                        print(system_prompt)
                        print("="*80)
                    else:
                        print(f"🤖 System prompt caricato per tenant {self.tenant_id} (debug_prompt=False)")
                    
                    return system_prompt
                    
                except Exception as e:
                    error_msg = (
                        f"❌ ERRORE CRITICO: Prompt SYSTEM obbligatorio non trovato per tenant {self.tenant_id}. "
                        f"Dettaglio: {e}. "
                        f"Configurazione prompt SYSTEM richiesta nel database: wopta:LLM:SYSTEM:intelligent_classifier_system"
                    )
                    if hasattr(self, 'logger'):
                        self.logger.error(error_msg)
                    
                    print(error_msg)
                    raise Exception(error_msg)
                    
            except AttributeError:
                error_msg = (
                    f"❌ ERRORE CRITICO: PromptManager non inizializzato per tenant {self.tenant_id}. "
                    f"Verificare configurazione database e connessione MySQL per SYSTEM prompt."
                )
                if hasattr(self, 'logger'):
                    self.logger.error(error_msg)
                print(error_msg)
                raise Exception(error_msg)
        
        # NESSUN FALLBACK - SE PromptManager NON DISPONIBILE IL SISTEMA SI DEVE BLOCCARE
        error_msg = (
            f"❌ ERRORE CRITICO: PromptManager non disponibile per SYSTEM prompt tenant {self.tenant_id}. "
            f"Sistema non può funzionare senza prompt DATABASE-DRIVEN."
        )
        if hasattr(self, 'logger'):
            self.logger.error(error_msg)
        print(error_msg)
        raise Exception(error_msg)
    
    def _get_system_prompt(self, conversation_context: str) -> str:
        """
        Carica il prompt di sistema dal database utilizzando PromptManager
        
        Scopo della funzione:
        - Caricare il prompt SYSTEM dal database per il tenant corrente
        - Segnalare errori di configurazione senza fallback nascosti
        
        Parametri di input e output:
        - conversation_context: str - Contesto della conversazione da classificare
        
        Valori di ritorno:
        - str: Prompt di sistema caricato dal database
        
        Tracciamento aggiornamenti:
        - 2025-08-25: Correzione per caricare SOLO dal database
        
        Args:
            conversation_context: Contesto della conversazione
            
        Returns:
            str: Prompt di sistema dal database
            
        Raises:
            Exception: Se il prompt non può essere caricato dal database
        """
        if not self.prompt_manager:
            error_msg = f"❌ ERRORE CRITICO: PromptManager non inizializzato per tenant {self.tenant_id}"
            print(error_msg)
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        try:
            # 🎯 USA TENANT CENTRALIZZATO invece di _resolve_tenant_id
            if self.tenant:
                resolved_tenant_id = self.tenant.tenant_id
            else:
                # Fallback legacy con risoluzione
                resolved_tenant_id = self.prompt_manager._resolve_tenant_id(self.tenant_id)
            
            # Variabili dinamiche per il prompt
            variables = {
                'available_labels': self._get_available_labels(),
                'priority_labels': self._get_priority_labels_hint(),
                'context_guidance': f"\nCONTESTO SPECIFICO: {conversation_context}" if conversation_context else ""
            }
            
            # Carica prompt dal database con validazione STRICT
            system_prompt = self.prompt_manager.get_prompt_strict(
                self.tenant,  # 🔧 FIX: Passa oggetto Tenant invece di tenant_id
                engine="LLM",
                prompt_type="SYSTEM",
                prompt_name="intelligent_classifier_system",
                variables=variables
            )
            
            self.logger.info(f"✅ Prompt SYSTEM caricato da database per tenant {self.tenant_id}")
            return system_prompt
            
        except Exception as e:
            error_msg = f"❌ ERRORE CARICAMENTO PROMPT: Impossibile caricare prompt SYSTEM per tenant {self.tenant_id}. Dettaglio: {e}"
            print(error_msg)
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def _get_user_prompt(self, conversation_text: str, context: Optional[str] = None) -> str:
        """
        Carica il prompt USER dal database utilizzando PromptManager
        
        Scopo della funzione:
        - Caricare il prompt USER/TEMPLATE dal database per il tenant corrente
        - Segnalare errori di configurazione senza fallback nascosti
        
        Parametri di input e output:
        - conversation_text: str - Testo da classificare
        - context: Optional[str] - Contesto opzionale aggiuntivo
        
        Valori di ritorno:
        - str: Prompt USER caricato dal database
        
        Tracciamento aggiornamenti:
        - 2025-08-25: Creazione metodo per caricare SOLO dal database
        
        Args:
            conversation_text: Testo da classificare
            context: Contesto opzionale
            
        Returns:
            str: Prompt USER dal database
            
        Raises:
            Exception: Se il prompt non può essere caricato dal database
        """
        if not self.prompt_manager:
            error_msg = f"❌ ERRORE CRITICO: PromptManager non inizializzato per tenant {self.tenant_id}"
            print(error_msg)
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        try:
            # 🎯 USA TENANT CENTRALIZZATO - controlla che esista l'oggetto tenant
            if not self.tenant:
                raise Exception(f"Oggetto tenant non disponibile per {self.tenant_id}")
            
            # Riassumi se troppo lungo
            processed_text = self._summarize_if_long(conversation_text)
            
            # Seleziona esempi dinamici
            examples = self._get_dynamic_examples(conversation_text, max_examples=5)
            
            # 🔧 CORREZIONE CRITICA: USA IL FORMATO ORIGINALE DAL DATABASE!
            # Non convertire in ##ESEMPIO##, usa il formato user:/assistant: originale
            examples_text = ""
            for i, ex in enumerate(examples, 1):
                if 'raw_content' in ex:
                    # 🎯 USA CONTENUTO ORIGINALE COMPLETO dal database
                    examples_text += f"{ex['raw_content']}\n\n"
                else:
                    # Fallback per esempi hardcoded nel vecchio formato
                    examples_text += f"""##ESEMPIO##
{ex["text"]}
{ex["label"]}

"""
            
            # Variabili dinamiche per il template
            variables = {
                'examples_text': examples_text,
                'context_section': f"\n\nCONTESTO AGGIUNTIVO:\n{context}" if context else "",
                'processed_text': processed_text,
            }
            
            # Carica prompt dal database con validazione STRICT - passa l'oggetto tenant
            user_prompt = self.prompt_manager.get_prompt_strict(
                self.tenant,  # Passa l'oggetto tenant completo
                engine="LLM",
                prompt_type="TEMPLATE",
                prompt_name="intelligent_classifier_user_template",
                variables=variables
            )
            
            self.logger.info(f"✅ Prompt USER caricato da database per tenant {self.tenant_id}")
            return user_prompt
            
        except Exception as e:
            error_msg = f"❌ ERRORE CARICAMENTO PROMPT: Impossibile caricare prompt USER per tenant {self.tenant_id}. Dettaglio: {e}"
            print(error_msg)
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def _get_dynamic_examples(self, conversation_text: str, max_examples: int = 4) -> List[Dict]:
        """
        Seleziona esempi dinamici più rilevanti per il testo di input
        Prima prova a caricare esempi dal database TAG, poi usa esempi curati come fallback
        
        EVOLUZIONE ARCHITETTURALE:
        - v1.0: Solo esempi hardcoded
        - v2.0: Esempi hardcoded + esempi reali da session_classifications (DEPRECATO)
        - v3.0 (CORRENTE): Esempi curati da TAG.esempi + esempi hardcoded fallback
        
        Scopo della funzione:
        - Selezionare esempi più rilevanti per guidare la classificazione LLM
        - Prioritizzare esempi dal database TAG per il tenant specifico
        
        Parametri di input e output:
        - conversation_text: str - Testo da classificare per selezione contestuale
        - max_examples: int - Numero massimo di esempi da restituire
        
        Valori di ritorno:
        - List[Dict]: Lista di esempi selezionati con chiavi text, label, motivation
        
        Tracciamento aggiornamenti:
        - 2025-09-06: Aggiunto tracing completo ENTER/EXIT
        
        Args:
            conversation_text: Testo da classificare
            max_examples: Numero massimo di esempi da includere
            
        Returns:
            Lista di esempi selezionati ottimizzati
        """
        
        # 🔍 TRACING ENTER
        trace_all("_get_dynamic_examples", "ENTER", 
                 called_from="classify_with_motivation",
                 conversation_length=len(conversation_text) if conversation_text else 0,
                 max_examples=max_examples,
                 has_embedder=self.embedder is not None,
                 enable_embeddings=self.enable_embeddings)
        
        if not conversation_text:
            # 🔍 TRACING EXIT - Empty Text Fallback
            trace_all("_get_dynamic_examples", "EXIT", 
                     called_from="classify_with_motivation",
                     source="CURATED_FALLBACK_EMPTY_TEXT",
                     examples_count=len(self.curated_examples[:max_examples]),
                     exit_reason="EMPTY_CONVERSATION_TEXT")
            
            return self.curated_examples[:max_examples]
        
        # STRATEGIA DINAMICA: Prima prova esempi database, poi fallback curati
        
        # FASE 1: Prova esempi dal database TAG per il tenant corrente
        database_examples = self._get_examples_from_database_tag(max_examples=max_examples)
        
        if database_examples:
            self.logger.debug(f"✅ Caricati {len(database_examples)} esempi dal database TAG per tenant {self.tenant_id}")
            
            # 🔍 TRACING EXIT - Database Success
            trace_all("_get_dynamic_examples", "EXIT", 
                     called_from="classify_with_motivation",
                     source="DATABASE_TAG",
                     examples_count=len(database_examples),
                     tenant_id=str(self.tenant_id),
                     exit_reason="DATABASE_SUCCESS")
            
            return database_examples
        
        # FASE 2: Fallback - Esempi curati hardcoded con selezione intelligente
        if self.enable_embeddings and self.embedder is not None:
            selected_examples = self._get_semantic_dynamic_examples(conversation_text, max_examples=max_examples)
            selection_method = "SEMANTIC_EMBEDDINGS"
        else:
            selected_examples = self._get_word_overlap_examples(conversation_text, max_examples=max_examples)
            selection_method = "WORD_OVERLAP"
        
        self.logger.debug(f"⚠️ Usando esempi curati hardcoded - {len(selected_examples)} esempi per tenant {self.tenant_id}")
        
        # 🔍 TRACING EXIT - Hardcoded Fallback
        trace_all("_get_dynamic_examples", "EXIT", 
                 called_from="classify_with_motivation",
                 source="HARDCODED_CURATED",
                 selection_method=selection_method,
                 examples_count=len(selected_examples),
                 tenant_id=str(self.tenant_id),
                 exit_reason="HARDCODED_FALLBACK")
        
        return selected_examples

    def _get_examples_from_database_tag(self, max_examples: int = 4) -> List[Dict]:
        """
        Carica esempi per il tenant corrente dal database TAG.esempi
        
        Scopo della funzione:
        - Caricare esempi curati dal database TAG per il tenant specifico
        - Convertire esempi dal formato database al formato prompt
        
        Parametri di input e output:
        - max_examples: int - Numero massimo di esempi da caricare
        
        Valori di ritorno:
        - List[Dict]: Lista di esempi con raw_content e esempio_name
        
        Tracciamento aggiornamenti:
        - 2025-09-06: Aggiunto tracing completo ENTER/EXIT/ERROR
        
        Args:
            max_examples: Numero massimo di esempi da caricare
            
        Returns:
            Lista di esempi dal database o lista vuota se non trovati
        """
        
        # 🔍 TRACING ENTER
        trace_all("_get_examples_from_database_tag", "ENTER", 
                 called_from="_get_dynamic_examples",
                 max_examples=max_examples,
                 has_prompt_manager=hasattr(self, 'prompt_manager') and self.prompt_manager is not None,
                 has_tenant=bool(self.tenant),
                 tenant_id=str(self.tenant_id) if hasattr(self, 'tenant_id') else 'unknown')
        
        try:
            # Usa il prompt_manager per caricare esempi dal database
            if not hasattr(self, 'prompt_manager') or self.prompt_manager is None:
                self.logger.debug("⚠️ PromptManager non disponibile per caricamento esempi")
                
                # 🔍 TRACING EXIT - No PromptManager
                trace_all("_get_examples_from_database_tag", "EXIT", 
                         called_from="_get_dynamic_examples",
                         examples_count=0,
                         exit_reason="NO_PROMPT_MANAGER")
                
                return []
            
            # 🎯 USA TENANT CENTRALIZZATO - passa l'oggetto tenant completo
            if not self.tenant:
                self.logger.error("❌ Oggetto tenant non disponibile per caricamento esempi")
                
                # 🔍 TRACING EXIT - No Tenant
                trace_all("_get_examples_from_database_tag", "EXIT", 
                         called_from="_get_dynamic_examples",
                         examples_count=0,
                         exit_reason="NO_TENANT_OBJECT")
                
                return []
            
            # Carica esempi dal database TAG usando il prompt_manager
            esempi_list = self.prompt_manager.get_examples_list(
                self.tenant,  # Passa l'oggetto tenant completo
                engine='LLM',
                esempio_type='CONVERSATION'
            )
            
            if not esempi_list:
                self.logger.debug(f"⚠️ Nessun esempio trovato nel database TAG per tenant {self.tenant.tenant_id}")
                
                # 🔍 TRACING EXIT - No Examples Found
                trace_all("_get_examples_from_database_tag", "EXIT", 
                         called_from="_get_dynamic_examples",
                         examples_count=0,
                         tenant_id=str(self.tenant.tenant_id),
                         exit_reason="NO_EXAMPLES_FOUND")
                
                return []
            
            # Converte gli esempi dal formato database al formato richiesto per il prompt
            converted_examples = []
            for esempio in esempi_list[:max_examples]:
                try:
                    # 🔧 CORREZIONE CRITICA: USA IL FORMATO ORIGINALE COMPLETO!
                    # Carica il contenuto completo dell'esempio così com'è dal database
                    full_content = self._get_example_full_content(self.tenant.tenant_id, esempio['esempio_name'])
                    if full_content:
                        # 🎯 NON CONVERTIRE! Usa il formato originale dal database:
                        # user: "testo"
                        # assistant: {"predicted_label": "...", "confidence": ..., "motivation": "..."}
                        converted_examples.append({
                            "raw_content": full_content,  # 🔑 CONTENUTO ORIGINALE COMPLETO
                            "esempio_name": esempio['esempio_name']
                        })
                except Exception as e:
                    self.logger.debug(f"⚠️ Errore caricamento esempio {esempio['esempio_name']}: {e}")
                    continue
            
            if converted_examples:
                self.logger.info(f"✅ Caricati {len(converted_examples)} esempi dal database TAG per tenant {self.tenant.tenant_id}")
            
            # 🔍 TRACING EXIT - Success
            trace_all("_get_examples_from_database_tag", "EXIT", 
                     called_from="_get_dynamic_examples",
                     examples_count=len(converted_examples),
                     tenant_id=str(self.tenant.tenant_id),
                     esempi_list_count=len(esempi_list) if esempi_list else 0,
                     exit_reason="SUCCESS")
            
            return converted_examples
            
        except Exception as e:
            self.logger.debug(f"⚠️ Errore caricamento esempi dal database TAG: {e}")
            
            # 🔍 TRACING ERROR
            trace_all("_get_examples_from_database_tag", "ERROR", 
                     called_from="_get_dynamic_examples",
                     error_type=type(e).__name__,
                     error_message=str(e))
            
            # 🔍 TRACING EXIT - Error
            trace_all("_get_examples_from_database_tag", "EXIT", 
                     called_from="_get_dynamic_examples",
                     examples_count=0,
                     exit_reason="ERROR")
            
            return []

    @staticmethod
    def _build_gpt5_json_schema(domain_labels: List[str]) -> Dict[str, Any]:
        """
        Costruisce il blocco `text.format=json_schema` per Responses API (GPT-5)
        in modalità SOLO JSON.

        Scopo:
            Forzare un output JSON con campi previsti (predicted_label,
            confidence, motivation) e `strict: true`.

        Parametri:
            domain_labels: Lista etichette valide; se vuota usa ['altro']

        Ritorno:
            Dict compatibile con campo 'text' della Responses API.

        Ultima modifica: 2025-10-25
        """
        try:
            labels = list(domain_labels) if domain_labels else ['altro']
        except Exception:
            labels = ['altro']

        schema = {
            "format": {
                "type": "json_schema",
                "name": "classification_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "predicted_label": {
                            "type": "string",
                            "enum": labels
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "motivation": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "predicted_label",
                        "confidence",
                        "motivation"
                    ],
                    "additionalProperties": False
                }
            }
        }
        return schema
    
    def _get_example_full_content(self, tenant_id: str, esempio_name: str) -> str:
        """
        Recupera il contenuto completo di un esempio dal database
        
        Scopo della funzione:
        - Recuperare il contenuto grezzo completo di un esempio dal database
        - Gestire connessione database e query SQL sicura
        
        Parametri di input e output:
        - tenant_id: str - ID del tenant per query sicura
        - esempio_name: str - Nome dell'esempio da recuperare
        
        Valori di ritorno:
        - str: Contenuto completo dell'esempio o stringa vuota se non trovato
        
        Tracciamento aggiornamenti:
        - 2025-09-06: Aggiunto tracing completo ENTER/EXIT/ERROR
        
        Args:
            tenant_id: ID del tenant
            esempio_name: Nome dell'esempio
            
        Returns:
            Contenuto dell'esempio o stringa vuota
        """
        
        # 🔍 TRACING ENTER
        trace_all("_get_example_full_content", "ENTER", 
                 called_from="_get_examples_from_database_tag",
                 tenant_id=tenant_id,
                 esempio_name=esempio_name,
                 has_prompt_manager=bool(self.prompt_manager))
        
        try:
            if not self.prompt_manager.connect():
                # 🔍 TRACING EXIT - Connection Failed
                trace_all("_get_example_full_content", "EXIT", 
                         called_from="_get_examples_from_database_tag",
                         content_length=0,
                         exit_reason="CONNECTION_FAILED")
                
                return ""
            
            cursor = self.prompt_manager.connection.cursor()
            cursor.execute("""
                SELECT esempio_content 
                FROM esempi 
                WHERE tenant_id = %s AND esempio_name = %s AND is_active = TRUE
                LIMIT 1
            """, (tenant_id, esempio_name))
            
            result = cursor.fetchone()
            cursor.close()
            
            content = result[0] if result else ""
            
            # 🔍 TRACING EXIT - Success
            trace_all("_get_example_full_content", "EXIT", 
                     called_from="_get_examples_from_database_tag",
                     content_length=len(content) if content else 0,
                     found=bool(result),
                     exit_reason="SUCCESS")
            
            return content
            
        except Exception as e:
            # 🔍 TRACING ERROR
            trace_all("_get_example_full_content", "ERROR", 
                     called_from="_get_examples_from_database_tag",
                     error_type=type(e).__name__,
                     error_message=str(e))
            
            # 🔍 TRACING EXIT - Error
            trace_all("_get_example_full_content", "EXIT", 
                     called_from="_get_examples_from_database_tag",
                     content_length=0,
                     exit_reason="ERROR")
            
            return ""
    
    def _extract_conversation_from_example(self, example_content: str) -> str:
        """
        Estrae il testo della conversazione dall'esempio con formato user:/assistant:
        
        Args:
            example_content: Contenuto esempio nel formato 'user: "..." \\n assistant: {...}'
            
        Returns:
            Testo della conversazione dell'utente
        """
        try:
            # Nuovo formato: user: "..." \n assistant: {...}
            if 'user: "' in example_content:
                import re
                # Estrae il testo tra le virgolette dopo user:
                user_match = re.search(r'user:\s*"(.+?)"', example_content, re.DOTALL)
                if user_match:
                    return user_match.group(1).strip()
            
            # Pattern legacy UTENTE: ... ASSISTENTE: ...
            if 'UTENTE:' in example_content:
                import re
                utente_match = re.search(r'UTENTE:\s*(.+?)(?:\s+ASSISTENTE:|$)', example_content, re.DOTALL)
                if utente_match:
                    return utente_match.group(1).strip()
            
            # Fallback: restituisce l'esempio così com'è
            return example_content[:200]  # Limita lunghezza
        except Exception:
            return example_content[:100]
    
    def _extract_label_from_example(self, example_content: str) -> str:
        """
        Estrae l'etichetta dal JSON dell'assistente nell'esempio
        
        Args:
            example_content: Contenuto esempio nel formato 'user: "..." \\n assistant: {...}'
            
        Returns:
            Etichetta estratta dal JSON dell'assistente
        """
        try:
            import json
            import re
            
            # Estrae il JSON dell'assistente
            assistant_match = re.search(r'assistant:\s*(\{.+\})', example_content, re.DOTALL)
            if assistant_match:
                json_str = assistant_match.group(1).strip()
                # Converte single quotes in double quotes per JSON valido
                json_str = json_str.replace("'", '"')
                
                try:
                    assistant_data = json.loads(json_str)
                    if 'predicted_label' in assistant_data:
                        return assistant_data['predicted_label']
                except json.JSONDecodeError:
                    # Fallback: estrazione con regex
                    label_match = re.search(r'"predicted_label":\s*"([^"]+)"', json_str)
                    if label_match:
                        return label_match.group(1)
            
            # Fallback usando il nome dell'esempio
            return "altro"
        except Exception:
            return "altro"

    def _infer_label_from_example_name(self, esempio_name: str) -> str:
        """Deduce una label dall'nome dell'esempio"""
        # Mapping nomi esempio -> etichette standard Wopta
        label_mapping = {
            'durata_polizza': 'info_durata_polizza',
            'beneficiario': 'info_beneficiario',
            'disdetta': 'disdetta_polizza',
            'condizioni': 'info_condizioni_polizza',
            'costi': 'info_costi_premio',
            'sede': 'info_sede_aziendale',
            'contatti': 'info_contatti'
        }
        
        esempio_lower = esempio_name.lower()
        for key, label in label_mapping.items():
            if key in esempio_lower:
                return label
        
        # Fallback: genera etichetta dal nome
        return f"info_{esempio_name.lower().replace(' ', '_')}"
    
    # CODICE COMMENTATO - FUNZIONE NON PIU' UTILIZZATA
    # Sostituita da _get_examples_from_database_tag per sistema TAG-based
    # Mantenuta per riferimento storico - Era usata nelle versioni precedenti
    # per implementare apprendimento dinamico con esempi reali da session_classifications
    
    # def _get_real_examples_from_database(self, conversation_text: str, max_examples: int = 2) -> List[Dict]:
    #     """
    #     FUNZIONE DEPRECATA - NON PIU' UTILIZZATA
    #     
    #     Recuperava esempi reali di classificazioni dal database per enrichment
    #     Sostituita dal sistema TAG.esempi per esempi curati
    #     
    #     Args:
    #         conversation_text: Testo di riferimento
    #         max_examples: Numero massimo di esempi reali
    #         
    #     Returns:
    #         Lista di esempi reali formattati
    #     """
    #     try:
    #         if not self.enable_embeddings or self.embedder is None:
    #             return []
    #         
    #         from MySql.connettore import ConnettoreDB
    #         
    #         connector = ConnettoreDB()
    #         if not connector.connetti():
    #             return []
    #         
    #         # Query per esempi recenti con buona confidence
    #         query = """
    #         SELECT 
    #             s.conversation_text,
    #             s.predicted_label,
    #             s.confidence,
    #             s.human_feedback
    #         FROM session_classifications s
    #         WHERE s.confidence >= 0.7
    #             AND s.predicted_label != 'altro'
    #             AND s.conversation_text IS NOT NULL
    #             AND LENGTH(s.conversation_text) BETWEEN 10 AND 200
    #             AND s.prediction_timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
    #         ORDER BY s.confidence DESC, s.prediction_timestamp DESC
    #         LIMIT 20
    #         """
    #         
    #         result = connector.esegui_query(query)
    #         connector.disconnetti()
    #         
    #         if not result:
    #             return []
    #         
    #         # Converte in formato compatibile e filtra per diversità
    #         real_examples = []
    #         used_labels = set()
    #         
    #         for row in result:
    #             text, label, confidence, feedback = row
    #             
    #             # Solo esempi con feedback positivo o neutrale
    #             if feedback and feedback.lower() in ['negative', 'wrong']:
    #                 continue
    #             
    #             # Diversifica le etichette
    #             if label not in used_labels:
    #                 real_examples.append({
    #                     'text': text[:150],  # Tronca per leggibilità
    #                     'label': label,
    #                     'motivation': f"Esempio reale (conf: {confidence:.2f})",
    #                     'source': 'database',
    #                     'confidence': confidence
    #                 })
    #                 used_labels.add(label)
    #             
    #             if len(real_examples) >= max_examples:
    #                 break
    #         
    #         self.logger.debug(f"Recuperati {len(real_examples)} esempi reali dal database")
    #         return real_examples
    #         
    #     except Exception as e:
    #         self.logger.warning(f"Errore recupero esempi reali: {e}")
    #         return []
    
    # CODICE COMMENTATO - FUNZIONE NON PIU' UTILIZZATA  
    # Era usata per combinare esempi curati + esempi reali da session_classifications
    # Sistema sostituito da _get_examples_from_database_tag (solo esempi curati TAG)
    # Mantenuta per riferimento storico
    
    # def _merge_examples_intelligently(self, curated: List[Dict], real: List[Dict], max_total: int) -> List[Dict]:
    #     """
    #     FUNZIONE DEPRECATA - NON PIU' UTILIZZATA
    #     
    #     Combinava esempi curati e reali in modo intelligente
    #     Era parte del sistema di apprendimento dinamico legacy
    #     
    #     Args:
    #         curated: Esempi curati (sempre affidabili)
    #         real: Esempi reali dal database
    #         max_total: Numero massimo totale
    #         
    #     Returns:
    #         Lista combinata ottimizzata
    #     """
    #     # Priorità agli esempi curati (più affidabili)
    #     combined = []
    #     used_labels = set()
    #     
    #     # FASE 1: Prendi i migliori esempi curati
    #     for example in curated:
    #         if len(combined) >= max_total:
    #             break
    #         combined.append(example)
    #         used_labels.add(example['label'])
    #     
    #     # FASE 2: Aggiungi esempi reali per diversità (se c'è spazio)
    #     for example in real:
    #         if len(combined) >= max_total:
    #             break
    #         
    #         # Solo se etichetta diversa o se abbiamo pochi esempi
    #         if example['label'] not in used_labels or len(combined) < 2:
    #             combined.append(example)
    #             used_labels.add(example['label'])
    #     
    #     self.logger.debug(f"Esempi combinati: {len(combined)} totali ({len(curated)} curati + {len([e for e in combined if e.get('source') == 'database'])} reali)")
    #     return combined
    
    def _get_semantic_dynamic_examples(self, conversation_text: str, max_examples: int = 4) -> List[Dict]:
        """
        Seleziona esempi usando similarità semantica (embedding)
        
        Args:
            conversation_text: Testo da classificare
            max_examples: Numero massimo di esempi da includere
            
        Returns:
            Lista di esempi selezionati semanticamente
        """
        try:
            # Genera embedding del testo input
            input_embedding = self.embedder.encode_single(conversation_text)
            
            # Calcola/recupera embedding degli esempi curati
            example_embeddings = self._get_or_compute_example_embeddings()
            
            # Calcola similarità con tutti gli esempi
            similarities = []
            for i, example_embedding in enumerate(example_embeddings):
                similarity = self._cosine_similarity(input_embedding, example_embedding)
                similarities.append((similarity, self.curated_examples[i]))
            
            # Ordina per similarità decrescente
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # ALGORITMO MIGLIORATO: Selezione bilanciata tra rilevanza e diversità
            selected = []
            used_labels = set()
            
            # FASE 1: Prendi i migliori esempi semanticamente rilevanti (threshold alto)
            high_similarity_threshold = 0.7
            for similarity, example in similarities:
                if len(selected) >= max_examples:
                    break
                    
                # Priorità alta: esempi molto simili con etichette diverse
                if (similarity >= high_similarity_threshold and 
                    example['label'] not in used_labels):
                    selected.append(example)
                    used_labels.add(example['label'])
                    self.logger.debug(f"Esempio alta similarità: '{example['text'][:30]}...' (sim: {similarity:.3f})")
            
            # FASE 2: Completa con esempi diversificati ma comunque rilevanti
            medium_similarity_threshold = 0.4
            for similarity, example in similarities:
                if len(selected) >= max_examples:
                    break
                    
                # Priorità media: diversità delle etichette mantenendo rilevanza minima
                if (similarity >= medium_similarity_threshold and 
                    example['label'] not in used_labels and 
                    example not in selected):
                    selected.append(example)
                    used_labels.add(example['label'])
                    self.logger.debug(f"Esempio diversificazione: '{example['text'][:30]}...' (sim: {similarity:.3f})")
            
            # FASE 3: Se ancora mancano esempi, prendi i migliori rimanenti
            for similarity, example in similarities:
                if len(selected) >= max_examples:
                    break
                if example not in selected:
                    selected.append(example)
                    self.logger.debug(f"Esempio riempimento: '{example['text'][:30]}...' (sim: {similarity:.3f})")
            
            return selected[:max_examples]
            
        except Exception as e:
            self.logger.warning(f"Errore selezione esempi semantici: {e}")
            # Fallback alla logica word overlap
            return self._get_word_overlap_examples(conversation_text, max_examples)
    
    def _get_or_compute_example_embeddings(self) -> List[np.ndarray]:
        """Recupera o calcola embedding degli esempi curati (con cache)"""
        if self._example_embeddings_cache is None:
            try:
                self.logger.debug("Calcolo embedding esempi curati...")
                self._example_embeddings_cache = []
                
                for example in self.curated_examples:
                    embedding = self.embedder.encode_single(example['text'])
                    self._example_embeddings_cache.append(embedding)
                    
                self.logger.debug(f"Calcolati {len(self._example_embeddings_cache)} embedding esempi")
                
            except Exception as e:
                self.logger.error(f"Errore calcolo embedding esempi: {e}")
                return []
        
        return self._example_embeddings_cache
    
    def _get_word_overlap_examples(self, conversation_text: str, max_examples: int = 4) -> List[Dict]:
        """
        Logica originale di selezione esempi basata su overlap di parole
        """
        # Parole chiave per matching euristico semplice
        text_lower = conversation_text.lower()
        scored_examples = []
        
        for example in self.curated_examples:
            score = 0
            example_words = example['text'].lower().split()
            
            # Score basato su overlap di parole (euristica semplice)
            for word in example_words:
                if len(word) > 3 and word in text_lower:
                    score += 1
            
            # Bonus per lunghezza simile
            len_diff = abs(len(conversation_text) - len(example['text']))
            if len_diff < 50:
                score += 0.5
            
            scored_examples.append((score, example))
        
        # Ordina per score e prendi i migliori
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        # Assicura diversità nelle etichette
        selected = []
        used_labels = set()
        
        for score, example in scored_examples:
            if len(selected) >= max_examples:
                break
            
            # Preferisci esempi con etichette diverse
            if example['label'] not in used_labels or len(selected) < 2:
                selected.append(example)
                used_labels.add(example['label'])
        
        # Se non abbiamo abbastanza esempi, aggiungi dai rimanenti
        if len(selected) < max_examples:
            for score, example in scored_examples:
                if len(selected) >= max_examples:
                    break
                if example not in selected:
                    selected.append(example)
        
        return selected[:max_examples]
    
    def _summarize_if_long(self, text: str, max_length: int = 300) -> str:
        """
        Riassume automaticamente testi troppo lunghi per il prompt
        
        Args:
            text: Testo da potenzialmente riassumere
            max_length: Lunghezza massima prima del riassunto
            
        Returns:
            Testo originale o riassunto
        """
        if len(text) <= max_length:
            return text
        
        # Riassunto semplice: prime e ultime frasi + indicatore
        sentences = text.split('.')
        if len(sentences) <= 2:
            return text[:max_length] + "..."
        
        first_part = sentences[0] + '.'
        last_part = sentences[-1] if sentences[-1].strip() else sentences[-2]
        
        summary = f"{first_part} [...] {last_part}"
        
        # Se ancora troppo lungo, tronca
        if len(summary) > max_length:
            return text[:max_length-3] + "..."
        
        return summary
    
    def _build_user_message(self, conversation_text: str, context: Optional[str] = None) -> str:
        """
        Costruisce il messaggio utente utilizzando _get_user_prompt per caricamento database
        
        Args:
            conversation_text: Testo da classificare
            context: Contesto opzionale
            
        Returns:
            User message strutturato e context-aware
            
        Raises:
            Exception: Se i prompt obbligatori non sono configurati per il tenant
        """
        # Usa il metodo dedicato per caricare il prompt USER dal database
        try:
            user_prompt = self._get_user_prompt(conversation_text, context)
            
            # 🔍 STAMPA DETTAGLIATA PROMPT USER (solo se debug_prompt=True)
            if self.debug_prompt:
                print("\n" + "="*80)
                print("👤 DEBUG PROMPT USER - DATABASE")
                print("="*80)
                print(f"📋 Prompt Name: LLM/TEMPLATE/intelligent_classifier_user_template")
                print(f"🏢 Tenant ID: {self.tenant_id}")
                print(f"📏 Text Length: {len(conversation_text)} chars")
                print("-"*80)
                print("📄 USER PROMPT CONTENT (dopo sostituzione placeholder):")
                print("-"*80)
                print(user_prompt)
                print("="*80)
            else:
                print(f"👤 User prompt generato per conversazione {len(conversation_text)} chars (debug_prompt=False)")
            
            return user_prompt
            
        except Exception as e:
            # Errore già gestito in _get_user_prompt, rilancialo
            raise e
    
    def _build_classification_prompt(self, 
                                   conversation_text: str,
                                   context: Optional[str] = None) -> str:
        """
        Costruisce il prompt ottimizzato con struttura ChatML-like intelligente
        e tokenizzazione preventiva per gestire conversazioni lunghe
        
        Args:
            conversation_text: Testo della conversazione da classificare
            context: Contesto aggiuntivo opzionale
            
        Returns:
            Prompt strutturato e ottimizzato con tokenizzazione preventiva
            
        Raises:
            Exception: Se i prompt obbligatori non sono configurati per il tenant
        """
        trace_all(
            'ENTER', 
            '_build_classification_prompt', 
            {
                'conversation_length': len(conversation_text),
                'has_context': context is not None,
                'tenant_id': self.tenant_id,
                'model_name': self.model_name,
                'tokenizer_available': self.tokenizer is not None
            }
        )
        
        # Analizza il testo per determinare context hints
        conversation_context = self._analyze_conversation_context(conversation_text)
        
        # Carica system message - SENZA conversazione per ora
        system_msg = self._build_system_message(conversation_context)
        
        # ========================================================================
        # 🔥 TOKENIZZAZIONE PREVENTIVA COME RICHIESTO DALL'UTENTE
        # ========================================================================
        
        processed_conversation = conversation_text
        tokenization_stats = None
        
        if self.tokenizer:
            print(f"\n🔍 TOKENIZZAZIONE PREVENTIVA LLM CLASSIFICATION")
            print(f"=" * 60)
            
            # Costruisce prompt template senza conversazione per calcolare token base
            prompt_template = f"""<|system|>
{system_msg}

<|user|>
{{conversation_placeholder}}

<|assistant|>"""
            
            # Usa TokenizationManager per processare con LLM context
            try:
                processed_conversation, tokenization_stats = self.tokenizer.process_conversation_for_llm(
                    conversation_text, prompt_template
                )
                
                print(f"✅ Tokenizzazione LLM completata:")
                print(f"   📊 Token prompt base: {tokenization_stats['prompt_tokens']}")
                print(f"   📊 Token conversazione originale: {tokenization_stats['conversation_tokens_original']}")
                print(f"   📊 Token conversazione finale: {tokenization_stats['conversation_tokens_final']}")
                print(f"   📊 Token totali: {tokenization_stats['total_tokens_final']}")
                print(f"   📊 Limite configurato: {tokenization_stats['max_tokens_limit']}")
                if tokenization_stats['truncated']:
                    print(f"   ✂️  Conversazione TRONCATA per rispettare limite token")
                    print(f"   ⚠️  Caratteri rimossi: {len(conversation_text) - len(processed_conversation)}")
                else:
                    print(f"   ✅ Conversazione entro i limiti, nessun troncamento")
                print(f"=" * 60)
                
            except Exception as e:
                print(f"⚠️  Errore durante tokenizzazione LLM: {e}")
                print(f"🔄 Fallback a conversazione originale")
                processed_conversation = conversation_text
        else:
            print(f"⚠️  TokenizationManager non disponibile per tokenizzazione LLM")
            print(f"📏 Usando conversazione originale ({len(conversation_text)} caratteri)")
        
        # Costruisce user message con conversazione processata (potenzialmente troncata)
        user_msg = self._build_user_message(processed_conversation, context)
        
        # Struttura ChatML-like per guida migliore del modello
        prompt = f"""<|system|>
{system_msg}

<|user|>
{user_msg}

<|assistant|>"""
        
        # 🚀 DEBUG DETTAGLIATO COME RICHIESTO DALL'UTENTE (solo se debug_prompt=True)
        if self.debug_prompt:
            print("\n" + "🔥"*80)
            print("🚀 PROMPT COMPLETO FINALE INVIATO ALL'LLM")
            print("🔥"*80)
            print(f"🤖 Model: {self.model_name}")
            print(f"🌐 Ollama URL: {self.ollama_url}")
            print(f"🏢 Tenant: {self.tenant_id}")
            print(f"📏 Total Prompt Length: {len(prompt)} characters")
            if tokenization_stats:
                print(f"🔢 Token Analysis:")
                print(f"   📊 Token prompt base: {tokenization_stats['prompt_tokens']}")
                print(f"   📊 Token conversazione: {tokenization_stats['conversation_tokens_final']}")
                print(f"   📊 Token totali stimati: {tokenization_stats['total_tokens_final']}")
                print(f"   📊 Limite configurato: {tokenization_stats['max_tokens_limit']}")
                if tokenization_stats['truncated']:
                    print(f"   ✂️  STATUS: Conversazione TRONCATA")
                else:
                    print(f"   ✅ STATUS: Conversazione COMPLETA")
            print("-"*80)
            print("📄 FULL PROMPT CONTENT:")
            print("-"*80)
            print(prompt)
            print("🔥"*80)
            print()
        else:
            print(f"🚀 Prompt finale generato per LLM {self.model_name} - {len(prompt)} chars (debug_prompt=False)")
        
        trace_all(
            'EXIT', 
            '_build_classification_prompt', 
            {
                'prompt_length': len(prompt),
                'tokenization_applied': tokenization_stats is not None,
                'conversation_truncated': tokenization_stats['truncated'] if tokenization_stats else False,
                'total_tokens': tokenization_stats['total_tokens_final'] if tokenization_stats else None,
                'conversation_context': conversation_context,
                'debug_enabled': self.debug_prompt
            }
        )
        
        return prompt
    
    def _analyze_conversation_context(self, conversation_text: str) -> Optional[str]:
        """
        Analizza il testo per determinare context hints per il prompt
        
        Args:
            conversation_text: Testo da analizzare
            
        Returns:
            Context hint specifico o None
        """
        text_lower = conversation_text.lower()
        
        # Pattern di urgenza/emergenza
        if any(word in text_lower for word in ['urgente', 'emergenza', 'subito', 'aiuto']):
            return "Richiesta urgente - valuta priorità massima"
        
        # Pattern di problemi tecnici/accesso (PRIORITÀ ALTA)
        access_problems = ['non riesco', 'non funziona', 'errore', 'sbagliato', 'non entra', 'non accede', 'login', 'password']
        portal_terms = ['portale', 'sito', 'accesso', 'login', 'account']
        
        if (any(problem in text_lower for problem in access_problems) and 
            any(portal in text_lower for portal in portal_terms)):
            return "Problema tecnico di accesso - priorità problema_accesso_portale su altre categorie"
        
        # Pattern di confusione/difficoltà  
        if any(word in text_lower for word in ['non capisco', 'confuso', 'sbagliato', 'errore']):
            return "Utente confuso - semplifica classificazione verso problemi tecnici"
        
        # Pattern di cortesia/ringraziamento (spesso info generiche)
        if any(word in text_lower for word in ['grazie', 'cortesia', 'informazione', 'sapere']):
            return "Richiesta informativa - preferisci etichette info_*"
        
        # Pattern operativi (azioni concrete) - ma non se c'è problema tecnico
        if any(word in text_lower for word in ['voglio', 'devo', 'prenotare', 'ritirare', 'cambiare']):
            return "Richiesta operativa - preferisci etichette di azione"
        
        return None
    
    def _validate_and_fix_json(self, response_text: str) -> Tuple[bool, str]:
        """
        Valida e cerca di correggere automaticamente risposte JSON malformate
        
        Args:
            response_text: Risposta raw del modello
            
        Returns:
            Tuple (is_valid, corrected_json_string)
        """
        try:
            # Test parsing diretto
            json.loads(response_text.strip())
            return True, response_text.strip()
        except json.JSONDecodeError:
            pass
        
        # Tentativi di correzione automatica
        cleaned = response_text.strip()
        
        # Rimuovi testo prima/dopo JSON
        json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if json_match:
            json_candidate = json_match.group(0)
            try:
                json.loads(json_candidate)
                return True, json_candidate
            except json.JSONDecodeError:
                pass
        
        # Cerca pattern mancanti comuni
        fixes = [
            (r'(\w+):', r'"\1":'),  # Aggiungi quote alle chiavi
            (r':\s*([^",\{\}\[\]]+)(\s*[,\}])', r': "\1"\2'),  # Quote ai valori
            (r',\s*}', '}'),  # Rimuovi virgole finali
        ]
        
        for pattern, replacement in fixes:
            fixed = re.sub(pattern, replacement, cleaned)
            try:
                json.loads(fixed)
                return True, fixed
            except json.JSONDecodeError:
                continue
        
        return False, cleaned
    
    def _call_ollama_api_with_retry(self, prompt: str, max_retries: int = 2) -> str:
        """
        Chiama API Ollama con retry automatico per JSON malformati
        
        Args:
            prompt: Prompt per il modello
            max_retries: Numero massimo di retry
            
        Returns:
            Risposta del modello
        """
        for attempt in range(max_retries + 1):
            try:
                response = self._call_ollama_api(prompt)
                
                # Verifica se la risposta è JSON valido
                is_valid, _ = self._validate_and_fix_json(response)
                
                if is_valid or attempt == max_retries:
                    return response
                
                # Se non valido, aggiungi istruzioni di correzione per retry
                if attempt < max_retries:
                    self.logger.debug(f"JSON malformato al tentativo {attempt + 1}, retry...")
                    retry_prompt = prompt + f"\n\nATTENZIONE: La risposta precedente non era JSON valido. Genera SOLO JSON corretto:\n{{"
                    response = self._call_ollama_api(retry_prompt)
                    
                    # Test finale del retry
                    is_valid, _ = self._validate_and_fix_json(response)
                    if is_valid:
                        return response
                
            except Exception as e:
                if attempt == max_retries:
                    raise e
                self.logger.debug(f"Errore al tentativo {attempt + 1}: {e}")
                continue
        
        return response
    
    def _parse_llm_response(self, response_text: str, conversation_text: str = "") -> Tuple[str, float, str, str]:
        """
        # 🔍 TRACING ENTER
        trace_all("_parse_llm_response", "ENTER", 
                 called_from="classification_pipeline")
        
        Parsa la risposta JSON del modello LLM con auto-validazione e correzione
        
        Args:
            response_text: Risposta raw del modello
            
        Returns:
            Tuple (predicted_label, confidence, motivation)
        """
        trace_all(
            'ENTER', 
            '_parse_llm_response', 
            {
                'response_length': len(response_text),
                'conversation_length': len(conversation_text),
                'response_preview': response_text[:100]
            }
        )
        
        try:
            self.logger.debug(f"Parsing risposta LLM: {response_text[:200]}...")
            
            # Auto-validazione e correzione
            is_valid, corrected_json = self._validate_and_fix_json(response_text)
            
            if not is_valid:
                self.logger.warning("JSON non valido anche dopo correzione automatica")
                fallback_result = self._fallback_parse_response(response_text)
                trace_all(
                    'EXIT', 
                    '_parse_llm_response', 
                    {
                        'predicted_label': fallback_result[0],
                        'confidence': fallback_result[1],
                        'motivation_length': len(fallback_result[2]),
                        'method': 'FALLBACK_PARSE',
                        'json_valid': False
                    }
                )
                return fallback_result
            
            # Parsing del JSON corretto
            result = json.loads(corrected_json)
            
            # Estrazione e validazione campi
            raw_predicted_label = result.get('predicted_label', '').strip()
            # Mantiene forma MAIUSCOLA: clean_label_text già normalizza in uppercase
            predicted_label = clean_label_text(raw_predicted_label)  # 🧹 PULIZIA ETICHETTA
            confidence = float(result.get('confidence', 0.0))
            motivation = result.get('motivation', '').strip()
            
            # 🐛 DEBUG: Log pulizia etichetta se necessaria
            if raw_predicted_label != predicted_label:
                self.logger.info(f"🧹 Etichetta pulita (parse): '{raw_predicted_label}' → '{predicted_label}'")
            
            # Validazione valori
            if not predicted_label:
                raise ValueError("predicted_label vuoto")
            
            # Normalizza confidence nel range corretto
            if not (0.0 <= confidence <= 1.0):
                self.logger.warning(f"Confidence {confidence} normalizzato")
                confidence = max(0.0, min(1.0, confidence))
            
            if not motivation:
                motivation = f"Classificato come {predicted_label}"
            
            # NUOVO: Risoluzione semantica intelligente con controllo soglia fallback
            resolved_label, resolution_method, semantic_confidence = self._semantic_label_resolution(
                predicted_label, conversation_text, confidence  # Passa anche la confidence iniziale
            )
            
            # Aggiorna confidence basandosi sulla qualità della risoluzione
            if resolution_method == "DIRECT_MATCH":
                final_confidence = confidence
            elif resolution_method == "SEMANTIC_MATCH":
                final_confidence = min(confidence, semantic_confidence + 0.1)
            elif resolution_method == "BERTOPIC_NEW_CATEGORY":
                final_confidence = min(confidence, semantic_confidence)
            elif resolution_method == "AUTO_CREATED":
                final_confidence = semantic_confidence  # Mantieni confidence LLM originale
                self.logger.info(f"✨ Nuovo tag creato automaticamente da LLM: '{resolved_label}' (conf: {final_confidence:.3f})")
            elif resolution_method == "BERTOPIC_FALLBACK_NEW_TAG":
                final_confidence = semantic_confidence  # Usa confidence BERTopic per nuovo tag
                self.logger.info(f"🆕 Nuovo tag scoperto da fallback BERTopic: '{resolved_label}' (conf: {final_confidence:.3f})")
            elif resolution_method == "BERTOPIC_FALLBACK_EXISTING":
                final_confidence = max(0.5, semantic_confidence)  # Boost per match BERTopic
                self.logger.info(f"🔄 Tag esistente suggerito da fallback BERTopic: '{resolved_label}' (conf: {final_confidence:.3f})")
            else:
                final_confidence = max(0.1, confidence - 0.2)  # Penalizza fallback
            
            self.logger.debug(f"Risoluzione semantica: '{predicted_label}' → '{resolved_label}' via {resolution_method} (conf: {final_confidence:.3f})")
            
            trace_all(
                'EXIT', 
                '_parse_llm_response', 
                {
                    'resolved_label': resolved_label,
                    'final_confidence': final_confidence,
                    'motivation_length': len(motivation),
                    'original_label': predicted_label,
                    'resolution_method': resolution_method,
                    'json_valid': True
                }
            )
            # 🔍 TRACING EXIT - Successo
            trace_all("_parse_llm_response", "EXIT", 
                     {"status": "SUCCESS", 
                      "resolved_label": resolved_label,
                      "final_confidence": final_confidence,
                      "method": "STANDARD_PARSE"})
            
            return resolved_label, final_confidence, motivation, predicted_label  # Include etichetta originale
            
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            # 🔍 TRACING ERROR
            trace_all("_parse_llm_response", "ERROR", 
                     {"error": str(e),
                      "error_type": type(e).__name__,
                      "response_length": len(response_text)})
            
            fallback_result = self._fallback_parse_response(response_text)
            
            # 🔍 TRACING EXIT - Fallback
            trace_all("_parse_llm_response", "EXIT", 
                     {"status": "FALLBACK_USED",
                      "fallback_label": fallback_result[0],
                      "fallback_confidence": fallback_result[1]})
            
            self.logger.warning(f"Errore parsing/validazione: {e}")
            return fallback_result
    
    def _fallback_parse_response(self, response_text: str) -> Tuple[str, float, str, str]:
        """
        Fallback per parsing quando il JSON non è valido
        NESSUN PATTERN RIGIDO - solo estrazione euristica generica
        
        Returns:
            Tuple (predicted_label, confidence, motivation, original_llm_label)
        """
        trace_all(
            'ENTER', 
            '_fallback_parse_response', 
            {
                'response_length': len(response_text),
                'domain_labels_count': len(self.domain_labels),
                'response_preview': response_text[:100]
            }
        )
        
        try:
            self.logger.debug("Parsing fallback per risposta LLM malformata - nessun pattern rigido")
            
            response_lower = response_text.lower()
            
            # Cerca se il LLM ha menzionato almeno una delle etichette note
            # ma SENZA usare pattern predefiniti per la classificazione
            mentioned_labels = []
            for label in self.domain_labels:
                if label in response_lower or label.replace('_', ' ') in response_lower:
                    mentioned_labels.append(label)
            
            # Se il LLM ha menzionato esattamente una etichetta, probabilmente è quella giusta
            if len(mentioned_labels) == 1:
                raw_predicted_label = mentioned_labels[0]
                predicted_label = clean_label_text(raw_predicted_label)  # 🧹 PULIZIA ETICHETTA
                
                # 🐛 DEBUG: Log pulizia etichetta se necessaria
                if raw_predicted_label != predicted_label:
                    self.logger.info(f"🧹 Etichetta pulita (fallback): '{raw_predicted_label}' → '{predicted_label}'")
                
                # Cerca confidence nel testo in modo generico
                confidence_match = re.search(r'(\d+\.?\d*)%?', response_text)
                if confidence_match:
                    confidence = float(confidence_match.group(1))
                    if confidence > 1.0:  # Se è percentuale
                        confidence = confidence / 100.0
                    confidence = max(0.0, min(1.0, confidence))  # Clamp 0-1
                else:
                    confidence = 0.4  # Confidence moderata per parsing imperfetto
                
                motivation = f"Etichetta estratta da risposta LLM non strutturata: {predicted_label}"
                
                trace_all(
                    'EXIT', 
                    '_fallback_parse_response', 
                    {
                        'predicted_label': predicted_label,
                        'confidence': confidence,
                        'mentioned_labels_count': len(mentioned_labels),
                        'extraction_method': 'SINGLE_LABEL_MENTION'
                    }
                )
                return predicted_label, confidence, motivation, predicted_label
            
            # Se il LLM ha menzionato multiple etichette, è ambiguo
            elif len(mentioned_labels) > 1:
                motivation = f"Risposta ambigua, multiple etichette menzionate: {mentioned_labels}"
                result = ("altro", 0.2, motivation, response_text[:50])
                
                trace_all(
                    'EXIT', 
                    '_fallback_parse_response', 
                    {
                        'predicted_label': result[0],
                        'confidence': result[1],
                        'mentioned_labels_count': len(mentioned_labels),
                        'mentioned_labels': mentioned_labels,
                        'extraction_method': 'AMBIGUOUS_MULTIPLE'
                    }
                )
                return result
            
            # Nessuna etichetta riconosciuta - fallback completo
            result = ("altro", 0.1, "Risposta LLM non interpretabile senza pattern rigidi", response_text[:50])
            
            trace_all(
                'EXIT', 
                '_fallback_parse_response', 
                {
                    'predicted_label': result[0],
                    'confidence': result[1],
                    'mentioned_labels_count': len(mentioned_labels),
                    'extraction_method': 'NO_LABELS_FOUND'
                }
            )
            return result
            
        except Exception as e:
            error_result = ("altro", 0.05, f"Errore fallback parsing: {str(e)[:50]}", response_text[:50])
            
            trace_all(
                'ERROR', 
                # TODO: Aggiungere trace_all EXIT e ERROR per _fallback_parse_response
                '_fallback_parse_response', 
                {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'response_length': len(response_text),
                    'fallback_label': error_result[0]
                }
            )
            return error_result
    
    def _normalize_label(self, label: str) -> str:
        """
        Normalizza le etichette con correzione intelligente di errori comuni
        """
        original_label = label
        label = label.strip().lower().replace(' ', '_')
        
        # 1. Verifica diretta se è già una etichetta valida
        if label in self.domain_labels:
            return label
        
        # 2. Correzioni intelligenti per errori comuni del LLM
        corrections = {
            # Errori di spelling comuni
            'amministativo': 'problema_amministrativo',
            'amminstrativo': 'problema_amministrativo', 
            'amministrativo': 'problema_amministrativo',
            'problema_amministativo': 'problema_amministrativo',
            'accesso_portale': 'problema_accesso_portale',
            'portale_accesso': 'problema_accesso_portale',
            'problema_portale': 'problema_accesso_portale',
            'prenotazione_portale': 'problema_prenotazione_portale',
            'portale_prenotazione': 'problema_prenotazione_portale',
            # Varianti semantiche comuni
            'contatti': 'info_contatti',
            'informazioni_contatti': 'info_contatti',
            'esami_info': 'info_esami',
            'informazioni_esami': 'info_esami',
            'parcheggi': 'info_parcheggio',
            'parking': 'info_parcheggio',
            'cartella_clinica': 'ritiro_cartella_clinica_referti',
            'ritiro_cartella': 'ritiro_cartella_clinica_referti',
            'referti': 'ritiro_cartella_clinica_referti',
            'prenotazione': 'prenotazione_esami',
            'prenotazioni': 'prenotazione_esami',
            'appuntamento': 'prenotazione_esami',
            'visita': 'prenotazione_esami',
            # Sinonimi e varianti
            'medicina': 'parere_medico',
            'medico': 'parere_medico',
            'consulto': 'parere_medico',
            'ricoveri': 'info_ricovero',
            'ospedalizzazione': 'info_ricovero',
            'intervento': 'info_interventi',
            'operazione': 'info_interventi',
            'chirurgia': 'info_interventi'
        }
        
        # Applica correzioni dirette
        if label in corrections:
            corrected = corrections[label]
            self.logger.debug(f"Etichetta corretta: '{original_label}' → '{corrected}'")
            return corrected
        
        # 3. Fuzzy matching per similarity alta (distanza di edit < 3)
        best_match = None
        best_distance = float('inf')
        
        for domain_label in self.domain_labels:
            if domain_label == 'altro':  # Skip 'altro' nel fuzzy matching
                continue
                
            distance = self._levenshtein_distance(label, domain_label)
            
            # Se la distanza è molto piccola e il match è ragionevole
            if distance < best_distance and distance <= 2:
                # Verifica che non sia un match spurio (lunghezze troppo diverse)
                length_ratio = min(len(label), len(domain_label)) / max(len(label), len(domain_label))
                if length_ratio >= 0.6:  # Almeno 60% di lunghezza simile
                    best_distance = distance
                    best_match = domain_label
        
        if best_match:
            self.logger.debug(f"Fuzzy match trovato: '{original_label}' → '{best_match}' (distanza: {best_distance})")
            return best_match
        
        # 4. Pattern matching per categorie semantiche
        semantic_patterns = {
            'info_': ['info', 'informazione', 'informazioni', 'come', 'dove', 'quando', 'cosa'],
            'problema_': ['problema', 'errore', 'sbagliato', 'non funziona', 'non riesco'],
            'prenotazione_': ['prenota', 'appuntamento', 'visita', 'esame'],
            'ritiro_': ['ritira', 'prende', 'recupera', 'preleva']
        }
        
        label_words = label.replace('_', ' ').split()
        for prefix, keywords in semantic_patterns.items():
            if any(keyword in ' '.join(label_words) for keyword in keywords):
                # Trova l'etichetta domain che inizia con questo prefix
                candidates = [dl for dl in self.domain_labels if dl.startswith(prefix)]
                if candidates:
                    # Se c'è solo una candidata, usala
                    if len(candidates) == 1:
                        selected = candidates[0]
                        self.logger.debug(f"Pattern semantico: '{original_label}' → '{selected}'")
                        return selected
                    # Se ci sono multiple candidate, usa la più generica o 'altro'
                    # (questo evita di indovinare tra opzioni specifiche)
        
        # 5. Fallback finale
        self.logger.debug(f"Etichetta '{original_label}' non riconosciuta, normalizzata ad 'altro'")
        return 'altro'
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calcola distanza di Levenshtein tra due stringhe"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _extract_system_user_from_chatmL_prompt(self, chatml_prompt: str) -> tuple[str, str]:
        """
        Estrae system e user content da prompt ChatML-like per uso con OpenAI API
        
        Scopo della funzione:
        Parsifica il prompt ChatML generato da _build_classification_prompt
        ed estrae le sezioni system e user per l'API OpenAI messages format
        
        Parametri di input:
        chatml_prompt: Prompt in formato ChatML con <|system|>, <|user|>, <|assistant|>
        
        Valori di ritorno:
        tuple: (system_content, user_content) estratti dal prompt ChatML
        
        Data ultima modifica: 2024-12-19
        """
        trace_all(
            'ENTER', 
            '_extract_system_user_from_chatmL_prompt', 
            {
                'prompt_length': len(chatml_prompt),
                'prompt_preview': chatml_prompt[:200] + '...' if len(chatml_prompt) > 200 else chatml_prompt
            }
        )
        
        try:
            # Estrae contenuto system (tra <|system|> e <|user|>)
            system_start = chatml_prompt.find('<|system|>')
            user_start = chatml_prompt.find('<|user|>')
            assistant_start = chatml_prompt.find('<|assistant|>')
            
            if system_start == -1 or user_start == -1:
                raise ValueError("Formato ChatML non valido: mancano tag <|system|> o <|user|>")
            
            # Estrae system content
            system_content = chatml_prompt[system_start + len('<|system|>'):user_start].strip()
            
            # Estrae user content
            if assistant_start != -1:
                user_content = chatml_prompt[user_start + len('<|user|>'):assistant_start].strip()
            else:
                user_content = chatml_prompt[user_start + len('<|user|>'):].strip()
            
            if self.enable_logging:
                print(f"🔧 [ChatML Parser] System content: {len(system_content)} chars")
                print(f"🔧 [ChatML Parser] User content: {len(user_content)} chars")
            
            trace_all(
                'EXIT', 
                '_extract_system_user_from_chatmL_prompt', 
                {
                    'system_length': len(system_content),
                    'user_length': len(user_content),
                    'extraction_success': True
                }
            )
            
            return system_content, user_content
            
        except Exception as e:
            trace_all(
                'ERROR', 
                '_extract_system_user_from_chatmL_prompt', 
                {
                    'error': str(e),
                    'prompt_preview': chatml_prompt[:500] if len(chatml_prompt) > 500 else chatml_prompt
                }
            )
            raise Exception(f"Errore estrazione ChatML: {e}")
    
    def _call_openai_api_structured(self, conversation_text: str) -> Dict[str, Any]:
        """
        Chiama API OpenAI per classificazione strutturata con Function Tools
        Carica tools dal database esattamente come Ollama per coerenza completa
        
        Args:
            conversation_text: Testo della conversazione da classificare
            
        Returns:
            Dict con predicted_label, confidence, motivation
            
        Data ultima modifica: 2025-09-07
        """
        trace_all(
            '_call_openai_api_structured', 
            'ENTER',
            called_from="classify_with_motivation",
            conversation_length=len(conversation_text),
            tenant_id=self.tenant_id,
            model_name=self.model_name,
            openai_service_available=self.openai_service is not None
        )
        
        if not self.openai_service:
            raise ValueError("Servizio OpenAI non inizializzato")
        
        # Rileva se è gpt-5 e se la modalità JSON-only è attiva da config.yaml
        is_gpt5 = (self.model_name or '').lower() == 'gpt-5'
        json_only = False
        try:
            models_avail = self.config.get('llm', {}).get('models', {}).get('available', [])
            for m in models_avail:
                if isinstance(m, dict) and m.get('name') == 'gpt-5':
                    json_only = bool(m.get('json_only', False))
                    break
        except Exception:
            json_only = False
        
        # 🔄 CARICAMENTO TOOLS DAL DATABASE (stessa logica di Ollama)
        classification_tools = []
        
        if self.prompt_manager and self.tool_manager:
            try:
                # 1. Recupera gli ID dei tool dal prompt "intelligent_classifier_system"
                resolved_tenant_id = self.prompt_manager._resolve_tenant_id(self.tenant_id)
                tool_ids = self.prompt_manager.get_prompt_tools(
                    tenant_id=resolved_tenant_id,
                    prompt_name="intelligent_classifier_system",
                    engine="LLM"
                )
                
                if self.enable_logging:
                    print(f"🤖 [OpenAI] Tool IDs recuperati dal prompt: {tool_ids}")
                
                # 2. Per ogni tool ID, recupera il tool completo dal database
                for tool_id in tool_ids:
                    try:
                        # tool_id deve essere un numero intero
                        if not isinstance(tool_id, int):
                            if isinstance(tool_id, str) and tool_id.isdigit():
                                tool_id = int(tool_id)
                            else:
                                if self.enable_logging:
                                    print(f"⚠️ [OpenAI] Tool ID non valido: {tool_id}")
                                continue
                        
                        db_tool = self.tool_manager.get_tool_by_id(tool_id)
                        
                        if db_tool:
                            # Estrai i parametri corretti dal function_schema
                            function_schema = db_tool['function_schema']
                            
                            if isinstance(function_schema, dict) and 'function' in function_schema:
                                parameters = function_schema['function']['parameters']
                            else:
                                parameters = function_schema
                            
                            # Costruisce il tool per OpenAI (stesso formato di Ollama)
                            classification_tool = {
                                "type": "function", 
                                "function": {
                                    "name": db_tool['tool_name'],
                                    "description": db_tool['description'],
                                    "parameters": parameters
                                }
                            }
                            
                            # Aggiorna l'enum dei domain_labels nel tool
                            if (self.domain_labels and 
                                'properties' in classification_tool['function']['parameters'] and
                                'predicted_label' in classification_tool['function']['parameters']['properties'] and
                                'enum' in classification_tool['function']['parameters']['properties']['predicted_label']):
                                
                                classification_tool['function']['parameters']['properties']['predicted_label']['enum'] = list(self.domain_labels)
                                if self.enable_logging:
                                    print(f"🔄 [OpenAI] Aggiornato enum domain_labels: {self.domain_labels}")
                            
                            classification_tools.append(classification_tool)
                            
                            if self.enable_logging:
                                print(f"✅ [OpenAI] Tool caricato: {db_tool['tool_name']}")
                        
                        else:
                            if self.enable_logging:
                                print(f"⚠️ [OpenAI] Tool non trovato per ID: {tool_id}")
                    
                    except Exception as tool_error:
                        if self.enable_logging:
                            print(f"❌ [OpenAI] Errore caricamento tool {tool_id}: {tool_error}")
                        continue
                        
            except Exception as e:
                if self.enable_logging:
                    print(f"❌ [OpenAI] Errore recupero tools dal database: {e}")
        
        # Verifica tools solo se NON siamo in modalità JSON-only
        if not json_only:
            if not classification_tools:
                raise ValueError(
                    "❌ ERRORE CRITICO OpenAI: Nessun tool di classificazione disponibile dal database. "
                    "Verificare che il prompt 'intelligent_classifier_system' abbia tool associati."
                )
        
        # 🆕 VALIDAZIONE TOOLS usando helper dell'OpenAIService (solo se non JSON-only)
        if not json_only:
            valid_tools = []
            for tool in classification_tools:
                if self.openai_service.validate_tool_schema(tool):
                    valid_tools.append(tool)
                    if self.enable_logging:
                        print(f"✅ [OpenAI] Tool '{tool['function']['name']}' validato correttamente")
                else:
                    if self.enable_logging:
                        print(f"❌ [OpenAI] Tool '{tool.get('function', {}).get('name', 'unknown')}' non valido!")
            if not valid_tools:
                raise ValueError("❌ ERRORE CRITICO OpenAI: Nessun tool valido dopo validazione schema")
            classification_tools = valid_tools  # Usa solo tools validati
        
        try:
            # 🔄 USA STESSA FUNZIONE DI OLLAMA: _build_classification_prompt
            # Questo garantisce prompt identico con esempi dinamici dal database
            full_prompt = self._build_classification_prompt(conversation_text, context=None)
            
            # 🔧 ESTRAZIONE system e user dal prompt ChatML di Ollama
            system_content, user_content = self._extract_system_user_from_chatmL_prompt(full_prompt)
            
            # Messaggi per chat completion con function tools
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            if self.enable_logging:
                print(f"🤖 [OpenAI] Using same prompt as Ollama via _build_classification_prompt")
                print(f"🤖 [OpenAI] System prompt length: {len(system_content)} chars")
                print(f"🤖 [OpenAI] User prompt length: {len(user_content)} chars")
                print(f"🤖 [OpenAI] Calling {self.model_name} with {len(classification_tools)} tools")
            
            # 🔧 NUOVO: Chiama OpenAI con function tools usando metodo asincrono per parallelismo
            import asyncio
            
            # 🎯 FORZA L'USO DEL TOOL SPECIFICO per il tenant corrente (se tools attivi)
            primary_tool_name = (classification_tools[0]['function']['name'] 
                                 if (not json_only and classification_tools) else 'classify_conversation')
            
            if self.enable_logging:
                print(f"🎯 [OpenAI] Forzando uso del tool: {primary_tool_name}")
            
            # is_gpt5 già calcolato sopra
            
            async def call_openai_async():
                if is_gpt5:
                    # GPT-5 usa l'API 'responses'
                    if json_only:
                        # Modalità SOLO JSON: disabilita tools e impone schema
                        if self.enable_logging:
                            print("🧩 [GPT-5/JSON-only] Attivo text.format=json_schema e disabilito tools")
                        # Costruisci schema JSON stretto su etichette del dominio
                        text_format = IntelligentClassifier._build_gpt5_json_schema(self.domain_labels)
                        return await self.openai_service.responses_completion(
                            model=self.model_name,
                            input_data=messages,
                            text=text_format,
                            max_tokens=self.max_tokens if hasattr(self, 'max_tokens') else None
                        )
                    else:
                        # Tools attivi: usa tools + tool_choice
                        if self.enable_logging:
                            print(f"🚀 [GPT-5] Usando API 'responses' con tools: {len(classification_tools)} e tool_choice mirato")
                        # 🔥 VINCOLO GPT-5: Aggiungi SEMPRE parametro 'text' anche con tools
                        return await self.openai_service.responses_completion(
                            model=self.model_name,
                            input_data=messages,
                            text={'format': {'type': 'text'}},  # ← OBBLIGATORIO per GPT-5
                            tools=classification_tools,
                            tool_choice={"type": "function", "function": {"name": primary_tool_name}},
                            max_tokens=self.max_tokens if hasattr(self, 'max_tokens') else None
                        )
                else:
                    # Modelli tradizionali (GPT-4o, etc) usano chat.completions
                    return await self.openai_service.chat_completion(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        tools=classification_tools,  # 🔑 AGGIUNTA: Function tools per OpenAI
                        tool_choice={"type": "function", "function": {"name": primary_tool_name}}  # 🔧 FORZA: Usa sempre il tool specifico del tenant
                    )
            
            # Esegui la chiamata asincrona in modo sincrono per compatibilità
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Se siamo già in un loop asincrono, usiamo un thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, call_openai_async())
                        response = future.result()
                else:
                    response = loop.run_until_complete(call_openai_async())
            except RuntimeError:
                # Nessun loop esistente, creane uno nuovo
                response = asyncio.run(call_openai_async())
            
            if self.enable_logging:
                print(f"✅ [OpenAI] Risposta ricevuta usando metodo asincrono per parallelismo")
            
            # 🆕 GESTIONE RISPOSTA GPT-5 vs GPT-4o
            if is_gpt5:
                if json_only:
                    # SOLO JSON: ignora tool calls, usa solo output_text
                    output_text = (response.get('output_text') or '').strip()
                    if not output_text:
                        raise ValueError("Risposta GPT-5 senza output_text in modalità JSON-only")
                    if self.enable_logging:
                        print(f"🧩 [GPT-5/JSON-only] Parsing output_text: {output_text[:100]}...")
                    try:
                        structured_result = json.loads(output_text)
                    except json.JSONDecodeError:
                        structured_result = self._extract_json_from_openai_response(output_text)
                else:
                    # 1) Prova a estrarre function tool calls dalla Responses API
                    tool_calls = []
                    try:
                        tool_calls = self.openai_service.extract_responses_tool_calls(response)
                    except Exception as _tcerr:
                        if self.enable_logging:
                            print(f"⚠️ [GPT-5] Errore estrazione tool calls: {_tcerr}")

                    if tool_calls:
                        # Usa gli arguments del tool specifico
                        used = False
                        for tc in tool_calls:
                            if tc.get('function', {}).get('name') == primary_tool_name and 'arguments' in tc['function']:
                                args = tc['function']['arguments']
                                if isinstance(args, str):
                                    structured_result = json.loads(args)
                                else:
                                    structured_result = args
                                used = True
                                if self.enable_logging:
                                    print(f"✅ [GPT-5] Function call estratto: {structured_result}")
                                break
                        if not used:
                            # Se non abbiamo trovato il tool atteso, prendi il primo disponibile
                            tc = tool_calls[0]
                            args = tc.get('function', {}).get('arguments', '{}')
                            structured_result = json.loads(args) if isinstance(args, str) else args
                            if self.enable_logging:
                                print(f"ℹ️ [GPT-5] Usato primo tool call disponibile: {structured_result}")
                    else:
                        # 2) Nessun tool call: usa output_text normalizzato
                        output_text = (response.get('output_text') or '').strip()
                        if not output_text:
                            raise ValueError("Risposta GPT-5 senza output_text né tool_calls")
                        if self.enable_logging:
                            print(f"🚀 [GPT-5] Parsing output_text: {output_text[:100]}...")
                        try:
                            structured_result = json.loads(output_text)
                        except json.JSONDecodeError:
                            structured_result = self._extract_json_from_openai_response(output_text)
            else:
                # Parsing tradizionale per GPT-4o e altri modelli
                # Estrae contenuto dalla risposta OpenAI con function call
                if 'choices' not in response or not response['choices']:
                    raise ValueError("Risposta OpenAI vuota o malformata")
                
                # 🆕 NUOVO: Usa i metodi helper dell'OpenAIService per gestire tool calls
                tool_calls = self.openai_service.extract_tool_calls(response)
                
                if tool_calls:
                    # Verifica che sia il tool call giusto
                    for tool_call in tool_calls:
                        if (tool_call.get('function', {}).get('name') == 'classify_conversation' and 
                            'arguments' in tool_call['function']):
                            
                            # Parse dei function call arguments
                            if isinstance(tool_call['function']['arguments'], str):
                                structured_result = json.loads(tool_call['function']['arguments'])
                            else:
                                structured_result = tool_call['function']['arguments']
                            
                            if self.enable_logging:
                                print(f"✅ [OpenAI] Function call received via helper: {structured_result}")
                            
                            break
                    else:
                        raise ValueError(f"Function call 'classify_conversation' non trovata nei tool calls: {tool_calls}")
                else:
                    # Fallback: Se non c'è function call, prova parsing del contenuto
                    message = response['choices'][0]['message']
                    if 'content' in message and message['content']:
                        content = message['content'].strip()
                        
                        if self.enable_logging:
                            print(f"⚠️ [OpenAI] No function call detected, parsing content: {content[:100]}...")
                        
                        try:
                            structured_result = json.loads(content)
                        except json.JSONDecodeError as e:
                            # Fallback con parsing robusto esistente
                            structured_result = self._extract_json_from_openai_response(content)
                    else:
                        raise ValueError("Risposta OpenAI senza tool_calls né content")
            
            # Valida struttura risposta
            required_fields = ['predicted_label', 'confidence', 'motivation']
            for field in required_fields:
                if field not in structured_result:
                    raise ValueError(f"Campo richiesto '{field}' mancante nella risposta OpenAI")
            
            # Normalizza confidence come float
            if isinstance(structured_result['confidence'], str):
                conf_str = structured_result['confidence'].replace('%', '')
                structured_result['confidence'] = float(conf_str) / 100.0
            elif isinstance(structured_result['confidence'], (int, float)):
                conf_val = float(structured_result['confidence'])
                if conf_val > 1.0:
                    structured_result['confidence'] = conf_val / 100.0
                else:
                    structured_result['confidence'] = conf_val
            
            if self.enable_logging:
                print(f"✅ [OpenAI] Classification successful: {structured_result['predicted_label']} (conf: {structured_result['confidence']:.3f})")
            
            trace_all(
                '_call_openai_api_structured',
                'EXIT',
                called_from="classify_with_motivation",
                predicted_label=structured_result['predicted_label'],
                confidence=structured_result['confidence'],
                tokens_used=response.get('usage', {}).get('total_tokens', 0),
                tools_used=len(classification_tools),
                exit_reason="SUCCESS"
            )
            
            return structured_result
            
        except Exception as e:
            if self.enable_logging:
                print(f"❌ [OpenAI] Errore durante classificazione: {e}")
            
            trace_all(
                '_call_openai_api_structured',
                'ERROR',
                called_from="classify_with_motivation",
                error_type=type(e).__name__,
                error_message=str(e),
                conversation_length=len(conversation_text),
                tools_loaded=len(classification_tools)
            )
            
            raise
    
    async def _call_openai_api_batch(self, conversations: List[str]) -> List[Dict[str, Any]]:
        """
        Chiama API OpenAI in batch per classificazione parallela di più conversazioni
        Utilizza il metodo batch_chat_completions dell'OpenAIService per ottimizzare le prestazioni
        
        Args:
            conversations: Lista di testi di conversazioni da classificare
            
        Returns:
            Lista di dizionari con predicted_label, confidence, motivation per ogni conversazione
            
        Data ultima modifica: 2025-01-31
        """
        trace_all(
            '_call_openai_api_batch', 
            'ENTER',
            called_from="classify_multiple_conversations",
            batch_size=len(conversations),
            tenant_id=self.tenant_id,
            model_name=self.model_name,
            openai_service_available=self.openai_service is not None
        )
        
        if not self.openai_service:
            raise ValueError("Servizio OpenAI non inizializzato")
        
        if not conversations:
            return []
        
        # 🔄 CARICAMENTO TOOLS DAL DATABASE (stesso sistema di _call_openai_api_structured)
        classification_tools = []
        
        if self.prompt_manager and self.tool_manager:
            try:
                # 1. Recupera gli ID dei tool dal prompt "intelligent_classifier_system"
                resolved_tenant_id = self.prompt_manager._resolve_tenant_id(self.tenant_id)
                tool_ids = self.prompt_manager.get_prompt_tools(
                    tenant_id=resolved_tenant_id,
                    prompt_name="intelligent_classifier_system",
                    engine="LLM"
                )
                
                # 2. Per ogni tool ID, recupera il tool completo dal database
                for tool_id in tool_ids:
                    try:
                        # tool_id deve essere un numero intero
                        if not isinstance(tool_id, int):
                            if isinstance(tool_id, str) and tool_id.isdigit():
                                tool_id = int(tool_id)
                            else:
                                continue
                        
                        db_tool = self.tool_manager.get_tool_by_id(tool_id)
                        
                        if db_tool:
                            # Estrai i parametri corretti dal function_schema
                            function_schema = db_tool['function_schema']
                            
                            if isinstance(function_schema, dict) and 'function' in function_schema:
                                parameters = function_schema['function']['parameters']
                            else:
                                parameters = function_schema
                            
                            # Costruisce il tool per OpenAI
                            classification_tool = {
                                "type": "function", 
                                "function": {
                                    "name": db_tool['tool_name'],
                                    "description": db_tool['description'],
                                    "parameters": parameters
                                }
                            }
                            
                            # Log su file tool_batch_openai.log
                            
                            log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔄 [OpenAI Batch] Caricato tool di classificazione: {db_tool['tool_name']}"
                            
                            # Determina path del file di log (nella stessa directory del file corrente)
                            log_file_path = os.path.join(os.path.dirname(__file__), 'tool_batch_openai.log')
                            
                            # Salva su file
                            try:
                                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                                    log_file.write(log_message + '\n')
                            except Exception as log_error:
                                if self.enable_logging:
                                    print(f"⚠️ Errore scrittura log su file: {log_error}")
                            
                          

                            # Aggiorna l'enum dei domain_labels nel tool
                            if (self.domain_labels and 
                                'properties' in classification_tool['function']['parameters'] and
                                'predicted_label' in classification_tool['function']['parameters']['properties'] and
                                'enum' in classification_tool['function']['parameters']['properties']['predicted_label']):
                                
                                classification_tool['function']['parameters']['properties']['predicted_label']['enum'] = list(self.domain_labels)
                            
                            classification_tools.append(classification_tool)
                        
                    except Exception as tool_error:
                        if self.enable_logging:
                            print(f"❌ [OpenAI Batch] Errore caricamento tool {tool_id}: {tool_error}")
                        continue
                        
            except Exception as e:
                if self.enable_logging:
                    print(f"❌ [OpenAI Batch] Errore recupero tools dal database: {e}")
        
        # Verifica che ci sia almeno un tool disponibile
        if not classification_tools:
            raise ValueError(
                "❌ ERRORE CRITICO OpenAI Batch: Nessun tool di classificazione disponibile dal database."
            )
        
        # 🆕 VALIDAZIONE TOOLS usando helper dell'OpenAIService
        valid_tools = []
        for tool in classification_tools:
            if self.openai_service.validate_tool_schema(tool):
                valid_tools.append(tool)
            else:
                if self.enable_logging:
                    print(f"❌ [OpenAI Batch] Tool '{tool.get('function', {}).get('name', 'unknown')}' non valido!")
        
        if not valid_tools:
            raise ValueError("❌ ERRORE CRITICO OpenAI Batch: Nessun tool valido dopo validazione schema")
        
        classification_tools = valid_tools
        
        # 🎯 FORZA L'USO DEL TOOL SPECIFICO per il tenant corrente nel batch
        primary_tool_name = classification_tools[0]['function']['name'] if classification_tools else 'classify_conversation'
        
        if self.enable_logging:
            print(f"🎯 [OpenAI Batch] Forzando uso del tool: {primary_tool_name}")
        
        # 🚀 PREPARAZIONE BATCH REQUESTS
        batch_requests = []
        
        for i, conversation_text in enumerate(conversations):
            try:
                # 🔄 USA STESSA FUNZIONE DI OLLAMA: _build_classification_prompt
                full_prompt = self._build_classification_prompt(conversation_text, context=None)
                
                # 🔧 ESTRAZIONE system e user dal prompt ChatML di Ollama
                system_content, user_content = self._extract_system_user_from_chatmL_prompt(full_prompt)
                
                # Messaggi per chat completion con function tools
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
                
                # Richiesta per il batch
                request = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "tools": classification_tools,
                    "tool_choice": {"type": "function", "function": {"name": primary_tool_name}}  # 🔧 FORZA: Usa sempre il tool specifico del tenant
                }
                
                batch_requests.append(request)
                
            except Exception as e:
                if self.enable_logging:
                    print(f"❌ [OpenAI Batch] Errore preparazione richiesta {i}: {e}")
                raise
        
        if self.enable_logging:
            print(f"🚀 [OpenAI Batch] Preparate {len(batch_requests)} richieste per batch processing")
        
        try:
            # 🔥 CHIAMATA BATCH PARALLELA usando OpenAIService.batch_chat_completions
            batch_responses = await self.openai_service.batch_chat_completions(batch_requests)
            
            if self.enable_logging:
                print(f"✅ [OpenAI Batch] Ricevute {len(batch_responses)} risposte dal batch processing")
            
            # 🔄 PROCESSING DELLE RISPOSTE
            results = []
            
            for i, response in enumerate(batch_responses):
                try:
                    # Gestisce errori specifici di singole richieste
                    if 'error' in response:
                        if self.enable_logging:
                            print(f"❌ [OpenAI Batch] Errore richiesta {i}: {response['error']}")
                        # Fallback per errore singolo
                        fallback_result = {
                            'predicted_label': 'altro',
                            'confidence': 0.1,
                            'motivation': f"Errore batch processing: {response['error']}"
                        }
                        results.append(fallback_result)
                        continue
                    
                    # Estrae contenuto dalla risposta OpenAI con function call
                    if 'choices' not in response or not response['choices']:
                        raise ValueError(f"Risposta OpenAI vuota o malformata per richiesta {i}")
                    
                    # 🆕 USA i metodi helper dell'OpenAIService per gestire tool calls
                    tool_calls = self.openai_service.extract_tool_calls(response)
                    
                    structured_result = None
                    
                    if tool_calls:
                        # Verifica che sia il tool call giusto
                        for tool_call in tool_calls:
                            if (tool_call.get('function', {}).get('name') == 'classify_conversation' and 
                                'arguments' in tool_call['function']):
                                
                                # Parse dei function call arguments
                                if isinstance(tool_call['function']['arguments'], str):
                                    structured_result = json.loads(tool_call['function']['arguments'])
                                else:
                                    structured_result = tool_call['function']['arguments']
                                break
                    
                    if not structured_result:
                        # Fallback: Se non c'è function call, prova parsing del contenuto
                        message = response['choices'][0]['message']
                        if 'content' in message and message['content']:
                            content = message['content'].strip()
                            try:
                                structured_result = json.loads(content)
                            except json.JSONDecodeError:
                                structured_result = self._extract_json_from_openai_response(content)
                        else:
                            raise ValueError(f"Risposta OpenAI senza tool_calls né content per richiesta {i}")
                    
                    # Valida struttura risposta
                    required_fields = ['predicted_label', 'confidence', 'motivation']
                    for field in required_fields:
                        if field not in structured_result:
                            raise ValueError(f"Campo richiesto '{field}' mancante nella risposta {i}")
                    
                    # Normalizza confidence come float
                    if isinstance(structured_result['confidence'], str):
                        conf_str = structured_result['confidence'].replace('%', '')
                        structured_result['confidence'] = float(conf_str) / 100.0
                    elif isinstance(structured_result['confidence'], (int, float)):
                        conf_val = float(structured_result['confidence'])
                        if conf_val > 1.0:
                            structured_result['confidence'] = conf_val / 100.0
                        else:
                            structured_result['confidence'] = conf_val
                    
                    results.append(structured_result)
                    
                except Exception as e:
                    if self.enable_logging:
                        print(f"❌ [OpenAI Batch] Errore processing risposta {i}: {e}")
                    
                    # Fallback per errore singolo
                    fallback_result = {
                        'predicted_label': 'altro',
                        'confidence': 0.1,
                        'motivation': f"Errore processing risposta: {str(e)}"
                    }
                    results.append(fallback_result)
            
            if self.enable_logging:
                print(f"✅ [OpenAI Batch] Processing completato: {len(results)} risultati")
            
            trace_all(
                '_call_openai_api_batch',
                'EXIT',
                called_from="classify_multiple_conversations",
                batch_size=len(conversations),
                results_count=len(results),
                tools_used=len(classification_tools),
                exit_reason="SUCCESS"
            )
            
            return results
            
        except Exception as e:
            if self.enable_logging:
                print(f"❌ [OpenAI Batch] Errore durante batch processing: {e}")
            
            trace_all(
                '_call_openai_api_batch',
                'ERROR',
                called_from="classify_multiple_conversations",
                error_type=type(e).__name__,
                error_message=str(e),
                batch_size=len(conversations),
                tools_loaded=len(classification_tools)
            )
            
            raise

    def classify_multiple_conversations_optimized(self, 
                                                conversations: List[str],
                                                context: Optional[str] = None) -> List[ClassificationResult]:
        """
        Classifica multiple conversazioni utilizzando batch processing ottimizzato per OpenAI
        
        Scopo della funzione:
        - Determina automaticamente se usare batch processing o chiamate singole
        - Utilizza classification_batch_size dal config per ottimizzare le performance
        - Mantiene compatibilità con chiamate singole per Ollama
        
        Parametri di input e output:
        - conversations: List[str] - Lista di testi di conversazioni da classificare
        - context: Optional[str] - Contesto aggiuntivo opzionale
        
        Valori di ritorno:
        - List[ClassificationResult]: Lista di risultati di classificazione
        
        Tracciamento aggiornamenti:
        - 2025-01-31: Creazione con supporto automatico batch/single processing
        
        Args:
            conversations: Lista di conversazioni da classificare
            context: Contesto aggiuntivo opzionale
            
        Returns:
            Lista di ClassificationResult con dettagli completi
        """
        # ============================================================================
        # 🔥🔥🔥 DEBUG CAZZO CHIARISSIMO - BATCH VS SINGLE PROCESSING 🔥🔥🔥
        # ============================================================================
        print(f"\n{'='*80}")
        print(f"🚨 [BATCH DEBUG] INIZIO classify_multiple_conversations_optimized")
        print(f"🚨 [BATCH DEBUG] Tenant: {self.tenant_id}")
        print(f"🚨 [BATCH DEBUG] Model: {self.model_name}")
        print(f"🚨 [BATCH DEBUG] Conversazioni da processare: {len(conversations)}")
        print(f"{'='*80}\n")
        
        trace_all("classify_multiple_conversations_optimized", "ENTER", 
                 called_from="pipeline_or_external",
                 conversations_count=len(conversations),
                 tenant_id=self.tenant_id,
                 model_name=self.model_name,
                 engine_type="to_be_determined")
        
        if not conversations:
            print(f"🚨 [BATCH DEBUG] NESSUNA CONVERSAZIONE - ESCO SUBITO")
            return []
        
        # 🔧 LETTURA CONFIGURAZIONE BATCH SIZE dal config
        batch_size = 1  # Default: chiamate singole
        
        print(f"🚨 [BATCH DEBUG] 1. LETTURA CONFIG BATCH_SIZE...")
        # Carica batch_size dal config se disponibile
        try:
            config = self._load_config()
            print(f"🚨 [BATCH DEBUG] Config caricato: {config is not None}")
            if config and 'pipeline' in config and 'classification_batch_size' in config['pipeline']:
                batch_size = config['pipeline']['classification_batch_size']
                print(f"🚨 [BATCH DEBUG] Batch size dal config: {batch_size}")
                
                # Validazione range (1-200 come specificato dall'utente)
                if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 200:
                    print(f"🚨 [BATCH DEBUG] ⚠️ Batch size non valido: {batch_size}, uso default 32")
                    batch_size = 32
                else:
                    print(f"🚨 [BATCH DEBUG] ✅ Batch size valido: {batch_size}")
            else:
                print(f"� [BATCH DEBUG] Config non contiene classification_batch_size, uso default 32")
                batch_size = 32
            
        except Exception as e:
            print(f"🚨 [BATCH DEBUG] ❌ Errore caricamento config: {e}")
            batch_size = 32
        
        print(f"🚨 [BATCH DEBUG] BATCH_SIZE FINALE: {batch_size}")
        
        # 🎯 RILEVAMENTO AUTOMATICO ENGINE TYPE
        print(f"\n🚨 [BATCH DEBUG] 2. RILEVAMENTO ENGINE TYPE...")
        print(f"🚨 [BATCH DEBUG] hasattr openai_service: {hasattr(self, 'openai_service')}")
        print(f"🚨 [BATCH DEBUG] openai_service value: {getattr(self, 'openai_service', 'NOT_SET')}")
        print(f"🚨 [BATCH DEBUG] openai_service is not None: {getattr(self, 'openai_service', None) is not None}")
        
        # Determina automaticamente il tipo di engine basato sui servizi disponibili
        if hasattr(self, 'openai_service') and self.openai_service is not None:
            engine_type = 'openai'
            print(f"🚨 [BATCH DEBUG] ✅ ENGINE TYPE: OPENAI")
            print(f"🚨 [BATCH DEBUG] OpenAI Service type: {type(self.openai_service).__name__}")
        else:
            engine_type = 'ollama'
            print(f"🚨 [BATCH DEBUG] ❌ ENGINE TYPE: OLLAMA (OpenAI non disponibile)")
        
        # 🎯 STRATEGIA BATCH: Solo per OpenAI, chiamate singole per Ollama
        print(f"\n🚨 [BATCH DEBUG] 3. DECISIONE STRATEGIA BATCH...")
        print(f"� [BATCH DEBUG] Engine type: {engine_type}")
        print(f"🚨 [BATCH DEBUG] Conversazioni > 1: {len(conversations) > 1} ({len(conversations)} conv)")
        print(f"🚨 [BATCH DEBUG] Batch size > 1: {batch_size > 1} (batch_size={batch_size})")
        
        use_batch_processing = (
            engine_type == 'openai' and      # Solo per OpenAI
            len(conversations) > 1 and        # Più di una conversazione
            batch_size > 1                    # Batch size configurato > 1
        )
        
        print(f"\n🚨 [BATCH DEBUG] ===== RISULTATO FINALE =====")
        if use_batch_processing:
            print(f"🔥🔥� [BATCH DEBUG] STRATEGIA: BATCH PROCESSING ATTIVATO! 🔥🔥🔥")
        else:
            print(f"❌❌❌ [BATCH DEBUG] STRATEGIA: CHIAMATE SINGOLE (NO BATCH) ❌❌❌")
        print(f"🚨 [BATCH DEBUG] Engine: {engine_type}")
        print(f"🚨 [BATCH DEBUG] Conversazioni: {len(conversations)}")
        print(f"🚨 [BATCH DEBUG] Batch size: {batch_size}")
        print(f"🚨 [BATCH DEBUG] Use batch: {use_batch_processing}")
        print(f"{'='*80}\n")
        
        # 📊 TRACE AGGIUNTIVO dopo determinazione strategia
        trace_all("classify_multiple_conversations_optimized", "STRATEGY_DETERMINED", 
                 called_from="pipeline_or_external",
                 conversations_count=len(conversations),
                 engine_type=engine_type,
                 use_batch_processing=use_batch_processing,
                 batch_size=batch_size)
        
        results = []
        
        if use_batch_processing:
            # 🚀 BATCH PROCESSING per OpenAI con parallelismo
            print(f"\n🔥🔥🔥 [BATCH DEBUG] ENTRANDO IN BATCH PROCESSING! 🔥🔥🔥")
            try:
                print(f"🔥 [OpenAI Batch] Avvio batch processing per {len(conversations)} conversazioni")
                
                # Dividi in chunks se necessario
                chunks = [conversations[i:i + batch_size] for i in range(0, len(conversations), batch_size)]
                print(f"🔥 [OpenAI Batch] Diviso in {len(chunks)} chunks con batch_size={batch_size}")
                
                # 🚀 OTTIMIZZAZIONE: PROCESSAMENTO PARALLELO DI TUTTI I CHUNK
                import asyncio
                
                async def process_all_chunks_parallel():
                    """Processa tutti i chunk in parallelo per massimizzare performance"""
                    
                    # Crea task per tutti i chunk simultaneamente
                    chunk_tasks = []
                    for chunk_idx, chunk in enumerate(chunks):
                        print(f"🎯 [OpenAI Parallel] Creando task per chunk {chunk_idx + 1}/{len(chunks)} "
                              f"({len(chunk)} conversazioni)")
                        
                        # Crea task asincrono per ogni chunk
                        task = self._call_openai_api_batch(chunk)
                        chunk_tasks.append(task)
                    
                    print(f"🚀 [OpenAI Parallel] Avvio processamento parallelo di {len(chunk_tasks)} chunk...")
                    
                    # Esegue tutti i chunk in parallelo con asyncio.gather
                    all_chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
                    
                    print(f"✅ [OpenAI Parallel] Tutti i chunk completati in parallelo!")
                    
                    # Processa risultati e gestisci eventuali eccezioni
                    combined_results = []
                    for chunk_idx, chunk_results in enumerate(all_chunk_results):
                        if isinstance(chunk_results, Exception):
                            print(f"❌ [OpenAI Parallel] Errore nel chunk {chunk_idx + 1}: {chunk_results}")
                            # Fallback per chunk fallito: crea risultati vuoti
                            chunk_size = len(chunks[chunk_idx])
                            chunk_results = [{'predicted_label': 'altro', 'confidence': 0.1, 'motivation': f'Chunk error: {chunk_results}'}] * chunk_size
                        
                        print(f"📊 [OpenAI Parallel] Chunk {chunk_idx + 1}: {len(chunk_results)} risultati ricevuti")
                        combined_results.extend(chunk_results)
                    
                    return combined_results
                
                # Esecuzione del processamento parallelo
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Se siamo già in un loop asincrono, usiamo un thread
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, process_all_chunks_parallel())
                            all_results = future.result()
                    else:
                        all_results = loop.run_until_complete(process_all_chunks_parallel())
                except RuntimeError:
                    # Nessun loop esistente, creane uno nuovo
                    all_results = asyncio.run(process_all_chunks_parallel())
                
                print(f"🎉 [OpenAI Parallel] Processamento parallelo completato: {len(all_results)} risultati totali")
                
                # Converte risultati dict in ClassificationResult
                for result_dict in all_results:
                    classification_result = ClassificationResult(
                        predicted_label=result_dict['predicted_label'],
                        confidence=result_dict['confidence'],
                        motivation=result_dict['motivation'],
                        method='openai_batch_parallel',  # Indica processamento parallelo
                        response_time=0.0,  # TODO: tracciare tempo se necessario
                        cache_hit=False,
                        embedding_similarity=None
                    )
                    results.append(classification_result)
                
                if self.enable_logging:
                    print(f"✅ [OpenAI Batch] Completato batch processing: {len(results)} risultati")
                
            except Exception as e:
                if self.enable_logging:
                    print(f"❌ [OpenAI Batch] Errore batch processing, fallback a chiamate singole: {e}")
                
                # Fallback a chiamate singole in caso di errore batch
                use_batch_processing = False
        
        if not use_batch_processing:
            # 🔄 CHIAMATE SINGOLE (per Ollama o fallback OpenAI)
            print(f"\n❌❌❌ [SINGLE DEBUG] ENTRANDO IN SINGLE PROCESSING! ❌❌❌")
            print(f"🔁 [Single] Processing {len(conversations)} conversazioni con chiamate singole")
            print(f"🔁 [Single] Engine: {engine_type}")
            
            for i, conversation_text in enumerate(conversations):
                try:
                    print(f"🔄 [Single] Processing conversazione {i + 1}/{len(conversations)}")
                    
                    result = self.classify_with_motivation(conversation_text, context)
                    results.append(result)
                    print(f"✅ [Single] Conversazione {i + 1} completata: {result.predicted_label}")
                    
                except Exception as e:
                    print(f"❌ [Single] Errore conversazione {i}: {e}")
                    
                    # Fallback result per errore singolo
                    fallback_result = ClassificationResult(
                        predicted_label='altro',
                        confidence=0.1,
                        motivation=f"Errore classificazione: {str(e)}",
                        method='fallback_error',
                        response_time=0.0,
                        cache_hit=False,
                        embedding_similarity=None
                    )
                    results.append(fallback_result)
        
        print(f"\n{'='*80}")
        print(f"🚨 [BATCH DEBUG] FINE classify_multiple_conversations_optimized")
        print(f"🚨 [BATCH DEBUG] Strategia usata: {'BATCH' if use_batch_processing else 'SINGLE'}")
        print(f"🚨 [BATCH DEBUG] Risultati totali: {len(results)}")
        print(f"🚨 [BATCH DEBUG] Engine type: {engine_type}")
        print(f"{'='*80}\n")
        
# TODO: Aggiungere trace_all EXIT e ERROR per classify_multiple_conversations_optimized
        
        trace_all("classify_multiple_conversations_optimized", "EXIT", 
                 called_from="pipeline_or_external",
                 conversations_count=len(conversations),
                 results_count=len(results),
                 processing_mode='batch' if use_batch_processing else 'single',
                 batch_size_used=batch_size,
                 exit_reason="SUCCESS")
        
        return results
    
    def _load_config(self) -> Optional[Dict]:
        """
        Carica configurazione integrata database + config.yaml per batch processing
        
        Scopo della funzione:
            Carica configurazione batch processing con priorità:
            1. Database (se disponibile e tenant configurato)
            2. config.yaml (fallback)
        
        Valori di ritorno:
            Dict con configurazione batch processing o None se errore
        
        Tracciamento aggiornamenti:
            2025-09-09: Integrazione database per batch processing dinamico
        """
        
        # 🗄️ PRIORITÀ 1: CARICA DA DATABASE (SE DISPONIBILE)
        if (hasattr(self, 'ai_config_service') and self.ai_config_service and 
            hasattr(self, 'tenant_id') and self.tenant_id):
            try:
                batch_config = self.ai_config_service.get_batch_processing_config(self.tenant_id)
                
                if batch_config and batch_config.get('source') == 'database':
                    # Costruisce config format compatibile con il sistema esistente
                    database_config = {
                        'pipeline': {
                            'classification_batch_size': batch_config.get('classification_batch_size', 32),
                            'max_parallel_calls': batch_config.get('max_parallel_calls', 200)
                        },
                        'source': 'database',
                        'updated_at': batch_config.get('updated_at')
                    }
                    
                    if self.enable_logging:
                        print(f"✅ [BATCH CONFIG] Caricato da database per tenant {self.tenant_id}: "
                              f"batch_size={batch_config.get('classification_batch_size')}")
                    
                    return database_config
                    
            except Exception as e:
                if self.enable_logging:
                    print(f"⚠️ [BATCH CONFIG] Errore caricamento database per tenant {getattr(self, 'tenant_id', 'unknown')}: {e}")
                    print("🔄 [BATCH CONFIG] Fallback a config.yaml...")
        
        # 📄 PRIORITÀ 2: FALLBACK A CONFIG.YAML
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        yaml_config['source'] = 'config.yaml'
                        if self.enable_logging:
                            batch_size = yaml_config.get('pipeline', {}).get('classification_batch_size', 'non configurato')
                            print(f"✅ [BATCH CONFIG] Caricato da config.yaml: batch_size={batch_size}")
                        return yaml_config
        except Exception as e:
            if self.enable_logging:
                print(f"⚠️ [BATCH CONFIG] Errore caricamento config.yaml: {e}")
        
        # ❌ NESSUNA CONFIGURAZIONE TROVATA
        if self.enable_logging:
            print(f"⚠️ [BATCH CONFIG] Nessuna configurazione disponibile - usando valori default")
        
        return None

    def _extract_json_from_openai_response(self, content: str) -> Dict[str, Any]:
        """
        Estrae JSON da risposta OpenAI potenzialmente malformata
        
        Args:
            content: Contenuto della risposta OpenAI
            
        Returns:
            Dizionario estratto dalla risposta
            
        Data ultima modifica: 2025-01-31
        """
        # 🔍 TRACING ENTER
        trace_all("_extract_json_from_openai_response", "ENTER", 
                 called_from="_call_openai_api_structured",
                 content_length=len(content) if content else 0,
                 content_preview=content[:100] if content else "None")
        
        # Pattern per trovare JSON nella risposta
        json_patterns = [
            r'\{[^{}]*"predicted_label"[^{}]*\}',
            r'\{.*?"predicted_label".*?\}',
            r'\{.*\}'
        ]
        
        for i, pattern in enumerate(json_patterns):
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    result = json.loads(match)
                    if 'predicted_label' in result:
                        # Aggiungi campi mancanti con valori di default
                        if 'confidence' not in result:
                            result['confidence'] = 0.5
                        if 'motivation' not in result:
                            result['motivation'] = "Classificazione estratta da risposta parziale"
                        
                        trace_all("_extract_json_from_openai_response", "EXIT", 
                                 called_from="_call_openai_api_structured",
                                 extraction_method=f"JSON_PATTERN_{i}",
                                 predicted_label=result['predicted_label'],
                                 confidence=result['confidence'])
                        return result
                except json.JSONDecodeError:
                    continue
        
        # Fallback finale: cerca informazioni con regex
        label_match = re.search(r'"predicted_label":\s*"([^"]+)"', content)
        confidence_match = re.search(r'"confidence":\s*([0-9.]+)', content)
        motivation_match = re.search(r'"motivation":\s*"([^"]+)"', content)
        
        fallback_result = {
            'predicted_label': label_match.group(1) if label_match else 'altro',
            'confidence': float(confidence_match.group(1)) if confidence_match else 0.5,
            'motivation': motivation_match.group(1) if motivation_match else 'Estratto da risposta malformata'
        }
        
        trace_all("_extract_json_from_openai_response", "EXIT", 
                 called_from="_call_openai_api_structured",
                 extraction_method="REGEX_FALLBACK",
                 predicted_label=fallback_result['predicted_label'],
                 confidence=fallback_result['confidence'])
        
        return fallback_result
    
    def _call_llm_api_structured(self, conversation_text: str) -> Dict[str, Any]:
        """
        Metodo unificato per chiamate strutturate LLM (delega a Ollama o OpenAI)
        
        Args:
            conversation_text: Testo della conversazione da classificare
            
        Returns:
            Dict con predicted_label, confidence, motivation
            
        Data ultima modifica: 2025-01-31
        """
        # 🔍 TRACING ENTER
        trace_all("_call_llm_api_structured", "ENTER", 
                 called_from="classify_with_motivation",
                 conversation_length=len(conversation_text) if conversation_text else 0,
                 is_openai_model=getattr(self, 'is_openai_model', False),
                 openai_service_available=getattr(self, 'openai_service', None) is not None)
        
        try:
            if self.is_openai_model and self.openai_service:
                # ✅ USA IL METODO STRUTTURATO NORMALE PER OPENAI (SINGOLE CONVERSAZIONI)
                result = self._call_openai_api_structured(conversation_text)
                trace_all("_call_llm_api_structured", "EXIT", 
                         called_from="classify_with_motivation",
                         llm_provider="OPENAI",
                         predicted_label=result.get('predicted_label', 'unknown'),
                         confidence=result.get('confidence', 0.0))
                return result
            else:
                result = self._call_ollama_api_structured(conversation_text)
                trace_all("_call_llm_api_structured", "EXIT", 
                         called_from="classify_with_motivation",
                         llm_provider="OLLAMA",
                         predicted_label=result.get('predicted_label', 'unknown'),
                         confidence=result.get('confidence', 0.0))
                return result
                
        except Exception as e:
            trace_all("_call_llm_api_structured", "ERROR", 
                     called_from="classify_with_motivation",
                     error_type=type(e).__name__,
                     error_message=str(e),
                     llm_provider="OPENAI" if getattr(self, 'is_openai_model', False) else "OLLAMA")
            raise e
    
    def _call_ollama_api_structured(self, conversation_text: str) -> Dict[str, Any]:
        """
        IMPLEMENTAZIONE DEFINITIVA: Usa Function Tools di Ollama/Mistral
        Supporta sia modalità standard che raw mode per Mistral 7B v0.3
        Elimina completamente il parsing manuale - Mistral restituisce JSON strutturato
        Recupera i tool dal database tramite il prompt e i tool ID associati
        
        Args:
            conversation_text: Testo della conversazione da classificare
            
        Returns:
            Dict con predicted_label, confidence, motivation (JSON garantito dalle function tools)
            
        Aggiornamenti:
            2025-08-31: Aggiunto supporto raw mode per Mistral 7B v0.3 con function calling
        """
        trace_all(
            'ENTER', 
            '_call_ollama_api_structured', 
            {
                'conversation_length': len(conversation_text),
                'tenant_id': self.tenant_id,
                'model_name': self.model_name,
                'prompt_manager_available': self.prompt_manager is not None,
                'tool_manager_available': self.tool_manager is not None,
                'domain_labels_count': len(self.domain_labels)
            }
        )
        
        # Recupero dei function tools dal database tramite PromptManager e ToolManager
        classification_tools = []
        
        if self.prompt_manager and self.tool_manager:
                try:
                    # 1. Recupera gli ID dei tool dal prompt "intelligent_classifier_system"
                    resolved_tenant_id = self.prompt_manager._resolve_tenant_id(self.tenant_id)
                    tool_ids = self.prompt_manager.get_prompt_tools(
                        tenant_id=resolved_tenant_id,
                        prompt_name="intelligent_classifier_system",
                        engine="LLM"
                    )
                    
                    if self.enable_logging:
                        print(f"🔍 Tool IDs recuperati dal prompt: {tool_ids}")
                    
                    # 2. Per ogni tool ID, recupera il tool completo dal database
                    for tool_id in tool_ids:
                        try:
                            # tool_id deve essere un numero intero
                            if not isinstance(tool_id, int):
                                if isinstance(tool_id, str) and tool_id.isdigit():
                                    tool_id = int(tool_id)
                                else:
                                    if self.enable_logging:
                                        print(f"⚠️ Tool ID non valido (deve essere numerico): {tool_id}")
                                    continue
                            
                            db_tool = self.tool_manager.get_tool_by_id(tool_id)
                            
                            if db_tool:
                                # 🔧 CORREZIONE CRITICAL: Estrai i parametri corretti dal function_schema
                                function_schema = db_tool['function_schema']
                                
                                # Se il function_schema ha la struttura annidata, estraiamo solo i parametri
                                if isinstance(function_schema, dict) and 'function' in function_schema:
                                    # Schema del database ha struttura errata con doppia nidificazione
                                    parameters = function_schema['function']['parameters']
                                    if self.enable_logging:
                                        print(f"⚠️ Schema corretto: estratti parametri da struttura annidata")
                                else:
                                    # Schema già nella forma corretta
                                    parameters = function_schema
                                
                                # Costruisce il tool per Ollama usando la struttura corretta secondo documentazione ufficiale
                                classification_tool = {
                                    "type": "function", 
                                    "function": {
                                        "name": db_tool['tool_name'],
                                        "description": db_tool['description'],
                                        "parameters": parameters  # 🔑 SOLO i parametri, senza doppia nidificazione
                                    }
                                }
                                
                                # Aggiorna l'enum dei domain_labels nel tool dal database
                                if (self.domain_labels and 
                                    'properties' in classification_tool['function']['parameters'] and
                                    'predicted_label' in classification_tool['function']['parameters']['properties']):
                                    classification_tool['function']['parameters']['properties']['predicted_label']['enum'] = list(self.domain_labels)
                                
                                classification_tools.append(classification_tool)
                                
                                if self.enable_logging:
                                    print(f"✅ Tool recuperato dal database: {db_tool['tool_name']} (ID: {tool_id}) per tenant {db_tool['tenant_id']}")
                            else:
                                if self.enable_logging:
                                    print(f"⚠️ Tool ID {tool_id} non trovato o non attivo nel database")
                                    
                        except Exception as e:
                            if self.enable_logging:
                                print(f"⚠️ Errore recupero tool ID {tool_id}: {e}")
                        
                except Exception as e:
                    if self.enable_logging:
                        print(f"⚠️ Errore nel recupero dei tool dal prompt: {e}")
        
        # Verifica che ci sia almeno un tool disponibile
        if not classification_tools:
            trace_all(
                'ERROR', 
                '_call_ollama_api_structured', 
                {
                    'error': 'No classification tools available',
                    'tools_found': len(classification_tools),
                    'prompt_manager_available': self.prompt_manager is not None,
                    'tool_manager_available': self.tool_manager is not None
                }
            )
            raise ValueError(
                "❌ ERRORE CRITICO: Nessun tool di classificazione disponibile dal database. "
                "Verificare che il prompt 'intelligent_classifier_system' abbia tool associati e che siano attivi."
            )
        
        # Messaggio di sistema ottimizzato per function calling
        system_message = f"""
        Sei un classificatore esperto di conversazioni mediche per l'ospedale Humanitas.            Le categorie disponibili sono:
            {', '.join(self.domain_labels) if self.domain_labels else 'altro'}
        
        Analizza la conversazione fornita e usa una delle function tools disponibili per restituire:
        - predicted_label: ESATTAMENTE una delle categorie elencate sopra
        - confidence: valore da 0.0 a 1.0 basato sulla chiarezza del contenuto
        - motivation: spiegazione breve e chiara in italiano (massimo 150 caratteri)
        
        DEVI SEMPRE usare una delle function tools disponibili per rispondere.
        """
        
        # 🚀 NUOVO: Verifica se usare raw mode per Mistral 7B v0.3 function calling
        raw_mode_config = self.config.get('llm', {}).get('ollama', {}).get('raw_mode', {})
        use_raw_mode = (
            raw_mode_config.get('enabled', False) and 
            self.model_name in raw_mode_config.get('models_requiring_raw', [])
        )
        
        if use_raw_mode and classification_tools:
            if self.enable_logging:
                print(f"🔧 Uso RAW MODE per {self.model_name} con function calling")
            return self._call_ollama_api_raw_mode(conversation_text, classification_tools, system_message)
        
        # Payload per function calling con Ollama/Mistral (modalità standard)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user", 
                    "content": f"Classifica questa conversazione medica:\n\n{conversation_text}"
                }
            ],
            "tools": classification_tools,  # 🔑 ARRAY DI FUNCTION TOOLS dal database
            "stream": False,
            "options": {
                "temperature": self.temperature,  # 🔧 USA CONFIG TENANT!
                "num_predict": self.max_tokens,   # 🔧 USA CONFIG TENANT!
                "top_p": self.top_p,              # 🔧 USA CONFIG TENANT!
                "top_k": self.top_k               # 🔧 USA CONFIG TENANT!
            }
        }
        
        # 🔍 DEBUG: Stampa il tool prima di mandarlo all'LLM
        if self.enable_logging:
            print(f"🛠️ DEBUG - Tools inviati all'LLM ({len(classification_tools)} tools):")
            for i, tool in enumerate(classification_tools):
                print(f"  Tool #{i+1}: {json.dumps(tool, indent=2, ensure_ascii=False)}")
            print(f"🌐 URL Ollama: {self.ollama_url}/api/chat")
            print(f"🤖 Modello: {self.model_name}")
            print(f"📦 Payload completo:")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        
        try:
            response = self.session.post(
                f"{self.ollama_url}/api/chat",  # Usa /api/chat per function tools
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # 🔍 DEBUG AVANZATO: Stampa la risposta RAW completa di Ollama
            if self.enable_logging:
                print(f"� DEBUG RAW - Status Code: {response.status_code}")
                print(f"🔥 DEBUG RAW - Headers: {dict(response.headers)}")
                print(f"🔥 DEBUG RAW - Risposta RAW completa da Ollama:")
                print("="*80)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                print("="*80)
                print(f"🔥 DEBUG RAW - Tipo risposta: {type(result)}")
                print(f"🔥 DEBUG RAW - Chiavi root: {list(result.keys())}")
                if 'message' in result:
                    print(f"🔥 DEBUG RAW - Tipo message: {type(result['message'])}")
                    print(f"🔥 DEBUG RAW - Chiavi message: {list(result['message'].keys())}")
                    if 'content' in result['message']:
                        content_raw = result['message']['content']
                        print(f"🔥 DEBUG RAW - Content RAW (lunghezza {len(content_raw)}):")
                        print(f"'{content_raw}'")
                        print(f"🔥 DEBUG RAW - Content tipo: {type(content_raw)}")
                        print(f"🔥 DEBUG RAW - Content bytes: {content_raw.encode('utf-8')}")
                    if 'tool_calls' in result['message']:
                        print(f"🔥 DEBUG RAW - tool_calls presente: {result['message']['tool_calls']}")
                        print(f"🔥 DEBUG RAW - tool_calls tipo: {type(result['message']['tool_calls'])}")
                    else:
                        print(f"🔥 DEBUG RAW - tool_calls ASSENTE!")
                print("="*80)
            
            # Verifica la struttura della risposta di function calling
            if 'message' not in result:
                raise ValueError(f"Risposta API Ollama malformata: manca 'message'")
                
            message = result['message']
            
            # 🔍 DEBUG AVANZATO: Analisi dettagliata del messaggio
            if self.enable_logging:
                print(f"� DEBUG PARSING - Inizio analisi messaggio:")
                print(f"  - Chiavi message: {list(message.keys())}")
                print(f"  - Ha tool_calls: {'tool_calls' in message}")
                print(f"  - Ha content: {'content' in message}")
                print(f"  - Ha role: {'role' in message}")
                if 'tool_calls' in message:
                    print(f"  - tool_calls valore: {message['tool_calls']}")
                    print(f"  - tool_calls tipo: {type(message['tool_calls'])}")
                    print(f"  - tool_calls lunghezza: {len(message['tool_calls']) if message['tool_calls'] else 0}")
                if 'content' in message:
                    content = message['content']
                    print(f"  - content valore: '{content}'")
                    print(f"  - content tipo: {type(content)}")
                    print(f"  - content lunghezza: {len(content) if content else 0}")
                    print(f"  - content è vuoto: {not content or content.strip() == ''}")
            
            # Controlla se il modello ha fatto una function call
            if 'tool_calls' in message and message['tool_calls']:
                if self.enable_logging:
                    print(f"🔥 DEBUG PARSING - PERCORSO FUNCTION CALL ATTIVATO")
                
                tool_call = message['tool_calls'][0]  # Prima function call
                
                if self.enable_logging:
                    print(f"🔥 DEBUG PARSING - Tool call ricevuta:")
                    print(f"  - tool_call completa: {tool_call}")
                    print(f"  - tool_call tipo: {type(tool_call)}")
                    print(f"  - tool_call chiavi: {list(tool_call.keys()) if isinstance(tool_call, dict) else 'NON DICT'}")
                
                if (tool_call.get('function', {}).get('name') == 'classify_conversation' and 
                    'arguments' in tool_call['function']):
                    
                    # Estrai gli argomenti della function call
                    arguments = tool_call['function']['arguments']
                    
                    if self.enable_logging:
                        print(f"🔥 DEBUG PARSING - Arguments ricevuti:")
                        print(f"  - arguments raw: {arguments}")
                        print(f"  - arguments tipo: {type(arguments)}")
                    
                    # Se arguments è una stringa, parsala come JSON
                    if isinstance(arguments, str):
                        if self.enable_logging:
                            print(f"🔥 DEBUG PARSING - Arguments è stringa, parsing JSON...")
                        try:
                            arguments = json.loads(arguments)
                            if self.enable_logging:
                                print(f"🔥 DEBUG PARSING - JSON parsed: {arguments}")
                        except json.JSONDecodeError as e:
                            if self.enable_logging:
                                print(f"🔥 DEBUG PARSING - ERRORE JSON parse: {e}")
                            raise
                    
                    # Valida che abbiamo tutti i campi richiesti
                    if all(key in arguments for key in ['predicted_label', 'confidence', 'motivation']):
                        if self.enable_logging:
                            print(f"🔥 DEBUG PARSING - SUCCESSO! Tutti i campi presenti: {arguments}")
                        self.logger.info(f"✅ Function call classificazione ricevuta: {arguments}")
                        trace_all(
                            'EXIT', 
                            '_call_ollama_api_structured', 
                            {
                                'predicted_label': arguments['predicted_label'],
                                'confidence': arguments['confidence'],
                                'motivation_length': len(arguments['motivation']),
                                'parsing_method': 'FUNCTION_CALL',
                                'tools_used': len(classification_tools),
                                'response_format': 'STRUCTURED'
                            }
                        )
                        return arguments
                    else:
                        missing_keys = [key for key in ['predicted_label', 'confidence', 'motivation'] if key not in arguments]
                        if self.enable_logging:
                            print(f"🔥 DEBUG PARSING - ERRORE! Campi mancanti: {missing_keys}")
                            print(f"🔥 DEBUG PARSING - Arguments disponibili: {list(arguments.keys())}")
                        raise ValueError(f"Function call incompleta: mancano campi richiesti {missing_keys} in {arguments}")
                else:
                    if self.enable_logging:
                        print(f"🔥 DEBUG PARSING - Function call non riconosciuta o malformata")
                        if 'function' in tool_call:
                            print(f"  - function name: {tool_call['function'].get('name', 'MISSING')}")
                            print(f"  - ha arguments: {'arguments' in tool_call['function']}")
                    raise ValueError(f"Function call non riconosciuta: {tool_call}")
            
            # Se non c'è function call, prova a parsare il contenuto come fallback
            elif 'content' in message:
                if self.enable_logging:
                    print(f"🔥 DEBUG PARSING - PERCORSO FALLBACK CONTENT ATTIVATO")
                
                content = message['content'].strip()
                if self.enable_logging:
                    print(f"🔥 DEBUG PARSING - Content per fallback: '{content}'")
                    print(f"🔥 DEBUG PARSING - Content lunghezza: {len(content)}")
                
                self.logger.warning(f"⚠️ Modello non ha usato function call, tento parsing contenuto: {content[:100]}...")
                
                # 🔍 DEBUG AGGIUNTO: Stampa la risposta LLM completa che ha causato l'errore di parsing
                if self.enable_logging:
                    print(f"🚨 RISPOSTA LLM COMPLETA CHE HA CAUSATO ERRORE PARSING:")
                    print("=" * 100)
                    print(content)
                    print("=" * 100)
                    print(f"📊 Lunghezza totale: {len(content)} caratteri")
                    print(f"📊 Tipo: {type(content)}")
                    print("=" * 100)
                
                # Fallback: prova a parsare come JSON
                if self.enable_logging:
                    print(f"🔥 DEBUG PARSING - Tentativo 1: JSON diretto")
                try:
                    parsed_content = json.loads(content)
                    if self.enable_logging:
                        print(f"🔥 DEBUG PARSING - JSON diretto riuscito: {parsed_content}")
                    if all(key in parsed_content for key in ['predicted_label', 'confidence', 'motivation']):
                        if self.enable_logging:
                            print(f"🔥 DEBUG PARSING - JSON diretto SUCCESSO con tutti i campi!")
                        return parsed_content
                    else:
                        if self.enable_logging:
                            print(f"🔥 DEBUG PARSING - JSON diretto manca campi: {list(parsed_content.keys())}")
                except json.JSONDecodeError as e:
                    if self.enable_logging:
                        print(f"🔥 DEBUG PARSING - JSON diretto fallito: {e}")
                    pass
                
                # 🔧 NUOVO: Fallback per formato Mistral-nemo pseudo function call
                if self.enable_logging:
                    print(f"🔥 DEBUG PARSING - Tentativo 2: Mistral-nemo pseudo function call")
                # Formato: **{"name": "classify_conversation", "arguments": {"predicted_label": "...", "confidence": ...}}
                try:
                    if '**{"name": "classify_conversation"' in content:
                        if self.enable_logging:
                            print(f"🔥 DEBUG PARSING - Trovato pattern Mistral-nemo!")
                        # Estrai solo la parte JSON dopo "arguments":
                        start_idx = content.find('"arguments": ')
                        if start_idx != -1:
                            start_idx += len('"arguments": ')
                            # Trova la fine del JSON degli arguments
                            json_content = content[start_idx:]
                            if self.enable_logging:
                                print(f"🔥 DEBUG PARSING - JSON content estratto: '{json_content}'")
                            
                            # 🔧 MIGLIORATO: Gestione JSON troncato - ricostruisce automaticamente
                            # Controlla se è troncato (mancano chiusure, virgole dangling, etc.)
                            if (json_content.endswith('}}') or 
                                json_content.endswith('...') or 
                                json_content.endswith(', ') or
                                json_content.endswith(',') or
                                not json_content.strip().endswith('}')):
                                
                                if self.enable_logging:
                                    print(f"🔥 DEBUG PARSING - JSON troncato/malformato rilevato, tento ricostruzione")
                                
                                # Pulisce e estrae quello che c'è
                                clean_json = json_content.replace('...', '').replace('}}', '').rstrip(', ')
                                if not clean_json.strip().endswith('}'):
                                    clean_json += '}'
                                
                                if self.enable_logging:
                                    print(f"🔥 DEBUG PARSING - JSON pulito: '{clean_json}'")
                                
                                try:
                                    partial = json.loads(clean_json)
                                    # Completa con valori mancanti
                                    if 'predicted_label' not in partial:
                                        partial['predicted_label'] = 'altro'
                                    if 'confidence' not in partial:
                                        partial['confidence'] = 0.3
                                    if 'motivation' not in partial:
                                        partial['motivation'] = 'Risposta incompleta/troncata dal modello'
                                    
                                    if self.enable_logging:
                                        print(f"🔥 DEBUG PARSING - JSON ricostruito con successo: {partial}")
                                    return partial
                                    
                                except json.JSONDecodeError as e:
                                    if self.enable_logging:
                                        print(f"🔥 DEBUG PARSING - Errore JSON decode: {e}")
                                        print(f"🔥 DEBUG PARSING - Tentativo ricostruzione manuale...")
                                    
                                    # Ultimo tentativo: ricostruzione manuale
                                    if '"predicted_label":' in clean_json:
                                        try:
                                            # Estrai almeno la label
                                            import re
                                            label_match = re.search(r'"predicted_label":\s*"([^"]*)"', clean_json)
                                            if label_match:
                                                label = label_match.group(1)
                                                if self.enable_logging:
                                                    print(f"🔥 DEBUG PARSING - Label estratta manualmente: {label}")
                                                return {
                                                    'predicted_label': label,
                                                    'confidence': 0.5,
                                                    'motivation': f'Label estratta da risposta malformata: {label}'
                                                }
                                        except Exception as manual_ex:
                                            if self.enable_logging:
                                                print(f"🔥 DEBUG PARSING - Anche ricostruzione manuale fallita: {manual_ex}")
                                
                                if self.enable_logging:
                                    print(f"🔥 DEBUG PARSING - Tutte le ricostruzioni fallite")
                            
                            if self.enable_logging:
                                print(f"🔥 DEBUG PARSING - JSON content pulito: '{json_content}'")
                            
                            try:
                                arguments = json.loads(json_content)
                                if self.enable_logging:
                                    print(f"🔥 DEBUG PARSING - Mistral arguments parsed: {arguments}")
                                if all(key in arguments for key in ['predicted_label', 'confidence', 'motivation']):
                                    if self.enable_logging:
                                        print(f"🔥 DEBUG PARSING - Mistral pseudo function call SUCCESSO!")
                                    self.logger.info(f"✅ Parsing Mistral-nemo pseudo function call riuscito: {arguments['predicted_label']} (conf: {arguments['confidence']})")
                                    return arguments
                                else:
                                    if self.enable_logging:
                                        print(f"🔥 DEBUG PARSING - Mistral manca campi: {list(arguments.keys())}")
                            except json.JSONDecodeError as e:
                                if self.enable_logging:
                                    print(f"🔥 DEBUG PARSING - Errore parsing Mistral JSON: {e}")
                                self.logger.debug(f"Errore parsing pseudo function call: {e}")
                        else:
                            if self.enable_logging:
                                print(f"🔥 DEBUG PARSING - Pattern Mistral trovato ma arguments non trovato")
                    else:
                        if self.enable_logging:
                            print(f"🔥 DEBUG PARSING - Pattern Mistral non trovato nel content")
                            
                    # 🔧 NUOVO: Parsing per formato semplice **Motivazione:** ...
                    if '**Motivazione:**' in content or '**Risposta:**' in content:
                        if self.enable_logging:
                            print(f"🔥 DEBUG PARSING - Trovato formato testo descrittivo")
                        # Prova a estrarre qualche informazione dal testo
                        lines = content.split('\n')
                        motivation = ""
                        predicted_label = "altro"
                        confidence = 0.3
                        
                        for line in lines:
                            if 'categoria' in line.lower() or 'appartiene' in line.lower():
                                # Cerca pattern come "categoria X" o "appartiene alla categoria Y"
                                words = line.split()
                                for i, word in enumerate(words):
                                    if word.strip('"').lower() in [label.lower() for label in self.domain_labels]:
                                        predicted_label = word.strip('"')
                                        confidence = 0.6
                                        if self.enable_logging:
                                            print(f"🔥 DEBUG PARSING - Trovata categoria nel testo: {predicted_label}")
                                        break
                            elif 'motivazione' in line.lower():
                                motivation = line.split(':', 1)[-1].strip() if ':' in line else line
                        
                        if not motivation:
                            motivation = "Estratto da risposta descrittiva"
                            
                        result = {
                            "predicted_label": predicted_label,
                            "confidence": confidence,
                            "motivation": motivation[:150]  # Limita lunghezza
                        }
                        if self.enable_logging:
                            print(f"🔥 DEBUG PARSING - Risultato da testo descrittivo: {result}")
                        return result
                except Exception as e:
                    if self.enable_logging:
                        print(f"🔥 DEBUG PARSING - Errore generale Mistral parsing: {e}")
                    self.logger.debug(f"Errore parsing Mistral-nemo format: {e}")
                    pass
                
                # 🔧 ALTRO: Fallback per formato chiave-valore (YAML-like)
                if self.enable_logging:
                    print(f"🔥 DEBUG PARSING - Tentativo 3: YAML-like parsing")
                try:
                    # Parsing per formato: predicted_label: valore\nconfidence: valore\nmotivation: valore
                    parsed_result = {}
                    lines = content.strip().split('\n')
                    if self.enable_logging:
                        print(f"🔥 DEBUG PARSING - Linee da parsare: {lines}")
                    
                    for line in lines:
                        line = line.strip()
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            if self.enable_logging:
                                print(f"🔥 DEBUG PARSING - Trovato key='{key}', value='{value}'")
                            
                            if key == 'predicted_label':
                                parsed_result['predicted_label'] = value
                            elif key == 'confidence':
                                try:
                                    parsed_result['confidence'] = float(value)
                                except ValueError:
                                    parsed_result['confidence'] = 0.5
                                    if self.enable_logging:
                                        print(f"🔥 DEBUG PARSING - Errore parsing confidence, uso 0.5")
                            elif key == 'motivation':
                                parsed_result['motivation'] = value
                    
                    if self.enable_logging:
                        print(f"🔥 DEBUG PARSING - YAML-like risultato: {parsed_result}")
                    
                    # Verifica che abbiamo tutti i campi richiesti
                    if all(key in parsed_result for key in ['predicted_label', 'confidence', 'motivation']):
                        if self.enable_logging:
                            print(f"🔥 DEBUG PARSING - YAML-like SUCCESSO!")
                        self.logger.info(f"✅ Parsing YAML-like riuscito: {parsed_result['predicted_label']} (conf: {parsed_result['confidence']})")
                        trace_all(
                            'EXIT', 
                            '_call_ollama_api_structured', 
                            {
                                'predicted_label': parsed_result['predicted_label'],
                                'confidence': parsed_result['confidence'],
                                'motivation_length': len(parsed_result['motivation']),
                                'parsing_method': 'YAML_LIKE',
                                'tools_used': len(classification_tools),
                                'response_format': 'FALLBACK'
                            }
                        )
                        return parsed_result
                    else:
                        missing_yaml = [key for key in ['predicted_label', 'confidence', 'motivation'] if key not in parsed_result]
                        if self.enable_logging:
                            print(f"🔥 DEBUG PARSING - YAML-like manca campi: {missing_yaml}")
                        
                except Exception as e:
                    if self.enable_logging:
                        print(f"🔥 DEBUG PARSING - Errore YAML-like: {e}")
                    self.logger.debug(f"Errore parsing YAML-like: {e}")
                    pass
                
                # 🔧 FALLBACK INTELLIGENTE: Estrae classificazione nascosta nel testo
                if self.enable_logging:
                    print(f"🔥 DEBUG PARSING - FALLBACK INTELLIGENTE attivato")
                
                # Cerca pattern di classificazione nascosta nel content
                import re
                
                # Pattern 1: Predicted_label: "LABEL"
                label_pattern = re.search(r'Predicted_label:\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
                # Pattern 2: "predicted_label": "LABEL" 
                json_pattern = re.search(r'"predicted_label":\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
                
                extracted_label = None
                extracted_confidence = 0.2
                
                if label_pattern:
                    extracted_label = label_pattern.group(1)
                    if self.enable_logging:
                        print(f"🔧 FALLBACK - Label estratta da pattern 1: {extracted_label}")
                elif json_pattern:
                    extracted_label = json_pattern.group(1)
                    if self.enable_logging:
                        print(f"🔧 FALLBACK - Label estratta da pattern 2: {extracted_label}")
                
                # Cerca confidence se disponibile
                conf_pattern = re.search(r'[Cc]onfidence:\s*([0-9.]+)', content)
                if conf_pattern:
                    try:
                        extracted_confidence = float(conf_pattern.group(1))
                        if self.enable_logging:
                            print(f"🔧 FALLBACK - Confidence estratta: {extracted_confidence}")
                    except ValueError:
                        pass
                
                # Valida che la label estratta sia nelle domain_labels
                if extracted_label and extracted_label.upper() in [dl.upper() for dl in self.domain_labels]:
                    if self.enable_logging:
                        print(f"🔧 FALLBACK - Label validata: {extracted_label}")
                    fallback_result = {
                        "predicted_label": extracted_label,
                        "confidence": extracted_confidence,
                        "motivation": f"Estratta da risposta non strutturata: {extracted_label}"
                    }
                else:
                    if self.enable_logging:
                        print(f"🔥 DEBUG PARSING - FALLBACK ESTREMO attivato (nessuna label valida estratta)")
                    fallback_result = {
                        "predicted_label": "altro",
                        "confidence": 0.2,
                        "motivation": f"Risposta non strutturata: {content[:100]}"
                    }
                trace_all(
                    'EXIT', 
                    '_call_ollama_api_structured', 
                    {
                        'predicted_label': fallback_result['predicted_label'],
                        'confidence': fallback_result['confidence'],
                        'motivation_length': len(fallback_result['motivation']),
                        'parsing_method': 'EXTREME_FALLBACK',
                        'tools_used': len(classification_tools),
                        'response_format': 'UNSTRUCTURED'
                    }
                )
                return fallback_result
            else:
                if self.enable_logging:
                    print(f"🔥 DEBUG PARSING - ERRORE: Nessun tool_calls né content nella risposta!")
                error_msg = "Risposta senza tool_calls né content"
                trace_all(
                    'ERROR', 
                    '_call_ollama_api_structured', 
                    {
                        'error': error_msg,
                        'response_keys': list(result.keys()) if 'result' in locals() else [],
                        'message_keys': list(message.keys()) if 'message' in locals() else [],
                        'tools_used': len(classification_tools)
                    }
                )
                raise ValueError(error_msg)
                
        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as e:
            error_result = {
                "predicted_label": "altro",
                "confidence": 0.1,
                "motivation": f"Errore API: {str(e)[:100]}"
            }
            trace_all(
                'ERROR', 
                '_call_ollama_api_structured', 
                {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'conversation_length': len(conversation_text),
                    'tools_used': len(classification_tools) if 'classification_tools' in locals() else 0,
                    'fallback_result': error_result
                }
            )
            self.logger.error(f"❌ Errore Ollama Function Tools API: {e}")
            return error_result

    def _call_ollama_api_raw_mode(self, conversation_text: str, tools: List[Dict], system_message: str) -> Dict[str, Any]:
        """
        Implementa function calling per Mistral 7B v0.3 usando raw mode di Ollama
        Formato richiesto: [AVAILABLE_TOOLS] [tools_json][/AVAILABLE_TOOLS][INST] query [/INST]
        
        Args:
            conversation_text: Testo della conversazione da classificare
            tools: Lista dei function tools dal database
            system_message: Messaggio di sistema
            
        Returns:
            Dict con predicted_label, confidence, motivation
            
        Autore: Valerio Bignardi
        Data: 2025-08-31
        """
        try:
            # Costruisce il prompt raw per Mistral 7B v0.3 function calling
            tools_json = json.dumps(tools, ensure_ascii=False)
            
            # Template Mistral per function calls con raw mode
            raw_prompt = f"[AVAILABLE_TOOLS] {tools_json}[/AVAILABLE_TOOLS][INST] {system_message}\n\nClassifica questa conversazione medica:\n\n{conversation_text} [/INST]"
            
            # Payload per raw mode
            payload = {
                "model": self.model_name,
                "prompt": raw_prompt,
                "raw": True,  # 🔑 CHIAVE: Abilita raw mode per Mistral function calling
                "stream": False,
                "options": {
                    "temperature": 0.01,
                    "num_predict": 300,
                    "top_p": 0.8,
                    "top_k": 20
                }
            }
            
            if self.enable_logging:
                print(f"🔧 RAW MODE - Prompt costruito per {self.model_name}:")
                print(f"📝 Raw prompt: {raw_prompt[:200]}...")
                print(f"📦 Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            
            response = self.session.post(
                f"{self.ollama_url}/api/generate",  # Raw mode usa /api/generate
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            if 'response' not in result:
                raise ValueError(f"Risposta API Ollama malformata: {result}")
            
            raw_content = result['response'].strip()
            
            if self.enable_logging:
                print(f"📥 RAW MODE - Risposta ricevuta: {raw_content[:200]}...")
            
            # Parsing della risposta Mistral raw mode function calling
            # Formato atteso: [TOOL_CALLS] [{"name": "classify_conversation", "arguments": {...}}]
            # O formato alternativo: predicted_label: xxx\nconfidence: xxx\nmotivation: xxx
            if '[TOOL_CALLS]' in raw_content:
                # Estrai il JSON tra le parentesi quadre
                start_idx = raw_content.find('[TOOL_CALLS]') + len('[TOOL_CALLS]')
                tool_calls_json = raw_content[start_idx:].strip()
                
                # Rimuovi eventuali caratteri extra
                if tool_calls_json.startswith('[') and tool_calls_json.endswith(']'):
                    tool_calls = json.loads(tool_calls_json)
                    
                    if tool_calls and 'arguments' in tool_calls[0]:
                        arguments = tool_calls[0]['arguments']
                        
                        # Valida che abbiamo tutti i campi richiesti
                        if all(key in arguments for key in ['predicted_label', 'confidence', 'motivation']):
                            self.logger.info(f"✅ RAW MODE Function call ricevuta: {arguments}")
                            return arguments
                        else:
                            raise ValueError(f"Function call incompleta nel raw mode: {arguments}")
            
            # 🔧 NUOVO: Parsing formato Mistral alternativo (predicted_label: xxx)
            elif 'predicted_label:' in raw_content and 'confidence:' in raw_content:
                try:
                    # Parse formato key: value
                    lines = raw_content.strip().split('\n')
                    result = {}
                    
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            if key == 'predicted_label':
                                result['predicted_label'] = value
                            elif key == 'confidence':
                                result['confidence'] = float(value)
                            elif key == 'motivation':
                                result['motivation'] = value
                    
                    # Valida che abbiamo tutti i campi
                    if all(key in result for key in ['predicted_label', 'confidence', 'motivation']):
                        self.logger.info(f"✅ RAW MODE Formato alternativo parsato: {result}")
                        return result
                    else:
                        self.logger.warning(f"⚠️ Campi mancanti nel formato alternativo: {result}")
                        
                except (ValueError, KeyError) as parse_error:
                    self.logger.warning(f"⚠️ Errore parsing formato alternativo: {parse_error}")
            
            # Fallback: prova parsing JSON diretto
            try:
                parsed_content = json.loads(raw_content)
                if all(key in parsed_content for key in ['predicted_label', 'confidence', 'motivation']):
                    self.logger.info(f"✅ RAW MODE JSON diretto: {parsed_content}")
                    return parsed_content
            except json.JSONDecodeError:
                pass
            
            # Fallback estremo per raw mode
            self.logger.warning(f"⚠️ RAW MODE - Risposta non strutturata: {raw_content[:100]}")
            return {
                "predicted_label": "altro",
                "confidence": 0.2,
                "motivation": f"Raw mode parsing fallito: {raw_content[:50]}"
            }
                
        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"❌ Errore RAW MODE API: {e}")
            return {
                "predicted_label": "altro",
                "confidence": 0.1,
                "motivation": f"Errore raw mode: {str(e)[:100]}"
            }

    def _call_ollama_api(self, prompt: str) -> str:
        """
        Chiama l'API Ollama con parametri ottimizzati per classificazione JSON
        
        Args:
            prompt: Prompt per il modello
            
        Returns:
            Risposta del modello
        """
        # Parametri ottimizzati per classificazione deterministrica
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.01,  # Molto bassa per output deterministico
                "num_predict": 150,  # Limitato per JSON compatto
                "top_p": 0.8,        # Ridotto per focalizzare su token probabili
                "top_k": 20,         # Ridotto per maggiore determinismo
                "repeat_penalty": 1.1,  # Evita ripetizioni
                "presence_penalty": 0.2,  # Incentiva diversità moderata
                "frequency_penalty": 0.1,  # Penalizza ripetizioni
                "stop": [
                    "\n}",           # Fine JSON
                    "```",           # Fine code block
                    "</assistant>",  # Fine risposta
                    "<|",            # Token speciali
                    "}\n\n",         # JSON + newlines
                    "Esempio",       # Evita generazione esempi aggiuntivi
                    "ESEMPIO"        # Variante maiuscola
                ]
            }
        }
        
        try:
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            if 'response' not in result:
                raise ValueError(f"Risposta API Ollama malformata: {result}")
            
            return result['response']
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Timeout nella chiamata Ollama dopo {self.timeout}s")
        
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Errore connessione Ollama: {e}")
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Errore parsing risposta Ollama: {e}")
    
    def classify_with_motivation(self, 
                               conversation_text: str,
                               context: Optional[str] = None,
                               session_id: Optional[str] = None) -> ClassificationResult:
        """
        Classifica una conversazione con motivazione dettagliata
        
        Scopo della funzione:
        - Classifica una singola conversazione usando LLM con motivazione dettagliata
        - Gestisce cache semantica, validazione embedding e fallback
        
        Parametri di input e output:
        - conversation_text: str - Testo della conversazione da classificare
        - context: Optional[str] - Contesto aggiuntivo opzionale
        
        Valori di ritorno:
        - ClassificationResult: Oggetto con predicted_label, confidence, motivation, method, etc.
        
        Tracciamento aggiornamenti:
        - 2025-09-07: Aggiunto tracing completo ENTER/EXIT/ERROR
        
        Args:
            conversation_text: Testo della conversazione da classificare
            context: Contesto aggiuntivo opzionale
            
        Returns:
            ClassificationResult con dettagli completi
        """
        # 🔍 TRACING ENTER
        trace_all("classify_with_motivation", "ENTER", 
                 called_from="batch_or_single_classification",
                 conversation_length=len(conversation_text) if conversation_text else 0,
                 context_provided=context is not None,
                 tenant_id=getattr(self, 'tenant_id', 'unknown'))
        
        if not conversation_text or not conversation_text.strip():
            return ClassificationResult(
                predicted_label="altro",
                confidence=0.0,
                motivation="Testo di input vuoto",
                method="VALIDATION_ERROR",
                timestamp=datetime.now().isoformat()
            )
        
        start_time = time.time()
        
        # FASE 1: Verifica cache semantica (se abilitata)
        if self.enable_embeddings:
            semantic_cached = self._get_semantic_cached_prediction(conversation_text, session_id=session_id)
            if semantic_cached:
                return semantic_cached
        
        # FASE 2: Verifica cache tradizionale
        cache_key = self._generate_cache_key(conversation_text, context)
        cached_result = self._get_cached_prediction(cache_key)
        if cached_result:
            return cached_result
        
        self.stats['cache_misses'] += 1
        
        try:
            # FASE 3: Verifica disponibilità del servizio
            if not self.is_available():
                return self._fallback_classification(conversation_text, "SERVICE_UNAVAILABLE")
            
            # FASE 4: Classificazione LLM con esempi ottimizzati (semantici se abilitati)
            prompt = self._build_classification_prompt(conversation_text, context)
            
            # DEBUG: Log input prima della chiamata LLM
            model_params = {
                'model': self.model_name,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'timeout': self.timeout
            }
            
            # � Verifica se usare raw mode per questo modello
            raw_mode_config = self.config.get('llm', {}).get('ollama', {}).get('raw_mode', {})
            use_raw_mode = (
                raw_mode_config.get('enabled', False) and 
                self.model_name in raw_mode_config.get('models_requiring_raw', [])
            )
            
            # �🚀 NUOVO: Usa Structured Outputs invece del parsing manuale
            raw_response = None  # 🔧 FIX: Inizializza raw_response per evitare errore
            try:
                structured_result = self._call_llm_api_structured(conversation_text)
                
                # 🐛 DEBUG: Stampa in console la risposta come richiesto
                print(f"🔍 STRUCTURED RESPONSE DEBUG: {structured_result}")
                
                # Estrai i risultati dal JSON strutturato (garantito valido)
                raw_predicted_label = structured_result["predicted_label"]
                predicted_label = clean_label_text(raw_predicted_label)  # 🧹 PULIZIA ETICHETTA
                confidence = float(structured_result["confidence"])
                motivation = structured_result["motivation"]
                original_llm_label = predicted_label  # È già l'etichetta originale
                
                # 🐛 DEBUG: Log pulizia etichetta se necessaria
                if raw_predicted_label != predicted_label:
                    print(f"🧹 LABEL CLEANING: '{raw_predicted_label}' → '{predicted_label}'")
                    self.logger.info(f"🧹 Etichetta pulita: '{raw_predicted_label}' → '{predicted_label}'")
                
                self.logger.info(f"✅ Structured Output: {predicted_label} (conf: {confidence:.3f})")
                
            except Exception as e:
                # Se stiamo usando un modello OpenAI (es. GPT-5), non effettuare fallback su Ollama
                print(f"🚨 EXCEPTION IN STRUCTURED OUTPUT: {e}")
                print(f"🔍 EXCEPTION TYPE: {type(e).__name__}")
                if getattr(self, 'is_openai_model', False):
                    self.logger.error(
                        f"❌ Structured Outputs fallito con modello OpenAI ({self.model_name}); evito fallback Ollama e propago l'errore"
                    )
                    # Rilancia per essere gestito dal catch esterno che applica solo fallback semantico/puro
                    raise
                else:
                    # Modelli non-OpenAI: usa il percorso di fallback attuale su Ollama
                    self.logger.error(f"❌ Structured Outputs fallito, uso metodo tradizionale Ollama: {e}")
                    raw_response = self._call_ollama_api_with_retry(prompt)
                    predicted_label, confidence, motivation, original_llm_label = self._parse_llm_response(raw_response, conversation_text)
            
            processing_time = time.time() - start_time
            
            # Crea risultato LLM base
            llm_result = ClassificationResult(
                predicted_label=predicted_label,
                confidence=confidence,
                motivation=motivation,
                method="LLM_STRUCTURED" if 'structured_result' in locals() else "LLM",
                raw_response=str(structured_result) if 'structured_result' in locals() else (raw_response or ""),
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
            # DEBUG: Log completo della chiamata LLM
            if self.llm_debugger and self.llm_debugger.enabled:
                session_id = f"classify_{int(time.time())}"
                self.llm_debugger.debug_llm_call(
                    session_id=session_id,
                    phase="classification",
                    input_text=conversation_text,
                    prompt=prompt,
                    model_params=model_params,
                    raw_response=str(structured_result) if 'structured_result' in locals() else (raw_response or ""),
                    parsed_response={
                        'predicted_label': predicted_label,
                        'confidence': confidence,
                        'motivation': motivation
                    },
                    confidence=confidence,
                    reasoning=motivation,
                    processing_time=processing_time,
                    context={
                        'model_used': self.model_name,
                        'cache_hit': False,
                        'context_provided': context is not None,
                        'embedding_validation_enabled': self.enable_embeddings,
                        'client_name': self.client_name,
                        'raw_mode_used': use_raw_mode if 'use_raw_mode' in locals() else False
                    }
                )
            
            # FASE 5: Validazione e correzione con embedding (se abilitata)
            if self.enable_embeddings and self.embedder is not None:
                validated_result = self._validate_llm_classification_with_embedding(conversation_text, llm_result)
                
                # FASE 6: Rilevazione potenziali nuove categorie con etichetta originale
                final_result = self._detect_potential_new_category(conversation_text, validated_result, original_llm_label)
            else:
                final_result = llm_result
            
            # Aggiorna statistiche
            self._update_stats(final_result.processing_time, success=True)
            
            # Cache del risultato (sia tradizionale che semantica)
            self._cache_prediction(cache_key, final_result)
            if self.enable_embeddings:
                self._cache_semantic_prediction(conversation_text, final_result, session_id=session_id)
            
            # Salvataggio in MongoDB (se disponibile e client specificato)
            self._save_to_mongodb(conversation_text, final_result, session_id=session_id)
            
            # Log con testo analizzato (troncato per leggibilità)
            text_preview = conversation_text[:200].replace('\n', ' ').replace('\r', ' ') if len(conversation_text) > 200 else conversation_text.replace('\n', ' ').replace('\r', ' ')
            self.logger.info(f"Classificazione completata: {final_result.predicted_label} (conf: {final_result.confidence:.3f}, tempo: {final_result.processing_time:.2f}s, metodo: {final_result.method}) - Testo: '{text_preview}{'...' if len(conversation_text) > 200 else ''}'")
            
            # 🔍 TRACING EXIT - Success
            trace_all("classify_with_motivation", "EXIT", 
                     called_from="batch_or_single_classification",
                     predicted_label=final_result.predicted_label,
                     confidence=final_result.confidence,
                     method=final_result.method,
                     processing_time=final_result.processing_time,
                     exit_reason="SUCCESS")
            
            return final_result
            
        except Exception as e:
            # 🔍 TRACING ERROR
            trace_all("classify_with_motivation", "ERROR", 
                     called_from="batch_or_single_classification",
                     error_type=type(e).__name__,
                     error_message=str(e),
                     conversation_length=len(conversation_text) if conversation_text else 0)
            
            self.logger.error(f"Errore nella classificazione: {e}")
            self._update_stats(time.time() - start_time, success=False)
            
            # VERIFICA SE È UN ERRORE DI CONFIGURAZIONE PROMPT - NON USARE FALLBACK
            error_str = str(e).lower()
            is_prompt_configuration_error = (
                'errore critico' in error_str and 
                ('prompt' in error_str or 'promptmanager' in error_str)
            )
            
            if is_prompt_configuration_error:
                # ERRORI DI CONFIGURAZIONE PROMPT DEVONO BLOCCARE IL SISTEMA
                error_msg = f"❌ SISTEMA BLOCCATO: {e}"
                self.logger.error(error_msg)
                print(error_msg)
                
                # 🔍 TRACING EXIT - Critical Error
                trace_all("classify_with_motivation", "EXIT", 
                         called_from="batch_or_single_classification",
                         exit_reason="CRITICAL_ERROR")
                
                raise e  # Rilancia l'errore originale per bloccare il sistema
            
            # Altri errori possono usare fallback
            fallback_result = self._fallback_classification(conversation_text, f"LLM_ERROR: {str(e)}")
            
            # 🔍 TRACING EXIT - Fallback
            trace_all("classify_with_motivation", "EXIT", 
                     called_from="batch_or_single_classification",
                     predicted_label=fallback_result.predicted_label,
                     confidence=fallback_result.confidence,
                     method=fallback_result.method,
                     exit_reason="FALLBACK_USED")
            
            return fallback_result
    
    def _fallback_classification(self, 
                               conversation_text: str, 
                               error_reason: str) -> ClassificationResult:
        """
        Classificazione di fallback con supporto embedding intelligente
        """
        # 🔍 TRACING ENTER
        trace_all("_fallback_classification", "ENTER", 
                 called_from="classify_with_motivation",
                 conversation_length=len(conversation_text) if conversation_text else 0,
                 error_reason=error_reason,
                 embeddings_enabled=self.enable_embeddings,
                 semantic_fallback_enabled=getattr(self, 'enable_semantic_fallback', False))
        
        self.logger.warning(f"LLM non disponibile, fallback per: {error_reason}")
        
        # Se embedding abilitati, prova fallback semantico intelligente
        if (self.enable_embeddings and self.enable_semantic_fallback and 
            self.embedder is not None and self.semantic_memory is not None):
            
            fallback_result = self._intelligent_semantic_fallback(conversation_text, error_reason)
            if fallback_result:
                self.stats['embedding_fallback_used'] += 1
                
                trace_all("_fallback_classification", "EXIT", 
                         called_from="classify_with_motivation",
                         exit_reason="SEMANTIC_FALLBACK_SUCCESS",
                         predicted_label=fallback_result.predicted_label,
                         confidence=fallback_result.confidence)
                return fallback_result
        
        # Altrimenti fallback puro attuale
        pure_result = self._pure_fallback_classification(conversation_text, error_reason)
        
        trace_all("_fallback_classification", "EXIT", 
                 called_from="classify_with_motivation",
                 exit_reason="PURE_FALLBACK",
                 predicted_label=pure_result.predicted_label,
                 confidence=pure_result.confidence)
        
        return pure_result
    
    def _intelligent_semantic_fallback(self, conversation_text: str, error_reason: str) -> Optional[ClassificationResult]:
        """
        Fallback intelligente usando embedding e memoria semantica
        """
        # 🔍 TRACING ENTER
        trace_all("_intelligent_semantic_fallback", "ENTER", 
                 called_from="_fallback_classification",
                 conversation_length=len(conversation_text) if conversation_text else 0,
                 error_reason=error_reason,
                 has_embedder=self.embedder is not None,
                 has_semantic_memory=self.semantic_memory is not None)
        
        try:
            # Genera embedding del testo
            embedding = self.embedder.encode_single(conversation_text)
            
            # Cerca classificazioni simili nella memoria semantica
            similar_classifications = self.semantic_memory.find_most_similar_classified(
                embedding, threshold=0.75, max_results=3
            )
            
            if similar_classifications:
                best_match = similar_classifications[0]
                
                # Usa classificazione più simile con confidence ridotta per incertezza
                result = ClassificationResult(
                    predicted_label=best_match['label'],
                    confidence=best_match['similarity'] * 0.7,  # Ridotta per incertezza
                    motivation=f"Classificazione per similarità semantica (sim: {best_match['similarity']:.3f}), LLM non disponibile",
                    method=f"EMBEDDING_FALLBACK_{error_reason}",
                    processing_time=0.05,  # Veloce
                    timestamp=datetime.now().isoformat()
                )
                
                # 🔍 TRACING EXIT - Success
                trace_all("_intelligent_semantic_fallback", "EXIT", 
                         called_from="_fallback_classification",
                         predicted_label=result.predicted_label,
                         confidence=result.confidence,
                         similarity=best_match['similarity'],
                         exit_reason="SEMANTIC_MATCH_FOUND")
                
                return result
            
            # 🔍 TRACING EXIT - No Match
            trace_all("_intelligent_semantic_fallback", "EXIT", 
                     called_from="_fallback_classification",
                     exit_reason="NO_SEMANTIC_MATCH",
                     similar_classifications_count=len(similar_classifications) if similar_classifications else 0)
            
            return None
            
        except Exception as e:
            # 🔍 TRACING ERROR
            trace_all("_intelligent_semantic_fallback", "ERROR", 
                     called_from="_fallback_classification",
                     error_type=type(e).__name__,
                     error_message=str(e),
                     conversation_length=len(conversation_text) if conversation_text else 0,
                     error_reason=error_reason)
            
            self.logger.warning(f"Errore fallback semantico: {e}")
            return None
    
    def _pure_fallback_classification(self, conversation_text: str, error_reason: str) -> ClassificationResult:
        """
        Classificazione di fallback PURA - logica attuale
        """
        # Analisi lunghezza per confidence euristica
        text_length = len(conversation_text.strip())
        
        # Confidence basata solo sulla lunghezza del testo (euristica semplice)
        if text_length < 10:
            confidence = 0.05  # Testo troppo breve
            motivation = "Testo troppo breve per classificazione affidabile"
        elif text_length > 500:
            confidence = 0.15  # Testo molto lungo, potrebbe essere complesso
            motivation = "Testo complesso, richiede analisi LLM non disponibile"
        else:
            confidence = 0.10  # Testo normale ma senza LLM
            motivation = "LLM non disponibile, classificazione generica"
        
        return ClassificationResult(
            predicted_label='altro',  # SEMPRE 'altro' senza pattern
            confidence=confidence,
            motivation=motivation,
            method=f"FALLBACK_PURE_{error_reason}",
            timestamp=datetime.now().isoformat()
        )
    
    def classify_conversation(self, conversation_text: str) -> Dict[str, Any]:
        """
        Classifica una conversazione (interfaccia compatibile)
        
        Scopo della funzione:
        - Wrapper per classify_with_motivation che mantiene compatibilità API
        - Conversione da ClassificationResult a Dict per retrocompatibilità
        
        Parametri di input e output:
        - conversation_text: str - Testo della conversazione da classificare
        
        Valori di ritorno:
        - Dict[str, Any]: Risultato classificazione in formato compatibile
        
        Tracciamento aggiornamenti:
        - 2025-09-06: Aggiunto tracing completo ENTER/EXIT/ERROR
        
        Args:
            conversation_text: Testo della conversazione
            
        Returns:
            Dict con risultato della classificazione
        """
        
        # 🔍 TRACING ENTER
        trace_all("classify_conversation", "ENTER", 
                 called_from="external_api_call",
                 conversation_length=len(conversation_text) if conversation_text else 0,
                 interface_type="COMPATIBILITY_WRAPPER")
        
        try:
            result = self.classify_with_motivation(conversation_text)
            
            # Formato compatibile con l'interfaccia esistente
            response_dict = {
                'predicted_label': result.predicted_label,
                'confidence': result.confidence,
                'motivation': result.motivation,
                'method': result.method,
                'processing_time': result.processing_time
            }
            
            # 🔍 TRACING EXIT - Success
            trace_all("classify_conversation", "EXIT", 
                     called_from="external_api_call",
                     predicted_label=result.predicted_label,
                     confidence=result.confidence,
                     method=result.method,
                     processing_time=result.processing_time,
                     exit_reason="SUCCESS")
            
            return response_dict
            
        except Exception as e:
            # 🔍 TRACING ERROR
            trace_all("classify_conversation", "ERROR", 
                     called_from="external_api_call",
                     error_type=type(e).__name__,
                     error_message=str(e))
            
            # 🔍 TRACING EXIT - Error
            trace_all("classify_conversation", "EXIT", 
                     called_from="external_api_call",
                     exit_reason="ERROR")
            
            # Rilancia l'errore per gestione upstream
            raise e
    
    def classify_batch(self, 
                      documents,
                      show_progress: bool = True) -> List[ClassificationResult]:
        """
        🚀 FIX METADATI: Classifica DocumentoProcessing preservando tutti i metadati
        
        Scopo della funzione:
        - Classifica oggetti DocumentoProcessing aggiornandoli con risultati
        - Preserva metadati cluster (outlier, representative, propagated)
        - Gestisce progress tracking e fallback per elementi falliti
        
        Parametri di input e output:
        - documents: List[DocumentoProcessing] O List[str] - Documenti o testi (compatibilità)
        - show_progress: bool - Se mostrare progress bar
        
        Valori di ritorno:
        - List[ClassificationResult]: Lista di risultati classificazione
        
        Tracciamento aggiornamenti:
        - 2025-09-07: Aggiunto tracing completo ENTER/EXIT/ERROR
        - 2025-01-13: Fix metadati DocumentoProcessing 
        
        Args:
            documents: Lista di DocumentoProcessing o testi da classificare
            show_progress: Se mostrare progress bar
            
        Returns:
            Lista di risultati classificazione
        """
        # 🚀 FIX: Rileva tipo input e gestisci DocumentoProcessing
        if documents and hasattr(documents[0], 'testo_completo'):
            # Input: List[DocumentoProcessing] - MANTIENI METADATI!
            print(f"🔍 classify_batch con {len(documents)} DocumentoProcessing (metadati preservati)")
            conversations = [doc.testo_completo for doc in documents]
            preserve_metadata = True
            document_objects = documents
        else:
            # Input: List[str] - Compatibilità retroattiva
            print(f"🔍 classify_batch con {len(documents)} stringhe (modalità compatibilità)")
            conversations = documents
            preserve_metadata = False
            document_objects = None
        
        # 🔍 TRACING ENTER
        trace_all("classify_batch", "ENTER", 
                 called_from="intelligent_intent_clusterer_or_api",
                 total_conversations=len(conversations),
                 show_progress=show_progress,
                 batch_size=len(conversations))
        
        if not conversations:
            # 🔍 TRACING EXIT - Empty Input
            trace_all("classify_batch", "EXIT", 
                     called_from="intelligent_intent_clusterer_or_api",
                     exit_reason="EMPTY_INPUT",
                     results_count=0)
            return []
        
        results = []
        total = len(conversations)
        successful_classifications = 0
        fallback_used = 0
        
        # Progress tracking se richiesto
        if show_progress and total > 1:
            self.logger.info(f"Classificazione batch di {total} conversazioni...")
        
        for i, conversation_text in enumerate(conversations):
            try:
                # Propaga session_id quando disponibile (DocumentoProcessing)
                sid = None
                if preserve_metadata and document_objects and i < len(document_objects):
                    try:
                        sid = getattr(document_objects[i], 'session_id', None)
                    except Exception:
                        sid = None
                result = self.classify_with_motivation(conversation_text, session_id=sid)
                results.append(result)
                
                # Traccia se è stato usato fallback
                if 'FALLBACK' in result.method.upper():
                    fallback_used += 1
                else:
                    successful_classifications += 1
                
                if show_progress and total > 10 and (i + 1) % max(1, total // 10) == 0:
                    self.logger.info(f"Progresso batch: {i + 1}/{total} ({((i + 1) / total * 100):.1f}%)")
                    
            except Exception as e:
                # 🔍 TRACING ERROR
                trace_all("classify_batch", "ERROR", 
                         called_from="intelligent_intent_clusterer_or_api",
                         error_type=type(e).__name__,
                         error_message=str(e),
                         current_item=i,
                         total_items=total)
                
                self.logger.error(f"Errore nella classificazione batch elemento {i}: {e}")
                # Crea risultato di fallback per l'elemento fallito
                fallback_result = self._fallback_classification(
                    conversation_text, f"BATCH_ERROR: {str(e)}"
                )
                results.append(fallback_result)
                fallback_used += 1
        
        if show_progress and total > 1:
            self.logger.info(f"Classificazione batch completata: {len(results)}/{total} elementi")
        
        # � FIX: Aggiorna metadati DocumentoProcessing con risultati classificazione
        if preserve_metadata and document_objects:
            for i, (doc, result) in enumerate(zip(document_objects, results)):
                doc.set_classification_result(
                    predicted_label=result.predicted_label,
                    confidence=result.confidence,
                    method=f"intelligent_classifier_{result.method}",
                    reasoning=result.motivation
                )
                print(f"  📝 Aggiornato DocumentoProcessing {doc.session_id}: {result.predicted_label} ({result.confidence:.3f})")
        
        # �🔍 TRACING EXIT - Success
        trace_all("classify_batch", "EXIT", 
                 called_from="intelligent_intent_clusterer_or_api",
                 total_conversations=total,
                 results_count=len(results),
                 successful_classifications=successful_classifications,
                 fallback_used=fallback_used,
                 success_rate=(successful_classifications / total * 100) if total > 0 else 0,
                 exit_reason="SUCCESS")
        
        return results
    
    def batch_classify(self, 
                      documents,
                      show_progress: bool = True) -> List[ClassificationResult]:
        """
        Alias per classify_batch (compatibilità schema operativo)
        
        Args:
            documents: Lista di DocumentoProcessing o testi da classificare
            show_progress: Se mostrare progress bar
            
        Returns:
            Lista di risultati classificazione
        """
        return self.classify_batch(documents, show_progress)
    
    def _update_stats(self, processing_time: float, success: bool = True) -> None:
        """Aggiorna statistiche interne"""
        self.stats['total_predictions'] += 1
        
        if not success:
            self.stats['errors'] += 1
        
        self.stats['response_times'].append(processing_time)
        
        # Mantieni solo gli ultimi 1000 tempi per calcolare la media
        if len(self.stats['response_times']) > 1000:
            self.stats['response_times'] = self.stats['response_times'][-1000:]
        
        if self.stats['response_times']:
            self.stats['avg_response_time'] = sum(self.stats['response_times']) / len(self.stats['response_times'])
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Recupera statistiche del classificatore
        
        Returns:
            Dict con statistiche dettagliate
        """
        stats = {
            'total_predictions': self.stats['total_predictions'],
            'cache_hit_rate': (self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])),
            'error_rate': (self.stats['errors'] / max(1, self.stats['total_predictions'])),
            'avg_response_time': self.stats['avg_response_time'],
            'cache_size': len(self._prediction_cache),
            'is_available': self.is_available(),
            'model_info': {
                'name': self.model_name,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'ollama_url': self.ollama_url
            },
            'embedding_features': {
                'enabled': self.enable_embeddings,
                'validation_enabled': self.enable_embedding_validation,
                'semantic_fallback_enabled': self.enable_semantic_fallback,
                'new_category_detection_enabled': self.enable_new_category_detection,
                'embedder_available': self.embedder is not None,
                'semantic_memory_available': self.semantic_memory is not None
            }
        }
        
        # Aggiungi statistiche embedding se disponibili
        if self.enable_embeddings:
            stats['semantic_cache_hits'] = self.stats['semantic_cache_hits']
            stats['semantic_cache_size'] = len(self._semantic_cache)
            stats['embedding_fallback_used'] = self.stats['embedding_fallback_used']
            stats['llm_corrections'] = self.stats['llm_corrections']
            stats['new_categories_detected'] = self.stats['new_categories_detected']
        
        return stats
    
    def clear_cache(self) -> int:
        """
        Pulisce la cache delle predizioni
        
        Returns:
            Numero di elementi rimossi dalla cache
        """
        with self._cache_lock:
            cache_size = len(self._prediction_cache)
            self._prediction_cache.clear()
            
        self.logger.info(f"Cache pulita: {cache_size} elementi rimossi")
        return cache_size
    
    def warm_up(self) -> bool:
        """
        Scalda il modello con una predizione di test
        
        Returns:
            True se il warm-up è riuscito
        """
        try:
            self.logger.info("Warm-up del modello LLM...")
            
            test_text = "Buongiorno, vorrei informazioni sugli orari di apertura"
            result = self.classify_with_motivation(test_text)
            
            success = result.method == "LLM"
            
            if success:
                self.logger.info(f"Warm-up completato con successo (tempo: {result.processing_time:.2f}s)")
            else:
                self.logger.warning(f"Warm-up fallito: {result.method}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Errore durante warm-up: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Controllo di salute completo del classificatore
        
        Returns:
            Dict con stato di salute dettagliato
        """
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        # Check 1: Disponibilità servizio
        try:
            service_available = self.is_available()
            health_status['checks']['service_availability'] = {
                'status': 'pass' if service_available else 'fail',
                'message': 'Ollama service available' if service_available else 'Ollama service unavailable'
            }
        except Exception as e:
            health_status['checks']['service_availability'] = {
                'status': 'fail',
                'message': f'Service check error: {e}'
            }
            health_status['overall_status'] = 'unhealthy'
        
        # Check 2: Test predizione
        try:
            test_result = self.classify_with_motivation("Test di classificazione")
            prediction_working = test_result.method == "LLM"
            health_status['checks']['prediction_test'] = {
                'status': 'pass' if prediction_working else 'warn',
                'message': f'Prediction test completed via {test_result.method}',
                'response_time': test_result.processing_time
            }
            
            if not prediction_working and health_status['overall_status'] == 'healthy':
                health_status['overall_status'] = 'degraded'
                
        except Exception as e:
            health_status['checks']['prediction_test'] = {
                'status': 'fail',
                'message': f'Prediction test failed: {e}'
            }
            health_status['overall_status'] = 'unhealthy'
        
        # Check 3: Statistiche performance
        stats = self.get_statistics()
        error_rate = stats['error_rate']
        avg_time = stats['avg_response_time']
        
        performance_status = 'pass'
        performance_message = f'Error rate: {error_rate:.1%}, Avg time: {avg_time:.2f}s'
        
        if error_rate > 0.1:  # >10% errori
            performance_status = 'warn'
            performance_message += ' (High error rate)'
        
        if avg_time > 10.0:  # >10s tempo medio
            performance_status = 'warn'
            performance_message += ' (Slow response time)'
        
        health_status['checks']['performance'] = {
            'status': performance_status,
            'message': performance_message,
            'error_rate': error_rate,
            'avg_response_time': avg_time
        }
        
        # Determina stato finale
        check_statuses = [check['status'] for check in health_status['checks'].values()]
        if 'fail' in check_statuses:
            health_status['overall_status'] = 'unhealthy'
        elif 'warn' in check_statuses and health_status['overall_status'] == 'healthy':
            health_status['overall_status'] = 'degraded'
        
        return health_status
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcola similarità coseno tra due embedding"""
        if self.embedder is None:
            return 0.0
        
        try:
            # Normalizza gli embedding
            emb1_norm = emb1 / np.linalg.norm(emb1)
            emb2_norm = emb2 / np.linalg.norm(emb2)
            
            # Calcola similarità coseno
            similarity = np.dot(emb1_norm, emb2_norm)
            return float(similarity)
        except Exception as e:
            self.logger.warning(f"Errore calcolo similarità coseno: {e}")
            return 0.0
    
    def _get_semantic_cached_prediction(self, conversation_text: str, session_id: Optional[str] = None) -> Optional[ClassificationResult]:
        """Cerca predizione in cache semantica se abilitata"""
        # 🔍 TRACING ENTER
        trace_all("_get_semantic_cached_prediction", "ENTER", 
                 called_from="classify_with_motivation",
                 conversation_length=len(conversation_text) if conversation_text else 0,
                 embeddings_enabled=self.enable_embeddings,
                 embedder_available=self.embedder is not None)
        
        if not self.enable_embeddings:
            trace_all("_get_semantic_cached_prediction", "EXIT", 
                     called_from="classify_with_motivation",
                     exit_reason="EMBEDDINGS_DISABLED",
                     result=None)
            return None
        
        try:
            input_embedding = None
            # 1) Tentativo: recupera embedding dalla embedding_cache centralizzata se ho session_id
            if session_id and hasattr(self, 'tenant') and self.tenant is not None:
                try:
                    import os, sys
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MongoDB'))
                    from embedding_store import EmbeddingStore  # type: ignore
                    store = EmbeddingStore()
                    fetched = store.get_embedding(self.tenant.tenant_id, session_id, consume=False)
                    if fetched:
                        input_embedding = fetched[0]
                        self.logger.debug("♻️ Semantic cache: embedding preso da embedding_cache (no ricalcolo)")
                except Exception as _e:
                    self.logger.debug(f"Semantic cache: nessun embedding in cache o errore store: {_e}")
            # 2) Fallback: calcola embedding singolo se embedder disponibile
            if input_embedding is None and self.embedder is not None and hasattr(self.embedder, 'encode_single'):
                input_embedding = self.embedder.encode_single(conversation_text)
            if input_embedding is None:
                trace_all("_get_semantic_cached_prediction", "EXIT", 
                         called_from="classify_with_motivation",
                         exit_reason="NO_EMBEDDING_AVAILABLE",
                         result=None)
                return None
            
            with self._semantic_cache_lock:
                # Cerca nella cache semantica
                for cached_embedding, cached_result, timestamp in self._semantic_cache:
                    # Verifica se cache è ancora valida
                    if datetime.now() - timestamp < timedelta(hours=self.cache_ttl_hours):
                        similarity = self._cosine_similarity(input_embedding, cached_embedding)
                        
                        # Se molto simile (>95%), considera cache hit
                        if similarity > 0.95:
                            self.stats['semantic_cache_hits'] += 1
                            self.logger.debug(f"Cache semantica hit (similarità: {similarity:.3f})")
                            
                            trace_all("_get_semantic_cached_prediction", "EXIT", 
                                     called_from="classify_with_motivation",
                                     exit_reason="CACHE_HIT",
                                     similarity=similarity,
                                     cached_label=cached_result.predicted_label)
                            return cached_result
            
            trace_all("_get_semantic_cached_prediction", "EXIT", 
                     called_from="classify_with_motivation",
                     exit_reason="CACHE_MISS",
                     result=None)
            return None
            
        except Exception as e:
            trace_all("_get_semantic_cached_prediction", "ERROR", 
                     called_from="classify_with_motivation",
                     error_type=type(e).__name__,
                     error_message=str(e))
            self.logger.warning(f"Errore cache semantica: {e}")
            return None
    
    def _cache_semantic_prediction(self, conversation_text: str, result: ClassificationResult, session_id: Optional[str] = None) -> None:
        """Salva predizione in cache semantica se abilitata"""
        if not self.enable_embeddings:
            return
        
        try:
            input_embedding = None
            # 1) Tentativo: usa embedding dalla cache centralizzata per evitare ricalcolo
            if session_id and hasattr(self, 'tenant') and self.tenant is not None:
                try:
                    import os, sys
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MongoDB'))
                    from embedding_store import EmbeddingStore  # type: ignore
                    store = EmbeddingStore()
                    fetched = store.get_embedding(self.tenant.tenant_id, session_id, consume=False)
                    if fetched:
                        input_embedding = fetched[0]
                        self.logger.debug("💾 Semantic cache: embedding preso da embedding_cache (no ricalcolo)")
                except Exception as _e:
                    self.logger.debug(f"Semantic cache store non disponibile: {_e}")
            # 2) Fallback: calcolo singolo se embedder supporta encode_single
            if input_embedding is None and self.embedder is not None and hasattr(self.embedder, 'encode_single'):
                input_embedding = self.embedder.encode_single(conversation_text)
            if input_embedding is None:
                return
            
            with self._semantic_cache_lock:
                # Aggiungi alla cache semantica
                self._semantic_cache.append((input_embedding, result, datetime.now()))
                
                # Limita dimensione cache semantica (max 500 elementi)
                if len(self._semantic_cache) > 500:
                    self._semantic_cache = self._semantic_cache[-400:]
                    
        except Exception as e:
            self.logger.warning(f"Errore salvataggio cache semantica: {e}")
    
    def _save_to_mongodb(self, conversation_text: str, result: ClassificationResult, session_id: Optional[str] = None) -> None:
        """
        Salva la classificazione in MongoDB se disponibile e client specificato
        
        Args:
            conversation_text: Testo della conversazione classificata
            result: Risultato della classificazione
            
        🔧 FIX 2025-09-05: Popola correttamente ml_result e llm_result per interfaccia
        """
        if not self.mongo_reader or not self.client_name:
            return
        
        try:
            # Genera embedding se disponibile
            embedding = None
            embedding_model = None
            
            # Lazy acquire embedder se non presente
            if self.embedder is None and hasattr(self, 'tenant') and self.tenant is not None:
                try:
                    import os, sys
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'EmbeddingEngine'))
                    from simple_embedding_manager import simple_embedding_manager
                    self.embedder = simple_embedding_manager.get_embedder_for_tenant(self.tenant)
                    self.logger.debug("Embedder acquisito lazy da SimpleEmbeddingManager per salvataggio MongoDB")
                except Exception as e:
                    self.logger.warning(f"Impossibile acquisire embedder lazy: {e}")
            
            # 1) Tentativo: recupera dallo store centralizzato per questa sessione (consume=True)
            if session_id and hasattr(self, 'tenant') and self.tenant is not None:
                try:
                    import os, sys
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MongoDB'))
                    from embedding_store import EmbeddingStore  # type: ignore
                    store = EmbeddingStore()
                    fetched = store.get_embedding(self.tenant.tenant_id, session_id, consume=True)
                    if fetched:
                        embedding, embedding_model = fetched[0], fetched[1]
                except Exception as _e:
                    self.logger.debug(f"Mongo save: embedding store non disponibile: {_e}")
            # 2) Fallback: calcolo singolo se embedder lo supporta
            if embedding is None and self.embedder is not None and hasattr(self.embedder, 'encode_single'):
                try:
                    embedding = self.embedder.encode_single(conversation_text)
                    embedding_model = getattr(self.embedder, 'model_name', None) or getattr(self.embedder, 'model_path', None) or 'unknown_embedder'
                except Exception as e:
                    self.logger.warning(f"Errore generazione embedding per MongoDB: {e}")
            
            # Usa session_id reale se fornito; altrimenti genera uno sintetico basato su testo+timestamp
            if not session_id:
                import hashlib
                session_id = hashlib.md5(f"{conversation_text[:100]}_{result.timestamp}".encode()).hexdigest()
            
            # Determina tenant_id/nome: preferisci oggetto tenant se disponibile
            if hasattr(self, 'tenant') and self.tenant is not None:
                tenant_id = getattr(self.tenant, 'tenant_id', None) or self.client_name
                tenant_name = getattr(self.tenant, 'tenant_name', self.client_name)
            else:
                # Fallback legacy basato su client_name
                import hashlib
                tenant_id = hashlib.md5(self.client_name.encode()).hexdigest()[:16]
                tenant_name = self.client_name
            
            # 🔧 FIX BUG N/A: Popola ml_result e llm_result correttamente per ensemble
            ml_result = None
            llm_result = None
            
            # Determina quale metodo è stato utilizzato dal result.method
            method = result.method.upper() if hasattr(result, 'method') and result.method else 'LLM'
            
            # 🔧 FIX CRITICO: Per ensemble classifier, popola ENTRAMBI i risultati
            if 'ENSEMBLE' in method:
                # Per ensemble, crea entrambi i risultati con stessi valori
                # Questo risolve il problema N/A nell'interfaccia di review
                ml_result = {
                    'predicted_label': result.predicted_label,
                    'confidence': result.confidence,
                    'method': f"{method}_ML_COMPONENT"
                }
                llm_result = {
                    'predicted_label': result.predicted_label,
                    'confidence': result.confidence,
                    'reasoning': result.motivation,
                    'method': f"{method}_LLM_COMPONENT"
                }
                self.logger.debug(f"🤖🧠 ENSEMBLE: Popolati entrambi ML+LLM result: {result.predicted_label} (conf: {result.confidence:.3f})")
            elif 'ML' in method:
                # Solo ML
                ml_result = {
                    'predicted_label': result.predicted_label,
                    'confidence': result.confidence,
                    'method': method
                }
                self.logger.debug(f"🤖 ML result salvato: {result.predicted_label} (conf: {result.confidence:.3f})")
            else:
                # Solo LLM (default per intelligent_clustering)
                llm_result = {
                    'predicted_label': result.predicted_label,
                    'confidence': result.confidence,
                    'reasoning': result.motivation,
                    'method': method
                }
                self.logger.debug(f"🧠 LLM result salvato: {result.predicted_label} (conf: {result.confidence:.3f})")
            
            # 🧹 PULIZIA CRITICA: Applica pulizia caratteri speciali PRIMA del salvataggio
            clean_predicted_label = clean_label_text(result.predicted_label)
            if clean_predicted_label != result.predicted_label:
                self.logger.info(f"🧹 Label pulita prima del salvataggio MongoDB: '{result.predicted_label}' → '{clean_predicted_label}'")
            
            # Aggiorna ml_result e llm_result con label pulita
            if ml_result:
                ml_result['predicted_label'] = clean_predicted_label
            if llm_result:
                llm_result['predicted_label'] = clean_predicted_label
            
            # Salva in MongoDB usando il metodo corretto
            success = self.mongo_reader.save_classification_result(
                session_id=session_id,
                client_name=self.client_name,
                ml_result=ml_result,  # 🔧 FIX: Aggiungi ml_result
                llm_result=llm_result,  # 🔧 FIX: Aggiungi llm_result
                final_decision={
                    'predicted_label': clean_predicted_label,  # 🧹 USA LABEL PULITA
                    'confidence': result.confidence,
                    'method': 'INTELLIGENT_CLUSTERING',  # 🔧 FIX 3: Sostituisci LLM_STRUCTURED con metodo clustering
                    'reasoning': result.motivation
                },
                conversation_text=conversation_text,
                needs_review=False,  # Per ora auto-classifica sempre
                classified_by='intelligent_clustering',  # 🔧 FIX: Specifica la provenienza per evitare LLM_STRUCTURED
                cluster_metadata={  # 🔧 FIX 2: Aggiungi cluster_metadata per outlier intelligenti
                    'cluster_id': -1,
                    'is_representative': False,
                    'is_outlier': True,
                    'outlier_score': 1.0 - result.confidence,  # Outlier score inversamente correlato alla confidenza
                    'method': 'intelligent_clustering_outlier',
                    'classified_individually': True,
                    'clustering_stage': 'intelligent_intent_extraction'
                },
                embedding=embedding,  # 🔧 FIX: Aggiungi embedding
                embedding_model=embedding_model  # 🔧 FIX: Aggiungi embedding_model
            )
            
            if success:
                self.logger.debug(f"💾 Classificazione salvata in MongoDB: {session_id[:8]}/{tenant_id}")
                
                # AGGIUNTA: Salva anche il tag scoperto nella tabella MySQL tags
                # Questo risolve il problema dei tag non visibili nel frontend
                try:
                    tag_description = f"Tag generato automaticamente durante classificazione (metodo: {result.method})"
                    tag_saved = self._add_new_validated_label(
                        tag_name=result.predicted_label,
                        tag_description=tag_description, 
                        confidence=result.confidence,
                        validation_method=result.method
                    )
                    
                    if tag_saved:
                        self.logger.debug(f"🏷️ Tag '{result.predicted_label}' aggiunto alla tabella tags per frontend")
                    else:
                        self.logger.debug(f"🏷️ Tag '{result.predicted_label}' già esistente nella tabella tags")
                        
                except Exception as tag_error:
                    self.logger.warning(f"⚠️ Errore salvataggio tag '{result.predicted_label}' in tabella MySQL: {tag_error}")
                    # Non blocca il flusso principale, è solo per il frontend
                    
            else:
                self.logger.warning(f"⚠️ Errore salvataggio MongoDB per tenant {tenant_id}")
                
        except Exception as e:
            self.logger.error(f"❌ Errore salvataggio MongoDB: {e}")
    
    def _validate_llm_classification_with_embedding(self, conversation_text: str, llm_result: ClassificationResult) -> ClassificationResult:
        """
        Valida e potenzialmente corregge la classificazione LLM usando embedding
        """
        if not self.enable_embedding_validation or self.embedder is None:
            return llm_result
        
        try:
            # Genera embedding del testo input
            input_embedding = self.embedder.encode_single(conversation_text)
            
            # Trova esempi della stessa categoria predetta dal LLM
            same_label_examples = [ex for ex in self.curated_examples if ex['label'] == llm_result.predicted_label]
            
            if same_label_examples:
                # Calcola similarità media con esempi della categoria predetta
                similarities = []
                for example in same_label_examples:
                    example_embedding = self.embedder.encode_single(example['text'])
                    sim = self._cosine_similarity(input_embedding, example_embedding)
                    similarities.append(sim)
                
                avg_similarity = np.mean(similarities)
                
                # Se similarità è troppo bassa e confidence LLM è moderata, cerca categoria migliore
                if avg_similarity < 0.5 and llm_result.confidence < 0.8:
                    better_category = self._find_most_similar_category(input_embedding)
                    
                    if better_category and better_category['similarity'] > avg_similarity + 0.2:
                        # Correggi la classificazione LLM
                        self.stats['llm_corrections'] += 1
                        # TODO: Aggiungere trace_all EXIT e ERROR per _validate_llm_classification_with_embedding
                        self.logger.info(f"Classificazione LLM corretta da {llm_result.predicted_label} a {better_category['label']} per bassa similarità semantica")
                        
                        return ClassificationResult(
                            predicted_label=better_category['label'],
                            confidence=better_category['similarity'] * 0.8,
                            motivation=f"Classificazione LLM corretta da {llm_result.predicted_label} a {better_category['label']} per bassa similarità semantica (orig_sim: {avg_similarity:.3f}, new_sim: {better_category['similarity']:.3f})",
                            method="LLM_CORRECTED_BY_EMBEDDING",
                            processing_time=llm_result.processing_time + 0.02,
                            timestamp=datetime.now().isoformat()
                        )
            
            return llm_result  # Mantieni classificazione LLM se valida
            
        except Exception as e:
            self.logger.warning(f"Errore validazione embedding: {e}")
            return llm_result
    
    def _find_most_similar_category(self, input_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """Trova la categoria più simile semanticamente all'input"""
        try:
            best_similarity = 0
            best_category = None
            
            # Raggruppa esempi per categoria
            categories = {}
            for example in self.curated_examples:
                label = example['label']
                if label not in categories:
                    categories[label] = []
                categories[label].append(example)
            
            # Per ogni categoria, calcola similarità media
            for label, examples in categories.items():
                similarities = []
                for example in examples:
                    example_embedding = self.embedder.encode_single(example['text'])
                    sim = self._cosine_similarity(input_embedding, example_embedding)
                    similarities.append(sim)
                
                avg_similarity = np.mean(similarities)
                
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_category = {
                        'label': label,
                        'similarity': avg_similarity
                    }
            
            return best_category
            
        except Exception as e:
            self.logger.warning(f"Errore ricerca categoria simile: {e}")
            return None
    
    def _detect_potential_new_category(self, conversation_text: str, 
                                      llm_result: ClassificationResult,
                                      original_llm_label: Optional[str] = None) -> ClassificationResult:
        """
        Rileva se il testo potrebbe richiedere una nuova categoria
        """
        if not self.enable_new_category_detection or self.embedder is None:
            return llm_result
        
        try:
            # NUOVO CHECK: Verifica se LLM ha proposto una categoria non esistente
            if (original_llm_label and 
                original_llm_label not in self.domain_labels and
                original_llm_label != 'altro' and
                llm_result.predicted_label == 'altro'):  # Normalizzazione ha forzato "altro"
                
                # LLM ha proposto una categoria nuova che è stata normalizzata!
                self.stats['new_categories_detected'] += 1
                self.logger.info(f"🆕 LLM ha proposto categoria inesistente: '{original_llm_label}' → normalizzata ad 'altro'")
                
                return ClassificationResult(
                    predicted_label="categoria_nuova_proposta",
                    confidence=min(0.8, llm_result.confidence + 0.2),
                    motivation=f"LLM ha proposto '{original_llm_label}' non presente nel dominio corrente. Potenziale nuova categoria.",
                    method="LLM_NEW_CATEGORY_PROPOSAL",
                    raw_response=llm_result.raw_response,
                    processing_time=llm_result.processing_time + 0.01,
                    timestamp=datetime.now().isoformat()
                )
            
            # CHECK ORIGINALE: Solo se LLM ha classificato come "altro" con bassa confidence
            if llm_result.predicted_label == "altro" and llm_result.confidence < 0.4:
                
                # Genera embedding
                embedding = self.embedder.encode_single(conversation_text)
                
                # Verifica se è davvero dissimilar da tutto
                max_similarity = 0
                for example in self.curated_examples:
                    example_embedding = self.embedder.encode_single(example['text'])
                    sim = self._cosine_similarity(embedding, example_embedding)
                    max_similarity = max(max_similarity, sim)
                
                # Se molto dissimilar da tutte le categorie esistenti
                if max_similarity < 0.4:
                    self.stats['new_categories_detected'] += 1
                    self.logger.info(f"Rilevato potenziale nuovo concetto: max_similarity = {max_similarity:.3f}")
                    
                    return ClassificationResult(
                        predicted_label="nuovo_concetto_potenziale",
                        confidence=0.8,
                        motivation=f"Testo semanticamente molto diverso da categorie esistenti (max_sim: {max_similarity:.3f}). Potrebbe richiedere nuova categoria.",
                        method="EMBEDDING_NEW_CATEGORY_DETECTION",
                        processing_time=llm_result.processing_time + 0.03,
                        timestamp=datetime.now().isoformat()
                    )
            
            return llm_result
            
        except Exception as e:
            self.logger.warning(f"Errore rilevazione nuova categoria: {e}")
            return llm_result
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Carica la configurazione con priorità database → config.yaml
        
        Scopo della funzione:
            Carica configurazione per il classificatore con sistema integrato:
            1. Tenta caricamento da database (se disponibile e tenant configurato)
            2. Fallback a config.yaml per configurazioni mancanti
            3. Configurazione di default come ultimo fallback
        
        Parametri di input:
            config_path: Percorso del file di configurazione YAML (opzionale)
            
        Valori di ritorno:
            Dict con la configurazione merged da database + YAML + default
        
        Tracciamento aggiornamenti:
            2025-09-09: Implementata integrazione database per batch processing
        """
        
        # 🔧 CONFIGURAZIONE DI DEFAULT BASE
        default_config = {
            'pipeline': {
                'intelligent_classifier_embedding': False,
                'embedding_validation': True,
                'semantic_fallback': True,
                'new_category_detection': True,
                'embedding_similarity_threshold': 0.85,
                'classification_batch_size': 32,  # Default batch size
                'max_parallel_calls': 200
            }
        }
        
        # 🚀 PASSO 1: CARICA CONFIG.YAML (BASE)
        yaml_config = {}
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)  
            config_path = os.path.join(project_root, 'config.yaml')
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f) or {}
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Errore caricamento config.yaml da {config_path}: {e}")
        
        # 🗄️ PASSO 2: INTEGRA CONFIGURAZIONI DATABASE (PRIORITÀ ALTA)
        # Solo se il servizio database è disponibile e il tenant è configurato
        if (hasattr(self, 'ai_config_service') and self.ai_config_service and 
            hasattr(self, 'tenant_id') and self.tenant_id):
            try:
                # Recupera configurazioni batch processing da database
                batch_config = self.ai_config_service.get_batch_processing_config(self.tenant_id)
                
                if batch_config and batch_config.get('source') == 'database':
                    # Sovrascrive configurazioni YAML con valori database (PRIORITÀ DATABASE)
                    if 'pipeline' not in yaml_config:
                        yaml_config['pipeline'] = {}
                    
                    yaml_config['pipeline']['classification_batch_size'] = batch_config.get('classification_batch_size', 32)
                    yaml_config['pipeline']['max_parallel_calls'] = batch_config.get('max_parallel_calls', 200)
                    
                    if hasattr(self, 'logger'):
                        self.logger.info(f"✅ Configurazione batch loading da database per tenant {self.tenant_id}: "
                                       f"batch_size={batch_config.get('classification_batch_size')}")
                        
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.warning(f"⚠️ Errore caricamento config database per tenant {getattr(self, 'tenant_id', 'unknown')}: {e}")
                    self.logger.info("🔄 Fallback a configurazione config.yaml")
        
        # 🔗 PASSO 3: MERGE CONFIGURAZIONI (DATABASE → YAML → DEFAULT)
        # Inizia con default, poi sovrascrive con YAML, poi con database
        merged_config = default_config.copy()
        
        # Merge YAML config
        if yaml_config:
            for key in yaml_config:
                if key == 'pipeline' and 'pipeline' in merged_config:
                    merged_config['pipeline'].update(yaml_config['pipeline'])
                else:
                    merged_config[key] = yaml_config[key]
        
        return merged_config
    
    # ==================== METODI FINE-TUNING ====================
    
    def has_finetuned_model(self) -> bool:
        """
        Verifica se questo cliente ha un modello fine-tuned disponibile
        
        Returns:
            True se ha un modello fine-tuned
        """
        if not self.finetuning_manager or not self.client_name:
            return False
        
        return self.finetuning_manager.has_finetuned_model(self.client_name)
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """
        Ottieni informazioni sul modello attualmente in uso
        
        Returns:
            Dizionario con informazioni del modello
        """
        info = {
            'base_model': self.base_model_name,
            'current_model': self.model_name,
            'is_finetuned': self.model_name != self.base_model_name,
            'client_name': self.client_name,
            'finetuning_enabled': self.enable_finetuning and FINETUNING_AVAILABLE,
            'finetuning_manager_available': self.finetuning_manager is not None
        }
        
        # Aggiungi info dettagliate se disponibili
        if self.finetuning_manager and self.client_name:
            detailed_info = self.finetuning_manager.get_model_info(self.client_name)
            info.update(detailed_info)
        
        return info
    
    def switch_to_finetuned_model(self, force_refresh: bool = False) -> bool:
        """
        Passa al modello fine-tuned per questo cliente (se disponibile)
        
        Args:
            force_refresh: Se forzare il refresh delle info del modello
            
        Returns:
            True se lo switch è riuscito
        """
        if not self.finetuning_manager or not self.client_name:
            self.logger.warning("Fine-tuning manager o client_name non disponibili")
            return False
        
        try:
            finetuned_model = self.finetuning_manager.get_client_model(self.client_name)
            
            if finetuned_model and finetuned_model != self.model_name:
                old_model = self.model_name
                self.model_name = finetuned_model
                
                self.logger.info(f"🎯 Switch da {old_model} a {finetuned_model} per {self.client_name}")
                return True
            elif not finetuned_model:
                self.logger.info(f"Nessun modello fine-tuned disponibile per {self.client_name}")
                return False
            else:
                self.logger.debug(f"Già in uso il modello fine-tuned: {finetuned_model}")
                return True
                
        except Exception as e:
            self.logger.error(f"Errore switch modello fine-tuned: {e}")
            return False
    
    def switch_to_base_model(self) -> bool:
        """
        Passa al modello base (disabilita fine-tuning temporaneamente)
        
        Returns:
            True se lo switch è riuscito
        """
        try:
            if self.model_name != self.base_model_name:
                old_model = self.model_name
                self.model_name = self.base_model_name
                
                self.logger.info(f"🔄 Switch da {old_model} al modello base {self.base_model_name}")
                return True
            else:
                self.logger.debug("Già in uso il modello base")
                return True
                
        except Exception as e:
            self.logger.error(f"Errore switch al modello base: {e}")
            return False
    
    # ==================== METODI GESTIONE TAG DATABASE ====================
    
    def _load_model_from_database(self) -> Optional[str]:
        """
        Carica il modello LLM configurato per il tenant dal database
        
        Returns:
            Nome del modello se trovato nel database, None altrimenti
            
        Autore: Valerio Bignardi
        Data: 2025-09-01 (Aggiornato: fix llm_engine field)
        """
        if not self.client_name:
            return None
            
        try:
            # Import AIConfigurationService per lettura database
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AIConfiguration'))
            from ai_configuration_service import AIConfigurationService
            
            ai_service = AIConfigurationService()
            config = ai_service.get_tenant_configuration(self.client_name, force_no_cache=True)
            
            print(f"🔍 DATABASE CONFIG DEBUG per {self.client_name}: {config}")
            
            # FIX: Cerca 'llm_engine' invece di 'llm_model.current'
            if config and 'llm_engine' in config:
                database_model = config['llm_engine']
                if database_model:
                    print(f"🎲 DATABASE: Modello LLM trovato per {self.client_name}: {database_model}")
                    return database_model
            elif config and 'llm_model' in config:
                # Fallback legacy per struttura vecchia
                database_model = config['llm_model'].get('current')
                if database_model:
                    print(f"🎲 DATABASE LEGACY: Modello trovato per {self.client_name}: {database_model}")
                    return database_model
                    
        except Exception as e:
            print(f"⚠️ Errore caricamento modello dal database per {self.client_name}: {e}")
                
        return None

    def _load_domain_labels_from_database(self):
        """
        Carica etichette esistenti dalla tabella TAG.tags
        """
        if not self.mysql_connector:
            self.logger.warning("TagDatabase connector non disponibile - skip caricamento etichette")
            return
        
        try:
            # Recupera tutti i tag dal database usando TagDatabaseConnector
            tags_data = self.mysql_connector.get_all_tags()
            
            if tags_data:
                self.domain_labels = []
                self.label_descriptions = {}
                
                for tag_info in tags_data:
                    tag_name = tag_info['tag_name']
                    tag_description = tag_info.get('tag_description', '')
                    
                    self.domain_labels.append(tag_name)
                    if tag_description:
                        self.label_descriptions[tag_name] = tag_description
                
                self.logger.info(f"✅ Caricate {len(self.domain_labels)} etichette da TAG.tags")
            else:
                self.logger.info("📋 Nessuna etichetta trovata in TAG.tags (prima esecuzione)")
            
        except Exception as e:
            self.logger.error(f"❌ Errore caricamento etichette: {e}")
            # Non solleva eccezione - fallback gestito dal chiamante
            self.mysql_connector.disconnetti()
            
        except Exception as e:
            self.logger.error(f"Errore caricamento etichette da database: {e}")
            try:
                self.mysql_connector.disconnetti()
            except:
                pass

    def _add_new_validated_label(self, tag_name: str, tag_description: str, 
                                confidence: float, validation_method: str) -> bool:
        """
        Aggiunge una nuova etichetta validata alla tabella TAG.tags
        
        Args:
            tag_name: Nome dell'etichetta
            tag_description: Descrizione semantica dell'etichetta
            confidence: Confidence della validazione
            validation_method: Metodo di validazione utilizzato
            
        Returns:
            True se inserimento riuscito
        """
        if not self.mysql_connector:
            self.logger.warning("TagDatabase connector non disponibile - skip inserimento etichetta")
            return False
        
        try:
            # 🧹 PULIZIA FINALE ETICHETTA prima del salvataggio
            clean_tag_name = clean_label_text(tag_name)
            if clean_tag_name != tag_name:
                self.logger.info(f"🧹 Etichetta pulita prima del salvataggio: '{tag_name}' → '{clean_tag_name}'")
                tag_name = clean_tag_name
            
            # Aggiungi il tag usando TagDatabaseConnector
            success = self.mysql_connector.add_tag_if_not_exists(
                tag_name=tag_name,
                tag_description=tag_description
            )
            
            if success:
                # Aggiungi anche alle strutture in memoria per uso immediato
                if tag_name not in self.domain_labels:
                    self.domain_labels.append(tag_name)
                    self.label_descriptions[tag_name] = tag_description
                
                self.logger.info(f"✅ Nuova etichetta aggiunta: '{tag_name}' (confidence: {confidence:.2f})")
                return True
            else:
                self.logger.warning(f"⚠️ Etichetta '{tag_name}' non aggiunta (potrebbe esistere già)")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Errore inserimento etichetta '{tag_name}': {e}")
            return False

    def _reload_domain_labels(self):
        """
        Ricarica le etichette dal database dopo un inserimento
        """
        self.logger.info("🔄 Ricaricamento etichette da TAG.tags...")
        old_count = len(self.domain_labels)
        self._load_domain_labels_from_database()
        new_count = len(self.domain_labels)
        
        if new_count > old_count:
            self.logger.info(f"📈 Etichette aggiornate: {old_count} → {new_count}")
        
    # ==================== METODI RISOLUZIONE SEMANTICA ====================
    
    def _semantic_label_resolution(self, proposed_label: str, 
                                  conversation_text: str, 
                                  initial_confidence: float = 1.0) -> Tuple[str, str, float]:
        """
        Risoluzione semantica dell'etichetta con sistema di fallback BERTopic intelligente
        
        Args:
            proposed_label: Etichetta proposta dall'LLM
            conversation_text: Testo originale della conversazione
            initial_confidence: Confidence iniziale della classificazione LLM
            
        Returns:
            (final_label, resolution_method, confidence_score)
        """
        trace_all(
            'ENTER', 
            '_semantic_label_resolution', 
            {
                'proposed_label': proposed_label,
                'conversation_length': len(conversation_text),
                'initial_confidence': initial_confidence,
                'domain_labels_count': len(self.domain_labels),
                'llm_confidence_threshold': self.llm_confidence_threshold,
                'auto_tag_creation': self.auto_tag_creation
            }
        )
        
        try:
            original_label = proposed_label.strip().lower().replace(' ', '_')
            
            self.logger.debug(f"🎯 Inizio risoluzione semantica: '{proposed_label}' (confidence: {initial_confidence:.3f})")
            
            # FASE 1: Verifica diretta (se è già nel dominio)
            if original_label in self.domain_labels:
                trace_all(
                    'EXIT', 
                    '_semantic_label_resolution', 
                    {
                        'result_label': original_label,
                        'resolution_method': 'DIRECT_MATCH',
                        'confidence': 1.0,
                        'phase': 'DIRECT_MATCH'
                    }
                )
                return original_label, "DIRECT_MATCH", 1.0
            
            # FASE 2: AUTO-CREAZIONE TAG (NUOVA LOGICA SEMPLIFICATA)
            # Se il tag non esiste e l'LLM è molto sicuro, crea automaticamente il tag
            if initial_confidence >= self.llm_confidence_threshold:
                self.logger.info(f"🎯 LLM confidence {initial_confidence:.3f} >= {self.llm_confidence_threshold:.3f} → Tentativo auto-creazione tag '{original_label}'")
                
                if self.auto_tag_creation:
                    # Prova a creare il nuovo tag automaticamente
                    success = self.add_new_label_to_database(original_label)
                    if success:
                        # Ricarica le etichette dal database per includere il nuovo tag
                        self._load_domain_labels_from_database()
                        self.logger.info(f"✅ Nuovo tag creato automaticamente: '{original_label}' (confidence: {initial_confidence:.3f})")
                        trace_all(
                            'EXIT', 
                            '_semantic_label_resolution', 
                            {
                                'result_label': original_label,
                                'resolution_method': 'AUTO_CREATED',
                                'confidence': initial_confidence,
                                'phase': 'AUTO_CREATION',
                                'database_labels_reloaded': True
                            }
                        )
                        return original_label, "AUTO_CREATED", initial_confidence
                    else:
                        self.logger.warning(f"⚠️ Fallimento creazione automatica tag: '{original_label}'")
                else:
                    self.logger.info(f"📝 Auto-creazione disabilitata, ma LLM suggerirebbe: '{original_label}'")

            # FASE 3: Risoluzione semantica con embeddings
            if self.embedder is not None:
                semantic_result = self._resolve_label_semantically(
                    original_label, conversation_text
                )
                if semantic_result['found_match']:
                    trace_all(
                        'EXIT', 
                        '_semantic_label_resolution', 
                        {
                            'result_label': semantic_result['matched_label'],
                            'resolution_method': 'SEMANTIC_MATCH',
                            'confidence': semantic_result['confidence'],
                            'phase': 'SEMANTIC_RESOLUTION'
                        }
                    )
                    return (
                        semantic_result['matched_label'], 
                        "SEMANTIC_MATCH", 
                        semantic_result['confidence']
                    )
            
            # FASE 3: Validazione con BERTopic per nuova categoria
            if hasattr(self, 'bertopic_provider') and self.bertopic_provider:
                bertopic_result = self._evaluate_new_category_with_bertopic(
                    original_label, conversation_text
                )
                if bertopic_result['is_valid_new_category']:
                    trace_all(
                        'EXIT', 
                        '_semantic_label_resolution', 
                        {
                            'result_label': bertopic_result['refined_label'],
                            'resolution_method': 'BERTOPIC_NEW_CATEGORY',
                            'confidence': bertopic_result['confidence'],
                            'phase': 'BERTOPIC_VALIDATION'
                        }
                    )
                    return (
                        bertopic_result['refined_label'],
                        "BERTOPIC_NEW_CATEGORY",
                        bertopic_result['confidence']
                    )
            
            # FASE 4: Fallback intelligente
            fallback_result = self._intelligent_fallback(original_label, conversation_text)
            trace_all(
                'EXIT', 
                '_semantic_label_resolution', 
                {
                    'result_label': fallback_result[0],
                    'resolution_method': fallback_result[1],
                    'confidence': fallback_result[2],
                    'phase': 'INTELLIGENT_FALLBACK'
                }
            )
            return fallback_result
            
        except Exception as e:
            trace_all(
                'ERROR', 
                '_semantic_label_resolution', 
                {
                    'error': str(e),
                    'proposed_label': proposed_label,
                    'conversation_length': len(conversation_text),
                    'initial_confidence': initial_confidence
                }
            )
            self.logger.error(f"❌ Errore in _semantic_label_resolution: {e}")
            # Fallback sicuro in caso di errore
            return 'altro', 'ERROR_FALLBACK', 0.0

    def _resolve_label_semantically(self, proposed_label: str, 
                                   conversation_text: str) -> Dict[str, Any]:
        """
        Risolve l'etichetta usando embedding similarity potenziata con descrizioni
        """
        trace_all(
            'ENTER', 
            '_resolve_label_semantically', 
            {
                'proposed_label': proposed_label,
                'conversation_length': len(conversation_text),
                'domain_labels_count': len(self.domain_labels),
                'embedder_available': self.embedder is not None,
                'label_descriptions_count': len(self.label_descriptions) if hasattr(self, 'label_descriptions') else 0
            }
        )
        
        try:
            # Embedding dell'etichetta proposta
            label_text = proposed_label.replace('_', ' ')
            proposed_embedding = self.embedder.encode_single(label_text)
            
            # Embedding del testo originale
            text_embedding = self.embedder.encode_single(conversation_text)
            
            best_match = None
            best_score = 0.0
            best_method = None
            matches_analyzed = 0
            
            # APPROCCIO POTENZIATO: Similarità con etichette + descrizioni esistenti
            for existing_label in self.domain_labels:
                if existing_label == 'altro':
                    continue
                    
                matches_analyzed += 1
                
                # Embedding dell'etichetta esistente
                existing_text = existing_label.replace('_', ' ')
                existing_embedding = self.embedder.encode_single(existing_text)
                
                # Similarità etichetta-etichetta
                label_similarity = self._cosine_similarity(proposed_embedding, existing_embedding)
                
                # Similarità testo-etichetta esistente
                text_similarity = self._cosine_similarity(text_embedding, existing_embedding)
                
                # NUOVO: Similarità con descrizione se disponibile
                description_similarity = 0.0
                if hasattr(self, 'label_descriptions') and existing_label in self.label_descriptions and self.label_descriptions[existing_label]:
                    description_text = self.label_descriptions[existing_label]
                    description_embedding = self.embedder.encode_single(description_text)
                    
                    # Similarità testo-descrizione e label-descrizione
                    text_desc_sim = self._cosine_similarity(text_embedding, description_embedding)
                    label_desc_sim = self._cosine_similarity(proposed_embedding, description_embedding)
                    description_similarity = (text_desc_sim + label_desc_sim) / 2
                
                # Score combinato POTENZIATO con descrizione
                if description_similarity > 0:
                    # Con descrizione: peso maggiore alla semantica
                    combined_score = (
                        0.4 * label_similarity +           # Similarità nomi
                        0.3 * text_similarity +            # Coerenza testo-etichetta
                        0.3 * description_similarity       # Coerenza semantica profonda
                    )
                    method_detail = f"Label:{label_similarity:.2f}, Text:{text_similarity:.2f}, Desc:{description_similarity:.2f}"
                else:
                    # Senza descrizione: logica originale
                    combined_score = 0.6 * label_similarity + 0.4 * text_similarity
                    method_detail = f"Label:{label_similarity:.2f}, Text:{text_similarity:.2f}, NoDesc"
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = existing_label
                    best_method = method_detail
            
            # APPROCCIO 2: Similarità con esempi (se score ancora basso)
            if best_score < 0.7:  # Se non trovato match diretto forte
                example_match_result = self._find_best_match_via_examples(
                    proposed_embedding, text_embedding, conversation_text
                )
                if example_match_result['score'] > best_score:
                    best_score = example_match_result['score']
                    best_match = example_match_result['label']
                    best_method = f"Example-based: {example_match_result['method']}"
            
            # Soglia di confidenza per accettare il match (più alta per descrizioni)
            confidence_threshold = 0.65
            if best_score >= confidence_threshold:
                trace_all(
                    'EXIT', 
                    '_resolve_label_semantically', 
                    {
                        'found_match': True,
                        'matched_label': best_match,
                        'confidence': best_score,
                        'method': best_method,
                        'matches_analyzed': matches_analyzed,
                        'confidence_threshold': confidence_threshold,
                        'original_label': proposed_label
                    }
                )
                return {
                    'found_match': True,
                    'matched_label': best_match,
                    'confidence': best_score,
                    'method': best_method,
                    'original_label': proposed_label
                }
            else:
                trace_all(
                    'EXIT', 
                    '_resolve_label_semantically', 
                    {
                        'found_match': False,
                        'best_candidate': best_match,
                        'best_score': best_score,
                        'method': best_method,
                        'matches_analyzed': matches_analyzed,
                        'confidence_threshold': confidence_threshold
                    }
                # TODO: Aggiungere trace_all EXIT e ERROR per _resolve_label_semantically
                )
                # 🔍 TRACING ENTER
                trace_all("_find_best_match_via_examples", "ENTER", 
                         called_from="classification_pipeline")
                
                return {
                    'found_match': False,
                    'best_candidate': best_match,
                    'best_score': best_score,
                    'method': best_method
                }
                
        except Exception as e:
            trace_all(
                'ERROR', 
                '_resolve_label_semantically', 
                {
                    'error': str(e),
                    'proposed_label': proposed_label,
                    'conversation_length': len(conversation_text),
                    'embedder_available': self.embedder is not None
                }
            )
            self.logger.warning(f"Errore risoluzione semantica: {e}")
            return {'found_match': False, 'error': str(e)}

    def _find_best_match_via_examples(self, proposed_embedding: np.ndarray,
                                     text_embedding: np.ndarray,
                                     conversation_text: str) -> Dict[str, Any]:
        """
        Trova match migliore attraverso esempi rappresentativi
        """
        best_score = 0.0
        best_label = None
        best_method = ""
        
        try:
            # Raggruppa esempi per categoria
            examples_by_category = {}
            for example in self.curated_examples:
                category = example['label']
                if category not in examples_by_category:
                    examples_by_category[category] = []
                examples_by_category[category].append(example)
            
            # Per ogni categoria, calcola similarità media
            for category, examples in examples_by_category.items():
                if category == 'altro':
                    continue
                
                # Calcola embedding degli esempi
                example_embeddings = []
                for example in examples:
                    ex_embedding = self.embedder.encode_single(example['text'])
                    example_embeddings.append(ex_embedding)
                
                if not example_embeddings:
                    continue
                
                # Similarità media del testo con esempi della categoria
                text_similarities = [
                    self._cosine_similarity(text_embedding, ex_emb) 
                    for ex_emb in example_embeddings
                ]
                avg_text_similarity = np.mean(text_similarities)
                max_text_similarity = np.max(text_similarities)
                
# TODO: Aggiungere trace_all EXIT e ERROR per _find_best_match_via_examples
                
                # 🔍 TRACING ENTER
                trace_all("_evaluate_new_category_with_bertopic", "ENTER", 
                         called_from="classification_pipeline")
                
                # Similarità etichetta con categoria (usando nome categoria)
                category_embedding = self.embedder.encode_single(category.replace('_', ' '))
                label_similarity = self._cosine_similarity(proposed_embedding, category_embedding)
                
                # Score combinato con peso maggiore sulla similarità del testo
                combined_score = (
                    0.5 * avg_text_similarity +
                    0.3 * max_text_similarity +
                    0.2 * label_similarity
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_label = category
                    best_method = f"Avg:{avg_text_similarity:.2f}, Max:{max_text_similarity:.2f}, Label:{label_similarity:.2f}"
            
            return {
                'score': best_score,
                'label': best_label,
                'method': best_method
            }
            
        except Exception as e:
            self.logger.warning(f"Errore match via esempi: {e}")
            return {'score': 0.0, 'label': None, 'method': f"Error: {e}"}

    def _evaluate_new_category_with_bertopic(self, proposed_label: str,
                                           conversation_text: str) -> Dict[str, Any]:
        """
        Valuta se l'etichetta proposta rappresenta una categoria genuinamente nuova
        usando BERTopic per l'analisi del topic
        """
        try:
            if not self.bertopic_provider or not self.bertopic_provider.model:
                return {
                    'is_valid_new_category': False,
                    'refined_label': proposed_label,
                    'reason': 'BERTopic provider non disponibile',
                    'confidence': 0.0,
                    'topic_info': None
                }
            
            # Analizza il topic del testo con BERTopic
            topics, probabilities = self.bertopic_provider.model.transform([conversation_text])
            topic_id = topics[0]
            topic_prob = probabilities[0] if len(probabilities) > 0 else [0.0]
            max_prob = max(topic_prob) if hasattr(topic_prob, '__iter__') else topic_prob
            
            # Se topic ID è -1, è considerato "noise" da BERTopic
            if topic_id == -1:
                return {
                    'is_valid_new_category': False,
                    'refined_label': proposed_label,
                    'reason': 'Testo classificato come rumore da BERTopic',
                    'confidence': 0.2,
                    'topic_info': {'topic_id': topic_id, 'probability': max_prob}
                }
            
            # Ottieni informazioni sul topic identificato
            topic_info = self.bertopic_provider.model.get_topic_info()
            if topic_info is not None and topic_id < len(topic_info):
                topic_words = self.bertopic_provider.model.get_topic(topic_id)
                
                # Crea label raffinata dalle parole chiave del topic
                if topic_words and len(topic_words) > 0:
                    # Prendi le prime 3-4 parole più rilevanti per creare una label
                    top_words = [word for word, _ in topic_words[:3] if word.isalpha()]
                    if top_words:
                        refined_label = '_'.join(top_words).lower()
                    else:
                        refined_label = proposed_label
                else:
                    refined_label = proposed_label
                
                # Verifica se questa label raffinata è semanticamente diversa da quelle esistenti
                if self.embedder and hasattr(self, 'domain_labels'):
                    refined_embedding = self.embedder.encode_single(refined_label.replace('_', ' '))
                    
                    max_similarity = 0.0
                    most_similar_label = None
                    
                    for existing_label in self.domain_labels:
                        existing_embedding = self.embedder.encode_single(existing_label.replace('_', ' '))
                        similarity = self._cosine_similarity(refined_embedding, existing_embedding)
                        
                        if similarity > max_similarity:
                            max_similarity = similarity
                            most_similar_label = existing_label
                    
                    # Se similarità < soglia configurabile, è effettivamente una nuova categoria
                    if max_similarity < self.new_tag_similarity_threshold:
                        return {
                            'is_valid_new_category': True,
                            'refined_label': refined_label,
                            'reason': f'Nuovo topic identificato, max similarità: {max_similarity:.3f}',
                            'confidence': min(0.8, max_prob + 0.1),
                            'topic_info': {
                                'topic_id': topic_id, 
                                'probability': max_prob,
                                'topic_words': topic_words[:5],
                                'most_similar_existing': most_similar_label,
                                'max_similarity': max_similarity
                            }
                        }
                    else:
                        # Topic rilevato ma troppo simile a categoria esistente
                        return {
                            'is_valid_new_category': False,
                            'refined_label': most_similar_label,
                            'reason': f'Topic simile a categoria esistente: {most_similar_label} (sim: {max_similarity:.3f})',
                            'confidence': min(0.7, max_similarity),
                            'topic_info': {
                                'topic_id': topic_id,
                                'probability': max_prob,
                                'topic_words': topic_words[:5],
                                'most_similar_existing': most_similar_label,
                                'max_similarity': max_similarity
                            }
                        # TODO: Aggiungere trace_all EXIT e ERROR per _evaluate_new_category_with_bertopic
                        }
                else:
                    # Senza embedder, accetta il topic BERTopic se ha probabilità sufficiente
                    return {
                        'is_valid_new_category': max_prob > 0.5,
                        'refined_label': refined_label,
                        'reason': f'Topic BERTopic con probabilità: {max_prob:.3f}',
                        'confidence': max_prob,
                        'topic_info': {
                            'topic_id': topic_id,
                            'probability': max_prob,
                            'topic_words': topic_words[:5]
                        }
                    }
            else:
                return {
                    'is_valid_new_category': False,
                    'refined_label': proposed_label,
                    'reason': 'Topic info non disponibile',
                    'confidence': 0.3,
                    'topic_info': {'topic_id': topic_id, 'probability': max_prob}
                }
                
        except Exception as e:
            self.logger.error(f"Errore in valutazione BERTopic: {e}")
            return {
                'is_valid_new_category': False,
                'refined_label': proposed_label,
                'reason': f"Errore BERTopic: {e}",
                'confidence': 0.0,
                'topic_info': None
            }

    def _create_new_tag_automatically(self, new_tag_name: str, 
                                     conversation_text: str, 
                                     bertopic_result: Dict[str, Any]) -> bool:
        """
        Crea automaticamente un nuovo tag nel database TAG.tags
        
        Args:
            new_tag_name: Nome del nuovo tag da creare
            conversation_text: Testo che ha generato il tag
            bertopic_result: Risultato dell'analisi BERTopic con topic_info
            
        Returns:
            bool: True se creazione riuscita, False altrimenti
        """
        try:
            if not self.mysql_connector:
                self.logger.warning("MySQL connector non disponibile per creazione automatica tag")
                return False
            
            # Prepara i dati del tag
            tag_data = {
                'tag': new_tag_name,
                'description': self._generate_tag_description(conversation_text, bertopic_result),
                'created_via': 'BERTopic_AutoDiscovery',
                'topic_info': json.dumps(bertopic_result.get('topic_info', {})),
                'creation_confidence': bertopic_result.get('confidence', 0.0),
                'sample_text': conversation_text[:500]  # Primi 500 caratteri come esempio
            }
            
            # Verifica che il tag non esista già
            existing_tags = self.mysql_connector.get_tags_list()
            if any(tag.get('tag', '').lower() == new_tag_name.lower() for tag in existing_tags):
                self.logger.info(f"Tag '{new_tag_name}' già esistente, skip creazione")
                return True  # Non è un errore, il tag esiste già
            
            # Inserisce il nuovo tag
            success = self.mysql_connector.create_new_tag(
                tag_name=new_tag_name,
                description=tag_data['description'],
                metadata={
                    'created_via': tag_data['created_via'],
                    'topic_info': tag_data['topic_info'], 
                    'creation_confidence': tag_data['creation_confidence']
                }
            )
            
            if success:
                # Aggiorna cache locale domain_labels
                if hasattr(self, 'domain_labels'):
                    self.domain_labels.add(new_tag_name)
                
                self.logger.info(f"✅ Nuovo tag creato automaticamente: '{new_tag_name}' (confidence: {tag_data['creation_confidence']:.3f})")
                return True
            else:
                self.logger.error(f"❌ Fallimento creazione tag nel database: '{new_tag_name}'")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Errore in creazione automatica tag '{new_tag_name}': {e}")
            return False
    
    def _generate_tag_description(self, conversation_text: str, bertopic_result: Dict[str, Any]) -> str:
        """
        Genera una descrizione automatica per il nuovo tag basata su BERTopic
        """
        try:
            topic_info = bertopic_result.get('topic_info', {})
            topic_words = topic_info.get('topic_words', [])
            
            # Costruisce descrizione dalle parole chiave del topic
            if topic_words:
                key_words = [word for word, _ in topic_words[:5]]
                base_description = f"Tag auto-generato da topic BERTopic. Parole chiave: {', '.join(key_words)}"
            else:
                base_description = "Tag auto-generato da analisi BERTopic del contenuto conversazione"
            
            # Aggiunge informazioni aggiuntive
            topic_id = topic_info.get('topic_id')
            confidence = bertopic_result.get('confidence', 0.0)
            
            full_description = f"{base_description}. Topic ID: {topic_id}, Confidence: {confidence:.3f}"
            
            return full_description[:200]  # Limita lunghezza descrizione
            
        except Exception as e:
            return f"Tag auto-generato da sistema BERTopic (errore descrizione: {e})"

    def _intelligent_fallback(self, original_label: str, 
                             conversation_text: str) -> Tuple[str, str, float]:
        """
        Fallback intelligente quando nessun approccio funziona
        """
        # Se abbiamo embedding, verifica se il testo è davvero un outlier
        if self.embedder:
            try:
                text_embedding = self.embedder.encode_single(conversation_text)
                
                # Calcola similarità massima con tutti gli esempi
                max_similarity = 0.0
                for example in self.curated_examples:
                    example_embedding = self.embedder.encode_single(example['text'])
                    similarity = self._cosine_similarity(text_embedding, example_embedding)
                    max_similarity = max(max_similarity, similarity)
                
                # Se molto dissimilar da tutto, potrebbe essere nuova categoria
                if max_similarity < 0.3:
                    return (
                        "categoria_emergente", 
                        "OUTLIER_DETECTION", 
                        0.6
                    )
            except:
                pass
        
        # Fallback finale
        return "altro", "FALLBACK", 0.2
    
    # ==================== METODI FINE-TUNING ====================
    
    def create_finetuned_model(self, 
                              min_confidence: float = 0.7,
                              force_retrain: bool = False) -> Dict[str, Any]:
        """
        Crea un modello fine-tuned per questo cliente
        
        Args:
            min_confidence: Confidence minima per includere esempi nel training
            force_retrain: Se forzare il re-training anche se esiste già un modello
            
        Returns:
            Dizionario con risultato del fine-tuning
        """
        if not self.finetuning_manager:
            return {
                'success': False,
                'error': 'Fine-tuning manager non disponibile'
            }
        
        if not self.client_name:
            return {
                'success': False,
                'error': 'client_name non specificato'
            }
        
        # Controlla se esiste già un modello (se non forziamo re-training)
        if not force_retrain and self.has_finetuned_model():
            return {
                'success': False,
                'error': f'Modello fine-tuned già esistente per {self.client_name}. Usa force_retrain=True per ricreare.'
            }
        
        try:
            self.logger.info(f"🚀 Avvio creazione modello fine-tuned per {self.client_name}")
            
            # Configura fine-tuning
            if FINETUNING_AVAILABLE:
                from mistral_finetuning_manager import FineTuningConfig
                config = FineTuningConfig()
                config.output_model_name = f"mistral_finetuned_{self.client_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Esegui fine-tuning
                print("\n" + "🎯"*80)
                print("🎯 INIZIO FINE-TUNING - DEBUG PROMPT")
                print("🎯"*80)
                print(f"🏢 Client: {self.client_name}")
                print(f"🤖 Output Model: {config.output_model_name}")
                print(f"⚙️  Config: {config.__dict__ if hasattr(config, '__dict__') else 'N/A'}")
                print("🎯"*80)
                
                result = self.finetuning_manager.execute_finetuning(
                    client_name=self.client_name,
                    config=config
                )
                
                print("\n" + "🎯"*80)
                print("🎯 RISULTATO FINE-TUNING")
                print("🎯"*80)
                print(f"✅ Successo: {result.success}")
                print(f"🤖 Model Name: {result.model_name}")
                print(f"📊 Training Samples: {result.training_samples}")
                print(f"📈 Validation Samples: {result.validation_samples}")
                print(f"⏱️  Training Time: {result.training_time_minutes} min")
                print(f"💾 Model Size: {result.model_size_mb} MB")
                if not result.success:
                    print(f"❌ Error: {result.error_message}")
                print("🎯"*80)
                
                if result.success:
                    # Auto-switch al nuovo modello
                    self.model_name = result.model_name
                    self.logger.info(f"✅ Fine-tuning completato: {result.model_name}")
                
                return {
                    'success': result.success,
                    'model_name': result.model_name,
                    'training_samples': result.training_samples,
                    'validation_samples': result.validation_samples,
                    'training_time_minutes': result.training_time_minutes,
                    'model_size_mb': result.model_size_mb,
                    'error': result.error_message if not result.success else None
                }
            else:
                return {
                    'success': False,
                    'error': 'Modulo fine-tuning non disponibile'
                }
            
        except Exception as e:
            error_msg = f"Errore durante fine-tuning: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

    def reload_model_configuration(self, force_from_database: bool = True) -> Dict[str, Any]:
        """
        Ricarica configurazione modello LLM dal database/configurazione
        
        FUNZIONE CRITICA: Implementa lo stesso pattern di reload dinamico
        del sistema embedding per risolvere il bug di cambio modello LLM.
        
        Args:
            force_from_database: Se True, forza lettura dal database
            
        Returns:
            Risultato del reload con dettagli
            
        Ultima modifica: 26 Agosto 2025
        """
        try:
            print(f"🔄 RELOAD MODEL CONFIG per tenant {self.client_name}")
            
            # Import AI Configuration Service per lettura database
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AIConfiguration'))
                from ai_configuration_service import AIConfigurationService
                ai_service = AIConfigurationService(use_database=True)
            except ImportError as e:
                print(f"⚠️ AIConfigurationService non disponibile: {e}")
                return {'success': False, 'error': 'AIConfigurationService non disponibile'}
            
            # Ottieni configurazione corrente dal database
            if self.client_name:
                config = ai_service.get_tenant_configuration(self.client_name, force_no_cache=force_from_database)
                
                if config and config.get('llm_model'):
                    new_model = config['llm_model'].get('current')
                    
                    if new_model and new_model != self.model_name:
                        old_model = self.model_name
                        print(f"🔄 CAMBIO MODELLO RILEVATO: {old_model} -> {new_model}")
                        
                        # Aggiorna modello corrente
                        self.model_name = new_model
                        
                        # Test nuovo modello
                        test_result = self._test_model_connection()
                        
                        if test_result.get('success', False):
                            print(f"✅ Modello {new_model} ricaricato e testato con successo")
                            
                            # Invalida cache predizioni (modello cambiato)
                            if hasattr(self, 'prediction_cache'):
                                self.prediction_cache.clear()
                                print(f"🧹 Cache predizioni invalidata dopo cambio modello")
                            
                            return {
                                'success': True,
                                'old_model': old_model,
                                'new_model': new_model,
                                'test_result': test_result,
                                'cache_cleared': True
                            }
                        else:
                            # Rollback su errore
                            print(f"❌ Test fallito per nuovo modello {new_model}, rollback a {old_model}")
                            self.model_name = old_model
                            return {
                                'success': False,
                                'error': f'Test fallito per nuovo modello {new_model}',
                                'test_result': test_result,
                                'rollback_to': old_model
                            }
                    else:
                        print(f"ℹ️  Nessun cambio modello necessario: {self.model_name}")
                        return {
                            'success': True,
                            'message': f'Modello {self.model_name} già aggiornato',
                            'model_name': self.model_name
                        }
                else:
                    return {
                        'success': False,
                        'error': 'Configurazione LLM non trovata per tenant'
                    }
            else:
                return {
                    'success': False,
                    'error': 'Nessun client_name configurato per reload'
                }
                
        except Exception as e:
            print(f"❌ Errore reload configurazione modello: {e}")
            return {
                'success': False,
                'error': f'Errore reload: {str(e)}'
            }
    
    def _test_model_connection(self) -> Dict[str, Any]:
        """
        Testa connessione al modello LLM corrente
        
        Returns:
            Risultato del test con dettagli
        """
        try:
            # Test semplice di generazione
            test_prompt = "Rispondi solo con 'OK' se questo modello funziona."
            
            response = self._make_ollama_request({
                'model': self.model_name,
                'prompt': test_prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,
                    'num_predict': 10
                }
            })
            
            if response and response.get('response'):
                return {
                    'success': True,
                    'model_name': self.model_name,
                    'test_response': response.get('response', '').strip(),
                    'response_time': response.get('total_duration', 0) / 1000000 if response.get('total_duration') else 0
                }
            else:
                return {
                    'success': False,
                    'error': 'Nessuna risposta dal modello'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Test modello fallito: {str(e)}'
            }

# Funzioni di utilità per compatibilità
def create_intelligent_classifier(embedder=None, semantic_memory=None, **kwargs) -> IntelligentClassifier:
    """
    Factory function per creare il classificatore con supporto embedding
    
    Args:
        embedder: Embedder per calcoli semantici (opzionale)
        semantic_memory: Gestore memoria semantica (opzionale)
        **kwargs: Altri parametri per IntelligentClassifier
    """
    return IntelligentClassifier(embedder=embedder, semantic_memory=semantic_memory, **kwargs)


def test_intelligent_classifier() -> None:
    """Test di base del classificatore"""
    print("🧪 Test IntelligentClassifier con supporto embedding")
    
    # Crea classificatore (senza embedding per test base)
    classifier = IntelligentClassifier()
    
    # Test disponibilità
    print(f"Disponibilità: {classifier.is_available()}")
    print(f"Embedding abilitati: {classifier.enable_embeddings}")
    print(f"Embedder disponibile: {classifier.embedder is not None}")
    
    # Test classificazione
    test_conversations = [
        "Buongiorno, non riesco ad accedere al portale con la mia password",
        "Vorrei prenotare una risonanza magnetica",
        "Quando posso ritirare i risultati delle analisi?",
        "Quanto costa la visita cardiologica privata?"
    ]
    
    for text in test_conversations:
        result = classifier.classify_with_motivation(text)
        print(f"'{text[:30]}...' → {result.predicted_label} ({result.confidence:.2f}) [{result.method}]")
    
    # Statistiche
    stats = classifier.get_statistics()
    print(f"\nStatistiche: {stats}")
    
    # Test con embedding se disponibile
    if classifier.enable_embeddings and classifier.embedder is not None:
        print("\n🧠 Test funzionalità embedding...")
        # Qui si potrebbero aggiungere test specifici per embedding
    else:
        print("\n💡 Per testare le funzionalità embedding, fornire embedder e semantic_memory")


if __name__ == "__main__":
    test_intelligent_classifier()
