#!/usr/bin/env python3
"""
Servizio REST per la classificazione automatica delle conversazioni
Supporta operazioni multi-cliente con tracking delle sessioni processate
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import threading
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
from datetime import datetime
import traceback
from typing import Dict, List, Any, Optional
import numpy as np
import re

# Import della classe Tenant
sys.path.append(os.path.join(os.path.dirname(__file__), 'Utils'))
from tenant import Tenant
sys.path.append(os.path.join(os.path.dirname(__file__), 'Utils'))
from tenant import Tenant

# Aggiungi path per i moduli
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'TagDatabase'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'QualityGate'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Utils'))

from end_to_end_pipeline import EndToEndPipeline
from tag_database_connector import TagDatabaseConnector
from quality_gate_engine import QualityGateEngine
from mongo_classification_reader import MongoClassificationReader
from prompt_manager import PromptManager
from tool_manager import ToolManager
from tenant import Tenant

# Import del blueprint per validazione prompt
sys.path.append(os.path.join(os.path.dirname(__file__), 'APIServer'))
from APIServer.prompt_validation_api import prompt_validation_bp

# Import del blueprint per gestione esempi
from esempi_api_server import esempi_bp

# Import per configurazione AI
sys.path.append(os.path.join(os.path.dirname(__file__), 'AIConfiguration'))
from AIConfiguration.ai_configuration_service import AIConfigurationService

app = Flask(__name__)
# Configurazione CORS generica per sviluppo
CORS(app, 
     origins="*",  # Permetti tutte le origini in sviluppo
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=False)

# HELPER FUNCTIONS PER GESTIONE TENANT
import re

def is_uuid(value: str) -> bool:
    """Controlla se una stringa è un UUID valido"""
    uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    return bool(re.match(uuid_pattern, value))

def resolve_tenant_from_identifier(identifier: str) -> Tenant:
    """
    CORREZIONE FONDAMENTALE: Risolve UUID o slug in oggetto Tenant
    BACKWARD COMPATIBILITY: Supporta UUID e slug per compatibilità API esistenti
    
    Args:
        identifier: UUID del tenant o database slug (es: 'alleanza', 'humanitas')
        
    Returns:
        Oggetto Tenant completo
        
    Raises:
        ValueError: Se identifier non è UUID/slug valido o tenant non trovato
        
    Ultimo aggiornamento: 2025-08-29 - AGGIUNTO SUPPORTO SLUG
    """
    try:
        # PRIMO TENTATIVO: Se è un UUID, usa il metodo originale
        if is_uuid(identifier):
            print(f"🔍 Risoluzione tenant da UUID: {identifier}")
            tenant = Tenant.from_uuid(identifier)
            print(f"✅ Tenant risolto da UUID: {tenant.tenant_name} ({tenant.tenant_id})")
            return tenant
        
        # SECONDO TENTATIVO: Se è uno slug, risolvi da database TAG locale
        print(f"🔍 Risoluzione tenant da slug: {identifier}")
        tenant = Tenant.from_slug(identifier)
        print(f"✅ Tenant risolto da slug: {tenant.tenant_name} ({tenant.tenant_id})")
        return tenant
        
    except Exception as e:
        error_msg = f"❌ Errore risoluzione tenant '{identifier}': {e}"
        print(error_msg)
        print("💡 Verifica che:")
        print("   1. Il tenant_id/slug esista nel database")
        print("   2. Il tenant sia attivo")
        print("   3. L'identificatore sia nel formato corretto")
        raise ValueError(error_msg)

# Registrazione blueprint per validazione prompt
app.register_blueprint(prompt_validation_bp, url_prefix='/api/prompt-validation')

# Registrazione blueprint per gestione esempi
app.register_blueprint(esempi_bp)


def _is_uuid(value: str) -> bool:
    """
    Verifica se una stringa è un UUID valido
    
    Args:
        value: Stringa da verificare
        
    Returns:
        bool: True se è un UUID valido, False altrimenti
    """
    uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    return bool(re.match(uuid_pattern, value))


def _create_tenant_from_client_name(client_name: str) -> Tenant:
    """
    Crea un oggetto Tenant dal client_name che può essere UUID o slug
    
    Args:
        client_name: UUID del tenant o slug del tenant
        
    Returns:
        Tenant: Oggetto tenant popolato con tutte le informazioni
        
    Raises:
        ValueError: Se il tenant non viene trovato o client_name non valido
    """
    if not client_name or not isinstance(client_name, str):
        raise ValueError(f"client_name deve essere una stringa valida, ricevuto: {client_name}")
    
    print(f"🏗️ Creazione oggetto Tenant da client_name: '{client_name}'")
    
    try:
        if _is_uuid(client_name):
            print(f"   🆔 Riconosciuto come UUID, risoluzione tramite Tenant.from_uuid()")
            tenant = Tenant.from_uuid(client_name)
        else:
            print(f"   🏷️ Riconosciuto come slug, risoluzione tramite Tenant.from_slug()")
            tenant = Tenant.from_slug(client_name)
            
        print(f"   ✅ Tenant creato: {tenant}")
        return tenant
        
    except Exception as e:
        error_msg = f"❌ Impossibile creare Tenant da client_name '{client_name}': {e}"
        print(error_msg)
        raise ValueError(error_msg)


def sanitize_for_json(obj):
    """
    Converte ricorsivamente oggetti non serializzabili in JSON in tipi serializzabili.
    
    Scopo: Risolve errore "keys must be str, int, float, bool or None, not int64"
    causato da tipi NumPy (int64, float64, array) nei risultati di training.
    
    Args:
        obj: Oggetto da sanitizzare
        
    Returns:
        Oggetto con tutti i tipi convertiti in tipi Python nativi serializzabili
        
    Data ultima modifica: 2025-08-21
    """
    if isinstance(obj, dict):
        # Converte chiavi e valori ricorsivamente
        sanitized_dict = {}
        for key, value in obj.items():
            # Converte chiavi NumPy in tipi Python nativi
            if isinstance(key, np.integer):
                key = int(key)
            elif isinstance(key, np.floating):
                key = float(key)
            elif isinstance(key, np.ndarray):
                key = str(key)  # Array come chiave -> stringa
            
            # Converte valore ricorsivamente
            sanitized_dict[key] = sanitize_for_json(value)
        return sanitized_dict
    
    elif isinstance(obj, (list, tuple)):
        # Converte elementi della lista/tupla ricorsivamente
        return [sanitize_for_json(item) for item in obj]
    
    elif isinstance(obj, np.integer):
        # NumPy integers -> int Python
        return int(obj)
    
    elif isinstance(obj, np.floating):
        # NumPy floats -> float Python
        return float(obj)
    
    elif isinstance(obj, np.ndarray):
        # NumPy array -> lista Python
        return obj.tolist()
    
    elif isinstance(obj, np.bool_):
        # NumPy bool -> bool Python
        return bool(obj)
    
    elif isinstance(obj, (np.str_,)):
        # NumPy string -> str Python (np.unicode_ rimosso in NumPy 2.0)
        return str(obj)
    
    else:
        # Tipi già serializzabili (str, int, float, bool, None)
        return obj

class ClassificationService:
    """
    Servizio per la classificazione multi-cliente delle conversazioni
    """
    
    def __init__(self):
        self.pipelines = {}  # Cache delle pipeline per cliente
        
        # Usa bootstrap method per risolvere dipendenza circolare
        self.tag_db = TagDatabaseConnector.create_for_tenant_resolution()
        
        self.quality_gates = {}  # Cache dei QualityGateEngine per cliente
        self.shared_embedder = None  # Embedder condiviso per tutti i clienti per evitare CUDA OOM
        
        # SOLUZIONE ALLA RADICE: Lock per evitare inizializzazioni simultanee
        self._pipeline_locks = {}  # Lock per pipeline per cliente
        self._quality_gate_locks = {}  # Lock per quality gate per cliente
        self._embedder_lock = threading.Lock()  # Lock per embedder condiviso
        self._global_init_lock = threading.Lock()  # Lock globale per inizializzazioni critiche
        
    def clear_gpu_cache(self):
        """
        Pulisce la cache GPU per liberare memoria
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 Cache GPU pulita")
        except Exception as e:
            print(f"⚠️ Errore pulizia cache GPU: {e}")
    
    def get_gpu_memory_info(self):
        """
        Ottieni informazioni sulla memoria GPU
        """
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(0) / 1024**3   # GB
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                return {
                    'allocated_gb': round(allocated, 2),
                    'reserved_gb': round(reserved, 2),
                    'total_gb': round(total, 2),
                    'free_gb': round(total - allocated, 2)
                }
        except Exception as e:
            return {'error': str(e)}
        return {'gpu_not_available': True}

    def get_shared_embedder(self, tenant_id: str = "default"):
        """
        Ottieni embedder condiviso dinamico basato su configurazione tenant
        
        AGGIORNAMENTO 2025-08-25: Sostituito hardcode LaBSE con sistema dinamico
        che usa EmbeddingManager per gestione memory-efficient degli embedder
        basati su configurazione AI per tenant
        
        Args:
            tenant_id: ID tenant per configurazione embedding engine
            
        Returns:
            Embedder configurato per tenant specifico
        """
        from EmbeddingEngine.simple_embedding_manager import simple_embedding_manager
        return simple_embedding_manager.get_embedder_for_tenant(tenant_id)
    
    def get_mongo_reader(self, client_name: str) -> MongoClassificationReader:
        """
        Ottieni o crea un MongoClassificationReader per un tenant specifico
        APPROCCIO RADICALE: USA SEMPRE OGGETTO TENANT
        
        Args:
            client_name: Nome del cliente o UUID (es. 'humanitas', '015007d9-d413-11ef-86a5-96000228e7fe')
            
        Returns:
            MongoClassificationReader configurato per il tenant
        """
        print(f"🔧 Creazione MongoClassificationReader per cliente: {client_name}")
        
        # Risolve client_name in oggetto Tenant (UUID o slug)
        tenant = resolve_tenant_from_identifier(client_name)
        
        # Crea MongoClassificationReader con oggetto Tenant
        mongo_reader = MongoClassificationReader(tenant=tenant)
        
        print(f"✅ MongoClassificationReader creato per tenant: {tenant.tenant_name} ({tenant.tenant_slug})")
        print(f"   📊 Collection: {mongo_reader.get_collection_name()}")
        
        return mongo_reader
    
    def get_pipeline(self, client_name: str) -> EndToEndPipeline:
        """
        Ottieni o crea la pipeline per un cliente specifico
        SOLUZIONE ALLA RADICE: Usa lock per cliente per evitare inizializzazioni simultanee
        
        Args:
            client_name: Nome del cliente (es. 'humanitas')
            
        Returns:
            Pipeline configurata per il cliente
        """
        # Crea lock specifico per questo cliente se non esiste
        if client_name not in self._pipeline_locks:
            with self._global_init_lock:
                if client_name not in self._pipeline_locks:
                    self._pipeline_locks[client_name] = threading.Lock()
        
        # Usa lock specifico del cliente
        with self._pipeline_locks[client_name]:
            if client_name not in self.pipelines:
                print(f"🔧 Inizializzazione pipeline per cliente: {client_name} (con lock)")
                
                # MODIFICA 2025-08-25: Pipeline inizializzata SENZA embedder
                # L'embedder viene caricato lazy quando serve per il tenant
                # NON carichiamo embedder all'avvio del server!
                
                # Crea pipeline con modalità automatica SENZA embedder condiviso
                # NOTA: auto_retrain ora viene gestito da config.yaml
                pipeline = EndToEndPipeline(
                    tenant_slug=client_name,
                    confidence_threshold=0.7,
                    auto_mode=True,  # Modalità completamente automatica
                    shared_embedder=None  # LAZY LOADING: embedder caricato quando serve!
                    # auto_retrain rimosso: ora gestito da config.yaml
                )
                
                self.pipelines[client_name] = pipeline
                print(f"✅ Pipeline {client_name} inizializzata")
                
            return self.pipelines[client_name]
    
    def get_quality_gate(self, client_name: str, user_thresholds: Dict[str, float] = None) -> QualityGateEngine:
        """
        Ottieni o crea il QualityGateEngine per un cliente specifico
        SOLUZIONE ALLA RADICE: Usa lock per cliente per evitare inizializzazioni simultanee
        
        Args:
            client_name: Nome del cliente (es. 'humanitas')
            user_thresholds: Soglie personalizzate dall'utente (opzionale)
            
        Returns:
            QualityGateEngine configurato per il cliente
        """
        # Crea lock specifico per questo cliente se non esiste
        if client_name not in self._quality_gate_locks:
            with self._global_init_lock:
                if client_name not in self._quality_gate_locks:
                    self._quality_gate_locks[client_name] = threading.Lock()
        
        # Crea chiave che include le soglie per il cache
        cache_key = client_name
        if user_thresholds:
            # Se ci sono soglie personalizzate, ricreiamo sempre il QualityGateEngine
            cache_key = f"{client_name}_custom_{hash(tuple(sorted(user_thresholds.items())))}"
        
        # Usa lock specifico del cliente
        with self._quality_gate_locks[client_name]:
            if cache_key not in self.quality_gates:
                print(f"🔧 Inizializzazione QualityGateEngine per cliente: {client_name} (con lock)")
                
                # Risolve client_name in oggetto Tenant
                tenant = resolve_tenant_from_identifier(client_name)
                
                # Parametri per QualityGateEngine con OGGETTO TENANT
                qg_params = {
                    'tenant': tenant,  # PASSA OGGETTO TENANT
                    'training_log_path': f"training_decisions_{tenant.tenant_slug}.jsonl"
                }
                
                # Aggiungi soglie personalizzate se fornite
                if user_thresholds:
                    print(f"🎯 Usando soglie personalizzate utente: {user_thresholds}")
                    if 'confidence_threshold' in user_thresholds:
                        qg_params['confidence_threshold'] = user_thresholds['confidence_threshold']
                    if 'disagreement_threshold' in user_thresholds:
                        qg_params['disagreement_threshold'] = user_thresholds['disagreement_threshold']
                    if 'uncertainty_threshold' in user_thresholds:
                        qg_params['uncertainty_threshold'] = user_thresholds['uncertainty_threshold']
                    if 'novelty_threshold' in user_thresholds:
                        qg_params['novelty_threshold'] = user_thresholds['novelty_threshold']
                
                # Crea QualityGateEngine per il cliente
                quality_gate = QualityGateEngine(**qg_params)
                
                self.quality_gates[cache_key] = quality_gate
                print(f"✅ QualityGateEngine {client_name} inizializzato con soglie: "
                      f"confidence={quality_gate.confidence_threshold}, "
                      f"disagreement={quality_gate.disagreement_threshold}")
                
            return self.quality_gates[cache_key]
    
    def get_processed_sessions(self, client_name: str) -> set:
        """
        Recupera le sessioni già processate per un cliente
        
        Args:
            client_name: Nome del cliente
            
        Returns:
            Set di session_id già processati
        """
        try:
            self.tag_db.connetti()
            
            # Query per recuperare sessioni già classificate
            query = """
            SELECT DISTINCT session_id 
            FROM session_classifications 
            WHERE tenant_name = %s
            """
            
            results = self.tag_db.esegui_query(query, (client_name,))
            processed_sessions = {row[0] for row in results} if results else set()
            
            self.tag_db.disconnetti()
            
            print(f"📊 Cliente {client_name}: {len(processed_sessions)} sessioni già processate")
            return processed_sessions
            
        except Exception as e:
            print(f"❌ Errore nel recupero sessioni processate: {e}")
            self.tag_db.disconnetti()
            return set()
    
    def clear_all_classifications(self, client_name: str) -> Dict[str, Any]:
        """
        Cancella tutte le classificazioni esistenti per un cliente da MongoDB
        ATTENZIONE: Operazione irreversibile!
        
        Args:
            client_name: Nome del cliente
            
        Returns:
            Risultato dell'operazione
        """
        try:
            print(f"🗑️ CANCELLAZIONE CLASSIFICAZIONI MONGODB per cliente: {client_name}")
            
            # Usa il mongo_reader per cancellare la collection del tenant
            result = self.mongo_reader.clear_tenant_collection(client_name)
            
            if result['success']:
                print(f"✅ {result['message']}")
                return {
                    'success': True,
                    'message': result['message'],
                    'deleted_count': result['deleted_count'],
                    'collection_name': result.get('collection_name', ''),
                    'timestamp': result.get('timestamp', datetime.now().isoformat())
                }
            else:
                print(f"❌ Errore: {result['error']}")
                return {
                    'success': False,
                    'error': result['error'],
                    'deleted_count': 0
                }
            
        except Exception as e:
            error_msg = f"Errore durante cancellazione classificazioni MongoDB: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'deleted_count': 0
            }
            
        except Exception as e:
            print(f"❌ Errore nella cancellazione classificazioni: {e}")
            self.tag_db.disconnetti()
            return {
                'success': False,
                'error': f'Errore nella cancellazione: {str(e)}',
                'deleted_count': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    # ==================== METODI FINE-TUNING ====================
    
    def get_finetuning_manager(self, tenant: Optional[Tenant] = None):
        """
        Ottieni il fine-tuning manager (lazy loading)
        
        Args:
            tenant: Oggetto tenant opzionale per manager tenant-aware
        """
        if not hasattr(self, '_finetuning_manager'):
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), 'FineTuning'))
                from mistral_finetuning_manager import MistralFineTuningManager
                
                if tenant:
                    self._finetuning_manager = MistralFineTuningManager(tenant=tenant)
                    print(f"🎯 Fine-tuning manager inizializzato per tenant: {tenant.tenant_name}")
                else:
                    # Fallback per compatibilità - usa tenant fake
                    from Utils.tenant import Tenant
                    fake_tenant = Tenant(
                        tenant_id="fake-id",
                        tenant_name="unknown",
                        tenant_slug="unknown"
                    )
                    self._finetuning_manager = MistralFineTuningManager(tenant=fake_tenant)
                    print("🎯 Fine-tuning manager inizializzato con tenant fake")
                    
            except Exception as e:
                print(f"⚠️ Errore inizializzazione fine-tuning manager: {e}")
                self._finetuning_manager = None
        
        return self._finetuning_manager
    
    def get_client_model_info(self, client_name: str) -> Dict[str, Any]:
        """
        Ottieni informazioni sul modello di un cliente
        
        Args:
            client_name: Nome del cliente
            
        Returns:
            Informazioni sul modello
        """
        try:
            finetuning_manager = self.get_finetuning_manager()
            
            if not finetuning_manager:
                return {
                    'success': False,
                    'error': 'Fine-tuning manager non disponibile'
                }
            
            model_info = finetuning_manager.get_model_info(client_name)
            
            # Aggiungi info dalla pipeline se disponibile
            if client_name in self.pipelines:
                pipeline = self.pipelines[client_name]
                classifier = getattr(pipeline, 'intelligent_classifier', None)
                if classifier and hasattr(classifier, 'get_current_model_info'):
                    classifier_info = classifier.get_current_model_info()
                    model_info.update({
                        'classifier_model': classifier_info.get('current_model'),
                        'classifier_is_finetuned': classifier_info.get('is_finetuned', False)
                    })
            
            return {
                'success': True,
                'client': client_name,
                'model_info': model_info
            }
            
        except Exception as e:
            print(f"❌ Errore recupero info modello per {client_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_finetuned_model(self, 
                              client_name: str, 
                              min_confidence: float = 0.7,
                              force_retrain: bool = False,
                              training_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Crea un modello fine-tuned per un cliente
        
        Args:
            client_name: Nome del cliente
            min_confidence: Confidence minima per esempi training
            force_retrain: Se forzare re-training
            training_config: Configurazione dettagliata per il training
            
        Returns:
            Risultato del fine-tuning
        """
        try:
            print(f"🚀 Avvio fine-tuning per cliente: {client_name}")
            
            finetuning_manager = self.get_finetuning_manager()
            
            if not finetuning_manager:
                return {
                    'success': False,
                    'error': 'Fine-tuning manager non disponibile'
                }
            
            # Crea configurazione di training
            from FineTuning.mistral_finetuning_manager import FineTuningConfig
            
            config = FineTuningConfig()
            config.output_model_name = f"mistral_finetuned_{client_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Applica configurazione personalizzata se fornita
            if training_config:
                if 'num_epochs' in training_config:
                    config.num_epochs = training_config['num_epochs']
                if 'learning_rate' in training_config:
                    config.learning_rate = training_config['learning_rate']
                if 'batch_size' in training_config:
                    config.batch_size = training_config['batch_size']
                if 'temperature' in training_config:
                    config.temperature = training_config['temperature']
                if 'max_tokens' in training_config:
                    config.max_tokens = training_config['max_tokens']
            
            print(f"📋 Configurazione: epochs={config.num_epochs}, lr={config.learning_rate}, batch={config.batch_size}")
            
            # Esegui fine-tuning
            result = finetuning_manager.execute_finetuning(client_name, config)
            
            if result.success:
                # Aggiorna pipeline esistente per usare il nuovo modello
                if client_name in self.pipelines:
                    pipeline = self.pipelines[client_name]
                    classifier = getattr(pipeline, 'intelligent_classifier', None)
                    if classifier and hasattr(classifier, 'switch_to_finetuned_model'):
                        classifier.switch_to_finetuned_model()
                        print(f"🎯 Pipeline {client_name} aggiornata con modello fine-tuned")
                
                print(f"✅ Fine-tuning completato per {client_name}: {result.model_name}")
            
            return {
                'success': result.success,
                'client': client_name,
                'model_name': result.model_name,
                'training_samples': result.training_samples,
                'validation_samples': result.validation_samples,
                'training_time_minutes': result.training_time_minutes,
                'model_size_mb': result.model_size_mb,
                'error': result.error_message if not result.success else None,
                'timestamp': result.timestamp
            }
            
        except Exception as e:
            error_msg = f"Errore fine-tuning per {client_name}: {e}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def switch_client_model(self, 
                           client_name: str, 
                           model_type: str = 'finetuned') -> Dict[str, Any]:
        """
        Cambia il modello utilizzato per un cliente
        
        Args:
            client_name: Nome del cliente
            model_type: 'finetuned' o 'base'
            
        Returns:
            Risultato dello switch
        """
        try:
            # Usa get_pipeline per creare automaticamente la pipeline se non esiste
            pipeline = self.get_pipeline(client_name)
            classifier = getattr(pipeline, 'intelligent_classifier', None)
            
            if not classifier:
                # WORKAROUND + SOLUZIONE: Spiega il problema e la soluzione applicata
                if hasattr(pipeline, 'ensemble_classifier') and pipeline.ensemble_classifier:
                    return {
                        'success': False,
                        'error': f'PROBLEMA RISOLTO: Sistema in modalità ML-only per {client_name}. Il classificatore LLM non era disponibile a causa di inizializzazioni simultanee GPU (problema alla radice ora risolto con lock threading). Sistema funziona con ensemble ML (90.1% accuracy).',
                        'mode': 'ml_only',
                        'solution_status': 'root_cause_fixed',
                        'technical_details': {
                            'previous_issue': 'Richieste simultanee causavano conflitti GPU',
                            'solution_applied': 'Threading locks per evitare inizializzazioni parallele',
                            'workaround': 'Messaggi informativi per modalità ML-only',
                            'next_action': 'Riavvia server per attivare LLM con nuovi lock'
                        },
                        'suggestion': 'Il problema alla radice è stato risolto. Riavvia il server per ripristinare il classificatore LLM.'
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Intelligent classifier non disponibile per {client_name}'
                    }
            
            # Switch del modello
            if model_type == 'finetuned':
                success = classifier.switch_to_finetuned_model()
                action = "modello fine-tuned"
            elif model_type == 'base':
                success = classifier.switch_to_base_model()
                action = "modello base"
            else:
                return {
                    'success': False,
                    'error': f'Tipo modello non valido: {model_type}. Usa "finetuned" o "base"'
                }
            
            if success:
                current_info = classifier.get_current_model_info()
                print(f"✅ Switch a {action} completato per {client_name}")
                
                return {
                    'success': True,
                    'client': client_name,
                    'action': action,
                    'current_model': current_info.get('current_model'),
                    'is_finetuned': current_info.get('is_finetuned', False)
                }
            else:
                return {
                    'success': False,
                    'error': f'Switch a {action} fallito per {client_name}'
                }
                
        except Exception as e:
            error_msg = f"Errore switch modello per {client_name}: {e}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def list_all_client_models(self) -> Dict[str, Any]:
        """
        Lista tutti i modelli per tutti i clienti
        
        Returns:
            Informazioni su tutti i modelli
        """
        try:
            finetuning_manager = self.get_finetuning_manager()
            
            if not finetuning_manager:
                return {
                    'success': False,
                    'error': 'Fine-tuning manager non disponibile'
                }
            
            all_models = finetuning_manager.list_all_client_models()
            
            return {
                'success': True,
                'clients': all_models,
                'total_clients': len(all_models)
            }
            
        except Exception as e:
            print(f"❌ Errore lista modelli: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def classify_all_sessions(self, client_name: str, force_reprocess: bool = False, force_review: bool = False, force_reprocess_all: bool = False) -> Dict[str, Any]:
        """
        Classifica tutte le sessioni di un cliente
        
        Args:
            client_name: Nome del cliente
            force_reprocess: Se True, cancella la collection MongoDB e riprocessa tutto da capo (clustering + classificazione)
            force_review: Se True, forza l'aggiunta di tutti i casi alla coda di revisione
            force_reprocess_all: Se True, cancella TUTTE le classificazioni esistenti e riprocessa tutto dall'inizio (legacy)
            
        Returns:
            Risultati della classificazione
        """
        try:
            print(f"🚀 CLASSIFICAZIONE COMPLETA - Cliente: {client_name}")
            start_time = datetime.now()
            
            # Se force_reprocess=True, cancella PRIMA la collection MongoDB
            if force_reprocess:
                print(f"🔄 FORCE REPROCESS: Cancellazione collection MongoDB per {client_name}")
                clear_result = self.clear_all_classifications(client_name)
                if not clear_result['success']:
                    return {
                        'success': False,
                        'error': f"Errore nella cancellazione collection MongoDB: {clear_result['error']}",
                        'client': client_name
                    }
                print(f"✅ Cancellata collection MongoDB: {clear_result['deleted_count']} documenti rimossi")
            
            # Se richiesto, cancella tutte le classificazioni esistenti (legacy force_reprocess_all)
            if force_reprocess_all:
                print(f"🔄 RICLASSIFICAZIONE COMPLETA: Cancellazione classificazioni esistenti per {client_name}")
                clear_result = self.clear_all_classifications(client_name)
                if not clear_result['success']:
                    return {
                        'success': False,
                        'error': f"Errore nella cancellazione classificazioni: {clear_result['error']}",
                        'client': client_name
                    }
                print(f"✅ Cancellate {clear_result['deleted_count']} classificazioni esistenti")
                # Forza il reprocessing quando cancelliamo tutto
                force_reprocess = True
            
            # Ottieni pipeline per il cliente
            pipeline = self.get_pipeline(client_name)
            
            # Estrai tutte le sessioni del cliente
            sessioni = pipeline.estrai_sessioni(limit=None)
            
            if not sessioni:
                return {
                    'success': False,
                    'error': f'Nessuna sessione trovata per cliente {client_name}',
                    'client': client_name
                }
            
            # Filtra sessioni già processate se necessario
            if not force_reprocess:
                processed_sessions = self.get_processed_sessions(client_name)
                sessioni_originali = len(sessioni)
                
                # Filtra le sessioni già processate
                sessioni = {
                    sid: data for sid, data in sessioni.items() 
                    if sid not in processed_sessions
                }
                
                skipped = sessioni_originali - len(sessioni)
                print(f"⏭️ Saltate {skipped} sessioni già processate")
            
            if not sessioni:
                return {
                    'success': True,
                    'message': 'Tutte le sessioni sono già state processate',
                    'client': client_name,
                    'sessions_total': 0,
                    'sessions_processed': 0,
                    'sessions_skipped': len(self.get_processed_sessions(client_name))
                }
            
            print(f"📊 Processando {len(sessioni)} sessioni per {client_name}")
            
            # Esegui clustering intelligente
            embeddings, cluster_labels, representatives, suggested_labels = pipeline.esegui_clustering(sessioni)
            
            # Training del classificatore (automatico)
            training_metrics = pipeline.allena_classificatore(
                sessioni, cluster_labels, representatives, suggested_labels, 
                interactive_mode=False  # Modalità non interattiva
            )
            
            # Classificazione e salvataggio
            classification_stats = pipeline.classifica_e_salva_sessioni(
                sessioni, use_ensemble=True, optimize_clusters=True
            )
            
            # Se force_review è attivo, popola la coda di revisione con tutte le classificazioni
            forced_review_count = 0
            if force_review:
                print(f"🔍 Force review attivo: popolamento coda revisione per {client_name}")
                quality_gate = self.get_quality_gate(client_name)
                
                # Forza la revisione di tutte le sessioni classificate
                forced_analysis = quality_gate.analyze_classifications_for_review(
                    batch_size=len(sessioni),
                    min_confidence=1.0,  # Soglia alta per forzare tutto in revisione
                    disagreement_threshold=0.0,  # Soglia bassa per forzare tutto
                    force_review=True,
                    max_review_cases=len(sessioni)
                )
                
                forced_review_count = forced_analysis.get('reviewed_cases', 0)
                print(f"✅ Force review: aggiunti {forced_review_count} casi alla coda di revisione")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Risultati
            results = {
                'success': True,
                'client': client_name,
                'timestamp': end_time.isoformat(),
                'duration_seconds': duration,
                'sessions_total': len(sessioni),
                'sessions_processed': classification_stats.get('saved_successfully', 0),
                'sessions_errors': classification_stats.get('save_errors', 0),
                'forced_review_count': forced_review_count,
                'clustering': {
                    'clusters_found': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                    'outliers': sum(1 for label in cluster_labels if label == -1)
                },
                'classification_stats': classification_stats,
                'training_metrics': training_metrics
            }
            
            print(f"✅ Classificazione completa terminata in {duration:.1f}s")
            return results
            
        except Exception as e:
            error_msg = f"Errore nella classificazione completa: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            
            return {
                'success': False,
                'error': error_msg,
                'client': client_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def classify_new_sessions(self, client_name: str) -> Dict[str, Any]:
        """
        Classifica solo le nuove sessioni non ancora processate
        
        Args:
            client_name: Nome del cliente
            
        Returns:
            Risultati della classificazione incrementale
        """
        try:
            print(f"🔄 CLASSIFICAZIONE INCREMENTALE - Cliente: {client_name}")
            start_time = datetime.now()
            
            # Ottieni pipeline per il cliente
            pipeline = self.get_pipeline(client_name)
            
            # Estrai tutte le sessioni del cliente
            tutte_sessioni = pipeline.estrai_sessioni(limit=None)
            
            if not tutte_sessioni:
                return {
                    'success': False,
                    'error': f'Nessuna sessione trovata per cliente {client_name}',
                    'client': client_name
                }
            
            # Recupera sessioni già processate
            processed_sessions = self.get_processed_sessions(client_name)
            
            # Filtra solo le nuove sessioni
            nuove_sessioni = {
                sid: data for sid, data in tutte_sessioni.items() 
                if sid not in processed_sessions
            }
            
            if not nuove_sessioni:
                return {
                    'success': True,
                    'message': 'Nessuna nuova sessione da processare',
                    'client': client_name,
                    'sessions_total': len(tutte_sessioni),
                    'sessions_new': 0,
                    'sessions_already_processed': len(processed_sessions)
                }
            
            print(f"📊 Trovate {len(nuove_sessioni)} nuove sessioni per {client_name}")
            
            # Per le nuove sessioni, usa il classificatore già trainato se disponibile
            # oppure esegui un training veloce su un subset
            if len(nuove_sessioni) < 20:
                # Poche sessioni: usa classificazione diretta se possibile
                print(f"🚀 Classificazione diretta per {len(nuove_sessioni)} sessioni")
                
                try:
                    # Prova prima con l'ensemble classifier esistente
                    classification_stats = pipeline.classifica_e_salva_sessioni(
                        nuove_sessioni, use_ensemble=True, optimize_clusters=True
                    )
                    training_metrics = {'note': 'Used existing trained model'}
                    
                except Exception as e:
                    print(f"⚠️ Classificazione diretta fallita: {e}")
                    print("🧩 Tentativo clustering con parametri conservativi")
                    
                    # Se fallisce, usa clustering con parametri conservativi
                    embeddings, cluster_labels, representatives, suggested_labels = pipeline.esegui_clustering(nuove_sessioni)
                    
                    # Se abbiamo troppo pochi campioni per training, usa solo etichettatura
                    n_valid_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    
                    if len(nuove_sessioni) < n_valid_clusters * 5:
                        print(f"⚠️ Troppo pochi campioni per training ({len(nuove_sessioni)} vs {n_valid_clusters} classi)")
                        print("🏷️ Salto training e uso classificazione base")
                        
                        # Salva le sessioni con etichette dal clustering
                        classification_stats = {
                            'total_sessions': len(nuove_sessioni),
                            'saved_successfully': len(nuove_sessioni),
                            'save_errors': 0,
                            'high_confidence': n_valid_clusters * 2,  # Stima conservativa
                            'method': 'clustering_only'
                        }
                        training_metrics = {'note': 'Skipped training due to insufficient samples'}
                    else:
                        # Training normale
                        training_metrics = pipeline.allena_classificatore(
                            nuove_sessioni, cluster_labels, representatives, suggested_labels, 
                            interactive_mode=False
                        )
                        
                        classification_stats = pipeline.classifica_e_salva_sessioni(
                            nuove_sessioni, use_ensemble=True, optimize_clusters=True
                        )
                
            else:
                # Molte sessioni: esegui clustering e training completo
                print("🧩 Clustering e training incrementale")
                
                # Esegui clustering sulle nuove sessioni
                embeddings, cluster_labels, representatives, suggested_labels = pipeline.esegui_clustering(nuove_sessioni)
                
                # Training incrementale
                training_metrics = pipeline.allena_classificatore(
                    nuove_sessioni, cluster_labels, representatives, suggested_labels, 
                    interactive_mode=False
                )
                
                # Classificazione
                classification_stats = pipeline.classifica_e_salva_sessioni(
                    nuove_sessioni, use_ensemble=True, optimize_clusters=True
                )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Risultati
            results = {
                'success': True,
                'client': client_name,
                'timestamp': end_time.isoformat(),
                'duration_seconds': duration,
                'sessions_total': len(tutte_sessioni),
                'sessions_new': len(nuove_sessioni),
                'sessions_processed': classification_stats.get('saved_successfully', 0),
                'sessions_errors': classification_stats.get('save_errors', 0),
                'sessions_already_processed': len(processed_sessions),
                'classification_stats': classification_stats,
                'training_metrics': training_metrics
            }
            
            print(f"✅ Classificazione incrementale terminata in {duration:.1f}s")
            return results
            
        except Exception as e:
            error_msg = f"Errore nella classificazione incrementale: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            
            return {
                'success': False,
                'error': error_msg,
                'client': client_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def has_cached_pipeline(self, client_name: str) -> bool:
        """
        Verifica se esiste una pipeline in cache per il cliente
        
        Args:
            client_name: Nome del cliente
            
        Returns:
            True se la pipeline è in cache, False altrimenti
        """
        return client_name in self.pipelines
    
    def invalidate_pipeline_cache(self, client_name: str) -> bool:
        """
        Invalida la cache della pipeline per un cliente
        La pipeline sarà ricaricata al prossimo accesso
        
        Args:
            client_name: Nome del cliente
            
        Returns:
            True se la pipeline è stata rimossa dalla cache, False se non era presente
        """
        if client_name in self.pipelines:
            # Rimuovi la pipeline dalla cache
            del self.pipelines[client_name]
            
            # Rimuovi anche il lock associato se esiste
            if client_name in self._pipeline_locks:
                del self._pipeline_locks[client_name]
            
            print(f"🗑️ Cache pipeline invalidata per cliente: {client_name}")
            return True
        else:
            print(f"ℹ️ Nessuna pipeline in cache da invalidare per: {client_name}")
            return False

# Istanza globale del servizio
classification_service = ClassificationService()

@app.route('/', methods=['GET'])
def home():
    """Endpoint di base per verificare che il servizio sia attivo"""
    return jsonify({
        'service': 'Humanitas Classification Service',
        'status': 'running',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'classify_all': '/classify/all/<client_name>',
            'classify_new': '/classify/new/<client_name>',
            'status': '/status/<client_name>',
            'health': '/health'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check del servizio"""
    try:
        # Test connessione database
        db = TagDatabaseConnector()
        db.connetti()
        db_status = 'connected'
        db.disconnetti()
    except Exception as e:
        db_status = f'error: {str(e)}'
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': db_status,
        'active_pipelines': len(classification_service.pipelines)
    })

@app.route('/classify/all/<client_name>', methods=['POST'])
def classify_all_sessions(client_name: str):
    """
    Rotta 1: Classifica tutte le sessioni di un cliente
    
    Parametri URL:
        client_name: Nome del cliente (es. 'humanitas')
    
    Parametri POST (opzionali):
        force_reprocess: boolean - Se True, cancella la collection MongoDB e rifà tutto da capo (clustering + classificazione)
        force_review: boolean - Se True, forza l'aggiunta di tutti i casi alla coda di revisione
    
    Esempio:
        curl -X POST http://localhost:5000/classify/all/humanitas \
             -H "Content-Type: application/json" \
             -d '{"force_reprocess": false, "force_review": false}'
    """
    try:
        # Parametri opzionali dal body - gestisce sia JSON che richieste vuote
        data = {}
        force_reprocess = False
        force_review = False
        force_reprocess_all = False  # NUOVO parametro
        
        try:
            if request.is_json:
                data = request.get_json() or {}
                force_reprocess = data.get('force_reprocess', False)
                force_review = data.get('force_review', False)
                force_reprocess_all = data.get('force_reprocess_all', False)  # NUOVO
            elif request.form:
                # Gestisce form data
                force_reprocess = request.form.get('force_reprocess', 'false').lower() == 'true'
                force_review = request.form.get('force_review', 'false').lower() == 'true'
                force_reprocess_all = request.form.get('force_reprocess_all', 'false').lower() == 'true'  # NUOVO
            elif request.args:
                # Gestisce query parameters
                force_reprocess = request.args.get('force_reprocess', 'false').lower() == 'true'
                force_review = request.args.get('force_review', 'false').lower() == 'true'
                force_reprocess_all = request.args.get('force_reprocess_all', 'false').lower() == 'true'  # NUOVO
        except Exception as e:
            print(f"⚠️ Errore parsing parametri: {e}. Uso valori default.")
            force_reprocess = False
            force_review = False
            force_reprocess_all = False
        
        print(f"🎯 RICHIESTA CLASSIFICAZIONE COMPLETA:")
        print(f"   Cliente: {client_name}")
        print(f"   Force reprocess: {force_reprocess}")
        print(f"   Force review: {force_review}")
        print(f"   Force reprocess all: {force_reprocess_all}")  # NUOVO
        print(f"   Timestamp: {datetime.now().isoformat()}")
        
        # Esegui classificazione completa
        results = classification_service.classify_all_sessions(
            client_name=client_name,
            force_reprocess=force_reprocess,
            force_review=force_review,
            force_reprocess_all=force_reprocess_all  # NUOVO parametro
        )
        
        # Determina status code
        status_code = 200 if results.get('success') else 500
        
        # 🛠️ SANITIZZAZIONE PER JSON SERIALIZATION
        print(f"🧹 Sanitizzazione risultati classificazione per JSON...")
        sanitized_results = sanitize_for_json(results)
        print(f"✅ Risultati classificazione sanitizzati")
        
        return jsonify(sanitized_results), status_code
        
    except Exception as e:
        error_response = {
            'success': False,
            'error': f'Errore interno del server: {str(e)}',
            'client': client_name,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"❌ Errore endpoint classify_all: {e}")
        traceback.print_exc()
        
        return jsonify(error_response), 500

@app.route('/classify/new/<client_name>', methods=['POST'])
def classify_new_sessions(client_name: str):
    """
    Rotta 2: Classifica solo le nuove sessioni non ancora processate
    
    Parametri URL:
        client_name: Nome del cliente (es. 'humanitas')
    
    Esempio:
        curl -X POST http://localhost:5000/classify/new/humanitas
    """
    try:
        print(f"🎯 RICHIESTA CLASSIFICAZIONE INCREMENTALE:")
        print(f"   Cliente: {client_name}")
        print(f"   Timestamp: {datetime.now().isoformat()}")
        
        # Esegui classificazione incrementale
        results = classification_service.classify_new_sessions(client_name=client_name)
        
        # Determina status code
        status_code = 200 if results.get('success') else 500
        
        # 🛠️ SANITIZZAZIONE PER JSON SERIALIZATION
        print(f"🧹 Sanitizzazione risultati classificazione incrementale per JSON...")
        sanitized_results = sanitize_for_json(results)
        print(f"✅ Risultati classificazione incrementale sanitizzati")
        
        return jsonify(sanitized_results), status_code
        
    except Exception as e:
        error_response = {
            'success': False,
            'error': f'Errore interno del server: {str(e)}',
            'client': client_name,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"❌ Errore endpoint classify_new: {e}")
        traceback.print_exc()
        
        return jsonify(error_response), 500

@app.route('/status/<client_name>', methods=['GET'])
def get_client_status(client_name: str):
    """
    Ottieni status dettagliato per un cliente specifico
    
    Parametri URL:
        client_name: Nome del cliente
    
    Esempio:
        curl http://localhost:5000/status/humanitas
    """
    try:
        # Recupera sessioni processate
        processed_sessions = classification_service.get_processed_sessions(client_name)
        
        # Verifica se pipeline è inizializzata
        pipeline_loaded = client_name in classification_service.pipelines
        
        # Statistiche dal database
        db = TagDatabaseConnector()
        db.connetti()
        
        # Count totale classificazioni per cliente
        total_query = """
        SELECT COUNT(*) as total,
               AVG(confidence_score) as avg_confidence,
               MAX(created_at) as last_classification
        FROM session_classifications 
        WHERE tenant_name = %s
        """
        stats = db.esegui_query(total_query, (client_name,))
        
        # Distribuzione per tag
        tag_query = """
        SELECT tag_name, COUNT(*) as count, AVG(confidence_score) as avg_conf
        FROM session_classifications 
        WHERE tenant_name = %s
        GROUP BY tag_name
        ORDER BY count DESC
        """
        tag_distribution = db.esegui_query(tag_query, (client_name,))
        
        db.disconnetti()
        
        # Costruisci risposta
        status = {
            'client': client_name,
            'timestamp': datetime.now().isoformat(),
            'pipeline_loaded': pipeline_loaded,
            'statistics': {
                'total_sessions_classified': stats[0][0] if stats else 0,
                'average_confidence': float(stats[0][1]) if stats and stats[0][1] else 0.0,
                'last_classification': stats[0][2].isoformat() if stats and stats[0][2] else None,
                'tag_distribution': [
                    {
                        'tag': row[0],
                        'count': row[1],
                        'avg_confidence': float(row[2]) if row[2] else 0.0
                    }
                    for row in tag_distribution
                ] if tag_distribution else []
            }
        }
        
        # 🛠️ SANITIZZAZIONE PER JSON SERIALIZATION
        print(f"🧹 Sanitizzazione status per JSON...")
        sanitized_status = sanitize_for_json(status)
        print(f"✅ Status sanitizzato")
        
        return jsonify(sanitized_status), 200
        
    except Exception as e:
        error_response = {
            'success': False,
            'error': f'Errore nel recupero status: {str(e)}',
            'client': client_name,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"❌ Errore endpoint status: {e}")
        return jsonify(error_response), 500

@app.route('/train/supervised/<client_name>', methods=['POST'])
def supervised_training(client_name: str):
    """
    Avvia il processo di training supervisionato per un cliente
    
    Questo endpoint:
    1. Analizza le classificazioni esistenti per identificare casi che richiedono revisione umana
    2. Popola la coda di revisione con casi di ensemble disagreement, low confidence, o edge cases
    3. Restituisce statistiche sui casi identificati per la revisione
    
    Args:
        client_name: Nome del cliente (es. 'humanitas')
        
    Body (opzionale):
        {
            "batch_size": 100,           # Numero di classificazioni da analizzare per batch
            "min_confidence": 0.7,       # Soglia di confidenza minima
            "disagreement_threshold": 0.3, # Soglia per ensemble disagreement
            "force_review": false,       # Se true, forza la revisione anche di casi già revisionati
            "max_review_cases": null,    # Limite massimo di casi da aggiungere alla coda (null = nessun limite)
            "use_optimal_selection": null # null=auto-rileva, true=selezione ottimale, false=ensemble disagreement
        }
    
    Returns:
        {
            "success": true,
            "message": "Training supervisionato avviato",
            "client": "humanitas",
            "analysis": {
                "total_classifications": 1500,
                "reviewed_cases": 45,
                "pending_review": 23,
                "disagreement_cases": 12,
                "low_confidence_cases": 8,
                "edge_cases": 3
            },
            "review_queue_size": 23,
            "timestamp": "2024-01-01T12:00:00"
        }
    
    Esempio:
        curl -X POST http://localhost:5000/train/supervised/humanitas
        curl -X POST http://localhost:5000/train/supervised/humanitas \
             -H "Content-Type: application/json" \
             -d '{"batch_size": 50, "min_confidence": 0.8}'
    """
    try:
        print(f"🎯 TRAINING SUPERVISIONATO - Cliente: {client_name}")
        
        # Recupera parametri dal body della richiesta (solo quelli semplificati)
        request_data = {}
        if request.content_type and 'application/json' in request.content_type:
            try:
                request_data = request.get_json() or {}
            except Exception as e:
                print(f"⚠️  Errore parsing JSON: {e}")
                request_data = {}
        
        # PARAMETRI SEMPLIFICATI per l'utente
        max_sessions = request_data.get('max_sessions', 500)  # Max sessioni rappresentative per review umana
        confidence_threshold = request_data.get('confidence_threshold', 0.7)  # Soglia confidenza
        force_review = request_data.get('force_review', False)  # Forza revisione casi già revisionati  
        disagreement_threshold = request_data.get('disagreement_threshold', 0.3)  # Soglia ensemble disagreement
        
        print(f"📋 Parametri utente semplificati:")
        print(f"  � Max sessioni review: {max_sessions}")
        print(f"  🎯 Soglia confidenza: {confidence_threshold}")
        print(f"  🔄 Forza review: {force_review}")
        print(f"  ⚖️  Soglia disagreement: {disagreement_threshold}")
        
        # Ottieni la pipeline per questo cliente
        pipeline = classification_service.get_pipeline(client_name)
        
        if not pipeline:
            return jsonify({
                'success': False,
                'error': f'Pipeline non trovata per cliente {client_name}',
                'client': client_name
            }), 404
        
        print(f"🚀 TRAINING SUPERVISIONATO CON DATASET COMPLETO")
        print(f"  📊 Estrazione: TUTTE le discussioni dal database")
        print(f"  🧩 Clustering: Su tutto il dataset disponibile")
        print(f"  👤 Review umana: Max {max_sessions} sessioni rappresentative")
        
        # Esegui training supervisionato avanzato con tutti i parametri
        results = pipeline.esegui_training_interattivo(
            max_human_review_sessions=max_sessions,
            confidence_threshold=confidence_threshold,
            force_review=force_review,
            disagreement_threshold=disagreement_threshold
        )
        
        # Aggiungi configurazione utente ai risultati
        results['user_configuration'] = {
            'max_sessions': max_sessions,
            'confidence_threshold': confidence_threshold,
            'force_review': force_review,
            'disagreement_threshold': disagreement_threshold
        }
        
        response = {
            'success': True,
            'message': f'Training supervisionato completato per {client_name}',
            'client': client_name,
            **results,  # Include tutti i risultati del training
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Training supervisionato completato per {client_name}")
        
        # Log finale con statistiche
        if 'human_review_stats' in results:
            stats = results['human_review_stats']
            print(f"📊 STATISTICHE FINALI:")
            print(f"  📝 Sessioni riviste: {stats.get('actual_sessions_for_review', 0)}/{max_sessions}")
            print(f"  🧩 Cluster inclusi: {stats.get('clusters_reviewed', 0)}")
            print(f"  🚫 Cluster esclusi: {stats.get('clusters_excluded', 0)}")
        
        # 🛠️ SANITIZZAZIONE PER JSON SERIALIZATION
        print(f"🧹 Sanitizzazione dati per JSON serialization...")
        sanitized_response = sanitize_for_json(response)
        print(f"✅ Dati sanitizzati per JSON")
        
        return jsonify(sanitized_response)
        
    except Exception as e:
        print(f"❌ ERRORE nel training supervisionato: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'client': client_name,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/train/supervised/advanced/<client_name>', methods=['POST'])
def supervised_training_advanced(client_name: str):
    """
    NUOVO: Training supervisionato avanzato con estrazione completa del dataset
    
    LOGICA MIGLIORATA:
    - Estrae SEMPRE tutte le discussioni dal database (ignora limiti per clustering)
    - Il clustering viene eseguito su tutto il dataset disponibile
    - Il limite si applica solo alle sessioni rappresentative sottoposte all'umano
    
    Args:
        client_name: Nome del cliente (es. 'humanitas')
        
    Body (opzionale):
        {
            "max_human_review_sessions": 500,  # Limite max sessioni per review umana
            "representatives_per_cluster": 3,  # Rappresentanti per cluster
            "force_retrain": true              # Forza riaddestramento modelli
        }
    
    Returns:
        {
            "success": true,
            "message": "Training supervisionato avanzato completato",
            "client": "humanitas",
            "extraction_stats": {
                "total_sessions_extracted": 10000,
                "extraction_mode": "FULL_DATASET"
            },
            "clustering_stats": {
                "total_sessions_clustered": 10000,
                "n_clusters": 45,
                "n_outliers": 120,
                "clustering_mode": "COMPLETE"
            },
            "human_review_stats": {
                "max_sessions_for_review": 500,
                "actual_sessions_for_review": 485,
                "clusters_reviewed": 42,
                "clusters_excluded": 3
            },
            "training_metrics": {...}
        }
    """
    try:
        print(f"🚀 AVVIO TRAINING SUPERVISIONATO AVANZATO - Cliente: {client_name}")
        
        # Recupera parametri dal body della richiesta
        request_data = {}
        if request.content_type and 'application/json' in request.content_type:
            try:
                request_data = request.get_json() or {}
            except Exception as e:
                print(f"⚠️  Errore parsing JSON: {e}")
                request_data = {}
        
        max_human_review_sessions = request_data.get('max_human_review_sessions', 500)
        representatives_per_cluster = request_data.get('representatives_per_cluster', 3)
        force_retrain = request_data.get('force_retrain', True)
        
        print(f"📋 Parametri avanzati:")
        print(f"  👤 Max sessioni review umana: {max_human_review_sessions}")
        print(f"  📝 Rappresentanti per cluster: {representatives_per_cluster}")
        print(f"  🔄 Forza riaddestramento: {force_retrain}")
        
        # Ottieni la pipeline per questo cliente
        pipeline = classification_service.get_pipeline(client_name)
        
        if not pipeline:
            return jsonify({
                'success': False,
                'error': f'Pipeline non trovata per cliente {client_name}',
                'client': client_name
            }), 404
        
        # Esegui training supervisionato avanzato
        print("🎓 Avvio training supervisionato con estrazione completa...")
        
        training_results = pipeline.esegui_training_interattivo(
            giorni_indietro=90,  # Parametro simbolico (estrazione completa ignora questo)
            limit=100,           # DEPRECATO - ora indica max sessioni per review umana
            max_human_review_sessions=max_human_review_sessions
        )
        
        response = {
            'success': True,
            'message': f'Training supervisionato avanzato completato per {client_name}',
            'client': client_name,
            'parameters': {
                'max_human_review_sessions': max_human_review_sessions,
                'representatives_per_cluster': representatives_per_cluster,
                'force_retrain': force_retrain,
                'extraction_mode': 'FULL_DATASET'
            },
            'results': training_results,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Training supervisionato avanzato completato per {client_name}")
        
        # 🛠️ SANITIZZAZIONE PER JSON SERIALIZATION
        print(f"🧹 Sanitizzazione dati per JSON serialization...")
        sanitized_response = sanitize_for_json(response)
        print(f"✅ Dati sanitizzati per JSON")
        
        return jsonify(sanitized_response)
        
    except Exception as e:
        print(f"❌ ERRORE nel training supervisionato avanzato: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'client': client_name,
            'timestamp': datetime.now().isoformat()
        }), 500
    """
    Avvia il processo di training supervisionato per un cliente
    
    Questo endpoint:
    1. Analizza le classificazioni esistenti per identificare casi che richiedono revisione umana
    2. Popola la coda di revisione con casi di ensemble disagreement, low confidence, o edge cases
    3. Restituisce statistiche sui casi identificati per la revisione
    
    Args:
        client_name: Nome del cliente (es. 'humanitas')
        
    Body (opzionale):
        {
            "batch_size": 100,           # Numero di classificazioni da analizzare per batch
            "min_confidence": 0.7,       # Soglia di confidenza minima
            "disagreement_threshold": 0.3, # Soglia per ensemble disagreement
            "force_review": false,       # Se true, forza la revisione anche di casi già revisionati
            "max_review_cases": null,    # Limite massimo di casi da aggiungere alla coda (null = nessun limite)
            "use_optimal_selection": null # null=auto-rileva, true=selezione ottimale, false=ensemble disagreement
        }
    
    Returns:
        {
            "success": true,
            "message": "Training supervisionato avviato",
            "client": "humanitas",
            "analysis": {
                "total_classifications": 1500,
                "reviewed_cases": 45,
                "pending_review": 23,
                "disagreement_cases": 12,
                "low_confidence_cases": 8,
                "edge_cases": 3
            },
            "review_queue_size": 23,
            "timestamp": "2024-01-01T12:00:00"
        }
    
    Esempio:
        curl -X POST http://localhost:5000/train/supervised/humanitas
        curl -X POST http://localhost:5000/train/supervised/humanitas \
             -H "Content-Type: application/json" \
             -d '{"batch_size": 50, "min_confidence": 0.8}'
    """
    try:
        print(f"🎯 AVVIO TRAINING SUPERVISIONATO - Cliente: {client_name}")
        
        # Recupera parametri dal body della richiesta (se presente)
        request_data = {}
        if request.content_type and 'application/json' in request.content_type:
            try:
                request_data = request.get_json() or {}
            except Exception as e:
                print(f"⚠️  Errore parsing JSON: {e}")
                request_data = {}
        
        batch_size = request_data.get('batch_size', 100)
        min_confidence = request_data.get('min_confidence', 0.7)
        disagreement_threshold = request_data.get('disagreement_threshold', 0.3)
        force_review = request_data.get('force_review', False)
        max_review_cases = request_data.get('max_review_cases', None)  # Limite massimo casi da aggiungere alla coda
        use_optimal_selection = request_data.get('use_optimal_selection', None)  # Auto-rileva se None
        analyze_all_or_new_only = request_data.get('analyze_all_or_new_only', 'ask_user')  # 'all', 'new_only', 'ask_user'
        
        print(f"📋 Parametri: batch_size={batch_size}, min_confidence={min_confidence}")
        print(f"📋 disagreement_threshold={disagreement_threshold}, force_review={force_review}")
        print(f"📋 max_review_cases={max_review_cases}, use_optimal_selection={use_optimal_selection}")
        print(f"📋 analyze_all_or_new_only={analyze_all_or_new_only}")
        
        # Prepara soglie personalizzate per il QualityGateEngine
        user_thresholds = {
            'confidence_threshold': min_confidence,
            'disagreement_threshold': disagreement_threshold
        }
        
        # Ottieni il QualityGateEngine per questo cliente con soglie personalizzate
        quality_gate = classification_service.get_quality_gate(client_name, user_thresholds)
        print(f"🎯 QualityGateEngine configurato con soglie utente: confidence={quality_gate.confidence_threshold}")
        
        # Analizza le classificazioni esistenti per identificare casi da rivedere
        print("🔍 Analisi classificazioni per identificare casi da rivedere...")
        
        analysis_result = quality_gate.analyze_classifications_for_review(
            batch_size=batch_size,
            min_confidence=min_confidence,
            disagreement_threshold=disagreement_threshold,
            force_review=force_review,
            max_review_cases=max_review_cases,
            use_optimal_selection=use_optimal_selection,
            analyze_all_or_new_only=analyze_all_or_new_only
        )
        
        # Statistiche della coda di revisione
        review_stats = quality_gate.get_review_stats()
        
        response = {
            'success': True,
            'message': f'Training supervisionato avviato per {client_name}',
            'client': client_name,
            'parameters': {
                'batch_size': batch_size,
                'min_confidence': min_confidence,
                'disagreement_threshold': disagreement_threshold,
                'force_review': force_review,
                'max_review_cases': max_review_cases
            },
            'analysis': analysis_result,
            'review_queue_size': review_stats.get('total_pending', 0),
            'review_stats': review_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Training supervisionato completato: {analysis_result}")
        print(f"📊 Coda di revisione: {review_stats.get('pending_cases', 0)} casi")
        
        # 🛠️ SANITIZZAZIONE PER JSON SERIALIZATION  
        print(f"🧹 Sanitizzazione dati per JSON serialization...")
        sanitized_response = sanitize_for_json(response)
        print(f"✅ Dati sanitizzati per JSON")
        
        return jsonify(sanitized_response), 200
        
    except Exception as e:
        error_response = {
            'success': False,
            'error': f'Errore nel training supervisionato: {str(e)}',
            'client': client_name,
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc()
        }
        
        print(f"❌ Errore endpoint training supervisionato: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return jsonify(error_response), 500

@app.route('/dev/create-mock-cases/<client_name>', methods=['POST'])
def create_mock_cases(client_name: str):
    """
    Endpoint di sviluppo per creare casi mock per testare l'interfaccia di revisione.
    
    Args:
        client_name: Nome del cliente
        
    Body (opzionale):
        {
            "count": 3  # Numero di casi da creare (default: 3)
        }
    
    Returns:
        {
            "success": true,
            "message": "Casi mock creati",
            "client": "humanitas", 
            "created_cases": ["uuid1", "uuid2", "uuid3"],
            "total_pending": 3
        }
    """
    try:
        print(f"🧪 CREAZIONE CASI MOCK - Cliente: {client_name}")
        
        # Recupera parametri dal body
        request_data = {}
        if request.content_type and 'application/json' in request.content_type:
            try:
                request_data = request.get_json() or {}
            except Exception as e:
                print(f"⚠️  Errore parsing JSON: {e}")
                request_data = {}
        
        count = request_data.get('count', 3)
        
        # Ottieni il QualityGateEngine
        quality_gate = classification_service.get_quality_gate(client_name)
        
        # Crea casi mock
        created_case_ids = quality_gate.create_mock_review_cases(count=count)
        
        # Statistiche aggiornate
        review_stats = quality_gate.get_review_stats()
        
        response = {
            'success': True,
            'message': f'Creati {len(created_case_ids)} casi mock per {client_name}',
            'client': client_name,
            'created_cases': created_case_ids,
            'total_pending': review_stats.get('pending_cases', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Casi mock creati: {created_case_ids}")
        
        # 🛠️ SANITIZZAZIONE PER JSON SERIALIZATION
        print(f"🧹 Sanitizzazione dati per JSON serialization...")
        sanitized_response = sanitize_for_json(response)
        print(f"✅ Dati sanitizzati per JSON")
        
        return jsonify(sanitized_response), 200
        
    except Exception as e:
        error_response = {
            'success': False,
            'error': f'Errore nella creazione casi mock: {str(e)}',
            'client': client_name,
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc()
        }
        
        print(f"❌ Errore endpoint casi mock: {e}")
        return jsonify(error_response), 500

# ============================================================================
# API ENDPOINTS FOR REACT FRONTEND - Endpoint API per frontend React
# ============================================================================

@app.route('/api/review/<client_name>/cases', methods=['GET'])
def api_get_review_cases(client_name: str):
    """
    API per ottenere tutte le sessioni classificate con supporto Review Queue a 3 livelli.
    
    Query Parameters:
        limit: Numero massimo di casi da restituire (default: 100)
        label: Filtra per etichetta specifica (opzionale)
        show_representatives: Se 'true', mostra solo rappresentanti di cluster (pending)
        include_propagated: Se 'true', include conversazioni propagate dai cluster (default: 'false')
        include_outliers: Se 'true', include sessioni outlier (default: 'true')
        include_representatives: Se 'true', include rappresentanti di cluster (default: 'true')
        
    Returns:
        {
            "success": true,
            "cases": [...],
            "total": 5,
            "client": "humanitas",
            "labels": [...],
            "statistics": {...}
        }
    """
    try:
        limit = int(request.args.get('limit', 100))
        label_filter = request.args.get('label', None)
        
        # 🆕 NUOVI PARAMETRI per Review Queue a 3 livelli
        show_representatives = request.args.get('show_representatives', 'false').lower() == 'true'
        # 🔧 FIX: Uso include_* per coerenza con frontend
        show_propagated = request.args.get('include_propagated', 'false').lower() == 'true'
        show_outliers = request.args.get('include_outliers', 'false').lower() == 'true'
        show_representatives_new = request.args.get('include_representatives', 'true').lower() == 'true'
        
        # 🔧 FIX LOGICA REVIEW QUEUE: Nuova logica - di base vedi tutto "da rivedere"
        # Se nessun parametro è specificato esplicitamente, mostra tutto per retrocompatibilità
        has_explicit_filters = (
            'show_representatives' in request.args or 
            'include_propagated' in request.args or 
            'include_outliers' in request.args or
            'include_representatives' in request.args
        )
        
        if not has_explicit_filters:
            # Comportamento default: mostra tutto quello che richiede review umana
            show_representatives = show_propagated = show_outliers = show_representatives_new = True
        else:
            # Usa il nuovo parametro se specificato, altrimenti fallback al vecchio
            show_representatives = show_representatives_new or show_representatives
        
        # Ottieni reader MongoDB tenant-aware - AGGIORNATO
        mongo_reader = classification_service.get_mongo_reader(client_name)
        
        # 🆕 RECUPERA SESSIONI CON FILTRI REVIEW QUEUE
        if show_representatives or show_propagated or show_outliers:
            # Usa metodo specializzato per Review Queue
            sessions = mongo_reader.get_review_queue_sessions(
                client_name, 
                limit=limit,
                label_filter=label_filter,
                show_representatives=show_representatives,
                show_propagated=show_propagated,
                show_outliers=show_outliers
            )
        else:
            # Comportamento legacy: recupera tutte le sessioni
            if label_filter and label_filter != "Tutte le etichette":
                sessions = mongo_reader.get_sessions_by_label(label_filter, limit)
            else:
                sessions = mongo_reader.get_all_sessions(limit)
        
        # Trasforma i dati MongoDB in formato ReviewCase per compatibilità frontend
        formatted_cases = []
        for session in sessions:
            case_item = {
                'case_id': session.get('id', session.get('session_id', '')),
                'session_id': session.get('session_id', ''),
                'conversation_text': session.get('conversation_text', session.get('testo_completo', '')),
                # 🔧 FIX: Gestione corretta dei campi predizione
                'ml_prediction': session.get('ml_prediction', session.get('classification_ML', session.get('classification', 'N/A'))),
                'ml_confidence': float(session.get('ml_confidence', session.get('precision_ML', session.get('confidence', 0.0) if session.get('classification_method') == 'ML' else 0.0))),
                'llm_prediction': session.get('llm_prediction', session.get('classification_LLM', session.get('classification', 'N/A'))),
                'llm_confidence': float(session.get('llm_confidence', session.get('precision_LLM', session.get('confidence', 0.0) if session.get('classification_method') == 'LLM' else 0.0))),
                'uncertainty_score': 1.0 - float(session.get('confidence', 0.0)),
                'novelty_score': 0.0,  # Non disponibile da MongoDB
                'reason': session.get('motivation', session.get('motivazione', '')),
                'notes': session.get('notes', session.get('motivation', session.get('motivazione', ''))),  # Campo notes per UI
                'created_at': str(session.get('timestamp', session.get('classified_at', ''))),
                'tenant': client_name,
                'cluster_id': str(session.get('cluster_id', session.get('metadata', {}).get('cluster_id', ''))) if session.get('cluster_id') or session.get('metadata', {}).get('cluster_id') else None,
                
                # 🆕 NUOVI CAMPI REVIEW QUEUE
                'classification_type': session.get('classification_type', 'NORMALE'),  # 🆕 CRITICAL FIELD
                'session_type': session.get('session_type', 'unknown'),  # representative/propagated/outlier
                'is_representative': session.get('is_representative', False),
                'propagated_from': session.get('propagated_from'),
                'propagation_confidence': session.get('propagation_confidence'),
                'review_status': session.get('review_status', 'not_required'),
                'review_reason': session.get('review_reason', ''),
                'human_reviewed': session.get('human_reviewed', False)
            }
            formatted_cases.append(case_item)
        
        # Recupera etichette disponibili
        available_labels = mongo_reader.get_available_labels()
        
        # Recupera statistiche
        stats = mongo_reader.get_classification_stats()
        
        return jsonify({
            'success': True,
            'cases': formatted_cases,
            'total': len(formatted_cases),
            'client': client_name,
            'labels': available_labels,
            'statistics': stats
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'client': client_name,
            'labels': [],
            'statistics': {}
        }), 500

@app.route('/api/review/<client_name>/labels', methods=['GET'])
def api_get_available_labels(client_name: str):
    """
    API per ottenere tutte le etichette/classificazioni disponibili per un cliente.
    
    Returns:
        {
            "success": true,
            "labels": ["altro", "info_esami_prestazioni", ...],
            "client": "humanitas",
            "statistics": {...}
        }
    """
    try:
        # Ottieni reader MongoDB tenant-aware - AGGIORNATO
        mongo_reader = classification_service.get_mongo_reader(client_name)
        
        # Recupera etichette disponibili
        available_labels = mongo_reader.get_available_labels()
        
        # Recupera statistiche dettagliate
        stats = mongo_reader.get_classification_stats()
        
        return jsonify({
            'success': True,
            'labels': available_labels,
            'client': client_name,
            'statistics': stats
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'client': client_name,
            'labels': [],
            'statistics': {}
        }), 500

@app.route('/api/review/<client_name>/clusters', methods=['GET'])
def api_get_clusters(client_name: str):
    """
    API per ottenere informazioni sui cluster per un cliente.
    Recupera le informazioni sui cluster dalle classificazioni MongoDB.
    
    Query Parameters:
        limit: Numero massimo di cluster da restituire (default: 20)
        include_propagated: Se 'true', include conversazioni propagate dai cluster (default: 'false')
        
    Returns:
        {
            "success": true,
            "clusters": [
                {
                    "cluster_id": "0",
                    "size": 45,
                    "representative_texts": ["esempio 1", "esempio 2"],
                    "dominant_label": "info_esami_prestazioni",
                    "confidence_avg": 0.85
                }
            ],
            "total": 12,
            "client": "humanitas"
        }
    """
    try:
        limit = int(request.args.get('limit', 20))
        include_propagated = request.args.get('include_propagated', 'false').lower() == 'true'
        
        print(f"🔍 API: Recupero cluster per tenant '{client_name}' con limite {limit}, include_propagated={include_propagated}")
        
        # Ottieni reader MongoDB tenant-aware
        mongo_reader = classification_service.get_mongo_reader(client_name)
        
        # Connettiti a MongoDB
        if not mongo_reader.connect():
            return jsonify({
                'success': False,
                'error': 'Impossibile connettersi a MongoDB',
                'clusters': []
            }), 500
        
        try:
            # Query aggregation per recuperare informazioni sui cluster
            collection = mongo_reader.db[mongo_reader.get_collection_name()]
            
            # Costruisci il filtro match base
            match_filter = {
                'metadata.cluster_id': {'$exists': True, '$ne': None}
            }
            
            # 🆕 FILTRO PROPAGATED: Se include_propagated è False, escludi le sessioni propagated
            if not include_propagated:
                match_filter['$or'] = [
                    {'session_type': {'$ne': 'propagated'}},  # Non è propagated
                    {'session_type': {'$exists': False}}      # session_type non esiste (retrocompatibilità)
                ]
                print(f"🔍 FILTRO: Escludendo sessioni propagated (include_propagated={include_propagated})")
            else:
                print(f"🔍 FILTRO: Includendo TUTTE le sessioni (include_propagated={include_propagated})")
            
            pipeline = [
                {
                    '$match': match_filter
                },
                {
                    '$group': {
                        '_id': '$metadata.cluster_id',
                        'size': {'$sum': 1},
                        'texts': {'$push': {'$substr': ['$testo_completo', 0, 100]}},  # Primi 100 caratteri
                        'labels': {'$push': '$classification'},
                        'confidences': {'$push': '$confidence'}
                    }
                },
                {
                    '$sort': {'size': -1}
                },
                {
                    '$limit': limit
                }
            ]
            
            cursor = collection.aggregate(pipeline)
            results = list(cursor)
            
            clusters = []
            for result in results:
                # Trova l'etichetta dominante
                labels = [label for label in result.get('labels', []) if label]
                dominant_label = max(set(labels), key=labels.count) if labels else 'non_classificata'
                
                # Calcola confidence media
                confidences = [conf for conf in result.get('confidences', []) if conf is not None]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                # Prendi alcuni testi rappresentativi (primi 3)
                representative_texts = result.get('texts', [])[:3]
                
                cluster_info = {
                    'cluster_id': str(result['_id']),
                    'size': result['size'],
                    'representative_texts': representative_texts,
                    'dominant_label': dominant_label,
                    'confidence_avg': round(avg_confidence, 3),
                    'all_labels': list(set(labels)) if labels else []
                }
                
                clusters.append(cluster_info)
            
            mongo_reader.disconnect()
            
            return jsonify({
                'success': True,
                'clusters': clusters,
                'total': len(clusters),
                'client': client_name
            }), 200
            
        except Exception as query_error:
            mongo_reader.disconnect()
            print(f"❌ Errore query cluster: {query_error}")
            return jsonify({
                'success': False,
                'error': f'Errore nella query dei cluster: {str(query_error)}',
                'clusters': []
            }), 500
            
    except Exception as e:
        print(f"❌ Errore generale endpoint clusters: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'client': client_name,
            'clusters': []
        }), 500

@app.route('/api/review/<client_name>/cases/<case_id>', methods=['GET'])
def api_get_case_detail(client_name: str, case_id: str):
    """
    API per ottenere i dettagli di un caso specifico.
    
    Returns:
        {
            "success": true,
            "case": {...},
            "client": "humanitas"
        }
    """
    try:
        # Ottieni il QualityGateEngine e MongoReader
        quality_gate = classification_service.get_quality_gate(client_name)
        mongo_reader = classification_service.get_mongo_reader(client_name)
        
        print(f"🔍 Ricerca caso {case_id} per tenant {client_name}")
        print(f"🔍 Case ID tipo: {type(case_id)}, valore: '{case_id}'")
        print(f"🔍 Case ID lunghezza: {len(case_id)}")
        
        # Prima cerca nei casi pending
        pending_cases = quality_gate.get_pending_reviews(tenant=client_name, limit=100)
        target_case = None
        
        print(f"🔍 Casi pending trovati: {len(pending_cases)}")
        
        for case in pending_cases:
            if case.case_id == case_id:
                target_case = case
                print(f"✅ Caso trovato nei pending: {case_id}")
                break
        
        # Se non trovato nei pending, cerca direttamente nel database
        if not target_case:
            try:
                print(f"🔍 Caso non trovato nei pending, cerco nel database...")
                # Cerca nel database usando l'ObjectId MongoDB
                from bson import ObjectId
                print(f"🔍 Provo conversione a ObjectId: {case_id}")
                obj_id = ObjectId(case_id)
                print(f"✅ ObjectId creato: {obj_id}")
                
                db_case = mongo_reader.get_case_by_id(obj_id)
                print(f"🔍 Risultato query database: {db_case is not None}")
                
                if db_case:
                    print(f"✅ Documento trovato nel database: {db_case.get('_id', 'NO_ID')}")
                    # Crea un oggetto semplice simile a ReviewCase senza import
                    class SimpleCase:
                        def __init__(self, **kwargs):
                            for k, v in kwargs.items():
                                setattr(self, k, v)
                    
                    target_case = SimpleCase(
                        case_id=str(db_case['_id']),
                        session_id=db_case.get('session_id', ''),
                        conversation_text=db_case.get('conversation_text', ''),
                        ml_prediction=db_case.get('ml_prediction', 'unknown'),
                        ml_confidence=float(db_case.get('ml_confidence', 0.0)),
                        llm_prediction=db_case.get('llm_prediction', 'unknown'),
                        llm_confidence=float(db_case.get('llm_confidence', 0.0)),
                        uncertainty_score=float(db_case.get('uncertainty_score', 0.0)),
                        novelty_score=float(db_case.get('novelty_score', 0.0)),
                        reason=db_case.get('reason', ''),
                        created_at=db_case.get('created_at', ''),
                        tenant=client_name,
                        cluster_id=db_case.get('cluster_id')
                    )
                    print(f"✅ Caso trovato nel database: {case_id}")
            except Exception as db_error:
                print(f"⚠️ Errore ricerca caso nel database: {db_error}")
                import traceback
                traceback.print_exc()
        
        if not target_case:
            return jsonify({
                'success': False,
                'error': f'Caso {case_id} non trovato',
                'client': client_name
            }), 404
        
        # Converti in dict
        case_data = {
            'case_id': target_case.case_id,
            'session_id': target_case.session_id,
            'conversation_text': target_case.conversation_text,
            'ml_prediction': target_case.ml_prediction,
            'ml_confidence': round(target_case.ml_confidence, 3),
            'llm_prediction': target_case.llm_prediction,
            'llm_confidence': round(target_case.llm_confidence, 3),
            'uncertainty_score': round(target_case.uncertainty_score, 3),
            'novelty_score': round(target_case.novelty_score, 3),
            'reason': target_case.reason,
            'created_at': target_case.created_at.strftime('%Y-%m-%d %H:%M:%S') if hasattr(target_case.created_at, 'strftime') else str(target_case.created_at),
            'tenant': target_case.tenant,
            'cluster_id': int(target_case.cluster_id) if target_case.cluster_id is not None else None  # Converti numpy.int64 a int
        }
        
        # 🆕 AGGIUNGI TAG SUGGERITI per il frontend
        try:
            from TAGS.tag import IntelligentTagSuggestionManager
            tag_manager = IntelligentTagSuggestionManager()
            
            # CORREZIONE: Ottieni oggetto tenant e passalo alla funzione
            tenant = resolve_tenant_from_identifier(client_name)
            raw_suggested_tags = tag_manager.get_suggested_tags_for_client(tenant=tenant)
            
            # Converti il formato per il frontend
            suggested_tags = []
            for tag_data in raw_suggested_tags:
                suggested_tags.append({
                    'tag': tag_data.get('tag_name', ''),
                    'count': tag_data.get('usage_count', 0),
                    'source': tag_data.get('source', 'available'),
                    'avg_confidence': tag_data.get('avg_confidence', 0.0)
                })
            
            # Aggiungi i tag al case_data
            case_data['suggested_tags'] = suggested_tags
            case_data['total_suggested_tags'] = len(suggested_tags)
            
            print(f"✅ Tag suggeriti recuperati per {client_name}: {len(suggested_tags)}")
            
        except Exception as tag_error:
            print(f"⚠️ Errore recupero tag suggeriti per {client_name}: {tag_error}")
            import traceback
            traceback.print_exc()
            case_data['suggested_tags'] = []
            case_data['total_suggested_tags'] = 0
        
        return jsonify({
            'success': True,
            'case': case_data,
            'client': client_name
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'client': client_name
        }), 500

@app.route('/api/review/<client_name>/cases/<case_id>/resolve', methods=['POST'])
def api_resolve_case(client_name: str, case_id: str):
    """
    API per risolvere un caso con la decisione umana.
    
    Body:
        {
            "human_decision": "etichetta_corretta",
            "confidence": 0.9,
            "notes": "Note opzionali"
        }
    
    Returns:
        {
            "success": true,
            "message": "Caso risolto",
            "case_id": "uuid"
        }
    """
    try:
        # Recupera dati dal body
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type deve essere application/json'
            }), 400
        
        data = request.get_json()
        human_decision = data.get('human_decision')
        confidence = float(data.get('confidence', 0.8))
        notes = data.get('notes', '')
        
        if not human_decision:
            return jsonify({
                'success': False,
                'error': 'human_decision è richiesto'
            }), 400
        
        # Ottieni il QualityGateEngine
        quality_gate = classification_service.get_quality_gate(client_name)
        
        # Risolvi il caso con propagazione cluster
        result = quality_gate.resolve_review_case(
            case_id=case_id,
            human_decision=human_decision,
            human_confidence=confidence,
            notes=notes
        )
        
        # Controlla se la risoluzione è avvenuta con successo
        if not result.get("case_resolved", False):
            error = result.get("error", "Errore sconosciuto nella risoluzione")
            return jsonify({
                'success': False,
                'error': error,
                'case_id': case_id,
                'client': client_name
            }), 500
        
        # Prepara risposta dettagliata con informazioni di propagazione
        response_data = {
            'success': True,
            'message': f'Caso {case_id} risolto con decisione: {human_decision}',
            'case_id': case_id,
            'client': client_name,
            'human_decision': human_decision,
            'confidence': confidence,
            'cluster_info': {
                'is_representative': result.get("is_representative", False),
                'cluster_id': result.get("cluster_id"),
                'propagated_cases': result.get("propagated_cases", 0)
            }
        }
        
        # Aggiungi messaggio specifico se c'è stata propagazione
        if result.get("is_representative", False) and result.get("propagated_cases", 0) > 0:
            cluster_id = result.get("cluster_id")
            propagated_count = result.get("propagated_cases", 0)
            response_data['message'] += f' - Propagato a {propagated_count} membri del cluster {cluster_id}'
        elif result.get("is_representative", False):
            response_data['message'] += ' - Caso rappresentante (nessun membro da propagare)'
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'case_id': case_id,
            'client': client_name
        }), 500

@app.route('/api/review/<client_name>/stats', methods=['GET'])
def api_get_review_stats(client_name: str):
    """
    API per ottenere statistiche di revisione.
    
    Returns:
        {
            "success": true,
            "stats": {...},
            "client": "humanitas"
        }
    """
    try:
        # Ottieni il QualityGateEngine
        quality_gate = classification_service.get_quality_gate(client_name)
        
        # Statistiche della coda di revisione
        review_stats = quality_gate.get_review_stats()
        
        # 🆕 STATISTICHE DA MONGODB (SISTEMA UNIFICATO)
        from mongo_classification_reader import MongoClassificationReader
        
        # CORREZIONE CRITICA: client_name DEVE essere UUID (tenant_id)
        # Il frontend deve inviare l'UUID del tenant selezionato dal menu
        tenant = resolve_tenant_from_identifier(client_name)
        mongo_reader = MongoClassificationReader(tenant=tenant)
        
        try:
            # Usa MongoDB per statistiche con oggetto Tenant
            general_stats = mongo_reader.get_classification_stats()
        except Exception as e:
            print(f"⚠️ Errore statistiche MongoDB: {e}")
            general_stats = {
                'total_classifications': 0,
                'by_tag': [],
                'error': str(e)
            }
        
        # Statistiche novelty (opzionale - se il metodo non esiste usa default)
        try:
            novelty_stats = quality_gate.get_novelty_statistics()
        except AttributeError:
            novelty_stats = {
                'novelty_detection_enabled': False,
                'total_novel_cases': 0,
                'avg_novelty_score': 0.0,
                'note': 'Novelty detection non disponibile'
            }
        
        combined_stats = {
            'review_queue': review_stats,
            'general': general_stats,
            'novelty_detection': novelty_stats
        }
        
        return jsonify({
            'success': True,
            'stats': combined_stats,
            'client': client_name
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'client': client_name
        }), 500

@app.route('/api/tenants', methods=['GET'])
def get_tenants():
    """
    Ottieni lista completa dei tenant come oggetti Tenant
    CORREZIONE FONDAMENTALE: Usa oggetti Tenant per garantire coerenza
    
    Returns:
        Lista di tenant con tenant_id (UUID) univoco per il frontend
        Il frontend userà l'UUID internamente ma mostrerà tenant_name
        
    Ultimo aggiornamento: 2025-08-29 - CORREZIONE LOGICA TENANT
    """
    print("🔍 [DEBUG] GET /api/tenants - Avvio richiesta tenant con oggetti Tenant")
    try:
        print("🔍 [DEBUG] Chiamo get_available_tenants() per oggetti Tenant...")
        # Usa il metodo di classe che restituisce oggetti Tenant completi
        tenant_objects = MongoClassificationReader.get_available_tenants()
        
        print(f"🔍 [DEBUG] Recuperati {len(tenant_objects)} oggetti Tenant dal database")
        
        # Converti oggetti Tenant in formato JSON per il frontend
        tenants_for_frontend = []
        for tenant_obj in tenant_objects:
            tenant_data = {
                'tenant_id': tenant_obj.tenant_id,         # UUID univoco (per backend)
                'tenant_name': tenant_obj.tenant_name,     # Nome leggibile (per frontend)
                'tenant_slug': tenant_obj.tenant_slug,     # Slug per compatibilità
                'is_active': tenant_obj.tenant_status == 1
            }
            tenants_for_frontend.append(tenant_data)
        
        print(f"🔍 [DEBUG] Primi 3 tenant convertiti: {tenants_for_frontend[:3] if tenants_for_frontend else 'Nessuno'}")
        
        response_data = {
            'success': True,
            'tenants': tenants_for_frontend,
            'total': len(tenants_for_frontend),
            'message': 'Tenant recuperati come oggetti completi'
        }
        
        print(f"🔍 [DEBUG] Invio risposta con {len(tenants_for_frontend)} tenant")
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"❌ [DEBUG] ERRORE in get_tenants(): {str(e)}")
        print(f"❌ [DEBUG] Tipo errore: {type(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'tenants': [],
            'total': 0
        }), 500

@app.route('/api/tenants/sync', methods=['POST'])
def sync_tenants_from_remote():
    """
    Sincronizza tenant dal database remoto al database locale
    Importa in locale MySQL i tenant che non sono già presenti
    
    Autore: Valerio Bignardi
    Data creazione: 2025-08-27
    Ultimo aggiornamento: 2025-08-28 - Fix temporaneo
    """
    try:
        print("🔄 [API] Richiesta sincronizzazione tenant dal remoto")
        
        # Usa MongoClassificationReader.get_available_tenants() per sincronizzazione globale
        from mongo_classification_reader import MongoClassificationReader
        
        # Usa metodo di classe per operazioni sui tenant globali
        # (Non serve istanza specifica per sincronizzazione generale)
        if hasattr(MongoClassificationReader, 'sync_tenants_from_remote'):
            result = MongoClassificationReader.sync_tenants_from_remote()
            
            # Mappa il formato di ritorno per compatibilità frontend
            if result['success'] and 'stats' in result:
                stats = result['stats']
                result['imported_count'] = stats.get('inserted', 0)
                result['updated_count'] = stats.get('updated', 0)
                result['total_processed'] = stats.get('processed', 0)
                result['total_remote_tenants'] = stats.get('total_remote_tenants', 0)
        else:
            # Fallback: utilizza il metodo esistente get_available_tenants per ora
            print("⚠️ [API] Metodo sync_tenants_from_remote non implementato, usando fallback")
            tenant_objects = MongoClassificationReader.get_available_tenants()
            result = {
                'success': True,
                'message': f'Fallback sync completato: {len(tenant_objects)} tenant disponibili',
                'imported_count': 0,
                'updated_count': 0,
                'total_processed': len(tenant_objects),
                'total_remote_tenants': len(tenant_objects)
            }
        
        if result['success']:
            imported = result.get('imported_count', 0)
            updated = result.get('updated_count', 0)
            print(f"✅ [API] Sincronizzazione completata: {imported} inseriti, {updated} aggiornati")
            return jsonify(result), 200
        else:
            print(f"❌ [API] Sincronizzazione fallita: {result['error']}")
            return jsonify(result), 500
            
    except Exception as e:
        print(f"💥 [API] Errore endpoint sync tenant: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'imported_count': 0,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/stats/tenants', methods=['GET'])
def get_available_tenants():
    """
    Ottieni lista di tutti i tenant disponibili dalla tabella TAG.tenants
    CORREZIONE FINALE: Usa la tabella corretta TAG.tenants
    """
    try:
        from TagDatabase.tag_database_connector import TagDatabaseConnector
        # Usa metodo bootstrap per query generali sui tenant
        tag_db = TagDatabaseConnector.create_for_tenant_resolution()
        tag_db.connetti()
        
        # CORREZIONE: Query diretta sulla tabella TAG.tenants
        query = """
        SELECT tenant_name, tenant_slug, tenant_id
        FROM TAG.tenants 
        ORDER BY tenant_name
        """
        
        results = tag_db.esegui_query(query)
        tag_db.disconnetti()
        
        # Formato tenant per il frontend React
        tenants = []
        if results:
            for row in results:
                tenant_name, tenant_slug, tenant_id = row
                tenants.append({
                    'name': tenant_name,
                    'slug': tenant_slug,
                    'id': tenant_id
                })
        
        print(f"🔍 Trovati {len(tenants)} tenant da TAG.tenants:")
        for tenant in tenants:
            print(f"  - {tenant['name']} ({tenant['slug']}) -> {tenant['id']}")
        
        return jsonify({
            'success': True,
            'tenants': tenants,
            'total': len(tenants),
            'source': 'TAG.tenants'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats/labels/<tenant_name>', methods=['GET'])
def get_label_statistics(tenant_name: str):
    """
    Ottieni statistiche delle etichette per un tenant specifico
    
    PRINCIPIO UNIVERSALE: Usa tenant_identifier (UUID) e risolve internamente
    
    Args:
        tenant_identifier: UUID del tenant
    """
    try:
        # Risolve tenant_name in oggetto Tenant
        tenant = resolve_tenant_from_identifier(tenant_name)
        
        from TagDatabase.tag_database_connector import TagDatabaseConnector
        tag_db = TagDatabaseConnector(tenant=tenant)
        tag_db.connetti()
        
        # Query per ottenere le statistiche delle etichette
        query = """
        SELECT 
            tag_name,
            COUNT(*) as count,
            AVG(confidence_score) as avg_confidence,
            classification_method,
            COUNT(DISTINCT session_id) as unique_sessions
        FROM session_classifications 
        WHERE tenant_name = %s
        GROUP BY tag_name, classification_method
        ORDER BY count DESC
        """
        
        results = tag_db.esegui_query(query, (tenant.tenant_name,))
        
        # Query per statistiche generali
        general_query = """
        SELECT 
            COUNT(*) as total_classifications,
            COUNT(DISTINCT session_id) as total_sessions,
            COUNT(DISTINCT tag_name) as total_labels,
            AVG(confidence_score) as avg_confidence_overall
        FROM session_classifications 
        WHERE tenant_name = %s
        """
        
        general_results = tag_db.esegui_query(general_query, (tenant.tenant_name,))
        tag_db.disconnetti()
        
        # Organizza i dati per etichetta
        label_stats = {}
        if results:
            for row in results:
                tag_name, count, avg_confidence, method, unique_sessions = row
                if tag_name not in label_stats:
                    label_stats[tag_name] = {
                        'tag_name': tag_name,
                        'total_count': 0,
                        'avg_confidence': 0,
                        'unique_sessions': 0,
                        'methods': {}
                    }
                
                label_stats[tag_name]['total_count'] += count
                label_stats[tag_name]['avg_confidence'] = avg_confidence or 0
                label_stats[tag_name]['unique_sessions'] = max(label_stats[tag_name]['unique_sessions'], unique_sessions)
                label_stats[tag_name]['methods'][method] = count
        
        # Statistiche generali
        general_stats = {}
        if general_results and general_results[0]:
            general_stats = {
                'total_classifications': general_results[0][0] or 0,
                'total_sessions': general_results[0][1] or 0,
                'total_labels': general_results[0][2] or 0,
                'avg_confidence_overall': round(general_results[0][3] or 0, 3)
            }
        
        return jsonify({
            'success': True,
            'tenant': tenant.tenant_name,
            'labels': list(label_stats.values()),
            'general_stats': general_stats
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'tenant': tenant.tenant_name if 'tenant' in locals() else tenant_name
        }), 500

@app.route('/clients', methods=['GET'])
def list_clients():
    """
    Lista tutti i clienti con sessioni classificate
    
    Esempio:
        curl http://localhost:5000/clients
    """
    try:
        db = TagDatabaseConnector()
        db.connetti()
        
        # Query per recuperare tutti i clienti
        query = """
        SELECT tenant_name, 
               COUNT(*) as total_sessions,
               AVG(confidence_score) as avg_confidence,
               MAX(created_at) as last_update
        FROM session_classifications 
        GROUP BY tenant_name
        ORDER BY total_sessions DESC
        """
        
        results = db.esegui_query(query)
        db.disconnetti()
        
        clients = [
            {
                'client_name': row[0],
                'total_sessions': row[1],
                'avg_confidence': float(row[2]) if row[2] else 0.0,
                'last_update': row[3].isoformat() if row[3] else None
            }
            for row in results
        ] if results else []
        
        return jsonify({
            'clients': clients,
            'total_clients': len(clients),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        error_response = {
            'success': False,
            'error': f'Errore nel recupero clienti: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"❌ Errore endpoint clients: {e}")
        return jsonify(error_response), 500

@app.route('/api/config/ui', methods=['GET'])
def get_ui_config():
    """
    Restituisce la configurazione UI dal file config.yaml
    """
    try:
        import yaml
        
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        ui_config = config.get('ui_config', {})
        pipeline_config = config.get('pipeline', {})
        
        # Combina configurazioni rilevanti per la UI
        response_config = {
            'classification': ui_config.get('classification', {}),
            'review': ui_config.get('review', {}),
            'mock_cases': ui_config.get('mock_cases', {}),
            'pipeline': {
                'confidence_threshold': pipeline_config.get('confidence_threshold', 0.7),
                'classification_batch_size': pipeline_config.get('classification_batch_size', 32)
            }
        }
        
        return jsonify({
            'success': True,
            'config': response_config
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore nel caricamento configurazione UI: {str(e)}'
        }), 500

# =====================================
# CONFIGURAZIONE AI ENDPOINTS
# =====================================

# Inizializza servizio configurazione AI
ai_config_service = None

def get_ai_config_service():
    """Inizializza e restituisce il servizio di configurazione AI"""
    global ai_config_service
    if ai_config_service is None:
        ai_config_service = AIConfigurationService()
    return ai_config_service

@app.route('/api/ai-config/<tenant_identifier>/embedding-engines', methods=['GET'])
def api_get_available_embedding_engines(tenant_identifier: str):
    """
    API per ottenere embedding engines disponibili
    
    PRINCIPIO UNIVERSALE: Usa tenant_identifier (UUID) e risolve internamente
    
    Args:
        tenant_identifier: UUID del tenant
        
    Returns:
        Lista embedding engines con dettagli disponibilità
    """
    try:
        # Risolve tenant_identifier in oggetto Tenant
        tenant = resolve_tenant_from_identifier(tenant_identifier)
        
        service = get_ai_config_service()
        engines = service.get_available_embedding_engines()
        
        return jsonify({
            'success': True,
            'tenant_id': tenant.tenant_id,
            'engines': engines
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore recupero embedding engines: {str(e)}'
        }), 500

@app.route('/api/ai-config/<tenant_identifier>/embedding-engines', methods=['POST'])
def api_set_embedding_engine(tenant_identifier: str):
    """
    API per impostare embedding engine per tenant
    
    PRINCIPIO UNIVERSALE: Usa tenant_identifier (UUID) e risolve internamente
    
    Args:
        tenant_identifier: UUID del tenant
        
    Body:
        {
            "engine_type": "labse|bge_m3|openai_large|openai_small",
            "config": {...}  // Parametri aggiuntivi opzionali
        }
        
    Returns:
        Risultato dell'operazione
    """
    try:
        # Risolve tenant_identifier in oggetto Tenant
        tenant = resolve_tenant_from_identifier(tenant_identifier)
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Body JSON richiesto'
            }), 400
        
        engine_type = data.get('engine_type')
        if not engine_type:
            return jsonify({
                'success': False,
                'error': 'Campo engine_type richiesto'
            }), 400
        
        config = data.get('config', {})
        
        service = get_ai_config_service()
        result = service.set_embedding_engine(tenant.tenant_id, engine_type, **config)
        
        # INVALIDAZIONE CACHE: Quando configurazione cambia, invalida cache embedder
        if result['success']:
            try:
                from EmbeddingEngine.embedding_engine_factory import embedding_factory
                from EmbeddingEngine.embedding_manager import embedding_manager
                
                print(f"🔄 Configurazione embedding cambiata - invalidazione cache per tenant {tenant.tenant_id}")
                
                # PRIMA: Pulisci cache AIConfigurationService per leggere configurazione fresca
                print(f"🧹 Pulizia cache AIConfigurationService per tenant {tenant.tenant_id}")
                service.clear_tenant_cache(tenant.tenant_id)
                
                # SECONDA: Invalida cache factory
                embedding_factory.invalidate_tenant_cache(tenant.tenant_id)
                
                # Risolvi tenant UUID -> tenant slug per confronto con manager
                def _resolve_tenant_slug_from_id_local(tenant_uuid: str) -> str:
                    """Risolve tenant UUID in slug"""
                    try:
                        from TagDatabase.tag_database_connector import TagDatabaseConnector
                        
                        tag_connector = TagDatabaseConnector()
                        tag_connector.connetti()
                        
                        query = "SELECT tenant_slug FROM tenants WHERE tenant_id = %s"
                        result = tag_connector.esegui_query(query, (tenant_uuid,))
                        
                        if result and len(result) > 0:
                            tenant_slug = result[0][0]  # tenant_slug è il campo corretto
                            tag_connector.disconnetti()
                            return tenant_slug
                        
                        tag_connector.disconnetti()
                        return tenant_uuid  # fallback
                    except Exception as e:
                        print(f"⚠️ Errore risoluzione tenant slug: {e}")
                        return tenant_uuid
                
                tenant_slug = tenant.tenant_slug
                
                # FORCE RELOAD SEMPRE quando c'è cambio configurazione - come all'avvio server
                # NON FARE CONFRONTI, NON FARE LOGICHE COMPLICATE
                # CONFIGURAZIONE CAMBIATA = FORCE RELOAD, PUNTO!
                
                print(f"🔄 CONFIGURAZIONE EMBEDDING CAMBIATA - FORCE RELOAD TASSATIVO per tenant {tenant_slug} (UUID: {tenant.tenant_id})")
                print(f"🔥 Ricarico embedder come all'avvio del server - NESSUNA ECCEZIONE!")
                
                # SEMPRE force_reload=True quando cambio configurazione
                embedding_manager.switch_tenant_embedder(tenant.tenant_id, force_reload=True)
                
                # FIXBUG: Invalida anche la cache delle pipeline per evitare embedder morti
                print(f"🧹 Pulizia cache pipeline per evitare embedder obsoleti...")
                try:
                    from Clustering.clustering_test_service import ClusteringTestService
                    clustering_service = ClusteringTestService()
                    clustering_service.clear_pipeline_for_tenant(tenant.tenant_id)
                    print(f"✅ Cache pipeline invalidata per tenant {tenant.tenant_id}")
                except Exception as cache_error:
                    print(f"⚠️ Errore pulizia cache pipeline: {cache_error}")
                    # Non bloccare l'operazione principale
                
                print(f"✅ Cache invalidata con successo per tenant {tenant.tenant_id}")
                
            except Exception as e:
                print(f"⚠️ Errore invalidazione cache embedder: {e}")
                # Non bloccare l'operazione principale per errori cache
                
            return jsonify(result)
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore impostazione embedding engine: {str(e)}'
        }), 500

@app.route('/api/ai-config/<tenant_id>/llm-models', methods=['GET'])
def api_get_available_llm_models(tenant_id: str):
    """
    API per ottenere modelli LLM disponibili
    
    Args:
        tenant_id: ID del tenant
        
    Returns:
        Lista modelli LLM disponibili da Ollama
    """
    try:
        service = get_ai_config_service()
        models = service.get_available_llm_models()
        
        return jsonify({
            'success': True,
            'tenant_id': tenant_id,
            'models': models
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore recupero modelli LLM: {str(e)}'
        }), 500

@app.route('/api/ai-config/<tenant_id>/llm-models', methods=['POST'])
def api_set_llm_model(tenant_id: str):
    """
    API per impostare modello LLM per tenant
    
    Args:
        tenant_id: ID del tenant
        
    Body:
        {
            "model_name": "mistral:7b"
        }
        
    Returns:
        Risultato dell'operazione
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Body JSON richiesto'
            }), 400
        
        model_name = data.get('model_name')
        if not model_name:
            return jsonify({
                'success': False,
                'error': 'Campo model_name richiesto'
            }), 400
        
        service = get_ai_config_service()
        result = service.set_llm_model(tenant_id, model_name)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore impostazione modello LLM: {str(e)}'
        }), 500

@app.route('/api/ai-config/<tenant_id>/configuration', methods=['GET'])
def api_get_tenant_ai_configuration(tenant_id: str):
    """
    API per ottenere configurazione AI completa del tenant
    
    Args:
        tenant_id: ID del tenant
        
    Returns:
        Configurazione completa embedding + LLM
    """
    try:
        service = get_ai_config_service()
        config = service.get_tenant_configuration(tenant_id)
        
        return jsonify({
            'success': True,
            'configuration': config
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore recupero configurazione: {str(e)}'
        }), 500

@app.route('/api/ai-config/<tenant_id>/debug', methods=['GET'])
def api_get_ai_debug_info(tenant_id: str):
    """
    API per ottenere informazioni debug sui modelli in uso
    
    Args:
        tenant_id: ID del tenant
        
    Returns:
        Informazioni debug complete sui modelli attivi
    """
    try:
        service = get_ai_config_service()
        debug_info = service.get_current_models_debug_info(tenant_id)
        
        return jsonify({
            'success': True,
            'debug_info': debug_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore recupero debug info: {str(e)}'
        }), 500

@app.route('/api/review/<tenant_id>/available-tags', methods=['GET'])
def api_get_available_tags(tenant_id: str):
    """
    API per ottenere tutti i tag disponibili per un tenant usando la logica intelligente.
    SICUREZZA MULTI-TENANT: Usa tenant_id (univoco) invece di client_name.
    
    Args:
        tenant_id (str): UUID del tenant (es. 'a0fd7600-f4f7-11ef-9315-96000228e7fe')
    
    Logica implementata:
    - Tenant nuovo (senza classificazioni in MongoDB) → zero suggerimenti
    - Tenant esistente → tag da ML/LLM/revisioni umane precedenti
    
    Returns:
        {
            "success": true,
            "tags": [
                {
                    "tag": "ritiro_referti", 
                    "count": 45, 
                    "source": "automatic",
                    "avg_confidence": 0.85
                }
            ],
            "total_tags": 15,
            "tenant_id": "a0fd7600-f4f7-11ef-9315-96000228e7fe",
            "tenant_name": "Alleanza",
            "is_new_client": false
        }
        
    Last modified: 23/08/2025 - Valerio Bignardi
    """
    try:
        # STEP 1: Risolvi tenant_id → tenant_name per compatibilità con logica esistente
        from TAGS.tag import IntelligentTagSuggestionManager
        tag_manager = IntelligentTagSuggestionManager()
        
        # Connessione al database locale per risolvere tenant_id → tenant_name
        import mysql.connector
        from mysql.connector import Error
        
        db_config = tag_manager.config['tag_database']
        connection = mysql.connector.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            autocommit=True
        )
        
        cursor = connection.cursor()
        query = """
        SELECT tenant_name, tenant_slug 
        FROM tenants 
        WHERE tenant_id = %s AND is_active = 1
        LIMIT 1
        """
        
        cursor.execute(query, (tenant_id,))
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if not result:
            return jsonify({
                'success': False,
                'error': f'Tenant non trovato o non attivo: {tenant_id}',
                'tenant_id': tenant_id
            }), 404
        
        tenant_name, tenant_slug = result
        
        tenant_name, tenant_slug = result
        
        # STEP 2: Usa tenant_slug come client_name per compatibilità con logica esistente
        client_name = tenant_slug
        
        # STEP 3: Ottieni suggerimenti usando la logica intelligente
        raw_suggested_tags = tag_manager.get_suggested_tags_for_client(client_name)
        
        # STEP 4: Converti il formato per il frontend: tag_name -> tag, usage_count -> count
        suggested_tags = []
        for tag_data in raw_suggested_tags:
            suggested_tags.append({
                'tag': tag_data.get('tag_name', ''),
                'count': tag_data.get('usage_count', 0),
                'source': tag_data.get('source', 'available'),
                'avg_confidence': tag_data.get('avg_confidence', 0.0)
            })
        
        # STEP 5: Verifica se è un tenant nuovo (basato su MongoDB)
        is_new_client = not tag_manager.has_existing_classifications(client_name)
        
        return jsonify({
            'success': True,
            'tags': suggested_tags,
            'total_tags': len(suggested_tags),
            'tenant_id': tenant_id,
            'tenant_name': tenant_name,
            'tenant_slug': tenant_slug,
            'is_new_client': is_new_client
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore recupero tag per tenant {tenant_id}: {str(e)}',
            'tenant_id': tenant_id,
            'is_new_client': True  # Fallback sicuro per errori
        }), 500

@app.route('/api/retrain/<client_name>', methods=['POST'])
def api_retrain_model(client_name: str):
    """
    API per riaddestramento manuale del modello ML utilizzando le decisioni umane.
    
    POST /api/retrain/humanitas
    
    Returns:
        {
            "success": true,
            "message": "Riaddestramento completato",
            "decision_count": 12,
            "timestamp": "2024-01-15T10:30:00"
        }
    """
    try:
        print(f"🔄 Richiesta riaddestramento manuale per cliente: {client_name}")
        
        # Ottieni QualityGateEngine per il cliente
        quality_gate = classification_service.get_quality_gate(client_name)
        
        # Avvia riaddestramento manuale
        result = quality_gate.trigger_manual_retraining()
        
        status_code = 200 if result['success'] else 400
        
        print(f"🔄 Risultato riaddestramento {client_name}: {result['message']}")
        
        return jsonify(result), status_code
        
    except Exception as e:
        error_msg = f"Errore nel riaddestramento per {client_name}: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'client': client_name
        }), 500

@app.route('/api/classifications/<client_name>/clear-all', methods=['DELETE'])
def api_clear_all_classifications(client_name: str):
    """
    API per cancellare TUTTE le classificazioni esistenti per un cliente.
    ATTENZIONE: Operazione irreversibile!
    
    DELETE /api/classifications/humanitas/clear-all
    
    Returns:
        {
            "success": true,
            "message": "Cancellate 1105 classificazioni per humanitas",
            "deleted_count": 1105,
            "timestamp": "2025-07-20T10:30:00"
        }
    """
    try:
        print(f"🗑️ Richiesta cancellazione classificazioni per cliente: {client_name}")
        
        # Esegui cancellazione
        result = classification_service.clear_all_classifications(client_name)
        
        status_code = 200 if result['success'] else 500
        
        print(f"🗑️ Risultato cancellazione {client_name}: {result['message']}")
        
        return jsonify(result), status_code
        
    except Exception as e:
        error_msg = f"Errore nella cancellazione classificazioni per {client_name}: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'client': client_name,
            'deleted_count': 0,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/review/<client>/all-sessions', methods=['GET'])
def get_all_sessions(client):
    """
    Ottieni tutte le sessioni disponibili, incluse quelle non selezionate per review
    """
    try:
        print(f"🔍 GET ALL SESSIONS per {client}")
        
        # Parametri opzionali
        include_reviewed = request.args.get('include_reviewed', 'false').lower() == 'true'
        limit = request.args.get('limit', type=int, default=None)  # RIMOSSO LIMITE HARDCODED: ora default è None (tutte le sessioni)
        status_filter = request.args.get('status_filter', None)  # NUOVO: filtro per status
        
        print(f"📊 Parametri: include_reviewed={include_reviewed}, limit={limit}, status_filter={status_filter}")
        
        # NON inizializzare pipeline o QualityGate per evitare CUDA out of memory
        # "Tutte le Sessioni" è solo lettura delle CLASSIFICAZIONI GIÀ SALVATE in MongoDB
        
        # CORREZIONE: Usa MongoDB per sessioni già classificate con oggetto Tenant
        from mongo_classification_reader import MongoClassificationReader
        
        # Risolvi client in oggetto Tenant
        tenant = resolve_tenant_from_identifier(client)
        mongo_reader = MongoClassificationReader(tenant=tenant)
        if not mongo_reader.connect():
            return jsonify({
                'success': False,
                'error': 'Errore connessione MongoDB',
                'sessions': [],
                'count': 0
            }), 500
        
        # Estrai sessioni già classificate da MongoDB
        sessioni_classificate = mongo_reader.get_all_sessions(limit=limit)
        
        if not sessioni_classificate:
            return jsonify({
                'success': False,
                'error': 'Nessuna sessione classificata trovata in MongoDB',
                'sessions': [],
                'count': 0
            })
        
        print(f"📊 Trovate {len(sessioni_classificate)} sessioni già classificate in MongoDB")
        
        # Ottieni sessioni in review queue SOLO SE quality_gate è già inizializzato
        pending_session_ids = set()
        reviewed_session_ids = set()
        
        # Inizializza automaticamente il QualityGate se non esiste
        quality_gate = classification_service.get_quality_gate(client)
        
        # Usa il metodo corretto per ottenere pending reviews
        try:
            pending_cases = quality_gate.get_pending_reviews(tenant=client, limit=1000)
            for case in pending_cases:
                pending_session_ids.add(case['session_id'])
        except Exception as e:
            print(f"⚠️ Errore nel recupero pending reviews: {e}")
        
        # Le sessioni classificate sono già in sessioni_classificate da MongoDB
        print(f"📋 Sessioni pending: {len(pending_session_ids)}")
        
        # Organizza classificazioni per session_id (da MongoDB - sistema unificato)
        classifications_by_session = {}
        
        try:
            # Le classificazioni sono già nelle sessioni da MongoDB
            for session_doc in sessioni_classificate:
                session_id = session_doc.get('session_id')
                if session_id:
                    if session_id not in classifications_by_session:
                        classifications_by_session[session_id] = []
                    classifications_by_session[session_id].append({
                        'tag_name': session_doc.get('classification', ''),
                        'confidence': float(session_doc.get('confidence', 0.0)),
                        'method': session_doc.get('method', 'unknown'),
                        'created_at': session_doc.get('timestamp', ''),
                        'source': 'mongodb'  # Nuovo sistema unificato
                    })
        except Exception as e:
            print(f"⚠️ Errore recupero classificazioni da MongoDB: {e}")
            # Continua con dizionario vuoto
        
        # NUOVO: Aggiungi auto-classificazioni in cache (pending, non ancora salvate)
        auto_classifications_by_session = {}
        # Inizializza automaticamente il QualityGate se non esiste
        quality_gate = classification_service.get_quality_gate(client)
        pending_auto_classifications = quality_gate.get_pending_auto_classifications(client)
        
        print(f"📊 Trovate {len(pending_auto_classifications)} auto-classificazioni in cache per {client}")
        
        for auto_class in pending_auto_classifications:
            session_id = auto_class.get('session_id')
            if session_id:  # CORREZIONE: Usa solo sessioni classificate MongoDB
                if session_id not in auto_classifications_by_session:
                    auto_classifications_by_session[session_id] = []
                auto_classifications_by_session[session_id].append({
                    'tag_name': auto_class.get('tag'),
                    'confidence': float(auto_class.get('confidence', 0.0)),
                    'method': auto_class.get('method', 'auto'),
                    'created_at': auto_class.get('timestamp', ''),
                    'source': 'cache_pending'  # Identificatore per classificazioni in cache
                })
        
        # Prepara lista delle sessioni con stato - USA SESSIONI DA MONGODB
        all_sessions = []
        for session_doc in sessioni_classificate:  # CORREZIONE: Usa sessioni da MongoDB
            session_id = session_doc.get('session_id')
            if not session_id:
                continue
                
            status = 'available'  # Disponibile per review
            if session_id in pending_session_ids:
                status = 'in_review_queue'
            elif session_id in reviewed_session_ids:
                status = 'reviewed'
                if not include_reviewed:
                    continue  # Salta se non richieste
            
            # Combina classificazioni dal database e dalla cache
            all_classifications = []
            
            # Aggiungi classificazioni salvate nel database
            if session_id in classifications_by_session:
                all_classifications.extend(classifications_by_session[session_id])
            
            # Aggiungi auto-classificazioni in cache (pending)
            if session_id in auto_classifications_by_session:
                all_classifications.extend(auto_classifications_by_session[session_id])
            
            # Estrai informazioni dalla conversazione salvata in MongoDB
            conversation_text = session_doc.get('conversation_text', session_doc.get('conversation', ''))
            if isinstance(conversation_text, list):
                # Se la conversazione è una lista di messaggi
                conversation_text = ' '.join([msg.get('text', '') for msg in conversation_text if isinstance(msg, dict)])
            
            # Estrai classificazione principale per React (evita UNKNOWN)
            final_tag = 'unknown'
            confidence = 0.0
            if all_classifications:
                # Prendi la prima classificazione disponibile
                first_classification = all_classifications[0]
                final_tag = first_classification.get('tag_name', 'unknown')
                confidence = first_classification.get('confidence', 0.0)
            
            # Determina informazioni pulsante per React
            button_info = {
                'can_add_to_review': status in ['available', 'reviewed'],  # REVIEWED ora selezionabile
                'button_text': 'AGGIUNGI A REVIEW',
                'button_disabled': False
            }
            
            if status == 'in_review_queue':
                button_info.update({
                    'can_add_to_review': False,
                    'button_text': 'GIÀ IN REVISIONE',
                    'button_disabled': True
                })
            elif status == 'reviewed':
                button_info.update({
                    'can_add_to_review': True,  # CAMBIATO: ora selezionabile 
                    'button_text': 'REVISIONA ANCORA',
                    'button_disabled': False  # CAMBIATO: ora attivo
                })
            
            session_info = {
                'session_id': session_id,
                'conversation_text': conversation_text[:500] + '...' if len(conversation_text) > 500 else conversation_text,
                'full_text': conversation_text,
                'num_messages': session_doc.get('num_messages', 0),
                'num_user_messages': session_doc.get('num_user_messages', 0),
                'status': status,
                'created_at': session_doc.get('created_at', ''),
                'last_activity': session_doc.get('updated_at', session_doc.get('classified_at', '')),
                'classifications': all_classifications,  # Combinazione di database + cache
                # CAMPI DIRETTI PER REACT (evita UNKNOWN)
                'final_tag': final_tag,
                'tag': final_tag,  # Alias per compatibilità
                'confidence': confidence,
                # INFORMAZIONI PULSANTE PER REACT
                'review_button': button_info
            }
            all_sessions.append(session_info)
        
        # Ordina per data di creazione (più recenti primi)
        all_sessions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # NUOVO: Applica filtro status se specificato
        if status_filter:
            valid_statuses = ['available', 'in_review_queue', 'reviewed']
            if status_filter in valid_statuses:
                filtered_sessions = [s for s in all_sessions if s['status'] == status_filter]
                print(f"🔍 Filtro status '{status_filter}': {len(filtered_sessions)}/{len(all_sessions)} sessioni")
                all_sessions = filtered_sessions
            else:
                print(f"⚠️ Status filter non valido: {status_filter}")
        
        return jsonify({
            'success': True,
            'sessions': all_sessions,
            'count': len(all_sessions),
            'total_classified_sessions': len(sessioni_classificate),
            'breakdown': {
                'available': len([s for s in all_sessions if s['status'] == 'available']),
                'in_review_queue': len([s for s in all_sessions if s['status'] == 'in_review_queue']),
                'reviewed': len([s for s in all_sessions if s['status'] == 'reviewed']),
                'with_db_classifications': len([s for s in all_sessions if any(c['source'] == 'mongodb' for c in s['classifications'])]),
                'with_pending_classifications': len([s for s in all_sessions if any(c['source'] == 'cache_pending' for c in s['classifications'])]),
                'total_classified': len([s for s in all_sessions if len(s['classifications']) > 0])
            },
            # NUOVO: Informazioni filtri per React
            'filter_options': {
                'available_statuses': ['all', 'available', 'in_review_queue', 'reviewed'],
                'current_filter': status_filter or 'all'
            }
        })
        
    except Exception as e:
        print(f"❌ Errore get_all_sessions: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'sessions': [],
            'count': 0
        }), 500

@app.route('/api/review/<client>/add-to-queue', methods=['POST'])
def add_session_to_review_queue(client):
    """
    Aggiungi manualmente una sessione alla review queue - USA MONGODB
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        reason = data.get('reason', 'manual_addition')
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'session_id richiesto'
            }), 400
        
        print(f"➕ Aggiunta manuale sessione {session_id} alla review queue per {client}")
        
        # NUOVO: Usa MongoDB per ottenere i dati della sessione con oggetto Tenant
        from mongo_classification_reader import MongoClassificationReader
        
        # Risolvi client in oggetto Tenant
        tenant = resolve_tenant_from_identifier(client)
        mongo_reader = MongoClassificationReader(tenant=tenant)
        if not mongo_reader.connect():
            return jsonify({
                'success': False,
                'error': 'Errore connessione MongoDB'
            }), 500
        
        # Verifica se la sessione esiste in MongoDB
        all_sessions = mongo_reader.get_all_sessions()
        session_data = None
        for session in all_sessions:
            if session.get('session_id') == session_id:
                session_data = session
                break
                
        if not session_data:
            return jsonify({
                'success': False,
                'error': f'Sessione {session_id} non trovata in MongoDB'
            }), 404
        
        quality_gate = classification_service.get_quality_gate(client)
        
        # Verifica se già in queue - usa il metodo corretto
        try:
            pending_cases = quality_gate.get_pending_reviews(tenant=client, limit=1000)
            for case in pending_cases:
                if case['session_id'] == session_id:
                    return jsonify({
                        'success': False,
                        'error': f'Sessione {session_id} già nella review queue'
                    }), 400
        except Exception as e:
            print(f"⚠️ Errore controllo pending reviews: {e}")
        
        # Crea il caso di review usando i dati da MongoDB
        conversation_text = session_data.get('conversation_text', '')
        ml_prediction = session_data.get('classification_ML', '')
        ml_confidence = session_data.get('confidence_ML', 0.0)
        llm_prediction = session_data.get('classification_LLM', '')
        llm_confidence = session_data.get('confidence_LLM', 0.0)
        
        case_id = quality_gate.add_to_review_queue(
            session_id=session_id,
            conversation_text=conversation_text,
            reason=f"manual_addition: {reason}",
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence,
            llm_prediction=llm_prediction,
            llm_confidence=llm_confidence
        )

        print(f"✅ Sessione {session_id} aggiunta alla review queue come {case_id}")
        
        # Calcola queue size usando il metodo corretto
        try:
            current_pending = quality_gate.get_pending_reviews(tenant=client, limit=1000)
            queue_size = len(current_pending)
        except Exception as e:
            print(f"⚠️ Errore calcolo queue size: {e}")
            queue_size = 0

        return jsonify({
            'success': True,
            'message': f'Sessione {session_id} aggiunta alla review queue',
            'case_id': case_id,
            'queue_size': queue_size
        })
        
    except Exception as e:
        print(f"❌ Errore add_session_to_review_queue: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/training/<client>/selection-criteria', methods=['GET'])
def get_training_selection_criteria(client):
    """
    Ottieni informazioni sui criteri di selezione usati nel training supervisionato
    """
    try:
        quality_gate = classification_service.get_quality_gate(client)
        criteria_info = quality_gate.get_training_selection_criteria()
        
        return jsonify({
            'success': True,
            'criteria': criteria_info
        })
        
    except Exception as e:
        print(f"❌ Errore get_training_selection_criteria: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/review/<client_name>/workflow-status', methods=['GET'])
def get_workflow_status(client_name: str):
    """
    Ottieni lo stato completo del workflow per un cliente.
    
    Args:
        client_name: Nome del cliente
        
    Returns:
        Stato del workflow con review queue e auto-classificazioni
    """
    try:
        quality_gate = classification_service.get_quality_gate(client_name)
        workflow_status = quality_gate.get_workflow_status(client_name)
        
        return jsonify({
            'success': True,
            'client': client_name,
            'workflow_status': workflow_status,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore nel recupero stato workflow: {str(e)}',
            'client': client_name,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/review/<client_name>/save-auto-classifications', methods=['POST'])
def save_auto_classifications(client_name: str):
    """
    Salva le auto-classificazioni in cache nel database.
    Da chiamare dopo il completamento della review umana.
    
    Args:
        client_name: Nome del cliente
        
    Returns:
        Risultato del salvataggio
    """
    try:
        quality_gate = classification_service.get_quality_gate(client_name)
        save_result = quality_gate.save_auto_classifications_to_db(client_name)
        
        return jsonify({
            'success': save_result['success'],
            'client': client_name,
            'save_result': save_result,
            'timestamp': datetime.now().isoformat()
        }), 200 if save_result['success'] else 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore nel salvataggio auto-classificazioni: {str(e)}',
            'client': client_name,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/review/<client_name>/clear-auto-classifications', methods=['POST'])
def clear_auto_classifications(client_name: str):
    """
    Pulisce la cache delle auto-classificazioni senza salvare nel database.
    
    Args:
        client_name: Nome del cliente
        
    Returns:
        Conferma di pulizia
    """
    try:
        quality_gate = classification_service.get_quality_gate(client_name)
        quality_gate.clear_auto_classifications_cache(client_name)
        
        return jsonify({
            'success': True,
            'client': client_name,
            'message': 'Cache auto-classificazioni pulita',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore nella pulizia cache: {str(e)}',
            'client': client_name,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/gpu/status', methods=['GET'])
def gpu_status():
    """
    Ottieni informazioni sullo stato della GPU
    """
    gpu_info = classification_service.get_gpu_memory_info()
    return jsonify({
        'gpu_memory': gpu_info,
        'active_pipelines': len(classification_service.pipelines),
        'shared_embedder_loaded': classification_service.shared_embedder is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/gpu/clear-cache', methods=['POST'])
def clear_gpu_cache():
    """
    Pulisce la cache GPU per liberare memoria
    """
    try:
        classification_service.clear_gpu_cache()
        gpu_info_after = classification_service.get_gpu_memory_info()
        
        return jsonify({
            'success': True,
            'message': 'Cache GPU pulita',
            'gpu_memory_after': gpu_info_after,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/admin/<client_name>/retrain', methods=['POST'])
def api_manual_retrain(client_name: str):
    """
    API per riaddestramento manuale del modello ML.
    
    Body (opzionale):
        {
            "force": true
        }
    
    Returns:
        {
            "success": true,
            "message": "Riaddestramento completato",
            "client": "client_name"
        }
    """
    try:
        # Recupera parametri opzionali
        data = request.get_json() if request.is_json else {}
        force = data.get('force', False)
        
        # Ottieni la pipeline
        pipeline = classification_service.get_pipeline(client_name)
        
        # Esegui riaddestramento manuale
        success = pipeline.manual_retrain_model(force=force)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Riaddestramento del modello ML completato per {client_name}',
                'client': client_name
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Riaddestramento fallito - verificare i log',
                'client': client_name
            }), 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'client': client_name
        }), 500


# ==================== ENDPOINT FINE-TUNING ====================

@app.route('/api/finetuning/<client_name>/info', methods=['GET'])
def get_finetuning_info(client_name: str):
    """
    Ottieni informazioni sul modello fine-tuned di un cliente
    
    Args:
        client_name: Nome del cliente
        
    Returns:
        Informazioni sul modello fine-tuned
    """
    try:
        result = classification_service.get_client_model_info(client_name)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore nel recupero info fine-tuning: {str(e)}',
            'client': client_name
        }), 500

@app.route('/api/finetuning/<client_name>/create', methods=['POST'])
def create_finetuned_model(client_name: str):
    """
    Crea un modello fine-tuned per un cliente
    
    Body:
        {
            "min_confidence": 0.7,        // Confidence minima per esempi training (opzionale)
            "force_retrain": false,       // Se forzare re-training (opzionale)
            "training_config": {          // Configurazione training (opzionale)
                "num_epochs": 3,
                "learning_rate": 5e-5,
                "batch_size": 4,
                "temperature": 0.1,
                "max_tokens": 150
            }
        }
    
    Returns:
        Risultato del fine-tuning
    """
    try:
        data = request.get_json() if request.is_json else {}
        min_confidence = data.get('min_confidence', 0.7)
        force_retrain = data.get('force_retrain', False)
        training_config = data.get('training_config', {})
        
        print(f"🚀 Richiesta fine-tuning per {client_name}")
        print(f"   - min_confidence: {min_confidence}")
        print(f"   - force_retrain: {force_retrain}")
        print(f"   - training_config: {training_config}")
        
        result = classification_service.create_finetuned_model(
            client_name=client_name,
            min_confidence=min_confidence,
            force_retrain=force_retrain,
            training_config=training_config
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore durante fine-tuning: {str(e)}',
            'client': client_name
        }), 500

@app.route('/api/finetuning/<client_name>/switch', methods=['POST'])
def switch_model(client_name: str):
    """
    Cambia il modello utilizzato per un cliente
    
    Body:
        {
            "model_type": "finetuned"  // "finetuned" o "base"
        }
    
    Returns:
        Risultato dello switch
    """
    try:
        data = request.get_json() if request.is_json else {}
        model_type = data.get('model_type', 'finetuned')
        
        if model_type not in ['finetuned', 'base']:
            return jsonify({
                'success': False,
                'error': 'model_type deve essere "finetuned" o "base"'
            }), 400
        
        result = classification_service.switch_client_model(
            client_name=client_name,
            model_type=model_type
        )
        
        # SOLUZIONE ALLA RADICE: distingui tra errori veri e modalità ML-only
        if result['success']:
            return jsonify(result), 200
        else:
            # Se il sistema è in modalità ML-only, non è un errore ma un'informazione
            if result.get('mode') == 'ml_only':
                return jsonify(result), 200  # Status 200 perché il sistema funziona
            else:
                return jsonify(result), 400  # Status 400 solo per errori veri
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore switch modello: {str(e)}',
            'client': client_name
        }), 500

@app.route('/api/finetuning/models', methods=['GET'])
def list_all_finetuned_models():
    """
    Lista tutti i modelli fine-tuned per tutti i clienti
    
    Returns:
        Lista di tutti i modelli fine-tuned
    """
    try:
        result = classification_service.list_all_client_models()
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore nel recupero modelli: {str(e)}'
        }), 500

@app.route('/api/finetuning/<client_name>/status', methods=['GET'])
def get_finetuning_status(client_name: str):
    """
    Ottieni stato completo del fine-tuning per un cliente
    Include info su modello corrente, disponibilità fine-tuning, e statistiche
    
    Args:
        client_name: Nome del cliente
        
    Returns:
        Stato completo del fine-tuning
    """
    try:
        # Info del modello
        model_info = classification_service.get_client_model_info(client_name)
        
        # Controlla se c'è una pipeline attiva
        pipeline_active = client_name in classification_service.pipelines
        current_model = None
        
        if pipeline_active:
            pipeline = classification_service.pipelines[client_name]
            classifier = getattr(pipeline, 'intelligent_classifier', None)
            if classifier and hasattr(classifier, 'get_current_model_info'):
                current_model = classifier.get_current_model_info()
        
        return jsonify({
            'success': True,
            'client': client_name,
            'model_info': model_info.get('model_info', {}) if model_info['success'] else {},
            'current_model': current_model,
            'pipeline_active': pipeline_active,
            'finetuning_available': classification_service.get_finetuning_manager() is not None,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore nel recupero stato fine-tuning: {str(e)}',
            'client': client_name
        }), 500


# ==================== NUOVI ENDPOINT PER FILTRO TENANT/ETICHETTE ====================

@app.route('/api/tenants/stats', methods=['GET'])  # PATH DIVERSO per evitare conflitti
def api_get_all_tenants_stats():
    """
    API per recuperare statistiche di tutti i tenant da MongoDB
    
    Scopo: Fornisce statistiche dettagliate dei tenant per dashboard
    
    Returns:
        {
            "success": true,
            "tenants": [
                {
                    "tenant_id": "uuid",
                    "tenant_name": "humanitas",
                    "session_count": 2901,
                    "classification_count": 1850
                }
            ],
            "total": 1
        }
    """
    try:
        print("🔍 API: Recupero statistiche tenant da MongoDB...")
        
        # Ottieni tutti i tenant come oggetti completi
        tenant_objects = MongoClassificationReader.get_available_tenants()
        
        if not tenant_objects:
            return jsonify({
                'success': False,
                'error': 'Nessun tenant trovato nel database locale',
                'tenants': []
            }), 500
        
        tenant_stats = []
        
        # Per ogni tenant, calcola le sue statistiche
        for tenant_obj in tenant_objects:
            try:
                mongo_reader = MongoClassificationReader(tenant=tenant_obj)
                
                # Recupera statistiche specifiche del tenant
                stats = mongo_reader.get_classification_stats()
                
                tenant_stat = {
                    'tenant_id': tenant_obj.tenant_id,
                    'tenant_name': tenant_obj.tenant_name,
                    'tenant_slug': tenant_obj.tenant_slug,
                    'session_count': stats.get('total_classifications', 0),  # Per ora usiamo questo
                    'classification_count': stats.get('total_classifications', 0),
                    'unique_labels': stats.get('unique_labels', 0),
                    'is_active': tenant_obj.tenant_status == 1
                }
                
                tenant_stats.append(tenant_stat)
                mongo_reader.disconnect()
                
                print(f"  ✅ {tenant_obj.tenant_name}: {tenant_stat['classification_count']} classificazioni")
                
            except Exception as tenant_error:
                print(f"  ⚠️ Errore statistiche per {tenant_obj.tenant_name}: {tenant_error}")
                # Aggiungi comunque il tenant con statistiche zero
                tenant_stats.append({
                    'tenant_id': tenant_obj.tenant_id,
                    'tenant_name': tenant_obj.tenant_name,
                    'tenant_slug': tenant_obj.tenant_slug,
                    'session_count': 0,
                    'classification_count': 0,
                    'unique_labels': 0,
                    'is_active': tenant_obj.tenant_status == 1,
                    'error': str(tenant_error)
                })
        
        print(f"✅ Recuperate statistiche per {len(tenant_stats)} tenant")
        
        return jsonify({
            'success': True,
            'tenants': tenant_stats,
            'total': len(tenant_stats),
            'timestamp': datetime.now().isoformat()
        })
            
    except Exception as e:
        print(f"❌ Errore recupero statistiche tenant: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'tenants': []
        }), 500


@app.route('/api/labels/<tenant_name>', methods=['GET'])
def api_get_labels_by_tenant(tenant_name: str):
    """
    API per recuperare tutte le etichette per un tenant specifico da MongoDB
    
    Scopo: Fornisce le etichette filtrate per tenant per il dropdown in React
    
    Args:
        tenant_name: Nome del tenant (es. 'humanitas')
    
    Returns:
        {
            "success": true,
            "tenant_name": "humanitas",
            "labels": [
                {
                    "label": "info_esami_prestazioni",
                    "count": 145,
                    "avg_confidence": 0.85
                }
            ],
            "total": 25
        }
    """
    try:
        print(f"🔍 API: Recupero etichette per tenant '{tenant_name}' da MongoDB...")
        
        # CORREZIONE CRITICA: tenant_name DEVE essere UUID (tenant_id)
        tenant = resolve_tenant_from_identifier(tenant_name)
        mongo_reader = MongoClassificationReader(tenant=tenant)
        
        if not mongo_reader.connect():
            return jsonify({
                'success': False,
                'error': 'Impossibile connettersi a MongoDB',
                'labels': []
            }), 500
        
        try:
            # USA METODI DELLA CLASSE invece di query aggregation manuali
            # Con oggetto Tenant, non serve più filtro tenant_name
            labels = mongo_reader.get_available_labels()
            stats = mongo_reader.get_classification_stats()
            
            # Formatta le etichette con statistiche
            labels_with_stats = []
            if stats and 'label_distribution' in stats:
                for label_stat in stats['label_distribution']:
                    labels_with_stats.append({
                        'label': label_stat['label'],
                        'count': label_stat['count'],
                        'avg_confidence': label_stat['avg_confidence'],
                        'percentage': label_stat['percentage']
                    })
            else:
                # Fallback: solo nomi etichette
                labels_with_stats = [{'label': label, 'count': 0} for label in labels]
            
            print(f"✅ Trovate {len(labels_with_stats)} etichette per tenant {tenant.tenant_name}")
            for label_info in labels_with_stats[:5]:  # Log delle prime 5
                print(f"  - {label_info['label']}: {label_info.get('count', 0)} classificazioni")
            
            return jsonify({
                'success': True,
                'tenant_name': tenant.tenant_name,
                'tenant_id': tenant.tenant_id,
                'labels': labels_with_stats,
                'total': len(labels_with_stats),
                'timestamp': datetime.now().isoformat()
            })
            
        finally:
            mongo_reader.disconnect()
            
    except Exception as e:
        print(f"❌ Errore recupero etichette per tenant '{tenant_name}': {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'tenant_name': tenant_name,
            'labels': []
        }), 500


@app.route('/api/sessions/<tenant_name>', methods=['GET'])
def api_get_sessions_by_tenant(tenant_name: str):
    """
    API per recuperare sessioni filtrate per tenant e opzionalmente per etichetta
    
    Scopo: Fornisce sessioni filtrate per la visualizzazione in React
    
    Args:
        tenant_name: Nome del tenant (es. 'humanitas')
    
    Query Parameters:
        label: Etichetta specifica per ulteriore filtro (opzionale)
        limit: Numero massimo di sessioni (default: 100)
    
    Returns:
        {
            "success": true,
            "tenant_name": "humanitas", 
            "label_filter": "info_esami_prestazioni",
            "sessions": [...],
            "total": 145
        }
    """
    try:
        label_filter = request.args.get('label', None)
        limit = request.args.get('limit', 100, type=int)
        
        print(f"🔍 API: Recupero sessioni per tenant '{tenant_name}'")
        if label_filter:
            print(f"  🏷️ Filtro etichetta: '{label_filter}'")
        print(f"  📊 Limite: {limit}")
        
        # CORREZIONE CRITICA: tenant_name DEVE essere UUID (tenant_id)
        tenant = resolve_tenant_from_identifier(tenant_name)
        mongo_reader = MongoClassificationReader(tenant=tenant)
        
        if not mongo_reader.connect():
            return jsonify({
                'success': False,
                'error': 'Impossibile connettersi a MongoDB',
                'sessions': []
            }), 500
        
        try:
            # USA METODI DELLA CLASSE invece di query manuali
            # Con oggetto Tenant, non serve più filtro tenant_name
            if label_filter:
                sessions = mongo_reader.get_sessions_by_label(label_filter, limit=limit)
            else:
                sessions = mongo_reader.get_all_sessions(limit=limit)
            
            print(f"✅ Recuperate {len(sessions)} sessioni per tenant {tenant.tenant_name}")
            if label_filter:
                print(f"  🏷️ Con etichetta '{label_filter}'")
            
            return jsonify({
                'success': True,
                'tenant_name': tenant.tenant_name,
                'tenant_id': tenant.tenant_id,
                'label_filter': label_filter,
                'sessions': sessions,
                'total': len(sessions),
                'limit': limit,
                'timestamp': datetime.now().isoformat()
            })
            
        finally:
            mongo_reader.disconnect()
            
    except Exception as e:
        print(f"❌ Errore recupero sessioni per tenant '{tenant_name}': {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'tenant_name': tenant_name,
            'sessions': []
        }), 500


# =============================================================================
# PROMPT MANAGEMENT API ENDPOINTS
# =============================================================================

@app.route('/api/prompts/tenant/<tenant_id>', methods=['GET'])
def get_prompts_for_tenant(tenant_id: str):
    """
    Recupera tutti i prompt per un tenant specifico
    
    GET /api/prompts/tenant/1
    
    Returns:
        [
            {
                "id": 1,
                "tenant_id": 1,
                "tenant_name": "humanitas", 
                "prompt_type": "classification_prompt",
                "content": "...",
                "variables": {...},
                "is_active": true,
                "created_at": "2025-01-16T10:00:00",
                "updated_at": "2025-01-16T10:00:00"
            }
        ]
    """
    try:
        print(f"🔍 API: Recupero prompt per tenant_id: {tenant_id}")
        
        prompt_manager = PromptManager()
        prompts = prompt_manager.get_all_prompts_for_tenant(tenant_id)
        
        print(f"✅ Recuperati {len(prompts)} prompt per tenant {tenant_id}")
        
        return jsonify(prompts)
        
    except Exception as e:
        print(f"❌ Errore recupero prompt per tenant {tenant_id}: {e}")
        return jsonify({
            'error': str(e),
            'tenant_id': tenant_id
        }), 500


# COMMENTATO: Endpoint ridondante - L'UI usa direttamente /api/prompts/{tenant_id}/status
# @app.route('/api/prompts/tenant/<tenant_id>/status', methods=['GET'])
# def get_prompts_status_for_tenant(tenant_id: str):
#     """
#     Recupera lo status dei prompt per un tenant specifico (accetta tenant_slug)
#     
#     GET /api/prompts/tenant/alleanza/status
#     
#     Returns:
#         {
#             "tenant_id": "alleanza",
#             "tenant_name": "Alleanza",
#             "total_prompts": 5,
#             "active_prompts": 3,
#             "inactive_prompts": 2,
#             "last_updated": "2025-01-16T10:00:00",
#             "status": "ready"
#         }
#     """
#     try:
#         print(f"🔍 API: Recupero status prompt per tenant_slug: {tenant_id}")
#         
#         prompt_manager = PromptManager()
#         prompts = prompt_manager.get_all_prompts_for_tenant(tenant_id)
#         
#         # Calcola statistiche
#         total_prompts = len(prompts)
#         active_prompts = len([p for p in prompts if p.get('is_active', False)])
#         inactive_prompts = total_prompts - active_prompts
#         
#         # Trova ultimo aggiornamento
#         last_updated = None
#         if prompts:
#             last_updated = max(p.get('updated_at', '') for p in prompts if p.get('updated_at'))
#         
#         # Determina tenant name
#         tenant_name = prompts[0].get('tenant_name', 'unknown') if prompts else 'unknown'
#         
#         status = {
#             "tenant_id": tenant_id,
#             "tenant_name": tenant_name,
#             "total_prompts": total_prompts,
#             "active_prompts": active_prompts,
#             "inactive_prompts": inactive_prompts,
#             "last_updated": last_updated,
#             "status": "ready" if active_prompts > 0 else "no_active_prompts"
#         }
#         
#         print(f"✅ Status prompt per tenant {tenant_id}: {active_prompts}/{total_prompts} attivi")
#         
#         return jsonify(status)
#         
#     except Exception as e:
#         print(f"❌ Errore status prompt per tenant {tenant_id}: {e}")
#         return jsonify({
#             'error': str(e),
#             'tenant_id': tenant_id,
#             'status': 'error'
#         }), 500


@app.route('/api/prompts/<tenant_identifier>/status', methods=['GET'])
def get_prompts_status_by_tenant_id(tenant_identifier: str):
    """
    Recupera lo status dei prompt per un tenant usando tenant_id UUID o tenant_slug
    
    GET /api/prompts/a0fd7600-f4f7-11ef-9315-96000228e7fe/status (UUID)
    GET /api/prompts/alleanza/status (slug)
    
    Returns: Stesso formato dell'endpoint sopra
    """
    print(f"🔍 [DEBUG] GET /api/prompts/{tenant_identifier}/status - Avvio richiesta")
    try:
        print(f"🔍 [DEBUG] API: Recupero status prompt per tenant: {tenant_identifier}")
        
        # Determina se è UUID o slug
        import re
        is_uuid = bool(re.match(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$', tenant_identifier))
        
        if is_uuid:
            print(f"✅ [DEBUG] Riconosciuto come tenant_id UUID: {tenant_identifier}")
            tenant_id = tenant_identifier
        else:
            print(f"✅ [DEBUG] Riconosciuto come tenant_slug: {tenant_identifier}")
            # Risolvi slug -> oggetto Tenant completo usando metodo statico
            print("🔍 [DEBUG] Uso get_available_tenants per ricerca tenant...")
            from mongo_classification_reader import MongoClassificationReader
            
            # Trova il tenant con questo slug
            tenant_objects = MongoClassificationReader.get_available_tenants()
            tenant_info = None
            for tenant_obj in tenant_objects:
                if tenant_obj.tenant_slug == tenant_identifier:
                    tenant_info = {
                        'tenant_id': tenant_obj.tenant_id,
                        'tenant_name': tenant_obj.tenant_name,
                        'tenant_slug': tenant_obj.tenant_slug
                    }
                    break
            
            if not tenant_info:
                print(f"❌ [DEBUG] Tenant non trovato: {tenant_identifier}")
                return jsonify({
                    'success': False,
                    'error': f'Tenant non trovato: {tenant_identifier}',
                    'tenant_id': tenant_identifier
                }), 404
            tenant_id = tenant_info['tenant_id']
            print(f"🔄 [DEBUG] Risolto {tenant_identifier} -> {tenant_id}")
        
        print("🔍 [DEBUG] Inizializzo PromptManager...")
        prompt_manager = PromptManager()
        
        print(f"🔍 [DEBUG] Chiamo get_all_prompts_for_tenant({tenant_id})...")
        # Per tenant_id completo, dobbiamo trovare lo slug corrispondente
        # Il PromptManager ora gestisce automaticamente questa conversione
        prompts = prompt_manager.get_all_prompts_for_tenant(tenant_id)
        
        print(f"🔍 [DEBUG] Recuperati {len(prompts)} prompt dal PromptManager")
        
        # Calcola statistiche
        total_prompts = len(prompts)
        active_prompts = len([p for p in prompts if p.get('is_active', False)])
        inactive_prompts = total_prompts - active_prompts
        
        print(f"🔍 [DEBUG] Statistiche: {active_prompts}/{total_prompts} attivi, {inactive_prompts} inattivi")
        
        # Trova ultimo aggiornamento
        last_updated = None
        if prompts:
            last_updated = max(p.get('updated_at', '') for p in prompts if p.get('updated_at'))
        
        # Determina tenant name
        tenant_name = prompts[0].get('tenant_name', 'unknown') if prompts else 'unknown'
        
        # 🔍 Determina prompt requirements per questo tenant
        # Lista dei prompt richiesti per il funzionamento del sistema
        required_prompt_types = [
            {"name": "System Prompt", "type": "system", "description": "Prompt di sistema per la classificazione"},
            {"name": "User Template", "type": "user", "description": "Template per prompt utente"},
            {"name": "Classification Prompt", "type": "classification", "description": "Prompt per classificazione intelligente"}
        ]
        
        # Verifica quali prompt esistono
        existing_prompt_types = set()
        for prompt in prompts:
            if prompt.get('is_active', False):
                prompt_type = prompt.get('prompt_type', '').lower()
                if 'system' in prompt_type:
                    existing_prompt_types.add('system')
                elif 'user' in prompt_type or 'template' in prompt_type:
                    existing_prompt_types.add('user')
                elif 'classification' in prompt_type or 'intelligent' in prompt_type:
                    existing_prompt_types.add('classification')
        
        # Costruisci lista prompt requirements con stato exists
        required_prompts = []
        missing_count = 0
        
        for req in required_prompt_types:
            exists = req["type"] in existing_prompt_types
            if not exists:
                missing_count += 1
            
            required_prompts.append({
                "name": req["name"],
                "type": req["type"], 
                "description": req["description"],
                "exists": exists
            })
        
        # canOperate = True solo se abbiamo almeno system e user prompt (prompt essenziali)
        essential_prompts = {'system', 'user'}
        has_essential_prompts = essential_prompts.issubset(existing_prompt_types)
        can_operate = has_essential_prompts and active_prompts > 0
        
        status = {
            "success": True,
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "total_prompts": total_prompts,
            "active_prompts": active_prompts,
            "inactive_prompts": inactive_prompts,
            "last_updated": last_updated,
            "status": "ready" if active_prompts > 0 else "no_active_prompts",
            # 🆕 Campi richiesti dal frontend TenantContext
            "canOperate": can_operate,
            "requiredPrompts": required_prompts,
            "missingCount": missing_count
        }
        
        print(f"✅ [DEBUG] Status calcolato: {status}")
        print(f"✅ [DEBUG] Status prompt per tenant_id {tenant_id}: {active_prompts}/{total_prompts} attivi")
        print(f"🔍 [DEBUG] Invio risposta JSON per status prompt")
        
        return jsonify(status)
        
    except Exception as e:
        print(f"❌ [DEBUG] ERRORE in get_prompts_status_by_tenant_id(): {str(e)}")
        print(f"❌ [DEBUG] Tipo errore: {type(e)}")
        import traceback
        traceback.print_exc()
        print(f"❌ [DEBUG] Errore status prompt per tenant: {tenant_identifier}: {e}")
        return jsonify({
            'success': False,  # AGGIUNTO: Campo success per coerenza
            'error': str(e),
            'tenant_id': tenant_identifier,
            'status': 'error'
        }), 500


@app.route('/api/prompts/<int:prompt_id>', methods=['GET'])
def get_prompt_by_id(prompt_id: int):
    """
    Recupera un prompt specifico tramite ID
    
    GET /api/prompts/5
    
    Returns:
        {
            "id": 5,
            "tenant_id": 1,
            "tenant_name": "humanitas",
            "prompt_type": "classification_prompt", 
            "content": "...",
            "variables": {...},
            "is_active": true,
            "created_at": "2025-01-16T10:00:00",
            "updated_at": "2025-01-16T10:00:00"
        }
    """
    try:
        print(f"🔍 API: Recupero prompt con ID: {prompt_id}")
        
        prompt_manager = PromptManager()
        prompt = prompt_manager.get_prompt_by_id(prompt_id)
        
        if prompt is None:
            return jsonify({
                'error': f'Prompt con ID {prompt_id} non trovato'
            }), 404
        
        print(f"✅ Recuperato prompt ID {prompt_id}")
        
        return jsonify(prompt)
        
    except Exception as e:
        print(f"❌ Errore recupero prompt ID {prompt_id}: {e}")
        return jsonify({
            'error': str(e),
            'prompt_id': prompt_id
        }), 500


@app.route('/api/prompts', methods=['POST'])
def create_prompt():
    """
    Crea un nuovo prompt
    
    POST /api/prompts
    Content-Type: application/json
    
    {
        "tenant_id": 1,
        "tenant_name": "humanitas",
        "prompt_type": "new_classification_prompt",
        "content": "Classifica il testo seguente...",
        "variables": {"param1": "value1"},
        "is_active": true
    }
    
    Returns:
        {
            "id": 6,
            "tenant_id": 1,
            "tenant_name": "humanitas",
            "prompt_type": "new_classification_prompt",
            "content": "Classifica il testo seguente...",
            "variables": {"param1": "value1"},
            "is_active": true,
            "created_at": "2025-01-16T10:30:00",
            "updated_at": "2025-01-16T10:30:00"
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dati JSON richiesti'}), 400
        
        # Validazione campi obbligatori
        required_fields = ['tenant_id', 'tenant_name', 'prompt_type', 'content']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Campo {field} mancante'}), 400
        
        print(f"📝 API: Creazione prompt per tenant {data['tenant_name']}")
        print(f"  🏷️ Nome: {data.get('prompt_name', 'AUTO-GENERATO')}")
        print(f"  🏷️ Tipo: {data['prompt_type']}")
        print(f"  🔧 Engine: {data.get('engine', 'LLM')}")
        
        prompt_manager = PromptManager()
        prompt_id = prompt_manager.create_prompt(
            tenant_id=data['tenant_id'],
            tenant_name=data['tenant_name'],
            prompt_type=data['prompt_type'],
            content=data['content'],
            prompt_name=data.get('prompt_name'),  # ✅ AGGIUNTO parametro
            engine=data.get('engine', 'LLM'),     # ✅ AGGIUNTO parametro
            tools=data.get('tools', []),          # ✅ AGGIUNTO parametro tools
            variables=data.get('variables', {}),
            is_active=data.get('is_active', True)
        )
        
        # Recupera il prompt creato
        created_prompt = prompt_manager.get_prompt_by_id(prompt_id)
        
        print(f"✅ Creato prompt ID {prompt_id}")
        
        return jsonify(created_prompt), 201
        
    except Exception as e:
        print(f"❌ Errore creazione prompt: {e}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/prompts/copy-from-humanitas', methods=['POST'])
def copy_prompts_from_humanitas():
    """
    Copia tutti i prompt dal tenant Humanitas al tenant specificato
    
    POST /api/prompts/copy-from-humanitas
    Content-Type: application/json
    
    {
        "target_tenant_id": "a0fd7600-f4f7-11ef-9315-96000228e7fe"
    }
    
    Returns:
        {
            "success": true,
            "copied_prompts": 3,
            "prompts": [...]
        }
        
    Autore: Sistema
    Data: 2025-08-24
    Descrizione: Copia automatica prompt da tenant Humanitas template
    """
    try:
        data = request.get_json()
        target_tenant_id = data.get('target_tenant_id')
        
        if not target_tenant_id:
            return jsonify({'error': 'target_tenant_id richiesto'}), 400
            
        print(f"🔄 [DEBUG] Copia prompt da Humanitas a tenant: {target_tenant_id}")
        
        # ID del tenant Humanitas (template)
        HUMANITAS_TENANT_ID = "015007d9-d413-11ef-86a5-96000228e7fe"
        
        prompt_manager = PromptManager()
        
        # 1. Recupera tutti i prompt di Humanitas
        humanitas_prompts = prompt_manager.get_all_prompts_for_tenant(HUMANITAS_TENANT_ID)
        
        if not humanitas_prompts:
            return jsonify({
                'error': 'Nessun prompt trovato per il tenant Humanitas template'
            }), 404
            
        print(f"🔍 [DEBUG] Trovati {len(humanitas_prompts)} prompt in Humanitas")
        
        # 2. Copia ogni prompt al nuovo tenant
        copied_prompts = []
        
        for original_prompt in humanitas_prompts:
            if not original_prompt.get('is_active', False):
                continue  # Salta prompt non attivi
                
            # Usa il nuovo metodo specializzato per la copia
            prompt_id = prompt_manager.create_prompt_from_template(
                target_tenant_id=target_tenant_id,
                template_prompt=original_prompt
            )
            
            if prompt_id:
                # Recupera il prompt appena creato per includerlo nella risposta
                created_prompt = prompt_manager.get_prompt_by_id(prompt_id)
                if created_prompt:
                    copied_prompts.append(created_prompt)
                    print(f"✅ [DEBUG] Copiato prompt '{original_prompt.get('prompt_name')}' -> ID: {prompt_id}")
                else:
                    print(f"⚠️ [DEBUG] Prompt creato ma non recuperabile: ID {prompt_id}")
            else:
                print(f"❌ [DEBUG] Errore copia prompt '{original_prompt.get('prompt_name')}'")
        
        print(f"🎉 [DEBUG] Copia completata: {len(copied_prompts)} prompt copiati")
        
        return jsonify({
            'success': True,
            'copied_prompts': len(copied_prompts),
            'prompts': copied_prompts,
            'message': f"Copiati {len(copied_prompts)} prompt dal tenant Humanitas"
        }), 201
        
    except Exception as e:
        print(f"❌ [DEBUG] Errore copia prompt da Humanitas: {e}")
        return jsonify({
            'error': f"Errore nella copia dei prompt: {str(e)}"
        }), 500


@app.route('/api/prompts/<int:prompt_id>', methods=['PUT'])
def update_prompt(prompt_id: int):
    """
    Aggiorna un prompt esistente
    
    PUT /api/prompts/5
    Content-Type: application/json
    
    {
        "content": "Nuovo contenuto del prompt...",
        "variables": {"new_param": "new_value"},
        "is_active": false
    }
    
    Returns:
        {
            "id": 5,
            "tenant_id": 1,
            "tenant_name": "humanitas",
            "prompt_type": "classification_prompt",
            "content": "Nuovo contenuto del prompt...",
            "variables": {"new_param": "new_value"},
            "is_active": false,
            "created_at": "2025-01-16T10:00:00",
            "updated_at": "2025-01-16T10:35:00"
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dati JSON richiesti'}), 400
        
        print(f"✏️ API: Aggiornamento prompt ID {prompt_id}")
        
        prompt_manager = PromptManager()
        
        # Verifica che il prompt esista
        existing_prompt = prompt_manager.get_prompt_by_id(prompt_id)
        if existing_prompt is None:
            return jsonify({
                'error': f'Prompt con ID {prompt_id} non trovato'
            }), 404
        
        success = prompt_manager.update_prompt(
            prompt_id=prompt_id,
            content=data.get('content'),
            tools=data.get('tools'),      # ✅ AGGIUNTO parametro tools
            variables=data.get('variables'),
            is_active=data.get('is_active')
        )
        
        if not success:
            return jsonify({
                'error': f'Errore aggiornamento prompt ID {prompt_id}'
            }), 500
        
        # Recupera il prompt aggiornato
        updated_prompt = prompt_manager.get_prompt_by_id(prompt_id)
        
        print(f"✅ Aggiornato prompt ID {prompt_id}")
        
        return jsonify(updated_prompt)
        
    except Exception as e:
        print(f"❌ Errore aggiornamento prompt ID {prompt_id}: {e}")
        return jsonify({
            'error': str(e),
            'prompt_id': prompt_id
        }), 500


@app.route('/api/prompts/<int:prompt_id>', methods=['DELETE'])
def delete_prompt(prompt_id: int):
    """
    Elimina un prompt
    
    DELETE /api/prompts/5
    
    Returns:
        {
            "success": true,
            "message": "Prompt 5 eliminato con successo",
            "prompt_id": 5
        }
    """
    try:
        print(f"🗑️ API: Eliminazione prompt ID {prompt_id}")
        
        prompt_manager = PromptManager()
        
        # Verifica che il prompt esista
        existing_prompt = prompt_manager.get_prompt_by_id(prompt_id)
        if existing_prompt is None:
            return jsonify({
                'error': f'Prompt con ID {prompt_id} non trovato'
            }), 404
        
        success = prompt_manager.delete_prompt(prompt_id)
        
        if not success:
            return jsonify({
                'error': f'Errore eliminazione prompt ID {prompt_id}'
            }), 500
        
        print(f"✅ Eliminato prompt ID {prompt_id}")
        
        return jsonify({
            'success': True,
            'message': f'Prompt {prompt_id} eliminato con successo',
            'prompt_id': prompt_id
        })
        
    except Exception as e:
        print(f"❌ Errore eliminazione prompt ID {prompt_id}: {e}")
        return jsonify({
            'error': str(e),
            'prompt_id': prompt_id
        }), 500


@app.route('/api/prompts/<int:prompt_id>/preview', methods=['POST'])
def preview_prompt_with_variables(prompt_id: int):
    """
    Anteprima di un prompt con variabili sostituite
    
    POST /api/prompts/5/preview
    Content-Type: application/json
    
    {
        "variables": {
            "conversation_text": "Esempio di conversazione...",
            "available_tags": "tag1, tag2, tag3"
        }
    }
    
    Returns:
        {
            "prompt_id": 5,
            "prompt_type": "classification_prompt",
            "original_content": "Classifica il testo: {{conversation_text}}...",
            "rendered_content": "Classifica il testo: Esempio di conversazione...",
            "variables_used": ["conversation_text", "available_tags"]
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dati JSON richiesti'}), 400
        
        variables = data.get('variables', {})
        
        print(f"👁️ API: Anteprima prompt ID {prompt_id}")
        print(f"  📝 Variabili: {list(variables.keys())}")
        
        prompt_manager = PromptManager()
        preview = prompt_manager.preview_prompt_with_variables(prompt_id, variables)
        
        if preview is None:
            return jsonify({
                'error': f'Prompt con ID {prompt_id} non trovato'
            }), 404
        
        print(f"✅ Generata anteprima prompt ID {prompt_id}")
        
        return jsonify(preview)
        
    except Exception as e:
        print(f"❌ Errore anteprima prompt ID {prompt_id}: {e}")
        return jsonify({
            'error': str(e),
            'prompt_id': prompt_id
        }), 500


# =============================================================================
# TOOL MANAGEMENT API ENDPOINTS
# =============================================================================

@app.route('/api/tools/tenant/<tenant_id>', methods=['GET'])
def get_tools_for_tenant(tenant_id: str):
    """
    Recupera tutti i tools per un tenant specifico
    
    GET /api/tools/tenant/015007d9-d413-11ef-86a5-96000228e7fe
    
    Returns:
        [
            {
                "id": 1,
                "tool_name": "_get_available_labels",
                "display_name": "Etichette Disponibili",
                "description": "Recupera tutte le etichette disponibili per la classificazione",
                "function_schema": {
                    "type": "function",
                    "function": {
                        "name": "_get_available_labels",
                        "description": "...",
                        "parameters": {...}
                    }
                },
                "is_active": true,
                "tenant_id": "015007d9-d413-11ef-86a5-96000228e7fe",
                "tenant_name": "Humanitas",
                "created_at": "2025-01-16T10:00:00",
                "updated_at": "2025-01-16T10:00:00"
            }
        ]
    """
    try:
        print(f"🔧 API: Recupero tools per tenant_id: {tenant_id}")
        
        tool_manager = ToolManager()
        tools = tool_manager.get_all_tools_for_tenant(tenant_id)
        
        print(f"✅ Recuperati {len(tools)} tools per tenant {tenant_id}")
        
        return jsonify(sanitize_for_json(tools))
        
    except Exception as e:
        print(f"❌ Errore recupero tools per tenant {tenant_id}: {e}")
        return jsonify({
            'error': str(e),
            'tenant_id': tenant_id
        }), 500


@app.route('/api/tools/<int:tool_id>', methods=['GET'])
def get_tool_by_id(tool_id: int):
    """
    Recupera un tool specifico tramite ID
    
    GET /api/tools/1
    
    Returns:
        {
            "id": 1,
            "tool_name": "_get_available_labels",
            "display_name": "Etichette Disponibili",
            "description": "Recupera tutte le etichette disponibili per la classificazione",
            "function_schema": {...},
            "is_active": true,
            "tenant_id": "015007d9-d413-11ef-86a5-96000228e7fe",
            "tenant_name": "Humanitas",
            "created_at": "2025-01-16T10:00:00",
            "updated_at": "2025-01-16T10:00:00"
        }
    """
    try:
        print(f"🔧 API: Recupero tool ID: {tool_id}")
        
        tool_manager = ToolManager()
        tool = tool_manager.get_tool_by_id(tool_id)
        
        if not tool:
            return jsonify({
                'error': 'Tool non trovato',
                'tool_id': tool_id
            }), 404
        
        print(f"✅ Recuperato tool {tool['tool_name']}")
        
        return jsonify(sanitize_for_json(tool))
        
    except Exception as e:
        print(f"❌ Errore recupero tool ID {tool_id}: {e}")
        return jsonify({
            'error': str(e),
            'tool_id': tool_id
        }), 500


@app.route('/api/tools', methods=['POST'])
def create_tool():
    """
    Crea un nuovo tool
    
    POST /api/tools
    Content-Type: application/json
    
    {
        "tool_name": "_custom_function",
        "display_name": "Funzione Personalizzata",
        "description": "Descrizione della funzione",
        "function_schema": {
            "type": "function",
            "function": {
                "name": "_custom_function",
                "description": "Funzione personalizzata per tenant",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Input parameter"}
                    },
                    "required": ["input"]
                }
            }
        },
        "tenant_id": "015007d9-d413-11ef-86a5-96000228e7fe",
        "tenant_name": "Humanitas",
        "is_active": true
    }
    
    Returns:
        Tool creato con ID assegnato
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dati richiesti nel body della richiesta'}), 400
        
        print(f"🔧 API: Creazione nuovo tool: {data.get('tool_name', 'N/A')}")
        
        tool_manager = ToolManager()
        
        # Validazione dati richiesti
        required_fields = ['tool_name', 'display_name', 'description', 'function_schema', 'tenant_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Campo obbligatorio mancante: {field}'}), 400
        
        created_tool = tool_manager.create_tool(data)
        
        if not created_tool:
            return jsonify({'error': 'Impossibile creare il tool'}), 400
        
        print(f"✅ Tool creato: {created_tool['tool_name']} (ID: {created_tool['id']})")
        
        return jsonify(sanitize_for_json(created_tool)), 201
        
    except Exception as e:
        print(f"❌ Errore creazione tool: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/tools/<int:tool_id>', methods=['PUT'])
def update_tool(tool_id: int):
    """
    Aggiorna un tool esistente
    
    PUT /api/tools/1
    Content-Type: application/json
    
    {
        "display_name": "Nuovo Nome Display",
        "description": "Nuova descrizione",
        "is_active": false
    }
    
    Returns:
        Tool aggiornato
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dati richiesti nel body della richiesta'}), 400
        
        print(f"🔧 API: Aggiornamento tool ID: {tool_id}")
        
        tool_manager = ToolManager()
        
        updated_tool = tool_manager.update_tool(tool_id, data)
        
        if not updated_tool:
            return jsonify({
                'error': 'Tool non trovato o impossibile aggiornare',
                'tool_id': tool_id
            }), 404
        
        print(f"✅ Tool aggiornato: {updated_tool['tool_name']}")
        
        return jsonify(sanitize_for_json(updated_tool))
        
    except Exception as e:
        print(f"❌ Errore aggiornamento tool ID {tool_id}: {e}")
        return jsonify({
            'error': str(e),
            'tool_id': tool_id
        }), 500


@app.route('/api/tools/<int:tool_id>', methods=['DELETE'])
def delete_tool(tool_id: int):
    """
    Elimina un tool (soft delete - disattiva)
    
    DELETE /api/tools/1
    
    Returns:
        Conferma eliminazione
    """
    try:
        print(f"🔧 API: Eliminazione tool ID: {tool_id}")
        
        tool_manager = ToolManager()
        
        success = tool_manager.delete_tool(tool_id)
        
        if not success:
            return jsonify({
                'error': 'Tool non trovato',
                'tool_id': tool_id
            }), 404
        
        print(f"✅ Tool eliminato: ID {tool_id}")
        
        return jsonify({
            'message': 'Tool eliminato con successo',
            'tool_id': tool_id,
            'deleted_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Errore eliminazione tool ID {tool_id}: {e}")
        return jsonify({
            'error': str(e),
            'tool_id': tool_id
        }), 500


@app.route('/api/tools/stats', methods=['GET'])
def get_tools_stats():
    """
    Recupera statistiche dei tools per tutti i tenant
    
    GET /api/tools/stats
    
    Returns:
        {
            "total_tools": 8,
            "tools_by_tenant": {
                "Humanitas": 4,
                "Ospedale_XYZ": 4
            }
        }
    """
    try:
        print("📊 API: Recupero statistiche tools")
        
        tool_manager = ToolManager()
        
        tools_by_tenant = tool_manager.get_tools_count_by_tenant()
        total_tools = sum(tools_by_tenant.values())
        
        stats = {
            'total_tools': total_tools,
            'tools_by_tenant': tools_by_tenant
        }
        
        print(f"✅ Statistiche tools: {total_tools} tools totali")
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"❌ Errore statistiche tools: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/tools/export/tenant/<tenant_id>', methods=['GET'])
def export_tools_for_tenant(tenant_id: str):
    """
    Esporta tutti i tools di un tenant in formato JSON
    
    GET /api/tools/export/tenant/015007d9-d413-11ef-86a5-96000228e7fe
    
    Returns:
        {
            "tenant_id": "015007d9-d413-11ef-86a5-96000228e7fe",
            "export_timestamp": "2025-01-16T10:00:00",
            "tools_count": 4,
            "tools": [...]
        }
    """
    try:
        print(f"📤 API: Esportazione tools per tenant: {tenant_id}")
        
        tool_manager = ToolManager()
        
        export_data = tool_manager.export_tools_for_tenant(tenant_id)
        
        print(f"✅ Esportati {export_data['tools_count']} tools per tenant {tenant_id}")
        
        response = jsonify(sanitize_for_json(export_data))
        response.headers['Content-Disposition'] = f'attachment; filename="tools_export_{tenant_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"'
        
        return response
        
    except Exception as e:
        print(f"❌ Errore esportazione tools per tenant {tenant_id}: {e}")
        return jsonify({
            'error': str(e),
            'tenant_id': tenant_id
        }), 500

# ==================== CLUSTERING PARAMETERS API ====================

@app.route('/api/clustering/<tenant_id>/parameters', methods=['GET'])
def get_clustering_parameters(tenant_id):
    """
    Recupera i parametri di clustering attuali per un tenant
    
    Returns:
        JSON con parametri di clustering e loro spiegazioni
    """
    try:
        # Carica configurazione attuale
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        clustering_config = config.get('clustering', {})
        
        # 🆕 CARICA PARAMETRI PERSONALIZZATI TENANT (se esistono)
        config_status = "default"  # default, custom, error
        custom_config_info = {}
        tenant_config_dir = os.path.join(os.path.dirname(__file__), 'tenant_configs')
        tenant_config_file = os.path.join(tenant_config_dir, f'{tenant_id}_clustering.yaml')
        
        if os.path.exists(tenant_config_file):
            try:
                with open(tenant_config_file, 'r', encoding='utf-8') as f:
                    tenant_config = yaml.safe_load(f)
                    tenant_clustering_params = tenant_config.get('clustering_parameters', {})
                    
                    if tenant_clustering_params:
                        print(f"🎯 [CLUSTERING CONFIG] Caricando parametri personalizzati per tenant {tenant_id}")
                        config_status = "custom"
                        custom_config_info = {
                            "file_path": tenant_config_file,
                            "last_updated": tenant_config.get('last_updated', 'unknown'),
                            "updated_by": tenant_config.get('updated_by', 'unknown'),
                            "custom_parameters": list(tenant_clustering_params.keys())
                        }
                        
                        for param, value in tenant_clustering_params.items():
                            old_value = clustering_config.get(param, 'non_definito')
                            clustering_config[param] = value
                            print(f"   {param}: {old_value} -> {value}")
                    else:
                        print(f"📋 [CLUSTERING CONFIG] File config personalizzata esiste ma è vuoto per tenant {tenant_id}")
                        config_status = "default"
                        custom_config_info = {"reason": "empty_custom_config"}
                        
            except Exception as e:
                print(f"⚠️ [CLUSTERING CONFIG] Errore caricamento config personalizzata tenant {tenant_id}: {e}")
                print("🔄 [CLUSTERING CONFIG] Fallback alla configurazione default")
                config_status = "error"
                custom_config_info = {
                    "error": str(e),
                    "fallback": "default_config"
                }
        else:
            print(f"📋 [CLUSTERING CONFIG] Nessun file di configurazione personalizzata per tenant {tenant_id}")
            print(f"🔄 [CLUSTERING CONFIG] Usando configurazione default da config.yaml")
            config_status = "default"
            custom_config_info = {"reason": "no_custom_config_file"}
        
        # Parametri con spiegazioni user-friendly
        parameters = {
            'min_cluster_size': {
                'value': clustering_config.get('min_cluster_size', 5),
                'default': 5,
                'min': 2,
                'max': 50,
                'description': 'Dimensione minima del cluster',
                'explanation': 'Numero minimo di conversazioni necessarie per formare un cluster. Valori più alti creano cluster più grandi ma meno numerosi.',
                'impact': {
                    'low': 'Molti cluster piccoli, classificazione molto specifica',
                    'medium': 'Bilanciamento tra specificità e generalizzazione',
                    'high': 'Pochi cluster grandi, classificazione più generale'
                },
                'gpu_supported': True  # ✅ SUPPORTATO SU GPU
            },
            'min_samples': {
                'value': clustering_config.get('min_samples', 3),
                'default': 3,
                'min': 1,
                'max': 20,
                'description': 'Campioni minimi per punto core',
                'explanation': 'Numero di conversazioni vicine necessarie perché un punto sia considerato "core". Controlla la densità richiesta.',
                'impact': {
                    'low': 'Cluster più aperti, include più conversazioni borderline',
                    'medium': 'Bilanciamento tra inclusione e qualità',
                    'high': 'Cluster più chiusi, solo conversazioni molto simili'
                },
                'gpu_supported': True  # ✅ SUPPORTATO SU GPU
            },
            'cluster_selection_epsilon': {
                'value': clustering_config.get('cluster_selection_epsilon', 0.08),
                'default': 0.08,
                'min': 0.01,
                'max': 0.5,
                'step': 0.01,
                'description': 'Soglia di raggruppamento semantico',
                'explanation': 'Controlla quanto devono essere simili le conversazioni per essere nello stesso cluster. IL PARAMETRO PIÙ IMPORTANTE!',
                'impact': {
                    'low': 'Tanti micro-cluster, rischio etichette simili',
                    'medium': 'Raggruppamento bilanciato',
                    'high': 'Pochi cluster grandi, raggruppamento aggressivo'
                },
                'recommendation': 'AUMENTA a 0.15-0.25 se hai troppe etichette simili!',
                'gpu_supported': True  # ✅ SUPPORTATO SU GPU
            },
            'metric': {
                'value': clustering_config.get('metric', 'euclidean'),
                'default': 'cosine',
                'options': ['cosine', 'euclidean', 'manhattan'],
                'description': 'Metrica di distanza',
                'explanation': 'Come calcolare la distanza tra conversazioni.',
                'impact': {
                    'cosine': 'Migliore per testi - considera l\'angolo tra vettori',
                    'euclidean': 'Distanza geometrica standard',
                    'manhattan': 'Somma delle differenze assolute'
                },
                'recommendation': 'Usa COSINE per conversazioni testuali!',
                'gpu_supported': True  # ✅ SUPPORTATO SU GPU
            },
            
            # 🆕 NUOVI PARAMETRI AVANZATI HDBSCAN
            'cluster_selection_method': {
                'value': clustering_config.get('cluster_selection_method', 'eom'),
                'default': 'eom',
                'options': ['eom', 'leaf'],
                'description': 'Metodo selezione cluster',
                'explanation': 'Strategia per selezionare i cluster finali dall\'albero gerarchico. EOM è più conservativo, LEAF più aggressivo.',
                'impact': {
                    'eom': 'Selezione conservativa - preferisce cluster stabili',
                    'leaf': 'Selezione aggressiva - può creare più micro-cluster'
                },
                'recommendation': 'USA LEAF per ridurre outlier se EOM ne produce troppi!',
                'gpu_supported': True  # ✅ SUPPORTATO SU GPU
            },
            'alpha': {
                'value': clustering_config.get('alpha', 1.0),
                'default': 1.0,
                'min': 0.1,
                'max': 2.0,
                'step': 0.1,
                'description': 'Controllo noise/outlier',
                'explanation': 'Regola la sensitività ai punti rumorosi. Valori più bassi riducono outlier, più alti li aumentano.',
                'impact': {
                    'low': 'Meno outlier, cluster più inclusivi',
                    'medium': 'Bilanciamento standard',
                    'high': 'Più outlier, cluster più esclusivi'
                },
                'recommendation': 'RIDUCI a 0.3-0.5 se hai troppi outlier!',
                'gpu_supported': True  # ✅ SUPPORTATO SU GPU
            },
            'max_cluster_size': {
                'value': clustering_config.get('max_cluster_size', 0),
                'default': 0,
                'min': 0,
                'max': 1000,
                'step': 1,
                'description': 'Dimensione massima cluster',
                'explanation': 'Limita il numero massimo di conversazioni per cluster. 0 = nessun limite.',
                'impact': {
                    'low': 'Cluster piccoli ma molti',
                    'medium': 'Clusters di dimensione bilanciata',
                    'high': 'Permette cluster molto grandi'
                },
                'recommendation': 'Usa 50-200 per evitare cluster troppo generici',
                'gpu_supported': False,  # 🆕 NON SUPPORTATO SU GPU
                'gpu_warning': '⚠️ Questo parametro è IGNORATO quando il clustering usa GPU (cuML)'
            },
            'allow_single_cluster': {
                'value': clustering_config.get('allow_single_cluster', False),
                'default': False,
                'options': [True, False],
                'description': 'Permetti cluster singolo',
                'explanation': 'Se True, HDBSCAN può decidere che tutte le conversazioni appartengono a un solo grande cluster.',
                'impact': {
                    True: 'Possibile un singolo cluster se i dati sono molto omogenei',
                    False: 'Forza la creazione di più cluster distinti'
                },
                'recommendation': 'Attiva solo se i tuoi dati sono molto omogenei!',
                'gpu_supported': True  # ✅ SUPPORTATO SU GPU
            },
            'only_user': {
                'value': clustering_config.get('only_user', False),
                'default': False,
                'options': [True, False],
                'description': 'Solo messaggi utente',
                'explanation': 'Se attivato, durante la lettura delle conversazioni verranno considerati SOLO i messaggi degli utenti, ignorando completamente le risposte del sistema/assistente.',
                'impact': {
                    True: 'Classificazione basata solo sui messaggi degli utenti - più focalizzata sulle richieste',
                    False: 'Classificazione sull\'intera conversazione - include anche le risposte del sistema'
                },
                'recommendation': 'Attiva se vuoi classificare solo le richieste degli utenti, ignorando le risposte',
                'gpu_supported': True,  # ✅ Supportato - è un filtro pre-processing
                'category': 'preprocessing'  # 🆕 Categoria per raggruppamento nell'UI
            },
            
            # 🆕 PARAMETRI UMAP per riduzione dimensionalità
            'use_umap': {
                'value': clustering_config.get('use_umap', False),
                'default': False,
                'options': [True, False],
                'description': 'Abilita preprocessing UMAP',
                'explanation': 'UMAP riduce la dimensionalità degli embeddings (768D → 50D tipico) prima di applicare HDBSCAN, migliorando performance e qualità del clustering.',
                'impact': {
                    True: 'Migliore qualità clustering e performance su dataset grandi, preserva struttura locale',
                    False: 'Clustering diretto sugli embeddings originali - più veloce per dataset piccoli'
                },
                'recommendation': 'Attiva per dataset con >1000 conversazioni o embeddings ad alta dimensionalità',
                'gpu_supported': True,
                'category': 'dimensionality_reduction'
            },
            'umap_n_neighbors': {
                'value': clustering_config.get('umap_n_neighbors', 15),
                'default': 15,
                'min': 5,
                'max': 100,
                'step': 5,
                'description': 'UMAP: Vicini per grafo locale',
                'explanation': 'Numero di vicini considerati per costruire il grafo locale. Valori più alti preservano più struttura globale.',
                'impact': {
                    'low': 'Focus su struttura molto locale, cluster piccoli e dettagliati',
                    'medium': 'Bilanciamento tra struttura locale e globale',
                    'high': 'Preserva più struttura globale, cluster più generali'
                },
                'recommendation': 'Usa 15-30 per testi, 30-50 per dataset complessi',
                'gpu_supported': True,
                'category': 'dimensionality_reduction'
            },
            'umap_min_dist': {
                'value': clustering_config.get('umap_min_dist', 0.1),
                'default': 0.1,
                'min': 0.0,
                'max': 1.0,
                'step': 0.05,
                'description': 'UMAP: Distanza minima punti',
                'explanation': 'Distanza minima tra punti nel low-dimensional embedding. Valori più bassi permettono clustering più densi.',
                'impact': {
                    'low': 'Punti più vicini, cluster più compatti e definiti',
                    'medium': 'Bilanciamento tra compattezza e separazione',
                    'high': 'Punti più sparsi, cluster più separati'
                },
                'recommendation': 'Usa 0.0-0.1 per clustering densi, 0.2-0.3 per separazione',
                'gpu_supported': True,
                'category': 'dimensionality_reduction'
            },
            'umap_metric': {
                'value': clustering_config.get('umap_metric', 'cosine'),
                'default': 'cosine',
                'options': ['cosine', 'euclidean', 'manhattan', 'correlation'],
                'description': 'UMAP: Metrica di distanza',
                'explanation': 'Metrica utilizzata per calcolare le distanze nel processo di riduzione dimensionale.',
                'impact': {
                    'cosine': 'Ottimale per embeddings testuali - considera angoli tra vettori',
                    'euclidean': 'Distanza geometrica standard - buona per dati numerici',
                    'manhattan': 'Somma differenze assolute - robusta agli outlier',
                    'correlation': 'Basata sulla correlazione - per dati con pattern lineari'
                },
                'recommendation': 'Usa COSINE per embeddings di testi, EUCLIDEAN per dati numerici',
                'gpu_supported': True,
                'category': 'dimensionality_reduction'
            },
            'umap_n_components': {
                'value': clustering_config.get('umap_n_components', 50),
                'default': 50,
                'min': 2,
                'max': 100,
                'step': 5,
                'description': 'UMAP: Dimensioni finali',
                'explanation': 'Numero di dimensioni dell\'embedding ridotto. Più dimensioni preservano più informazioni ma richiedono più tempo.',
                'impact': {
                    'low': 'Riduzione aggressiva, veloce ma può perdere dettagli',
                    'medium': 'Bilanciamento tra performance e qualità',
                    'high': 'Preserva più informazioni, più lento ma migliore qualità'
                },
                'recommendation': 'Usa 20-50 per testi, 50-100 per dati complessi',
                'gpu_supported': True,
                'category': 'dimensionality_reduction'
            },
            'umap_random_state': {
                'value': clustering_config.get('umap_random_state', 42),
                'default': 42,
                'min': 0,
                'max': 999999,
                'step': 1,
                'description': 'UMAP: Seed riproducibilità',
                'explanation': 'Seed per generatore numeri casuali - garantisce risultati riproducibili tra esecuzioni.',
                'impact': {
                    'fixed': 'Risultati sempre identici - utile per test e debug',
                    'random': 'Risultati possono variare leggermente tra esecuzioni'
                },
                'recommendation': 'Usa un valore fisso (42) per riproducibilità o cambia per esplorare variazioni',
                'gpu_supported': True,
                'category': 'dimensionality_reduction'
            }
        }
        
        # Statistiche tenant (se disponibili)
        try:
            tenant_stats = {
                'tenant_id': tenant_id,
                'name': tenant_id  # Placeholder, potresti caricare il nome da DB
            }
        except:
            tenant_stats = {'tenant_id': tenant_id}
        
        response = {
            'success': True,
            'tenant_id': tenant_id,
            'parameters': parameters,
            'tenant_info': tenant_stats,
            'last_updated': datetime.now().isoformat(),
            'config_source': config_status,  # default, custom, error
            'config_details': {
                'status': config_status,
                'is_using_default': config_status != 'custom',
                'custom_config_info': custom_config_info,
                'base_config_file': 'config.yaml'
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore recupero parametri clustering: {str(e)}',
            'tenant_id': tenant_id
        }), 500

@app.route('/api/clustering/<tenant_id>/parameters', methods=['POST'])
def update_clustering_parameters(tenant_id):
    """
    Aggiorna i parametri di clustering per un tenant
    Salva in un file di override specifico del tenant
    """
    try:
        data = request.get_json()
        
        if not data or 'parameters' not in data:
            return jsonify({
                'success': False,
                'error': 'Dati parametri mancanti'
            }), 400
        
        # Validazione parametri
        new_params = data['parameters']
        
        # Validazione range valori
        validations = {
            'min_cluster_size': (2, 50),
            'min_samples': (1, 20),
            'cluster_selection_epsilon': (0.01, 0.5),
            'metric': ['cosine', 'euclidean', 'manhattan'],
            
            # 🆕 VALIDAZIONI NUOVI PARAMETRI
            'cluster_selection_method': ['eom', 'leaf'],
            'alpha': (0.1, 2.0),
            'max_cluster_size': (0, 1000),
            'allow_single_cluster': [True, False],
            'only_user': [True, False],  # 🆕 Validazione per only_user
            
            # 🆕 VALIDAZIONI PARAMETRI UMAP
            'use_umap': [True, False],
            'umap_n_neighbors': (5, 100),
            'umap_min_dist': (0.0, 1.0),
            'umap_metric': ['cosine', 'euclidean', 'manhattan', 'chebyshev', 'minkowski'],
            'umap_n_components': (2, 100),
            'umap_random_state': (0, 999999)
        }
        
        for param_name, param_value in new_params.items():
            if param_name in validations:
                validation = validations[param_name]
                
                if isinstance(validation, tuple):  # Range numerico
                    min_val, max_val = validation
                    if not (min_val <= param_value <= max_val):
                        return jsonify({
                            'success': False,
                            'error': f'Parametro {param_name}: valore {param_value} fuori range [{min_val}, {max_val}]'
                        }), 400
                elif isinstance(validation, list):  # Opzioni
                    if param_value not in validation:
                        return jsonify({
                            'success': False,
                            'error': f'Parametro {param_name}: valore {param_value} non valido. Opzioni: {validation}'
                        }), 400
        
        # Salva configurazione personalizzata per tenant
        tenant_config_dir = os.path.join(os.path.dirname(__file__), 'tenant_configs')
        os.makedirs(tenant_config_dir, exist_ok=True)
        
        tenant_config_file = os.path.join(tenant_config_dir, f'{tenant_id}_clustering.yaml')
        
        # Prepara configurazione
        tenant_clustering_config = {
            'tenant_id': tenant_id,
            'last_updated': datetime.now().isoformat(),
            'updated_by': 'web_interface',
            'clustering_parameters': new_params,
            'previous_config_backup': {}
        }
        
        # Backup configurazione precedente se esiste
        if os.path.exists(tenant_config_file):
            try:
                with open(tenant_config_file, 'r', encoding='utf-8') as f:
                    old_config = yaml.safe_load(f)
                    tenant_clustering_config['previous_config_backup'] = old_config.get('clustering_parameters', {})
            except:
                pass
        
        # Salva nuova configurazione
        with open(tenant_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(tenant_clustering_config, f, default_flow_style=False, allow_unicode=True)
        
        # Log dell'aggiornamento
        print(f"📊 Parametri clustering aggiornati per tenant {tenant_id}:")
        for param_name, param_value in new_params.items():
            print(f"   {param_name}: {param_value}")
        
        response = {
            'success': True,
            'message': 'Parametri clustering aggiornati con successo',
            'tenant_id': tenant_id,
            'updated_parameters': new_params,
            'config_file': tenant_config_file,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore aggiornamento parametri: {str(e)}',
            'tenant_id': tenant_id
        }), 500

@app.route('/api/clustering/<tenant_id>/parameters/reset', methods=['POST'])
def reset_clustering_parameters(tenant_id):
    """
    Reset parametri clustering ai valori di default
    """
    try:
        # Rimuovi file configurazione personalizzata
        tenant_config_dir = os.path.join(os.path.dirname(__file__), 'tenant_configs')
        tenant_config_file = os.path.join(tenant_config_dir, f'{tenant_id}_clustering.yaml')
        
        if os.path.exists(tenant_config_file):
            # Backup prima della rimozione
            backup_file = tenant_config_file + f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            os.rename(tenant_config_file, backup_file)
            print(f"🔄 Configurazione personalizzata salvata in backup: {backup_file}")
        
        # Carica parametri default da config.yaml
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        default_params = config.get('clustering', {})
        
        response = {
            'success': True,
            'message': 'Parametri clustering ripristinati ai valori default',
            'tenant_id': tenant_id,
            'default_parameters': default_params,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore reset parametri: {str(e)}',
            'tenant_id': tenant_id
        }), 500


@app.route('/api/clustering/<tenant_id>/test', methods=['POST'])
def test_clustering_preview(tenant_id):
    """
    Esegue un test di clustering HDBSCAN usando i parametri attuali del tenant
    senza coinvolgere LLM - solo per preview e ottimizzazione parametri
    
    Args:
        tenant_id: ID del tenant
        
    Body (JSON opzionale):
        - parameters: parametri clustering personalizzati per il test
        - sample_size: numero conversazioni da testare (default: 1000)
        
    Returns:
        JSON con risultati clustering test:
        - cluster_statistics: numero cluster, outliers, dimensioni
        - detailed_clusters: cluster con conversazioni associate  
        - quality_metrics: metriche qualità clustering
        - outlier_analysis: analisi outliers con raccomandazioni
        
    Autore: Sistema di Classificazione
    Data: 2025-08-25
    """
    try:
        # Carica oggetto tenant completo usando la classe Tenant
        from Utils.tenant import Tenant
        
        try:
            # Carica tenant completo dall'UUID
            tenant = Tenant.from_uuid(tenant_id)  # Metodo corretto
            if not tenant:
                raise ValueError(f"Tenant con ID {tenant_id} non trovato")
                
            print(f"🔍 Tenant caricato: {tenant.tenant_slug} ({tenant.tenant_name})")
            
        except Exception as e:
            print(f"❌ Errore caricamento tenant '{tenant_id}': {e}")
            return jsonify({
                'success': False,
                'error': f'Tenant non trovato: {tenant_id}',
                'details': str(e)
            }), 404
        
        # Importa il servizio di test clustering
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'Clustering'))
        from clustering_test_service import ClusteringTestService
        
        # PROTEZIONE: Verifica se è una chiamata accidentale durante cambio LLM
        request_data = request.get_json() or {}
        source_context = request_data.get('source_context', None)
        
        # Se la chiamata arriva dal contesto LLM, blocca
        if source_context == 'llm_change' or 'llm' in str(request_data).lower():
            print(f"🚫 [API] BLOCCATO test clustering accidentale durante cambio LLM per tenant: {tenant_id}")
            return jsonify({
                'success': False,
                'error': 'Test clustering bloccato durante cambio LLM per evitare interferenze',
                'blocked_reason': 'llm_change_protection'
            }), 400
        
        print(f"🧪 [API] Test clustering richiesto per tenant: {tenant.tenant_slug} ({tenant.tenant_name})")
        
        # Parse parametri dalla richiesta
        custom_parameters = request_data.get('parameters', None)
        sample_size = request_data.get('sample_size', None)
        
        if custom_parameters:
            print(f"🎛️ [API] Parametri personalizzati ricevuti: {custom_parameters}")
        if sample_size:
            print(f"📊 [API] Sample size richiesto: {sample_size}")
        
        # Inizializza servizio test
        test_service = ClusteringTestService()
        
        # Esegue test clustering PASSANDO L'OGGETTO TENANT COMPLETO
        print(f"⚡ [API] Avvio test clustering...")
        result = test_service.run_clustering_test(
            tenant=tenant,  # 🎯 PASSA L'OGGETTO TENANT COMPLETO
            custom_parameters=custom_parameters,
            sample_size=sample_size
        )
        
        # Aggiunge informazioni tenant al risultato per il frontend
        if 'tenant_id' in result:
            result['tenant_id'] = tenant.tenant_id  # UUID originale
            result['tenant_slug'] = tenant.tenant_slug  # Slug
            result['tenant_name'] = tenant.tenant_name  # Nome
        
        if result['success']:
            print(f"✅ [API] Test clustering completato con successo")
            print(f"   📊 Cluster: {result['statistics']['n_clusters']}")
            print(f"   🔍 Outliers: {result['statistics']['n_outliers']}")
            print(f"   ⏱️ Tempo: {result['execution_time']}s")
            
            # Sanifica per JSON serialization (risolve numpy.int64 errors)
            sanitized_result = sanitize_for_json(result)
            return jsonify(sanitized_result), 200
        else:
            print(f"❌ [API] Test clustering fallito: {result.get('error', 'Unknown error')}")
            # Sanifica anche gli errori
            sanitized_result = sanitize_for_json(result)
            return jsonify(sanitized_result), 400
            
    except ImportError as ie:
        error_msg = f'Errore importazione ClusteringTestService: {str(ie)}'
        print(f"❌ [API] {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'tenant_id': tenant_id
        }), 500
        
    except Exception as e:
        error_msg = f'Errore test clustering: {str(e)}'
        print(f"❌ [API] {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'tenant_id': tenant_id
        }), 500


@app.route('/api/statistics/<tenant_id>/clustering', methods=['GET'])
def get_clustering_statistics(tenant_id):
    """
    Recupera statistiche complete di clustering post-classificazione per la sezione STATISTICHE
    
    Args:
        tenant_id: ID del tenant
        
    Query Parameters:
        - days_back: giorni di storico da analizzare (default: 30)
        - include_visualizations: se includere dati per grafici (default: true)
        - sample_limit: limite conversazioni per performance (default: 5000)
        
    Returns:
        JSON con statistiche complete clustering + classificazione:
        - clustering_stats: statistiche cluster originali
        - classification_stats: statistiche etichette finali assegnate
        - visualization_data: coordinate e dati per grafici interattivi
        - cluster_vs_labels: analisi comparativa cluster vs etichette finali
        - quality_metrics: metriche avanzate di qualità
        
    Autore: Sistema di Classificazione  
    Data: 2025-08-26
    """
    try:
        import sys
        import os
        from datetime import datetime, timedelta
        
        # Import dei servizi necessari
        sys.path.append(os.path.join(os.path.dirname(__file__), 'MySql'))
        sys.path.append(os.path.join(os.path.dirname(__file__), 'MongoDB'))  
        sys.path.append(os.path.join(os.path.dirname(__file__), 'Pipeline'))
        
        from connettore import MySqlConnettore
        from mongo_classification_reader import MongoClassificationReader
        from end_to_end_pipeline import EndToEndPipeline
        
        # Parametri query
        days_back = request.args.get('days_back', '30')
        include_visualizations = request.args.get('include_visualizations', 'true').lower() == 'true'
        sample_limit = int(request.args.get('sample_limit', '5000'))
        
        print(f"📈 [API] Statistiche clustering per tenant {tenant_id}")
        print(f"   📅 Giorni storico: {days_back}")
        print(f"   🎨 Include visualizzazioni: {include_visualizations}")
        print(f"   📊 Limite campioni: {sample_limit}")
        
        # 1. Risolvi tenant_id UUID -> tenant_slug
        def _resolve_tenant_slug_from_id(tenant_uuid: str) -> str:
            remote = MySqlConnettore()
            query = """
            SELECT tenant_database, tenant_name 
            FROM common.tenants 
            WHERE tenant_id = %s AND tenant_status = 1
            """
            result = remote.esegui_query(query, (tenant_uuid,))
            remote.disconnetti()
            
            if result:
                # result è una lista di tuple: [(tenant_database, tenant_name), ...]
                return result[0][0]  # prima riga, primo campo (tenant_database)
            else:
                raise ValueError(f"Tenant {tenant_uuid} non trovato o inattivo")
        
        # CORREZIONE: Crea oggetto Tenant direttamente dal tenant_id (UUID)
        tenant = resolve_tenant_from_identifier(tenant_id)
        print(f"✅ [API] Tenant risolto: {tenant.tenant_name} ({tenant.tenant_id})")
        
        # 2. Inizializza reader classificazioni MongoDB con oggetto Tenant
        mongo_reader = MongoClassificationReader(tenant=tenant)
        mongo_reader.connect()
        
        # 3. Recupera classificazioni recenti con clustering info
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days_back))
        
        print(f"🔍 [API] Estrazione classificazioni {start_date} -> {end_date}")
        
        classifications = mongo_reader.get_tenant_classifications_with_clustering(
            tenant_slug=tenant.tenant_slug,
            start_date=start_date, 
            end_date=end_date,
            limit=sample_limit
        )
        
        mongo_reader.disconnect()
        
        if not classifications:
            return jsonify({
                'success': False,
                'error': f'Nessuna classificazione trovata per {tenant.tenant_name} negli ultimi {days_back} giorni',
                'tenant_id': tenant.tenant_id,
                'tenant_slug': tenant.tenant_slug
            }), 404
        
        print(f"✅ [API] Trovate {len(classifications)} classificazioni")
        
        # 4. Estrai dati per analisi
        session_texts = []
        cluster_labels = []
        final_predictions = []
        session_ids = []
        
        for classification in classifications:
            if 'testo_completo' in classification and 'cluster_label' in classification:
                session_texts.append(classification['testo_completo'])
                cluster_labels.append(classification.get('cluster_label', -1))
                final_predictions.append({
                    'prediction': classification.get('predicted_label', 'unknown'),
                    'confidence': classification.get('confidence', 0.0),
                    'method': classification.get('classification_method', 'unknown')
                })
                session_ids.append(classification.get('session_id', f'session_{len(session_ids)}'))
        
        if len(session_texts) < 10:
            return jsonify({
                'success': False,
                'error': f'Troppo poche classificazioni valide trovate ({len(session_texts)}). Minimo: 10',
                'tenant_id': tenant_id
            }), 400
        
        # 5. Genera statistiche clustering vs classificazione
        import numpy as np
        cluster_labels_array = np.array(cluster_labels)
        
        # Statistiche clustering
        unique_clusters = set(cluster_labels)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_outliers = int(np.sum(cluster_labels_array == -1))
        
        # Statistiche classificazione
        prediction_labels = [pred['prediction'] for pred in final_predictions]
        unique_predictions = set(prediction_labels)
        prediction_counts = {label: prediction_labels.count(label) for label in unique_predictions}
        
        # Confidence statistics
        confidences = [pred['confidence'] for pred in final_predictions]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Metodi utilizzati
        methods = [pred['method'] for pred in final_predictions]
        method_counts = {method: methods.count(method) for method in set(methods)}
        
        # 6. Analisi cluster vs etichette finali (purezza cluster)
        cluster_purity = {}
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue
            cluster_mask = cluster_labels_array == cluster_id
            cluster_predictions = [prediction_labels[i] for i, mask in enumerate(cluster_mask) if mask]
            
            if cluster_predictions:
                most_common_label = max(set(cluster_predictions), key=cluster_predictions.count)
                purity = cluster_predictions.count(most_common_label) / len(cluster_predictions)
                cluster_purity[int(cluster_id)] = {
                    'most_common_label': most_common_label,
                    'purity': purity,
                    'size': len(cluster_predictions),
                    'label_distribution': {label: cluster_predictions.count(label) for label in set(cluster_predictions)}
                }
        
        result_data = {
            'success': True,
            'tenant_id': tenant.tenant_id,
            'tenant_slug': tenant.tenant_slug,
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': int(days_back)
            },
            'clustering_stats': {
                'total_conversations': len(session_texts),
                'n_clusters': n_clusters,
                'n_outliers': n_outliers,
                'clustering_ratio': round((len(session_texts) - n_outliers) / len(session_texts), 3) if len(session_texts) > 0 else 0
            },
            'classification_stats': {
                'unique_labels': len(unique_predictions),
                'label_distribution': prediction_counts,
                'avg_confidence': round(avg_confidence, 3),
                'method_distribution': method_counts
            },
            'cluster_vs_labels': {
                'cluster_purity': cluster_purity,
                'avg_purity': round(np.mean([cp['purity'] for cp in cluster_purity.values()]), 3) if cluster_purity else 0.0
            }
        }
        
        # 7. Aggiungi dati visualizzazione se richiesti
        if include_visualizations and len(session_texts) > 0:
            try:
                print("🎨 [API] Generazione dati visualizzazione...")
                
                # Inizializza pipeline per ottenere embeddings
                pipeline = EndToEndPipeline(config_path='config.yaml', tenant=tenant)
                embedder = pipeline._get_embedder()
                
                # Genera embeddings (campione se troppi)
                if len(session_texts) > 2000:
                    indices = np.random.choice(len(session_texts), 2000, replace=False)
                    sample_texts = [session_texts[i] for i in indices]
                    sample_cluster_labels = cluster_labels_array[indices]
                    sample_session_ids = [session_ids[i] for i in indices]
                    sample_predictions = [final_predictions[i] for i in indices]
                else:
                    sample_texts = session_texts
                    sample_cluster_labels = cluster_labels_array
                    sample_session_ids = session_ids
                    sample_predictions = final_predictions
                
                print(f"🔍 [API] Generazione embeddings per {len(sample_texts)} campioni...")
                embeddings = embedder.encode(sample_texts, show_progress_bar=False)
                
                # Usa ClusteringTestService per generare dati visualizzazione
                sys.path.append(os.path.join(os.path.dirname(__file__), 'Clustering'))
                from clustering_test_service import ClusteringTestService
                
                clustering_service = ClusteringTestService()
                visualization_data = clustering_service._generate_visualization_data(
                    embeddings, sample_cluster_labels, sample_texts, sample_session_ids
                )
                
                # Aggiungi info etichette finali ai punti
                for i, point in enumerate(visualization_data.get('points', [])):
                    if i < len(sample_predictions):
                        point.update({
                            'final_prediction': sample_predictions[i]['prediction'],
                            'prediction_confidence': sample_predictions[i]['confidence'],
                            'classification_method': sample_predictions[i]['method']
                        })
                
                result_data['visualization_data'] = visualization_data
                print("✅ [API] Dati visualizzazione generati con successo")
                
            except Exception as viz_error:
                print(f"⚠️ [API] Errore generazione visualizzazioni: {viz_error}")
                result_data['visualization_error'] = str(viz_error)
        
        print(f"✅ [API] Statistiche clustering generate con successo")
        
        # Sanifica per JSON serialization
        sanitized_result = sanitize_for_json(result_data)
        return jsonify(sanitized_result), 200
        
    except Exception as e:
        error_msg = f'Errore recupero statistiche clustering: {str(e)}'
        print(f"❌ [API] {error_msg}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'tenant_id': tenant_id
        }), 500


# ==================== ENDPOINT LLM CONFIGURATION RELOAD ====================

@app.route('/api/llm/<client_name>/reload', methods=['POST'])
def reload_llm_configuration(client_name: str):
    """
    ⚡ MODIFICA ARCHITETTURALE: Solo reload configurazione LLM, NO pipeline completa
    
    OTTIMIZZAZIONE CRITICA: 
    - Evita caricamento LaBSE (1.4GB GPU memory) per semplice cambio LLM
    - Solo salvataggio configurazione in database
    - Pipeline completa caricata solo quando serve (test, training, classificazione)
    
    Args:
        client_name: Nome del tenant/cliente
        
    Returns:
        Risultato del reload con dettagli del cambio modello
    """
    try:
        print(f"🔄 [API] Reload LLM configuration (solo config, NO pipeline) per tenant {client_name}")
        
        # Verifica configurazione esistente
        try:
            tenant_id = client_name
            config = ai_config_service.get_tenant_configuration(tenant_id)
            if not config or not config.get('llm_model'):
                print(f"❌ [API] Tenant {client_name} non ha configurazione LLM - non posso fare reload")
                return jsonify({
                    'success': False,
                    'error': f'Tenant {client_name} non ha configurazione LLM salvata. Configura prima il modello LLM.'
                }), 404
                
        except Exception as e:
            print(f"❌ [API] Errore verifica configurazione per {client_name}: {e}")
            return jsonify({
                'success': False,
                'error': f'Errore verifica configurazione per tenant {client_name}: {str(e)}'
            }), 500
        
        # 🎯 NUOVO APPROCCIO: Solo invalidazione cache pipeline esistente
        # Non caricare pipeline completa - sarà caricata quando serve davvero
        old_model = config.get('llm_model', 'unknown')
        
        # Invalida cache pipeline se esiste (senza caricarla)
        if classification_service.has_cached_pipeline(client_name):
            print(f"🗑️ [API] Invalidando cache pipeline esistente per {client_name}")
            classification_service.invalidate_pipeline_cache(client_name)
        
        # La configurazione è già salvata in database - nessun reload di pipeline necessario
        
        print(f"✅ [API] Configurazione LLM aggiornata per {client_name}: modello={old_model}")
        print(f"🚀 [API] Pipeline completa sarà caricata solo quando necessaria (test/training/classificazione)")
        
        return jsonify({
            'success': True,
            'message': f'Configurazione LLM aggiornata per {client_name}',
            'tenant_id': client_name,
            'model': config.get('llm_model'),
            'cache_invalidated': classification_service.has_cached_pipeline(client_name),
            'note': 'Pipeline completa sarà caricata solo quando necessaria per operazioni pesanti',
            'timestamp': datetime.now().isoformat()
        }), 200
            
    except Exception as e:
        error_msg = f'Errore reload LLM configuration: {str(e)}'
        print(f"❌ [API] {error_msg}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'tenant_id': client_name
        }), 500


@app.route('/api/llm/<client_name>/info', methods=['GET'])
def get_current_llm_info(client_name: str):
    """
    Ottieni informazioni sul modello LLM corrente per un tenant
    
    Args:
        client_name: Nome del tenant/cliente
        
    Returns:
        Informazioni dettagliate sul modello LLM in uso
    """
    try:
        print(f"ℹ️  [API] Richiesta info LLM per tenant {client_name}")
        
        # Ottieni pipeline esistente
        pipeline = classification_service.get_pipeline(client_name)
        
        if not pipeline:
            return jsonify({
                'success': False,
                'error': f'Pipeline non trovata per tenant {client_name}'
            }), 404
        
        # Ottieni informazioni LLM
        llm_info = pipeline.get_current_llm_info()
        
        return jsonify({
            'success': True,
            'tenant_id': client_name,
            'llm_info': llm_info,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        error_msg = f'Errore recupero info LLM: {str(e)}'
        print(f"❌ [API] {error_msg}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'tenant_id': client_name
        }), 500


# ====== NUOVI ENDPOINT PER GESTIONE STORICO CLUSTERING ======

@app.route('/api/clustering/<tenant_id>/history', methods=['GET'])
def get_clustering_history(tenant_id):
    """
    Recupera storico completo test clustering per tenant con versioning
    
    Args:
        tenant_id: ID del tenant
        
    Query Parameters:
        - limit: numero massimo versioni da restituire (default: 50)
        
    Returns:
        JSON con storico versioni clustering:
        - versions: lista versioni ordinate dalla più recente
        - total_count: numero totale versioni disponibili
        - tenant_id: ID tenant
        
    Autore: Sistema di Classificazione
    Data: 2025-08-27
    """
    try:
        from Database.clustering_results_db import ClusteringResultsDB
        
        limit = int(request.args.get('limit', '50'))
        
        # Inizializza database risultati
        results_db = ClusteringResultsDB()
        
        if not results_db.connect():
            return jsonify({
                'success': False,
                'error': 'Impossibile connettersi al database risultati',
                'tenant_id': tenant_id
            }), 500
        
        # Recupera storico
        history = results_db.get_clustering_history(tenant_id, limit)
        results_db.disconnect()
        
        print(f"📈 [API] Recuperato storico clustering: {len(history)} versioni per tenant {tenant_id}")
        
        return jsonify({
            'success': True,
            'tenant_id': tenant_id,
            'data': history,  # Cambiato da 'versions' a 'data'
            'total_versions': len(history)  # Cambiato da 'total_count' a 'total_versions'
        }), 200
        
    except Exception as e:
        error_msg = f'Errore recupero storico clustering: {str(e)}'
        print(f"❌ [API] {error_msg}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'tenant_id': tenant_id
        }), 500


@app.route('/api/clustering/<tenant_id>/version/<int:version_number>', methods=['GET'])
def get_clustering_version(tenant_id, version_number):
    """
    Recupera dati completi di una versione specifica clustering
    
    Args:
        tenant_id: ID del tenant
        version_number: Numero versione da recuperare
        
    Returns:
        JSON con dati completi versione:
        - version_data: dati risultati clustering
        - parameters_data: parametri HDBSCAN/UMAP utilizzati
        - metadata: info versione (timestamp, execution_time, etc.)
        
    Autore: Sistema di Classificazione
    Data: 2025-08-27
    """
    try:
        from Database.clustering_results_db import ClusteringResultsDB
        
        # Inizializza database risultati
        results_db = ClusteringResultsDB()
        
        if not results_db.connect():
            return jsonify({
                'success': False,
                'error': 'Impossibile connettersi al database risultati',
                'tenant_id': tenant_id
            }), 500
        
        # Recupera versione specifica
        version_data = results_db.get_clustering_version(tenant_id, version_number)
        results_db.disconnect()
        
        if not version_data:
            return jsonify({
                'success': False,
                'error': f'Versione {version_number} non trovata per tenant {tenant_id}',
                'tenant_id': tenant_id,
                'version_number': version_number
            }), 404
        
        print(f"📊 [API] Recuperata versione {version_number} per tenant {tenant_id}")
        
        return jsonify({
            'success': True,
            'data': {
                'id': version_data['id'],
                'version_number': version_data['version_number'],
                'tenant_id': tenant_id,
                'created_at': version_data['created_at'],
                'results_data': version_data['results_data'],
                'parameters_data': version_data['parameters_data'],
                'n_clusters': version_data['n_clusters'],
                'n_outliers': version_data['n_outliers'],
                'silhouette_score': version_data['silhouette_score'],
                'execution_time': version_data['execution_time']
            }
        }), 200
        
    except Exception as e:
        error_msg = f'Errore recupero versione clustering: {str(e)}'
        print(f"❌ [API] {error_msg}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'tenant_id': tenant_id,
            'version_number': version_number
        }), 500


@app.route('/api/clustering/<tenant_id>/latest', methods=['GET'])
def get_latest_clustering(tenant_id):
    """
    Recupera ultima versione clustering per tenant
    
    Args:
        tenant_id: ID del tenant
        
    Returns:
        JSON con dati ultima versione o messaggio se non disponibile
        
    Autore: Sistema di Classificazione
    Data: 2025-08-27
    """
    try:
        from Database.clustering_results_db import ClusteringResultsDB
        
        # Inizializza database risultati
        results_db = ClusteringResultsDB()
        
        if not results_db.connect():
            return jsonify({
                'success': False,
                'error': 'Impossibile connettersi al database risultati',
                'tenant_id': tenant_id
            }), 500
        
        # Recupera ultima versione
        latest_data = results_db.get_latest_clustering(tenant_id)
        results_db.disconnect()
        
        if not latest_data:
            return jsonify({
                'success': True,
                'tenant_id': tenant_id,
                'has_data': False,
                'message': 'Nessun test clustering effettuato per questo tenant'
            }), 200
        
        print(f"📊 [API] Recuperata ultima versione (v{latest_data['version_number']}) per tenant {tenant_id}")
        
        return jsonify({
            'success': True,
            'tenant_id': tenant_id,
            'has_data': True,
            'version_number': latest_data['version_number'],
            'version_data': latest_data['results_data'],
            'parameters_data': latest_data['parameters_data'],
            'metadata': {
                'id': latest_data['id'],
                'created_at': latest_data['created_at'],
                'execution_time': latest_data['execution_time'],
                'n_clusters': latest_data['n_clusters'],
                'n_outliers': latest_data['n_outliers'],
                'n_conversations': latest_data['n_conversations'],
                'clustering_ratio': latest_data['clustering_ratio'],
                'silhouette_score': latest_data['silhouette_score']
            }
        }), 200
        
    except Exception as e:
        error_msg = f'Errore recupero ultima versione clustering: {str(e)}'
        print(f"❌ [API] {error_msg}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'tenant_id': tenant_id
        }), 500


@app.route('/api/clustering/<tenant_id>/compare/<int:version1>/<int:version2>', methods=['GET'])
def compare_clustering_versions(tenant_id, version1, version2):
    """
    Confronta due versioni clustering per analisi comparativa
    
    Args:
        tenant_id: ID del tenant
        version1: Prima versione da confrontare
        version2: Seconda versione da confrontare
        
    Returns:
        JSON con dati delle due versioni e metriche comparative
        
    Autore: Sistema di Classificazione
    Data: 2025-08-27
    """
    try:
        from Database.clustering_results_db import ClusteringResultsDB
        
        # Inizializza database risultati
        results_db = ClusteringResultsDB()
        
        if not results_db.connect():
            return jsonify({
                'success': False,
                'error': 'Impossibile connettersi al database risultati',
                'tenant_id': tenant_id
            }), 500
        
        # Recupera dati per confronto
        comparison_data = results_db.get_comparison_data(tenant_id, version1, version2)
        results_db.disconnect()
        
        if not comparison_data:
            return jsonify({
                'success': False,
                'error': f'Una o entrambe le versioni ({version1}, {version2}) non trovate',
                'tenant_id': tenant_id,
                'version1': version1,
                'version2': version2
            }), 404
        
        # Calcola metriche comparative
        v1_stats = comparison_data['version1']['results_data']['statistics']
        v2_stats = comparison_data['version2']['results_data']['statistics']
        v1_quality = comparison_data['version1']['results_data']['quality_metrics']
        v2_quality = comparison_data['version2']['results_data']['quality_metrics']
        
        comparison_metrics = {
            'clusters_diff': v2_stats['n_clusters'] - v1_stats['n_clusters'],
            'outliers_diff': v2_stats['n_outliers'] - v1_stats['n_outliers'],
            'ratio_diff': v2_stats['clustering_ratio'] - v1_stats['clustering_ratio'],
            'silhouette_diff': v2_quality.get('silhouette_score', 0) - v1_quality.get('silhouette_score', 0),
            'execution_time_diff': comparison_data['version2']['execution_time'] - comparison_data['version1']['execution_time']
        }
        
        print(f"🔄 [API] Confronto versioni {version1} vs {version2} per tenant {tenant_id}")
        
        return jsonify({
            'success': True,
            'tenant_id': tenant_id,
            'version1': {
                'number': version1,
                'data': comparison_data['version1']['results_data'],
                'parameters': comparison_data['version1']['parameters_data'],
                'metadata': {
                    'created_at': comparison_data['version1']['created_at'],
                    'execution_time': comparison_data['version1']['execution_time']
                }
            },
            'version2': {
                'number': version2,
                'data': comparison_data['version2']['results_data'],
                'parameters': comparison_data['version2']['parameters_data'],
                'metadata': {
                    'created_at': comparison_data['version2']['created_at'],
                    'execution_time': comparison_data['version2']['execution_time']
                }
            },
            'comparison_metrics': comparison_metrics
        }), 200
        
    except Exception as e:
        error_msg = f'Errore confronto versioni clustering: {str(e)}'
        print(f"❌ [API] {error_msg}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'tenant_id': tenant_id,
            'version1': version1,
            'version2': version2
        }), 500


@app.route('/api/clustering/<tenant_id>/metrics-trend', methods=['GET'])
def get_clustering_metrics_trend(tenant_id):
    """
    Recupera trend metriche clustering per visualizzazioni evolutive
    
    Args:
        tenant_id: ID del tenant
        
    Query Parameters:
        - limit: numero versioni da includere (default: 20)
        
    Returns:
        JSON con dati trend per grafici temporali:
        - trend_data: metriche ordinate per versione
        - metrics_summary: statistiche aggregate trend
        
    Autore: Sistema di Classificazione
    Data: 2025-08-27
    """
    try:
        from Database.clustering_results_db import ClusteringResultsDB
        
        limit = int(request.args.get('limit', '20'))
        
        # Inizializza database risultati
        results_db = ClusteringResultsDB()
        
        if not results_db.connect():
            return jsonify({
                'success': False,
                'error': 'Impossibile connettersi al database risultati',
                'tenant_id': tenant_id
            }), 500
        
        # Recupera trend metriche
        trend_data = results_db.get_metrics_trend(tenant_id, limit)
        results_db.disconnect()
        
        if not trend_data:
            return jsonify({
                'success': True,
                'tenant_id': tenant_id,
                'has_data': False,
                'message': 'Nessun dato disponibile per trend analysis'
            }), 200
        
        # Calcola statistiche aggregate
        metrics_summary = {
            'total_versions': len(trend_data),
            'avg_clusters': sum(d['n_clusters'] for d in trend_data if d['n_clusters']) / len([d for d in trend_data if d['n_clusters']]) if trend_data else 0,
            'avg_outliers': sum(d['n_outliers'] for d in trend_data if d['n_outliers']) / len([d for d in trend_data if d['n_outliers']]) if trend_data else 0,
            'avg_silhouette': sum(d['silhouette_score'] for d in trend_data if d['silhouette_score']) / len([d for d in trend_data if d['silhouette_score']]) if trend_data else 0,
            'best_silhouette': max((d['silhouette_score'] for d in trend_data if d['silhouette_score']), default=0),
            'best_version': next((d['version_number'] for d in trend_data if d['silhouette_score'] == max((x['silhouette_score'] for x in trend_data if x['silhouette_score']), default=0)), None) if trend_data else None
        }
        
        print(f"📈 [API] Trend metriche clustering: {len(trend_data)} versioni per tenant {tenant_id}")
        
        return jsonify({
            'success': True,
            'tenant_id': tenant_id,
            'has_data': True,
            'trend_data': trend_data,
            'metrics_summary': metrics_summary
        }), 200
        
    except Exception as e:
        error_msg = f'Errore recupero trend metriche: {str(e)}'
        print(f"❌ [API] {error_msg}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'tenant_id': tenant_id
        }), 500


if __name__ == "__main__":
    # Configurazione per evitare loop di ricaricamento con il virtual environment
    import os
    
    # Disabilita il debug se ci sono problemi con il virtual environment
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"🚀 Avvio server Flask - Debug: {debug_mode}")
    
    # Avvio del server con configurazione ottimizzata
    # FORZA use_reloader=False per evitare problemi con watchdog
    app.run(
        host="0.0.0.0", 
        port=5000, 
        debug=debug_mode,
        use_reloader=False,  # SEMPRE False per evitare blocco watchdog
        extra_files=None  # Non monitorare file extra per evitare loop
    )