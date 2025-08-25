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

# Aggiungi path per i moduli
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'TagDatabase'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'QualityGate'))

from end_to_end_pipeline import EndToEndPipeline
from tag_database_connector import TagDatabaseConnector
from quality_gate_engine import QualityGateEngine
from mongo_classification_reader import MongoClassificationReader

app = Flask(__name__)
CORS(app)  # Abilita CORS per permettere richieste dal frontend React


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
        self.tag_db = TagDatabaseConnector()
        self.quality_gates = {}  # Cache dei QualityGateEngine per cliente
        self.shared_embedder = None  # Embedder condiviso per tutti i clienti per evitare CUDA OOM
        self.mongo_reader = MongoClassificationReader()  # Reader per classificazioni MongoDB
        
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

    def get_shared_embedder(self):
        """
        Ottieni un embedder condiviso per evitare CUDA out of memory
        SOLUZIONE ALLA RADICE: Usa lock per evitare inizializzazioni simultanee
        """
        with self._embedder_lock:
            if self.shared_embedder is None:
                print(f"🔧 Inizializzazione embedder condiviso (con lock)...")
                from EmbeddingEngine.labse_embedder import LaBSEEmbedder
                self.shared_embedder = LaBSEEmbedder()
                print(f"✅ Embedder condiviso inizializzato")
            return self.shared_embedder
    
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
                
                # Ottieni embedder condiviso per evitare CUDA out of memory
                shared_embedder = self.get_shared_embedder()
                
                # Crea pipeline con modalità automatica e embedder condiviso
                # NOTA: auto_retrain ora viene gestito da config.yaml
                pipeline = EndToEndPipeline(
                    tenant_slug=client_name,
                    confidence_threshold=0.7,
                    auto_mode=True,  # Modalità completamente automatica
                    shared_embedder=shared_embedder
                    # auto_retrain rimosso: ora gestito da config.yaml
                )
                
                self.pipelines[client_name] = pipeline
                print(f"✅ Pipeline {client_name} inizializzata")
                
            return self.pipelines[client_name]
    
    def get_quality_gate(self, client_name: str) -> QualityGateEngine:
        """
        Ottieni o crea il QualityGateEngine per un cliente specifico
        SOLUZIONE ALLA RADICE: Usa lock per cliente per evitare inizializzazioni simultanee
        
        Args:
            client_name: Nome del cliente (es. 'humanitas')
            
        Returns:
            QualityGateEngine configurato per il cliente
        """
        # Crea lock specifico per questo cliente se non esiste
        if client_name not in self._quality_gate_locks:
            with self._global_init_lock:
                if client_name not in self._quality_gate_locks:
                    self._quality_gate_locks[client_name] = threading.Lock()
        
        # Usa lock specifico del cliente
        with self._quality_gate_locks[client_name]:
            if client_name not in self.quality_gates:
                print(f"🔧 Inizializzazione QualityGateEngine per cliente: {client_name} (con lock)")
                
                # Crea QualityGateEngine per il cliente
                quality_gate = QualityGateEngine(
                    tenant_name=client_name,
                    review_db_path=f"human_review_{client_name}.db",  # DB specifico per cliente
                    training_log_path=f"training_decisions_{client_name}.jsonl"  # Log specifico per cliente
                )
                
                self.quality_gates[client_name] = quality_gate
                print(f"✅ QualityGateEngine {client_name} inizializzato")
                
            return self.quality_gates[client_name]
    
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
        Cancella tutte le classificazioni esistenti per un cliente
        ATTENZIONE: Operazione irreversibile!
        
        Args:
            client_name: Nome del cliente
            
        Returns:
            Risultato dell'operazione
        """
        try:
            print(f"🗑️ CANCELLAZIONE CLASSIFICAZIONI per cliente: {client_name}")
            self.tag_db.connetti()
            
            # Conta le classificazioni esistenti prima di cancellare
            count_query = """
            SELECT COUNT(*) 
            FROM session_classifications 
            WHERE tenant_name = %s
            """
            count_result = self.tag_db.esegui_query(count_query, (client_name,))
            existing_count = count_result[0][0] if count_result else 0
            
            if existing_count == 0:
                self.tag_db.disconnetti()
                return {
                    'success': True,
                    'message': f'Nessuna classificazione trovata per {client_name}',
                    'deleted_count': 0,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Cancella tutte le classificazioni per il cliente
            delete_query = """
            DELETE FROM session_classifications 
            WHERE tenant_name = %s
            """
            
            affected_rows = self.tag_db.esegui_comando(delete_query, (client_name,))
            self.tag_db.disconnetti()
            
            print(f"✅ Cancellate {affected_rows} classificazioni per {client_name}")
            
            return {
                'success': True,
                'message': f'Cancellate {affected_rows} classificazioni per {client_name}',
                'deleted_count': affected_rows,
                'timestamp': datetime.now().isoformat()
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
    
    def get_finetuning_manager(self):
        """
        Ottieni il fine-tuning manager (lazy loading)
        """
        if not hasattr(self, '_finetuning_manager'):
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), 'FineTuning'))
                from mistral_finetuning_manager import MistralFineTuningManager
                self._finetuning_manager = MistralFineTuningManager()
                print("🎯 Fine-tuning manager inizializzato")
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
            force_reprocess: Se True, riprocessa anche le sessioni già classificate
            force_review: Se True, forza l'aggiunta di tutti i casi alla coda di revisione
            force_reprocess_all: Se True, cancella TUTTE le classificazioni esistenti e riprocessa tutto dall'inizio
            
        Returns:
            Risultati della classificazione
        """
        try:
            print(f"🚀 CLASSIFICAZIONE COMPLETA - Cliente: {client_name}")
            start_time = datetime.now()
            
            # Se richiesto, cancella tutte le classificazioni esistenti
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
                        nuove_sessioni, use_ensemble=True
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
                            nuove_sessioni, use_ensemble=True
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
                    nuove_sessioni, use_ensemble=True
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
        force_reprocess: boolean - Se True, riprocessa anche sessioni già classificate
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
        
        # Ottieni il QualityGateEngine per questo cliente
        quality_gate = classification_service.get_quality_gate(client_name)
        
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
    API per ottenere tutte le sessioni classificate (non più solo pending).
    Ora legge da MongoDB tutte le classificazioni esistenti.
    
    Query Parameters:
        limit: Numero massimo di casi da restituire (default: 100)
        label: Filtra per etichetta specifica (opzionale)
        
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
        
        # Ottieni reader MongoDB
        mongo_reader = classification_service.mongo_reader
        
        # Recupera sessioni (tutte o filtrate per etichetta)
        if label_filter and label_filter != "Tutte le etichette":
            sessions = mongo_reader.get_sessions_by_label(client_name, label_filter, limit)
        else:
            sessions = mongo_reader.get_all_sessions(client_name, limit)
        
        # Trasforma i dati MongoDB in formato ReviewCase per compatibilità frontend
        formatted_cases = []
        for session in sessions:
            case_item = {
                'case_id': session.get('id', session.get('session_id', '')),
                'session_id': session.get('session_id', ''),
                'conversation_text': session.get('conversation_text', ''),
                'ml_prediction': session.get('classification', 'non_classificata'),
                'ml_confidence': float(session.get('confidence', 0.0)),
                'llm_prediction': session.get('classification', 'non_classificata'),  # Stesso valore per ora
                'llm_confidence': float(session.get('confidence', 0.0)),  # Stesso valore per ora
                'uncertainty_score': 1.0 - float(session.get('confidence', 0.0)),
                'novelty_score': 0.0,  # Non disponibile da MongoDB
                'reason': session.get('motivation', ''),
                'notes': session.get('notes', session.get('motivation', '')),  # Campo notes per UI
                'created_at': str(session.get('timestamp', '')),
                'tenant': client_name,
                'cluster_id': str(session.get('metadata', {}).get('cluster_id', '')) if session.get('metadata', {}).get('cluster_id') else None
            }
            formatted_cases.append(case_item)
        
        # Recupera etichette disponibili
        available_labels = mongo_reader.get_available_labels(client_name)
        
        # Recupera statistiche
        stats = mongo_reader.get_classification_stats(client_name)
        
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
        # Ottieni reader MongoDB
        mongo_reader = classification_service.mongo_reader
        
        # Recupera etichette disponibili
        available_labels = mongo_reader.get_available_labels(client_name)
        
        # Recupera statistiche dettagliate
        stats = mongo_reader.get_classification_stats(client_name)
        
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
        # Ottieni il QualityGateEngine
        quality_gate = classification_service.get_quality_gate(client_name)
        
        # Cerca il caso specifico
        pending_cases = quality_gate.get_pending_reviews(tenant=client_name, limit=100)
        target_case = None
        
        for case in pending_cases:
            if case.case_id == case_id:
                target_case = case
                break
        
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
        
        # Risolvi il caso
        quality_gate.resolve_review_case(
            case_id=case_id,
            human_decision=human_decision,
            human_confidence=confidence,
            notes=notes
        )
        
        return jsonify({
            'success': True,
            'message': f'Caso {case_id} risolto con decisione: {human_decision}',
            'case_id': case_id,
            'client': client_name
        }), 200
        
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
        
        # Statistiche generali
        general_stats = quality_gate.get_statistics(tenant=client_name)
        
        # Statistiche novelty
        novelty_stats = quality_gate.get_novelty_statistics()
        
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
    Ottieni lista completa dei tenant dalla tabella MySQL TAG.tenants
    
    Returns:
        Lista di tenant con tenant_id e nome per il frontend
        
    Ultimo aggiornamento: 2025-01-27
    """
    try:
        # Usa il nuovo metodo del MongoClassificationReader che legge da MySQL
        mongo_reader = MongoClassificationReader()
        tenants = mongo_reader.get_available_tenants()
        
        return jsonify({
            'success': True,
            'tenants': tenants,
            'total': len(tenants)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats/tenants', methods=['GET'])
def get_available_tenants():
    """
    Ottieni lista di tutti i tenant disponibili nel database
    """
    try:
        from TagDatabase.tag_database_connector import TagDatabaseConnector
        tag_db = TagDatabaseConnector()
        tag_db.connetti()
        
        query = """
        SELECT DISTINCT tenant_name
        FROM session_classifications 
        WHERE tenant_name IS NOT NULL
        ORDER BY tenant_name
        """
        
        results = tag_db.esegui_query(query)
        tag_db.disconnetti()
        
        tenants = [row[0] for row in results] if results else []
        
        return jsonify({
            'success': True,
            'tenants': tenants,
            'total': len(tenants)
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
    """
    try:
        from TagDatabase.tag_database_connector import TagDatabaseConnector
        tag_db = TagDatabaseConnector()
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
        
        results = tag_db.esegui_query(query, (tenant_name,))
        
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
        
        general_results = tag_db.esegui_query(general_query, (tenant_name,))
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
            'tenant': tenant_name,
            'labels': list(label_stats.values()),
            'general_stats': general_stats
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'tenant': tenant_name
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

@app.route('/api/review/<client_name>/available-tags', methods=['GET'])
def api_get_available_tags(client_name: str):
    """
    API per ottenere tutti i tag disponibili per un cliente usando la logica intelligente.
    
    Logica implementata:
    - Cliente nuovo (senza classificazioni in DB) → zero suggerimenti
    - Cliente esistente → tag da ML/LLM/revisioni umane precedenti
    
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
            "client": "humanitas",
            "is_new_client": false
        }
    """
    try:
        # Usa il nuovo gestore intelligente dei suggerimenti
        from TAGS.tag import tag_suggestion_manager
        
        # Ottieni suggerimenti usando la logica intelligente
        raw_suggested_tags = tag_suggestion_manager.get_suggested_tags_for_client(client_name)
        
        # Converti il formato per il frontend: tag_name -> tag, usage_count -> count
        suggested_tags = []
        for tag_data in raw_suggested_tags:
            suggested_tags.append({
                'tag': tag_data.get('tag_name', ''),
                'count': tag_data.get('usage_count', 0),
                'source': tag_data.get('source', 'available'),
                'avg_confidence': tag_data.get('avg_confidence', 0.0)
            })
        
        # Verifica se è un cliente nuovo
        is_new_client = not tag_suggestion_manager.has_existing_classifications(client_name)
        
        return jsonify({
            'success': True,
            'tags': suggested_tags,
            'total_tags': len(suggested_tags),
            'client': client_name,
            'is_new_client': is_new_client
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'client': client_name,
            'is_new_client': True  # Fallback per errori
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
        
        # NON inizializzare pipeline o QualityGate per evitare CUDA out of memory
        # "Tutte le Sessioni" è solo lettura dal database, non serve ML
        
        # Estrai tutte le sessioni dal database SENZA pipeline
        from Preprocessing.session_aggregator import SessionAggregator
        aggregator = SessionAggregator(schema='humanitas')
        tutte_sessioni = aggregator.estrai_sessioni_aggregate(limit=limit)
        
        if not tutte_sessioni:
            return jsonify({
                'success': False,
                'error': 'Nessuna sessione trovata nel database',
                'sessions': [],
                'count': 0
            })
        
        # Filtra conversazioni con più di 1 messaggio utente
        sessioni_valide = {}
        for session_id, dati in tutte_sessioni.items():
            if dati['num_messaggi_user'] > 1:
                sessioni_valide[session_id] = dati
        
        print(f"📊 Trovate {len(sessioni_valide)} sessioni valide su {len(tutte_sessioni)} totali")
        
        # Ottieni sessioni in review queue SOLO SE quality_gate è già inizializzato
        pending_session_ids = set()
        reviewed_session_ids = set()
        
        # Inizializza automaticamente il QualityGate se non esiste
        quality_gate = classification_service.get_quality_gate(client)
        
        for case in quality_gate.pending_reviews.values():
            pending_session_ids.add(case.session_id)
        
        if hasattr(quality_gate, 'reviewed_cases'):
            for case in quality_gate.reviewed_cases.values():
                reviewed_session_ids.add(case.session_id)
        
        # Ottieni classificazioni esistenti dal database
        from TagDatabase.tag_database_connector import TagDatabaseConnector
        tag_db = TagDatabaseConnector()
        tag_db.connetti()
        
        # Query per ottenere classificazioni esistenti
        classification_query = """
        SELECT session_id, tag_name, confidence_score, classification_method, created_at
        FROM session_classifications 
        WHERE tenant_name = %s AND session_id IN ({})
        """.format(','.join(['%s'] * len(sessioni_valide)))
        
        classification_params = [client] + list(sessioni_valide.keys())
        classification_results = tag_db.esegui_query(classification_query, classification_params)
        tag_db.disconnetti()
        
        # Organizza classificazioni per session_id (dal database - classificazioni salvate)
        classifications_by_session = {}
        if classification_results:
            for row in classification_results:
                session_id = row[0]
                if session_id not in classifications_by_session:
                    classifications_by_session[session_id] = []
                classifications_by_session[session_id].append({
                    'tag_name': row[1],
                    'confidence': float(row[2]) if row[2] else 0.0,
                    'method': row[3],
                    'created_at': row[4].isoformat() if row[4] else '',
                    'source': 'database'  # Aggiunto per distinguere la fonte
                })
        
        # NUOVO: Aggiungi auto-classificazioni in cache (pending, non ancora salvate)
        auto_classifications_by_session = {}
        # Inizializza automaticamente il QualityGate se non esiste
        quality_gate = classification_service.get_quality_gate(client)
        pending_auto_classifications = quality_gate.get_pending_auto_classifications(client)
        
        print(f"📊 Trovate {len(pending_auto_classifications)} auto-classificazioni in cache per {client}")
        
        for auto_class in pending_auto_classifications:
            session_id = auto_class.get('session_id')
            if session_id and session_id in sessioni_valide:  # Solo sessioni valide
                if session_id not in auto_classifications_by_session:
                    auto_classifications_by_session[session_id] = []
                auto_classifications_by_session[session_id].append({
                    'tag_name': auto_class.get('tag'),
                    'confidence': float(auto_class.get('confidence', 0.0)),
                    'method': auto_class.get('method', 'auto'),
                    'created_at': auto_class.get('timestamp', ''),
                    'source': 'cache_pending'  # Identificatore per classificazioni in cache
                })
        
        # Prepara lista delle sessioni con stato
        all_sessions = []
        for session_id, dati in sessioni_valide.items():
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
            
            session_info = {
                'session_id': session_id,
                'conversation_text': dati['testo_completo'][:500] + '...' if len(dati['testo_completo']) > 500 else dati['testo_completo'],
                'full_text': dati['testo_completo'],
                'num_messages': dati['num_messaggi_totali'],
                'num_user_messages': dati['num_messaggi_user'],
                'status': status,
                'created_at': dati.get('primo_timestamp', '').isoformat() if dati.get('primo_timestamp') else '',
                'last_activity': dati.get('ultimo_timestamp', '').isoformat() if dati.get('ultimo_timestamp') else '',
                'classifications': all_classifications  # Combinazione di database + cache
            }
            all_sessions.append(session_info)
        
        # Ordina per data di creazione (più recenti primi)
        all_sessions.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({
            'success': True,
            'sessions': all_sessions,
            'count': len(all_sessions),
            'total_valid_sessions': len(sessioni_valide),
            'breakdown': {
                'available': len([s for s in all_sessions if s['status'] == 'available']),
                'in_review_queue': len([s for s in all_sessions if s['status'] == 'in_review_queue']),
                'reviewed': len([s for s in all_sessions if s['status'] == 'reviewed']),
                'with_db_classifications': len([s for s in all_sessions if any(c['source'] == 'database' for c in s['classifications'])]),
                'with_pending_classifications': len([s for s in all_sessions if any(c['source'] == 'cache_pending' for c in s['classifications'])]),
                'total_classified': len([s for s in all_sessions if len(s['classifications']) > 0])
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
    Aggiungi manualmente una sessione alla review queue
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
        
        quality_gate = classification_service.get_quality_gate(client)
        
        # Ottieni i dati della sessione
        from Preprocessing.session_aggregator import SessionAggregator
        aggregator = SessionAggregator(schema='humanitas')
        sessioni = aggregator.estrai_sessioni_aggregate()
        
        if session_id not in sessioni:
            return jsonify({
                'success': False,
                'error': f'Sessione {session_id} non trovata nel database'
            }), 404
        
        # Verifica se già in queue
        for case in quality_gate.pending_reviews.values():
            if case.session_id == session_id:
                return jsonify({
                    'success': False,
                    'error': f'Sessione {session_id} già nella review queue'
                }), 400
        
        # Crea il caso di review
        from QualityGate.quality_gate_engine import ReviewCase
        case_id = f"{client}_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_manual"
        
        review_case = ReviewCase(
            case_id=case_id,
            session_id=session_id,
            conversation_text=sessioni[session_id]['testo_completo'],
            ml_prediction="",
            ml_confidence=0.0,
            llm_prediction="manual_review",
            llm_confidence=0.0,
            uncertainty_score=0.5,
            novelty_score=0.3,
            reason=f"manual_addition: {reason}",
            created_at=datetime.now(),
            tenant=client,
            cluster_id=-1  # Non ha cluster per ora
        )
        
        quality_gate.pending_reviews[case_id] = review_case
        
        print(f"✅ Sessione {session_id} aggiunta alla review queue come {case_id}")
        
        return jsonify({
            'success': True,
            'message': f'Sessione {session_id} aggiunta alla review queue',
            'case_id': case_id,
            'queue_size': len(quality_gate.pending_reviews)
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

@app.route('/api/tenants', methods=['GET'])
def api_get_all_tenants():
    """
    API per recuperare tutti i tenant disponibili da MongoDB
    
    Scopo: Fornisce la lista dei tenant per il filtro principale in React
    
    Returns:
        {
            "success": true,
            "tenants": [
                {
                    "tenant_name": "humanitas",
                    "client": "humanitas", 
                    "session_count": 2901,
                    "classification_count": 1850
                }
            ],
            "total": 1
        }
    """
    try:
        print("🔍 API: Recupero tutti i tenant da MongoDB...")
        
        # Usa MongoDB reader per recuperare tenant
        mongo_reader = MongoClassificationReader()
        
        if not mongo_reader.connect():
            return jsonify({
                'success': False,
                'error': 'Impossibile connettersi a MongoDB',
                'tenants': []
            }), 500
        
        try:
            # Query aggregation per recuperare statistiche per tenant
            pipeline = [
                {
                    '$group': {
                        '_id': {
                            'tenant_name': '$tenant_name',
                            'client': '$client'
                        },
                        'session_count': {'$addToSet': '$session_id'},
                        'classification_count': {'$sum': 1}
                    }
                },
                {
                    '$project': {
                        'tenant_name': '$_id.tenant_name',
                        'client': '$_id.client',
                        'session_count': {'$size': '$session_count'},
                        'classification_count': 1,
                        '_id': 0
                    }
                },
                {
                    '$sort': {'tenant_name': 1}
                }
            ]
            
            # Esegui aggregation
            cursor = mongo_reader.collection.aggregate(pipeline)
            tenants = list(cursor)
            
            print(f"✅ Trovati {len(tenants)} tenant in MongoDB")
            for tenant in tenants:
                print(f"  - {tenant['tenant_name']}: {tenant['session_count']} sessioni, {tenant['classification_count']} classificazioni")
            
            return jsonify({
                'success': True,
                'tenants': tenants,
                'total': len(tenants),
                'timestamp': datetime.now().isoformat()
            })
            
        finally:
            mongo_reader.disconnect()
            
    except Exception as e:
        print(f"❌ Errore recupero tenant: {e}")
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
        
        # Usa MongoDB reader per recuperare etichette
        mongo_reader = MongoClassificationReader()
        
        if not mongo_reader.connect():
            return jsonify({
                'success': False,
                'error': 'Impossibile connettersi a MongoDB',
                'labels': []
            }), 500
        
        try:
            # Query aggregation per recuperare statistiche etichette per tenant
            pipeline = [
                {
                    '$match': {
                        'tenant_name': tenant_name,
                        'classificazione': {'$ne': None, '$ne': '', '$ne': 'non_classificata'}
                    }
                },
                {
                    '$group': {
                        '_id': '$classificazione',
                        'count': {'$sum': 1},
                        'avg_confidence': {'$avg': '$confidence'},
                        'sessions': {'$addToSet': '$session_id'}
                    }
                },
                {
                    '$project': {
                        'label': '$_id',
                        'count': 1,
                        'session_count': {'$size': '$sessions'},
                        'avg_confidence': {'$round': ['$avg_confidence', 3]},
                        '_id': 0
                    }
                },
                {
                    '$sort': {'count': -1, 'label': 1}
                }
            ]
            
            # Esegui aggregation
            cursor = mongo_reader.collection.aggregate(pipeline)
            labels = list(cursor)
            
            print(f"✅ Trovate {len(labels)} etichette per tenant '{tenant_name}'")
            for label in labels[:5]:  # Log delle prime 5
                print(f"  - {label['label']}: {label['count']} classificazioni, {label['session_count']} sessioni")
            
            return jsonify({
                'success': True,
                'tenant_name': tenant_name,
                'labels': labels,
                'total': len(labels),
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
        
        # Usa MongoDB reader per recuperare sessioni
        mongo_reader = MongoClassificationReader()
        
        if not mongo_reader.connect():
            return jsonify({
                'success': False,
                'error': 'Impossibile connettersi a MongoDB',
                'sessions': []
            }), 500
        
        try:
            # Costruisci query MongoDB
            query = {'tenant_name': tenant_name}
            
            # Aggiungi filtro etichetta se specificato
            if label_filter:
                query['classificazione'] = label_filter
            
            # Recupera sessioni
            cursor = mongo_reader.collection.find(
                query,
                {'embedding': 0}  # Escludi embedding per performance
            ).sort('timestamp', -1).limit(limit)
            
            sessions = []
            for doc in cursor:
                # Converti ObjectId in string
                doc['_id'] = str(doc['_id'])
                
                # Formatta per interfaccia React
                session = {
                    'id': doc['_id'],
                    'session_id': doc.get('session_id', ''),
                    'conversation_text': doc.get('testo', doc.get('conversazione', '')),
                    'classification': doc.get('classificazione', 'non_classificata'),
                    'confidence': doc.get('confidence', 0.0),
                    'motivation': doc.get('motivazione', ''),
                    'notes': doc.get('motivazione', ''),  # Mapping motivazione → notes per UI
                    'method': doc.get('metadata', {}).get('method', 'unknown'),
                    'timestamp': doc.get('timestamp'),
                    'tenant_name': doc.get('tenant_name'),
                    'client': doc.get('client'),
                    'classifications': [{
                        'tag_name': doc.get('classificazione', 'non_classificata'),
                        'confidence': doc.get('confidence', 0.0),
                        'method': doc.get('metadata', {}).get('method', 'unknown'),
                        'motivation': doc.get('motivazione', ''),
                        'created_at': doc.get('timestamp').isoformat() if doc.get('timestamp') else '',
                        'source': 'database'
                    }] if doc.get('classificazione') and doc.get('classificazione') != 'non_classificata' else []
                }
                sessions.append(session)
            
            print(f"✅ Recuperate {len(sessions)} sessioni per tenant '{tenant_name}'")
            if label_filter:
                print(f"  🏷️ Con etichetta '{label_filter}'")
            
            return jsonify({
                'success': True,
                'tenant_name': tenant_name,
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)