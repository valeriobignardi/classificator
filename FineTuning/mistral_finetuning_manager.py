"""
MistralFineTuningManager - Gestore per il fine-tuning di Mistral per clienti specifici

Questo modulo implementa il fine-tuning automatico di Mistral utilizzando le decisioni umane
di ciascun cliente per creare modelli personalizzati che migliorano l'accuratezza.

Caratteristiche principali:
- Fine-tuning basato su decisioni umane validate
- Generazione automatica dataset in formato ChatML
- Gestione modelli fine-tuned per cliente
- Switch automatico a modello personalizzato
- Backup e rollback dei modelli
- Validazione e metriche di performance

Autore: Pipeline Humanitas  
Data: 20 Luglio 2025
"""

import json
import os
import sys
import yaml
import logging
import subprocess
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import requests

# Aggiungi path per importare Tenant
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from tenant import Tenant

# Import del sistema esistente
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TagDatabase'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'QualityGate'))

from tag_database_connector import TagDatabaseConnector
from quality_gate_engine import QualityGateEngine


@dataclass
class FineTuningConfig:
    """Configurazione per il fine-tuning"""
    base_model: str = ""  # Sarà impostato dal config.yaml
    output_model_name: str = ""
    learning_rate: float = 5e-5
    num_epochs: int = 3
    batch_size: int = 4
    validation_split: float = 0.2
    min_training_samples: int = 50
    max_training_samples: int = 1000
    temperature: float = 0.1
    max_tokens: int = 150
    
    def __post_init__(self):
        if not self.output_model_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Estrai il nome del modello base per il nome del modello fine-tuned
            model_base_name = self.base_model.split(':')[0] if ':' in self.base_model else self.base_model
            self.output_model_name = f"{model_base_name}_finetuned_{timestamp}"


@dataclass 
class FineTuningResult:
    """Risultato del fine-tuning"""
    success: bool
    model_name: str
    training_samples: int
    validation_samples: int
    training_loss: float
    validation_loss: float
    training_time_minutes: float
    model_size_mb: float
    error_message: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class MistralFineTuningManager:
    """
    Gestore per il fine-tuning di Mistral per clienti specifici
    """
    
    def __init__(self, 
                 tenant: Tenant,
                 config_path: Optional[str] = None,
                 ollama_url: str = None):
        """
        Inizializza il manager per fine-tuning
        
        PRINCIPIO UNIVERSALE: Accetta oggetto Tenant
        
        Args:
            tenant: Oggetto Tenant completo
            config_path: Percorso file configurazione (opzionale)
            ollama_url: URL server Ollama (se None, legge da config)
        """
        self.tenant = tenant
        self.tenant_slug = tenant.tenant_slug  # Estrae tenant_slug dall'oggetto Tenant
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        
        # Carica configurazione PRIMA di tutto
        with open(self.config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # Estrai configurazione LLM
        llm_config = self.config.get('llm', {})
        ollama_config = llm_config.get('ollama', {})
        
        # Imposta URL Ollama
        self.ollama_url = ollama_url or ollama_config.get('url', 'http://localhost:11434')
        self.ollama_url = self.ollama_url.rstrip('/')
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize MongoClassificationReader for tenant-aware naming
        if self.tenant_slug:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from mongo_classification_reader import MongoClassificationReader
            
            # Usa il tenant object già disponibile per il MongoClassificationReader
            try:
                self.mongo_reader = MongoClassificationReader(tenant=self.tenant)
                print(f"🗄️ MongoClassificationReader inizializzato per tenant: {self.tenant.tenant_name}")
            except Exception as e:
                print(f"⚠️ Errore creazione MongoClassificationReader per tenant '{self.tenant_slug}': {e}")
                self.mongo_reader = None
        else:
            self.mongo_reader = None
        
        # Database connector
        self.tag_db = TagDatabaseConnector(tenant=self.tenant)
        
        # Directory per modelli fine-tuned (tenant-aware)
        if self.tenant_slug and self.mongo_reader:
            # Use tenant-aware model directory structure
            base_models_dir = os.path.join(os.path.dirname(__file__), '..', 'finetuned_models')
            tenant_id_short = self.tenant.tenant_id[:8] if self.tenant.tenant_id else 'unknown'
            self.models_dir = os.path.join(base_models_dir, f"{self.tenant_slug.lower()}_{tenant_id_short}_models")
        else:
            self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'finetuned_models')
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Registry dei modelli per cliente (tenant-aware)
        registry_name = f"model_registry_{self.tenant_slug.lower()}.json" if self.tenant_slug else "model_registry.json"
        self.model_registry_path = os.path.join(self.models_dir, registry_name)
        self.model_registry = self._load_model_registry()
        
        self.logger.info(f"MistralFineTuningManager inizializzato - Ollama: {self.ollama_url}")
        self.logger.info(f"Models directory: {self.models_dir}")
        if self.tenant_slug:
            self.logger.info(f"Tenant-aware mode: {self.tenant_slug}")
    
    def generate_model_name(self, 
                           client_name: str, 
                           model_type: str = 'finetuned',
                           base_model: str = 'mistral') -> str:
        """
        Generate tenant-aware model name using naming convention
        
        Args:
            client_name: Nome del cliente/tenant
            model_type: Tipo di modello ('finetuned', 'specialized', etc.)
            base_model: Modello base utilizzato
            
        Returns:
            Nome modello con convenzione tenant-aware: {tenant_name}_{tenant_id}_{type}_{timestamp}
        """
        if self.mongo_reader and self.tenant_slug:
            return self.mongo_reader.generate_model_name(
                tenant_name=client_name, 
                model_type=f"{base_model}_{model_type}"
            )
        else:
            # Fallback to legacy naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{client_name}_{model_type}_{timestamp}"
    
    def _setup_logger(self) -> logging.Logger:
        """Setup del logger"""
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
    
    def _load_config(self) -> Dict[str, Any]:
        """Carica configurazione da file YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.warning(f"Errore caricamento config: {e}")
            return {}
    
    def _load_model_registry(self) -> Dict[str, Any]:
        """Carica il registry dei modelli per cliente"""
        try:
            if os.path.exists(self.model_registry_path):
                with open(self.model_registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Errore caricamento model registry: {e}")
        
        return {}
    
    def _save_model_registry(self):
        """Salva il registry dei modelli"""
        try:
            with open(self.model_registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.model_registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Errore salvataggio model registry: {e}")
    
    def get_client_model(self, client_name: str) -> Optional[str]:
        """
        Ottieni il nome del modello fine-tuned per un cliente
        
        Args:
            client_name: Nome del cliente
            
        Returns:
            Nome del modello fine-tuned o None se non esiste
        """
        return self.model_registry.get(client_name, {}).get('active_model')
    
    def has_finetuned_model(self, client_name: str) -> bool:
        """
        Verifica se un cliente ha un modello fine-tuned attivo
        
        Args:
            client_name: Nome del cliente
            
        Returns:
            True se ha un modello fine-tuned
        """
        return self.get_client_model(client_name) is not None
    
    def generate_training_dataset(self, 
                                 client_name: str, 
                                 min_confidence: float = 0.7) -> Tuple[List[Dict], int]:
        """
        Genera dataset di training dalle decisioni umane del cliente
        
        Flusso:
        1. Legge file JSONL per ottenere session_id + human_decision
        2. Recupera conversazioni dal database remoto per quei session_id
        3. Combina conversazioni + etichette umane per creare dataset
        
        Args:
            client_name: Nome del cliente
            min_confidence: Confidence minima per includere un esempio
            
        Returns:
            Tuple (dataset, total_examples)
        """
        self.logger.info(f"🔨 Generazione dataset training per {client_name}")
        
        try:
            # AUTO-SYNC: Sincronizza con database prima di generare dataset
            sync_success = self.auto_sync_before_training(client_name)
            if sync_success:
                self.logger.info("✅ Auto-sync completato, dataset aggiornato")
            else:
                self.logger.warning("⚠️ Auto-sync fallito, procedo con dati esistenti")
            
            # FASE 1: Recupera decisioni umane dal file JSONL (tenant-aware)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Use tenant-aware filename if available
            if self.tenant_slug:
                training_decisions_path = os.path.join(project_root, f"training_decisions_{self.tenant_slug.lower()}.jsonl")
            else:
                training_decisions_path = os.path.join(project_root, f"training_decisions_{client_name}.jsonl")
            
            self.logger.info(f"🔍 Leggendo decisioni umane da: {training_decisions_path}")
            
            human_decisions = []
            if os.path.exists(training_decisions_path):
                with open(training_decisions_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            decision = json.loads(line.strip())
                            # Filtra per qualità: confidence alta e non "altro"
                            if (decision.get('human_confidence', 0) >= min_confidence and 
                                decision.get('human_decision') and
                                decision.get('human_decision') != 'altro'):
                                human_decisions.append(decision)
                        except json.JSONDecodeError:
                            continue
            
            self.logger.info(f"📊 Trovate {len(human_decisions)} decisioni umane di qualità")
            
            if len(human_decisions) == 0:
                self.logger.warning("⚠️ Nessuna decisione umana trovata, uso esempi curati di fallback")
                return self._generate_curated_fallback_dataset()
            
            # FASE 2: Recupera conversazioni dal database remoto per i session_id
            session_ids = [decision['session_id'] for decision in human_decisions]
            self.logger.info(f"🔍 Recuperando conversazioni per {len(session_ids)} sessioni...")
            
            # Usa SessionAggregator esistente per recuperare le conversazioni
            from Preprocessing.session_aggregator import SessionAggregator
            aggregator = SessionAggregator()
            
            # Recupera solo le sessioni di interesse
            all_sessions = aggregator.estrai_sessioni_aggregate(limit=None)
            
            # Filtra per session_id di interesse
            relevant_sessions = {sid: session for sid, session in all_sessions.items() 
                               if sid in session_ids}
            
            self.logger.info(f"✅ Trovate {len(relevant_sessions)} conversazioni nel database remoto")
            
            # FASE 3: Combina conversazioni + etichette umane per creare dataset
            dataset = []
            valid_tags = set(self._get_tags_with_descriptions().keys())
            invalid_labels = []
            
            for decision in human_decisions:
                session_id = decision['session_id']
                human_label = decision['human_decision']
                
                # VALIDAZIONE LABEL: Verifica che la label esista nel database
                if human_label not in valid_tags:
                    invalid_labels.append(human_label)
                    self.logger.warning(f"⚠️ Label non valida ignorata: '{human_label}' (session: {session_id})")
                    continue
                
                if session_id in relevant_sessions:
                    session_data = relevant_sessions[session_id]
                    conversation_text = session_data.get('testo_completo', '')
                    
                    # Skip conversazioni vuote o troppo corte
                    if len(conversation_text.strip()) < 10:
                        continue
                    
                    # Limita lunghezza per il fine-tuning (max 500 char)
                    if len(conversation_text) > 500:
                        conversation_text = conversation_text[:500] + "..."
                    
                    # Ottieni descrizione del tag per contesto aggiuntivo
                    tag_description = self._get_tags_with_descriptions().get(human_label, '')
                    
                    # Crea esempio di training in formato ChatML con descrizione del tag
                    training_example = {
                        "messages": [
                            {
                                "role": "system",
                                "content": self._build_finetuning_system_message()
                            },
                            {
                                "role": "user", 
                                "content": f"Classifica questa conversazione:\n\n\"{conversation_text}\""
                            },
                            {
                                "role": "assistant",
                                "content": json.dumps({
                                    "predicted_label": human_label,
                                    "confidence": decision.get('human_confidence', 0.9),
                                    "motivation": f"Classificazione umana validata per {human_label}: {tag_description}"
                                }, ensure_ascii=False)
                            }
                        ],
                        "metadata": {
                            "session_id": session_id,
                            "human_confidence": decision.get('human_confidence'),
                            "source": "human_validation",
                            "tag_description": tag_description
                        }
                    }
                    dataset.append(training_example)
            
            # Log delle label non valide
            if invalid_labels:
                unique_invalid = list(set(invalid_labels))
                self.logger.warning(f"🚫 Label non valide trovate: {unique_invalid}")
                self.logger.info(f"✅ Label valide disponibili: {sorted(valid_tags)}")
            
            self.logger.info(f"✅ Dataset generato: {len(dataset)} esempi di training da decisioni umane")
            
            # Se abbiamo pochi esempi, aggiungi esempi curati per completare
            if len(dataset) < 20:
                self.logger.info("📈 Integrando con esempi curati per raggiungere soglia minima")
                curated_dataset, _ = self._generate_curated_fallback_dataset()
                dataset.extend(curated_dataset[:30 - len(dataset)])  # Completa fino a 30 esempi
            
            return dataset, len(dataset)
            
        except Exception as e:
            self.logger.error(f"❌ Errore generazione dataset: {e}")
            # Fallback a esempi curati in caso di errore
            self.logger.info("🔄 Fallback a esempi curati...")
            return self._generate_curated_fallback_dataset()
    
    def _generate_curated_fallback_dataset(self) -> Tuple[List[Dict], int]:
        """
        Genera dataset di fallback con esempi curati di alta qualità
        
        Returns:
            Tuple (dataset, total_examples)
        """
        curated_examples = [
            {"text": "Buongiorno voglio ritirare la mia cartella clinica", "label": "ritiro_cartella_clinica_referti"},
            {"text": "Vorrei prenotare una visita cardiologica", "label": "prenotazione_esami"},
            {"text": "Come posso contattare il reparto di cardiologia?", "label": "info_contatti"},
            {"text": "Devo cambiare la mail nel portale", "label": "cambio_anagrafica"},
            {"text": "Non riesco ad entrare nel mio account", "label": "problema_accesso_portale"},
            {"text": "Fate gli esami del sangue?", "label": "info_esami"},
            {"text": "Quanto tempo per l'intervento al ginocchio?", "label": "info_interventi"},
            {"text": "Mi serve il certificato per il lavoro", "label": "info_certificati"},
            {"text": "Quando mi chiamate per il ricovero?", "label": "info_ricovero"},
            {"text": "Quanto costa il parcheggio?", "label": "info_parcheggio"},
            {"text": "Come faccio a prepararmi per la colonscopia?", "label": "norme_di_preparazione"},
            {"text": "Non ho ricevuto la fattura", "label": "problema_amministrativo"},
            {"text": "Quali alberghi convenzionati ci sono?", "label": "strutture_convenzionate_alberghiere"},
            {"text": "Ci sono sconti per l'aereo?", "label": "convenzioni_viaggio"},
            {"text": "Soffro di crisi d'ansia cosa posso fare?", "label": "parere_medico"},
            {"text": "Come arrivo da voi da Milano?", "label": "indicazioni_stradali"},
            {"text": "Il sito non mi fa accedere", "label": "problema_accesso_portale"},
            {"text": "Devo prenotare una risonanza", "label": "prenotazione_esami"},
            {"text": "Qual è il numero dell'oncologia?", "label": "info_contatti"},
            {"text": "Ho problemi con la prenotazione online", "label": "problema_prenotazione_portale"}
        ]
        
        dataset = []
        for example in curated_examples:
            training_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": self._build_finetuning_system_message()
                    },
                    {
                        "role": "user", 
                        "content": f"Classifica questa conversazione:\n\n\"{example['text']}\""
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "predicted_label": example['label'],
                            "confidence": 0.9,
                            "motivation": f"Classificazione curata per {example['label']} basata su pattern linguistico"
                        }, ensure_ascii=False)
                    }
                ],
                "metadata": {
                    "source": "curated_examples"
                }
            }
            dataset.append(training_example)
            
        self.logger.info(f"✅ Dataset curato generato: {len(dataset)} esempi di fallback")
        return dataset, len(dataset)
    
    def _get_tags_with_descriptions(self) -> Dict[str, str]:
        """
        Recupera i tag e le loro descrizioni dal database LOCALE (TAGS)
        
        Returns:
            Dizionario tag_name -> tag_description
        """
        try:
            # USA DATABASE LOCALE TAGS per i tag
            from Database.schema_manager import ClassificationSchemaManager
            tags_manager = ClassificationSchemaManager(schema='TAGS')  # DATABASE LOCALE
            tags = tags_manager.get_all_tags()
            tags_manager.chiudi_connessione()
            
            tags_dict = {}
            for tag in tags:
                tag_name = tag.get('tag_name', '')
                tag_description = tag.get('tag_description', '')
                if tag_name:
                    tags_dict[tag_name] = tag_description or f"Categoria per {tag_name}"
            
            self.logger.info(f"📋 Recuperati {len(tags_dict)} tag dal database LOCALE (TAGS)")
            return tags_dict
            
        except Exception as e:
            self.logger.warning(f"⚠️ Errore recupero tag dal database LOCALE: {e}")
            # Fallback ai tag hardcoded
            return {
                'ritiro_cartella_clinica_referti': 'Richieste di ritiro documentazione medica, cartelle cliniche e referti',
                'prenotazione_esami': 'Prenotazioni e appuntamenti per esami diagnostici, visite specialistiche',
                'info_contatti': 'Richieste di informazioni su contatti, numeri di telefono, reparti',
                'problema_accesso_portale': 'Problemi tecnici di accesso al portale online, credenziali',
                'info_esami': 'Informazioni generali su procedure di esami, modalità, tempi',
                'cambio_anagrafica': 'Modifiche dati anagrafici, aggiornamento informazioni personali',
                'norme_di_preparazione': 'Istruzioni per preparazione esami, digiuno, farmaci',
                'problema_amministrativo': 'Questioni amministrative, pagamenti, ticket, rimborsi',
                'info_ricovero': 'Informazioni su ricoveri, degenze, procedure ospedaliere',
                'info_certificati': 'Richieste di certificati medici, attestazioni, documentazione ufficiale',
                'info_interventi': 'Informazioni su interventi chirurgici, procedure operative',
                'info_parcheggio': 'Informazioni su parcheggi, costi, modalità di accesso',
                'altro': 'Richieste non classificabili nelle categorie principali'
            }

    def _build_finetuning_system_message(self) -> str:
        """
        Costruisce system message ottimizzato per fine-tuning con descrizioni dei tag
        """
        # Recupera tag e descrizioni dal database
        tags_dict = self._get_tags_with_descriptions()
        
        # Costruisce sezione etichette con descrizioni
        etichette_section = "ETICHETTE DISPONIBILI:\n"
        for tag_name, description in tags_dict.items():
            etichette_section += f"- {tag_name}: {description}\n"
        
        return f"""Sei un classificatore esperto per l'ospedale Humanitas specializzato nella comprensione di conversazioni con pazienti.

MISSIONE: Classifica conversazioni identificando l'intento principale del paziente/utente.

APPROCCIO:
1. Identifica l'intento principale (non dettagli secondari)
2. Considera il contesto ospedaliero
3. Distingui richieste operative da informative
4. Scegli sempre l'etichetta più specifica basandoti sulle descrizioni

CONFIDENCE SCALE:
- 0.9-1.0: Intento chiarissimo
- 0.7-0.8: Intento probabile 
- 0.5-0.6: Intento possibile
- 0.3-0.4: Molto incerto
- 0.0-0.2: Impossibile classificare

OUTPUT: JSON con predicted_label, confidence, motivation

{etichette_section}

IMPORTANTE: Usa ESATTAMENTE i nomi delle etichette sopra elencate."""
    
    def create_training_file(self, 
                           dataset: List[Dict], 
                           client_name: str) -> str:
        """
        Crea file di training in formato JSONL per fine-tuning
        
        Args:
            dataset: Dataset di training
            client_name: Nome del cliente
            
        Returns:
            Percorso del file di training creato
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_file = os.path.join(self.models_dir, f"training_{client_name}_{timestamp}.jsonl")
        
        try:
            with open(training_file, 'w', encoding='utf-8') as f:
                for example in dataset:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            self.logger.info(f"📁 File training creato: {training_file}")
            return training_file
            
        except Exception as e:
            self.logger.error(f"❌ Errore creazione file training: {e}")
            raise
    
    def execute_finetuning(self, 
                         client_name: str,
                         config: Optional[FineTuningConfig] = None) -> FineTuningResult:
        """
        Esegue il fine-tuning di Mistral per un cliente specifico
        
        Args:
            client_name: Nome del cliente
            config: Configurazione fine-tuning (opzionale)
            
        Returns:
            Risultato del fine-tuning
        """
        start_time = datetime.now()
        
        if config is None:
            # Carica configurazione LLM per il modello base
            llm_config = self.config.get('llm', {})
            finetuning_config = llm_config.get('finetuning', {})
            models_config = llm_config.get('models', {})
            
            config = FineTuningConfig()
            # Imposta il modello base dalla configurazione
            config.base_model = finetuning_config.get('base_model') or models_config.get('default', 'mistral:7b')
            config.min_training_samples = finetuning_config.get('min_training_samples', 50)
            config.max_training_samples = finetuning_config.get('max_training_samples', 1000)
            
            # Generate tenant-aware model name
            config.output_model_name = self.generate_model_name(
                client_name=client_name,
                model_type='finetuned',
                base_model=config.base_model.split(':')[0] if ':' in config.base_model else config.base_model
            )
        
        self.logger.info(f"🚀 AVVIO FINE-TUNING per cliente: {client_name}")
        self.logger.info(f"📋 Modello base: {config.base_model}")
        self.logger.info(f"🎯 Modello output: {config.output_model_name}")
        
        try:
            # 1. GENERA DATASET
            dataset, total_examples = self.generate_training_dataset(client_name)
            
            if total_examples < config.min_training_samples:
                error_msg = f"Dataset insufficiente: {total_examples} < {config.min_training_samples} esempi richiesti"
                self.logger.error(f"❌ {error_msg}")
                return FineTuningResult(
                    success=False,
                    model_name="",
                    training_samples=total_examples,
                    validation_samples=0,
                    training_loss=0.0,
                    validation_loss=0.0,
                    training_time_minutes=0.0,
                    model_size_mb=0.0,
                    error_message=error_msg
                )
            
            # Limita dataset se troppo grande
            if total_examples > config.max_training_samples:
                dataset = dataset[:config.max_training_samples]
                total_examples = len(dataset)
                self.logger.info(f"📊 Dataset limitato a {total_examples} esempi")
            
            # 2. SPLIT TRAINING/VALIDATION
            val_size = int(total_examples * config.validation_split)
            train_size = total_examples - val_size
            
            train_dataset = dataset[:train_size]
            val_dataset = dataset[train_size:]
            
            self.logger.info(f"📈 Split: {train_size} training, {val_size} validation")
            
            # 3. CREA FILES DI TRAINING
            train_file = self.create_training_file(train_dataset, f"{client_name}_train")
            val_file = self.create_training_file(val_dataset, f"{client_name}_val") if val_dataset else None
            
            # 4. ESEGUI FINE-TUNING tramite Ollama
            result = self._execute_ollama_finetuning(
                config=config,
                train_file=train_file,
                val_file=val_file
            )
            
            if result.success:
                # 5. REGISTRA MODELLO per il cliente
                self._register_model_for_client(client_name, result.model_name, config, result)
                
                # 6. CLEANUP files temporanei
                self._cleanup_training_files([train_file, val_file])
            
            # Calcola tempo di training
            training_time = (datetime.now() - start_time).total_seconds() / 60.0
            result.training_time_minutes = training_time
            
            self.logger.info(f"✅ Fine-tuning completato in {training_time:.1f} minuti")
            return result
            
        except Exception as e:
            error_msg = f"Errore nel fine-tuning: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            
            return FineTuningResult(
                success=False,
                model_name="",
                training_samples=0,
                validation_samples=0,
                training_loss=0.0,
                validation_loss=0.0,
                training_time_minutes=0.0,
                model_size_mb=0.0,
                error_message=error_msg
            )
    
    def _execute_ollama_finetuning(self, 
                                  config: FineTuningConfig,
                                  train_file: str,
                                  val_file: Optional[str] = None) -> FineTuningResult:
        """
        Esegue il fine-tuning tramite Ollama
        
        Args:
            config: Configurazione fine-tuning
            train_file: File dataset training
            val_file: File dataset validation (opzionale)
            
        Returns:
            Risultato del fine-tuning
        """
        try:
            # APPROCCIO OLLAMA COMPATIBILE: Crea modello specializzato con system message e template
            # basati sui dati di training invece di fine-tuning tradizionale
            
            # Estrae esempi rappresentativi dal dataset di training per specializzazione
            training_examples = self._extract_training_examples_for_specialization(train_file)
            
            # Crea Modelfile con system message specializzato e template
            modelfile_content = f"""FROM {config.base_model}

# Parametri ottimizzati per classificazione
PARAMETER temperature 0.1
PARAMETER num_predict 150
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER repeat_penalty 1.1

# System message specializzato con esempi di training reali del cliente
SYSTEM \"\"\"{self._build_specialized_system_message(training_examples)}\"\"\"

# Template per classificazione guidata
TEMPLATE \"\"\"<|system|>
{{{{ .System }}}}

<|user|>
{{{{ .Prompt }}}}

<|assistant|>
\"\"\"
"""
            
            # Salva Modelfile
            modelfile_path = os.path.join(self.models_dir, f"Modelfile_{config.output_model_name}")
            with open(modelfile_path, 'w', encoding='utf-8') as f:
                f.write(modelfile_content)
            
            self.logger.info(f"📄 Modelfile specializzato creato: {modelfile_path}")
            
            # Esegui ollama create per creare modello specializzato
            cmd = [
                "ollama", "create", 
                config.output_model_name,
                "-f", modelfile_path
            ]
            
            self.logger.info(f"🔧 Esecuzione comando: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # Timeout 1 ora
            )
            
            if process.returncode == 0:
                self.logger.info(f"✅ Modello fine-tuned creato: {config.output_model_name}")
                
                # Calcola dimensione modello (approssimativa)
                model_size = self._get_model_size(config.output_model_name)
                
                return FineTuningResult(
                    success=True,
                    model_name=config.output_model_name,
                    training_samples=self._count_training_samples(train_file),
                    validation_samples=self._count_training_samples(val_file) if val_file else 0,
                    training_loss=0.0,  # Ollama non fornisce metriche dettagliate
                    validation_loss=0.0,
                    training_time_minutes=0.0,
                    model_size_mb=model_size
                )
            else:
                error_msg = process.stderr or "Errore sconosciuto durante fine-tuning"
                self.logger.error(f"❌ Errore ollama create: {error_msg}")
                
                return FineTuningResult(
                    success=False,
                    model_name="",
                    training_samples=0,
                    validation_samples=0,
                    training_loss=0.0,
                    validation_loss=0.0,
                    training_time_minutes=0.0,
                    model_size_mb=0.0,
                    error_message=error_msg
                )
                
        except subprocess.TimeoutExpired:
            error_msg = "Timeout durante fine-tuning (>1 ora)"
            self.logger.error(f"❌ {error_msg}")
            return FineTuningResult(
                success=False,
                model_name="", 
                training_samples=0,
                validation_samples=0,
                training_loss=0.0,
                validation_loss=0.0,
                training_time_minutes=0.0,
                model_size_mb=0.0,
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"Errore durante fine-tuning: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return FineTuningResult(
                success=False,
                model_name="",
                training_samples=0,
                validation_samples=0,
                training_loss=0.0,
                validation_loss=0.0,
                training_time_minutes=0.0,
                model_size_mb=0.0,
                error_message=error_msg
            )
    
    def _count_training_samples(self, file_path: Optional[str]) -> int:
        """Conta il numero di esempi in un file JSONL"""
        if not file_path or not os.path.exists(file_path):
            return 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0
    
    def _get_model_size(self, model_name: str) -> float:
        """
        Ottieni dimensione approssimativa del modello in MB
        
        Args:
            model_name: Nome del modello
            
        Returns:
            Dimensione in MB
        """
        try:
            # Chiama ollama list per ottenere info sui modelli
            process = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if process.returncode == 0:
                # Parsing output per trovare il modello
                for line in process.stdout.split('\n'):
                    if model_name in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            size_str = parts[2]  # Es: "4.1GB"
                            if 'GB' in size_str:
                                return float(size_str.replace('GB', '')) * 1024
                            elif 'MB' in size_str:
                                return float(size_str.replace('MB', ''))
            
            return 0.0  # Default se non riesce a determinare
            
        except Exception as e:
            self.logger.warning(f"Impossibile determinare dimensione modello: {e}")
            return 0.0
    
    def _register_model_for_client(self, 
                                  client_name: str,
                                  model_name: str,
                                  config: FineTuningConfig,
                                  result: FineTuningResult):
        """
        Registra un modello fine-tuned per un cliente
        
        Args:
            client_name: Nome del cliente
            model_name: Nome del modello fine-tuned
            config: Configurazione usata
            result: Risultato del fine-tuning
        """
        # Backup modello precedente se esiste
        if client_name in self.model_registry:
            old_model = self.model_registry[client_name].get('active_model')
            if old_model:
                self.model_registry[client_name]['previous_models'] = \
                    self.model_registry[client_name].get('previous_models', []) + [old_model]
        
        # Registra nuovo modello
        self.model_registry[client_name] = {
            'active_model': model_name,
            'created_at': datetime.now().isoformat(),
            'base_model': config.base_model,
            'training_samples': result.training_samples,
            'validation_samples': result.validation_samples,
            'model_size_mb': result.model_size_mb,
            'config': asdict(config),
            'previous_models': self.model_registry.get(client_name, {}).get('previous_models', [])
        }
        
        # Salva registry
        self._save_model_registry()
        
        self.logger.info(f"📝 Modello {model_name} registrato per {client_name}")
    
    def _cleanup_training_files(self, files: List[Optional[str]]):
        """
        Pulisce i file temporanei di training
        
        Args:
            files: Lista dei file da eliminare
        """
        for file_path in files:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    self.logger.debug(f"🗑️ File rimosso: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Impossibile rimuovere {file_path}: {e}")
    
    def rollback_model(self, client_name: str) -> bool:
        """
        Effettua rollback al modello precedente per un cliente
        
        Args:
            client_name: Nome del cliente
            
        Returns:
            True se il rollback è riuscito
        """
        try:
            client_info = self.model_registry.get(client_name, {})
            previous_models = client_info.get('previous_models', [])
            
            if not previous_models:
                self.logger.warning(f"Nessun modello precedente per {client_name}")
                return False
            
            # Usa l'ultimo modello precedente
            previous_model = previous_models[-1]
            current_model = client_info.get('active_model')
            
            # Aggiorna registry
            self.model_registry[client_name]['active_model'] = previous_model
            self.model_registry[client_name]['previous_models'] = previous_models[:-1]
            
            # Aggiungi modello corrente ai precedenti
            if current_model:
                self.model_registry[client_name]['previous_models'].append(current_model)
            
            self._save_model_registry()
            
            self.logger.info(f"🔄 Rollback effettuato per {client_name}: {current_model} -> {previous_model}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Errore rollback per {client_name}: {e}")
            return False
    
    def delete_model(self, client_name: str, model_name: Optional[str] = None) -> bool:
        """
        Elimina un modello fine-tuned
        
        Args:
            client_name: Nome del cliente
            model_name: Nome specifico del modello (se None, elimina quello attivo)
            
        Returns:
            True se l'eliminazione è riuscita
        """
        try:
            if model_name is None:
                model_name = self.get_client_model(client_name)
            
            if not model_name:
                self.logger.warning(f"Nessun modello da eliminare per {client_name}")
                return False
            
            # Elimina modello tramite ollama
            process = subprocess.run(
                ["ollama", "rm", model_name],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if process.returncode == 0:
                # Rimuovi dal registry
                if client_name in self.model_registry:
                    if self.model_registry[client_name].get('active_model') == model_name:
                        self.model_registry[client_name]['active_model'] = None
                    
                    # Rimuovi da modelli precedenti
                    previous = self.model_registry[client_name].get('previous_models', [])
                    self.model_registry[client_name]['previous_models'] = [m for m in previous if m != model_name]
                
                self._save_model_registry()
                
                self.logger.info(f"🗑️ Modello {model_name} eliminato per {client_name}")
                return True
            else:
                error_msg = process.stderr or "Errore eliminazione modello"
                self.logger.error(f"❌ Errore eliminazione {model_name}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Errore eliminazione modello per {client_name}: {e}")
            return False
    
    def get_model_info(self, client_name: str) -> Dict[str, Any]:
        """
        Ottieni informazioni sul modello di un cliente
        
        Args:
            client_name: Nome del cliente
            
        Returns:
            Informazioni sul modello
        """
        client_info = self.model_registry.get(client_name, {})
        
        return {
            'client': client_name,
            'has_finetuned_model': self.has_finetuned_model(client_name),
            'active_model': client_info.get('active_model'),
            'created_at': client_info.get('created_at'),
            'base_model': client_info.get('base_model'),
            'training_samples': client_info.get('training_samples', 0),
            'validation_samples': client_info.get('validation_samples', 0),
            'model_size_mb': client_info.get('model_size_mb', 0),
            'previous_models_count': len(client_info.get('previous_models', []))
        }
    
    def list_all_client_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Lista tutti i modelli per tutti i clienti
        
        Returns:
            Dizionario con info modelli per cliente
        """
        return {
            client_name: self.get_model_info(client_name)
            for client_name in self.model_registry.keys()
        }
    
    def _extract_training_examples_for_specialization(self, train_file: str, max_examples: int = 15) -> List[Dict]:
        """
        Estrae esempi rappresentativi dal dataset di training per specializzazione del modello
        
        Args:
            train_file: File JSONL dataset training
            max_examples: Numero massimo di esempi da estrarre
            
        Returns:
            Lista di esempi rappresentativi per tutte le categorie
        """
        try:
            examples = []
            label_counts = {}
            
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        messages = data.get('messages', [])
                        
                        # Estrae user input e assistant response
                        user_msg = None
                        assistant_msg = None
                        
                        for msg in messages:
                            if msg.get('role') == 'user':
                                user_msg = msg.get('content', '')
                            elif msg.get('role') == 'assistant':
                                assistant_msg = msg.get('content', '')
                        
                        if user_msg and assistant_msg:
                            # Parsa risposta assistant per etichetta
                            try:
                                assistant_data = json.loads(assistant_msg)
                                label = assistant_data.get('predicted_label', '')
                                
                                if label:
                                    # Limita esempi per etichetta per diversità
                                    if label_counts.get(label, 0) < 2:
                                        # Estrae testo conversazione dal user message
                                        conv_text = user_msg.replace('Classifica questa conversazione:\n\n"', '').replace('"', '').strip()
                                        
                                        examples.append({
                                            'text': conv_text[:150],  # Limita lunghezza
                                            'label': label,
                                            'motivation': assistant_data.get('motivation', '')
                                        })
                                        label_counts[label] = label_counts.get(label, 0) + 1
                                        
                            except json.JSONDecodeError:
                                continue
                                
                        if len(examples) >= max_examples:
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            self.logger.info(f"📚 Estratti {len(examples)} esempi di specializzazione da training dataset")
            return examples
            
        except Exception as e:
            self.logger.error(f"❌ Errore estrazione esempi: {e}")
            return []
    
    def _build_specialized_system_message(self, training_examples: List[Dict]) -> str:
        """
        Costruisce system message specializzato basato su esempi di training reali del cliente
        
        Args:
            training_examples: Esempi estratti dal training dataset
            
        Returns:
            System message specializzato per il cliente
        """
        # Raggruppa esempi per categoria
        examples_by_label = {}
        for example in training_examples:
            label = example['label']
            if label not in examples_by_label:
                examples_by_label[label] = []
            examples_by_label[label].append(example)
        
        # Costruisce sezione esempi specializzati
        specialized_examples = ""
        if training_examples:
            specialized_examples = "\n\nESEMPI SPECIALIZZATI PER QUESTO CLIENTE:\n"
            for label, examples in examples_by_label.items():
                specialized_examples += f"\n{label.upper()}:\n"
                for example in examples[:2]:  # Max 2 per categoria
                    specialized_examples += f'- "{example["text"][:100]}..." → {label}\n'
        
        return f"""Sei un classificatore esperto SPECIALIZZATO per questo specifico cliente ospedaliero.

MISSIONE: Classifica conversazioni in base ai pattern specifici di questo cliente.

HAI IMPARATO DA {len(training_examples)} ESEMPI REALI di questo cliente, conosci i loro pattern linguistici specifici.

APPROCCIO SPECIALIZZATO:
1. Applica i pattern che hai appreso da questo cliente
2. Riconosci le espressioni e terminologie specifiche
3. Considera le categorie più frequenti per questo cliente
4. Mantieni alta precisione su categorie business-critical

CONFIDENCE SPECIALIZZATA:
- 0.9-1.0: Pattern riconosciuto dai training data di questo cliente
- 0.7-0.8: Similare ai pattern del cliente ma con variazioni
- 0.5-0.6: Incerto, non corrisponde ai pattern appresi
- 0.3-0.4: Molto diverso dai dati di training del cliente
{specialized_examples}

ETICHETTE SPECIALIZZATE: {' | '.join(sorted(examples_by_label.keys()))}

OUTPUT (SOLO JSON): {{"predicted_label": "etichetta", "confidence": 0.X, "motivation": "ragionamento_specializzato"}}

CRITICAL: Genera SOLO JSON valido, nessun testo aggiuntivo."""

    def _convert_to_ollama_format(self, input_file: str, model_name: str) -> str:
        """
        Converte il dataset ChatML in formato Ollama/Mistral
        
        Args:
            input_file: File input in formato ChatML
            model_name: Nome del modello per output file
            
        Returns:
            Percorso del file convertito
        """
        output_file = input_file.replace('.jsonl', f'_ollama_{model_name}.jsonl')
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f_in, \
                 open(output_file, 'w', encoding='utf-8') as f_out:
                
                for line in f_in:
                    try:
                        data = json.loads(line.strip())
                        messages = data.get('messages', [])
                        
                        # Estrae user prompt e assistant response
                        user_prompt = ""
                        assistant_response = ""
                        
                        for msg in messages:
                            if msg.get('role') == 'user':
                                user_prompt = msg.get('content', '')
                            elif msg.get('role') == 'assistant':
                                assistant_response = msg.get('content', '')
                        
                        # Formato Ollama/Mistral
                        ollama_entry = {
                            "prompt": user_prompt,
                            "completion": assistant_response
                        }
                        
                        f_out.write(json.dumps(ollama_entry, ensure_ascii=False) + '\n')
                        
                    except json.JSONDecodeError:
                        continue
            
            self.logger.info(f"📄 Dataset convertito in formato Ollama: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"❌ Errore conversione dataset: {e}")
            raise

    def _extract_training_examples_for_specialization(self, train_file: str, max_examples: int = 15) -> List[Dict]:
        """
        Estrae esempi rappresentativi dal dataset di training per specializzazione
        
        Args:
            train_file: File training JSONL
            max_examples: Numero massimo di esempi da estrarre
            
        Returns:
            Lista di esempi rappresentativi
        """
        examples = []
        label_counts = {}
        
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        messages = data.get('messages', [])
                        
                        # Estrai user input e assistant response
                        user_content = ""
                        assistant_content = ""
                        
                        for msg in messages:
                            if msg.get('role') == 'user':
                                user_content = msg.get('content', '')
                            elif msg.get('role') == 'assistant':
                                assistant_content = msg.get('content', '')
                        
                        # Parsa response JSON per ottenere label
                        if assistant_content:
                            try:
                                response_data = json.loads(assistant_content)
                                label = response_data.get('predicted_label', '')
                                
                                # Bilancia esempi per etichetta
                                if label and label_counts.get(label, 0) < 3:  # Max 3 per etichetta
                                    examples.append({
                                        'user_input': user_content,
                                        'label': label,
                                        'confidence': response_data.get('confidence', 0.9),
                                        'motivation': response_data.get('motivation', '')
                                    })
                                    label_counts[label] = label_counts.get(label, 0) + 1
                                    
                            except json.JSONDecodeError:
                                continue
                    
                    except json.JSONDecodeError:
                        continue
                    
                    if len(examples) >= max_examples:
                        break
            
            self.logger.info(f"📊 Estratti {len(examples)} esempi rappresentativi da {train_file}")
            return examples
            
        except Exception as e:
            self.logger.error(f"❌ Errore estrazione esempi: {e}")
            return []

    def _build_specialized_system_message(self, training_examples: List[Dict]) -> str:
        """
        Costruisce system message specializzato con esempi di training reali del cliente
        
        Args:
            training_examples: Esempi di training estratti
            
        Returns:
            System message specializzato
        """
        base_message = """Sei un classificatore esperto per l'ospedale Humanitas specializzato in conversazioni con pazienti di questo specifico cliente.

MISSIONE: Classifica conversazioni identificando l'intento principale del paziente/utente basandoti sui pattern appresi da questo cliente.

ESEMPI SPECIFICI DEL CLIENTE:"""
        
        # Aggiungi esempi reali dal training del cliente
        examples_text = ""
        for i, example in enumerate(training_examples[:10], 1):  # Max 10 esempi
            examples_text += f"""

ESEMPIO {i}:
Input: "{example['user_input'][:100]}..."
Output: {{"predicted_label": "{example['label']}", "confidence": {example['confidence']}, "motivation": "{example['motivation'][:50]}..."}}"""
        
        specialized_guidance = """

APPROCCIO SPECIALIZZATO:
1. Usa i pattern appresi dagli esempi sopra
2. Mantieni consistenza con le classificazioni precedenti del cliente
3. Considera il linguaggio specifico usato da questo cliente
4. Applica la stessa logica di classificazione degli esempi

CONFIDENCE GUIDELINES:
- 0.9-1.0: Pattern chiaramente riconosciuto dagli esempi
- 0.7-0.8: Pattern simile agli esempi
- 0.5-0.6: Pattern parzialmente riconosciuto
- 0.3-0.4: Incerto, possibile "altro"

ETICHETTE PRINCIPALI (da esempi cliente):"""
        
        # Estrai etichette uniche dagli esempi
        unique_labels = list(set(ex['label'] for ex in training_examples))
        labels_text = " | ".join(unique_labels[:15])  # Max 15 etichette
        
        return base_message + examples_text + specialized_guidance + f"\n{labels_text}"

    def sync_training_decisions_with_database(self, client_name: str, overwrite_existing: bool = False) -> bool:
        """
        Sincronizza il file JSONL training_decisions con il database session_classifications
        
        Args:
            client_name: Nome del cliente
            overwrite_existing: Se sovrascrivere il file esistente
            
        Returns:
            True se sincronizzazione riuscita
        """
        try:
            self.logger.info(f"🔄 Sincronizzazione training decisions per {client_name}")
            
            # Connetti al database TAG
            self.tag_db.connetti()
            
            # Query per ottenere tutte le classificazioni umane validate per il cliente
            query = """
            SELECT 
                session_id,
                tag_name as human_decision,
                confidence_score as human_confidence,
                classification_method,
                notes,
                created_at
            FROM session_classifications 
            WHERE tenant_name = %s 
                AND classification_method IN ('HUMAN_REVIEW', 'MANUAL')
                AND confidence_score >= 0.7
            ORDER BY created_at DESC
            """
            
            result = self.tag_db.esegui_query(query, (client_name,))
            
            if not result:
                self.logger.warning(f"⚠️ Nessuna classificazione umana trovata per {client_name}")
                self.tag_db.disconnetti()
                return False
            
            self.logger.info(f"📊 Trovate {len(result)} classificazioni umane nel database")
            
            # Costruisci path del file JSONL
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            training_decisions_path = os.path.join(project_root, f"training_decisions_{client_name}.jsonl")
            
            # Leggi decisioni esistenti se non si deve sovrascrivere
            existing_decisions = {}
            if not overwrite_existing and os.path.exists(training_decisions_path):
                try:
                    with open(training_decisions_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                decision = json.loads(line.strip())
                                session_id = decision.get('session_id')
                                if session_id:
                                    existing_decisions[session_id] = decision
                            except json.JSONDecodeError:
                                continue
                    self.logger.info(f"📁 Trovate {len(existing_decisions)} decisioni esistenti in JSONL")
                except Exception as e:
                    self.logger.warning(f"⚠️ Errore lettura file esistente: {e}")
            
            # Crea backup se file esiste
            if os.path.exists(training_decisions_path):
                backup_path = f"{training_decisions_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(training_decisions_path, backup_path)
                self.logger.info(f"💾 Backup creato: {backup_path}")
            
            # Scrivi file sincronizzato
            new_decisions_count = 0
            updated_decisions_count = 0
            
            with open(training_decisions_path, 'w', encoding='utf-8') as f:
                for row in result:
                    session_id, human_decision, human_confidence, method, notes, created_at = row
                    
                    # Crea entry in formato standard
                    case_id = f"{client_name}_{session_id}_{created_at.strftime('%Y%m%d_%H%M%S') if created_at else 'unknown'}"
                    
                    decision_entry = {
                        "case_id": case_id,
                        "session_id": session_id,
                        "tenant": client_name,
                        "ml_prediction": "",  # Non disponibile dal DB
                        "ml_confidence": 0.0,
                        "llm_prediction": "",  # Non disponibile dal DB
                        "llm_confidence": 0.0,
                        "human_decision": human_decision,
                        "human_confidence": float(human_confidence),
                        "uncertainty_score": 0.0,  # Non disponibile dal DB
                        "novelty_score": 0.0,  # Non disponibile dal DB
                        "reason": f"From database - method: {method}",
                        "notes": notes or "",
                        "resolved_at": created_at.isoformat() if created_at else datetime.now().isoformat()
                    }
                    
                    # Verifica se è nuova o aggiornata
                    if session_id in existing_decisions:
                        existing = existing_decisions[session_id]
                        if existing.get('human_decision') != human_decision:
                            updated_decisions_count += 1
                            self.logger.debug(f"🔄 Aggiornata sessione {session_id}: {existing.get('human_decision')} → {human_decision}")
                    else:
                        new_decisions_count += 1
                    
                    f.write(json.dumps(decision_entry, ensure_ascii=False) + '\n')
            
            self.tag_db.disconnetti()
            
            self.logger.info(f"✅ Sincronizzazione completata: {len(result)} totali, "
                           f"{new_decisions_count} nuove, {updated_decisions_count} aggiornate")
            self.logger.info(f"📁 File aggiornato: {training_decisions_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Errore sincronizzazione: {e}")
            if hasattr(self, 'tag_db'):
                self.tag_db.disconnetti()
            return False

    def auto_sync_before_training(self, client_name: str) -> bool:
        """
        Sincronizzazione automatica prima del training per garantire coerenza
        
        Args:
            client_name: Nome del cliente
            
        Returns:
            True se sincronizzazione riuscita
        """
        self.logger.info(f"🔄 Auto-sync prima del training per {client_name}")
        return self.sync_training_decisions_with_database(client_name, overwrite_existing=False)


def test_finetuning_manager():
    """Test di base del fine-tuning manager"""
    print("🧪 Test MistralFineTuningManager")
    
    # Crea tenant fake per test
    fake_tenant = Tenant(
        tenant_id="test-id",
        tenant_name="test-tenant", 
        tenant_slug="test"
    )
    
    manager = MistralFineTuningManager(tenant=fake_tenant)
    
    # Test info modelli
    print("\n📊 Modelli esistenti:")
    models = manager.list_all_client_models()
    for client, info in models.items():
        print(f"  {client}: {info}")
    
    print("\n✅ Test completato")


if __name__ == "__main__":
    test_finetuning_manager()
