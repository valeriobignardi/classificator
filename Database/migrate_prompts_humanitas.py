#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
SCRIPT DI MIGRAZIONE PROMPT HUMANITAS AL DATABASE
=====================================================================
Autore: Sistema di Classificazione AI
Data: 2025-08-21
Descrizione: Migra tutti i prompt esistenti di Humanitas dal codice hardcoded
             al database TAG.prompts mantenendo variabili dinamiche
=====================================================================
"""

import yaml
import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime
import os
from config_loader import load_config

class PromptMigrator:
    """
    Migra prompt esistenti dal codice al database TAG.prompts
    """
    
    def __init__(self):
        """Inizializza il migrator"""
        self.config = self._load_config()
        self.connection = None
        
        # Tenant Humanitas (dal database)
        self.humanitas_tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
        self.humanitas_tenant_name = "Humanitas"
        
    def _load_config(self):
        """Carica configurazione"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        return load_config()
    
    def connect(self):
        """Connessione al database TAG locale"""
        try:
            db_config = self.config['tag_database']
            self.connection = mysql.connector.connect(
                host=db_config['host'],
                port=db_config['port'], 
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database']
            )
            print("✅ Connesso al database TAG locale")
            return True
        except Error as e:
            print(f"❌ Errore connessione: {e}")
            return False
    
    def disconnect(self):
        """Disconnette dal database"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
    
    def migrate_all_prompts(self):
        """
        Migra tutti i prompt di Humanitas identificati nell'analisi
        """
        print("🚀 AVVIO MIGRAZIONE PROMPT HUMANITAS")
        print("="*60)
        
        if not self.connect():
            return False
        
        # 1. PROMPT LLM - SYSTEM MESSAGE (IntelligentClassifier)  
        self._migrate_llm_system_prompt()
        
        # 2. PROMPT LLM - USER TEMPLATE (IntelligentClassifier)
        self._migrate_llm_user_template()
        
        # 3. PROMPT FINE-TUNING - SYSTEM MESSAGE (MistralFineTuningManager)
        self._migrate_finetuning_system_prompt()
        
        # 4. PROMPT FINE-TUNING - SPECIALIZED SYSTEM (MistralFineTuningManager)
        self._migrate_finetuning_specialized_prompt()
        
        # 5. PROMPT FINE-TUNING - MODELFILE TEMPLATE
        self._migrate_finetuning_modelfile_template()
        
        self.disconnect()
        print("\\n✅ MIGRAZIONE COMPLETATA!")
        return True
    
    def _insert_prompt(self, engine, prompt_type, prompt_name, content, 
                      dynamic_variables=None, config_parameters=None, description=None):
        """
        Inserisce un prompt nel database
        """
        cursor = self.connection.cursor()
        
        try:
            query = """
            INSERT INTO prompts (
                tenant_id, tenant_name, engine, prompt_type, prompt_name,
                prompt_content, dynamic_variables, config_parameters, 
                description, created_by, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                self.humanitas_tenant_id,
                self.humanitas_tenant_name, 
                engine,
                prompt_type,
                prompt_name,
                content,
                json.dumps(dynamic_variables) if dynamic_variables else None,
                json.dumps(config_parameters) if config_parameters else None,
                description,
                'migration_script',
                datetime.now()
            )
            
            cursor.execute(query, values)
            self.connection.commit()
            
            print(f"✅ Migrato: {engine}/{prompt_type}/{prompt_name}")
            
        except Error as e:
            print(f"❌ Errore migrazione {prompt_name}: {e}")
        finally:
            cursor.close()
    
    def _migrate_llm_system_prompt(self):
        """
        Migra il system prompt dell'IntelligentClassifier
        """
        print("\\n📝 MIGRAZIONE: LLM System Prompt (IntelligentClassifier)")
        
        content = """Sei un classificatore esperto per l'ospedale {{tenant_name}} con profonda conoscenza del dominio sanitario.

MISSIONE: Classifica conversazioni con pazienti/utenti in base al loro intento principale.

APPROCCIO ANALITICO:
1. Identifica l'intento principale (non dettagli secondari)
2. Considera il contesto ospedaliero (prenotazioni, referti, informazioni)
3. Distingui richieste operative da richieste informative
4. Se incerto tra 2 etichette, scegli la più specifica

CONFIDENCE GUIDELINES:
- 0.9-1.0: Intento chiarissimo e inequivocabile
- 0.7-0.8: Intento probabile con piccole ambiguità
- 0.5-0.6: Intento possibile ma con dubbi ragionevoli  
- 0.3-0.4: Molto incerto, probabilmente "altro"
- 0.0-0.2: Impossibile classificare

ETICHETTE DISPONIBILI:
{{available_labels}}

{{priority_labels}}
{{context_guidance}}

OUTPUT FORMAT (SOLO JSON):
{{"predicted_label": "etichetta_precisa", "confidence": 0.X, "motivation": "ragionamento_breve"}}

CRITICAL: Genera ESCLUSIVAMENTE JSON valido. Zero testo aggiuntivo."""

        dynamic_variables = {
            "tenant_name": {
                "type": "string",
                "source": "database",
                "query": "SELECT tenant_name FROM tenants WHERE tenant_id = %tenant_id%",
                "description": "Nome del tenant corrente"
            },
            "available_labels": {
                "type": "text",
                "source": "function",
                "function": "_get_available_labels",
                "description": "Elenco etichette caricate dinamicamente dal database TAG.tag_definitions"
            },
            "priority_labels": {
                "type": "text", 
                "source": "function",
                "function": "_get_priority_labels_hint",
                "description": "Suggerimenti su etichette più frequenti per prioritizzazione"
            },
            "context_guidance": {
                "type": "text",
                "source": "parameter",
                "parameter": "conversation_context", 
                "description": "Contesto specifico analizzato dal conversation text",
                "optional": True
            }
        }
        
        config_parameters = {
            "temperature": 0.1,
            "max_tokens": 150,
            "model_selection": "auto"  # usa il modello configurato per il tenant
        }
        
        description = "System prompt principale per classificazione LLM tramite IntelligentClassifier. Include variabili dinamiche per etichette, contesto e tenant-specific guidance."
        
        self._insert_prompt(
            engine='LLM',
            prompt_type='SYSTEM', 
            prompt_name='intelligent_classifier_system',
            content=content,
            dynamic_variables=dynamic_variables,
            config_parameters=config_parameters,
            description=description
        )
    
    def _migrate_llm_user_template(self):
        """
        Migra il template per user message dell'IntelligentClassifier
        """
        print("📝 MIGRAZIONE: LLM User Template (IntelligentClassifier)")
        
        content = """Analizza questo testo seguendo l'approccio degli esempi:
{{examples_text}}
{{context_section}}

TESTO DA CLASSIFICARE:
"{{processed_text}}"

RAGIONA STEP-BY-STEP:
1. Identifica l'intento principale
2. Confronta con gli esempi
3. Scegli l'etichetta più appropriata
4. Valuta la tua certezza

OUTPUT (SOLO JSON):"""

        dynamic_variables = {
            "examples_text": {
                "type": "text",
                "source": "function",
                "function": "_get_dynamic_examples",
                "parameters": ["conversation_text", "max_examples"],
                "description": "Esempi dinamici selezionati semanticamente (curati + reali dal database)"
            },
            "context_section": {
                "type": "text",
                "source": "parameter", 
                "parameter": "context",
                "description": "Sezione di contesto aggiuntivo opzionale",
                "optional": True
            },
            "processed_text": {
                "type": "text",
                "source": "function",
                "function": "_summarize_if_long", 
                "parameters": ["conversation_text", "max_length"],
                "description": "Testo della conversazione processato (riassunto se troppo lungo)"
            }
        }
        
        config_parameters = {
            "max_examples": 5,
            "max_text_length": 300,
            "use_semantic_selection": True,
            "use_real_examples": True
        }
        
        description = "Template per user message con esempi dinamici e reasoning chain. Supporta selezione semantica di esempi e processamento intelligente del testo."
        
        self._insert_prompt(
            engine='LLM',
            prompt_type='TEMPLATE',
            prompt_name='intelligent_classifier_user_template', 
            content=content,
            dynamic_variables=dynamic_variables,
            config_parameters=config_parameters,
            description=description
        )
    
    def _migrate_finetuning_system_prompt(self):
        """
        Migra il system prompt per fine-tuning di MistralFineTuningManager
        """
        print("📝 MIGRAZIONE: Fine-Tuning System Prompt")
        
        content = """Sei un classificatore esperto per l'ospedale {{tenant_name}} specializzato nella comprensione di conversazioni con pazienti.

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

{{etichette_section}}

IMPORTANTE: Usa ESATTAMENTE i nomi delle etichette sopra elencate."""

        dynamic_variables = {
            "tenant_name": {
                "type": "string",
                "source": "database",
                "query": "SELECT tenant_name FROM tenants WHERE tenant_id = %tenant_id%",
                "description": "Nome del tenant per fine-tuning"
            },
            "etichette_section": {
                "type": "text",
                "source": "function",
                "function": "_get_tags_with_descriptions",
                "description": "Sezione etichette con descrizioni dettagliate caricate dal database"
            }
        }
        
        config_parameters = {
            "use_tag_descriptions": True,
            "format": "jsonl_chatml",
            "training_type": "supervised"
        }
        
        description = "System prompt per training di modelli fine-tuned. Include descrizioni dettagliate delle etichette e guidance specifica per il processo di apprendimento."
        
        self._insert_prompt(
            engine='FINETUNING',
            prompt_type='SYSTEM',
            prompt_name='mistral_finetuning_system',
            content=content,
            dynamic_variables=dynamic_variables,
            config_parameters=config_parameters,
            description=description
        )
    
    def _migrate_finetuning_specialized_prompt(self):
        """
        Migra il system prompt specializzato per modelli fine-tuned
        """
        print("📝 MIGRAZIONE: Fine-Tuning Specialized System Prompt")
        
        content = """Sei un classificatore esperto SPECIALIZZATO per {{tenant_name}}.

MISSIONE: Classifica conversazioni in base ai pattern specifici di questo cliente.

HAI IMPARATO DA {{training_examples_count}} ESEMPI REALI di questo cliente, conosci i loro pattern linguistici specifici.

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

{{specialized_examples}}

ETICHETTE SPECIALIZZATE: {{specialized_labels}}

OUTPUT (SOLO JSON): {{"predicted_label": "etichetta", "confidence": 0.X, "motivation": "ragionamento_specializzato"}}

CRITICAL: Genera SOLO JSON valido, nessun testo aggiuntivo."""

        dynamic_variables = {
            "tenant_name": {
                "type": "string",
                "source": "database",
                "query": "SELECT tenant_name FROM tenants WHERE tenant_id = %tenant_id%",
                "description": "Nome del tenant specializzato"
            },
            "training_examples_count": {
                "type": "integer",
                "source": "parameter",
                "parameter": "len(training_examples)",
                "description": "Numero di esempi di training utilizzati"
            },
            "specialized_examples": {
                "type": "text",
                "source": "function", 
                "function": "_build_specialized_examples_section",
                "parameters": ["training_examples"],
                "description": "Sezione esempi estratti dal training dataset specifico del cliente"
            },
            "specialized_labels": {
                "type": "text",
                "source": "function",
                "function": "_extract_specialized_labels",
                "parameters": ["training_examples"],
                "description": "Lista etichette specializzate basate sui training data del cliente"
            }
        }
        
        config_parameters = {
            "max_examples_per_label": 2,
            "use_client_patterns": True,
            "specialization_level": "high"
        }
        
        description = "System prompt per modelli fine-tuned specializzati per cliente specifico. Incorpora esempi e pattern appresi dal training dataset del cliente."
        
        self._insert_prompt(
            engine='FINETUNING',
            prompt_type='SPECIALIZED',
            prompt_name='mistral_specialized_system',
            content=content,
            dynamic_variables=dynamic_variables,
            config_parameters=config_parameters,
            description=description
        )
    
    def _migrate_finetuning_modelfile_template(self):
        """
        Migra il template Modelfile per Ollama
        """
        print("📝 MIGRAZIONE: Fine-Tuning Modelfile Template")
        
        content = """# Fine-tuned model per {{tenant_name}} - {{model_name}}
FROM {{base_model}}

# System message specializzato con esempi di training reali del cliente
SYSTEM \"\"\"{{specialized_system_message}}\"\"\"

# Template per classificazione guidata
TEMPLATE \"\"\"<|system|>
{{{{ .System }}}}

<|user|>
{{{{ .Prompt }}}}

<|assistant|>
\"\"\"

# Parametri ottimizzati per classificazione
PARAMETER temperature {{temperature}}
PARAMETER top_p {{top_p}} 
PARAMETER top_k {{top_k}}
PARAMETER repeat_penalty {{repeat_penalty}}"""

        dynamic_variables = {
            "tenant_name": {
                "type": "string",
                "source": "database",
                "query": "SELECT tenant_name FROM tenants WHERE tenant_id = %tenant_id%",
                "description": "Nome del tenant per il modello"
            },
            "model_name": {
                "type": "string",
                "source": "function",
                "function": "generate_model_name",
                "parameters": ["tenant_name", "tenant_id", "model_type", "timestamp"],
                "description": "Nome generato per il modello fine-tuned"
            },
            "base_model": {
                "type": "string",
                "source": "config",
                "config_path": "llm.finetuning.base_model",
                "default": "mistral:7b",
                "description": "Modello base per fine-tuning"
            },
            "specialized_system_message": {
                "type": "text",
                "source": "function",
                "function": "_build_specialized_system_message",
                "parameters": ["training_examples"],
                "description": "System message specializzato generato dinamicamente"
            },
            "temperature": {
                "type": "float",
                "source": "config", 
                "config_path": "llm.generation.temperature",
                "default": 0.1,
                "description": "Parametro temperatura del modello"
            },
            "top_p": {
                "type": "float",
                "source": "config",
                "config_path": "llm.generation.top_p", 
                "default": 0.9,
                "description": "Parametro top_p del modello"
            },
            "top_k": {
                "type": "integer",
                "source": "config",
                "config_path": "llm.generation.top_k",
                "default": 40,
                "description": "Parametro top_k del modello"
            },
            "repeat_penalty": {
                "type": "float",
                "source": "config",
                "config_path": "llm.generation.repeat_penalty",
                "default": 1.1,
                "description": "Parametro repeat_penalty del modello"
            }
        }
        
        config_parameters = {
            "format": "modelfile",
            "target": "ollama",
            "supports_system": True,
            "supports_template": True
        }
        
        description = "Template Modelfile per creazione modelli fine-tuned in Ollama. Include configurazione parametri e system message specializzato per il tenant."
        
        self._insert_prompt(
            engine='FINETUNING', 
            prompt_type='TEMPLATE',
            prompt_name='ollama_modelfile_template',
            content=content,
            dynamic_variables=dynamic_variables,
            config_parameters=config_parameters,
            description=description
        )

if __name__ == "__main__":
    migrator = PromptMigrator()
    migrator.migrate_all_prompts()
