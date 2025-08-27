#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
PROMPT MANAGER - GESTIONE PROMPT MULTI-TENANT DA DATABASE
=====================================================================
Autore: Sistema di Classificazione AI
Data: 2025-08-21
Descrizione: Classe per gestione prompt con variabili dinamiche
             caricati dal database TAG.prompts
=====================================================================
"""

import yaml
import mysql.connector
from mysql.connector import Error
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import logging

# Importazione classe Tenant per eliminare conversioni ridondanti
try:
    from Utils.tenant import Tenant
    TENANT_AVAILABLE = True
except ImportError:
    TENANT_AVAILABLE = False
    print("⚠️ PROMPT MANAGER: Classe Tenant non disponibile, uso retrocompatibilità")

class PromptManager:
    """
    Gestore centralizzato per prompt multi-tenant con variabili dinamiche
    """
    
    def __init__(self, config_path: str = None):
        """
        Inizializza il PromptManager
        
        Args:
            config_path: Percorso file configurazione (default: config.yaml nella root)
        """
        # Usa percorso relativo alla root del progetto
        if config_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(project_root, 'config.yaml')
        
        self.config_path = config_path
        self.config = self._load_config()
        self.connection = None
        self._cache = {}  # Cache per prompt caricati
        self._cache_ttl = 300  # TTL cache in secondi (5 minuti)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Import delle funzioni per variabili dinamiche (lazy loading)
        self._classification_functions = None
        self._finetuning_functions = None
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger per il PromptManager"""
        logger = logging.getLogger(f"PromptManager.{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - PromptManager - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self) -> Dict:
        """Carica configurazione"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Errore caricamento config: {e}")
    
    def connect(self) -> bool:
        """
        Connette al database TAG
        
        Returns:
            True se connessione riuscita, False altrimenti
        """
        try:
            if self.connection and self.connection.is_connected():
                return True
                
            db_config = self.config['tag_database']
            self.connection = mysql.connector.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database']
            )
            
            self.logger.debug("✅ Connesso al database TAG")
            return True
            
        except Error as e:
            self.logger.error(f"❌ Errore connessione database TAG: {e}")
            return False
    
    def disconnect(self):
        """Disconnette dal database"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
    
    def validate_tenant_prompts_strict(self, tenant_id: str, required_prompts: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Validazione STRICT dei prompt obbligatori per un tenant
        
        Args:
            tenant_id: ID del tenant
            required_prompts: Lista di dict con 'engine', 'prompt_type', 'prompt_name'
            
        Returns:
            Dict con:
            - 'valid': bool - True se tutti i prompt sono presenti
            - 'missing_prompts': List - Lista prompt mancanti
            - 'errors': List - Lista errori riscontrati
            
        Raises:
            Exception: Se tenant non ha prompt obbligatori configurati
        """
        try:
            validation_result = {
                'valid': True,
                'missing_prompts': [],
                'errors': []
            }
            
            self.logger.info(f"🔍 Validazione STRICT prompt per tenant: {tenant_id}")
            
            if not self.connect():
                raise Exception(f"Impossibile connettersi al database per validare prompt tenant {tenant_id}")
            
            for prompt_req in required_prompts:
                engine = prompt_req['engine']
                prompt_type = prompt_req['prompt_type']
                prompt_name = prompt_req['prompt_name']
                
                # Cache key per il prompt
                cache_key = f"{tenant_id}:{engine}:{prompt_type}:{prompt_name}"
                
                # Controlla esistenza del prompt
                prompt_data = self._load_prompt_from_db(tenant_id, engine, prompt_type, prompt_name)
                
                if not prompt_data:
                    missing_prompt = {
                        'tenant_id': tenant_id,
                        'engine': engine,
                        'prompt_type': prompt_type,
                        'prompt_name': prompt_name,
                        'cache_key': cache_key
                    }
                    validation_result['missing_prompts'].append(missing_prompt)
                    validation_result['valid'] = False
                    
                    error_msg = f"Prompt OBBLIGATORIO mancante: {cache_key}"
                    validation_result['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
            
            if not validation_result['valid']:
                total_missing = len(validation_result['missing_prompts'])
                raise Exception(
                    f"Tenant {tenant_id} ha {total_missing} prompt obbligatori mancanti. "
                    f"Configurazione richiesta prima di procedere."
                )
            
            self.logger.info(f"✅ Validazione STRICT completata per tenant {tenant_id}: tutti i prompt obbligatori sono presenti")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Errore validazione STRICT prompt tenant {tenant_id}: {e}")
            raise e
    
    def get_prompt_strict(self, 
                         tenant_id: str,
                         engine: str, 
                         prompt_type: str,
                         prompt_name: str,
                         variables: Dict[str, Any] = None) -> str:
        """
        Recupera prompt con validazione STRICT (nessun fallback)
        
        Args:
            tenant_id: ID del tenant
            engine: Tipo di engine ('LLM', 'ML', 'FINETUNING')
            prompt_type: Tipo di prompt ('SYSTEM', 'USER', 'TEMPLATE', 'SPECIALIZED')
            prompt_name: Nome identificativo del prompt
            variables: Variabili aggiuntive da passare per la sostituzione
            
        Returns:
            Prompt processato con variabili sostituite
            
        Raises:
            Exception: Se prompt non trovato o non disponibile
        """
        try:
            cache_key = f"{tenant_id}:{engine}:{prompt_type}:{prompt_name}"
            self.logger.debug(f"🔍 Caricamento STRICT prompt: {cache_key}")
            
            # Utilizza il metodo standard ma con controllo strict
            prompt_content = self.get_prompt(tenant_id, engine, prompt_type, prompt_name, variables)
            
            if prompt_content is None:
                raise Exception(
                    f"Prompt OBBLIGATORIO non trovato o non disponibile: {cache_key}. "
                    f"Configurazione richiesta per il tenant {tenant_id}."
                )
            
            self.logger.debug(f"✅ Prompt STRICT caricato con successo: {cache_key}")
            return prompt_content
            
        except Exception as e:
            self.logger.error(f"❌ Errore caricamento STRICT prompt {cache_key}: {e}")
            raise e
    
    def get_prompt(self, 
                   tenant_or_id,
                   engine: str, 
                   prompt_type: str,
                   prompt_name: str,
                   variables: Dict[str, Any] = None) -> Optional[str]:
        """
        Recupera e processa un prompt con variabili dinamiche
        
        Args:
            tenant_or_id: Oggetto Tenant o tenant_id (slug/UUID) per compatibilità
            engine: Tipo di engine ('LLM', 'ML', 'FINETUNING')
            prompt_type: Tipo di prompt ('SYSTEM', 'USER', 'TEMPLATE', 'SPECIALIZED')
            prompt_name: Nome identificativo del prompt
            variables: Variabili aggiuntive da passare per la sostituzione
            
        Returns:
            Prompt processato con variabili sostituite, o None se non trovato
        """
        try:
            # Gestione compatibilità Tenant vs tenant_id string
            if TENANT_AVAILABLE and hasattr(tenant_or_id, 'tenant_id'):
                # Oggetto Tenant - usa direttamente i suoi dati
                tenant = tenant_or_id
                resolved_tenant_id = tenant.tenant_id
            else:
                # Retrocompatibilità: tenant_id string - normalizza
                resolved_tenant_id = self._resolve_tenant_id(tenant_or_id)
            
            # Cache key per il prompt
            cache_key = f"{resolved_tenant_id}:{engine}:{prompt_type}:{prompt_name}"
            
            # Controlla cache
            if self._is_cached_valid(cache_key):
                prompt_data = self._cache[cache_key]['data']
            else:
                # Carica dal database (usa tenant_id GIÀ RISOLTO)
                prompt_data = self._load_prompt_from_db_resolved(resolved_tenant_id, engine, prompt_type, prompt_name)
                if not prompt_data:
                    self.logger.warning(f"Prompt non trovato: {cache_key}")
                    return None
                
                # Cache il risultato
                self._cache[cache_key] = {
                    'data': prompt_data,
                    'timestamp': datetime.now()
                }
            
            # Processa le variabili dinamiche (usa tenant_id GIÀ RISOLTO)
            processed_content = self._process_dynamic_variables(
                prompt_data['content'],
                prompt_data['dynamic_variables'],
                resolved_tenant_id,
                variables or {}
            )
            
            self.logger.debug(f"✅ Prompt processato: {cache_key}")
            return processed_content
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero prompt {cache_key}: {e}")
            return None
    
    def _load_prompt_from_db(self, tenant_or_id, engine: str, 
                           prompt_type: str, prompt_name: str) -> Optional[Dict]:
        """
        Carica prompt dal database
        
        Args:
            tenant_or_id: Oggetto Tenant o tenant_id per compatibilità
        
        Returns:
            Dict con 'content', 'dynamic_variables', 'config_parameters' o None
        """
        if not self.connect():
            return None
        
        try:
            # Gestione compatibilità Tenant vs tenant_id string
            if TENANT_AVAILABLE and hasattr(tenant_or_id, 'tenant_id'):
                # Oggetto Tenant - usa direttamente i suoi dati
                tenant = tenant_or_id
                resolved_tenant_id = tenant.tenant_id
            else:
                # Retrocompatibilità: tenant_id string - normalizza
                resolved_tenant_id = self._resolve_tenant_id(tenant_or_id)
            
            cursor = self.connection.cursor()
            
            query = """
            SELECT prompt_content, dynamic_variables, config_parameters
            FROM prompts 
            WHERE tenant_id = %s 
                AND engine = %s 
                AND prompt_type = %s 
                AND prompt_name = %s 
                AND is_active = TRUE
            ORDER BY version DESC
            LIMIT 1
            """
            
            cursor.execute(query, (resolved_tenant_id, engine, prompt_type, prompt_name))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                content, dynamic_vars_json, config_params_json = result
                return {
                    'content': content,
                    'dynamic_variables': json.loads(dynamic_vars_json) if dynamic_vars_json else {},
                    'config_parameters': json.loads(config_params_json) if config_params_json else {}
                }
            
            return None
            
        except Error as e:
            self.logger.error(f"❌ Errore query prompt: {e}")
            return None
    
    def _load_prompt_from_db_resolved(self, resolved_tenant_id: str, engine: str, 
                           prompt_type: str, prompt_name: str) -> Optional[Dict]:
        """
        Carica prompt dal database con tenant_id GIÀ RISOLTO
        
        Returns:
            Dict con 'content', 'dynamic_variables', 'config_parameters' o None
        """
        if not self.connect():
            return None
        
        try:
            cursor = self.connection.cursor()
            
            query = """
            SELECT prompt_content, dynamic_variables, config_parameters
            FROM prompts 
            WHERE tenant_id = %s 
                AND engine = %s 
                AND prompt_type = %s 
                AND prompt_name = %s 
                AND is_active = TRUE
            ORDER BY version DESC
            LIMIT 1
            """
            
            cursor.execute(query, (resolved_tenant_id, engine, prompt_type, prompt_name))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                content, dynamic_vars_json, config_params_json = result
                return {
                    'content': content,
                    'dynamic_variables': json.loads(dynamic_vars_json) if dynamic_vars_json else {},
                    'config_parameters': json.loads(config_params_json) if config_params_json else {}
                }
            
            return None
            
        except Error as e:
            self.logger.error(f"❌ Errore query prompt: {e}")
            return None
    
    def _process_dynamic_variables(self, 
                                 content: str,
                                 dynamic_vars: Dict,
                                 tenant_id: str,
                                 runtime_variables: Dict[str, Any]) -> str:
        """
        Processa e sostituisce le variabili dinamiche nel content
        
        Args:
            content: Contenuto del prompt con placeholder {{variabile}}
            dynamic_vars: Configurazione variabili dinamiche 
            tenant_id: ID tenant per query database
            runtime_variables: Variabili passate a runtime
            
        Returns:
            Contenuto con variabili sostituite
        """
        processed_content = content
        
        # Pattern per trovare placeholder {{variabile}} - CORRETTO
        placeholder_pattern = r'\{\{\s*([^}]+)\s*\}\}'
        placeholders = re.findall(placeholder_pattern, content)
        
        # AGGIUNTA: Variabili di base per tutti i tenant
        base_variables = self._get_base_tenant_variables(tenant_id)
        
        for placeholder in placeholders:
            placeholder_clean = placeholder.strip()
            
            # Valore della variabile
            value = None
            
            # 1. Prima controlla le variabili di base (tenant_name, tenant_id, etc.)
            if placeholder_clean in base_variables:
                value = str(base_variables[placeholder_clean])
            
            # 2. Poi controlla le variabili runtime
            elif placeholder_clean in runtime_variables:
                value = str(runtime_variables[placeholder_clean])
            
            # 3. Poi controlla le variabili dinamiche configurate
            elif placeholder_clean in dynamic_vars:
                var_config = dynamic_vars[placeholder_clean]
                value = self._resolve_dynamic_variable(var_config, tenant_id, runtime_variables)
            
            # 4. Fallback: lascia il placeholder se non risolto
            if value is None:
                self.logger.warning(f"⚠️ Variabile non risolta: {placeholder_clean}")
                continue
            
            # Sostituisce il placeholder
            pattern = f"{{{{{placeholder}}}}}"
            processed_content = processed_content.replace(pattern, str(value))
        
        return processed_content
    
    def _get_base_tenant_variables(self, tenant_id: str) -> Dict[str, str]:
        """
        Recupera variabili di base per un tenant dal database
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Dict con variabili di base (tenant_name, tenant_id, etc.)
        """
        base_vars = {
            'tenant_id': tenant_id
        }
        
        try:
            if not self.connect():
                return base_vars
            
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT DISTINCT tenant_name
                FROM prompts 
                WHERE tenant_id = %s 
                LIMIT 1
            """, (tenant_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                tenant_name = result[0]
                base_vars.update({
                    'tenant_name': tenant_name,
                    'tenant_display_name': tenant_name.title(),
                    'tenant_upper': tenant_name.upper(),
                    'tenant_lower': tenant_name.lower()
                })
            
        except Error as e:
            self.logger.debug(f"⚠️ Errore recupero variabili base tenant: {e}")
        
        return base_vars
    
    def _get_tenant_name(self, tenant_id: str) -> str:
        """
        Recupera il nome del tenant dal database
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Nome del tenant o 'unknown' se non trovato
            
        Autore: Sistema di Classificazione AI
        Data: 2025-08-25
        Ultimo aggiornamento: 2025-08-25
        """
        try:
            if not self.connect():
                return 'unknown'
            
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT DISTINCT tenant_name
                FROM prompts 
                WHERE tenant_id = %s 
                LIMIT 1
            """, (tenant_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return result[0]
            else:
                return 'unknown'
                
        except Error as e:
            self.logger.debug(f"⚠️ Errore recupero nome tenant: {e}")
            return 'unknown'
    
    def _resolve_dynamic_variable(self, 
                                var_config: Dict,
                                tenant_id: str,
                                runtime_variables: Dict[str, Any]) -> Optional[str]:
        """
        Risolve una singola variabile dinamica in base alla sua configurazione
        
        Args:
            var_config: Configurazione della variabile dinamica
            tenant_id: ID del tenant
            runtime_variables: Variabili runtime disponibili
            
        Returns:
            Valore risolto della variabile o None
        """
        try:
            source = var_config.get('source')
            
            if source == 'database':
                return self._resolve_database_variable(var_config, tenant_id)
            
            elif source == 'function':
                return self._resolve_function_variable(var_config, runtime_variables)
            
            elif source == 'parameter':
                param_name = var_config.get('parameter')
                return runtime_variables.get(param_name)
            
            elif source == 'config':
                return self._resolve_config_variable(var_config)
            
            else:
                self.logger.warning(f"Tipo source non supportato: {source}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Errore risoluzione variabile: {e}")
            return None
    
    def _resolve_database_variable(self, var_config: Dict, tenant_id: str) -> Optional[str]:
        """
        Risolve variabile da query database
        """
        if not self.connect():
            return None
        
        try:
            query = var_config.get('query', '')
            # Sostituisce %tenant_id% con il tenant attuale
            query = query.replace('%tenant_id%', f"'{tenant_id}'")
            
            cursor = self.connection.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            cursor.close()
            
            return str(result[0]) if result else None
            
        except Error as e:
            self.logger.error(f"❌ Errore query database variable: {e}")
            return None
    
    def _resolve_function_variable(self, var_config: Dict, runtime_variables: Dict) -> Optional[str]:
        """
        Risolve variabile chiamando funzione specifica
        """
        function_name = var_config.get('function')
        parameters = var_config.get('parameters', [])
        
        try:
            # Import lazy delle funzioni di classificazione
            if not self._classification_functions:
                self._import_classification_functions()
            
            if function_name in self._classification_functions:
                func = self._classification_functions[function_name]
                
                # Prepara parametri per la funzione
                func_args = []
                for param in parameters:
                    if param in runtime_variables:
                        func_args.append(runtime_variables[param])
                    else:
                        self.logger.warning(f"Parametro mancante per {function_name}: {param}")
                
                # Chiama la funzione
                if func_args:
                    result = func(*func_args)
                else:
                    result = func()
                
                return str(result) if result is not None else None
            
            else:
                self.logger.warning(f"Funzione non trovata: {function_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Errore chiamata funzione {function_name}: {e}")
            return None
    
    def _resolve_config_variable(self, var_config: Dict) -> Optional[str]:
        """
        Risolve variabile da configurazione
        """
        try:
            config_path = var_config.get('config_path', '')
            default_value = var_config.get('default')
            
            # Naviga nel dizionario config usando il path (es: "llm.generation.temperature")
            value = self.config
            for key in config_path.split('.'):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return str(default_value) if default_value is not None else None
            
            return str(value)
            
        except Exception as e:
            self.logger.error(f"❌ Errore risoluzione config variable: {e}")
            return var_config.get('default')
    
    def _import_classification_functions(self):
        """
        Import lazy delle funzioni di classificazione necessarie per le variabili dinamiche
        """
        try:
            # SEMPLIFICAZIONE: Usa funzioni stub invece di import complessi
            self._classification_functions = {
                '_get_available_labels': self._stub_get_available_labels,
                '_get_priority_labels_hint': self._stub_get_priority_labels_hint,
                '_get_dynamic_examples': self._stub_get_dynamic_examples,
                '_summarize_if_long': self._stub_summarize_if_long,
            }
            
            self.logger.debug("✅ Stub functions caricate per variabili dinamiche")
            
        except Exception as e:
            self.logger.error(f"❌ Errore import classification functions: {e}")
            self._classification_functions = {}
    
    def _stub_get_available_labels(self) -> str:
        """Stub function per available_labels"""
        return "prenotazione_esami, ritiro_cartella_clinica_referti, info_contatti, altro"
    
    def _stub_get_priority_labels_hint(self) -> str:
        """Stub function per priority_labels"""
        return "Focus principale: prenotazione_esami (45%), ritiro_cartella_clinica_referti (35%)"
    
    def _stub_get_dynamic_examples(self, text: str = "") -> str:
        """Stub function per dynamic_examples"""
        return """ESEMPIO 1:
Input: "Vorrei prenotare una visita"
Output: prenotazione_esami
Motivazione: Richiesta diretta di prenotazione"""
    
    def _stub_summarize_if_long(self, text: str) -> str:
        """Stub function per summarize"""
        return text[:500] + "..." if len(text) > 500 else text
    
    def _is_cached_valid(self, cache_key: str) -> bool:
        """
        Controlla se il prompt in cache è ancora valido
        """
        if cache_key not in self._cache:
            return False
        
        cache_time = self._cache[cache_key]['timestamp']
        elapsed = (datetime.now() - cache_time).total_seconds()
        
        return elapsed < self._cache_ttl
    
    def clear_cache(self):
        """
        Svuota la cache dei prompt
        """
        self._cache.clear()
        self.logger.debug("🧹 Cache prompt svuotata")
    
    def update_prompt(self, 
                     tenant_id: str,
                     engine: str,
                     prompt_type: str,
                     prompt_name: str,
                     new_content: str,
                     updated_by: str = 'web_interface') -> bool:
        """
        Aggiorna un prompt nel database (per interfaccia web)
        
        Args:
            tenant_id: ID del tenant
            engine: Tipo di engine
            prompt_type: Tipo di prompt  
            prompt_name: Nome del prompt
            new_content: Nuovo contenuto
            updated_by: Chi ha fatto la modifica
            
        Returns:
            True se aggiornamento riuscito
        """
        if not self.connect():
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Prima salva la versione attuale nella history
            self._save_to_history(cursor, tenant_id, engine, prompt_type, prompt_name, 'UPDATE', updated_by)
            
            # Poi aggiorna il prompt
            update_query = """
            UPDATE prompts 
            SET prompt_content = %s, 
                updated_by = %s, 
                updated_at = %s,
                version = version + 1
            WHERE tenant_id = %s 
                AND engine = %s 
                AND prompt_type = %s 
                AND prompt_name = %s 
                AND is_active = TRUE
            """
            
            cursor.execute(update_query, (
                new_content, updated_by, datetime.now(),
                tenant_id, engine, prompt_type, prompt_name
            ))
            
            if cursor.rowcount > 0:
                self.connection.commit()
                
                # Invalida cache
                cache_key = f"{tenant_id}:{engine}:{prompt_type}:{prompt_name}"
                if cache_key in self._cache:
                    del self._cache[cache_key]
                
                self.logger.info(f"✅ Prompt aggiornato: {cache_key}")
                cursor.close()
                return True
            else:
                self.logger.warning(f"⚠️ Nessun prompt trovato per update: {tenant_id}:{engine}:{prompt_type}:{prompt_name}")
                cursor.close()
                return False
                
        except Error as e:
            self.logger.error(f"❌ Errore aggiornamento prompt: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def _save_to_history(self, cursor, tenant_id: str, engine: str, 
                        prompt_type: str, prompt_name: str, 
                        change_type: str, changed_by: str):
        """
        Salva la versione attuale del prompt nella tabella history prima della modifica
        """
        try:
            # Recupera il contenuto attuale
            select_query = """
            SELECT id, prompt_content FROM prompts 
            WHERE tenant_id = %s AND engine = %s AND prompt_type = %s 
                AND prompt_name = %s AND is_active = TRUE
            """
            cursor.execute(select_query, (tenant_id, engine, prompt_type, prompt_name))
            result = cursor.fetchone()
            
            if result:
                prompt_id, old_content = result
                
                # Inserisce nella history
                history_query = """
                INSERT INTO prompt_history 
                (prompt_id, tenant_id, old_content, change_type, changed_by)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(history_query, (prompt_id, tenant_id, old_content, change_type, changed_by))
                
        except Error as e:
            self.logger.error(f"❌ Errore salvataggio history: {e}")
    
    def list_prompts_for_tenant(self, tenant_or_id) -> List[Dict]:
        """
        Elenca tutti i prompt disponibili per un tenant
        
        Args:
            tenant_or_id: Oggetto Tenant o tenant_id per compatibilità
        
        Returns:
            Lista di dict con info sui prompt
        """
        if not self.connect():
            return []
        
        # Gestione compatibilità Tenant vs tenant_id string
        if TENANT_AVAILABLE and hasattr(tenant_or_id, 'tenant_id'):
            # Oggetto Tenant - usa direttamente i suoi dati
            tenant = tenant_or_id
            resolved_tenant_id = tenant.tenant_id
        else:
            # Retrocompatibilità: tenant_id string - normalizza
            resolved_tenant_id = self._resolve_tenant_id(tenant_or_id)
        
        try:
            cursor = self.connection.cursor()
            
            query = """
            SELECT engine, prompt_type, prompt_name, description,
                   SUBSTRING(prompt_content, 1, 100) as preview,
                   updated_at, version
            FROM prompts 
            WHERE tenant_id = %s AND is_active = TRUE
            ORDER BY engine, prompt_type, prompt_name
            """
            
            cursor.execute(query, (resolved_tenant_id,))
            results = cursor.fetchall()
            cursor.close()
            
            prompts = []
            for row in results:
                prompts.append({
                    'engine': row[0],
                    'prompt_type': row[1], 
                    'prompt_name': row[2],
                    'description': row[3],
                    'preview': row[4],
                    'updated_at': row[5],
                    'version': row[6]
                })
            
            return prompts
            
        except Error as e:
            self.logger.error(f"❌ Errore lista prompt: {e}")
            return []

    def _resolve_tenant_id(self, tenant_identifier: str) -> str:
        """
        Risolve un identificatore tenant (slug o id) nel tenant_id corretto.
        
        REGOLA: Tutte le query devono usare SOLO tenant_id per coerenza multi-tenant.
        tenant_slug e tenant_name servono solo per visualizzazione umana.
        
        Args:
            tenant_identifier: Può essere tenant_id completo o tenant_slug
            
        Returns:
            tenant_id corretto da usare nelle query
        """
        if not self.connection:
            self.connect()
        
        try:
            # Se sembra già un tenant_id completo (UUID format), usalo direttamente
            if len(tenant_identifier) > 20 and '-' in tenant_identifier:
                self.logger.debug(f"🎯 '{tenant_identifier}' sembra già un tenant_id UUID")
                return tenant_identifier
            
            # Altrimenti cerca per nome nella tabella tenants
            cursor = self.connection.cursor()
            query = """
            SELECT tenant_id, tenant_name 
            FROM tenants 
            WHERE tenant_name = %s AND is_active = 1
            LIMIT 1
            """
            
            cursor.execute(query, (tenant_identifier,))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                tenant_id, tenant_name = result
                self.logger.info(f"✅ Risolto '{tenant_identifier}' → tenant_id: {tenant_id} ({tenant_name})")
                return tenant_id
            else:
                self.logger.warning(f"⚠️ Tenant '{tenant_identifier}' non trovato in tenants table")
                # Fallback: restituisce l'identifier originale nel caso sia già corretto
                return tenant_identifier
                
        except Error as e:
            self.logger.error(f"❌ Errore risoluzione tenant_id per '{tenant_identifier}': {e}")
            # Fallback: restituisce l'identifier originale
            return tenant_identifier

    def get_all_prompts_for_tenant(self, tenant_or_id) -> List[Dict[str, Any]]:
        """
        Recupera tutti i prompt di un tenant con dettagli completi
        
        Args:
            tenant_id: ID del tenant (può essere tenant_slug o tenant_id completo)
            
        Returns:
            Lista completa di prompt con tutti i dettagli
        """
        if not self.connection:
            self.connect()
        
        # Gestione compatibilità Tenant vs tenant_id string
        if TENANT_AVAILABLE and hasattr(tenant_or_id, 'tenant_id'):
            # Oggetto Tenant - usa direttamente i suoi dati
            tenant = tenant_or_id
            resolved_tenant_id = tenant.tenant_id
            tenant_display = f"{tenant.tenant_name} ({resolved_tenant_id})"
        else:
            # Retrocompatibilità: tenant_id string - normalizza
            resolved_tenant_id = self._resolve_tenant_id(tenant_or_id)
            tenant_display = str(tenant_or_id)
        
        try:
            cursor = self.connection.cursor()
            
            query = """
            SELECT id, tenant_id, tenant_name, engine, prompt_type, prompt_name, 
                   prompt_content, dynamic_variables, is_active, created_at, updated_at
            FROM prompts 
            WHERE tenant_id = %s
            ORDER BY created_at DESC
            """
            
            cursor.execute(query, (resolved_tenant_id,))
            results = cursor.fetchall()
            cursor.close()
            
            prompts = []
            for row in results:
                prompts.append({
                    'id': row[0],
                    'tenant_id': row[1],
                    'tenant_name': row[2],
                    'engine': row[3],
                    'prompt_type': row[4],
                    'prompt_name': row[5],
                    'content': row[6],  # prompt_content
                    'variables': json.loads(row[7]) if row[7] else {},  # dynamic_variables
                    'is_active': bool(row[8]),
                    'created_at': row[9].isoformat() if row[9] else None,
                    'updated_at': row[10].isoformat() if row[10] else None
                })
            
            self.logger.info(f"📋 Recuperati {len(prompts)} prompt per tenant {resolved_tenant_id}")
            return prompts
            
        except Error as e:
            self.logger.error(f"❌ Errore recupero prompt per tenant {tenant_display}: {e}")
            return []

    def create_prompt(self,
                     tenant_id: int,
                     tenant_name: str,
                     prompt_type: str,
                     content: str,
                     variables: Dict[str, Any] = None,
                     is_active: bool = True) -> Optional[int]:
        """
        Crea un nuovo prompt nel database - Versione API compatibile
        
        Args:
            tenant_id: ID del tenant (intero)
            tenant_name: Nome del tenant
            prompt_type: Tipo di prompt specifico
            content: Contenuto del prompt
            variables: Variabili dinamiche (opzionale)
            is_active: Se il prompt è attivo
            
        Returns:
            ID del prompt creato, None se errore
        """
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            
            query = """
            INSERT INTO prompts (tenant_id, tenant_name, engine, prompt_type, prompt_name, prompt_content, dynamic_variables, is_active, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            """
            
            variables_json = json.dumps(variables) if variables else '{}'
            # Determina engine in base al prompt_type
            engine = 'LLM' if 'llm' in prompt_type.lower() else 'ML'
            
            cursor.execute(query, (
                tenant_id,
                tenant_name,
                engine,
                prompt_type,
                f"{prompt_type}_prompt",  # prompt_name derivato dal prompt_type
                content,  # prompt_content
                variables_json,  # dynamic_variables
                is_active
            ))
            
            self.connection.commit()
            prompt_id = cursor.lastrowid
            
            self.logger.info(f"✅ Creato prompt ID {prompt_id} per tenant {tenant_name}")
            return prompt_id
            
        except Error as e:
            self.logger.error(f"❌ Errore creazione prompt: {e}")
            if self.connection:
                self.connection.rollback()
            return None

    def create_prompt_legacy(self, 
                     tenant_id: str,
                     engine: str,
                     prompt_type: str,
                     prompt_name: str,
                     content: str,
                     variables: Dict[str, Any] = None,
                     is_active: bool = True) -> Optional[int]:
        """
        Crea un nuovo prompt nel database
        
        Args:
            tenant_id: ID del tenant
            engine: Tipo di engine (LLM, ML, FINETUNING)
            prompt_type: Tipo di prompt (SYSTEM, TEMPLATE, SPECIALIZED)
            prompt_name: Nome identificativo del prompt
            content: Contenuto del prompt
            variables: Variabili dinamiche (opzionale)
            is_active: Se il prompt è attivo
            
        Returns:
            ID del prompt creato, None se errore
        """
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            
            # Inserisci nuovo prompt
            query = """
            INSERT INTO prompts 
            (tenant_id, engine, prompt_type, prompt_name, content, variables, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                tenant_id,
                engine.upper(),
                prompt_type.upper(),
                prompt_name,
                content,
                json.dumps(variables) if variables else '{}',
                is_active
            )
            
            cursor.execute(query, values)
            prompt_id = cursor.lastrowid
            
            # Inserisci nella history
            history_query = """
            INSERT INTO prompt_history 
            (prompt_id, tenant_id, engine, prompt_type, prompt_name, content, variables, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(history_query, (prompt_id,) + values)
            
            self.connection.commit()
            cursor.close()
            
            self.logger.info(f"✅ Prompt creato: ID={prompt_id}, nome={prompt_name}")
            
            # Invalida cache
            self._invalidate_cache()
            
            return prompt_id
            
        except Error as e:
            self.logger.error(f"❌ Errore creazione prompt: {e}")
            if self.connection:
                self.connection.rollback()
            return None

    def update_prompt(self, 
                     prompt_id: int,
                     content: str = None,
                     variables: Dict[str, Any] = None,
                     is_active: bool = None) -> bool:
        """
        Aggiorna un prompt esistente
        
        Args:
            prompt_id: ID del prompt da aggiornare
            content: Nuovo contenuto (opzionale)
            variables: Nuove variabili (opzionale)
            is_active: Nuovo stato attivo (opzionale)
            
        Returns:
            True se aggiornamento riuscito, False altrimenti
        """
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            
            # Prepara i campi da aggiornare
            updates = []
            values = []
            
            if content is not None:
                updates.append("prompt_content = %s")
                values.append(content)
                
            if variables is not None:
                updates.append("dynamic_variables = %s")
                values.append(json.dumps(variables))
                
            if is_active is not None:
                updates.append("is_active = %s")
                values.append(is_active)
                
            if not updates:
                self.logger.warning(f"⚠️ Nessun campo da aggiornare per prompt {prompt_id}")
                return False
            
            # Aggiungi timestamp
            updates.append("updated_at = NOW()")
            values.append(prompt_id)
            
            # Esegui aggiornamento
            query = f"""
            UPDATE prompts 
            SET {', '.join(updates)}
            WHERE id = %s
            """
            
            result = cursor.execute(query, values)
            
            if cursor.rowcount > 0:
                # TODO: Implementare history se necessario
                self.connection.commit()
                cursor.close()
                
                self.logger.info(f"✅ Prompt {prompt_id} aggiornato")
                cursor.close()
                
                self.logger.info(f"✅ Prompt {prompt_id} aggiornato")
                
                # Invalida cache
                self._invalidate_cache()
                
                return True
            else:
                self.logger.warning(f"⚠️ Prompt {prompt_id} non trovato per aggiornamento")
                return False
                
        except Error as e:
            self.logger.error(f"❌ Errore aggiornamento prompt {prompt_id}: {e}")
            if self.connection:
                self.connection.rollback()
            return False

    def delete_prompt(self, prompt_id: int) -> bool:
        """
        Elimina un prompt (soft delete - disattiva invece di eliminare)
        
        Args:
            prompt_id: ID del prompt da eliminare
            
        Returns:
            True se eliminazione riuscita, False altrimenti
        """
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            
            # Soft delete - disattiva il prompt
            query = """
            UPDATE prompts 
            SET is_active = FALSE, updated_at = NOW()
            WHERE id = %s AND is_active = TRUE
            """
            
            cursor.execute(query, (prompt_id,))
            
            if cursor.rowcount > 0:
                # Crea entry nella history per tracciare l'eliminazione
                history_query = """
                INSERT INTO prompt_history 
                SELECT id, tenant_id, engine, prompt_type, prompt_name, 
                       content, variables, FALSE, NOW() 
                FROM prompts WHERE id = %s
                """
                cursor.execute(history_query, (prompt_id,))
                
                self.connection.commit()
                cursor.close()
                
                self.logger.info(f"✅ Prompt {prompt_id} eliminato (disattivato)")
                
                # Invalida cache
                self._invalidate_cache()
                
                return True
            else:
                self.logger.warning(f"⚠️ Prompt {prompt_id} non trovato o già eliminato")
                return False
                
        except Error as e:
            self.logger.error(f"❌ Errore eliminazione prompt {prompt_id}: {e}")
            if self.connection:
                self.connection.rollback()
            return False

    def get_prompt_by_id(self, prompt_id: int) -> Optional[Dict[str, Any]]:
        """
        Recupera un prompt specifico per ID
        
        Args:
            prompt_id: ID del prompt
            
        Returns:
            Dati del prompt o None se non trovato
        """
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            
            query = """
            SELECT id, tenant_id, tenant_name, engine, prompt_type, prompt_name, 
                   prompt_content, dynamic_variables, is_active, created_at, updated_at
            FROM prompts 
            WHERE id = %s
            """
            
            cursor.execute(query, (prompt_id,))
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                return {
                    'id': row[0],
                    'tenant_id': row[1],
                    'tenant_name': row[2],
                    'engine': row[3],
                    'prompt_type': row[4],
                    'prompt_name': row[5],
                    'content': row[6],  # prompt_content
                    'variables': json.loads(row[7]) if row[7] else {},  # dynamic_variables
                    'is_active': bool(row[8]),
                    'created_at': row[9].isoformat() if row[9] else None,
                    'updated_at': row[10].isoformat() if row[10] else None
                }
            return None
            
        except Error as e:
            self.logger.error(f"❌ Errore recupero prompt {prompt_id}: {e}")
            return None

    def preview_prompt_with_variables(self, prompt_id: int) -> Optional[Dict[str, Any]]:
        """
        Genera anteprima di un prompt con variabili risolte
        
        Args:
            prompt_id: ID del prompt
            
        Returns:
            Dizionario con contenuto processato e variabili risolte
        """
        prompt = self.get_prompt_by_id(prompt_id)
        if not prompt:
            return None
        
        try:
            # Simula risoluzione variabili per anteprima
            # In una implementazione reale, qui chiameresti i resolver delle variabili
            resolved_variables = {}
            processed_content = prompt['content']
            
            # Simula alcune variabili comuni
            if 'available_labels' in processed_content:
                resolved_variables['available_labels'] = "prenotazione_esami, ritiro_cartella_clinica_referti, info_contatti"
                processed_content = processed_content.replace(
                    '{available_labels}', 
                    resolved_variables['available_labels']
                )
            
            if 'priority_labels' in processed_content:
                resolved_variables['priority_labels'] = "Focus su: prenotazione_esami (40%), ritiro_cartella_clinica_referti (35%)"
                processed_content = processed_content.replace(
                    '{priority_labels}', 
                    resolved_variables['priority_labels']
                )
            
            if 'context_guidance' in processed_content:
                resolved_variables['context_guidance'] = "CONTESTO: Ospedale Humanitas - Assistenza pazienti"
                processed_content = processed_content.replace(
                    '{context_guidance}', 
                    resolved_variables['context_guidance']
                )
            
            self.logger.info(f"👀 Anteprima generata per prompt {prompt_id}")
            
            return {
                'prompt_id': prompt_id,
                'content': processed_content,
                'variables_resolved': resolved_variables,
                'original_variables': prompt['variables']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Errore generazione anteprima prompt {prompt_id}: {e}")
            return None

    # =============================
    # GESTIONE TOOLS EMBEDDED
    # =============================
    
    def get_prompt_tools(self, 
                        tenant_id: str, 
                        prompt_name: str,
                        engine: str = "LLM") -> List[Dict[str, Any]]:
        """
        Recupera tutti i tools associati a un prompt specifico
        
        Args:
            tenant_id: ID del tenant
            prompt_name: Nome del prompt
            engine: Engine del prompt (default: LLM)
        
        Returns:
            Lista di tools associati al prompt
        """
        try:
            if not self.connect():
                return []
            
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT tools
                FROM prompts 
                WHERE tenant_id = %s 
                AND prompt_name = %s 
                AND engine = %s
                AND is_active = 1
            """, (tenant_id, prompt_name, engine))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result and result['tools']:
                tools = json.loads(result['tools']) if isinstance(result['tools'], str) else result['tools']
                self.logger.info(f"✅ Tools recuperati per {prompt_name}: {len(tools)} tools")
                return tools
            
            return []
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero tools per prompt {prompt_name}: {e}")
            return []
    
    def update_prompt_tools(self, 
                           tenant_id: str, 
                           prompt_name: str, 
                           tools: List[Dict[str, Any]],
                           engine: str = "LLM") -> bool:
        """
        Aggiorna i tools associati a un prompt
        
        Args:
            tenant_id: ID del tenant
            prompt_name: Nome del prompt
            tools: Lista dei tools da associare
            engine: Engine del prompt (default: LLM)
        
        Returns:
            True se aggiornamento riuscito
        """
        try:
            if not self.connect():
                return False
            
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE prompts 
                SET tools = %s,
                    updated_at = CURRENT_TIMESTAMP,
                    updated_by = 'prompt_manager'
                WHERE tenant_id = %s 
                AND prompt_name = %s 
                AND engine = %s
            """, (json.dumps(tools, ensure_ascii=False), tenant_id, prompt_name, engine))
            
            success = cursor.rowcount > 0
            self.connection.commit()
            cursor.close()
            
            if success:
                self.logger.info(f"✅ Tools aggiornati per prompt {prompt_name}")
                self._invalidate_cache()
            else:
                self.logger.warning(f"⚠️ Nessun prompt aggiornato per {prompt_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Errore aggiornamento tools per prompt {prompt_name}: {e}")
            return False
    
    def add_tool_to_prompt(self, 
                          tenant_id: str, 
                          prompt_name: str, 
                          tool: Dict[str, Any],
                          engine: str = "LLM") -> bool:
        """
        Aggiunge un nuovo tool a un prompt (solo se non esiste già)
        
        Args:
            tenant_id: ID del tenant
            prompt_name: Nome del prompt
            tool: Dati del tool da aggiungere
            engine: Engine del prompt (default: LLM)
        
        Returns:
            True se aggiunto con successo
        """
        try:
            # Recupera tools esistenti
            existing_tools = self.get_prompt_tools(tenant_id, prompt_name, engine)
            
            # Verifica se tool esiste già
            tool_name = tool.get('tool_name', '')
            if any(t.get('tool_name') == tool_name for t in existing_tools):
                self.logger.warning(f"⚠️ Tool {tool_name} già esistente per prompt {prompt_name}")
                return False
            
            # Aggiunge nuovo tool
            existing_tools.append(tool)
            
            # Aggiorna prompt
            success = self.update_prompt_tools(tenant_id, prompt_name, existing_tools, engine)
            
            if success:
                self.logger.info(f"✅ Tool {tool_name} aggiunto a prompt {prompt_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Errore aggiunta tool a prompt {prompt_name}: {e}")
            return False
    
    def remove_tool_from_prompt(self, 
                               tenant_id: str, 
                               prompt_name: str, 
                               tool_name: str,
                               engine: str = "LLM") -> bool:
        """
        Rimuove un tool da un prompt
        
        Args:
            tenant_id: ID del tenant
            prompt_name: Nome del prompt
            tool_name: Nome del tool da rimuovere
            engine: Engine del prompt (default: LLM)
        
        Returns:
            True se rimosso con successo
        """
        try:
            # Recupera tools esistenti
            existing_tools = self.get_prompt_tools(tenant_id, prompt_name, engine)
            
            # Filtra tool da rimuovere
            updated_tools = [t for t in existing_tools if t.get('tool_name') != tool_name]
            
            if len(updated_tools) == len(existing_tools):
                self.logger.warning(f"⚠️ Tool {tool_name} non trovato per prompt {prompt_name}")
                return False
            
            # Aggiorna prompt
            success = self.update_prompt_tools(tenant_id, prompt_name, updated_tools, engine)
            
            if success:
                self.logger.info(f"✅ Tool {tool_name} rimosso da prompt {prompt_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Errore rimozione tool da prompt {prompt_name}: {e}")
            return False

    def _invalidate_cache(self):
        """
        Invalida la cache dei prompt
        """
        self._cache = {}
        self.logger.debug("🔄 Cache prompt invalidata")

    def create_prompt_from_template(self,
                                   target_tenant_id: str,
                                   template_prompt: Dict[str, Any]) -> Optional[int]:
        """
        Crea un nuovo prompt copiando da un template
        Se esiste già, lo aggiorna con il contenuto del template
        
        Args:
            target_tenant_id: ID del tenant destinazione (UUID string)
            template_prompt: Dizionario con i dati del prompt template
            
        Returns:
            ID del prompt creato/aggiornato, None se errore
            
        Autore: Sistema
        Data: 2025-08-25
        Ultimo aggiornamento: 2025-08-25
        """
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            
            # CORREZIONE: Recupera il nome reale del tenant
            tenant_query = "SELECT tenant_name FROM tenants WHERE tenant_id = %s LIMIT 1"
            cursor.execute(tenant_query, (target_tenant_id,))
            tenant_result = cursor.fetchone()
            tenant_name = tenant_result[0] if tenant_result else 'unknown'
            
            self.logger.debug(f"🏢 Tenant {target_tenant_id} -> Nome: {tenant_name}")
            
            # Dati del prompt da copiare - CORREZIONE: usa 'content' non 'prompt_content'
            engine = template_prompt.get('engine', 'LLM')
            prompt_type = template_prompt.get('prompt_type', 'USER')
            prompt_name = template_prompt.get('prompt_name', f"{prompt_type.lower()}_prompt")
            content = template_prompt.get('content', '')  # CORREZIONE: chiave corretta è 'content'
            variables = template_prompt.get('variables', {})  # CORREZIONE: chiave corretta è 'variables'
            
            # Assicurati che variables sia un dict, non una stringa JSON
            if isinstance(variables, str):
                variables = json.loads(variables) if variables else {}
            
            # DEBUG: Log del contenuto che stiamo per copiare
            self.logger.debug(f"🔍 Copiando prompt '{prompt_name}': contenuto {len(content)} caratteri")
            
            if not content:
                self.logger.warning(f"⚠️ Prompt '{prompt_name}' ha contenuto vuoto!")
            
            # CORREZIONE: Prima controlla se esiste già il prompt
            check_query = """
            SELECT id FROM prompts 
            WHERE tenant_id = %s AND engine = %s AND prompt_type = %s AND prompt_name = %s
            """
            
            cursor.execute(check_query, (target_tenant_id, engine, prompt_type, prompt_name))
            existing_prompt = cursor.fetchone()
            
            self.logger.debug(f"🔍 Controllo esistenza: {prompt_name} -> {'Trovato ID ' + str(existing_prompt[0]) if existing_prompt else 'Non trovato'}")
            
            if existing_prompt:
                # AGGIORNA il prompt esistente
                prompt_id = existing_prompt[0]
                update_query = """
                UPDATE prompts SET 
                    tenant_name = %s,
                    prompt_content = %s,
                    dynamic_variables = %s,
                    is_active = %s,
                    updated_at = NOW()
                WHERE id = %s
                """
                
                cursor.execute(update_query, (
                    tenant_name,  # CORREZIONE: usa nome tenant reale
                    content,
                    json.dumps(variables),
                    True,  # is_active
                    prompt_id
                ))
                
                self.logger.info(f"✅ Aggiornato prompt esistente: ID {prompt_id} per tenant {target_tenant_id}")
            else:
                self.logger.debug(f"📝 Creazione nuovo prompt: {prompt_name}")
                # CREA nuovo prompt
                insert_query = """
                INSERT INTO prompts (
                    tenant_id, tenant_name, engine, prompt_type, prompt_name, 
                    prompt_content, dynamic_variables, is_active, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                )
                """
                
                cursor.execute(insert_query, (
                    target_tenant_id,
                    tenant_name,  # CORREZIONE: usa nome tenant reale
                    engine,
                    prompt_type,
                    prompt_name,
                    content,
                    json.dumps(variables),
                    True  # is_active
                ))
                
                prompt_id = cursor.lastrowid
                self.logger.info(f"✅ Creato prompt da template: ID {prompt_id} per tenant {target_tenant_id}")
            
            self.connection.commit()
            
            # Invalida cache
            self._invalidate_cache()
            
            return prompt_id
            
        except Error as e:
            self.logger.error(f"❌ Errore creazione prompt da template: {e}")
            if self.connection:
                self.connection.rollback()
            return None

    # =====================================================================
    # GESTIONE ESEMPI MULTI-TENANT
    # =====================================================================
    
    def get_examples_for_placeholder(
        self, 
        tenant_or_id, 
        engine: str = 'LLM', 
        esempio_type: str = 'CONVERSATION',
        limit: int = None
    ) -> str:
        """
        Recupera esempi formattati per sostituire placeholder {{examples_text}}
        
        Args:
            tenant_or_id: Oggetto Tenant o tenant_id per compatibilità
            engine: Tipo di engine (LLM, ML, FINETUNING) 
            esempio_type: Tipo di esempio (CONVERSATION, CLASSIFICATION, TEMPLATE)
            limit: Numero massimo di esempi (opzionale)
            
        Returns:
            Stringa con esempi formattati per il placeholder
            
        Autore: Sistema di Classificazione AI
        Data: 2025-08-25
        Ultimo aggiornamento: 2025-08-25
        """
        if not self.connection:
            self.connect()
            
        try:
            cursor = self.connection.cursor()
            
            # Gestione compatibilità Tenant vs tenant_id string
            if TENANT_AVAILABLE and hasattr(tenant_or_id, 'tenant_id'):
                # Oggetto Tenant - usa direttamente i suoi dati
                tenant = tenant_or_id
                resolved_tenant_id = tenant.tenant_id
                tenant_display = f"{tenant.tenant_name} ({resolved_tenant_id})"
            else:
                # Retrocompatibilità: tenant_id string - normalizza
                resolved_tenant_id = self._resolve_tenant_id(tenant_or_id)
                tenant_display = str(tenant_or_id)
            
            # Query per recuperare esempi attivi
            query = """
            SELECT esempio_content, esempio_name, categoria, description
            FROM esempi 
            WHERE tenant_id = %s AND engine = %s AND esempio_type = %s AND is_active = 1
            ORDER BY created_at DESC
            """
            
            params = [resolved_tenant_id, engine, esempio_type]
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            cursor.execute(query, params)
            esempi = cursor.fetchall()
            
            if not esempi:
                self.logger.warning(f"⚠️ Nessun esempio trovato per tenant {tenant_display}")
                return ""
            
            # Formatta gli esempi
            formatted_examples = []
            for esempio_content, esempio_name, categoria, description in esempi:
                # Gli esempi sono già formattati con UTENTE:/ASSISTENTE:
                formatted_examples.append(esempio_content)
            
            result = "\n\n".join(formatted_examples)
            
            self.logger.info(f"✅ Caricati {len(esempi)} esempi per placeholder {{examples_text}}")
            return result
            
        except Error as e:
            self.logger.error(f"❌ Errore caricamento esempi: {e}")
            return ""
    
    def create_example(
        self,
        tenant_or_id,
        esempio_name: str,
        esempio_content: str,
        engine: str = 'LLM',
        esempio_type: str = 'CONVERSATION',
        description: str = None,
        categoria: str = None,
        livello_difficolta: str = 'MEDIO'
    ) -> Optional[int]:
        """
        Crea nuovo esempio nel database
        
        Args:
            tenant_or_id: Oggetto Tenant o tenant_id per compatibilità
            esempio_name: Nome identificativo dell'esempio
            esempio_content: Contenuto formattato UTENTE:/ASSISTENTE:
            engine: Tipo di engine (LLM, ML, FINETUNING)
            esempio_type: Tipo esempio (CONVERSATION, CLASSIFICATION, TEMPLATE)  
            description: Descrizione dell'esempio
            categoria: Categoria dell'esempio
            livello_difficolta: Livello difficoltà (FACILE, MEDIO, DIFFICILE)
            
        Returns:
            ID dell'esempio creato o None se errore
            
        Autore: Sistema di Classificazione AI
        Data: 2025-08-25
        Ultimo aggiornamento: 2025-08-25
        """
        if not self.connection:
            self.connect()
            
        try:
            cursor = self.connection.cursor()
            
            # Gestione compatibilità Tenant vs tenant_id string
            if TENANT_AVAILABLE and hasattr(tenant_or_id, 'tenant_id'):
                # Oggetto Tenant - usa direttamente i suoi dati
                tenant = tenant_or_id
                resolved_tenant_id = tenant.tenant_id
                tenant_name = tenant.tenant_name
                tenant_display = f"{tenant.tenant_name} ({resolved_tenant_id})"
            else:
                # Retrocompatibilità: tenant_id string - normalizza
                resolved_tenant_id = self._resolve_tenant_id(tenant_or_id)
                tenant_name = self._get_tenant_name(resolved_tenant_id)
                tenant_display = str(tenant_or_id)
            
            # Controlla se esempio già esiste
            check_query = """
            SELECT id FROM esempi 
            WHERE tenant_id = %s AND engine = %s AND esempio_type = %s AND esempio_name = %s
            """
            
            cursor.execute(check_query, (resolved_tenant_id, engine, esempio_type, esempio_name))
            existing = cursor.fetchone()
            
            if existing:
                self.logger.warning(f"⚠️ Esempio '{esempio_name}' già esiste per tenant {tenant_display}")
                return existing[0]
            
            # Inserisci nuovo esempio
            insert_query = """
            INSERT INTO esempi (
                tenant_id, tenant_name, engine, esempio_type, esempio_name,
                esempio_content, description, categoria, livello_difficolta,
                is_active, created_by, created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
            )
            """
            
            cursor.execute(insert_query, (
                resolved_tenant_id,
                tenant_name,
                engine,
                esempio_type,
                esempio_name,
                esempio_content,
                description,
                categoria,
                livello_difficolta,
                1,  # is_active = 1 (TRUE)
                'prompt_manager'
            ))
            
            esempio_id = cursor.lastrowid
            self.connection.commit()
            
            self.logger.info(f"✅ Creato esempio '{esempio_name}' ID {esempio_id} per tenant {tenant_display}")
            return esempio_id
            
        except Error as e:
            self.logger.error(f"❌ Errore creazione esempio: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def get_examples_list(
        self,
        tenant_or_id,
        engine: str = 'LLM',
        esempio_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        Recupera lista esempi per un tenant
        
        Args:
            tenant_or_id: Oggetto Tenant o tenant_id per compatibilità
            engine: Tipo di engine (LLM, ML, FINETUNING)
            esempio_type: Tipo esempio (opzionale per filtrare)
            
        Returns:
            Lista dizionari con dati esempi
            
        Autore: Sistema di Classificazione AI  
        Data: 2025-08-25
        Ultimo aggiornamento: 2025-08-25
        """
        if not self.connection:
            self.connect()
            
        try:
            cursor = self.connection.cursor()
            
            # Gestione compatibilità Tenant vs tenant_id string  
            if TENANT_AVAILABLE and hasattr(tenant_or_id, 'tenant_id'):
                # Oggetto Tenant - usa direttamente i suoi dati
                tenant = tenant_or_id
                resolved_tenant_id = tenant.tenant_id
                tenant_display = f"{tenant.tenant_name} ({resolved_tenant_id})"
            else:
                # Retrocompatibilità: tenant_id string - normalizza
                resolved_tenant_id = self._resolve_tenant_id(tenant_or_id)
                tenant_display = str(tenant_or_id)
            
            # Query dinamica in base ai filtri - Solo esempi attivi
            query = """
            SELECT id, esempio_name, esempio_type, categoria, livello_difficolta, 
                   description, is_active, created_at, updated_at
            FROM esempi 
            WHERE tenant_id = %s AND engine = %s AND is_active = TRUE
            """
            
            params = [resolved_tenant_id, engine]
            
            if esempio_type:
                query += " AND esempio_type = %s"
                params.append(esempio_type)
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            esempi = cursor.fetchall()
            
            # Formatta risultato
            result = []
            for esempio in esempi:
                result.append({
                    'id': esempio[0],
                    'esempio_name': esempio[1],
                    'esempio_type': esempio[2],
                    'categoria': esempio[3],
                    'livello_difficolta': esempio[4],
                    'description': esempio[5],
                    'is_active': bool(esempio[6]),
                    'created_at': esempio[7].strftime('%Y-%m-%d %H:%M:%S'),
                    'updated_at': esempio[8].strftime('%Y-%m-%d %H:%M:%S')
                })
            
            self.logger.info(f"✅ Recuperati {len(result)} esempi per tenant {tenant_display}")
            return result
            
        except Error as e:
            self.logger.error(f"❌ Errore recupero lista esempi: {e}")
            return []
    
    def update_example(
        self,
        esempio_id: int,
        tenant_or_id,
        **updates
    ) -> bool:
        """
        Aggiorna esempio esistente
        
        Args:
            esempio_id: ID dell'esempio da aggiornare
            tenant_or_id: Oggetto Tenant o tenant_id per sicurezza
            **updates: Campi da aggiornare
            
        Returns:
            True se aggiornamento riuscito, False altrimenti
            
        Autore: Sistema di Classificazione AI
        Data: 2025-08-25  
        Ultimo aggiornamento: 2025-08-25
        """
        if not self.connection:
            self.connect()
            
        try:
            cursor = self.connection.cursor()
            
            # Gestione compatibilità Tenant vs tenant_id string
            if TENANT_AVAILABLE and hasattr(tenant_or_id, 'tenant_id'):
                # Oggetto Tenant - usa direttamente i suoi dati
                tenant = tenant_or_id
                resolved_tenant_id = tenant.tenant_id
                tenant_display = f"{tenant.tenant_name} ({resolved_tenant_id})"
            else:
                # Retrocompatibilità: tenant_id string - normalizza
                resolved_tenant_id = self._resolve_tenant_id(tenant_or_id)
                tenant_display = str(tenant_or_id)
            
            # Costruisci query dinamica
            set_clauses = []
            params = []
            
            allowed_fields = [
                'esempio_name', 'esempio_content', 'description', 
                'categoria', 'livello_difficolta', 'is_active'
            ]
            
            for field, value in updates.items():
                if field in allowed_fields:
                    set_clauses.append(f"{field} = %s")
                    params.append(value)
            
            if not set_clauses:
                self.logger.warning("⚠️ Nessun campo valido da aggiornare")
                return False
            
            # Aggiungi timestamp e condizioni WHERE
            set_clauses.append("updated_at = NOW()")
            set_clauses.append("updated_by = %s")
            params.extend(['prompt_manager', esempio_id, resolved_tenant_id])
            
            query = f"""
            UPDATE esempi 
            SET {', '.join(set_clauses)}
            WHERE id = %s AND tenant_id = %s
            """
            
            cursor.execute(query, params)
            
            if cursor.rowcount > 0:
                self.connection.commit()
                self.logger.info(f"✅ Aggiornato esempio ID {esempio_id}")
                return True
            else:
                self.logger.warning(f"⚠️ Esempio ID {esempio_id} non trovato per tenant {tenant_display}")
                return False
                
        except Error as e:
            self.logger.error(f"❌ Errore aggiornamento esempio: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def delete_example(self, esempio_id: int, tenant_or_id) -> bool:
        """
        Elimina esempio (soft delete - imposta is_active = False)
        
        Args:
            esempio_id: ID dell'esempio da eliminare
            tenant_or_id: Oggetto Tenant o tenant_id per sicurezza
            
        Returns:
            True se eliminazione riuscita, False altrimenti
            
        Autore: Sistema di Classificazione AI
        Data: 2025-08-25
        Ultimo aggiornamento: 2025-08-25
        """
        return self.update_example(esempio_id, tenant_or_id, is_active=False)
    
    def get_prompt_with_examples(
        self,
        tenant_id: str,
        engine: str,
        prompt_type: str,
        prompt_name: str,
        examples_limit: int = None
    ) -> Optional[str]:
        """
        Recupera prompt e sostituisce {{examples_text}} con esempi reali
        
        Args:
            tenant_id: ID del tenant
            engine: Tipo di engine  
            prompt_type: Tipo di prompt
            prompt_name: Nome del prompt
            examples_limit: Limite numero esempi
            
        Returns:
            Prompt con esempi sostituiti o None se errore
            
        Autore: Sistema di Classificazione AI
        Data: 2025-08-25
        Ultimo aggiornamento: 2025-08-25
        """
        try:
            # Carica prompt base
            prompt = self.get_prompt_strict(tenant_id, engine, prompt_type, prompt_name)
            
            # Se non contiene placeholder, restituisci com'è
            if '{{examples_text}}' not in prompt:
                return prompt
            
            # Carica esempi
            examples = self.get_examples_for_placeholder(
                tenant_id, engine, 'CONVERSATION', examples_limit
            )
            
            # Sostituisci placeholder
            if examples:
                final_prompt = prompt.replace('{{examples_text}}', examples)
                self.logger.info(f"✅ Sostituito {{examples_text}} con {len(examples.split('UTENTE:'))-1} esempi")
                return final_prompt
            else:
                # Se non ci sono esempi, rimuovi il placeholder
                final_prompt = prompt.replace('{{examples_text}}', '')
                self.logger.warning(f"⚠️ Nessun esempio disponibile, rimosso placeholder {{examples_text}}")
                return final_prompt
                
        except Exception as e:
            self.logger.error(f"❌ Errore sostituzione esempi nel prompt: {e}")
            return None
