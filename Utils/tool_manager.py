"""
ToolManager - Gestore centralizzato per tools/funzioni LLM multi-tenant

Questo modulo gestisce i tools (function calling) disponibili per l'LLM per ogni tenant,
permettendo la configurazione dinamica delle funzioni disponibili.

Funzionalità:
- CRUD completo per tools per tenant
- Validazione schema function calling
- Gestione multi-tenant con isolamento
- Integrazione con PromptManager per variabili dinamiche
- Caching per performance

Autore: Sistema di Classificazione Humanitas  
Data: 2025-08-22
"""

import os
import json
import logging
import yaml
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import uuid

class ToolManager:
    """
    Gestore centralizzato per tools multi-tenant con variabili dinamiche
    """
    
    def __init__(self, config_path: str = None):
        """
        Inizializza ToolManager
        
        Args:
            config_path: Percorso del file di configurazione
        """
        self.logger = logging.getLogger(__name__)
        
        # Carica configurazione
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # Configurazione database
        self.db_config = self.config.get('tag_database', {})
        
        # Cache per tools
        self._tools_cache = {}
        self._cache_expiry = {}
        self.cache_duration = 300  # 5 minuti
        
        self.logger.info("✅ ToolManager inizializzato")
    
    def _get_connection(self):
        """Crea connessione al database MySQL"""
        try:
            connection = mysql.connector.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 3306),
                user=self.db_config.get('user', 'root'),
                password=self.db_config.get('password'),
                database=self.db_config.get('database', 'TAG'),
                charset='utf8mb4',
                autocommit=True
            )
            return connection
        except Error as e:
            self.logger.error(f"❌ Errore connessione database: {e}")
            raise
    
    def get_all_tools_for_tenant(self, tenant_id: str) -> List[Dict[str, Any]]:
        """
        Recupera tutti i tools attivi per un tenant
        
        Args:
            tenant_id: UUID del tenant
            
        Returns:
            Lista dei tools del tenant
        """
        cache_key = f"tools_{tenant_id}"
        
        # Controlla cache
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"🔄 Cache hit per tools tenant {tenant_id}")
            return self._tools_cache[cache_key]
        
        try:
            connection = self._get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Query per recuperare tools del tenant
            query = """
            SELECT id, tool_name, display_name, description, function_schema, 
                   is_active, tenant_id, tenant_name, created_at, updated_at
            FROM tools 
            WHERE tenant_id = %s AND is_active = TRUE
            ORDER BY tool_name ASC
            """
            
            cursor.execute(query, (tenant_id,))
            tools = cursor.fetchall()
            
            # Converte function_schema da JSON string a dict
            for tool in tools:
                if tool['function_schema']:
                    try:
                        tool['function_schema'] = json.loads(tool['function_schema'])
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"⚠️ Schema JSON non valido per tool {tool['tool_name']}: {e}")
                        tool['function_schema'] = {}
                
                # Converte datetime in string per serializzazione
                tool['created_at'] = tool['created_at'].isoformat() if tool['created_at'] else None
                tool['updated_at'] = tool['updated_at'].isoformat() if tool['updated_at'] else None
            
            # Aggiorna cache
            self._tools_cache[cache_key] = tools
            self._cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_duration
            
            cursor.close()
            connection.close()
            
            self.logger.info(f"📊 Recuperati {len(tools)} tools per tenant {tenant_id}")
            return tools
            
        except Error as e:
            self.logger.error(f"❌ Errore recupero tools per tenant {tenant_id}: {e}")
            return []
    
    def get_tool_by_id(self, tool_id: int) -> dict:
        """
        Recupera un tool specifico per ID univoco
        
        Args:
            tool_id: ID univoco del tool
            
        Returns:
            Dict con dati del tool o None se non trovato
            
        Autore: Valerio Bignardi
        Data: 2025-08-30
        """
        if not self.connection:
            self.connect()
            
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            query = """
            SELECT id, tool_name, display_name, description, function_schema, 
                   is_active, tenant_id, tenant_name
            FROM tools 
            WHERE id = %s AND is_active = 1
            """
            
            cursor.execute(query, (tool_id,))
            result = cursor.fetchone()
            
            if result and result['function_schema']:
                try:
                    # Parse JSON schema se è stringa
                    if isinstance(result['function_schema'], str):
                        result['function_schema'] = json.loads(result['function_schema'])
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ Errore parsing JSON schema tool ID {tool_id}: {e}")
                    result['function_schema'] = {}
            
            cursor.close()
            return result
            
        except Error as e:
            self.logger.error(f"❌ Errore recupero tool ID {tool_id}: {e}")
            return None

    def get_tool_by_id(self, tool_id: int) -> Optional[Dict[str, Any]]:
        """
        Recupera un tool specifico per ID
        
        Args:
            tool_id: ID del tool
            
        Returns:
            Dati del tool o None se non trovato
        """
        try:
            connection = self._get_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = """
            SELECT id, tool_name, display_name, description, function_schema, 
                   is_active, tenant_id, tenant_name, created_at, updated_at
            FROM tools 
            WHERE id = %s AND is_active = 1
            """
            
            cursor.execute(query, (tool_id,))
            tool = cursor.fetchone()
            
            if tool and tool['function_schema']:
                try:
                    tool['function_schema'] = json.loads(tool['function_schema'])
                except json.JSONDecodeError as e:
                    self.logger.warning(f"⚠️ Schema JSON non valido per tool {tool['tool_name']}: {e}")
                    tool['function_schema'] = {}
            
            if tool:
                # Converte datetime in string
                tool['created_at'] = tool['created_at'].isoformat() if tool['created_at'] else None
                tool['updated_at'] = tool['updated_at'].isoformat() if tool['updated_at'] else None
            
            cursor.close()
            connection.close()
            
            return tool
            
        except Error as e:
            self.logger.error(f"❌ Errore recupero tool {tool_id}: {e}")
            return None
    
    def create_tool(self, tool_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Crea un nuovo tool
        
        Args:
            tool_data: Dati del tool da creare
            
        Returns:
            Tool creato con ID o None se errore
        """
        required_fields = ['tool_name', 'display_name', 'description', 'function_schema', 'tenant_id']
        
        # Validazione campi obbligatori
        for field in required_fields:
            if field not in tool_data:
                self.logger.error(f"❌ Campo obbligatorio mancante: {field}")
                return None
        
        # Validazione schema function
        if not self._validate_function_schema(tool_data['function_schema']):
            self.logger.error("❌ Schema function non valido")
            return None
        
        try:
            connection = self._get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Verifica che il tool_name sia univoco
            cursor.execute("SELECT id FROM tools WHERE tool_name = %s", (tool_data['tool_name'],))
            if cursor.fetchone():
                self.logger.error(f"❌ Tool con nome '{tool_data['tool_name']}' già esistente")
                return None
            
            # Inserimento nuovo tool
            insert_query = """
            INSERT INTO tools (tool_name, display_name, description, function_schema, 
                              is_active, tenant_id, tenant_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            function_schema_json = json.dumps(tool_data['function_schema'], ensure_ascii=False)
            
            cursor.execute(insert_query, (
                tool_data['tool_name'],
                tool_data['display_name'], 
                tool_data['description'],
                function_schema_json,
                tool_data.get('is_active', True),
                tool_data['tenant_id'],
                tool_data.get('tenant_name')
            ))
            
            # Recupera il tool appena creato
            tool_id = cursor.lastrowid
            created_tool = self.get_tool_by_id(tool_id)
            
            cursor.close()
            connection.close()
            
            # Invalida cache per questo tenant
            self._invalidate_cache_for_tenant(tool_data['tenant_id'])
            
            self.logger.info(f"✅ Tool creato: {tool_data['tool_name']} (ID: {tool_id})")
            return created_tool
            
        except Error as e:
            self.logger.error(f"❌ Errore creazione tool: {e}")
            return None
    
    def update_tool(self, tool_id: int, tool_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Aggiorna un tool esistente
        
        Args:
            tool_id: ID del tool da aggiornare
            tool_data: Nuovi dati del tool
            
        Returns:
            Tool aggiornato o None se errore
        """
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            # Verifica che il tool esista
            cursor.execute("SELECT tenant_id FROM tools WHERE id = %s", (tool_id,))
            result = cursor.fetchone()
            if not result:
                self.logger.error(f"❌ Tool {tool_id} non trovato")
                return None
            
            tenant_id = result[0]
            
            # Costruisce query di update dinamica
            update_fields = []
            values = []
            
            updateable_fields = ['tool_name', 'display_name', 'description', 'function_schema', 'is_active', 'tenant_name']
            
            for field in updateable_fields:
                if field in tool_data:
                    update_fields.append(f"{field} = %s")
                    if field == 'function_schema':
                        # Validazione e conversione schema
                        if not self._validate_function_schema(tool_data[field]):
                            self.logger.error("❌ Schema function non valido")
                            return None
                        values.append(json.dumps(tool_data[field], ensure_ascii=False))
                    else:
                        values.append(tool_data[field])
            
            if not update_fields:
                self.logger.warning("⚠️ Nessun campo da aggiornare")
                return self.get_tool_by_id(tool_id)
            
            # Aggiunge timestamp di aggiornamento
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            
            update_query = f"UPDATE tools SET {', '.join(update_fields)} WHERE id = %s"
            values.append(tool_id)
            
            cursor.execute(update_query, values)
            
            cursor.close()
            connection.close()
            
            # Invalida cache per questo tenant
            self._invalidate_cache_for_tenant(tenant_id)
            
            updated_tool = self.get_tool_by_id(tool_id)
            self.logger.info(f"✅ Tool aggiornato: ID {tool_id}")
            return updated_tool
            
        except Error as e:
            self.logger.error(f"❌ Errore aggiornamento tool {tool_id}: {e}")
            return None
    
    def delete_tool(self, tool_id: int) -> bool:
        """
        Elimina un tool (soft delete - disattiva)
        
        Args:
            tool_id: ID del tool da eliminare
            
        Returns:
            True se eliminato con successo, False altrimenti
        """
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            # Verifica che il tool esista
            cursor.execute("SELECT tenant_id FROM tools WHERE id = %s", (tool_id,))
            result = cursor.fetchone()
            if not result:
                self.logger.error(f"❌ Tool {tool_id} non trovato")
                return False
            
            tenant_id = result[0]
            
            # Soft delete - disattiva il tool
            cursor.execute(
                "UPDATE tools SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                (tool_id,)
            )
            
            cursor.close()
            connection.close()
            
            # Invalida cache per questo tenant
            self._invalidate_cache_for_tenant(tenant_id)
            
            self.logger.info(f"✅ Tool disattivato: ID {tool_id}")
            return True
            
        except Error as e:
            self.logger.error(f"❌ Errore eliminazione tool {tool_id}: {e}")
            return False
    
    def _validate_function_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Valida lo schema della function per OpenAI function calling
        
        Args:
            schema: Schema da validare
            
        Returns:
            True se valido, False altrimenti
        """
        try:
            # Schema base richiesto
            if not isinstance(schema, dict):
                return False
            
            if schema.get('type') != 'function':
                return False
            
            function_def = schema.get('function', {})
            if not isinstance(function_def, dict):
                return False
            
            # Campi obbligatori in function
            required_function_fields = ['name', 'description', 'parameters']
            for field in required_function_fields:
                if field not in function_def:
                    self.logger.error(f"Campo function.{field} mancante")
                    return False
            
            # Validazione parameters (deve essere JSON Schema)
            parameters = function_def['parameters']
            if not isinstance(parameters, dict):
                return False
            
            if parameters.get('type') != 'object':
                self.logger.error("function.parameters.type deve essere 'object'")
                return False
            
            self.logger.debug("✅ Schema function valido")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Errore validazione schema: {e}")
            return False
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Controlla se la cache è ancora valida"""
        if cache_key not in self._tools_cache:
            return False
        
        if cache_key not in self._cache_expiry:
            return False
        
        return datetime.now().timestamp() < self._cache_expiry[cache_key]
    
    def _invalidate_cache_for_tenant(self, tenant_id: str):
        """Invalida la cache per un tenant specifico"""
        cache_key = f"tools_{tenant_id}"
        if cache_key in self._tools_cache:
            del self._tools_cache[cache_key]
        if cache_key in self._cache_expiry:
            del self._cache_expiry[cache_key]
        
        self.logger.debug(f"🗑️ Cache invalidata per tenant {tenant_id}")
    
    def get_tools_count_by_tenant(self) -> Dict[str, int]:
        """
        Recupera il conteggio dei tools per ogni tenant
        
        Returns:
            Dizionario tenant_name -> count
        """
        try:
            connection = self._get_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = """
            SELECT tenant_name, COUNT(*) as tool_count
            FROM tools 
            WHERE is_active = TRUE
            GROUP BY tenant_name
            ORDER BY tenant_name
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            return {row['tenant_name']: row['tool_count'] for row in results if row['tenant_name']}
            
        except Error as e:
            self.logger.error(f"❌ Errore conteggio tools: {e}")
            return {}
    
    def export_tools_for_tenant(self, tenant_id: str) -> Dict[str, Any]:
        """
        Esporta tutti i tools di un tenant in formato JSON
        
        Args:
            tenant_id: UUID del tenant
            
        Returns:
            Dizionario con i tools esportati
        """
        tools = self.get_all_tools_for_tenant(tenant_id)
        
        export_data = {
            'tenant_id': tenant_id,
            'export_timestamp': datetime.now().isoformat(),
            'tools_count': len(tools),
            'tools': tools
        }
        
        self.logger.info(f"📤 Esportati {len(tools)} tools per tenant {tenant_id}")
        return export_data

    def get_tool_by_name(self, tool_name: str, tenant_or_id=None) -> Optional[Dict[str, Any]]:
        """
        Recupera un tool attivo per nome e tenant
        
        Args:
            tool_name: Nome del tool da cercare
            tenant_or_id: Oggetto Tenant o tenant_id per filtrare
            
        Returns:
            Dict con dati del tool o None se non trovato
            
        Autore: Valerio Bignardi
        Data: 2025-08-30
        """
        try:
            connection = self._get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Gestione compatibilità Tenant vs tenant_id string
            if tenant_or_id is not None and hasattr(tenant_or_id, 'tenant_id'):
                # Oggetto Tenant - usa direttamente i suoi dati
                resolved_tenant_id = tenant_or_id.tenant_id
            else:
                # Retrocompatibilità: tenant_id string
                resolved_tenant_id = tenant_or_id
            
            # Query per tool attivo
            query = """
            SELECT id, tool_name, display_name, description, function_schema, 
                   is_active, tenant_id, tenant_name, created_at, updated_at
            FROM tools 
            WHERE tool_name = %s AND is_active = 1
            """
            params = [tool_name]
            
            # Aggiungi filtro tenant se specificato
            if resolved_tenant_id:
                query += " AND tenant_id = %s"
                params.append(resolved_tenant_id)
            
            query += " ORDER BY updated_at DESC LIMIT 1"
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            cursor.close()
            connection.close()
            
            if result:
                # Parse function_schema da JSON
                function_schema = result['function_schema']
                if isinstance(function_schema, str):
                    try:
                        function_schema = json.loads(function_schema)
                    except json.JSONDecodeError:
                        self.logger.warning(f"⚠️ Schema non valido per tool {tool_name}")
                        function_schema = {}
                
                tool_data = dict(result)
                tool_data['function_schema'] = function_schema
                
                self.logger.info(f"✅ Trovato tool '{tool_name}' per tenant {resolved_tenant_id}")
                return tool_data
            else:
                self.logger.warning(f"⚠️ Tool '{tool_name}' non trovato per tenant {resolved_tenant_id}")
                return None
                
        except Error as e:
            self.logger.error(f"❌ Errore recupero tool per nome: {e}")
            return None

if __name__ == "__main__":
    # Test del ToolManager
    logging.basicConfig(level=logging.INFO)
    
    tool_manager = ToolManager()
    
    # Test recupero tools per Humanitas
    humanitas_tools = tool_manager.get_all_tools_for_tenant('015007d9-d413-11ef-86a5-96000228e7fe')
    print(f"\n🔧 Tools Humanitas: {len(humanitas_tools)}")
    
    for tool in humanitas_tools:
        print(f"  • {tool['tool_name']}: {tool['display_name']}")
    
    # Test conteggio
    counts = tool_manager.get_tools_count_by_tenant()
    print(f"\n📊 Conteggio tools per tenant: {counts}")
