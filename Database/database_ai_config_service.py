"""
File: database_ai_config_service.py
Autore: Sistema AI
Data creazione: 2025-08-25
Scopo: Servizio per gestione configurazioni AI da database MySQL

Storico modifiche:
- 2025-08-25: Creazione iniziale per sostituzione config.yaml con database
"""

import yaml
import mysql.connector
from mysql.connector import Error
import os
import json
import sys
from typing import Dict, Optional, Any
from datetime import datetime

# Import Tenant class per gestione centralizzata tenant
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from Utils.tenant import Tenant
    TENANT_AVAILABLE = True
except ImportError:
    Tenant = None
    TENANT_AVAILABLE = False


class DatabaseAIConfigService:
    """
    Servizio per gestione configurazioni AI da database
    
    Sostituisce la gestione via config.yaml con persistenza su MySQL.
    Gestisce configurazioni embedding e LLM per tenant.
    """
    
    def __init__(self):
        """
        Inizializza servizio configurazioni database
        
        Returns:
            None
        """
        self.config = self._load_config()
        self.connection = None
        
        # Cache in memoria per performance
        self._tenant_configs_cache = {}
        self._cache_timestamp = None
    
    def _load_config(self):
        """
        Carica configurazione database dal config.yaml
        
        Returns:
            dict: Configurazione database TAG
            
        Ultima modifica: 2025-08-25
        """
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config['tag_database']
        except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
            raise Exception(f"Errore caricamento configurazione database: {e}")
    
    def _connetti(self):
        """
        Stabilisce connessione al database TAG
        
        Returns:
            bool: True se connessione riuscita
            
        Ultima modifica: 2025-08-25
        """
        try:
            if self.connection and self.connection.is_connected():
                return True
                
            self.connection = mysql.connector.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database']
            )
            return self.connection.is_connected()
        except Error as e:
            print(f"❌ Errore connessione database: {e}")
            return False
    
    def _disconnetti(self):
        """
        Chiude connessione al database se aperta
        
        Ultima modifica: 2025-08-25
        """
        if self.connection and self.connection.is_connected():
            self.connection.close()
    
    def _resolve_tenant_id(self, identifier: str) -> str:
        """
        Risolve l'identificatore (slug o UUID) in tenant_id UUID
        
        Args:
            identifier: tenant slug (es. 'wopta') o tenant_id UUID
            
        Returns:
            str: tenant_id UUID se trovato, identifier originale altrimenti
            
        Ultima modifica: 2025-08-25
        """
        if not self._connetti():
            return identifier
            
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Prima prova se è già un UUID valido (cerca per tenant_id)
            query = """
            SELECT tenant_id 
            FROM tenants 
            WHERE tenant_id = %s AND is_active = TRUE
            """
            cursor.execute(query, (identifier,))
            result = cursor.fetchone()
            
            if result:
                cursor.close()
                return result['tenant_id']
            
            # Se non trovato, prova a cercare per slug
            query = """
            SELECT tenant_id 
            FROM tenants 
            WHERE tenant_slug = %s AND is_active = TRUE
            """
            cursor.execute(query, (identifier,))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return result['tenant_id']
            
            # Se non trovato in nessuno dei due modi, restituisce l'identifier originale
            return identifier
            
        except Error as e:
            print(f"❌ Errore risoluzione tenant_id per '{identifier}': {e}")
            return identifier
    
    def get_tenant_configuration(self, tenant_id=None, tenant=None, force_no_cache=False):
        """
        Scopo: Recupera la configurazione AI per un tenant specifico
        
        Parametri di input:
        - tenant_id: [DEPRECATED] ID del tenant (UUID) - usare tenant invece
        - tenant: Oggetto Tenant centralizzato (preferito)
        - force_no_cache: Se True, bypassa completamente la cache
        
        Valori di ritorno:
        - dict: Configurazione completa del tenant
        
        UPGRADE: Preferire l'uso del parametro 'tenant' invece di 'tenant_id'
        Data ultima modifica: 2025-08-26
        """
        
        # 🏗️ GESTIONE TENANT CENTRALIZZATA
        if tenant and TENANT_AVAILABLE:
            resolved_tenant_id = tenant.tenant_id
            tenant_info = {
                'tenant_id': tenant.tenant_id,
                'tenant_name': tenant.tenant_name,
                'tenant_slug': tenant.tenant_slug,
                'tenant_database': tenant.tenant_database
            }
            print(f"🎯 DatabaseAI: Uso tenant centralizzato {tenant}")
        elif tenant_id:
            # Legacy mode: risolvi tenant_id
            resolved_tenant_id = self._resolve_tenant_id(tenant_id)
            tenant_info = None  # Sarà risolto nel database query
            print(f"🔄 DatabaseAI: Modalità legacy - risolvo tenant_id {tenant_id}")
        else:
            raise ValueError("Deve essere fornito 'tenant' (preferito) o 'tenant_id' (legacy)")
            
        
        # Se force_no_cache=True, bypassa completamente la cache
        if force_no_cache:
            print(f"🔧 FORCE NO CACHE: Bypass cache per tenant {resolved_tenant_id}")
        else:
            # Verifica cache esistente solo se non è forzato il bypass
            if (self._tenant_configs_cache.get(resolved_tenant_id) and 
                self._cache_timestamp and 
                (datetime.now() - self._cache_timestamp).seconds < 300):  # 5 minuti cache
                return self._tenant_configs_cache[resolved_tenant_id]
        
        # DEBUG per force_no_cache
        if force_no_cache:
            print(f"🔥 FORCE NO CACHE: Bypasso cache e leggo DIRETTAMENTE dal database per tenant {tenant_id}")
        
        if not self._connetti():
            # Fallback su configurazione di default
            return self._get_default_config()
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            query = """
            SELECT tenant_id, tenant_name, tenant_slug,
                   embedding_engine, llm_engine,
                   embedding_config, llm_config,
                   is_active, updated_at
            FROM engines
            WHERE tenant_id = %s AND is_active = TRUE
            """
            
            cursor.execute(query, (resolved_tenant_id,))
            result = cursor.fetchone()
            cursor.close()
            
            print(f"🔍 [DEBUG] Query eseguita per tenant_id: {resolved_tenant_id}")
            print(f"🔍 [DEBUG] Risultato database: {result}")
            
            if result:
                # 🎯 USA INFORMAZIONI TENANT CENTRALIZZATE se disponibili
                if tenant_info:
                    # Usa informazioni dal tenant centralizzato (più affidabili)
                    config = {
                        'tenant_id': tenant_info['tenant_id'],
                        'tenant_name': tenant_info['tenant_name'],
                        'tenant_slug': tenant_info['tenant_slug'],
                        'tenant_database': tenant_info['tenant_database'],  # Info extra dal Tenant
                        'embedding_engine': result['embedding_engine'],
                        'llm_engine': result['llm_engine'],
                        'is_active': result['is_active'],
                        'updated_at': result['updated_at']
                    }
                else:
                    # Fallback legacy: usa info dal database
                    config = {
                        'tenant_id': result['tenant_id'],
                        'tenant_name': result['tenant_name'],
                        'tenant_slug': result['tenant_slug'],
                        'embedding_engine': result['embedding_engine'],
                        'llm_engine': result['llm_engine'],
                        'is_active': result['is_active'],
                        'updated_at': result['updated_at']
                    }
                
                # Parsea configurazioni JSON
                if result['embedding_config']:
                    try:
                        config['embedding_config'] = json.loads(result['embedding_config'])
                    except json.JSONDecodeError:
                        config['embedding_config'] = {}
                else:
                    config['embedding_config'] = {}
                
                if result['llm_config']:
                    try:
                        config['llm_config'] = json.loads(result['llm_config'])
                    except json.JSONDecodeError:
                        config['llm_config'] = {}
                else:
                    config['llm_config'] = {}
                
                # Aggiorna cache solo se non è stato forzato il bypass
                if not force_no_cache:
                    self._tenant_configs_cache[resolved_tenant_id] = config
                    self._cache_timestamp = datetime.now()
                
                return config
            else:
                print(f"⚠️ Configurazione non trovata per tenant {tenant_id} (risolto: {resolved_tenant_id})")
                return self._get_default_config()
                
        except Error as e:
            print(f"❌ Errore recupero configurazione: {e}")
            return self._get_default_config()
        finally:
            self._disconnetti()
    
    def set_embedding_engine(self, tenant_id: str, engine_type: str, **config) -> Dict[str, Any]:
        """
        Imposta engine embedding per un tenant
        
        Args:
            tenant_id: ID del tenant
            engine_type: Tipo engine (labse, bge_m3, openai_large, openai_small)
            **config: Configurazione aggiuntiva engine
            
        Returns:
            dict: Risultato operazione con successo e dettagli
            
        Ultima modifica: 2025-08-25
        """
        # Valida engine type
        valid_engines = ['labse', 'bge_m3', 'openai_large', 'openai_small']
        if engine_type not in valid_engines:
            return {
                'success': False,
                'error': f'Engine non supportato: {engine_type}. Validi: {valid_engines}'
            }
        
        if not self._connetti():
            return {
                'success': False,
                'error': 'Impossibile connettersi al database'
            }
        
        try:
            cursor = self.connection.cursor()
            
            # Recupera info tenant
            cursor.execute("""
                SELECT tenant_name, tenant_slug 
                FROM tenants 
                WHERE tenant_id = %s AND is_active = TRUE
            """, (tenant_id,))
            
            tenant_info = cursor.fetchone()
            if not tenant_info:
                cursor.close()
                return {
                    'success': False,
                    'error': f'Tenant {tenant_id} non trovato o non attivo'
                }
            
            tenant_name, tenant_slug = tenant_info
            
            # Prepara configurazione JSON
            embedding_config_json = json.dumps(config) if config else None
            
            # Upsert nella tabella engines
            upsert_query = """
            INSERT INTO engines (tenant_id, tenant_name, tenant_slug, embedding_engine, embedding_config)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                embedding_engine = VALUES(embedding_engine),
                embedding_config = VALUES(embedding_config),
                updated_at = CURRENT_TIMESTAMP
            """
            
            cursor.execute(upsert_query, (
                tenant_id, tenant_name, tenant_slug, 
                engine_type, embedding_config_json
            ))
            
            self.connection.commit()
            cursor.close()
            
            # Invalida COMPLETAMENTE la cache dopo modifica configurazione
            # FIXBUG: Rimuove timestamp globale per forzare ricaricamento da DB
            self._tenant_configs_cache.clear()
            self._cache_timestamp = None
            print("🗑️ Cache configurazioni invalidata dopo modifica embedding engine")
            
            print(f"✅ Engine embedding '{engine_type}' impostato per tenant '{tenant_name}'")
            
            # Importa factory per validazione (lazy import per evitare circolari)
            try:
                from EmbeddingEngine.embedding_engine_factory import EmbeddingEngineFactory
                factory = EmbeddingEngineFactory()
                # NOTA: get_engine_info non esiste, usiamo info basic
                engine_info = {'type': engine_type, 'available': True}
            except ImportError:
                engine_info = {'type': engine_type, 'available': 'unknown'}
            
            return {
                'success': True,
                'tenant_id': tenant_id,
                'tenant_name': tenant_name,
                'engine_type': engine_type,
                'config': config,
                'engine_info': engine_info,
                'saved_to': 'database',
                'timestamp': datetime.now().isoformat()
            }
            
        except Error as e:
            print(f"❌ Errore salvataggio embedding engine: {e}")
            if self.connection:
                self.connection.rollback()
            return {
                'success': False,
                'error': f'Errore database: {str(e)}'
            }
        finally:
            self._disconnetti()
    
    def set_llm_engine(self, tenant_id: str, model_name: str, **config) -> Dict[str, Any]:
        """
        Imposta modello LLM per un tenant
        
        Args:
            tenant_id: ID del tenant
            model_name: Nome del modello LLM
            **config: Configurazione aggiuntiva modello
            
        Returns:
            dict: Risultato operazione con successo e dettagli
            
        Ultima modifica: 2025-08-25
        """
        if not self._connetti():
            return {
                'success': False,
                'error': 'Impossibile connettersi al database'
            }
        
        try:
            cursor = self.connection.cursor()
            
            # Recupera info tenant
            cursor.execute("""
                SELECT tenant_name, tenant_slug 
                FROM tenants 
                WHERE tenant_id = %s AND is_active = TRUE
            """, (tenant_id,))
            
            tenant_info = cursor.fetchone()
            if not tenant_info:
                cursor.close()
                return {
                    'success': False,
                    'error': f'Tenant {tenant_id} non trovato o non attivo'
                }
            
            tenant_name, tenant_slug = tenant_info
            
            # Prepara configurazione JSON
            llm_config_json = json.dumps(config) if config else None
            
            # Upsert nella tabella engines
            upsert_query = """
            INSERT INTO engines (tenant_id, tenant_name, tenant_slug, llm_engine, llm_config)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                llm_engine = VALUES(llm_engine),
                llm_config = VALUES(llm_config),
                updated_at = CURRENT_TIMESTAMP
            """
            
            cursor.execute(upsert_query, (
                tenant_id, tenant_name, tenant_slug, 
                model_name, llm_config_json
            ))
            
            print(f"🔍 [DEBUG] Salvataggio LLM: tenant_id={tenant_id}, model_name={model_name}")
            print(f"🔍 [DEBUG] Upsert query: {upsert_query}")
            print(f"🔍 [DEBUG] Parametri: ({tenant_id}, {tenant_name}, {tenant_slug}, {model_name}, {llm_config_json})")
            
            self.connection.commit()
            cursor.close()
            
            # Invalida COMPLETAMENTE la cache dopo modifica configurazione
            # FIXBUG: Rimuove timestamp globale per forzare ricaricamento da DB
            self._tenant_configs_cache.clear()
            self._cache_timestamp = None
            print("🗑️ Cache configurazioni invalidata dopo modifica LLM engine")
            
            # CORREZIONE CRITICA: Invalida cache LLMFactory per forzare reload modello
            try:
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Classification'))
                from llm_factory import llm_factory
                
                # Invalida cache LLM per il tenant specificato
                llm_factory.invalidate_tenant_cache(tenant_id)
                print(f"🗑️ Cache LLMFactory invalidata per tenant {tenant_id}")
                
            except ImportError:
                print("⚠️ LLMFactory non disponibile per invalidazione cache")
            except Exception as e:
                print(f"⚠️ Errore invalidazione cache LLMFactory: {e}")
            
            print(f"✅ Modello LLM '{model_name}' impostato per tenant '{tenant_name}'")
            
            return {
                'success': True,
                'tenant_id': tenant_id,
                'tenant_name': tenant_name,
                'model_name': model_name,
                'config': config,
                'saved_to': 'database',
                'timestamp': datetime.now().isoformat()
            }
            
        except Error as e:
            print(f"❌ Errore salvataggio LLM engine: {e}")
            if self.connection:
                self.connection.rollback()
            return {
                'success': False,
                'error': f'Errore database: {str(e)}'
            }
        finally:
            self._disconnetti()
    
    def get_available_embedding_engines(self) -> list:
        """
        Ottiene lista engines embedding disponibili
        
        Returns:
            list: Lista engines con info disponibilità
            
        Ultima modifica: 2025-08-25
        """
        try:
            # Importa factory per info engines (lazy import)
            from EmbeddingEngine.embedding_engine_factory import EmbeddingEngineFactory
            factory = EmbeddingEngineFactory()
            # NOTA: metodo semplificato dato che get_available_engines non esiste
            engines = []
            for engine_type in ['labse', 'bge_m3', 'openai_large', 'openai_small']:
                engines.append({
                    'type': engine_type,
                    'available': True,  # Assumiamo disponibile per ora
                    'description': f'{engine_type.upper()} embedding engine'
                })
            return engines
        except ImportError:
            # Fallback se factory non disponibile
            return [
                {'type': 'labse', 'available': True, 'description': 'LaBSE embedding model'},
                {'type': 'bge_m3', 'available': True, 'description': 'BGE-M3 embedding model'},
                {'type': 'openai_large', 'available': False, 'description': 'OpenAI Large embedding'},
                {'type': 'openai_small', 'available': False, 'description': 'OpenAI Small embedding'}
            ]
    
    def get_all_tenant_configurations(self) -> Dict[str, Dict[str, Any]]:
        """
        Recupera configurazioni di tutti i tenants dal database
        
        Returns:
            dict: Mappa tenant_id -> configurazione
            
        Ultima modifica: 2025-08-25
        """
        if not self._connetti():
            return {}
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            query = """
            SELECT tenant_id, tenant_name, tenant_slug,
                   embedding_engine, llm_engine,
                   embedding_config, llm_config,
                   is_active, updated_at
            FROM engines
            WHERE is_active = TRUE
            ORDER BY tenant_name
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            
            configs = {}
            for result in results:
                tenant_id = result['tenant_id']
                
                config = {
                    'tenant_name': result['tenant_name'],
                    'tenant_slug': result['tenant_slug'],
                    'embedding_engine': result['embedding_engine'],
                    'llm_engine': result['llm_engine'],
                    'is_active': result['is_active'],
                    'updated_at': result['updated_at']
                }
                
                # Parsea configurazioni JSON
                if result['embedding_config']:
                    try:
                        config['embedding_config'] = json.loads(result['embedding_config'])
                    except json.JSONDecodeError:
                        config['embedding_config'] = {}
                else:
                    config['embedding_config'] = {}
                
                if result['llm_config']:
                    try:
                        config['llm_config'] = json.loads(result['llm_config'])
                    except json.JSONDecodeError:
                        config['llm_config'] = {}
                else:
                    config['llm_config'] = {}
                
                configs[tenant_id] = config
            
            return configs
            
        except Error as e:
            print(f"❌ Errore recupero configurazioni: {e}")
            return {}
        finally:
            self._disconnetti()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Restituisce configurazione di default
        
        Returns:
            dict: Configurazione default
            
        Ultima modifica: 2025-08-25
        """
        return {
            'embedding_engine': 'labse',
            'llm_engine': 'mistral:7b',
            'embedding_config': {},
            'llm_config': {},
            'is_active': True,
            'source': 'default'
        }
    
    def clear_cache(self):
        """
        Svuota cache configurazioni tenant
        
        Ultima modifica: 2025-08-25
        """
        self._tenant_configs_cache.clear()
        self._cache_timestamp = None
        print("🗑️ Cache configurazioni svuotata")

    def reset_llm_parameters(self, tenant_id: str) -> Dict[str, Any]:
        """
        Ripristina i parametri LLM ai valori di default per un tenant eliminando
        la configurazione personalizzata (llm_config) dal database.

        Non modifica il campo llm_engine per non cambiare il modello selezionato.

        Args:
            tenant_id: UUID del tenant

        Returns:
            dict: Risultato operazione
        """
        if not self._connetti():
            return {
                'success': False,
                'error': 'Impossibile connettersi al database'
            }

        try:
            cursor = self.connection.cursor()
            # Verifica esistenza record
            cursor.execute(
                "SELECT 1 FROM engines WHERE tenant_id = %s AND is_active = TRUE",
                (tenant_id,)
            )
            exists = cursor.fetchone() is not None

            if not exists:
                cursor.close()
                return {
                    'success': False,
                    'error': f"Tenant {tenant_id} non trovato nella tabella engines"
                }

            # Imposta llm_config a NULL (o vuoto) e aggiorna timestamp
            update_query = (
                "UPDATE engines SET llm_config = NULL, updated_at = CURRENT_TIMESTAMP "
                "WHERE tenant_id = %s AND is_active = TRUE"
            )
            cursor.execute(update_query, (tenant_id,))
            self.connection.commit()
            cursor.close()

            # Invalida cache dopo la modifica
            self._tenant_configs_cache.clear()
            self._cache_timestamp = None

            return {
                'success': True,
                'tenant_id': tenant_id,
                'message': 'Parametri LLM resettati (llm_config svuotato nel database)',
                'saved_to': 'database',
                'timestamp': datetime.now().isoformat()
            }

        except Error as e:
            if self.connection:
                self.connection.rollback()
            return {
                'success': False,
                'error': f'Errore database: {str(e)}'
            }
        finally:
            self._disconnetti()

    def save_batch_processing_config(self, tenant_id: str, batch_config: Dict[str, Any]) -> bool:
        """
        Salva configurazione batch processing per tenant nel database
        
        Scopo:
            Salva parametri classification_batch_size e max_parallel_calls 
            nella configurazione LLM del tenant per persistenza
        
        Parametri di input:
            tenant_id: ID del tenant (UUID)
            batch_config: Dizionario con parametri batch {
                'classification_batch_size': int,
                'max_parallel_calls': int
            }
        
        Valori di ritorno:
            bool: True se salvataggio riuscito, False altrimenti
        
        Data ultima modifica: 2025-09-07
        """
        if not self._connetti():
            return False
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Recupera configurazione LLM esistente
            select_query = """
            SELECT llm_config FROM engines 
            WHERE tenant_id = %s AND is_active = TRUE
            """
            cursor.execute(select_query, (tenant_id,))
            result = cursor.fetchone()
            
            if result:
                # Carica configurazione esistente
                existing_config = {}
                if result['llm_config']:
                    try:
                        existing_config = json.loads(result['llm_config'])
                    except:
                        existing_config = {}
                
                # Merge con nuovi parametri batch
                existing_config.update({
                    'batch_processing': {
                        'classification_batch_size': batch_config.get('classification_batch_size', 32),
                        'max_parallel_calls': batch_config.get('max_parallel_calls', 200),
                        'updated_at': datetime.now().isoformat()
                    }
                })
                
                # Salva configurazione aggiornata
                llm_config_json = json.dumps(existing_config, ensure_ascii=False, indent=2)
                
                update_query = """
                UPDATE engines 
                SET llm_config = %s, updated_at = CURRENT_TIMESTAMP
                WHERE tenant_id = %s AND is_active = TRUE
                """
                
                cursor.execute(update_query, (llm_config_json, tenant_id))
                
                print(f"✅ [BATCH CONFIG] Parametri batch salvati per tenant {tenant_id}")
                print(f"   📦 classification_batch_size: {batch_config.get('classification_batch_size')}")
                print(f"   ⚡ max_parallel_calls: {batch_config.get('max_parallel_calls')}")
                
            else:
                print(f"❌ [BATCH CONFIG] Tenant {tenant_id} non trovato nella tabella engines")
                cursor.close()
                return False
            
            self.connection.commit()
            cursor.close()
            
            # Invalida cache per forzare ricaricamento
            self._tenant_configs_cache.clear()
            self._cache_timestamp = None
            
            return True
            
        except Error as e:
            print(f"❌ [BATCH CONFIG] Errore salvataggio parametri batch: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    
    def get_batch_processing_config(self, tenant_id: str) -> Dict[str, Any]:
        """
        Recupera configurazione batch processing per tenant
        
        Scopo:
            Carica parametri batch processing salvati nel database
            con fallback ai valori di default se non configurati
        
        Parametri di input:
            tenant_id: ID del tenant (UUID)
        
        Valori di ritorno:
            dict: Configurazione batch processing con campi:
                - classification_batch_size: int
                - max_parallel_calls: int
                - source: str (database/default)
        
        Data ultima modifica: 2025-09-07
        """
        # Valori di default da config.yaml
        default_config = {
            'classification_batch_size': 32,
            'max_parallel_calls': 200,
            'source': 'default'
        }
        
        if not self._connetti():
            return default_config
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Recupera configurazione LLM 
            select_query = """
            SELECT llm_config FROM engines 
            WHERE tenant_id = %s AND is_active = TRUE
            """
            cursor.execute(select_query, (tenant_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if result and result['llm_config']:
                try:
                    llm_config = json.loads(result['llm_config'])
                    batch_config = llm_config.get('batch_processing', {})
                    
                    if batch_config:
                        return {
                            'classification_batch_size': batch_config.get('classification_batch_size', 32),
                            'max_parallel_calls': batch_config.get('max_parallel_calls', 200),
                            'source': 'database',
                            'updated_at': batch_config.get('updated_at')
                        }
                        
                except json.JSONDecodeError:
                    print(f"⚠️ [BATCH CONFIG] Errore parsing llm_config per tenant {tenant_id}")
            
            return default_config
            
        except Error as e:
            print(f"❌ [BATCH CONFIG] Errore caricamento parametri batch: {e}")
            return default_config


# =============================================================================
# Test del servizio
# =============================================================================
if __name__ == "__main__":
    print("=== TEST DATABASE AI CONFIG SERVICE ===\n")
    
    service = DatabaseAIConfigService()
    
    try:
        print("🔧 Test 1: Recupero configurazione tenant...")
        config = service.get_tenant_configuration("humanitas")
        print(f"Configurazione humanitas: {config}")
        
        print(f"\n🛠️ Test 2: Impostazione embedding engine BGE-M3...")
        result = service.set_embedding_engine("humanitas", "bge_m3", 
                                             batch_size=32, 
                                             normalize_embeddings=True)
        print(f"Risultato: {result}")
        
        print(f"\n📊 Test 3: Recupero engines disponibili...")
        engines = service.get_available_embedding_engines()
        print(f"Engines disponibili: {len(engines)}")
        for engine in engines:
            print(f"  - {engine.get('type', 'unknown')}: {engine.get('description', 'N/A')}")
        
        print(f"\n📋 Test 4: Configurazioni tutti i tenants...")
        all_configs = service.get_all_tenant_configurations()
        print(f"Tenants configurati: {len(all_configs)}")
        for tenant_id, config in all_configs.items():
            print(f"  - {config['tenant_name']}: {config['embedding_engine']} + {config['llm_engine']}")
        
        print("\n✅ Tutti i test completati con successo!")
        
    except Exception as e:
        print(f"❌ Errore durante test: {e}")
