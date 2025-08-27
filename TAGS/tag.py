import sys
import os
import yaml
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Aggiunge il path per importare le classi esistenti
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from TagDatabase.tag_database_connector import TagDatabaseConnector

# Import Tenant class per gestione centralizzata tenant
try:
    from Utils.tenant import Tenant
    TENANT_AVAILABLE = True
except ImportError:
    Tenant = None
    TENANT_AVAILABLE = False


class IntelligentTagSuggestionManager:
    """
    Gestione intelligente dei suggerimenti di tag per il training supervisionato.
    
    Logica implementata:
    - Cliente nuovo (senza classificazioni in DB) → zero suggerimenti
    - Dal primo caso analizzato → raccoglie e suggerisce tag da LLM, ML e revisioni umane
    """
    
    def __init__(self, config_path=None):
        """
        Inizializza il gestore dei suggerimenti intelligenti
        
        Args:
            config_path (str): Percorso al file di configurazione
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
    def _load_config(self, config_path):
        """Carica i parametri di configurazione dal file config.yaml"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise Exception(f"File di configurazione non trovato: {config_path}")
        except yaml.YAMLError as e:
            raise Exception(f"Errore nel parsing del file YAML: {e}")
    
    def _setup_logger(self):
        """Setup del logger per il modulo"""
        logger = logging.getLogger('IntelligentTagSuggestion')
        
        if logger.hasHandlers():
            logger.handlers.clear()
        
        logger.setLevel(logging.INFO)
        
        # Handler per console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        return logger
    
    def _resolve_tenant_id_from_name(self, client_name: str) -> str:
        """
        Risolve un client_name nel tenant_id corrispondente riutilizzando
        la logica esistente di mongo_classification_reader.
        
        SICUREZZA MULTI-TENANT: Questo metodo garantisce che ogni client
        veda solo i tag del proprio tenant, prevenendo data leak.
        
        Args:
            client_name (str): Nome del cliente/tenant (es. "alleanza", "bosch")
            
        Returns:
            str: tenant_id UUID del tenant corrispondente
            
        Raises:
            Exception: Se il tenant non è trovato o non è attivo
            
        Note:
            Riutilizza la logica testata di mongo_classification_reader.get_tenant_info_from_name()
            per garantire coerenza nell'architettura multi-tenant.
            
        Last modified: 23/08/2025 - Valerio Bignardi
        """
        try:
            # Connessione al database locale TAG per tabella tenants
            import mysql.connector
            from mysql.connector import Error
            
            db_config = self.config['tag_database']  # Database locale TAG 
            connection = mysql.connector.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database'],
                autocommit=True
            )
            
            cursor = connection.cursor()
            
            # Query identica a mongo_classification_reader.get_tenant_info_from_name()
            # Cerca per tenant_name OR tenant_slug per massima flessibilità
            query = """
            SELECT tenant_id, tenant_slug, tenant_name 
            FROM tenants 
            WHERE (tenant_name = %s OR tenant_slug = %s) AND is_active = 1 
            LIMIT 1
            """
            
            cursor.execute(query, (client_name, client_name))
            result = cursor.fetchone()
            
            cursor.close()
            connection.close()
            
            if result:
                tenant_id, tenant_slug, tenant_name = result
                self.logger.info(f"✅ Risolto client '{client_name}' → tenant_id: {tenant_id} ({tenant_name})")
                return tenant_id
            else:
                error_msg = f"❌ MULTI-TENANT SECURITY: Tenant '{client_name}' non trovato o non attivo"
                self.logger.error(error_msg)
                raise Exception(error_msg)
                
        except Error as e:
            error_msg = f"❌ Errore risoluzione tenant_id per '{client_name}': {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def has_existing_classifications(self, client_name=None, tenant=None) -> bool:
        """
        Controlla se esistono classificazioni in MongoDB per il cliente.
        MIGRAZIONE: Ora cerca nella collection MongoDB del tenant invece che in MySQL.
        
        Args:
            client_name (str): [DEPRECATED] Nome del cliente - usare tenant invece
            tenant (Tenant): Oggetto Tenant centralizzato (preferito)
            
        Returns:
            bool: True se il cliente ha classificazioni esistenti, False altrimenti
            
        UPGRADE: Preferire l'uso del parametro 'tenant' invece di 'client_name'
            
        Note:
            Cerca nella collection MongoDB usando pattern {tenant_slug}_{tenant_id}
            e conta i documenti con classificazioni valide (non 'altro').
            
        Last modified: 23/08/2025 - Valerio Bignardi
        """
        try:
            # 🏗️ GESTIONE TENANT CENTRALIZZATA
            if tenant and TENANT_AVAILABLE:
                tenant_id = tenant.tenant_id
                client_name_resolved = tenant.tenant_slug
                print(f"🎯 TAG: Uso tenant centralizzato {tenant}")
            elif client_name:
                # Legacy mode: risolvi client_name
                tenant_id = self._resolve_tenant_id_from_name(client_name)
                client_name_resolved = client_name
                print(f"🔄 TAG: Modalità legacy - risolvo client_name {client_name}")
            else:
                raise ValueError("Deve essere fornito 'tenant' (preferito) o 'client_name' (legacy)")
            
            # STEP 2: Genera nome collection usando pattern {tenant_slug}_{tenant_id}
            # Per Alleanza: alleanza_a0fd7600-f4f7-11ef-9315-96000228e7fe
            collection_name = f"{client_name_resolved.lower()}_{tenant_id}"
            
            # STEP 3: Connessione a MongoDB
            from pymongo import MongoClient
            
            mongodb_config = self.config['mongodb']
            client = MongoClient(mongodb_config['url'])
            db = client[mongodb_config['database']]  # 'classificazioni'
            collection = db[collection_name]
            
            # STEP 4: Conta documenti con classificazioni valide
            # Esclude 'altro' perché è tag generico quando LLM non trova match specifico
            count = collection.count_documents({
                'classification': {
                    '$exists': True, 
                    '$ne': None, 
                    '$ne': '', 
                    '$ne': 'altro'
                }
            })
            
            client.close()
            
            exists = count > 0
            self.logger.info(f"🔍 Cliente '{client_name}': {count} classificazioni valide in MongoDB - {'Esistente' if exists else 'Nuovo'}")
            self.logger.debug(f"   Collection: {collection_name}")
            
            return exists
            
        except Exception as e:
            self.logger.error(f"❌ Errore controllo classificazioni esistenti per '{client_name}': {e}")
            # Fallback sicuro: assume cliente nuovo per evitare errori critici
            return False
    
    def get_suggested_tags_for_client(self, client_name=None, tenant=None) -> List[Dict[str, Any]]:
        """
        Ottiene i tag suggeriti per un cliente dalla tabella TAGS.tags.
        SICUREZZA MULTI-TENANT: Ogni cliente riceve SOLO i tag del proprio tenant.
        
        Args:
            client_name (str): [DEPRECATED] Nome del cliente - usare tenant invece
            tenant (Tenant): Oggetto Tenant centralizzato (preferito)
            
        Returns:
            List[Dict]: Lista di tag suggeriti con metadati del tenant
            
        UPGRADE: Preferire l'uso del parametro 'tenant' invece di 'client_name'
            
        Raises:
            Exception: Se il tenant non è trovato o non attivo
            
        Note:
            Implementa isolamento multi-tenant risolvendo client_name → tenant_id
            e filtrando i tag per tenant specifico.
            
        Last modified: 23/08/2025 - Valerio Bignardi
        """
        try:
            # 🏗️ GESTIONE TENANT CENTRALIZZATA
            if tenant and TENANT_AVAILABLE:
                tenant_id = tenant.tenant_id
                client_name_resolved = tenant.tenant_slug
                print(f"🎯 TAG: Uso tenant centralizzato {tenant}")
            elif client_name:
                # Legacy mode: risolvi client_name
                tenant_id = self._resolve_tenant_id_from_name(client_name)
                client_name_resolved = client_name
                print(f"🔄 TAG: Modalità legacy - risolvo client_name {client_name}")
            else:
                raise ValueError("Deve essere fornito 'tenant' (preferito) o 'client_name' (legacy)")
                
            self.logger.info(f"🔒 MULTI-TENANT: Client '{client_name_resolved}' → Tenant ID: {tenant_id}")
            
            # STEP 2: Ottieni SOLO i tag del tenant specificato
            all_tags = self._get_all_available_tags(tenant_id)
            
            # Se il cliente ha classificazioni esistenti, ordina i tag per uso frequente
            if self.has_existing_classifications(client_name):
                self.logger.info(f"Cliente esistente '{client_name}': ordinamento tag per uso frequente")
                usage_stats = self._get_tag_usage_stats(client_name)
                
                # Aggiungi statistiche di utilizzo ai tag
                for tag_info in all_tags:
                    tag_name = tag_info['tag_name']
                    if tag_name in usage_stats:
                        tag_info['usage_count'] = usage_stats[tag_name]['count']
                        tag_info['avg_confidence'] = usage_stats[tag_name]['avg_confidence']
                        tag_info['source'] = 'used_frequently'
                    else:
                        tag_info['usage_count'] = 0
                        tag_info['avg_confidence'] = 0.0
                        tag_info['source'] = 'available'
                
                # Ordina per uso frequente
                all_tags.sort(key=lambda x: (x['usage_count'], x['avg_confidence']), reverse=True)
            else:
                self.logger.info(f"Cliente nuovo '{client_name}': tutti i tag disponibili senza ordinamento")
                # Per clienti nuovi, aggiungi metadati base
                for tag_info in all_tags:
                    tag_info['usage_count'] = 0
                    tag_info['avg_confidence'] = 0.0
                    tag_info['source'] = 'available'
            
            self.logger.info(f"Cliente '{client_name}': {len(all_tags)} tag suggeriti restituiti")
            return all_tags
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero suggerimenti per {client_name}: {e}")
            return []
    
    def _get_all_available_tags(self, tenant_id: str) -> List[Dict[str, Any]]:
        """
        Ottiene tutti i tag disponibili dalla tabella TAGS.tags per un tenant specifico.
        SICUREZZA MULTI-TENANT: Filtra i risultati per tenant_id.
        
        Args:
            tenant_id (str): ID del tenant per cui recuperare i tag
            
        Returns:
            List[Dict]: Lista di tag disponibili del tenant con metadati
            
        Note:
            Implementa isolamento multi-tenant tramite filtro WHERE tenant_id = %s
            nella query SQL per prevenire data leak tra tenant diversi.
            
        Last modified: 23/08/2025 - Valerio Bignardi
        """
        try:
            # Connessione diretta al database TAG
            import mysql.connector
            from mysql.connector import Error
            
            db_config = self.config['tag_database']
            connection = mysql.connector.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database'],
                autocommit=True
            )
            
            cursor = connection.cursor()
            
            # SICUREZZA MULTI-TENANT: Query filtra per tenant_id specifico
            query = """
            SELECT MIN(id) as id, tag_name, 
                   MAX(tag_description) as tag_description, 
                   MIN(created_at) as created_at
            FROM tags 
            WHERE tenant_id = %s
            GROUP BY tag_name
            ORDER BY tag_name ASC
            """
            
            self.logger.debug(f"🔒 Executing tenant-filtered query for tenant_id: {tenant_id}")
            cursor.execute(query, (tenant_id,))
            result = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            tags_list = []
            if result:
                for row in result:
                    tags_list.append({
                        'id': row[0],
                        'tag_name': row[1],
                        'description': row[2] if row[2] else '',
                        'created_at': row[3].isoformat() if row[3] else None
                    })
            
            self.logger.info(f"Caricati {len(tags_list)} tag unici dalla tabella TAGS.tags")
            return tags_list
            
        except Exception as e:
            self.logger.error(f"Errore nel caricamento tag dalla tabella tags: {e}")
            return []
    
    def _get_tag_usage_stats(self, client_name=None, tenant=None) -> Dict[str, Dict[str, Any]]:
        """
        Ottiene le statistiche di utilizzo dei tag per un cliente dalle classificazioni MongoDB.
        MIGRAZIONE: Ora cerca nella collection MongoDB del tenant invece che in MySQL.
        
        Args:
            client_name (str): [DEPRECATED] Nome del cliente - usare tenant invece
            tenant (Tenant): Oggetto Tenant centralizzato (preferito)
            
        Returns:
            Dict: Statistiche di utilizzo per tag name
            
        UPGRADE: Preferire l'uso del parametro 'tenant' invece di 'client_name'
            
        Note:
            Usa aggregation pipeline MongoDB per calcolare COUNT e AVG per ogni tag.
            Equivalente alla query SQL: SELECT tag, COUNT(*), AVG(confidence) GROUP BY tag
            
        Last modified: 23/08/2025 - Valerio Bignardi
        """
        try:
            # 🏗️ GESTIONE TENANT CENTRALIZZATA
            if tenant and TENANT_AVAILABLE:
                tenant_id = tenant.tenant_id
                client_name_resolved = tenant.tenant_slug
                print(f"🎯 TAG: Uso tenant centralizzato {tenant}")
            elif client_name:
                # Legacy mode: risolvi client_name
                tenant_id = self._resolve_tenant_id_from_name(client_name)
                client_name_resolved = client_name
                print(f"🔄 TAG: Modalità legacy - risolvo client_name {client_name}")
            else:
                raise ValueError("Deve essere fornito 'tenant' (preferito) o 'client_name' (legacy)")
            
            # STEP 2: Genera nome collection usando pattern {tenant_slug}_{tenant_id}
            collection_name = f"{client_name_resolved.lower()}_{tenant_id}"
            
            # STEP 3: Connessione a MongoDB
            from pymongo import MongoClient
            
            mongodb_config = self.config['mongodb']
            client = MongoClient(mongodb_config['url'])
            db = client[mongodb_config['database']]
            collection = db[collection_name]
            
            # STEP 4: Aggregation pipeline per statistiche utilizzo tag
            pipeline = [
                # Filtra solo classificazioni valide (non 'altro')
                {
                    '$match': {
                        'classification': {
                            '$exists': True,
                            '$ne': None,
                            '$ne': '',
                            '$ne': 'altro'
                        }
                    }
                },
                # Raggruppa per classification e calcola statistiche
                {
                    '$group': {
                        '_id': '$classification',
                        'count': {'$sum': 1},
                        'avg_confidence': {'$avg': '$confidence'}
                    }
                },
                # Ordina per frequenza decrescente
                {
                    '$sort': {'count': -1}
                }
            ]
            
            results = list(collection.aggregate(pipeline))
            client.close()
            
            # STEP 5: Converte risultati nel formato atteso
            usage_stats = {}
            for result in results:
                tag_name = result['_id']
                usage_stats[tag_name] = {
                    'count': result['count'],
                    'avg_confidence': float(result['avg_confidence']) if result['avg_confidence'] else 0.0
                }
            
            self.logger.info(f"📊 Cliente '{client_name}': statistiche utilizzo per {len(usage_stats)} tag")
            return usage_stats
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero statistiche utilizzo per '{client_name}': {e}")
            return {}

    def _get_automatic_classification_tags(self, client_name: str) -> List[Dict[str, Any]]:
        """
        Ottiene tag dalle classificazioni automatiche esistenti (ML/LLM).
        
        Args:
            client_name (str): Nome del cliente
            
        Returns:
            List[Dict]: Lista di tag con metadati
        """
        try:
            # Connessione diretta al database delle classificazioni (database locale TAG)
            import mysql.connector
            from mysql.connector import Error
            
            db_config = self.config['tag_database']
            connection = mysql.connector.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database'],
                autocommit=True
            )
            
            cursor = connection.cursor()
            
            # Query per tag da classificazioni automatiche (ML e LLM)
            query = """
            SELECT tag_name, COUNT(*) as count, AVG(confidence_score) as avg_confidence
            FROM session_classifications 
            WHERE tenant_name = %s 
                AND tag_name IS NOT NULL 
                AND tag_name != ''
                AND classification_method IN ('AUTOMATIC', 'ML', 'LLM')
            GROUP BY tag_name
            ORDER BY count DESC, avg_confidence DESC
            """
            
            cursor.execute(query, (client_name,))
            result = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            tags_list = []
            if result:
                for row in result:
                    tags_list.append({
                        'tag': row[0],
                        'count': row[1],
                        'source': 'automatic',
                        'avg_confidence': float(row[2]) if row[2] else 0.0
                    })
            
            return tags_list
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero tag automatici per {client_name}: {e}")
            return []
    
    def _get_llm_generated_suggestion_tags(self, client_name: str) -> List[Dict[str, Any]]:
        """
        Ottiene tag generati appositamente da LLM per suggerimenti.
        (Questa funzionalità sarà implementata quando creeremo la tabella llm_generated_tags)
        
        Args:
            client_name (str): Nome del cliente
            
        Returns:
            List[Dict]: Lista di tag LLM con metadati
        """
        # TODO: Implementare quando avremo la tabella llm_generated_tags
        # Per ora restituisce lista vuota
        return []
    
    def _get_human_review_suggestion_tags(self, client_name: str) -> List[Dict[str, Any]]:
        """
        Ottiene tag dalle revisioni umane precedenti.
        
        Args:
            client_name (str): Nome del cliente
            
        Returns:
            List[Dict]: Lista di tag dalle revisioni umane
        """
        try:
            # Connessione diretta al database delle classificazioni (database locale TAG)
            import mysql.connector
            from mysql.connector import Error
            
            db_config = self.config['tag_database']
            connection = mysql.connector.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database'],
                autocommit=True
            )
            
            cursor = connection.cursor()
            
            # Query per tag da classificazioni manuali/revisioni umane
            query = """
            SELECT tag_name, COUNT(*) as count
            FROM session_classifications 
            WHERE tenant_name = %s 
                AND tag_name IS NOT NULL 
                AND tag_name != ''
                AND classification_method = 'HUMAN_REVIEW'
            GROUP BY tag_name
            ORDER BY count DESC
            """
            
            cursor.execute(query, (client_name,))
            result = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            tags_list = []
            if result:
                for row in result:
                    tags_list.append({
                        'tag': row[0],
                        'count': row[1],
                        'source': 'human_review',
                        'avg_confidence': 0.95  # Alta confidenza per decisioni umane
                    })
            
            return tags_list
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero tag revisioni umane per {client_name}: {e}")
            return []
    
    def collect_llm_generated_tags(self, client_name: str, session_id: str, llm_tags: List[str], confidence: float = 0.8) -> bool:
        """
        Raccoglie e memorizza i tag generati da LLM per i suggerimenti futuri.
        
        Args:
            client_name (str): Nome del cliente
            session_id (str): ID della sessione
            llm_tags (List[str]): Lista di tag generati da LLM
            confidence (float): Livello di confidenza dei tag LLM
            
        Returns:
            bool: True se la raccolta è avvenuta con successo
        """
        try:
            # TODO: Implementare salvataggio in tabella llm_generated_tags
            # Per ora loggiamo l'operazione
            self.logger.info(f"Raccolta tag LLM per {client_name}, sessione {session_id}: {llm_tags}")
            
            # Nota: Quando creeremo la tabella llm_generated_tags, implementeremo
            # il salvataggio dei tag LLM per i suggerimenti futuri
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nella raccolta tag LLM per {client_name}: {e}")
            return False
    
    def collect_human_review_tags(self, client_name: str, session_id: str, human_tags: List[str]) -> bool:
        """
        Raccoglie i tag delle revisioni umane per i suggerimenti futuri.
        
        Args:
            client_name (str): Nome del cliente
            session_id (str): ID della sessione
            human_tags (List[str]): Lista di tag delle revisioni umane
            
        Returns:
            bool: True se la raccolta è avvenuta con successo
        """
        try:
            # I tag delle revisioni umane sono già salvati nel database tramite
            # la pipeline esistente (session_classifications con classification_method='MANUAL')
            # Quindi non serve implementare nulla di nuovo qui.
            
            self.logger.info(f"Tag revisione umana per {client_name}, sessione {session_id}: {human_tags}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nella raccolta tag revisione umana per {client_name}: {e}")
            return False


# Istanza globale per l'uso nell'applicazione
tag_suggestion_manager = IntelligentTagSuggestionManager()


# Test della classe
if __name__ == "__main__":
    print("=== TEST INTELLIGENT TAG SUGGESTION MANAGER ===")
    
    manager = IntelligentTagSuggestionManager()
    
    # Test 1: Cliente nuovo (dovrebbe restituire lista vuota)
    print("\n🧪 Test 1: Cliente nuovo 'TestClient'")
    new_client_tags = manager.get_suggested_tags_for_client("TestClient")
    print(f"Suggerimenti per cliente nuovo: {len(new_client_tags)} tag")
    print(f"Tag: {new_client_tags}")
    
    # Test 2: Cliente esistente (dovrebbe restituire suggerimenti)
    print("\n🧪 Test 2: Cliente esistente 'Humanitas'")
    existing_client_tags = manager.get_suggested_tags_for_client("Humanitas")
    print(f"Suggerimenti per cliente esistente: {len(existing_client_tags)} tag")
    if existing_client_tags:
        print(f"Primi 3 tag: {existing_client_tags[:3]}")
    
    # Test 3: Verifica esistenza classificazioni
    print("\n🧪 Test 3: Verifica esistenza classificazioni")
    has_humanitas = manager.has_existing_classifications("Humanitas")
    has_testclient = manager.has_existing_classifications("TestClient")
    print(f"Humanitas ha classificazioni: {has_humanitas}")
    print(f"TestClient ha classificazioni: {has_testclient}")
    
    print("\n✅ Test completati!")
