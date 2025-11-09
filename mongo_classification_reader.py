#!/usr/bin/env python3
"""
Author: Valerio Bignardi
Date: 2025-08-29
Description: Reader per classificazioni da MongoDB
Last Update: 2025-08-29

MongoDB Classification Reader - Legge le classificazioni direttamente da MongoDB
per l'interfaccia web React

AGGIORNATO: Supporta oggetto Tenant centralizzato per eliminare conversioni ridondanti
"""

from pymongo import MongoClient
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import os
import sys
import numpy as np  # 🆕 Per supporto embedding

# Import trace_all centralizzato per evitare import circolari
try:
    from Utils.tracing import trace_all
except ImportError:
    # Fallback se trace_all non è disponibile
    def trace_all(*args, **kwargs):
        pass

# Import della classe Tenant
sys.path.append(os.path.join(os.path.dirname(__file__), 'Utils'))
from tenant import Tenant

# Import config_loader per caricare config.yaml con variabili ambiente
from config_loader import load_config
from config_loader import get_mongodb_config


class MongoClassificationReader:
    """
    Scopo: Legge le classificazioni dal database MongoDB per l'interfaccia web
    
    AGGIORNATO: Supporta collection per tenant ({tenant}_classifications)
    
    Parametri input:
        - mongodb_url: URL di connessione MongoDB
        - database_name: Nome del database
        - tenant_name: Nome del tenant (per collection specifica)
        - collection_name: Nome della collection (legacy, opzionale)
        
    Output:
        - Classificazioni formattate per l'interfaccia React
        
    Ultimo aggiornamento: 2025-08-21
    """
    
    def __init__(self, tenant: Tenant,
                 mongodb_url: str = "mongodb://localhost:27017",
                 database_name: str = "classificazioni"):
        """
        Inizializza il reader MongoDB con OGGETTO TENANT OBBLIGATORIO
        CAMBIO RADICALE: Elimina tutte le stringhe, USA SOLO OGGETTO TENANT
        
        Args:
            tenant: Oggetto Tenant completo con tutti i dati (OBBLIGATORIO)
            mongodb_url: URL di connessione MongoDB
            database_name: Nome del database
        """
        if not hasattr(tenant, 'tenant_id') or not hasattr(tenant, 'tenant_name') or not hasattr(tenant, 'tenant_slug'):
            raise TypeError(f"Il parametro 'tenant' deve essere un oggetto Tenant con attributi tenant_id, tenant_name, tenant_slug. Ricevuto: {type(tenant)} - {tenant}")
            
        self.tenant = tenant

        # Carica configurazione MongoDB e applica in modo sicuro
        mongo_cfg = {}
        try:
            mongo_cfg = get_mongodb_config() or {}
        except Exception:
            mongo_cfg = {}

        # Flag abilitazione: se manca URL o è placeholder, disabilita in modo esplicito
        configured_url = mongo_cfg.get('url')
        url_is_placeholder = isinstance(configured_url, str) and ('${' in configured_url or '}' in configured_url)

        self.mongo_enabled = True
        if configured_url and not url_is_placeholder:
            # Se il chiamante ha passato l'URL di default, sovrascrivi con quello da config
            if mongodb_url == "mongodb://localhost:27017":
                self.mongodb_url = configured_url
            else:
                self.mongodb_url = mongodb_url
        else:
            # Nessun URL valido configurato → disabilita Mongo per evitare spam errori
            self.mongo_enabled = False
            self.mongodb_url = mongodb_url

        # Database name: preferisci config se presente
        configured_db = mongo_cfg.get('database')
        if configured_db and not (isinstance(configured_db, str) and ('${' in configured_db)):
            self.database_name = configured_db
        else:
            self.database_name = database_name

        # Per evitare log ripetuti quando Mongo è disabilitato
        self._disabled_notice_emitted = False
        
        # Collection name basata su tenant object
        self.collection_name = self.get_collection_name()
            
        # Database e Client
        self.client = None
        self.db = None
        
        print(f"✅ MongoClassificationReader inizializzato per tenant: {tenant.tenant_name}")
        print(f"   📊 Collection: {self.collection_name}")
        print(f"   🏢 Database: {self.database_name}")
    
    def get_tenant_id_from_name(self, tenant_name: str) -> Optional[str]:
        """
        Scopo: Recupera tenant_id dal nome tenant via query MySQL
        
        Args:
            tenant_name: Nome del tenant
            
        Returns:
            tenant_id (UUID) se trovato, None altrimenti
            
        Ultimo aggiornamento: 2025-01-27
        """
        try:
            import yaml
            import mysql.connector
            import os
            
            # Carica configurazione TAG database
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
            config = load_config()
            
            tag_db_config = config.get('tag_database', {})
            if not tag_db_config:
                return None
                
            # Connessione MySQL
            connection = mysql.connector.connect(
                host=tag_db_config['host'],
                port=tag_db_config['port'],
                user=tag_db_config['user'],
                password=tag_db_config['password'],
                database=tag_db_config['database']
            )
            
            cursor = connection.cursor()
            # Debug: proviamo sia tenant_name che tenant_slug per compatibilità
            cursor.execute("SELECT tenant_id FROM tenants WHERE (tenant_name = %s OR tenant_slug = %s) AND is_active = 1", (tenant_name, tenant_name.lower()))
            result = cursor.fetchone()
            
            cursor.close()
            connection.close()
            
            if result:
                return result[0]
            else:
                # Debug: se non trovato, proviamo case insensitive
                connection = mysql.connector.connect(
                    host=tag_db_config['host'],
                    port=tag_db_config['port'],
                    user=tag_db_config['user'],
                    password=tag_db_config['password'],
                    database=tag_db_config['database']
                )
                cursor = connection.cursor()
                cursor.execute("SELECT tenant_id FROM tenants WHERE LOWER(tenant_name) = LOWER(%s) AND is_active = 1", (tenant_name,))
                result = cursor.fetchone()
                
                cursor.close()
                connection.close()
                
                return result[0] if result else None
            
        except Exception as e:
            print(f"Errore nel recupero tenant_id per '{tenant_name}': {e}")
            return None

    def get_tenant_name_from_id(self, tenant_id: str) -> Optional[str]:
        """
        Scopo: Recupera tenant_name dal tenant_id via query MySQL
        
        Args:
            tenant_id: ID del tenant (UUID)
            
        Returns:
            tenant_name se trovato, None altrimenti
            
        Ultimo aggiornamento: 2025-01-27
        """
        try:
            import yaml
            import mysql.connector
            import os
            
            # Carica configurazione TAG database
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
            config = load_config()
            
            tag_db_config = config.get('tag_database', {})
            if not tag_db_config:
                return None
                
            # Connessione MySQL
            connection = mysql.connector.connect(
                host=tag_db_config['host'],
                port=tag_db_config['port'],
                user=tag_db_config['user'],
                password=tag_db_config['password'],
                database=tag_db_config['database']
            )
            
            cursor = connection.cursor()
            cursor.execute("SELECT tenant_name FROM tenants WHERE tenant_id = %s AND is_active = 1", (tenant_id,))
            result = cursor.fetchone()
            
            cursor.close()
            connection.close()
            
            return result[0] if result else None
            
        except Exception as e:
            print(f"Errore nel recupero tenant_name per ID '{tenant_id}': {e}")
            return None

    def get_tenant_info_from_name(self, tenant_name: str) -> Optional[Dict[str, str]]:
        """
        Scopo: Recupera informazioni complete del tenant (id, slug, name) dal nome
        
        Args:
            tenant_name: Nome del tenant da cercare
            
        Returns:
            Dict con tenant_id, tenant_slug, tenant_name se trovato, None altrimenti
            
        Ultimo aggiornamento: 2025-01-22
        """
        try:
            import yaml
            import mysql.connector
            import os
            
            # Carica configurazione TAG database
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
            config = load_config()
            
            tag_db_config = config.get('tag_database', {})
            if not tag_db_config:
                return None
                
            # Connessione MySQL
            connection = mysql.connector.connect(
                host=tag_db_config['host'],
                port=tag_db_config['port'],
                user=tag_db_config['user'],
                password=tag_db_config['password'],
                database=tag_db_config['database']
            )
            
            cursor = connection.cursor()
            # Cerca per tenant_name o tenant_slug (più robusto)
            cursor.execute("""
                SELECT tenant_id, tenant_slug, tenant_name 
                FROM tenants 
                WHERE (tenant_name = %s OR tenant_slug = %s) AND is_active = 1
            """, (tenant_name, tenant_name))
            result = cursor.fetchone()
            
            cursor.close()
            connection.close()
            
            if result:
                return {
                    'tenant_id': result[0],
                    'tenant_slug': result[1], 
                    'tenant_name': result[2]
                }
            return None
            
        except Exception as e:
            print(f"Errore nel recupero info tenant '{tenant_name}': {e}")
            return None

    def get_collection_name(self) -> str:
        """
        Genera il nome della collection MongoDB per il tenant usando l'oggetto Tenant
        CAMBIO RADICALE: USA SOLO L'OGGETTO TENANT INTERNO
        
        Returns:
            Nome della collection per il tenant: {tenant_slug}_{tenant_id}
            
        Ultimo aggiornamento: 2025-08-29
        """
        # Usa direttamente l'oggetto Tenant per generare nome collection
        safe_tenant_slug = self.tenant.tenant_slug.replace(' ', '_').replace('-', '_').lower()
        
        # FORMATO MONGODB: {tenant_slug}_{tenant_id_completo}
        collection_name = f"{safe_tenant_slug}_{self.tenant.tenant_id}"
        
        print(f"📊 Collection generata: {collection_name} per tenant {self.tenant.tenant_name}")
        return collection_name
    
    def connect(self) -> bool:
        """
        Stabilisce connessione a MongoDB con configurazione pool ottimizzata
        
        Output:
            - True se connesso con successo, False altrimenti
            
        Ultimo aggiornamento: 2025-08-23
        """
        try:
            if not self.mongo_enabled:
                # Avvisa una sola volta per sessione
                if not self._disabled_notice_emitted:
                    print("ℹ️ MongoDB disabilitato o URL non configurato: skip connessione/salvataggio")
                    self._disabled_notice_emitted = True
                return False
            if not self.client:
                # Configurazione pool connessioni per evitare esaurimento
                self.client = MongoClient(
                    self.mongodb_url,
                    maxPoolSize=50,          # Pool più grande
                    minPoolSize=5,           # Connessioni minime sempre aperte
                    maxIdleTimeMS=30000,     # Timeout idle 30 secondi
                    serverSelectionTimeoutMS=5000,  # Timeout selezione server 5 secondi
                    socketTimeoutMS=20000,   # Timeout socket 20 secondi
                    connectTimeoutMS=10000,  # Timeout connessione 10 secondi
                    heartbeatFrequencyMS=10000,  # Heartbeat ogni 10 secondi
                    retryWrites=True,        # Retry automatici su scritture
                    retryReads=True          # Retry automatici su letture
                )
                self.db = self.client[self.database_name]
                
                # Test connessione con ping
                self.client.admin.command('ping')
                print(f"✅ Connesso a MongoDB database: {self.database_name}")
                return True
            else:
                # Verifica che la connessione esistente sia ancora valida
                try:
                    self.client.admin.command('ping')
                    return True
                except Exception:
                    # Connessione morta, chiudi e ricrea
                    self.client.close()
                    self.client = None
                    # Ricorsione singola per riconnessione
                    self.client = MongoClient(
                        self.mongodb_url,
                        maxPoolSize=50, minPoolSize=5, maxIdleTimeMS=30000,
                        serverSelectionTimeoutMS=5000, socketTimeoutMS=20000,
                        connectTimeoutMS=10000, heartbeatFrequencyMS=10000,
                        retryWrites=True, retryReads=True
                    )
                    self.db = self.client[self.database_name]
                    self.client.admin.command('ping')
                    print(f"🔄 Riconnesso a MongoDB database: {self.database_name}")
                    return True
            
        except Exception as e:
            print(f"❌ Errore connessione MongoDB: {e}")
            if self.client:
                try:
                    self.client.close()
                except:
                    pass
                self.client = None
                self.db = None
            return False
    
    def disconnect(self):
        """
        Scopo: Chiude connessione MongoDB in modo sicuro
        Input: Nessuno
        Output: Connessione chiusa
        Ultimo aggiornamento: 2025-08-23
        """
        try:
            if self.client:
                self.client.close()
                self.client = None
                self.db = None
                print("🔌 Connessione MongoDB chiusa correttamente")
        except Exception as e:
            print(f"⚠️ Errore nella chiusura connessione MongoDB: {e}")
            # Force cleanup anche in caso di errore
            self.client = None
            self.db = None
    
    def __enter__(self):
        """Context manager entry"""
        if self.connect():
            return self
        else:
            raise Exception("Impossibile connettersi a MongoDB")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - chiude sempre la connessione"""
        self.disconnect()
    
    def ensure_connection(self) -> bool:
        """
        Assicura che ci sia una connessione attiva, riconnette se necessario
        
        Returns:
            bool: True se connessione disponibile
        """
        if not self.client:
            return self.connect()
        
        try:
            # Test rapido connessione
            self.client.admin.command('ping')
            return True
        except Exception:
            print("⚠️ Connessione MongoDB morta, riconnessione...")
            return self.connect()
    
    def get_all_sessions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scopo: Recupera tutte le sessioni classificate per il tenant
        CAMBIO RADICALE: USA SEMPRE L'OGGETTO TENANT INTERNO
        
        Parametri input:
            - limit: Numero massimo di sessioni da recuperare
            
        Output:
            - Lista di sessioni con classificazioni
            
        Ultimo aggiornamento: 2025-08-29
        """
        # Ensure we're connected to MongoDB
        if not self.ensure_connection():
            print("❌ Impossibile connettersi a MongoDB")
            return []

        try:
            # Usa la collection del tenant
            collection = self.db[self.collection_name]
            
            print(f"📊 Recupero sessioni per tenant: {self.tenant.tenant_name}")
            print(f"   📂 Collection: {self.collection_name}")
            
            # Query semplificata: usa sempre collection per tenant (non serve filtro client)
            query = {"review_status": {"$in": ["auto_classified", "resolved", "not_required"]}}
            
            # Proiezione per escludere embedding (troppo pesante)
            projection = {
                "embedding": 0
            }
            
            # Ordinamento per timestamp di classificazione (più recenti prima)
            sort_order = [("classified_at", -1)]
            
            cursor = collection.find(query, projection).sort(sort_order)
            
            if limit:
                cursor = cursor.limit(limit)
            
            sessions = []
            for doc in cursor:
                # Converti ObjectId in string per JSON
                doc['_id'] = str(doc['_id'])
                
                # Formatta per interfaccia React con nuovi campi ML/LLM
                session = {
                    'id': doc['_id'],
                    'session_id': doc.get('session_id', ''),
                    'conversation_text': doc.get('testo_completo', doc.get('testo', doc.get('conversazione', ''))),
                    'classification': doc.get('classification', doc.get('classificazione', 'non_classificata')),
                    'confidence': doc.get('confidence', 0.0),
                    'motivation': doc.get('motivazione', ''),
                    'notes': doc.get('motivazione', ''),  # Mapping motivazione → notes per UI
                    'method': doc.get('classification_method', doc.get('metadata', {}).get('method', 'unknown')),
                    'processing_time': doc.get('metadata', {}).get('processing_time', 0.0),
                    'timestamp': doc.get('classified_at', doc.get('timestamp')),
                    'tenant_name': self.tenant.tenant_name,  # USA OGGETTO TENANT
                    'tenant_id': self.tenant.tenant_id,      # AGGIUNGI ANCHE TENANT_ID
                    
                    # 🆕 METADATI CLUSTER per UI - AGGIUNTI
                    'metadata': doc.get('metadata', {}),  # Include tutti i metadati MongoDB
                    
                    # NUOVI CAMPI: Dettagli ML/LLM 
                    'ml_prediction': doc.get('ml_prediction', ''),
                    'ml_confidence': doc.get('ml_confidence', 0.0),
                    'llm_prediction': doc.get('llm_prediction', ''),
                    'llm_confidence': doc.get('llm_confidence', 0.0),
                    'llm_reasoning': doc.get('llm_reasoning', ''),
                    'review_status': doc.get('review_status', 'unknown'),
                    
                    'classifications': [{
                        'label': doc.get('classification', doc.get('classificazione', 'non_classificata')),
                        'confidence': doc.get('confidence', 0.0),
                        'method': doc.get('classification_method', doc.get('metadata', {}).get('method', 'unknown')),
                        'motivation': doc.get('motivazione', ''),
                        'ml_prediction': doc.get('ml_prediction', ''),
                        'ml_confidence': doc.get('ml_confidence', 0.0),
                        'llm_prediction': doc.get('llm_prediction', ''),
                        'llm_confidence': doc.get('llm_confidence', 0.0),
                        'created_at': doc.get('classified_at', doc.get('timestamp')),
                        # 🆕 CLUSTER_ID nelle classificazioni - AGGIUNTO
                        'cluster_id': doc.get('metadata', {}).get('cluster_id')
                    }] if doc.get('classification') or doc.get('classificazione') else []
                }
                
                sessions.append(session)
            
            return sessions
            
        except Exception as e:
            print(f"Errore nel recupero sessioni: {e}")
            return []
    
    def get_case_by_id(self, case_id) -> Optional[Dict[str, Any]]:
        """
        Scopo: Recupera un singolo caso per ObjectId
        
        Parametri input:
            - case_id: ObjectId MongoDB del caso (_id)
        
        Output:
            - Dizionario con i dettagli del caso o None se non trovato
            
        Ultimo aggiornamento: 2025-08-30
        """
        try:
            if not self.ensure_connection():
                return None
            
            collection = self.db[self.get_collection_name()]
            
            # Cerca il documento per _id
            case = collection.find_one({'_id': case_id})
            
            if case:
                print(f"✅ Caso trovato: {case_id}")
                return case
            else:
                print(f"❌ Caso non trovato: {case_id}")
                return None
                
        except Exception as e:
            print(f"❌ Errore nel recupero caso {case_id}: {e}")
            return None
    
    def get_available_labels(self) -> List[str]:
        """
        Scopo: Recupera tutte le etichette/classificazioni disponibili per il tenant
        CAMBIO RADICALE: USA SEMPRE L'OGGETTO TENANT INTERNO
        
        Output:
            - Lista di etichette unique per il tenant
            
        Ultimo aggiornamento: 2025-08-29
        """
        try:
            if not self.ensure_connection():
                # Mongo non disponibile/disabilitato → nessuna etichetta
                return []
                
            # Usa la collection del tenant
            collection = self.db[self.collection_name]
            
            print(f"🏷️ Recupero etichette per tenant: {self.tenant.tenant_name}")
            
            # Query semplificata: usa collection per tenant (non serve filtro client)
            filter_query = {}
                
            # Recupera etichette distinct
            labels = collection.distinct("classification", filter_query)
            if not labels:
                # Fallback per campo legacy
                labels = collection.distinct("classificazione", filter_query)
            
            # Rimuovi valori null/vuoti
            labels = [label for label in labels if label and label.strip()]
            
            # Ordina alfabeticamente
            labels.sort()
            
            return labels
            
        except Exception as e:
            print(f"Errore nel recupero etichette: {e}")
            return []
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """
        Scopo: Recupera statistiche sulle classificazioni per il tenant
        CAMBIO RADICALE: USA SEMPRE L'OGGETTO TENANT INTERNO
        
        Output:
            - Dizionario con statistiche dettagliate per il tenant
            
        Ultimo aggiornamento: 2025-08-29
        """
        try:
            if not self.ensure_connection():
                return {}
                
            # Usa la collection del tenant
            collection = self.db[self.collection_name]
            
            print(f"📊 Calcolo statistiche per tenant: {self.tenant.tenant_name}")
            
            # Pipeline semplificata: usa collection per tenant (non serve filtro client)
            pipeline = [
                {"$match": {}},  # Nessun filtro necessario
                {
                    "$group": {
                        "_id": {"$ifNull": ["$classification", "$classificazione"]},  # Supporta entrambi i campi
                        "count": {"$sum": 1},
                        "avg_confidence": {"$avg": "$confidence"},
                        "min_confidence": {"$min": "$confidence"},
                        "max_confidence": {"$max": "$confidence"}
                    }
                },
                {"$sort": {"count": -1}}
            ]
            
            results = list(collection.aggregate(pipeline))
            
            # Calcola statistiche totali
            total_classifications = sum(r['count'] for r in results)
            
            # Formatta risultati
            label_stats = []
            for result in results:
                label_stats.append({
                    'label': result['_id'],
                    'count': result['count'],
                    'percentage': round((result['count'] / total_classifications) * 100, 2),
                    'avg_confidence': round(result['avg_confidence'], 3),
                    'min_confidence': round(result['min_confidence'], 3),
                    'max_confidence': round(result['max_confidence'], 3)
                })
            
            return {
                'total_classifications': total_classifications,
                'unique_labels': len(results),
                'label_distribution': label_stats,
                'tenant_name': self.tenant.tenant_name,
                'tenant_id': self.tenant.tenant_id
            }
            
        except Exception as e:
            print(f"Errore nel calcolo statistiche: {e}")
            return {}
    
    def get_sessions_by_label(self, label: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scopo: Recupera sessioni filtrate per etichetta specifica per il tenant
        CAMBIO RADICALE: USA SEMPRE L'OGGETTO TENANT INTERNO
        
        Parametri input:
            - label: Etichetta da filtrare
            - limit: Numero massimo di risultati
            
        Output:
            - Lista di sessioni con l'etichetta specificata per il tenant
            
        Ultimo aggiornamento: 2025-08-29
        """
        try:
            if not self.ensure_connection():
                return False
                
            # Usa la collection del tenant
            collection = self.db[self.collection_name]
            
            print(f"🔍 Filtraggio sessioni per label '{label}' del tenant: {self.tenant.tenant_name}")
            
            # Query semplificata: filtra per classificazione in collection tenant-specifica
            query = {"classification": label}
                
            # Usa la collection del tenant
            collection = self.db[self.collection_name]
            
            # Proiezione esclude embedding
            projection = {"embedding": 0}
            
            cursor = collection.find(query, projection).sort([("confidence", -1)])
            
            if limit:
                cursor = cursor.limit(limit)
            
            sessions = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                
                session = {
                    'id': doc['_id'],
                    'session_id': doc.get('session_id', ''),
                    'conversation_text': doc.get('testo', doc.get('conversazione', '')),
                    'classification': doc.get('classificazione', ''),
                    'confidence': doc.get('confidence', 0.0),
                    'motivation': doc.get('motivazione', ''),
                    'notes': doc.get('motivazione', ''),  # Mapping motivazione → notes per UI
                    'method': doc.get('metadata', {}).get('method', 'unknown'),
                    'timestamp': doc.get('timestamp'),
                    'classifications': [{
                        'label': doc.get('classificazione', ''),
                        'confidence': doc.get('confidence', 0.0),
                        'method': doc.get('metadata', {}).get('method', 'unknown'),
                        'motivation': doc.get('motivazione', ''),
                        'created_at': doc.get('timestamp')
                    }]
                }
                
                sessions.append(session)
            
            return sessions
            
        except Exception as e:
            print(f"Errore nel filtro per etichetta: {e}")
            return []

    def get_pending_review_sessions(self, client_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scopo: Recupera sessioni che necessitano di review umana
        
        Parametri input:
            - client_name: Nome del cliente
            - limit: Numero massimo di risultati
            
        Output:
            - Lista di sessioni pending review
            
        Ultimo aggiornamento: 2025-08-21
        """
        try:
            if not self.ensure_connection():
                return False
                
            # Usa la collection appropriata per il tenant
            collection = self.db[self.get_collection_name()]
            
            # Query per sessioni che necessitano review
            # Con oggetto Tenant non serve più filtro client
            query = {"review_status": "pending"}
            
            # Proiezione esclude embedding
            projection = {"embedding": 0}
            
            cursor = collection.find(query, projection).sort([("timestamp", -1)])
            
            if limit:
                cursor = cursor.limit(limit)
            
            sessions = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                
                session = {
                    'id': doc['_id'],
                    'case_id': doc['_id'],  # Per compatibilità API
                    'session_id': doc.get('session_id', ''),
                    'conversation_text': doc.get('testo_completo', doc.get('testo', doc.get('conversazione', ''))),
                    'classification': doc.get('classification', doc.get('classificazione', 'non_classificata')),
                    'confidence': doc.get('confidence', 0.0),
                    'motivation': doc.get('motivazione', ''),
                    'notes': doc.get('motivazione', ''),  # Mapping motivazione → notes per UI
                    'method': doc.get('classification_method', doc.get('metadata', {}).get('method', 'unknown')),
                    'timestamp': doc.get('classified_at', doc.get('timestamp')),
                    'tenant_name': doc.get('tenant_name', client_name),
                    'review_status': doc.get('review_status', 'not_required'),
                    'review_reason': doc.get('review_reason', ''),
                    'human_reviewed': doc.get('human_reviewed', False),
                    
                    # NUOVI CAMPI: Dettagli ML/LLM per Review Queue
                    'ml_prediction': doc.get('ml_prediction', ''),
                    'ml_confidence': doc.get('ml_confidence', 0.0),
                    'llm_prediction': doc.get('llm_prediction', ''),
                    'llm_confidence': doc.get('llm_confidence', 0.0),
                    'llm_reasoning': doc.get('llm_reasoning', ''),
                    
                    'human_decision': doc.get('human_decision'),
                    'human_confidence': doc.get('human_confidence'),
                    'human_notes': doc.get('human_notes', ''),
                    'human_reviewed_at': doc.get('human_reviewed_at')
                }
                
                sessions.append(session)
            
            return sessions
            
        except Exception as e:
            print(f"Errore nel recupero sessioni pending review: {e}")
            return []

    def mark_session_for_review(self, session_id: str, client_name: str, review_reason: str, 
                                conversation_text: str = None) -> bool:
        """
        Scopo: Marca una sessione come da revisionare
        
        Parametri input:
            - session_id: ID della sessione
            - client_name: Nome del cliente  
            - review_reason: Motivo della review (low_confidence, disagreement, etc.)
            - conversation_text: Testo della conversazione (opzionale per test)
            
        Output:
            - True se aggiornato con successo
            
        Ultimo aggiornamento: 2025-01-27
        """
        try:
            if not self.ensure_connection():
                return []
                
            # Usa la collection appropriata per il tenant
            collection = self.db[self.get_collection_name()]
            
            # Prepara il filtro per trovare il documento
            # Con oggetto Tenant, filtra solo per session_id
            filter_dict = {"session_id": session_id}
            
            # Prima verifica se il documento esiste
            existing_doc = collection.find_one(filter_dict)
            
            if existing_doc:
                # Aggiorna il documento esistente per richiedere review
                result = collection.update_one(
                    filter_dict,
                    {
                        "$set": {
                            "review_status": "pending",
                            "review_reason": review_reason,
                            "human_reviewed": False,
                            "marked_for_review_at": datetime.now().isoformat()
                        }
                    }
                )
                return result.modified_count > 0
            else:
                # Se il documento non esiste, creane uno nuovo (per test o casi speciali)
                new_doc = {
                    "session_id": session_id,
                    "testo": conversation_text or f"Test conversation for {session_id}",
                    "conversazione": conversation_text or f"Test conversation for {session_id}",
                    "classificazione": "da_classificare",
                    "confidence": 0.0,
                    "motivazione": f"Documento creato per review: {review_reason}",
                    "timestamp": datetime.now().isoformat(),
                    "review_status": "pending",
                    "review_reason": review_reason,
                    "human_reviewed": False,
                    "marked_for_review_at": datetime.now().isoformat(),
                    "metadata": {
                        "method": "manual_review_creation",
                        "processing_time": 0.0
                    }
                }
                
                # Con oggetto Tenant, aggiungi sempre i dati tenant
                new_doc["client"] = self.tenant.tenant_name
                new_doc["tenant_name"] = self.tenant.tenant_name
                new_doc["tenant_id"] = self.tenant.tenant_id
                
                result = collection.insert_one(new_doc)
                return result.inserted_id is not None
            
        except Exception as e:
            print(f"Errore nel marcare sessione per review: {e}")
            return False

    def resolve_review_session(self, case_id: str, client_name: str, human_decision: str, 
                              human_confidence: float, human_notes: str = "") -> bool:
        """
        Scopo: Risolve un caso di review con la decisione umana
        
        Parametri input:
            - case_id: ID del caso (MongoDB _id)
            - client_name: Nome del cliente
            - human_decision: Etichetta scelta dall'umano
            - human_confidence: Confidenza della decisione
            - human_notes: Note aggiuntive
            
        Output:
            - True se risolto con successo
            
        Ultimo aggiornamento: 2025-01-27
        """
        try:
            if not self.ensure_connection():
                return []
                
            # Usa la collection appropriata per il tenant
            collection = self.db[self.get_collection_name()]
            
            from bson import ObjectId
            
            # Converte case_id in ObjectId se necessario
            try:
                object_id = ObjectId(case_id)
            except:
                # Se non è un ObjectId valido, prova come stringa
                object_id = case_id
            
            # Prepara il filtro per trovare il documento
            filter_dict = {
                "_id": object_id,
                "review_status": "pending"
            }
            # CON OGGETTO TENANT: Non serve più filtro client, usiamo collection specifica
            
            # Prima recupera il documento per ottenere la confidence attuale
            existing_doc = collection.find_one(filter_dict, {"confidence": 1})
            current_confidence = existing_doc.get("confidence", 0) if existing_doc else 0
            
            # Aggiorna il documento con la decisione umana
            result = collection.update_one(
                filter_dict,
                {
                    "$set": {
                        "review_status": "completed",
                        "human_reviewed": True,
                        "human_decision": human_decision,
                        "human_confidence": human_confidence,
                        "human_notes": human_notes,
                        "human_reviewed_at": datetime.now().isoformat(),
                        # Aggiorna anche la classificazione principale se diversa
                        "classificazione": human_decision,
                        "confidence": max(human_confidence, current_confidence)
                    }
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            print(f"Errore nella risoluzione caso review: {e}")
            return False

    def resolve_review_session_with_cluster_propagation(self, case_id: str, client_name: str, 
                                                       human_decision: str, human_confidence: float, 
                                                       human_notes: str = "") -> Dict[str, Any]:
        """
        Scopo: Risolve un caso di review e propaga la decisione agli altri membri del cluster se è rappresentante
        
        Parametri input:
            - case_id: ID del caso (MongoDB _id)
            - client_name: Nome del cliente
            - human_decision: Etichetta scelta dall'umano
            - human_confidence: Confidenza della decisione umana
            - human_notes: Note aggiuntive
            
        Output:
            - Dizionario con risultati: {
                "case_resolved": bool,
                "propagated_cases": int,
                "cluster_id": str,
                "is_representative": bool,
                "error": str (se presente)
              }
            
        Ultimo aggiornamento: 2025-08-27
        """
        try:
            if not self.ensure_connection():
                return {"case_resolved": False, "propagated_cases": 0, "error": "Connessione database fallita"}
            
            # Usa la collection appropriata per il tenant
            collection = self.db[self.get_collection_name()]
            
            from bson import ObjectId
            
            # Converte case_id in ObjectId se necessario
            try:
                object_id = ObjectId(case_id)
            except:
                object_id = case_id
            
            # Prepara il filtro per trovare il documento
            filter_dict = {
                "_id": object_id,
                "review_status": "pending"
            }
            # CON OGGETTO TENANT: Non serve più filtro client
            
            # 1. Recupera il documento completo per ottenere i metadati cluster
            case_doc = collection.find_one(filter_dict, {
                "confidence": 1,
                "metadata": 1,
                "session_id": 1,
                "testo_completo": 1,
                "testo": 1,
                "conversazione": 1
            })
            
            if not case_doc:
                return {"case_resolved": False, "propagated_cases": 0, "error": "Caso non trovato o già risolto"}
            
            # Estrai metadati cluster
            metadata = case_doc.get("metadata", {})
            cluster_metadata = metadata.get("cluster_metadata", {})
            is_representative = cluster_metadata.get("is_representative", False)
            cluster_id = cluster_metadata.get("cluster_id")
            current_confidence = case_doc.get("confidence", 0)
            # Estrai session_id e testo conversazione per training log
            resolved_session_id = case_doc.get("session_id")
            resolved_conversation_text = (
                case_doc.get("testo_completo")
                or case_doc.get("testo")
                or case_doc.get("conversazione")
            )
            
            print(f"🔍 Risoluzione caso {case_id}: rappresentante={is_representative}, cluster_id={cluster_id}")
            
            # 2. Risolvi il caso specifico
            case_resolved = self.resolve_review_session(case_id, client_name, human_decision, 
                                                       human_confidence, human_notes)
            
            if not case_resolved:
                return {"case_resolved": False, "propagated_cases": 0, "error": "Errore nella risoluzione del caso"}
            
            propagated_count = 0
            
            # 3. Se è rappresentante e ha cluster_id, propaga la decisione
            if is_representative and cluster_id:
                print(f"🔄 Propagazione decisione '{human_decision}' a membri cluster {cluster_id}")
                
                # Trova tutti i membri del cluster non ancora revisionati (escluso il rappresentante)
                cluster_filter = {
                    "metadata.cluster_metadata.cluster_id": cluster_id,
                    "review_status": {"$ne": "completed"},
                    "_id": {"$ne": object_id}  # Escludi il rappresentante già aggiornato
                }
                # CON OGGETTO TENANT: Non serve più filtro client, usiamo collection specifica
                
                # Applica la propagazione con confidenza leggermente ridotta
                propagation_confidence = max(human_confidence * 0.9, 0.1)  # Minimo 0.1
                current_timestamp = datetime.now().isoformat()
                
                propagation_result = collection.update_many(
                    cluster_filter,
                    {
                        "$set": {
                            "classificazione": human_decision,
                            "confidence": propagation_confidence,
                            "metadata.cluster_metadata.propagated_from_human_review": True,
                            "metadata.cluster_metadata.human_review_case_id": str(object_id),
                            "metadata.cluster_metadata.human_review_propagated_at": current_timestamp,
                            "metadata.cluster_metadata.human_review_decision": human_decision,
                            "metadata.cluster_metadata.human_review_confidence": human_confidence,
                            "human_reviewed_by_propagation": True,
                            "human_review_propagation_notes": f"Propagato da rappresentante {case_id}: {human_notes}" if human_notes else f"Propagato da rappresentante {case_id}"
                        }
                    }
                )
                
                propagated_count = propagation_result.modified_count
                print(f"✅ Propagata decisione a {propagated_count} membri del cluster {cluster_id}")
                
                # Log dettagliato per debugging
                if propagated_count > 0:
                    print(f"   📊 Cluster {cluster_id}: rappresentante={case_id}, propagati={propagated_count}")
                    print(f"   🎯 Decisione: '{human_decision}' (confidenza: {human_confidence} → {propagation_confidence})")
            
            return {
                "case_resolved": True,
                "propagated_cases": propagated_count,
                "cluster_id": cluster_id,
                "is_representative": is_representative,
                "human_decision": human_decision,
                "human_confidence": human_confidence,
                "session_id": resolved_session_id,
                "conversation_text": resolved_conversation_text
            }
            
        except Exception as e:
            error_msg = f"Errore nella risoluzione con propagazione: {e}"
            print(error_msg)
            import traceback
            print(f"Stack trace: {traceback.format_exc()}")
            return {"case_resolved": False, "propagated_cases": 0, "error": error_msg}

    def get_review_statistics(self, client_name: str) -> Dict[str, Any]:
        """
        Scopo: Recupera statistiche sui casi di review
        
        Parametri input:
            - client_name: Nome del cliente
            
        Output:
            - Dizionario con statistiche review
            
        Ultimo aggiornamento: 2025-01-27
        """
        try:
            if not self.ensure_connection():
                return []
                
            # Usa la collection appropriata per il tenant
            collection = self.db[self.get_collection_name()]
            
            # Pipeline di aggregazione per statistiche review
            # CON OGGETTO TENANT: Usa sempre collection tenant-specifica
            pipeline = [
                {
                    "$group": {
                        "_id": "$review_status",
                        "count": {"$sum": 1}
                    }
                }
            ]
            
            results = list(collection.aggregate(pipeline))
            
            stats = {
                "pending": 0,
                "completed": 0,
                "not_required": 0,
                "total": 0
            }
            
            for result in results:
                status = result["_id"] or "not_required"
                count = result["count"]
                stats[status] = count
                stats["total"] += count
            
            return stats
            
        except Exception as e:
            print(f"Errore nel calcolo statistiche review: {e}")
            return {"pending": 0, "completed": 0, "not_required": 0, "total": 0}

    @classmethod
    def get_available_tenants(cls) -> List[Tenant]:
        """
        Scopo: Recupera tutti i tenant disponibili e restituisce oggetti Tenant completi
        CORREZIONE FONDAMENTALE: Questa è l'UNICA funzione che genera oggetti Tenant 
        perché non ha bisogno di ricevere un tenant - è quella che LI CREA
        
        Output:
            - Lista di oggetti Tenant completi con tutti i dati necessari
            
        Ultimo aggiornamento: 2025-11-08 - FIX config_loader
        """
        try:
            # Importa le librerie necessarie
            import mysql.connector
            from mysql.connector import Error
            
            # Carica configurazione TAG database con config_loader
            config = load_config()
            tag_db_config = config.get('tag_database', {})
            
            # Connessione diretta al database TAG
            connection = mysql.connector.connect(
                host=tag_db_config['host'],
                port=tag_db_config['port'],
                user=tag_db_config['user'],
                password=tag_db_config['password'],
                database=tag_db_config['database']
            )
            
            cursor = connection.cursor()
            
            # Query per recuperare i tenant completi dalla tabella tenants
            query = """
                SELECT tenant_id, tenant_name, tenant_slug, is_active
                FROM tenants 
                WHERE is_active = 1
                ORDER BY tenant_name
            """
            cursor.execute(query)
            risultati = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            tenant_objects = []
            if risultati:
                for tenant_id, tenant_name, tenant_slug, is_active in risultati:
                    # Crea oggetto Tenant completo
                    tenant = Tenant(
                        tenant_id=str(tenant_id),
                        tenant_name=tenant_name,
                        tenant_slug=tenant_slug,
                        tenant_database=tenant_slug,  # Per ora, tenant_database = tenant_slug
                        tenant_status=is_active
                    )
                    tenant_objects.append(tenant)
            
            print(f"🏢 Recuperati {len(tenant_objects)} oggetti Tenant dalla tabella TAG.tenants")
            return tenant_objects
            
        except Exception as e:
            print(f"❌ Errore nel recupero tenant da MySQL: {e}")
            return []

    def generate_collection_name(self, tenant_name: str) -> str:
        """
        Scopo: Genera nome collezione MongoDB nel formato {tenant_slug}_{tenant_id}
        
        Args:
            tenant_name: Nome del tenant (slug)
            
        Returns:
            Nome collezione nel formato: {tenant_slug}_{tenant_id}
            
        Ultimo aggiornamento: 2025-08-23
        """
        try:
            # Ottieni tenant_id dal nome
            tenant_id = self.get_tenant_id_from_name(tenant_name)
            
            # Sanitizza il nome tenant per uso in filename
            safe_tenant_name = tenant_name.replace(' ', '_').replace('-', '_').lower()
            
            if tenant_id:
                # FORMATO MONGODB: {tenant_slug}_{tenant_id_completo}
                # USARE L'ID COMPLETO, NON TRONCATO!
                return f"{safe_tenant_name}_{tenant_id}"
            else:
                # Fallback: solo nome
                print(f"⚠️ Tenant ID non trovato per '{tenant_name}', uso solo nome")
                return f"{safe_tenant_name}"
                
        except Exception as e:
            print(f"Errore nella generazione nome collezione: {e}")
            return f"unknown_tenant"

    def generate_model_name(self, tenant_name: str, model_type: str = "classifier", timestamp: str = None) -> str:
        """
        Scopo: Genera nome modello nel formato {tenant_slug}_{tenant_id_short}_{timestamp}
        
        Args:
            tenant_name: Nome del tenant (slug)
            model_type: Tipo di modello (compatibilità backward, ignorato)
            timestamp: Timestamp personalizzato (opzionale, default: now)
            
        Returns:
            Nome modello nel formato: {tenant_slug}_{tenant_id_short}_{timestamp}
            
        Ultimo aggiornamento: 2025-08-23
        """
        try:
            # Ottieni tenant_id dal nome
            tenant_id = self.get_tenant_id_from_name(tenant_name)
            
            # Genera timestamp se non fornito
            if not timestamp:
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Sanitizza il nome tenant per uso in filename
            safe_tenant_name = tenant_name.replace(' ', '_').replace('-', '_').lower()
            
            if tenant_id:
                # FORMATO MODELLI: {tenant_slug}_{tenant_id_short}_{timestamp}
                # Per i modelli usiamo ID corto per evitare nomi troppo lunghi
                short_id = tenant_id[:8]
                return f"{safe_tenant_name}_{short_id}_{timestamp}"
            else:
                # Fallback: nome_timestamp
                print(f"⚠️ Tenant ID non trovato per '{tenant_name}', uso solo nome")
                return f"{safe_tenant_name}_{timestamp}"
                
        except Exception as e:
            print(f"Errore nella generazione nome modello: {e}")
            return f"unknown_tenant_{timestamp or 'unknown'}"

    def generate_semantic_cache_path(self, tenant_name: str, cache_type: str = "memory") -> str:
        """
        Scopo: Genera percorso cache semantica tenant-aware
        
        Args:
            tenant_name: Nome del tenant
            cache_type: Tipo di cache (memory, embeddings, etc.)
            
        Returns:
            Percorso cache nel formato: semantic_cache/{tenant_name}_{tenant_id}_{cache_type}/
            
        Ultimo aggiornamento: 2025-01-27
        """
        try:
            # Ottieni tenant_id dal nome
            tenant_id = self.get_tenant_id_from_name(tenant_name)
            
            # Sanitizza il nome tenant per uso in path
            safe_tenant_name = tenant_name.replace(' ', '_').replace('-', '_').lower()
            
            if tenant_id:
                # Usa prime 8 caratteri dell'ID per brevità
                short_id = tenant_id[:8]
                folder_name = f"{safe_tenant_name}_{short_id}_{cache_type}"
            else:
                # Fallback: solo nome_tipo
                print(f"⚠️ Tenant ID non trovato per '{tenant_name}', uso solo nome per cache")
                folder_name = f"{safe_tenant_name}_{cache_type}"
            
            return os.path.join("semantic_cache", folder_name)
                
        except Exception as e:
            print(f"Errore nella generazione path cache semantica: {e}")
            # Fallback di sicurezza
            return os.path.join("semantic_cache", f"unknown_tenant_{cache_type}")

    def save_classification_result(self, session_id: str, client_name: str, 
                                   ml_result: dict = None, llm_result: dict = None,
                                   final_decision: dict = None, conversation_text: str = None,
                                   needs_review: bool = False, review_reason: str = None,
                                   classified_by: str = None, notes: str = None,
                                   cluster_metadata: dict = None,
                                   embedding: Union[np.ndarray, List[float]] = None,
                                   embedding_model: str = None) -> bool:
        """
        Scopo: Salva il risultato completo di una classificazione in MongoDB
        
        Parametri input:
            - session_id: ID della sessione
            - client_name: Nome del cliente
            - ml_result: Risultato del classificatore ML (predicted_label, confidence)
            - llm_result: Risultato del classificatore LLM (predicted_label, confidence)
            - final_decision: Decisione finale del sistema (label, confidence, method)
            - conversation_text: Testo della conversazione
            - needs_review: Se True, marca per review, altrimenti auto_classified
            - review_reason: Motivo della review (se needs_review=True)
            - classified_by: Chi/cosa ha fatto la classificazione (es: 'human_supervisor', 'ensemble_classifier')
            - notes: Note aggiuntive separate dalla motivazione
            - cluster_metadata: Metadati del clustering (cluster_id, is_representative, propagated_from, etc.)
            - embedding: Array numpy o lista dei valori embedding del testo (🆕)
            - embedding_model: Nome del modello utilizzato per generare l'embedding (🆕)
            
        Output:
            - True se salvato con successo
            
        Ultimo aggiornamento: 2025-09-02 - Aggiunto supporto embedding
        """
        trace_all("save_classification_result", "ENTER",
                 session_id=session_id,
                 client_name=client_name,
                 has_ml_result=ml_result is not None,
                 has_llm_result=llm_result is not None,
                 has_final_decision=final_decision is not None,
                 needs_review=needs_review,
                 classified_by=classified_by)
        # � DEBUG CLASSIFIED_BY: Traccia TUTTI i parametri ricevuti dalla funzione
        print(f"🚨 [MONGO-DEBUG] save_classification_result chiamata per session {session_id}:")
        print(f"   📋 client_name: '{client_name}'")
        print(f"   📋 classified_by: '{classified_by}' (type: {type(classified_by)})")
        print(f"   📋 cluster_metadata: {cluster_metadata is not None} ({type(cluster_metadata)})")
        if cluster_metadata:
            print(f"   📋 cluster_metadata keys: {list(cluster_metadata.keys())}")
        print(f"   📋 final_decision: {final_decision is not None}")
        if final_decision:
            print(f"   📋 final_decision method: '{final_decision.get('method', 'N/A')}'")
        print(f"   📋 needs_review: {needs_review}")
        print(f"   📋 review_reason: '{review_reason}'")
        
        # 🔧 RISOLUZIONE TENANT: garantisce disponibilità tenant_obj anche per legacy call-site
        tenant_obj = getattr(self, 'tenant', None)
        if tenant_obj is None:
            try:
                resolved_info = self.get_tenant_info_from_name(client_name)
            except Exception as tenant_lookup_error:
                raise RuntimeError(
                    f"Impossibile risolvere tenant '{client_name}': {tenant_lookup_error}"
                ) from tenant_lookup_error

            if not resolved_info:
                raise RuntimeError(
                    f"Tenant '{client_name}' non trovato nel database TAG locale"
                )

            tenant_obj = Tenant(
                tenant_id=resolved_info['tenant_id'],
                tenant_name=resolved_info['tenant_name'],
                tenant_slug=resolved_info['tenant_slug'],
                tenant_database=resolved_info.get('tenant_database', resolved_info['tenant_slug']),
                tenant_status=resolved_info.get('tenant_status', 1)
            )
            self.tenant = tenant_obj

        tenant_id = tenant_obj.tenant_id
        tenant_name = tenant_obj.tenant_name
        tenant_slug = tenant_obj.tenant_slug

        # � DEBUG STACK TRACE: Se classified_by è None o unknown, traccia chi ha chiamato
        if not classified_by or classified_by == "unknown":
            print(f"🚨 [STACK-TRACE] CLASSIFIED_BY='{classified_by}' per session {session_id}")
            import traceback
            print(f"🚨 [STACK-TRACE] Call stack:")
            stack = traceback.format_stack()
            # Mostra solo le ultime 5 chiamate per non intasare il log
            for i, line in enumerate(stack[-6:-1]):  # Esclude questa riga
                print(f"   📋 #{i+1}: {line.strip()}")
        
        # �🔍 DEBUG: Import utility debug e trace del salvataggio
        try:
            import sys
            import os
            pipeline_path = os.path.join(os.path.dirname(__file__), 'Pipeline')
            if pipeline_path not in sys.path:
                sys.path.append(pipeline_path)
            from debug_pipeline import debug_metadata, debug_pipeline
            
            debug_metadata(session_id, cluster_metadata, classified_by or "unknown", "save_classification_result")
            
            debug_pipeline("save_classification_result", f"SAVING session {session_id}", {
                "client_name": client_name,
                "classified_by": classified_by,
                "has_cluster_metadata": bool(cluster_metadata),
                "cluster_metadata_content": cluster_metadata if cluster_metadata else "None"
            }, "INFO")
            
        except ImportError as e:
            print(f"⚠️ DEBUG: Impossibile importare debug_pipeline: {e}")
            # Continua senza debug se il modulo non è disponibile
        
        try:
            if not self.ensure_connection():
                return False
            
            # Usa la collection appropriata per il tenant (ora usa l'oggetto tenant interno)
            collection = self.db[self.get_collection_name()]
            tenant_obj = self.tenant
            
            # Prepara il documento base con tutti i campi necessari
            current_time = datetime.now().isoformat()
            doc = {
                "session_id": session_id,
                "classified_at": current_time,
                "created_at": current_time,  # 🆕 Campo MySQL
                "updated_at": current_time,  # 🆕 Campo MySQL
                "human_reviewed": False,
                "classified_by": classified_by or "system",  # 🆕 Campo MySQL
                "notes": notes or ""  # 🆕 Campo MySQL (separato da motivazione)
            }
            
            # SEMPRE aggiungi tenant info come chiave univoca di ricerca
            if tenant_obj is not None:
                doc["tenant_id"] = getattr(tenant_obj, 'tenant_id', tenant_id or tenant_slug)
                doc["tenant_name"] = getattr(tenant_obj, 'tenant_name', tenant_name)
                doc["tenant_slug"] = getattr(tenant_obj, 'tenant_slug', tenant_slug)
            else:
                doc["tenant_id"] = tenant_id or tenant_slug
                doc["tenant_name"] = tenant_name
                doc["tenant_slug"] = tenant_slug
            doc["client"] = client_name or tenant_slug
            
            # Aggiunge testo conversazione se fornito
            if conversation_text:
                doc["testo_completo"] = conversation_text
            
            # 🆕 AGGIUNGE EMBEDDING E MODELLO SE FORNITI O RECUPERATI DALLO STORE
            if embedding is not None:
                # Converte embedding in lista se è numpy array
                if isinstance(embedding, np.ndarray):
                    doc["embedding"] = embedding.tolist()
                else:
                    doc["embedding"] = list(embedding)
                    
                print(f"💾 [EMBEDDING] Salvato embedding per session {session_id}: {len(doc['embedding'])} dimensioni")
            
            # Se NON fornito, prova a recuperare embedding dallo store centralizzato (tenant-aware)
            if embedding is None:
                try:
                    import sys, os
                    # Prova a risolvere sia dal CWD che dalla root progetto
                    sys.path.append(os.path.join(os.path.dirname(__file__), 'MongoDB'))
                    from embedding_store import EmbeddingStore  # type: ignore
                except Exception:
                    try:
                        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MongoDB'))
                        from embedding_store import EmbeddingStore  # type: ignore
                    except Exception:
                        EmbeddingStore = None

                if EmbeddingStore is not None:
                    try:
                        store = EmbeddingStore()
                        fetched = store.get_embedding(doc.get("tenant_id"), session_id, consume=True)
                        if fetched:
                            fetched_emb, fetched_model = fetched
                            doc["embedding"] = fetched_emb
                            doc["embedding_model"] = fetched_model or "unknown_embedder"
                            print(f"💾 [EMBED STORE] Recuperato embedding per session {session_id} ({len(fetched_emb)}D)")
                        else:
                            print(f"ℹ️ [EMBED STORE] Nessun embedding in cache per session {session_id}")
                    except Exception as store_error:
                        print(f"⚠️ [EMBED STORE] Errore recupero embedding per {session_id}: {store_error}")

            # Aggiunge modello embedding se fornito esplicitamente
            if embedding_model and "embedding_model" not in doc:
                doc["embedding_model"] = embedding_model
                print(f"🏷️ [EMBEDDING] Modello: {embedding_model}")
            else:
                # Se non specificato ma c'è embedding e manca embedding_model, usa default
                if doc.get("embedding") is not None and "embedding_model" not in doc:
                    doc["embedding_model"] = "unknown"
                    print(f"⚠️ [EMBEDDING] Modello non specificato, usato 'unknown'")
            
            # Helper normalizzazione etichette (uppercase + underscore, rimuove speciali)
            def _normalize_label_local(label_value: Any) -> str:
                try:
                    label_str = str(label_value or "").strip()
                except Exception:
                    label_str = ""
                if not label_str:
                    return ""
                import re as _re
                label_str = label_str.replace('\\', ' ')
                label_str = _re.sub(r'[\"\']', '', label_str)
                label_str = _re.sub(r'[^A-Za-z0-9_\s]', '_', label_str)
                label_str = _re.sub(r'\s+', '_', label_str)
                label_str = _re.sub(r'_+', '_', label_str).strip('_')
                return label_str.upper()

            # Aggiunge risultati ML se forniti (normalizzati)
            if ml_result and ml_result.get("predicted_label"):
                _ml_label = _normalize_label_local(ml_result.get("predicted_label", ""))
                doc["ml_prediction"] = _ml_label
                doc["ml_confidence"] = float(ml_result.get("confidence", 0.0))
                # NUOVO: Aggiungi campi separati per tracciabilità
                doc["classification_ML"] = _ml_label
                doc["precision_ML"] = float(ml_result.get("confidence", 0.0))
            else:
                # 🚨 FIX: Anche quando ML non disponibile, salva campi vuoti per coerenza UI
                doc["ml_prediction"] = ""
                doc["ml_confidence"] = 0.0
            
            # Aggiunge risultati LLM se forniti (normalizzati)
            if llm_result and llm_result.get("predicted_label"):
                _llm_label = _normalize_label_local(llm_result.get("predicted_label", ""))
                doc["llm_prediction"] = _llm_label
                doc["llm_confidence"] = float(llm_result.get("confidence", 0.0))
                # NUOVO: Aggiungi campi separati per tracciabilità
                doc["classification_LLM"] = _llm_label
                doc["precision_LLM"] = float(llm_result.get("confidence", 0.0))
                if "reasoning" in llm_result:
                    doc["llm_reasoning"] = llm_result["reasoning"]
            else:
                # 🚨 FIX: Anche quando LLM non disponibile, salva campi vuoti per coerenza UI
                doc["llm_prediction"] = ""
                doc["llm_confidence"] = 0.0
            
            # 🆕 CALCOLA E SALVA DISAGREEMENT METRICS
            if ml_result and llm_result:
                ml_label = _normalize_label_local(ml_result.get("predicted_label", ""))
                llm_label = _normalize_label_local(llm_result.get("predicted_label", ""))
                ml_conf = float(ml_result.get("confidence", 0.0))
                llm_conf = float(llm_result.get("confidence", 0.0))
                
                # Disagreement binario (etichette diverse)
                has_disagreement = (ml_label != llm_label) if (ml_label and llm_label) else False
                doc["disagreement"] = has_disagreement
                
                # Disagreement confidenza (differenza assoluta tra confidenze)
                confidence_disagreement = abs(ml_conf - llm_conf) if (ml_conf > 0 and llm_conf > 0) else 0.0
                doc["confidence_disagreement"] = confidence_disagreement
                
                # Disagreement score complessivo (combina etichette + confidenze)
                if has_disagreement:
                    # Se etichette diverse, disagreement = 1.0 (massimo)
                    disagreement_score = 1.0
                else:
                    # Se etichette uguali, disagreement = differenza confidenze normalizzata
                    disagreement_score = confidence_disagreement
                doc["disagreement_score"] = disagreement_score
                
            else:
                # Solo un modello disponibile - nessun disagreement
                doc["disagreement"] = False
                doc["confidence_disagreement"] = 0.0
                doc["disagreement_score"] = 0.0
            
            # Aggiunge decisione finale
            if final_decision:
                _final_label = _normalize_label_local(final_decision.get("predicted_label", ""))
                doc["classification"] = _final_label
                doc["predicted_label"] = _final_label  # 🔧 FIX: Campo per interfaccia
                doc["confidence"] = float(final_decision.get("confidence", 0.0))
                doc["classification_method"] = final_decision.get("method", "unknown")
                if "reasoning" in final_decision:
                    doc["motivazione"] = final_decision["reasoning"]
                
                # 🚨 FIX CRITICO: Estrai anche ml_prediction e llm_prediction da final_decision
                # Questo serve per i documenti propagati che non hanno ml_result/llm_result separati
                if "ml_prediction" in final_decision and not ml_result:
                    doc["ml_prediction"] = _normalize_label_local(final_decision.get("ml_prediction", ""))
                    doc["ml_confidence"] = float(final_decision.get("ml_confidence", 0.0)) if final_decision.get("ml_confidence") is not None else 0.0
                
                if "llm_prediction" in final_decision and not llm_result:
                    doc["llm_prediction"] = _normalize_label_local(final_decision.get("llm_prediction", ""))
                    doc["llm_confidence"] = float(final_decision.get("llm_confidence", 0.0)) if final_decision.get("llm_confidence") is not None else 0.0
            
            # Determina review status
            if needs_review:
                doc["review_status"] = "pending"
                doc["review_reason"] = review_reason or "quality_gate_decision"
                doc["marked_for_review_at"] = datetime.now().isoformat()
            else:
                doc["review_status"] = "auto_classified"
            
            # 🆕 AGGIUNGI METADATI CLUSTER per Review Queue
            if cluster_metadata:
                print(f"🐛 DEBUG save_classification_result - cluster_metadata ricevuto: {cluster_metadata}")
                
                # Crea sezione metadata se non esiste
                if "metadata" not in doc:
                    doc["metadata"] = {}
                
                # Salva informazioni cluster (converti numpy types a Python native)
                if "cluster_id" in cluster_metadata:
                    # Converte numpy.int64 a int Python per compatibilità MongoDB
                    cluster_id_value = cluster_metadata["cluster_id"]
                    if hasattr(cluster_id_value, 'item'):  # numpy type
                        doc["metadata"]["cluster_id"] = int(cluster_id_value.item())
                    else:
                        doc["metadata"]["cluster_id"] = int(cluster_id_value)
                    
                if "is_representative" in cluster_metadata:
                    doc["metadata"]["is_representative"] = bool(cluster_metadata["is_representative"])
                    print(f"🐛 DEBUG save_classification_result - SALVATO is_representative: {bool(cluster_metadata['is_representative'])}")
                    
                if "propagated_from" in cluster_metadata:
                    doc["metadata"]["propagated_from"] = str(cluster_metadata["propagated_from"])
                    
                if "propagation_confidence" in cluster_metadata:
                    # Converte numpy.float64 a float Python se necessario
                    confidence_value = cluster_metadata["propagation_confidence"]
                    if hasattr(confidence_value, 'item'):  # numpy type
                        doc["metadata"]["propagation_confidence"] = float(confidence_value.item())
                    else:
                        doc["metadata"]["propagation_confidence"] = float(confidence_value)
                
                # 🆕 METADATI BOOLEANI STRUTTURATI (SEMPRE PRESENTI)
                # Determina i flag in base ai metadati del cluster
                is_representative = cluster_metadata.get("is_representative", False)
                cluster_id = cluster_metadata.get("cluster_id", None)
                propagated_from = cluster_metadata.get("propagated_from", None)
                
                # Logica per determinare i tipi
                is_outlier = (cluster_id == -1 or cluster_id == "-1")
                is_propagated = bool(propagated_from) or (classified_by == "cluster_propagation")
                
                # Salva i metadati booleani strutturati
                doc["metadata"]["representative"] = bool(is_representative)
                doc["metadata"]["outlier"] = bool(is_outlier)
                doc["metadata"]["propagated"] = bool(is_propagated)
                
                print(f"🏷️ METADATI BOOLEANI AGGIUNTI: representative={is_representative}, outlier={is_outlier}, propagated={is_propagated}")
                
                print(f"🐛 DEBUG save_classification_result - doc[metadata] finale: {doc['metadata']}")
                
                # Log per debug
                print(f"🏷️  Salvati metadati cluster per sessione {session_id}: cluster_id={cluster_metadata.get('cluster_id', 'N/A')}, is_representative={cluster_metadata.get('is_representative', False)}")
                
                # 🆕 DETERMINA CLASSIFICATION_TYPE per Review Queue
                # FIX: Usa doc["metadata"] (metadati salvati nel DB) invece di cluster_metadata (parametro originale)
                saved_metadata = doc.get("metadata", {})
                print(f"🐛 DEBUG save_classification_result - saved_metadata passato a _determine_classification_type: {saved_metadata}")
                classification_type = self._determine_classification_type(saved_metadata)
                doc["classification_type"] = classification_type
                print(f"🎯 Classification type determinato: {classification_type}")
                
            else:
                # � PUNTO CRITICO: cluster_metadata è None - qui si attiva LLM_STRUCTURED!
                print(f"🚨 [DEBUG CRITICO] cluster_metadata=None per session_id={session_id}")
                print(f"🚨 [DEBUG CRITICO] classified_by='{classified_by}'")
                print(f"🚨 [DEBUG CRITICO] QUESTO CAUSERÀ LLM_STRUCTURED!")
                
                # �🔧 FIX CRITICO: cluster_metadata None può significare:
                # 1. Classificazione normale (senza clustering) - NON è outlier
                # 2. Classificazione ottimizzata ma fallback - potrebbe essere outlier
                # 
                # STRATEGIA: Controlla se il metodo di classificazione indica clustering
                if "metadata" not in doc:
                    doc["metadata"] = {}
                
                # Determina se si tratta di classificazione con clustering basandosi sul classified_by
                classified_by = classified_by or "unknown"
                is_cluster_based = any(keyword in classified_by.lower() for keyword in [
                    'cluster', 'optimized', 'ens_pipe', 'ensemble'
                ])
                
                print(f"🚨 [DEBUG CRITICO] CONTROLLO CLUSTER-BASED per session {session_id}:")
                print(f"   📋 classified_by: '{classified_by}' (type: {type(classified_by)})")
                print(f"   📋 is_cluster_based: {is_cluster_based}")
                print(f"   📋 cluster_metadata: {cluster_metadata is not None}")
                print(f"   📋 keywords check: cluster={('cluster' in classified_by.lower())}, optimized={('optimized' in classified_by.lower())}, ens_pipe={('ens_pipe' in classified_by.lower())}, ensemble={('ensemble' in classified_by.lower())}")
                
                if is_cluster_based:
                    # 🎯 CLUSTERING-BASED: Sessione senza cluster metadata = outlier
                    print(f"🚨 [DEBUG CRITICO] ASSEGNAZIONE AUTOMATICA A OUTLIER per {session_id}")
                    outlier_counter = self._get_next_outlier_counter()
                    outlier_cluster_id = f"outlier_{outlier_counter}"
                    
                    doc["metadata"]["cluster_id"] = outlier_cluster_id
                    doc["metadata"]["is_representative"] = False
                    doc["metadata"]["outlier_score"] = 1.0
                    doc["metadata"]["method"] = "auto_outlier_assignment"
                    doc["session_type"] = "outlier"
                    doc["classification_type"] = "OUTLIER"  # 🆕 EXPLICIT TYPE
                    
                    # 🆕 METADATI BOOLEANI PER OUTLIER AUTOMATICO
                    doc["metadata"]["representative"] = False
                    doc["metadata"]["outlier"] = True
                    doc["metadata"]["propagated"] = False
                    
                    print(f"🎯 CLUSTERING OUTLIER DETECTED: Sessione {session_id} classificata come outlier (metodo={classified_by}) → cluster {outlier_cluster_id}")
                    print(f"🏷️ METADATI BOOLEANI OUTLIER: representative=False, outlier=True, propagated=False")
                else:
                    # 🔍 CLASSIFICAZIONE LLM DIRETTA (non cluster-based)
                    doc["classification_type"] = "NORMALE"  # 🆕 EXPLICIT TYPE
                    
                    # 🆕 METADATI BOOLEANI PER CLASSIFICAZIONE NORMALE
                    doc["metadata"]["representative"] = False
                    doc["metadata"]["outlier"] = False
                    doc["metadata"]["propagated"] = False
                    
                    print(f"🤖 LLM DIRECT CLASSIFICATION: Sessione {session_id} classificata tramite LLM diretto (classificazione={classified_by})")
                    print(f"🏷️ METADATI BOOLEANI NORMALE: representative=False, outlier=False, propagated=False")
                    # Non aggiungere cluster_id per classificazioni normali
            
            # 🆕 AGGIUNGI SEMPRE session_type basato sui metadata per consistenza UI
            if "session_type" not in doc:
                if doc.get("metadata", {}).get("is_representative", False):
                    doc["session_type"] = "representative"
                elif doc.get("metadata", {}).get("propagated_from"):
                    doc["session_type"] = "propagated"
                elif doc.get("metadata", {}).get("cluster_id"):
                    # Ha cluster_id = è parte di un sistema di clustering
                    doc["session_type"] = "outlier"
                else:
                    # Nessun cluster metadata = classificazione normale
                    doc["session_type"] = "normal"
            
            # Prepara il filtro per upsert usando tenant_id come chiave univoca
            filter_dict = {"session_id": session_id}
            if tenant_obj is not None and getattr(tenant_obj, 'tenant_id', None):
                filter_dict["tenant_id"] = tenant_obj.tenant_id
            elif tenant_id:
                filter_dict["tenant_id"] = tenant_id
            else:
                filter_dict["tenant_slug"] = tenant_slug
            
            # Upsert del documento con aggiornamento updated_at
            doc["updated_at"] = datetime.now().isoformat()  # Aggiorna timestamp
            
            result = collection.update_one(
                filter_dict,
                {"$set": doc},
                upsert=True
            )
            
            success = result.upserted_id is not None or result.modified_count > 0
            trace_all("save_classification_result", "EXIT", 
                     return_value=success,
                     upserted=result.upserted_id is not None,
                     modified=result.modified_count > 0)
            return success
            
        except Exception as e:
            trace_all("save_classification_result", "ERROR", 
                     error=str(e), error_type=type(e).__name__)
            print(f"Errore nel salvataggio risultato classificazione: {e}")
            return False

    def _get_next_outlier_counter(self) -> int:
        """
        Scopo: Genera il prossimo numero progressivo per outlier_X
        
        Output:
            - Numero progressivo per il prossimo outlier
            
        Ultimo aggiornamento: 2025-08-25
        """
        try:
            if not self.ensure_connection():
                return 1
                
            collection = self.db[self.get_collection_name()]
            
            # Trova tutti i cluster_id che iniziano con "outlier_"
            pipeline = [
                {"$match": {"metadata.cluster_id": {"$regex": "^outlier_"}}},
                {"$project": {"cluster_id": "$metadata.cluster_id"}},
                {"$group": {"_id": "$cluster_id"}}
            ]
            
            existing_outliers = list(collection.aggregate(pipeline))
            
            # Estrai i numeri dai cluster_id esistenti
            max_counter = 0
            for outlier_doc in existing_outliers:
                cluster_id = outlier_doc.get('_id', '')
                if cluster_id.startswith('outlier_'):
                    try:
                        counter = int(cluster_id.replace('outlier_', ''))
                        max_counter = max(max_counter, counter)
                    except ValueError:
                        continue
            
            return max_counter + 1
            
        except Exception as e:
            print(f"Errore nel calcolo outlier counter: {e}")
            return 1

    def get_review_queue_sessions(self, client_name: str, limit: int = 100, 
                                  label_filter: str = None,
                                  show_representatives: bool = True,
                                  show_propagated: bool = True, 
                                  show_outliers: bool = True) -> List[Dict[str, Any]]:
        """
        Scopo: Recupera sessioni per Review Queue a 3 livelli (Rappresentanti/Propagate/Outliers)
        OTTIMIZZAZIONE PERFORMANCE: Usa singola query + filtri in-memory invece di complessi $or
        
        Strategia ottimizzata:
        1. Singola query MongoDB semplice: {"review_status": "pending"}
        2. Filtri booleani applicati in-memory durante l'iterazione
        3. Early termination quando raggiunto il limite
        4. Logging minimale per evitare overhead I/O
        
        Parametri input:
            - client_name: Nome del cliente
            - limit: Numero massimo di risultati
            - label_filter: Filtro per etichetta specifica
            - show_representatives: Include rappresentanti di cluster (metadata.representative=True)
            - show_propagated: Include conversazioni propagate dai cluster (metadata.propagated=True)
            - show_outliers: Include outliers (metadata.outlier=True)
            
        Output:
            - Lista di sessioni filtrate per Review Queue (solo status "pending")
            
        Ultimo aggiornamento: 2025-09-06 - Valerio Bignardi - Ottimizzazione performance con single-query + in-memory filtering
        """
        try:
            if not self.ensure_connection():
                return []
            
            # 🚀 EARLY EXIT: Se tutti i filtri sono disattivati, restituisci subito array vuoto
            if not show_representatives and not show_propagated and not show_outliers:
                print(f"🚫 [PERFORMANCE] Tutti i filtri disattivati - Return immediato")
                return []
                
            # Usa la collection appropriata per il tenant
            collection = self.db[self.get_collection_name()]
            
            # 🚀 QUERY SEMPLIFICATA: Solo review_status + label_filter (se presente)
            base_query = {
                "review_status": "pending"  # Solo casi realmente da rivedere
            }
            
            # Aggiungi filtro etichetta se specificato
            if label_filter and label_filter != "Tutte le etichette":
                base_query["classification"] = label_filter
            
            print(f"� [PERFORMANCE] MongoDB Query semplificata: {base_query}")
            print(f"🔍 [PERFORMANCE] Filtri in-memory: representatives={show_representatives}, propagated={show_propagated}, outliers={show_outliers}")
            
            # Proiezione esclude embedding per ridurre traffico rete
            projection = {"embedding": 0}
            
            # 🚀 STREAMING CURSOR: Non usiamo sort() per evitare overhead, processiamo in streaming
            cursor = collection.find(base_query, projection)
            
            # 🚀 IN-MEMORY FILTERING con early termination
            sessions = []
            processed_count = 0
            filtered_count = 0
            
            for doc in cursor:
                processed_count += 1
                
                # � EARLY TERMINATION: Se abbiamo già abbastanza risultati, stoppa l'iterazione
                if len(sessions) >= limit:
                    print(f"🏁 [PERFORMANCE] Limite {limit} raggiunto - Stop iterazione dopo {processed_count} documenti processati")
                    break
                
                # 🚀 IN-MEMORY BOOLEAN FILTERING: Applica filtri sui metadati senza $or MongoDB
                metadata = doc.get('metadata', {})
                
                # Determina tipo sessione con logica booleana ottimizzata
                is_representative = metadata.get('representative', False)
                is_propagated = metadata.get('propagated', False)
                is_outlier = metadata.get('outlier', False)
                
                # Fallback per compatibilità con dati legacy (solo se metadati booleani non presenti)
                if not (is_representative or is_propagated or is_outlier):
                    if metadata.get('is_representative', False):
                        is_representative = True
                    elif metadata.get('propagated_from'):
                        is_propagated = True
                    elif metadata.get('cluster_id') in [-1, "-1"] or not metadata.get('cluster_id'):
                        is_outlier = True
                
                # 🚀 FAST BOOLEAN CHECK: Verifica se il documento soddisfa i filtri
                include_document = (
                    (show_representatives and is_representative) or
                    (show_propagated and is_propagated) or 
                    (show_outliers and is_outlier)
                )
                
                if not include_document:
                    filtered_count += 1
                    continue
                
                # 🚀 MINIMAL OBJECT CREATION: Costruisci solo gli oggetti che passano i filtri
                doc['_id'] = str(doc['_id'])
                
                # Determina session_type per compatibilità
                if is_representative:
                    session_type = "representative"
                elif is_propagated:
                    session_type = "propagated"
                elif is_outlier:
                    session_type = "outlier"
                else:
                    session_type = "unknown"
                
                # 🔧 FIX REVIEW QUEUE: Mapping intelligente per casi propagati (manteniamo la logica esistente)
                final_classification = doc.get('classification', doc.get('classificazione', ''))
                final_confidence = doc.get('confidence', 0.0)
                
                # Per casi propagati, usa classificazione finale come fallback per display ML/LLM
                if session_type == 'propagated':
                    ml_prediction = doc.get('ml_prediction', '') or final_classification  
                    llm_prediction = doc.get('llm_prediction', '') or final_classification
                    ml_confidence = doc.get('ml_confidence', 0.0) or final_confidence
                    llm_confidence = doc.get('llm_confidence', 0.0) or final_confidence
                else:
                    ml_prediction = doc.get('ml_prediction', '')
                    llm_prediction = doc.get('llm_prediction', '')
                    ml_confidence = doc.get('ml_confidence', 0.0)
                    llm_confidence = doc.get('llm_confidence', 0.0)

                session = {
                    'id': doc['_id'],
                    'case_id': doc['_id'],  # Per compatibilità API
                    'session_id': doc.get('session_id', ''),
                    'conversation_text': doc.get('testo_completo', doc.get('testo', doc.get('conversazione', ''))),
                    'classification': final_classification,
                    'confidence': final_confidence,
                    'motivation': doc.get('motivazione', ''),
                    'notes': doc.get('motivazione', ''),  # Mapping motivazione → notes per UI
                    'method': doc.get('classification_method', metadata.get('method', 'unknown')),
                    'timestamp': doc.get('classified_at', doc.get('timestamp')),
                    'tenant_name': doc.get('tenant_name', client_name),
                    'review_status': doc.get('review_status', 'not_required'),
                    'review_reason': doc.get('review_reason', ''),
                    'human_reviewed': doc.get('human_reviewed', False),
                    
                    # 🆕 METADATI REVIEW QUEUE
                    'session_type': session_type,  # representative/propagated/outlier
                    'cluster_id': metadata.get('cluster_id'),
                    'is_representative': is_representative,  # Usa il valore booleano calcolato
                    'propagated_from': metadata.get('propagated_from'),
                    'propagation_confidence': metadata.get('propagation_confidence'),
                    
                    # 🔧 DETTAGLI ML/LLM CON MAPPING INTELLIGENTE PER PROPAGATI
                    'ml_prediction': ml_prediction,
                    'ml_confidence': ml_confidence,
                    'llm_prediction': llm_prediction,
                    'llm_confidence': llm_confidence,
                    'llm_reasoning': doc.get('llm_reasoning', ''),
                    
                    'human_decision': doc.get('human_decision'),
                    'human_confidence': doc.get('human_confidence'),
                    'human_notes': doc.get('human_notes', ''),
                    'human_reviewed_at': doc.get('human_reviewed_at')
                }
                
                sessions.append(session)
            
            # 🚀 PERFORMANCE SUMMARY: Log delle statistiche di ottimizzazione
            print(f"✅ [PERFORMANCE] Review Queue completata:")
            print(f"   📊 Risultati: {len(sessions)} sessioni recuperate")
            print(f"   🔍 Processati: {processed_count} documenti totali")
            print(f"   🚫 Filtrati: {filtered_count} documenti esclusi")
            print(f"   ⚡ Efficienza: {len(sessions)}/{processed_count} = {(len(sessions)/max(processed_count,1)*100):.1f}% match rate")
            
            return sessions
            
        except Exception as e:
            print(f"❌ [PERFORMANCE] Errore nel recupero Review Queue ottimizzato: {e}")
            return []





    def get_tenant_classifications_with_clustering(self, 
                                                 tenant_slug: str,
                                                 start_date: datetime,
                                                 end_date: datetime,
                                                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Recupera classificazioni di un tenant con informazioni di clustering
        per generare statistiche avanzate
        
        Args:
            tenant_slug: Slug del tenant (es: 'humanitas')
            start_date: Data inizio ricerca
            end_date: Data fine ricerca  
            limit: Limite massimo risultati
            
        Returns:
            Lista classificazioni con cluster info
            
        Data ultima modifica: 2025-08-26
        """
        try:
            if not self.ensure_connection():
                print("❌ Impossibile connettersi a MongoDB per recupero classificazioni cluster")
                return []

            # Trova tenant_id dal tenant_slug
            tenant_info = self.get_tenant_info_from_name(tenant_slug)
            if not tenant_info:
                print(f"❌ Tenant '{tenant_slug}' non trovato")
                return []
            
            tenant_id = tenant_info['tenant_id']
            collection_name = self.get_collection_name()
            collection = self.db[collection_name]
            
            print(f"🔍 Ricerca classificazioni in {collection_name}")
            print(f"   📅 Periodo: {start_date} -> {end_date}")
            
            # Query MongoDB - filtra solo documenti con classificazione disponibile
            query = {
                'classification': {'$exists': True, '$ne': None}
            }
            
            # Projection per includere campi necessari per statistiche
            projection = {
                'session_id': 1,
                'testo_completo': 1,
                'classification': 1,
                'predicted_label': 1,
                'confidence': 1,
                'classification_method': 1,
                'cluster_label': 1,  # Campo legacy se disponibile
                'metadata.cluster_id': 1,
                'cluster_confidence': 1,
                'embedding': 1,  # Per eventuali ri-clustering
                'timestamp': 1,
                'tenant_slug': 1,
                'tenant_id': 1
            }
            
            # Esegui query con ordinamento per timestamp (più recenti prima)
            cursor = collection.find(query, projection).sort([
                ('classified_at', -1),
                ('timestamp', -1),
                ('created_at', -1)
            ])
            
            if limit:
                cursor = cursor.limit(limit)
            
            classifications = list(cursor)
            
            # Post-processing per garantire compatibilità
            processed_classifications = []
            for cls in classifications:
                # Filtra per range temporale supportando sia datetime che ISO string
                timestamp_value = cls.get('timestamp') or cls.get('classified_at') or cls.get('created_at')
                ts_datetime = None
                if timestamp_value is not None:
                    if isinstance(timestamp_value, datetime):
                        ts_datetime = timestamp_value
                    elif isinstance(timestamp_value, str):
                        try:
                            ts_datetime = datetime.fromisoformat(timestamp_value)
                        except ValueError:
                            ts_datetime = None
                
                if ts_datetime is not None:
                    if ts_datetime < start_date or ts_datetime > end_date:
                        continue

                # Recupera cluster_id da differenti fonti (legacy e nuova struttura metadata)
                cluster_id_raw = cls.get('cluster_label', None)
                if cluster_id_raw is None:
                    cluster_id_raw = cls.get('metadata', {}).get('cluster_id')

                cluster_id = -1  # Default: outlier
                cluster_label_friendly = None
                if cluster_id_raw is not None:
                    if isinstance(cluster_id_raw, (int, float)):
                        cluster_id = int(cluster_id_raw)
                    elif isinstance(cluster_id_raw, str):
                        cluster_id_raw_str = cluster_id_raw.strip()
                        if cluster_id_raw_str.lower().startswith('outlier'):
                            cluster_id = -1
                        else:
                            try:
                                cluster_id = int(cluster_id_raw_str)
                            except ValueError:
                                import zlib
                                # Usa hash deterministico (crc32) per ottenere ID numerico
                                cluster_id = int(zlib.crc32(cluster_id_raw_str.encode('utf-8')))
                            cluster_label_friendly = cluster_id_raw_str

                # Etichetta finale preferisce campo 'classification' (decisione finale)
                final_label = cls.get('classification') or cls.get('predicted_label', 'unknown')
                classification_method = cls.get('classification_method', 'unknown')

                processed_cls = {
                    'session_id': cls.get('session_id', str(cls.get('_id'))),
                    'testo_completo': cls.get('testo_completo', ''),
                    'predicted_label': final_label or 'unknown',
                    'confidence': float(cls.get('confidence', 0.0)),
                    'classification_method': classification_method,
                    'cluster_label': cluster_id,
                    'cluster_label_raw': cluster_id_raw,
                    'cluster_label_display': cluster_label_friendly,
                    'cluster_confidence': cls.get('cluster_confidence', 0.0),
                    'timestamp': ts_datetime.isoformat() if ts_datetime else cls.get('timestamp'),
                    'tenant_slug': tenant_slug,
                    'tenant_id': tenant_id
                }
                
                # Verifica che abbia contenuto minimo
                if processed_cls['testo_completo'] and len(processed_cls['testo_completo'].strip()) > 10:
                    processed_classifications.append(processed_cls)
            
            print(f"✅ Trovate {len(processed_classifications)} classificazioni valide")
            
            return processed_classifications
            
        except Exception as e:
            print(f"❌ Errore recupero classificazioni con clustering: {e}")
            import traceback
            traceback.print_exc()
            return []


    def sync_tenants_from_remote(self) -> Dict[str, Any]:
        """
        Scopo: Sincronizza i tenant dal database remoto alla tabella locale TAG.tenants
        
        Input: Nessun parametro
        Output: Dizionario con risultato dell'operazione
        
        - Legge i tenant dal database remoto (solo lettura)
        - Scrive/aggiorna i tenant nella tabella locale TAG.tenants
        - Mantiene sincronizzazione tra remoto e locale
        
        Autore: Valerio Bignardi
        Data creazione: 2025-08-28
        Ultima modifica: 2025-08-28 - Implementazione iniziale
        """
        try:
            import yaml
            import mysql.connector
            from mysql.connector import Error
            import os
            
            # Carica configurazione
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
            config = load_config()
            
            remote_db_config = config.get('database', {})  # Database remoto (solo lettura)
            local_db_config = config.get('tag_database', {})  # Database locale TAG (scrittura)
            
            # ✅ STEP 1: Leggi tenant dal database remoto
            print("🔍 [SYNC] Connessione al database remoto per lettura tenant...")
            remote_connection = mysql.connector.connect(
                host=remote_db_config['host'],
                port=remote_db_config['port'],
                user=remote_db_config['user'],
                password=remote_db_config['password'],
                database=remote_db_config['database']
            )
            
            remote_cursor = remote_connection.cursor()
            
            # Query per leggere tenant dal remoto (compatibile con struttura reale)
            remote_query = """
                SELECT tenant_id, tenant_name, tenant_database, tenant_status, created_at
                FROM tenants 
                ORDER BY tenant_name
            """
            remote_cursor.execute(remote_query)
            remote_tenants = remote_cursor.fetchall()
            
            remote_cursor.close()
            remote_connection.close()
            
            print(f"✅ [SYNC] Recuperati {len(remote_tenants)} tenant dal database remoto")
            
            # ✅ STEP 2: Connetti al database locale TAG per scrittura
            print("🔍 [SYNC] Connessione al database locale TAG per sincronizzazione...")
            local_connection = mysql.connector.connect(
                host=local_db_config['host'],
                port=local_db_config['port'],
                user=local_db_config['user'],
                password=local_db_config['password'],
                database=local_db_config['database']
            )
            
            local_cursor = local_connection.cursor()
            
            # ✅ STEP 3: Sincronizzazione (INSERT o UPDATE)
            sync_count = 0
            updated_count = 0
            inserted_count = 0
            
            for tenant_data in remote_tenants:
                tenant_id, tenant_name, tenant_database, tenant_status, created_at = tenant_data
                
                # Mapping dei campi: tenant_status (1=active, 0=inactive) -> is_active
                is_active = 1 if tenant_status == 1 else 0
                
                # Usa tenant_database dal remoto come tenant_slug (già formattato correttamente)
                tenant_slug = tenant_database
                
                # Verifica se il tenant esiste già localmente (per UUID - CORRETTO!)
                check_query = "SELECT tenant_id, updated_at FROM tenants WHERE tenant_id = %s"
                local_cursor.execute(check_query, (tenant_id,))
                existing_tenant = local_cursor.fetchone()
                
                if existing_tenant:
                    # UPDATE: tenant esiste, aggiorna nome, slug e status
                    update_query = """
                        UPDATE tenants 
                        SET tenant_name = %s, tenant_slug = %s, is_active = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE tenant_id = %s
                    """
                    local_cursor.execute(update_query, (tenant_name, tenant_slug, is_active, tenant_id))
                    updated_count += 1
                    print(f"🔄 [SYNC] Aggiornato tenant: {tenant_name} ({tenant_id})")
                else:
                    # INSERT: nuovo tenant - evita conflitto con tenant_slug
                    try:
                        insert_query = """
                            INSERT INTO tenants (tenant_id, tenant_name, tenant_slug, is_active, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        """
                        local_cursor.execute(insert_query, (tenant_id, tenant_name, tenant_slug, is_active, created_at))
                        inserted_count += 1
                        print(f"➕ [SYNC] Inserito nuovo tenant: {tenant_name} ({tenant_id})")
                    except mysql.connector.IntegrityError as ie:
                        # Conflitto tenant_slug, usa tenant_id come slug alternativo
                        if "Duplicate entry" in str(ie) and "tenant_slug" in str(ie):
                            tenant_slug_alt = tenant_id  # Usa direttamente il tenant_id come slug univoco
                            print(f"⚠️ [SYNC] Conflitto slug per {tenant_name}, uso tenant_id: {tenant_slug_alt}")
                            local_cursor.execute(insert_query, (tenant_id, tenant_name, tenant_slug_alt, is_active, created_at))
                            inserted_count += 1
                        else:
                            raise ie
                
                sync_count += 1
            
            # Commit delle modifiche
            local_connection.commit()
            
            local_cursor.close()
            local_connection.close()
            
            result = {
                'success': True,
                'message': f'Sincronizzazione completata con successo',
                'stats': {
                    'total_remote_tenants': len(remote_tenants),
                    'processed': sync_count,
                    'inserted': inserted_count,
                    'updated': updated_count
                }
            }
            
            print(f"✅ [SYNC] Sincronizzazione completata: {inserted_count} inseriti, {updated_count} aggiornati")
            return result
            
        except mysql.connector.Error as db_error:
            error_msg = f'Errore database durante sincronizzazione tenant: {str(db_error)}'
            print(f"❌ [SYNC] {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'error_type': 'database_error'
            }
            
        except Exception as e:
            error_msg = f'Errore generico durante sincronizzazione tenant: {str(e)}'
            print(f"❌ [SYNC] {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'error_type': 'generic_error'
            }

    def clear_tenant_collection(self, client_name: str) -> Dict[str, Any]:
        """
        Scopo: Cancella completamente la collection MongoDB di un tenant
        ATTENZIONE: Operazione irreversibile!
        
        Parametri input:
            - client_name: Nome del tenant
            
        Output:
            - Dizionario con risultato dell'operazione
            
        Ultimo aggiornamento: 2025-08-28
        """
        try:
            if not self.ensure_connection():
                return {
                    'success': False,
                    'error': 'Impossibile connettersi a MongoDB',
                    'deleted_count': 0
                }
            
            # Ottieni nome collection per il tenant (ora usa l'oggetto tenant interno)
            collection_name = self.get_collection_name()
            collection = self.db[collection_name]
            
            # Conta documenti prima della cancellazione
            count_before = collection.count_documents({})
            
            if count_before == 0:
                return {
                    'success': True,
                    'message': f'Collection {collection_name} già vuota',
                    'deleted_count': 0,
                    'collection_name': collection_name
                }
            
            # Drop completo della collection per reset totale (più sicuro di delete_many)
            collection.drop()
            print(f"🗑️ Collection '{collection_name}' DROPPED (rimossi {count_before} documenti)")
            
            return {
                'success': True,
                'message': f"Collection {collection_name} droppata con successo",
                'deleted_count': count_before,
                'collection_name': collection_name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f'Errore durante cancellazione collection per {client_name}: {str(e)}'
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'deleted_count': 0
            }
    
    def get_cluster_representatives(self, client_name: str, cluster_id: int) -> List[Dict]:
        """
        Recupera tutti i rappresentanti di un cluster specifico
        
        Scopo della funzione: Ottenere rappresentanti reviewed per logica consenso
        Parametri di input: client_name, cluster_id
        Parametri di output: Lista rappresentanti con metadata
        Valori di ritorno: Lista dizionari sessioni rappresentanti
        Tracciamento aggiornamenti: 2025-08-28 - Nuovo per logica propagated
        
        Args:
            client_name: Nome/UUID del tenant
            cluster_id: ID del cluster
            
        Returns:
            Lista di rappresentanti del cluster con info review
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        
        try:
            # Determina nome collection
            collection_name = self.generate_collection_name(client_name)
            if not collection_name:
                print(f"❌ Collection non trovata per client: {client_name}")
                return []
            
            collection = self.db[collection_name]
            
            # Query per rappresentanti del cluster specifico
            query = {
                "cluster_metadata.cluster_id": cluster_id,
                "cluster_metadata.is_representative": True
            }
            
            # Recupera rappresentanti con proiezione dei campi necessari
            representatives = list(collection.find(
                query,
                {
                    "session_id": 1,
                    "classification": 1,
                    "human_reviewed": 1,
                    "review_status": 1,
                    "confidence": 1,
                    "cluster_metadata": 1,
                    "_id": 0
                }
            ))
            
            print(f"🔍 Trovati {len(representatives)} rappresentanti per cluster {cluster_id}")
            return representatives
            
        except Exception as e:
            print(f"❌ Errore recupero rappresentanti cluster {cluster_id}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def update_cluster_propagated(self, 
                                client_name: str,
                                cluster_id: int,
                                final_label: str,
                                review_status: str = 'auto_classified',
                                classified_by: str = 'consensus',
                                notes: str = '') -> int:
        """
        Aggiorna tutte le sessioni propagate di un cluster con label consenso
        
        Scopo della funzione: Auto-classificazione propagated per consenso rappresentanti
        Parametri di input: client_name, cluster_id, final_label, metadata
        Parametri di output: numero sessioni aggiornate
        Valori di ritorno: conteggio aggiornamenti MongoDB
        Tracciamento aggiornamenti: 2025-08-28 - Nuovo per consenso automatico
        
        Args:
            client_name: Nome/UUID del tenant
            cluster_id: ID del cluster
            final_label: Label finale da assegnare
            review_status: Status review (default: auto_classified)
            classified_by: Chi ha classificato (default: consensus)
            notes: Note aggiuntive
            
        Returns:
            Numero di sessioni propagate aggiornate
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        
        try:
            # Determina nome collection
            collection_name = self.generate_collection_name(client_name)
            if not collection_name:
                print(f"❌ Collection non trovata per client: {client_name}")
                return 0
            
            collection = self.db[collection_name]
            
            # Query per sessioni propagate del cluster (non-rappresentanti)
            query = {
                "cluster_metadata.cluster_id": cluster_id,
                "cluster_metadata.is_representative": False,  # Solo propagated
                "review_status": "pending"  # Solo quelle ancora in review
            }
            
            # Update data con timestamp
            update_data = {
                "$set": {
                    "classification": final_label,
                    "review_status": review_status,
                    "classified_by": classified_by,
                    "human_reviewed": False,  # Auto-classificate, non umane
                    "updated_at": datetime.now().isoformat(),
                    "notes": notes,
                    "classification_method": "consensus_propagation"
                }
            }
            
            # Esegui aggiornamento bulk
            result = collection.update_many(query, update_data)
            
            print(f"✅ Aggiornate {result.modified_count} sessioni propagate cluster {cluster_id} con label '{final_label}'")
            return result.modified_count
            
        except Exception as e:
            print(f"❌ Errore aggiornamento propagated cluster {cluster_id}: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def _determine_classification_type(self, cluster_metadata: dict) -> str:
        """
        Scopo: Determina il tipo di classificazione basato sui metadati cluster
        CORREZIONE: Restituisce solo i 3 tipi validi: RAPPRESENTANTE, OUTLIER, PROPAGATO
        
        Parametri input:
            - cluster_metadata: Metadati del clustering
            
        Output:
            - Stringa che indica il tipo: RAPPRESENTANTE, OUTLIER, PROPAGATO
            
        Ultimo aggiornamento: 2025-09-04 - CORREZIONE TIPI
        """
        print(f"🐛 DEBUG _determine_classification_type - Input: {cluster_metadata}")
        
        if not cluster_metadata:
            print(f"🐛 DEBUG _determine_classification_type - Nessun metadata → OUTLIER (default)")
            # CORREZIONE: Se non ci sono metadata, è probabilmente un outlier
            return "OUTLIER"
            
        # RAPPRESENTANTE: è esplicitamente marcato come rappresentativo
        is_representative = cluster_metadata.get('is_representative', False)
        print(f"🐛 DEBUG _determine_classification_type - is_representative: {is_representative} (type: {type(is_representative)})")
        
        if is_representative:
            print(f"🐛 DEBUG _determine_classification_type - RISULTATO: RAPPRESENTANTE")
            return "RAPPRESENTANTE"
            
        # OUTLIER: PRIORITÀ ALTA - cluster_id è -1 o contiene "outlier" o ha outlier_score > 0.5
        # DEVE essere controllato PRIMA di PROPAGATO perché un outlier non può essere propagato!
        cluster_id = cluster_metadata.get('cluster_id')
        outlier_score = cluster_metadata.get('outlier_score', 0.0)
        
        print(f"🐛 DEBUG _determine_classification_type - cluster_id: {cluster_id}, outlier_score: {outlier_score}")
        
        if (cluster_id == -1 or 
            (isinstance(cluster_id, str) and 'outlier' in cluster_id.lower()) or
            outlier_score > 0.5):
            print(f"🐛 DEBUG _determine_classification_type - RISULTATO: OUTLIER")
            return "OUTLIER"
            
        # PROPAGATO: ha cluster_id valido (>=0) E propagated_from
        # NOTA: Questo controllo è DOPO outlier per evitare contraddizioni
        propagated_from = cluster_metadata.get('propagated_from')
        print(f"🐛 DEBUG _determine_classification_type - propagated_from: {propagated_from}")
        
        if propagated_from and cluster_id is not None and cluster_id >= 0:
            print(f"🐛 DEBUG _determine_classification_type - RISULTATO: PROPAGATO")
            return "PROPAGATO"
            
        # DEFAULT: Se ha cluster_id valido ma non è rappresentante né propagato
        # Assumiamo sia un OUTLIER (caso edge)
        print(f"🐛 DEBUG _determine_classification_type - RISULTATO: OUTLIER (fallback)")
        return "OUTLIER"



# DA VERIFICARE SE VA BENE QUESTA FUNZIONE. VALERIO DEVE VEDERE
    def get_reviewed_sessions_for_training(self, 
                                         include_representatives: bool = True,
                                         include_outliers: bool = True,
                                         only_human_reviewed: bool = True) -> List[Dict[str, Any]]:
        """
        Recupera tutte le sessioni corrette dalla review umana per il riaddestramento ML.
        
        Scopo della funzione: Estrarre dati puliti da MongoDB per il training ML
        Parametri di input: filtri per tipo di sessioni da includere
        Parametri di output: lista sessioni con conversation_text e final_label
        Valori di ritorno: Lista di dict con dati training-ready
        Tracciamento aggiornamenti: 2025-09-07 - Valerio Bignardi - Nuovo per riaddestramento
        
        Args:
            include_representatives: Include sessioni rappresentanti cluster
            include_outliers: Include sessioni outlier (cluster_id = -1)
            only_human_reviewed: Solo sessioni con review_status = 'completed'
            
        Returns:
            List[Dict]: Lista sessioni per training con campi:
            - session_id: ID sessione
            - conversation_text: Testo conversazione
            - final_label: Etichetta finale corretta (ML o umana)
            - confidence: Confidenza finale
            - cluster_id: ID cluster (-1 per outlier)
            - is_representative: Se è rappresentante cluster
            - review_status: Stato review umana
            
        Autore: Valerio Bignardi
        Data: 2025-09-07
        """
        trace_all("get_reviewed_sessions_for_training", "ENTER",
                 include_representatives=include_representatives,
                 include_outliers=include_outliers,
                 only_human_reviewed=only_human_reviewed)
        
        try:
            if not self.client:
                print("❌ [TRAINING DATA] Connessione MongoDB non disponibile")
                return []
            
            # Costruisci filtro per query MongoDB
            filter_query = {}
            
            # Filtro per stato review se richiesto
            if only_human_reviewed:
                filter_query["review_status"] = "completed"
            
            # Filtro per tipo sessioni
            if not include_representatives and not include_outliers:
                print("⚠️ [TRAINING DATA] Nessun tipo di sessione specificato")
                return []
            elif include_representatives and include_outliers:
                # Include tutto (rappresentanti + outlier)
                pass  # Nessun filtro aggiuntivo
            elif include_representatives and not include_outliers:
                # Solo rappresentanti (cluster_id != -1 e is_representative = true)
                filter_query["$and"] = [
                    {"cluster_id": {"$ne": -1}},
                    {"is_representative": True}
                ]
            elif include_outliers and not include_representatives:
                # Solo outlier (cluster_id = -1)
                filter_query["cluster_id"] = -1
            
            print(f"🔍 [TRAINING DATA] Query filtro: {filter_query}")
            
            # Esegui query con proiezione ottimizzata
            collection = self.db[self.collection_name]
            
            projection = {
                "session_id": 1,
                "conversation_text": 1,
                "classification": 1,
                "human_label": 1,
                "confidence": 1,
                "cluster_id": 1,
                "is_representative": 1,
                "review_status": 1,
                "timestamp": 1,
                "_id": 0
            }
            
            cursor = collection.find(filter_query, projection)
            sessions = list(cursor)
            
            print(f"📊 [TRAINING DATA] Trovate {len(sessions)} sessioni raw")
            
            # Processa risultati per formato training
            training_data = []
            
            for session in sessions:
                # Determina etichetta finale (umana se disponibile, altrimenti ML)
                final_label = session.get('human_label') or session.get('classification')
                
                if not final_label:
                    print(f"⚠️ [TRAINING DATA] Sessione {session.get('session_id')} senza etichetta, skip")
                    continue
                
                conversation_text = session.get('conversation_text', '')
                if not conversation_text or len(conversation_text.strip()) == 0:
                    print(f"⚠️ [TRAINING DATA] Sessione {session.get('session_id')} senza testo, skip")
                    continue
                
                training_record = {
                    'session_id': session.get('session_id'),
                    'conversation_text': conversation_text,
                    'final_label': final_label,
                    'confidence': session.get('confidence', 0.5),
                    'cluster_id': session.get('cluster_id', -1),
                    'is_representative': session.get('is_representative', False),
                    'review_status': session.get('review_status', 'pending'),
                    'timestamp': session.get('timestamp')
                }
                
                training_data.append(training_record)
            
            # Statistiche finali
            representatives_count = sum(1 for d in training_data if d['is_representative'])
            outliers_count = sum(1 for d in training_data if d['cluster_id'] == -1)
            human_reviewed_count = sum(1 for d in training_data if d['review_status'] == 'completed')
            
            print(f"✅ [TRAINING DATA] Dati training preparati:")
            print(f"   📊 Totale sessioni: {len(training_data)}")
            print(f"   👥 Rappresentanti: {representatives_count}")
            print(f"   🔍 Outliers: {outliers_count}")
            print(f"   👤 Human reviewed: {human_reviewed_count}")
            print(f"   🏷️ Etichette uniche: {len(set(d['final_label'] for d in training_data))}")
            
            trace_all("get_reviewed_sessions_for_training", "EXIT",
                     total_sessions=len(training_data),
                     representatives=representatives_count,
                     outliers=outliers_count,
                     human_reviewed=human_reviewed_count)
            
            return training_data
            
        except Exception as e:
            error_msg = f"Errore recupero dati training: {str(e)}"
            print(f"❌ [TRAINING DATA] {error_msg}")
            import traceback
            traceback.print_exc()
            
            trace_all("get_reviewed_sessions_for_training", "ERROR", error_message=error_msg)
            return []


def main():
    """
    Test del MongoDB Classification Reader
    """
    print("🔍 Test MongoDB Classification Reader")
    
    # CORREZIONE: Crea oggetto Tenant di esempio per il test
    try:
        # Usa il primo tenant disponibile per il test
        tenant_objects = MongoClassificationReader.get_available_tenants()
        if tenant_objects:
            test_tenant = tenant_objects[0]
            print(f"📝 Test con tenant: {test_tenant.tenant_name}")
        else:
            # Fallback: crea tenant di test (potrebbe fallire se non esiste)
            print("⚠️ Nessun tenant trovato, uso tenant di test 'humanitas'")
            test_tenant = Tenant.from_slug('humanitas')
    except Exception as e:
        print(f"❌ Errore creazione tenant di test: {e}")
        print("💡 Assicurati che ci siano tenant nel database locale TAG")
        return
    
    reader = MongoClassificationReader(tenant=test_tenant)
    
    if reader.connect():
        print("✅ Connesso a MongoDB")
        
        # Test recupero etichette (usa l'oggetto tenant interno)
        labels = reader.get_available_labels()
        print(f"📋 Etichette trovate: {len(labels)}")
        for label in labels[:5]:
            print(f"  - {label}")
        
        # Test recupero sessioni (usa l'oggetto tenant interno)
        sessions = reader.get_all_sessions(limit=3)
        print(f"📊 Sessioni trovate: {len(sessions)}")
        for session in sessions:
            print(f"  - {session['session_id']}: {session['classification']} ({session['confidence']})")
        
        # Test statistiche
        stats = reader.get_classification_stats("humanitas")
        print(f"📈 Totale classificazioni: {stats.get('total_classifications', 0)}")
        
        reader.disconnect()
        print("✅ Disconnesso da MongoDB")
    else:
        print("❌ Impossibile connettersi a MongoDB")


if __name__ == "__main__":
    main()
