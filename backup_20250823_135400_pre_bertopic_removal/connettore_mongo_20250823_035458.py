"""
Connettore MongoDB per salvare sessioni di classificazione con embeddings
Gestisce la collection client_session_classifications per ogni tenant
"""

import os
import sys
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import numpy as np
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import yaml

# Aggiungi il path del progetto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

class MongoDBConnector:
    """
    Connettore MongoDB per gestire le classificazioni delle sessioni con embeddings
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inizializza il connettore MongoDB
        
        Args:
            config_path: Path del file di configurazione YAML
        """
        self.logger = logging.getLogger(__name__)
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.config = self._load_config(config_path)
        
        # Connessione MongoDB
        self.mongo_url = self.config.get('mongodb', {}).get('url', 'mongodb://localhost:27017')
        self.database_name = self.config.get('mongodb', {}).get('database', 'classificazioni')
        
        # Configurazione collection
        self.collection_name = 'client_session_classifications'
        
        self._connect()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Carica la configurazione da file YAML"""
        if config_path is None:
            config_path = os.path.join(project_root, 'config.yaml')
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.warning(f"Impossibile caricare config da {config_path}: {e}")
            return {}
    
    def _connect(self) -> None:
        """Stabilisce la connessione a MongoDB"""
        try:
            self.logger.info(f"🔗 Connessione a MongoDB: {self.mongo_url}")
            
            # Crea client MongoDB con timeout
            self.client = MongoClient(
                self.mongo_url,
                serverSelectionTimeoutMS=5000,  # 5 secondi timeout
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            
            # Test connessione
            self.client.admin.command('ping')
            
            # Seleziona database
            self.db = self.client[self.database_name]
            
            self.logger.info(f"✅ Connesso a MongoDB database: {self.database_name}")
            
            # Crea indici necessari
            self._create_indexes()
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.logger.error(f"❌ Errore connessione MongoDB: {e}")
            self.client = None
            self.db = None
        except Exception as e:
            self.logger.error(f"❌ Errore generico MongoDB: {e}")
            self.client = None
            self.db = None
    
    def _create_indexes(self) -> None:
        """Crea gli indici necessari per la collection"""
        if self.db is None:
            return
        
        try:
            collection = self.db[self.collection_name]
            
            # Indice su session_id + tenant_id per chiave univoca
            collection.create_index([
                ('session_id', ASCENDING),
                ('tenant_id', ASCENDING)
            ], background=True, unique=True)
            
            # Indice su client + timestamp per query ordinate
            collection.create_index([
                ('client', ASCENDING),
                ('timestamp', DESCENDING)
            ], background=True)
            
            # Indice su tenant_id + timestamp per query per tenant
            collection.create_index([
                ('tenant_id', ASCENDING),
                ('timestamp', DESCENDING)
            ], background=True)
            
            # Indice su embedding_model per query per modello
            collection.create_index([
                ('embedding_model', ASCENDING)
            ], background=True)
            
            self.logger.info("✅ Indici MongoDB creati")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Errore creazione indici: {e}")
    
    def save_session_classification(self,
                                   client: str,
                                   session_id: str,
                                   tenant_id: str,
                                   tenant_name: str,
                                   testo: str,
                                   conversazione: str,
                                   embedding: Union[np.ndarray, List[float]],
                                   embedding_model: str,
                                   classificazione: Optional[str] = None,
                                   confidence: Optional[float] = None,
                                   motivazione: Optional[str] = None,
                                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Salva una classificazione di sessione in MongoDB
        
        Args:
            client: Nome del tenant/schema (es. 'humanitas')
            session_id: ID univoco della sessione
            tenant_id: ID numerico del tenant
            tenant_name: Nome del tenant
            testo: Testo della conversazione
            conversazione: Conversazione completa
            embedding: Array di embedding del testo
            embedding_model: Nome del modello usato per l'embedding
            classificazione: Etichetta di classificazione (opzionale)
            confidence: Confidence della classificazione (opzionale)
            motivazione: Motivazione della classificazione (opzionale)
            metadata: Metadati aggiuntivi (opzionale)
            
        Returns:
            True se salvato con successo, False altrimenti
        """
        if self.db is None:
            self.logger.error("❌ Database MongoDB non connesso")
            return False
        
        try:
            # Converte embedding in lista se è numpy array
            if isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)
            
            # Prepara il documento
            documento = {
                'client': client.lower(),
                'session_id': session_id,
                'tenant_id': tenant_id,
                'tenant_name': tenant_name,
                'testo': testo,
                'conversazione': conversazione,
                'embedding': embedding_list,
                'embedding_model': embedding_model,
                'timestamp': datetime.now(timezone.utc),
                'classificazione': classificazione,
                'confidence': confidence,
                'motivazione': motivazione,
                'metadata': metadata or {}
            }
            
            # Inserisce o aggiorna il documento
            collection = self.db[self.collection_name]
            
            # Usa upsert per evitare duplicati sulla combinazione session_id+tenant_id
            filter_query = {
                'session_id': session_id,
                'tenant_id': tenant_id
            }
            
            result = collection.replace_one(
                filter_query,
                documento,
                upsert=True
            )
            
            if result.upserted_id or result.modified_count > 0:
                self.logger.debug(f"✅ Sessione salvata: {client}/{session_id}")
                return True
            else:
                self.logger.warning(f"⚠️ Nessuna modifica per sessione: {client}/{session_id}")
                return True  # Considera comunque successo
                
        except Exception as e:
            self.logger.error(f"❌ Errore salvataggio sessione {client}/{session_id}: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def get_sessions_by_client(self,
                              client: str,
                              limit: int = 100,
                              skip: int = 0) -> List[Dict[str, Any]]:
        """
        Recupera le sessioni per un cliente specifico
        
        Args:
            client: Nome del cliente
            limit: Numero massimo di risultati
            skip: Numero di risultati da saltare
            
        Returns:
            Lista di documenti sessione
        """
        if self.db is None:
            self.logger.error("❌ Database MongoDB non connesso")
            return []
        
        try:
            collection = self.db[self.collection_name]
            
            cursor = collection.find(
                {'client': client.lower()},
                {'embedding': 0}  # Esclude embedding per performance
            ).sort('timestamp', DESCENDING).limit(limit).skip(skip)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero sessioni per {client}: {e}")
            return []
    
    def get_session_by_id(self,
                         client: str,
                         session_id: str,
                         include_embedding: bool = True) -> Optional[Dict[str, Any]]:
        """
        Recupera una sessione specifica per compatibilità (deprecato - usare get_session_by_unique_key)
        
        Args:
            client: Nome del cliente
            session_id: ID della sessione
            include_embedding: Se includere l'embedding nel risultato
            
        Returns:
            Documento sessione o None se non trovato
        """
        if self.db is None:
            self.logger.error("❌ Database MongoDB non connesso")
            return None
        
        try:
            collection = self.db[self.collection_name]
            
            projection = None if include_embedding else {'embedding': 0}
            
            return collection.find_one(
                {
                    'client': client.lower(),
                    'session_id': session_id
                },
                projection
            )
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero sessione {client}/{session_id}: {e}")
            return None
    
    def get_session_by_unique_key(self,
                                 session_id: str,
                                 tenant_id: str,
                                 include_embedding: bool = True) -> Optional[Dict[str, Any]]:
        """
        Recupera una sessione specifica usando la chiave univoca
        
        Args:
            session_id: ID della sessione
            tenant_id: ID del tenant
            include_embedding: Se includere l'embedding nel risultato
            
        Returns:
            Documento sessione o None se non trovato
        """
        if self.db is None:
            self.logger.error("❌ Database MongoDB non connesso")
            return None
        
        try:
            collection = self.db[self.collection_name]
            
            projection = None if include_embedding else {'embedding': 0}
            
            return collection.find_one(
                {
                    'session_id': session_id,
                    'tenant_id': tenant_id
                },
                projection
            )
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero sessione {session_id}/{tenant_id}: {e}")
            return None
    
    def get_sessions_by_client(self,
                              client: str,
                              limit: int = 1000,
                              include_embedding: bool = False) -> List[Dict[str, Any]]:
        """
        Recupera tutte le sessioni per un cliente (deprecato - usare get_sessions_by_tenant)
        
        Args:
            client: Nome del cliente
            limit: Numero massimo di risultati
            include_embedding: Se includere gli embeddings
            
        Returns:
            Lista di documenti sessione
        """
        if self.db is None:
            self.logger.error("❌ Database MongoDB non connesso")
            return []
        
        try:
            collection = self.db[self.collection_name]
            
            projection = None if include_embedding else {'embedding': 0}
            
            cursor = collection.find(
                {'client': client.lower()},
                projection
            ).limit(limit)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero sessioni per cliente {client}: {e}")
            return []
    
    def get_sessions_by_tenant(self,
                              tenant_id: str,
                              limit: int = 1000,
                              include_embedding: bool = False) -> List[Dict[str, Any]]:
        """
        Recupera tutte le sessioni per un tenant
        
        Args:
            tenant_id: ID del tenant
            limit: Numero massimo di risultati
            include_embedding: Se includere gli embeddings
            
        Returns:
            Lista di documenti sessione
        """
        if self.db is None:
            self.logger.error("❌ Database MongoDB non connesso")
            return []
        
        try:
            collection = self.db[self.collection_name]
            
            projection = None if include_embedding else {'embedding': 0}
            
            cursor = collection.find(
                {'tenant_id': tenant_id},
                projection
            ).sort('timestamp', -1).limit(limit)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero sessioni per tenant {tenant_id}: {e}")
            return []
    
    def get_embeddings_by_model(self,
                               client: str,
                               embedding_model: str,
                               limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Recupera embeddings per un modello specifico
        
        Args:
            client: Nome del cliente
            embedding_model: Nome del modello di embedding
            limit: Numero massimo di risultati
            
        Returns:
            Lista di documenti con embedding
        """
        if self.db is None:
            self.logger.error("❌ Database MongoDB non connesso")
            return []
        
        try:
            collection = self.db[self.collection_name]
            
            cursor = collection.find(
                {
                    'client': client.lower(),
                    'embedding_model': embedding_model
                },
                {
                    'session_id': 1,
                    'embedding': 1,
                    'classificazione': 1,
                    'confidence': 1,
                    'timestamp': 1
                }
            ).sort('timestamp', DESCENDING).limit(limit)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero embeddings {client}/{embedding_model}: {e}")
            return []
    
    def delete_sessions_by_client(self, client: str) -> int:
        """
        Elimina tutte le sessioni per un cliente
        
        Args:
            client: Nome del cliente
            
        Returns:
            Numero di documenti eliminati
        """
        if self.db is None:
            self.logger.error("❌ Database MongoDB non connesso")
            return 0
        
        try:
            collection = self.db[self.collection_name]
            
            result = collection.delete_many({'client': client.lower()})
            
            self.logger.info(f"🗑️ Eliminati {result.deleted_count} documenti per client {client}")
            return result.deleted_count
            
        except Exception as e:
            self.logger.error(f"❌ Errore eliminazione sessioni per {client}: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Recupera statistiche della collection
        
        Returns:
            Dizionario con statistiche
        """
        if self.db is None:
            return {'error': 'Database non connesso'}
        
        try:
            collection = self.db[self.collection_name]
            
            # Conta totale documenti
            total_count = collection.count_documents({})
            
            # Conta per cliente
            pipeline = [
                {'$group': {'_id': '$client', 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}}
            ]
            client_stats = list(collection.aggregate(pipeline))
            
            # Conta per modello embedding
            pipeline = [
                {'$group': {'_id': '$embedding_model', 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}}
            ]
            model_stats = list(collection.aggregate(pipeline))
            
            return {
                'total_documents': total_count,
                'by_client': client_stats,
                'by_embedding_model': model_stats,
                'collection_name': self.collection_name
            }
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero statistiche: {e}")
            return {'error': str(e)}
    
    def close(self) -> None:
        """Chiude la connessione MongoDB"""
        if self.client:
            self.client.close()
            self.logger.info("🔌 Connessione MongoDB chiusa")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Funzione di utilità per creare una connessione MongoDB
def create_mongo_connector(config_path: Optional[str] = None) -> MongoDBConnector:
    """
    Crea un connettore MongoDB configurato
    
    Args:
        config_path: Path del file di configurazione
        
    Returns:
        Istanza di MongoDBConnector
    """
    return MongoDBConnector(config_path)


if __name__ == "__main__":
    # Test del connettore
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Test connettore MongoDB...")
    
    with create_mongo_connector() as mongo:
        # Test statistiche
        stats = mongo.get_stats()
        print(f"📊 Statistiche: {stats}")
        
        # Test salvataggio
        embedding_test = np.random.rand(768).tolist()
        
        success = mongo.save_session_classification(
            client='test',
            session_id='test_session_001',
            testo='Questo è un testo di test',
            conversazione='Utente: Ciao\nBot: Ciao! Come posso aiutarti?',
            embedding=embedding_test,
            embedding_model='labse-test',
            classificazione='test',
            confidence=0.95,
            motivazione='Classificazione di test basata su saluto utente'
        )
        
        print(f"💾 Salvataggio test: {'✅' if success else '❌'}")
        
        # Test recupero
        session = mongo.get_session_by_id('test', 'test_session_001', include_embedding=False)
        print(f"🔍 Recupero test: {'✅' if session else '❌'}")
        
        if session:
            print(f"   Sessione: {session['session_id']}")
            print(f"   Classificazione: {session.get('classificazione')}")
            print(f"   Motivazione: {session.get('motivazione')}")
            print(f"   Timestamp: {session.get('timestamp')}")
        
    print("✅ Test completato!")
