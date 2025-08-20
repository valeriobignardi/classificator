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
        """Setup del logger per questo modulo"""
        logger = logging.getLogger(f"{__name__}.IntelligentTagSuggestionManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def has_existing_classifications(self, client_name: str) -> bool:
        """
        Controlla se esistono classificazioni nel database per il cliente.
        
        Args:
            client_name (str): Nome del cliente (es. 'Humanitas')
            
        Returns:
            bool: True se il cliente ha classificazioni esistenti, False altrimenti
        """
        try:
            # Connessione diretta al database delle classificazioni (database locale TAG)
            import mysql.connector
            from mysql.connector import Error
            
            db_config = self.config['tag_database']  # Usa il database locale TAG
            connection = mysql.connector.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database'],
                autocommit=True
            )
            
            cursor = connection.cursor()
            
            # Query per verificare se esistono classificazioni per questo tenant
            query = """
            SELECT COUNT(*) as count
            FROM session_classifications 
            WHERE tenant_name = %s AND tag_name IS NOT NULL AND tag_name != ''
            """
            
            cursor.execute(query, (client_name,))
            result = cursor.fetchone()
            
            cursor.close()
            connection.close()
            
            if result:
                count = result[0]
                exists = count > 0
                self.logger.info(f"Cliente '{client_name}': {count} classificazioni esistenti - {'Esistente' if exists else 'Nuovo'}")
                return exists
            
            return False
            
        except Exception as e:
            self.logger.error(f"Errore nel controllo classificazioni esistenti per {client_name}: {e}")
            return False
    
    def get_suggested_tags_for_client(self, client_name: str) -> List[Dict[str, Any]]:
        """
        Ottiene i tag suggeriti per un cliente dalla tabella TAGS.tags.
        Tutti i clienti ricevono tutti i tag disponibili come suggerimenti.
        
        Args:
            client_name (str): Nome del cliente
            
        Returns:
            List[Dict]: Lista di tag suggeriti con metadati
        """
        try:
            # Ottieni tutti i tag dalla tabella TAGS.tags
            all_tags = self._get_all_available_tags()
            
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
    
    def _get_all_available_tags(self) -> List[Dict[str, Any]]:
        """
        Ottiene tutti i tag disponibili dalla tabella TAGS.tags.
        Rimuove duplicati basandosi sul tag_name.
        
        Returns:
            List[Dict]: Lista di tutti i tag disponibili con metadati
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
            
            # Query per ottenere tutti i tag UNICI dalla tabella tags
            query = """
            SELECT MIN(id) as id, tag_name, 
                   MAX(tag_description) as tag_description, 
                   MIN(created_at) as created_at
            FROM tags 
            GROUP BY tag_name
            ORDER BY tag_name ASC
            """
            
            cursor.execute(query)
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
    
    def _get_tag_usage_stats(self, client_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Ottiene le statistiche di utilizzo dei tag per un cliente dalle classificazioni esistenti.
        
        Args:
            client_name (str): Nome del cliente
            
        Returns:
            Dict: Statistiche di utilizzo per tag name
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
            
            # Query per statistiche di utilizzo
            query = """
            SELECT tag_name, COUNT(*) as count, AVG(confidence_score) as avg_confidence
            FROM session_classifications 
            WHERE tenant_name = %s 
                AND tag_name IS NOT NULL 
                AND tag_name != ''
            GROUP BY tag_name
            ORDER BY count DESC
            """
            
            cursor.execute(query, (client_name,))
            result = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            usage_stats = {}
            if result:
                for row in result:
                    usage_stats[row[0]] = {
                        'count': row[1],
                        'avg_confidence': float(row[2]) if row[2] else 0.0
                    }
            
            return usage_stats
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero statistiche utilizzo per {client_name}: {e}")
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
