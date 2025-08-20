import sys
import os
import yaml
import mysql.connector
from mysql.connector import Error

class TagDatabaseManager:
    """
    Classe per gestire il database locale dei tag
    """
    
    def __init__(self, config_path=None):
        """
        Inizializza il gestore del database dei tag
        
        Args:
            config_path (str): Percorso al file di configurazione
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        self.config = self._load_config(config_path)
        self.connection = None
        
    def _load_config(self, config_path):
        """Carica i parametri di configurazione dal file config.yaml"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise Exception(f"File di configurazione non trovato: {config_path}")
        except yaml.YAMLError as e:
            raise Exception(f"Errore nel parsing del file YAML: {e}")
    
    def connetti(self):
        """Stabilisce la connessione al database MySQL locale"""
        try:
            db_config = self.config['tag_database']
            self.connection = mysql.connector.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                autocommit=True
            )
            if self.connection.is_connected():
                print("✅ Connessione al database locale MySQL stabilita con successo")
                return True
        except Error as e:
            print(f"❌ Errore durante la connessione al database locale: {e}")
            return False
    
    def crea_database_e_tabelle(self):
        """Crea il database TAG e le tabelle necessarie"""
        if not self.connection or not self.connection.is_connected():
            if not self.connetti():
                return False
        
        try:
            cursor = self.connection.cursor()
            
            # Crea il database TAG se non esiste
            print("🔧 Creazione database TAG...")
            cursor.execute("CREATE DATABASE IF NOT EXISTS TAG")
            print("✅ Database TAG creato/verificato")
            
            # Seleziona il database
            cursor.execute("USE TAG")
            
            # Crea la tabella per i tag delle sessioni
            create_table_query = """
            CREATE TABLE IF NOT EXISTS session_tags (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                tag VARCHAR(255) NOT NULL,
                tenant_name VARCHAR(255) NOT NULL,
                tenant_id VARCHAR(255) NOT NULL,
                confidence FLOAT DEFAULT NULL,
                reviewed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY unique_session (session_id),
                INDEX idx_tenant (tenant_name),
                INDEX idx_tag (tag),
                INDEX idx_reviewed (reviewed)
            )
            """
            
            print("🔧 Creazione tabella session_tags...")
            cursor.execute(create_table_query)
            print("✅ Tabella session_tags creata/verificata")
            
            # Crea una tabella per memorizzare i tenant disponibili
            create_tenants_table = """
            CREATE TABLE IF NOT EXISTS tenants_cache (
                tenant_id VARCHAR(255) PRIMARY KEY,
                tenant_name VARCHAR(255) NOT NULL,
                tenant_slug VARCHAR(255) NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY unique_slug (tenant_slug)
            )
            """
            
            print("🔧 Creazione tabella tenants_cache...")
            cursor.execute(create_tenants_table)
            print("✅ Tabella tenants_cache creata/verificata")
            
            cursor.close()
            return True
            
        except Error as e:
            print(f"❌ Errore durante la creazione del database/tabelle: {e}")
            return False
    
    def inserisci_tag_sessione(self, session_id, tag, tenant_name, tenant_id, confidence=None):
        """
        Inserisce o aggiorna un tag per una sessione
        
        Args:
            session_id (str): ID della sessione
            tag (str): Tag da assegnare
            tenant_name (str): Nome del tenant (es. 'Humanitas')
            tenant_id (str): ID del tenant
            confidence (float): Livello di confidenza del tag (opzionale)
        """
        if not self.connection or not self.connection.is_connected():
            if not self.connetti():
                return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("USE TAG")
            
            query = """
            INSERT INTO session_tags (session_id, tag, tenant_name, tenant_id, confidence)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                tag = VALUES(tag),
                confidence = VALUES(confidence),
                updated_at = CURRENT_TIMESTAMP
            """
            
            cursor.execute(query, (session_id, tag, tenant_name, tenant_id, confidence))
            print(f"✅ Tag '{tag}' assegnato alla sessione {session_id}")
            cursor.close()
            return True
            
        except Error as e:
            print(f"❌ Errore durante l'inserimento del tag: {e}")
            return False
    
    def leggi_tag_sessione(self, session_id):
        """
        Legge il tag assegnato a una sessione
        
        Args:
            session_id (str): ID della sessione
            
        Returns:
            dict: Informazioni sul tag o None se non trovato
        """
        if not self.connection or not self.connection.is_connected():
            if not self.connetti():
                return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("USE TAG")
            
            query = """
            SELECT session_id, tag, tenant_name, tenant_id, confidence, reviewed, created_at, updated_at
            FROM session_tags
            WHERE session_id = %s
            """
            
            cursor.execute(query, (session_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return {
                    'session_id': result[0],
                    'tag': result[1],
                    'tenant_name': result[2],
                    'tenant_id': result[3],
                    'confidence': result[4],
                    'reviewed': result[5],
                    'created_at': result[6],
                    'updated_at': result[7]
                }
            return None
            
        except Error as e:
            print(f"❌ Errore durante la lettura del tag: {e}")
            return None
    
    def aggiorna_cache_tenants(self, tenants_data):
        """
        Aggiorna la cache dei tenants
        
        Args:
            tenants_data (list): Lista di tuple (tenant_id, tenant_name, tenant_slug)
        """
        if not self.connection or not self.connection.is_connected():
            if not self.connetti():
                return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("USE TAG")
            
            # Pulisce la cache esistente
            cursor.execute("DELETE FROM tenants_cache")
            
            # Inserisce i nuovi dati
            query = """
            INSERT INTO tenants_cache (tenant_id, tenant_name, tenant_slug)
            VALUES (%s, %s, %s)
            """
            
            cursor.executemany(query, tenants_data)
            print(f"✅ Cache tenants aggiornata con {len(tenants_data)} record")
            cursor.close()
            return True
            
        except Error as e:
            print(f"❌ Errore durante l'aggiornamento cache tenants: {e}")
            return False
    
    def ottieni_tenant_id(self, tenant_slug):
        """
        Ottiene l'ID del tenant dal slug
        
        Args:
            tenant_slug (str): Slug del tenant (es. 'humanitas')
            
        Returns:
            str: ID del tenant o None se non trovato
        """
        if not self.connection or not self.connection.is_connected():
            if not self.connetti():
                return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("USE TAG")
            
            query = "SELECT tenant_id FROM tenants_cache WHERE tenant_slug = %s"
            cursor.execute(query, (tenant_slug,))
            result = cursor.fetchone()
            cursor.close()
            
            return result[0] if result else None
            
        except Error as e:
            print(f"❌ Errore durante la ricerca tenant: {e}")
            return None
    
    def disconnetti(self):
        """Chiude la connessione al database"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("🔌 Connessione al database locale chiusa")

# Test della classe
if __name__ == "__main__":
    print("=== TEST TAG DATABASE MANAGER ===")
    
    manager = TagDatabaseManager()
    
    try:
        # Test connessione e creazione database
        if manager.crea_database_e_tabelle():
            print("\n✅ Database e tabelle creati con successo!")
            
            # Test inserimento di un tag di esempio
            test_session_id = "test-session-123"
            test_tag = "Richiesta Prenotazione Esame"
            test_tenant_name = "Humanitas"
            test_tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
            
            if manager.inserisci_tag_sessione(test_session_id, test_tag, test_tenant_name, test_tenant_id, 0.95):
                print("\n✅ Tag di test inserito con successo!")
                
                # Test lettura del tag
                tag_info = manager.leggi_tag_sessione(test_session_id)
                if tag_info:
                    print(f"\n📋 Tag letto: {tag_info}")
                else:
                    print("\n❌ Errore nella lettura del tag")
        
    finally:
        manager.disconnetti()
