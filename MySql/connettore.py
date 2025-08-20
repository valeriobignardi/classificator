import yaml
import mysql.connector
from mysql.connector import Error
import os

#questa classe ha il compito di connettersi al database MySql e offrire le funzioni per leggere e scrivere
class MySqlConnettore:
    def __init__(self):
        self.config = self._load_config()
        self.connection = None
    
    def _load_config(self):
        """Carica i parametri di configurazione dal file config.yaml"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise Exception(f"File di configurazione non trovato: {config_path}")
        except yaml.YAMLError as e:
            raise Exception(f"Errore nel parsing del file YAML: {e}")
    
    def connetti(self):
        """Stabilisce la connessione al database MySQL"""
        try:
            db_config = self.config['database']
            self.connection = mysql.connector.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database']
            )
            if self.connection.is_connected():
                print("Connessione al database MySQL stabilita con successo")
                return True
        except Error as e:
            print(f"Errore durante la connessione al database: {e}")
            return False
    
    def disconnetti(self):
        """Chiude la connessione al database"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Connessione al database chiusa")
    
    def esegui_query(self, query, parametri=None):
        """Esegue una query SELECT e restituisce i risultati"""
        if not self.connection or not self.connection.is_connected():
            self.connetti()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, parametri)
            risultati = cursor.fetchall()
            cursor.close()
            return risultati
        except Error as e:
            print(f"Errore durante l'esecuzione della query: {e}")
            return None
    
    def esegui_comando(self, comando, parametri=None):
        """Esegue un comando INSERT, UPDATE o DELETE"""
        if not self.connection or not self.connection.is_connected():
            self.connetti()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(comando, parametri)
            self.connection.commit()
            righe_modificate = cursor.rowcount
            cursor.close()
            return righe_modificate
        except Error as e:
            print(f"Errore durante l'esecuzione del comando: {e}")
            self.connection.rollback()
            return None
