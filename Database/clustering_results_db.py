#!/usr/bin/env python3
"""
File: clustering_results_db.py
Autore: Sistema di Classificazione
Data: 2025-08-27

Descrizione: Gestore database per salvare e recuperare risultati dei test clustering
Supporta versioning, confronti e trend analytics
"""

import mysql.connector
import json
import yaml
import os
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

class NumpyJSONEncoder(json.JSONEncoder):
    """
    JSON Encoder personalizzato per gestire tipi numpy
    Converte bool numpy in bool standard Python
    """
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import logging

class ClusteringResultsDB:
    """
    Gestore database per risultati test clustering con versioning
    
    Scopo: Salvare ogni test clustering con numero versione incrementale
    e permettere confronti e analisi trend delle prestazioni clustering
    """
    
    def __init__(self, config_path: str = None):
        """
        Inizializza connessione al database locale TAG
        
        Args:
            config_path: Percorso del file config.yaml (opzionale)
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'config.yaml'
        )
        self.connection = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._load_config()
        
    def _load_config(self):
        """Carica configurazione database dal config.yaml"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            self.db_config = config.get('tag_database', {})
            self.logger.info(f"Configurazione database caricata da: {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Errore caricamento configurazione: {e}")
            raise RuntimeError(f"Impossibile caricare config database: {e}")
    
    def connect(self) -> bool:
        """
        Stabilisce connessione al database locale TAG
        
        Returns:
            bool: True se connessione riuscita
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database'],
                autocommit=True
            )
            self.logger.info("✅ Connesso al database TAG locale")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Errore connessione database TAG: {e}")
            return False
    
    def disconnect(self):
        """Chiude connessione database"""
        if self.connection:
            self.connection.close()
            self.logger.info("Connessione database chiusa")
    
    def save_clustering_result(self, 
                             tenant_id: str,
                             results_data: Dict[str, Any],
                             parameters_data: Dict[str, Any],
                             execution_time: float) -> Optional[int]:
        """
        Salva risultato test clustering con versione incrementale
        
        Args:
            tenant_id: ID del tenant
            results_data: JSON con risultati clustering completi
            parameters_data: JSON con parametri HDBSCAN/UMAP utilizzati
            execution_time: Tempo di esecuzione in secondi
            
        Returns:
            int: ID del record salvato, None se errore
        """
        try:
            if not self.connection:
                if not self.connect():
                    return None
            
            cursor = self.connection.cursor()
            
            # Genera numero versione incrementale per tenant
            cursor.execute("""
                SELECT COALESCE(MAX(version_number), 0) + 1 
                FROM clustering_test_results 
                WHERE tenant_id = %s
            """, (tenant_id,))
            
            version_number = cursor.fetchone()[0]
            
            # Estrai statistiche per query rapide
            stats = results_data.get('statistics', {})
            quality = results_data.get('quality_metrics', {})
            
            # Inserisci record
            insert_query = """
                INSERT INTO clustering_test_results (
                    tenant_id, version_number, execution_time,
                    results_json, parameters_json,
                    n_clusters, n_outliers, n_conversations,
                    clustering_ratio, silhouette_score,
                    davies_bouldin_score, calinski_harabasz_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                tenant_id,
                version_number,
                execution_time,
                json.dumps(results_data, ensure_ascii=False, cls=NumpyJSONEncoder),
                json.dumps(parameters_data, ensure_ascii=False, cls=NumpyJSONEncoder),
                stats.get('n_clusters'),
                stats.get('n_outliers'),
                stats.get('total_conversations'),
                stats.get('clustering_ratio'),
                quality.get('silhouette_score'),
                quality.get('davies_bouldin_score'),
                quality.get('calinski_harabasz_score')
            ))
            
            record_id = cursor.lastrowid
            cursor.close()
            
            self.logger.info(f"✅ Risultato clustering salvato: tenant={tenant_id}, "
                           f"version={version_number}, id={record_id}")
            
            return {
                'record_id': record_id,
                'version_number': version_number
            }
            
        except Exception as e:
            self.logger.error(f"❌ Errore salvataggio risultato clustering: {e}")
            return None
    
    def get_clustering_history(self, tenant_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Recupera storico test clustering per tenant
        
        Args:
            tenant_id: ID del tenant
            limit: Numero massimo record da restituire
            
        Returns:
            Lista dizionari con storico clustering
        """
        try:
            if not self.connection:
                if not self.connect():
                    return []
            
            cursor = self.connection.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT id, version_number, created_at, execution_time,
                       n_clusters, n_outliers, n_conversations,
                       clustering_ratio, silhouette_score,
                       davies_bouldin_score, calinski_harabasz_score
                FROM clustering_test_results
                WHERE tenant_id = %s
                ORDER BY version_number DESC
                LIMIT %s
            """, (tenant_id, limit))
            
            results = cursor.fetchall()
            cursor.close()
            
            # Converti timestamp in string per JSON serialization
            for result in results:
                if result['created_at']:
                    result['created_at'] = result['created_at'].isoformat()
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero storico clustering: {e}")
            return []
    
    def get_clustering_version(self, tenant_id: str, version_number: int) -> Optional[Dict[str, Any]]:
        """
        Recupera dati completi di una versione specifica
        
        Args:
            tenant_id: ID del tenant
            version_number: Numero versione
            
        Returns:
            Dizionario con dati completi della versione
        """
        try:
            if not self.connection:
                if not self.connect():
                    return None
            
            cursor = self.connection.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT id, version_number, created_at, execution_time,
                       results_json, parameters_json,
                       n_clusters, n_outliers, n_conversations,
                       clustering_ratio, silhouette_score,
                       davies_bouldin_score, calinski_harabasz_score
                FROM clustering_test_results
                WHERE tenant_id = %s AND version_number = %s
            """, (tenant_id, version_number))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                # Parse JSON fields
                result['results_data'] = json.loads(result['results_json'])
                result['parameters_data'] = json.loads(result['parameters_json'])
                
                # Converti timestamp
                if result['created_at']:
                    result['created_at'] = result['created_at'].isoformat()
                
                # Rimuovi campi JSON raw per pulizia
                del result['results_json']
                del result['parameters_json']
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero versione clustering: {e}")
            return None
    
    def get_latest_clustering(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """
        Recupera ultima versione clustering per tenant
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Dizionario con dati ultima versione
        """
        try:
            if not self.connection:
                if not self.connect():
                    return None
            
            cursor = self.connection.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT version_number
                FROM clustering_test_results
                WHERE tenant_id = %s
                ORDER BY version_number DESC
                LIMIT 1
            """, (tenant_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return self.get_clustering_version(tenant_id, result['version_number'])
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero ultimo clustering: {e}")
            return None
    
    def get_metrics_trend(self, tenant_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Recupera trend metriche clustering per grafici evolutivi
        
        Args:
            tenant_id: ID del tenant
            limit: Numero versioni da includere
            
        Returns:
            Lista con trend metriche ordinate per versione
        """
        try:
            if not self.connection:
                if not self.connect():
                    return []
            
            cursor = self.connection.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT version_number, created_at,
                       n_clusters, n_outliers, n_conversations,
                       clustering_ratio, silhouette_score,
                       davies_bouldin_score, calinski_harabasz_score,
                       execution_time
                FROM clustering_test_results
                WHERE tenant_id = %s
                ORDER BY version_number ASC
                LIMIT %s
            """, (tenant_id, limit))
            
            results = cursor.fetchall()
            cursor.close()
            
            # Converti timestamp per serialization
            for result in results:
                if result['created_at']:
                    result['created_at'] = result['created_at'].isoformat()
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero trend metriche: {e}")
            return []
    
    def get_comparison_data(self, tenant_id: str, 
                          version1: int, version2: int) -> Optional[Dict[str, Any]]:
        """
        Recupera dati per confronto tra due versioni
        
        Args:
            tenant_id: ID del tenant
            version1: Prima versione da confrontare
            version2: Seconda versione da confrontare
            
        Returns:
            Dizionario con dati delle due versioni per confronto
        """
        try:
            data1 = self.get_clustering_version(tenant_id, version1)
            data2 = self.get_clustering_version(tenant_id, version2)
            
            if not data1 or not data2:
                return None
            
            return {
                'version1': data1,
                'version2': data2,
                'tenant_id': tenant_id
            }
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero dati confronto: {e}")
            return None
    
    def __del__(self):
        """Destructor - chiude connessione se ancora aperta"""
        self.disconnect()
