#!/usr/bin/env python3
"""
Author: System
Date: 2025-08-21
Description: Reader per classificazioni da MongoDB
Last Update: 2025-08-21

MongoDB Classification Reader - Legge le classificazioni direttamente da MongoDB
per l'interfaccia web React
"""

from pymongo import MongoClient
from typing import List, Dict, Any, Optional
from datetime import datetime
import os


class MongoClassificationReader:
    """
    Scopo: Legge le classificazioni dal database MongoDB per l'interfaccia web
    
    Parametri input:
        - mongodb_url: URL di connessione MongoDB
        - database_name: Nome del database
        - collection_name: Nome della collection
        
    Output:
        - Classificazioni formattate per l'interfaccia React
        
    Ultimo aggiornamento: 2025-08-21
    """
    
    def __init__(self, mongodb_url: str = "mongodb://localhost:27017", 
                 database_name: str = "classificazioni", 
                 collection_name: str = "client_session_classifications"):
        """
        Inizializza il reader MongoDB
        """
        self.mongodb_url = mongodb_url
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
    
    def connect(self):
        """
        Scopo: Stabilisce connessione a MongoDB
        Input: Nessuno
        Output: Connessione MongoDB attiva
        Ultimo aggiornamento: 2025-08-21
        """
        try:
            self.client = MongoClient(self.mongodb_url)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            return True
        except Exception as e:
            print(f"Errore connessione MongoDB: {e}")
            return False
    
    def disconnect(self):
        """
        Scopo: Chiude connessione MongoDB
        Input: Nessuno
        Output: Connessione chiusa
        Ultimo aggiornamento: 2025-08-21
        """
        if self.client:
            self.client.close()
    
    def get_all_sessions(self, client_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scopo: Recupera tutte le sessioni classificate per un cliente
        
        Parametri input:
            - client_name: Nome del cliente (es. 'humanitas')
            - limit: Numero massimo di sessioni da recuperare
            
        Output:
            - Lista di sessioni con classificazioni
            
        Ultimo aggiornamento: 2025-08-21
        """
        if self.collection is None:
            self.connect()
        
        try:
            # Query per recuperare tutte le classificazioni del cliente
            query = {"client": client_name}
            
            # Proiezione per escludere embedding (troppo pesante)
            projection = {
                "embedding": 0
            }
            
            # Ordinamento per timestamp (più recenti prima)
            sort_order = [("timestamp", -1)]
            
            cursor = self.collection.find(query, projection).sort(sort_order)
            
            if limit:
                cursor = cursor.limit(limit)
            
            sessions = []
            for doc in cursor:
                # Converti ObjectId in string per JSON
                doc['_id'] = str(doc['_id'])
                
                # Formatta per interfaccia React
                session = {
                    'id': doc['_id'],
                    'session_id': doc.get('session_id', ''),
                    'conversation_text': doc.get('testo', doc.get('conversazione', '')),
                    'classification': doc.get('classificazione', 'non_classificata'),
                    'confidence': doc.get('confidence', 0.0),
                    'motivation': doc.get('motivazione', ''),
                    'method': doc.get('metadata', {}).get('method', 'unknown'),
                    'processing_time': doc.get('metadata', {}).get('processing_time', 0.0),
                    'timestamp': doc.get('timestamp'),
                    'tenant_name': doc.get('tenant_name', client_name),
                    'classifications': [{
                        'label': doc.get('classificazione', 'non_classificata'),
                        'confidence': doc.get('confidence', 0.0),
                        'method': doc.get('metadata', {}).get('method', 'unknown'),
                        'motivation': doc.get('motivazione', ''),
                        'created_at': doc.get('timestamp')
                    }] if doc.get('classificazione') else []
                }
                
                sessions.append(session)
            
            return sessions
            
        except Exception as e:
            print(f"Errore nel recupero sessioni: {e}")
            return []
    
    def get_available_labels(self, client_name: str) -> List[str]:
        """
        Scopo: Recupera tutte le etichette/classificazioni disponibili per un cliente
        
        Parametri input:
            - client_name: Nome del cliente
            
        Output:
            - Lista di etichette unique
            
        Ultimo aggiornamento: 2025-08-21
        """
        if self.collection is None:
            self.connect()
        
        try:
            # Recupera etichette distinct per il cliente
            labels = self.collection.distinct("classificazione", {"client": client_name})
            
            # Rimuovi valori null/vuoti
            labels = [label for label in labels if label and label.strip()]
            
            # Ordina alfabeticamente
            labels.sort()
            
            return labels
            
        except Exception as e:
            print(f"Errore nel recupero etichette: {e}")
            return []
    
    def get_classification_stats(self, client_name: str) -> Dict[str, Any]:
        """
        Scopo: Recupera statistiche sulle classificazioni per un cliente
        
        Parametri input:
            - client_name: Nome del cliente
            
        Output:
            - Dizionario con statistiche dettagliate
            
        Ultimo aggiornamento: 2025-08-21
        """
        if self.collection is None:
            self.connect()
        
        try:
            # Pipeline di aggregazione per statistiche
            pipeline = [
                {"$match": {"client": client_name}},
                {
                    "$group": {
                        "_id": "$classificazione",
                        "count": {"$sum": 1},
                        "avg_confidence": {"$avg": "$confidence"},
                        "min_confidence": {"$min": "$confidence"},
                        "max_confidence": {"$max": "$confidence"}
                    }
                },
                {"$sort": {"count": -1}}
            ]
            
            results = list(self.collection.aggregate(pipeline))
            
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
                'client': client_name
            }
            
        except Exception as e:
            print(f"Errore nel calcolo statistiche: {e}")
            return {}
    
    def get_sessions_by_label(self, client_name: str, label: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scopo: Recupera sessioni filtrate per etichetta specifica
        
        Parametri input:
            - client_name: Nome del cliente
            - label: Etichetta da filtrare
            - limit: Numero massimo di risultati
            
        Output:
            - Lista di sessioni con l'etichetta specificata
            
        Ultimo aggiornamento: 2025-08-21
        """
        if self.collection is None:
            self.connect()
        
        try:
            # Query per etichetta specifica
            query = {
                "client": client_name,
                "classificazione": label
            }
            
            # Proiezione esclude embedding
            projection = {"embedding": 0}
            
            cursor = self.collection.find(query, projection).sort([("confidence", -1)])
            
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


def main():
    """
    Test del MongoDB Classification Reader
    """
    print("🔍 Test MongoDB Classification Reader")
    
    reader = MongoClassificationReader()
    
    if reader.connect():
        print("✅ Connesso a MongoDB")
        
        # Test recupero etichette
        labels = reader.get_available_labels("humanitas")
        print(f"📋 Etichette trovate: {len(labels)}")
        for label in labels[:5]:
            print(f"  - {label}")
        
        # Test recupero sessioni
        sessions = reader.get_all_sessions("humanitas", limit=3)
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
