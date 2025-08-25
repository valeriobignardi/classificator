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
    
    def __init__(self, mongodb_url: str = "mongodb://localhost:27017", 
                 database_name: str = "classificazioni", 
                 tenant_name: str = None,
                 collection_name: str = None):
        """
        Inizializza il reader MongoDB con supporto tenant-based collections
        
        Args:
            tenant_name: Se specificato, usa collection {tenant_name}_classifications
            collection_name: Collection legacy (per backward compatibility)
        """
        self.mongodb_url = mongodb_url
        self.database_name = database_name
        self.tenant_name = tenant_name
        
        # Determina collection name
        if tenant_name:
            # Usa il metodo get_collection_name per coerenza
            tenant_id = self.get_tenant_id_from_name(tenant_name)
            if tenant_id:
                self.collection_name = f"{tenant_name}_{tenant_id}"
            else:
                self.collection_name = f"{tenant_name}_classifications"  # Fallback
        elif collection_name:
            self.collection_name = collection_name
        else:
            # Fallback per backward compatibility
            self.collection_name = "client_session_classifications"
            
        # Database e Client
        self.client = None
        self.db = None
    
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
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
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
            # Query case-insensitive per il nome tenant
            cursor.execute(
                "SELECT tenant_id FROM tenants WHERE LOWER(tenant_name) = LOWER(%s) AND is_active = 1", 
                (tenant_name,)
            )
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
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
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

    def get_collection_name(self, tenant_name: str = None) -> str:
        """
        Genera il nome della collection MongoDB per il tenant
        
        Args:
            tenant_name: Nome del tenant (opzionale)
            
        Returns:
            Nome della collection per il tenant nel formato {tenant_name}_{tenant_id}
            
        Ultimo aggiornamento: 2025-01-27
        """
        effective_tenant = tenant_name or self.tenant_name
        
        if effective_tenant:
            # Cerca tenant_id dal nome
            tenant_id = self.get_tenant_id_from_name(effective_tenant)
            
            if tenant_id:
                # Usa formato {tenant_name}_{tenant_id} per collection name
                return f"{effective_tenant}_{tenant_id}"
            else:
                # Fallback al nome per compatibilità con tenant non in MySQL
                print(f"⚠️ Tenant '{effective_tenant}' non trovato in MySQL, uso nome come fallback")
                return f"{effective_tenant}_classifications"
        else:
            return "client_session_classifications"  # Fallback legacy
    
    def set_tenant(self, tenant_name: str):
        """
        Cambia il tenant attivo e riconfigura la collection
        
        Args:
            tenant_name: Nome del nuovo tenant
        """
        self.tenant_name = tenant_name
        self.collection_name = self.get_collection_name(tenant_name)
        
        # Se già connesso, aggiorna la collection
        if self.db is not None:
            self.collection = self.db[self.collection_name]
    
    def connect(self):
        """
        Stabilisce connessione a MongoDB
        
        Ultimo aggiornamento: 2025-08-21
        """
        try:
            if not self.client:
                self.client = MongoClient(self.mongodb_url)
                self.db = self.client[self.database_name]
            print(f"Connesso a MongoDB database: {self.database_name}")
            
        except Exception as e:
            print(f"Errore connessione MongoDB: {e}")
    
    def disconnect(self):
        """
        Scopo: Chiude connessione MongoDB
        Input: Nessuno
        Output: Connessione chiusa
        Ultimo aggiornamento: 2025-08-21
        """
        if self.client:
            self.client.close()
    
    def get_all_sessions(self, client_name: str = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scopo: Recupera tutte le sessioni classificate per un cliente/tenant
        
        AGGIORNATO: Supporta tenant-based collections
        
        Parametri input:
            - client_name: Nome del cliente/tenant (opzionale se già impostato nell'istanza)
            - limit: Numero massimo di sessioni da recuperare
            
        Output:
            - Lista di sessioni con classificazioni
            
        Ultimo aggiornamento: 2025-08-21
        """
        # Determina il tenant effettivo
        effective_tenant = client_name or self.tenant_name
        
        # Se il tenant è diverso da quello attuale, cambia collection
        if client_name and client_name != self.tenant_name:
            self.set_tenant(client_name)
        
        # Ensure we're connected to MongoDB
        if self.db is None:
            self.connect()
        
        try:
            # Usa la collection appropriata per il tenant
            collection = self.db[self.get_collection_name()]
            
            # NUOVO: Query semplificata (no filtro "client" per tenant-based collections)
            if self.tenant_name:
                # Collection per tenant: non serve filtro client
                query = {"review_status": {"$in": ["auto_classified", "resolved", "not_required"]}}
            else:
                # Fallback legacy: usa filtro client per collection condivisa
                query = {
                    "client": effective_tenant,
                    "review_status": {"$in": ["auto_classified", "resolved", "not_required"]}
                }
            
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
                    'tenant_name': effective_tenant,  # AGGIORNATO: usa tenant effettivo
                    
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
                        'created_at': doc.get('classified_at', doc.get('timestamp'))
                    }] if doc.get('classification') or doc.get('classificazione') else []
                }
                
                sessions.append(session)
            
            return sessions
            
        except Exception as e:
            print(f"Errore nel recupero sessioni: {e}")
            return []
    
    def get_available_labels(self, client_name: str = None) -> List[str]:
        """
        Scopo: Recupera tutte le etichette/classificazioni disponibili per un cliente/tenant
        
        AGGIORNATO: Supporta tenant-based collections
        
        Parametri input:
            - client_name: Nome del cliente/tenant (opzionale se già impostato nell'istanza)
            
        Output:
            - Lista di etichette unique
            
        Ultimo aggiornamento: 2025-08-21
        """
        # Determina il tenant effettivo
        effective_tenant = client_name or self.tenant_name
        
        # Se il tenant è diverso da quello attuale, cambia collection
        if client_name and client_name != self.tenant_name:
            self.set_tenant(client_name)
            
        try:
            if not self.client:
                self.connect()
                
            # Usa la collection appropriata per il tenant
            collection = self.db[self.get_collection_name()]
            
            # NUOVO: Query semplificata per tenant-based collections
            if self.tenant_name:
                # Collection per tenant: non serve filtro client
                filter_query = {}
            else:
                # Fallback legacy: usa filtro client
                filter_query = {"client": effective_tenant}
                
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
    
    def get_classification_stats(self, client_name: str = None) -> Dict[str, Any]:
        """
        Scopo: Recupera statistiche sulle classificazioni per un cliente/tenant
        
        AGGIORNATO: Supporta tenant-based collections
        
        Parametri input:
            - client_name: Nome del cliente/tenant (opzionale se già impostato nell'istanza)
            
        Output:
            - Dizionario con statistiche dettagliate
            
        Ultimo aggiornamento: 2025-08-21
        """
        # Determina il tenant effettivo
        effective_tenant = client_name or self.tenant_name
        
        # Se il tenant è diverso da quello attuale, cambia collection
        if client_name and client_name != self.tenant_name:
            self.set_tenant(client_name)
            
        try:
            if not self.client:
                self.connect()
                
            # Usa la collection appropriata per il tenant
            collection = self.db[self.get_collection_name()]
            
            # NUOVO: Pipeline semplificata per tenant-based collections
            if self.tenant_name:
                # Collection per tenant: non serve filtro client
                match_stage = {}
            else:
                # Fallback legacy: usa filtro client
                match_stage = {"client": effective_tenant}
            
            pipeline = [
                {"$match": match_stage},
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
        try:
            if not self.client:
                self.connect()
                
            # Usa la collection appropriata per il tenant
            collection = self.db[self.get_collection_name()]
            
            # Query per etichetta specifica
            if self.tenant_name:
                # Collection tenant: nessun filtro client
                query = {"classificazione": label}
            else:
                # Collection legacy: filtro client
                query = {
                    "client": client_name,
                    "classificazione": label
                }
            
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
            if not self.client:
                self.connect()
                
            # Usa la collection appropriata per il tenant
            collection = self.db[self.get_collection_name()]
            
            # Query per sessioni che necessitano review
            if self.tenant_name:
                # Collection tenant: nessun filtro client
                query = {"review_status": "pending"}
            else:
                # Collection legacy: filtro client
                query = {
                    "client": client_name,
                    "review_status": "pending"
                }
            
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
            if not self.client:
                self.connect()
                
            # Usa la collection appropriata per il tenant
            collection = self.db[self.get_collection_name()]
            
            # Prepara il filtro per trovare il documento
            filter_dict = {"session_id": session_id}
            if not self.tenant_name:
                filter_dict["client"] = client_name
            
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
                
                # Aggiungi campo "client" solo se non usiamo tenant collection
                if not self.tenant_name:
                    new_doc["client"] = client_name
                    new_doc["tenant_name"] = client_name
                
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
            if not self.client:
                self.connect()
                
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
            if not self.tenant_name:
                filter_dict["client"] = client_name
            
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
            if not self.client:
                self.connect()
                
            # Usa la collection appropriata per il tenant
            collection = self.db[self.get_collection_name()]
            
            # Pipeline di aggregazione per statistiche review
            if self.tenant_name:
                # Per collection tenant, non filtrare per client
                pipeline = [
                    {
                        "$group": {
                            "_id": "$review_status",
                            "count": {"$sum": 1}
                        }
                    }
                ]
            else:
                # Per collection legacy, filtra per client
                pipeline = [
                    {"$match": {"client": client_name}},
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

    def get_available_tenants(self) -> List[Dict[str, str]]:
        """
        Scopo: Recupera tutti i tenant disponibili dalla tabella MySQL TAG.tenants
        
        Output:
            - Lista di dizionari con tenant_id e nome del tenant
            
        Ultimo aggiornamento: 2025-01-27
        """
        try:
            # Importa le librerie necessarie
            import yaml
            import mysql.connector
            from mysql.connector import Error
            import os
            
            # Carica configurazione TAG database
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
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
            
            # Query per recuperare i tenant dalla tabella tenants
            query = "SELECT tenant_id, tenant_name FROM tenants WHERE is_active = 1 ORDER BY tenant_name"
            cursor.execute(query)
            risultati = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            tenants = []
            if risultati:
                for tenant_id, tenant_name in risultati:
                    tenants.append({
                        "tenant_id": str(tenant_id),
                        "nome": tenant_name
                    })
            
            print(f"Recuperati {len(tenants)} tenant dalla tabella TAG.tenants")
            return tenants
            
        except Exception as e:
            print(f"Errore nel recupero tenant da MySQL: {e}")
            
            # Fallback: prova a leggere dalle collection MongoDB esistenti
            try:
                if not self.client:
                    self.connect()
                
                # Ottieni lista collection del database
                collection_names = self.db.list_collection_names()
                
                tenants = []
                for collection_name in collection_names:
                    if collection_name.endswith("_classifications"):
                        # Estrai tenant_id dalla collection
                        tenant_id = collection_name.replace("_classifications", "")
                        # Escludi collection legacy
                        if tenant_id != "client_session":
                            # Cerca nome tenant dal tenant_id
                            tenant_name = self.get_tenant_name_from_id(tenant_id)
                            tenants.append({
                                "tenant_id": tenant_id,
                                "nome": tenant_name if tenant_name else tenant_id.title()
                            })
                
                # Ordina alfabeticamente
                tenants.sort(key=lambda x: x['nome'])
                
                # Se non ci sono collection tenant, controlla la collection legacy
                if not tenants and "client_session_classifications" in collection_names:
                    legacy_collection = self.db["client_session_classifications"]
                    legacy_tenants = legacy_collection.distinct("client")
                    for tenant in legacy_tenants:
                        if tenant and tenant.strip():
                            tenants.append({
                                "tenant_id": tenant,
                                "nome": tenant.title()
                            })
                    tenants.sort(key=lambda x: x['nome'])
                
                print(f"Fallback MongoDB: recuperati {len(tenants)} tenant dalle collection")
                return tenants
                
            except Exception as fallback_error:
                print(f"Errore anche nel fallback MongoDB: {fallback_error}")
                return []

    def generate_model_name(self, tenant_name: str, model_type: str = "classifier", timestamp: str = None) -> str:
        """
        Scopo: Genera nome modello tenant-aware nel formato nome_id_timestamp
        
        Args:
            tenant_name: Nome del tenant
            model_type: Tipo di modello (classifier, bertopic, etc.)
            timestamp: Timestamp personalizzato (opzionale, default: now)
            
        Returns:
            Nome modello nel formato: {tenant_name}_{tenant_id}_{model_type}_{timestamp}
            
        Ultimo aggiornamento: 2025-01-27
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
                # Formato: nome_id_tipo_timestamp
                # Usa solo prime 8 caratteri dell'ID per brevità
                short_id = tenant_id[:8]
                return f"{safe_tenant_name}_{short_id}_{model_type}_{timestamp}"
            else:
                # Fallback: solo nome_tipo_timestamp
                print(f"⚠️ Tenant ID non trovato per '{tenant_name}', uso solo nome")
                return f"{safe_tenant_name}_{model_type}_{timestamp}"
                
        except Exception as e:
            print(f"Errore nella generazione nome modello: {e}")
            # Fallback di sicurezza
            return f"unknown_tenant_{model_type}_{timestamp or 'unknown'}"

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
                                   needs_review: bool = False, review_reason: str = None) -> bool:
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
            
        Output:
            - True se salvato con successo
            
        Ultimo aggiornamento: 2025-01-27
        """
        try:
            if not self.client:
                self.connect()
                
            # Usa la collection appropriata per il tenant
            collection = self.db[self.get_collection_name()]
            
            # Prepara il documento base
            doc = {
                "session_id": session_id,
                "classified_at": datetime.now().isoformat(),
                "human_reviewed": False
            }
            
            # Aggiunge sempre i campi tenant (anche in collection tenant-specific)
            if self.tenant_name:
                doc["tenant_name"] = self.tenant_name
                tenant_id = self.get_tenant_id_from_name(self.tenant_name)
                if tenant_id:
                    doc["tenant_id"] = tenant_id
            
            # Aggiungi campo "client" per backward compatibility
            if not self.tenant_name:
                doc["client"] = client_name
            else:
                # Anche in collection tenant, mantieni client come riferimento
                doc["client"] = client_name
            
            # Aggiunge testo conversazione se fornito
            if conversation_text:
                doc["testo_completo"] = conversation_text
            
            # Aggiunge risultati ML se forniti
            if ml_result:
                doc["ml_prediction"] = ml_result.get("predicted_label", "")
                doc["ml_confidence"] = float(ml_result.get("confidence", 0.0))
            
            # Aggiunge risultati LLM se forniti
            if llm_result:
                doc["llm_prediction"] = llm_result.get("predicted_label", "")
                doc["llm_confidence"] = float(llm_result.get("confidence", 0.0))
                if "reasoning" in llm_result:
                    doc["llm_reasoning"] = llm_result["reasoning"]
            
            # Aggiunge decisione finale
            if final_decision:
                doc["classification"] = final_decision.get("predicted_label", "")
                doc["confidence"] = float(final_decision.get("confidence", 0.0))
                doc["classification_method"] = final_decision.get("method", "unknown")
                if "reasoning" in final_decision:
                    doc["motivazione"] = final_decision["reasoning"]
            
            # Determina review status
            if needs_review:
                doc["review_status"] = "pending"
                doc["review_reason"] = review_reason or "quality_gate_decision"
                doc["marked_for_review_at"] = datetime.now().isoformat()
            else:
                doc["review_status"] = "auto_classified"
            
            # Prepara il filtro per upsert
            filter_dict = {"session_id": session_id}
            if not self.tenant_name:
                filter_dict["client"] = client_name
            else:
                # Per collection tenant-specific, include sempre tenant info nel filtro
                filter_dict["tenant_name"] = self.tenant_name
                filter_dict["client"] = client_name
            
            # Upsert del documento
            result = collection.update_one(
                filter_dict,
                {"$set": doc},
                upsert=True
            )
            
            return result.upserted_id is not None or result.modified_count > 0
            
        except Exception as e:
            print(f"Errore nel salvataggio risultato classificazione: {e}")
            return False


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
