"""
Gestore della memoria semantica per classificazioni esistenti.

Questo modulo gestisce lo storico delle classificazioni, calcola similarità semantiche
e decide quando riutilizzare etichette esistenti vs creare nuove etichette.
"""

import os
import sys
import yaml
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import logging
import re
from difflib import SequenceMatcher

# Aggiunge percorsi per importare altri moduli
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'TagDatabase'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EmbeddingEngine'))

from tag_database_connector import TagDatabaseConnector
# RIMOSSO: from labse_embedder import LaBSEEmbedder - Ora usa Docker service

# Import per simple_embedding_manager
from simple_embedding_manager import SimpleEmbeddingManager

# Import per MongoDB (sostituisce MySQL per le classificazioni)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from mongo_classification_reader import MongoClassificationReader

# Import per oggetto Tenant
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))
from tenant import Tenant

class SemanticMemoryManager:
    """
    Gestisce la memoria semantica delle classificazioni esistenti per
    ottimizzare il riutilizzo di etichette e la creazione di nuove categorie.
    """
    
    def __init__(self, 
                 tenant: Tenant,  # OGGETTO TENANT OBBLIGATORIO
                 config_path: str = None,
                 embedder = None):  # Any embedder type
        """
        Inizializza il gestore della memoria semantica
        CAMBIO RADICALE: USA OGGETTO TENANT
        
        Args:
            tenant: Oggetto Tenant completo con tutti i dati (OBBLIGATORIO)
            config_path: Percorso del file di configurazione
            embedder: Embedder per generare rappresentazioni semantiche
        """
        # VALIDA OGGETTO TENANT
        if not hasattr(tenant, 'tenant_id') or not hasattr(tenant, 'tenant_name') or not hasattr(tenant, 'tenant_slug'):
            raise TypeError("Il parametro 'tenant' deve essere un oggetto Tenant valido")
            
        # Salva oggetto tenant
        self.tenant = tenant
        self.tenant_name = tenant.tenant_name  # Mantieni per compatibilità
        
        # Carica configurazione
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        self.config = self._load_config(config_path)
        self.similarity_config = self.config.get('similarity_thresholds', {})
        self.memory_config = self.config.get('semantic_memory', {})
        
        # Soglie di similarità
        self.reuse_threshold = self.similarity_config.get('reuse_existing_label', 0.75)
        self.semantic_threshold = self.similarity_config.get('semantic_similarity', 0.65)
        self.confidence_threshold = self.similarity_config.get('classification_confidence', 0.70)
        
        # Configurazione memoria
        self.max_samples_per_label = self.memory_config.get('max_samples_per_label', 50)
        
        # Cache path tenant-aware
        base_cache_path = self.memory_config.get('cache_path', 'semantic_cache/')
        self.cache_path = self._get_tenant_cache_path(base_cache_path)
        
        self.enable_logging = self.memory_config.get('enable_detailed_logging', True)
        
        # Componenti - usa embedder Docker se non fornito
        if embedder is None:
            try:
                # Usa SimpleEmbeddingManager per ottenere embedder configurato
                sem = SimpleEmbeddingManager()
                self.embedder = sem.get_embedder_for_tenant(self.tenant)
            except Exception as e:
                # 🚫 NESSUN FALLBACK LOCALE - Solo Docker service
                raise RuntimeError(f"Embedder Docker richiesto ma non disponibile: {e}")
        else:
            self.embedder = embedder
            
        self.db_connector = TagDatabaseConnector(tenant=self.tenant)
        
        # MongoDB reader per classificazioni con OGGETTO TENANT
        self.mongo_reader = MongoClassificationReader(tenant=self.tenant)
        
        # Crea directory cache prima di tutto
        os.makedirs(self.cache_path, exist_ok=True)
        
        # Setup logging
        if self.enable_logging:
            self._setup_logging()
        
        # Memoria semantica
        self.semantic_memory = {
            'embeddings': {},      # tag_name -> list of embeddings
            'texts': {},          # tag_name -> list of session texts
            'session_ids': {},    # tag_name -> list of session_ids
            'last_updated': None,
            'total_sessions': 0
        }
        
        print(f"🧠 SemanticMemoryManager inizializzato")
        print(f"  🎯 Soglia riutilizzo etichette: {self.reuse_threshold}")
        print(f"  🔍 Soglia similarità semantica: {self.semantic_threshold}")
        print(f"  💾 Cache path: {self.cache_path}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Carica la configurazione dal file YAML"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Errore caricamento config: {e}. Uso valori default.")
            return {}
    
    def _get_tenant_cache_path(self, base_path: str) -> str:
        """
        Genera percorso cache tenant-aware
        
        Args:
            base_path: Percorso base della cache
            
        Returns:
            Percorso cache tenant-specifico nel formato: base_path/{tenant_name}_{tenant_id}/
        """
        try:
            if not self.tenant_name:
                # Se non c'è tenant, usa percorso standard
                return base_path
                
            # Importa MongoClassificationReader per ottenere tenant_id
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from mongo_classification_reader import MongoClassificationReader
            
            # Usa la funzione helper per generare il percorso con OGGETTO TENANT
            mongo_reader = MongoClassificationReader(tenant=self.tenant)
            tenant_cache_path = mongo_reader.generate_semantic_cache_path(self.tenant.tenant_name, "memory")
            
            return tenant_cache_path
            
        except Exception as e:
            print(f"⚠️ Errore nella generazione cache path tenant-aware: {e}")
            # Fallback: usa percorso base con nome tenant
            if self.tenant_name:
                safe_tenant = self.tenant_name.replace(' ', '_').replace('-', '_').lower()
                return os.path.join(base_path, safe_tenant)
            return base_path
    
    def _setup_logging(self):
        """Configura il logging dettagliato"""
        log_file = os.path.join(self.cache_path, 'semantic_memory.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_semantic_memory(self, force_reload: bool = False) -> bool:
        """
        Carica la memoria semantica dal database e cache
        
        Args:
            force_reload: Se True, ricarica completamente dal database
            
        Returns:
            True se caricamento riuscito
        """
        cache_file = os.path.join(self.cache_path, 'semantic_memory.pkl')
        
        # Verifica se usare cache esistente
        if not force_reload and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_memory = pickle.load(f)
                    
                # Verifica se cache è abbastanza recente
                if cached_memory.get('last_updated'):
                    last_update = datetime.fromisoformat(cached_memory['last_updated'])
                    time_diff = datetime.now() - last_update
                    
                    if time_diff.days < 1:  # Cache valida per 24 ore
                        self.semantic_memory = cached_memory
                        
                        # Assicurati che total_sessions sia presente
                        total_sessions = cached_memory.get('total_sessions', 0)
                        if total_sessions == 0 and cached_memory.get('session_ids'):
                            # Calcola dal session_ids se manca
                            total_sessions = sum(len(session_list) for session_list in cached_memory['session_ids'].values())
                            self.semantic_memory['total_sessions'] = total_sessions
                        
                        print(f"📦 Memoria semantica caricata da cache ({total_sessions} sessioni)")
                        return True
            except Exception as e:
                if self.enable_logging:
                    self.logger.warning(f"Errore caricamento cache: {e}")
        
        # Carica dal database
        print(f"🔄 Caricamento memoria semantica dal database...")
        return self._load_from_database()
    
    def _load_from_database(self) -> bool:
        """Carica la memoria semantica dal database MongoDB (sostituisce MySQL)"""
        try:
            print(f"🔄 Caricamento memoria semantica da MongoDB...")
            
            # Usa MongoDB invece di MySQL per le classificazioni
            # Recupera tutte le sessioni classificate per il tenant corrente
            tenant_slug = self.tenant_name or "humanitas"  # Default fallback
            
            classificazioni = self.mongo_reader.get_all_sessions(
                limit=None  # Tutte le classificazioni
            )
            
            if not classificazioni:
                print("📭 Nessuna classificazione trovata nel database MongoDB")
                return self._initialize_empty_memory()
            
            print(f"📚 Trovate {len(classificazioni)} classificazioni esistenti in MongoDB")
            
            # Reset memoria
            self.semantic_memory = {
                'embeddings': {},
                'texts': {},
                'session_ids': {},
                'last_updated': datetime.now().isoformat(),
                'total_sessions': 0
            }
            
            # Raggruppa per tag e normalizza le etichette
            metadata_by_tag = {}
            label_mappings = {}  # originale -> normalizzata
            
            for doc in classificazioni:
                # Estrae informazioni dal documento MongoDB
                session_id = doc.get('session_id', '')
                tag_name = doc.get('classification', 'altro')  # Campo principale
                confidence = doc.get('confidence', 0.0)
                method = doc.get('method', 'UNKNOWN')
                created_at = doc.get('timestamp', '')
                conversation_text = doc.get('conversation_text', '')
                
                # Skip se dati incompleti
                if not session_id or not tag_name or not conversation_text:
                    continue
                
                # Normalizza l'etichetta
                normalized_tag = self._normalize_label(tag_name)
                
                # Tieni traccia del mapping per riferimento
                if tag_name != normalized_tag:
                    label_mappings[tag_name] = normalized_tag
                    if self.enable_logging:
                        self.logger.info(f"Etichetta normalizzata: '{tag_name}' → '{normalized_tag}'")
                
                # Usa l'etichetta normalizzata
                if normalized_tag not in metadata_by_tag:
                    metadata_by_tag[normalized_tag] = []
                
                metadata_by_tag[normalized_tag].append({
                    'session_id': session_id,
                    'original_tag': tag_name,
                    'confidence': confidence,
                    'method': method,
                    'created_at': created_at,
                    'conversation_text': conversation_text[:200] + '...' if len(conversation_text) > 200 else conversation_text
                })
            
            # Salva i metadati nella memoria semantica
            total_db_sessions = 0
            for tag_name, metadata_list in metadata_by_tag.items():
                original_count = len(metadata_list)
                total_db_sessions += original_count
                print(f"  📝 Tag '{tag_name}': {original_count} sessioni classificate")
                
                # Limita il numero di campioni per tag nella memoria
                limited_metadata_list = metadata_list
                if len(metadata_list) > self.max_samples_per_label:
                    # Prendi i più recenti e quelli con alta confidenza
                    limited_metadata_list = sorted(metadata_list, key=lambda x: (x['confidence'] or 0, x['created_at']), reverse=True)
                    limited_metadata_list = limited_metadata_list[:self.max_samples_per_label]
                    print(f"    🔄 Limitato a {len(limited_metadata_list)} campioni per la memoria")
                
                # Salva metadati (embedding verranno generati al bisogno)
                self.semantic_memory['session_ids'][tag_name] = [m['session_id'] for m in limited_metadata_list]
            
            # Salva il conteggio totale dal database
            self.semantic_memory['total_sessions'] = total_db_sessions
            
            # Salva cache
            self._save_cache()
            
            print(f"✅ Memoria semantica caricata: {len(metadata_by_tag)} tag, {len(classificazioni)} sessioni")
            return True
            
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Errore caricamento da database: {e}")
            print(f"❌ Errore caricamento memoria semantica: {e}")
            return self._initialize_empty_memory()
        
        finally:
            self.db_connector.disconnetti()
    
    def _initialize_empty_memory(self) -> bool:
        """Inizializza una memoria semantica vuota"""
        self.semantic_memory = {
            'embeddings': {},
            'texts': {},
            'session_ids': {},
            'last_updated': datetime.now().isoformat(),
            'total_sessions': 0
        }
        print("📭 Inizializzata memoria semantica vuota")
        return True
    
    def _save_cache(self):
        """Salva la memoria semantica in cache"""
        try:
            cache_file = os.path.join(self.cache_path, 'semantic_memory.pkl')
            with open(cache_file, 'wb') as f:
                pickle.dump(self.semantic_memory, f)
                
            if self.enable_logging:
                self.logger.info(f"Cache salvata: {cache_file}")
                
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Errore salvataggio cache: {e}")
    
    def find_similar_labels(self, 
                          session_text: str, 
                          top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Trova le etichette più simili per una nuova sessione
        
        Args:
            session_text: Testo della sessione da classificare
            top_k: Numero massimo di etichette simili da restituire
            
        Returns:
            Lista di dizionari con etichette simili e scores
        """
        # Se non abbiamo embeddings ma abbiamo session_ids, per ora restituiamo lista vuota
        # In una versione futura, potremmo generare embedding al volo dal database remoto
        if not self.semantic_memory.get('embeddings'):
            if self.enable_logging:
                self.logger.warning("Nessun embedding disponibile per calcolo similarità")
            return []
        
        # Genera embedding per la nuova sessione
        try:
            new_embedding = self.embedder.encode_single(session_text)
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Errore generazione embedding: {e}")
            return []
        
        # Calcola similarità con ogni tag
        similarities = []
        
        for tag_name, tag_embeddings in self.semantic_memory['embeddings'].items():
            if len(tag_embeddings) == 0:
                continue
            
            # Calcola similarità coseno con tutti gli embedding del tag
            tag_similarities = cosine_similarity([new_embedding], tag_embeddings)[0]
            
            # Prendi il massimo, media e mediana delle similarità
            max_sim = np.max(tag_similarities)
            avg_sim = np.mean(tag_similarities)
            median_sim = np.median(tag_similarities)
            
            # Score complessivo (peso maggiore al massimo)
            overall_score = 0.6 * max_sim + 0.3 * avg_sim + 0.1 * median_sim
            
            similarities.append({
                'tag_name': tag_name,
                'max_similarity': float(max_sim),
                'avg_similarity': float(avg_sim),
                'median_similarity': float(median_sim),
                'overall_score': float(overall_score),
                'num_samples': len(tag_embeddings)
            })
        
        # Ordina per score complessivo
        similarities = sorted(similarities, key=lambda x: x['overall_score'], reverse=True)
        
        # Restituisci i top_k risultati
        return similarities[:top_k]
    
    def should_reuse_label(self, 
                          session_text: str, 
                          candidate_label: str = None) -> Tuple[bool, str, float]:
        """
        Determina se riutilizzare un'etichetta esistente per una nuova sessione
        
        Args:
            session_text: Testo della sessione da classificare
            candidate_label: Etichetta candidata specifica (opzionale)
            
        Returns:
            Tuple (should_reuse, best_label, confidence_score)
        """
        # Trova etichette simili
        similar_labels = self.find_similar_labels(session_text, top_k=5)
        
        if not similar_labels:
            return False, None, 0.0
        
        # Se abbiamo un candidato specifico, controllalo
        if candidate_label:
            for sim in similar_labels:
                if sim['tag_name'] == candidate_label:
                    should_reuse = sim['overall_score'] >= self.reuse_threshold
                    return should_reuse, candidate_label, sim['overall_score']
        
        # Altrimenti prendi il migliore
        best_match = similar_labels[0]
        should_reuse = best_match['overall_score'] >= self.reuse_threshold
        
        if self.enable_logging:
            self.logger.info(f"Migliore match per nuova sessione: {best_match['tag_name']} "
                           f"(score: {best_match['overall_score']:.3f})")
        
        return should_reuse, best_match['tag_name'], best_match['overall_score']
    
    def add_new_classification(self, 
                             session_id: str, 
                             session_text: str, 
                             tag_name: str, 
                             confidence: float = None):
        """
        Aggiunge una nuova classificazione alla memoria semantica
        
        Args:
            session_id: ID della sessione
            session_text: Testo della sessione
            tag_name: Nome del tag assegnato
            confidence: Confidenza della classificazione
        """
        try:
            # Genera embedding
            embedding = self.embedder.encode_single(session_text)
            
            # Inizializza il tag se non esiste
            if tag_name not in self.semantic_memory['embeddings']:
                self.semantic_memory['embeddings'][tag_name] = []
                self.semantic_memory['texts'][tag_name] = []
                self.semantic_memory['session_ids'][tag_name] = []
            
            # Aggiungi alla memoria
            self.semantic_memory['embeddings'][tag_name].append(embedding)
            self.semantic_memory['texts'][tag_name].append(session_text)
            self.semantic_memory['session_ids'][tag_name].append(session_id)
            
            # Limita il numero di campioni per tag
            if len(self.semantic_memory['embeddings'][tag_name]) > self.max_samples_per_label:
                # Rimuovi il più vecchio
                self.semantic_memory['embeddings'][tag_name].pop(0)
                self.semantic_memory['texts'][tag_name].pop(0)
                self.semantic_memory['session_ids'][tag_name].pop(0)
            
            self.semantic_memory['total_sessions'] += 1
            self.semantic_memory['last_updated'] = datetime.now().isoformat()
            
            if self.enable_logging:
                self.logger.info(f"Aggiunta classificazione: {session_id} -> {tag_name}")
        
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Errore aggiunta classificazione: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche sulla memoria semantica"""
        # Controlla se abbiamo dati (embeddings o session_ids)
        has_embeddings = bool(self.semantic_memory.get('embeddings'))
        has_sessions = bool(self.semantic_memory.get('session_ids'))
        
        if not has_embeddings and not has_sessions:
            return {'status': 'empty', 'total_sessions': 0, 'total_tags': 0}
        
        # Calcola statistiche dai session_ids (più affidabile)
        total_sessions_in_memory = 0
        tags_distribution_memory = {}
        
        if has_sessions:
            for tag_name, session_list in self.semantic_memory['session_ids'].items():
                count = len(session_list)
                tags_distribution_memory[tag_name] = count
                total_sessions_in_memory += count
        elif has_embeddings:
            for tag_name, embeddings in self.semantic_memory['embeddings'].items():
                count = len(embeddings)
                tags_distribution_memory[tag_name] = count
                total_sessions_in_memory += count
        
        # Ottieni i conteggi totali dal database
        total_sessions_database = self.semantic_memory.get('total_sessions', 0)
        tags_distribution_database = {}
        
        # Se total_sessions è 0 o non presente, prova a recuperarlo fresco dal database
        if total_sessions_database == 0:
            try:
                # Usa MongoDB per ottenere i conteggi freschi
                tenant_slug = self.tenant_name or "humanitas"
                all_classifications = self.mongo_reader.get_all_sessions(
                    limit=None
                )
                
                total_fresh_count = 0
                for doc in all_classifications:
                    tag_name = doc.get('classification', 'altro')
                    if tag_name:
                        normalized_tag = self._normalize_label(tag_name)
                        if normalized_tag in tags_distribution_database:
                            tags_distribution_database[normalized_tag] += 1
                        else:
                            tags_distribution_database[normalized_tag] = 1
                        total_fresh_count += 1
                
                # Aggiorna la memoria con i conteggi freschi
                total_sessions_database = total_fresh_count
                self.semantic_memory['total_sessions'] = total_sessions_database
                
                if self.enable_logging:
                    self.logger.info(f"Aggiornati conteggi database: {total_sessions_database} sessioni totali")
                    
            except Exception as e:
                if self.enable_logging:
                    self.logger.warning(f"Errore recupero conteggi database: {e}")
                # Fallback: usa i conteggi in memoria
                total_sessions_database = total_sessions_in_memory
                tags_distribution_database = tags_distribution_memory.copy()
            finally:
                self.db_connector.disconnetti()
        else:
            # Usa i conteggi in memoria come approssimazione per i tag
            tags_distribution_database = tags_distribution_memory.copy()
        
        stats = {
            'status': 'loaded',
            'memory_sessions': total_sessions_in_memory,
            'database_sessions': total_sessions_database,
            'total_tags': len(tags_distribution_memory),
            'last_updated': self.semantic_memory.get('last_updated'),
            'tags_distribution_memory': tags_distribution_memory,
            'tags_distribution_database': tags_distribution_database,
            'has_embeddings': has_embeddings,
            'has_session_metadata': has_sessions,
            'max_samples_per_label': self.max_samples_per_label
        }
        
        return stats
    
    def clear_cache(self):
        """Cancella la cache della memoria semantica"""
        cache_file = os.path.join(self.cache_path, 'semantic_memory.pkl')
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print("🗑️ Cache memoria semantica cancellata")
    
    def _normalize_label(self, label: str) -> str:
        """
        Normalizza un'etichetta per evitare duplicati
        
        Args:
            label: Etichetta da normalizzare
            
        Returns:
            Etichetta normalizzata
        """
        if not label:
            return label
            
        # Converti in minuscolo
        normalized = label.lower().strip()
        
        # Sostituisci spazi con underscore
        normalized = re.sub(r'\s+', '_', normalized)
        
        # Rimuovi caratteri speciali (mantieni solo lettere, numeri, underscore)
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        
        # Rimuovi underscore multipli
        normalized = re.sub(r'_+', '_', normalized)
        
        # Rimuovi underscore all'inizio e alla fine
        normalized = normalized.strip('_')
        
        return normalized
    
    def _calculate_label_similarity(self, label1: str, label2: str) -> float:
        """
        Calcola la similarità tra due etichette
        
        Args:
            label1, label2: Etichette da confrontare
            
        Returns:
            Score di similarità (0.0 - 1.0)
        """
        # Normalizza entrambe
        norm1 = self._normalize_label(label1)
        norm2 = self._normalize_label(label2)
        
        # Se sono identiche dopo normalizzazione
        if norm1 == norm2:
            return 1.0
        
        # Calcola similarità con SequenceMatcher
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        return similarity
    
    def _find_duplicate_labels(self, labels: List[str], similarity_threshold: float = 0.85) -> Dict[str, List[str]]:
        """
        Trova etichette duplicate o molto simili
        
        Args:
            labels: Lista di etichette da analizzare
            similarity_threshold: Soglia di similarità per considerare duplicati
            
        Returns:
            Dizionario {etichetta_canonica: [etichette_simili]}
        """
        duplicates = {}
        processed = set()
        
        for i, label1 in enumerate(labels):
            if label1 in processed:
                continue
                
            similar_labels = [label1]
            
            for j, label2 in enumerate(labels[i+1:], i+1):
                if label2 in processed:
                    continue
                    
                similarity = self._calculate_label_similarity(label1, label2)
                
                if similarity >= similarity_threshold:
                    similar_labels.append(label2)
                    processed.add(label2)
            
            if len(similar_labels) > 1:
                # Usa l'etichetta normalizzata come canonica
                canonical = self._normalize_label(label1)
                duplicates[canonical] = similar_labels
                
            processed.add(label1)
        
        return duplicates
    
    def clean_duplicate_labels(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Pulisce le etichette duplicate dal database
        
        Args:
            dry_run: Se True, mostra solo cosa verrebbe fatto senza eseguire
            
        Returns:
            Rapporto delle operazioni eseguite
        """
        print("🧹 Analisi duplicati etichette...")
        
        try:
            # Usa MongoDB per ottenere le etichette uniche
            tenant_slug = self.tenant_name or "humanitas"
            all_classifications = self.mongo_reader.get_all_sessions(
                limit=None
            )
            
            if not all_classifications:
                print("📭 Nessuna etichetta trovata")
                return {'status': 'no_labels'}
            
            # Estrai le etichette uniche
            labels = set()
            for doc in all_classifications:
                tag_name = doc.get('classification', '')
                if tag_name and tag_name.strip():
                    labels.add(tag_name)
            
            labels = sorted(list(labels))
            print(f"📋 Trovate {len(labels)} etichette uniche: {labels}")
            
            # Trova duplicati
            duplicates = self._find_duplicate_labels(labels, similarity_threshold=0.85)
            
            if not duplicates:
                print("✅ Nessun duplicato trovato")
                return {'status': 'no_duplicates', 'labels': labels}
            
            print(f"🔍 Trovati {len(duplicates)} gruppi di duplicati:")
            
            report = {
                'status': 'duplicates_found',
                'total_labels': len(labels),
                'duplicate_groups': len(duplicates),
                'operations': [],
                'dry_run': dry_run
            }
            
            for canonical, similar_labels in duplicates.items():
                print(f"\n📝 Gruppo: {canonical}")
                print(f"  🔄 Unifica: {similar_labels}")
                
                if not dry_run:
                    # Esegui unificazione usando MongoDB
                    for similar_label in similar_labels[1:]:  # Salta il primo (quello da mantenere)
                        # Per ora skip l'unificazione automatica su MongoDB
                        # Questo richiederebbe una funzione update_many sul mongo_reader
                        print(f"    ⚠️ Unificazione {similar_label} → {canonical} SKIPPED (MongoDB)")
                        print(f"    💡 Implementare update_classification_tags nel MongoClassificationReader")
                        
                        report['operations'].append({
                            'from': similar_label,
                            'to': canonical,
                            'affected_rows': 0,  # Non implementato ancora
                            'status': 'skipped_mongodb'
                        })
                else:
                    # Solo dry run - conta usando MongoDB
                    for similar_label in similar_labels[1:]:
                        # Conta documenti con questo tag in MongoDB
                        tenant_slug = self.tenant_name or "humanitas"
                        all_docs = self.mongo_reader.get_all_sessions(
                            limit=None
                        )
                        count = sum(1 for doc in all_docs if doc.get('classification') == similar_label)
                        
                        print(f"    🔄 {similar_label} → {canonical} ({count} documenti)")
                        report['operations'].append({
                            'from': similar_label,
                            'to': canonical,
                            'affected_rows': count,
                            'dry_run': True
                        })
            
            if dry_run:
                print(f"\n⚠️  MODALITÀ DRY RUN - Nessuna modifica eseguita")
                print(f"   Esegui con dry_run=False per applicare le modifiche")
            else:
                print(f"\n✅ Pulizia duplicati completata!")
                # Invalida la cache per ricaricare
                self.clear_cache()
            
            return report
            
        except Exception as e:
            print(f"❌ Errore durante pulizia duplicati: {e}")
            return {'status': 'error', 'error': str(e)}
            
        finally:
            self.db_connector.disconnetti()

    # ============================================================================
    # NOVELTY DETECTION METHODS - Metodi per rilevare conversazioni semanticamente nuove
    # ============================================================================
    
    def calculate_novelty_score(self, embedding: np.ndarray, tenant: str = None) -> float:
        """
        Calcola il novelty score per un embedding basandosi sulla distanza semantica
        dagli embeddings esistenti nella memoria.
        
        Questo metodo implementa la novelty detection richiesta dal QualityGateEngine
        per identificare conversazioni su argomenti mai visti prima.
        
        Args:
            embedding: Vector embedding della conversazione da valutare
            tenant: Nome del tenant (opzionale, per filtrare memoria per tenant)
            
        Returns:
            float: Score di novelty [0.0, 1.0] dove 1.0 = completamente nuovo
        """
        if not hasattr(self, 'session_embeddings') or len(self.session_embeddings) == 0:
            # Nessuna memoria disponibile, considera tutto nuovo
            self.logger.debug("Nessuna memoria semantica disponibile per novelty detection")
            return 0.0
        
        try:
            # Normalizza l'embedding di input per calcoli coseno efficienti
            normalized_input = embedding / np.linalg.norm(embedding)
            
            # Prepara matrice degli embeddings esistenti
            if tenant and hasattr(self, 'session_metadata'):
                # Filtra per tenant se specificato e se abbiamo metadati
                tenant_embeddings = []
                for session_id, emb in self.session_embeddings.items():
                    session_meta = self.session_metadata.get(session_id, {})
                    if session_meta.get('tenant') == tenant:
                        tenant_embeddings.append(emb)
                
                if not tenant_embeddings:
                    self.logger.debug(f"Nessun embedding trovato per tenant {tenant}")
                    return 0.0
                    
                embeddings_matrix = np.array(tenant_embeddings)
            else:
                # Usa tutti gli embeddings disponibili
                embeddings_matrix = np.array(list(self.session_embeddings.values()))
            
            # Calcola similarità coseno con tutti gli embeddings esistenti
            similarities = cosine_similarity([normalized_input], embeddings_matrix)[0]
            
            # Trova la similarità massima (esempio più simile)
            max_similarity = np.max(similarities)
            
            # Converti in novelty score: alta similarità = bassa novelty
            novelty_score = 1.0 - max_similarity
            
            # Applica soglia configurabile
            novelty_threshold = self.similarity_config.get('novelty_threshold', 0.3)
            
            self.logger.debug(f"Novelty detection: max_similarity={max_similarity:.3f}, "
                            f"novelty_score={novelty_score:.3f}, threshold={novelty_threshold}")
            
            return float(max(0.0, min(1.0, novelty_score)))
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo novelty score: {e}")
            # In caso di errore, considera come nuovo per sicurezza
            return 1.0
    
    def add_embedding_to_memory(self, session_id: str, embedding: np.ndarray, 
                               tenant: str = None, label: str = None) -> bool:
        """
        Aggiunge un nuovo embedding alla memoria semantica per future novelty detection.
        
        Args:
            session_id: ID della sessione
            embedding: Vector embedding della conversazione
            tenant: Nome del tenant (opzionale)
            label: Etichetta assegnata (opzionale)
            
        Returns:
            bool: True se aggiunto con successo
        """
        try:
            # Inizializza strutture dati se necessario
            if not hasattr(self, 'session_embeddings'):
                self.session_embeddings = {}
            if not hasattr(self, 'session_metadata'):
                self.session_metadata = {}
            
            # Normalizza embedding per efficienza
            normalized_embedding = embedding / np.linalg.norm(embedding)
            
            # Aggiungi alla memoria
            self.session_embeddings[session_id] = normalized_embedding
            
            # Aggiungi metadati
            self.session_metadata[session_id] = {
                'tenant': tenant,
                'label': label,
                'timestamp': datetime.now().isoformat(),
                'embedding_dim': len(embedding)
            }
            
            # Gestisci dimensione massima memoria
            max_memory_size = self.memory_config.get('max_embeddings', 10000)
            if len(self.session_embeddings) > max_memory_size:
                # Rimuovi gli embeddings più vecchi
                oldest_sessions = sorted(self.session_metadata.items(), 
                                       key=lambda x: x[1]['timestamp'])[:100]
                for old_session_id, _ in oldest_sessions:
                    if old_session_id in self.session_embeddings:
                        del self.session_embeddings[old_session_id]
                    if old_session_id in self.session_metadata:
                        del self.session_metadata[old_session_id]
                
                self.logger.info(f"Rimossi {len(oldest_sessions)} embeddings vecchi dalla memoria")
            
            self.logger.debug(f"Aggiunto embedding per sessione {session_id} alla memoria semantica")
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiunta embedding {session_id}: {e}")
            return False
    
    def get_similar_conversations(self, embedding: np.ndarray, tenant: str = None, 
                                 top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Trova le conversazioni più simili nella memoria semantica.
        
        Utile per analizzare il contesto di decisioni di novelty e per
        fornire esempi simili agli operatori umani durante la revisione.
        
        Args:
            embedding: Vector embedding di riferimento
            tenant: Nome del tenant (opzionale)
            top_k: Numero di conversazioni simili da restituire
            
        Returns:
            Lista di conversazioni simili con metadati e score di similarità
        """
        if not hasattr(self, 'session_embeddings') or len(self.session_embeddings) == 0:
            return []
        
        try:
            # Normalizza embedding input
            normalized_input = embedding / np.linalg.norm(embedding)
            
            # Prepara dati per calcolo similarità
            session_ids = []
            embeddings_list = []
            
            for session_id, emb in self.session_embeddings.items():
                if tenant and hasattr(self, 'session_metadata'):
                    session_meta = self.session_metadata.get(session_id, {})
                    if session_meta.get('tenant') != tenant:
                        continue
                
                session_ids.append(session_id)
                embeddings_list.append(emb)
            
            if not embeddings_list:
                return []
            
            # Calcola similarità
            embeddings_matrix = np.array(embeddings_list)
            similarities = cosine_similarity([normalized_input], embeddings_matrix)[0]
            
            # Trova i top_k più simili
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            similar_conversations = []
            for idx in top_indices:
                session_id = session_ids[idx]
                similarity_score = similarities[idx]
                
                conversation_data = {
                    'session_id': session_id,
                    'similarity_score': float(similarity_score),
                    'distance_score': float(1.0 - similarity_score)
                }
                
                # Aggiungi metadati se disponibili
                if hasattr(self, 'session_metadata') and session_id in self.session_metadata:
                    conversation_data.update(self.session_metadata[session_id])
                
                similar_conversations.append(conversation_data)
            
            return similar_conversations
            
        except Exception as e:
            self.logger.error(f"Errore nella ricerca conversazioni simili: {e}")
            return []
    
    def get_novelty_statistics(self, tenant: str = None) -> Dict[str, Any]:
        """
        Restituisce statistiche sulla novelty detection per un tenant.
        
        Args:
            tenant: Nome del tenant (opzionale)
            
        Returns:
            Dizionario con statistiche di novelty
        """
        try:
            if not hasattr(self, 'session_embeddings'):
                return {
                    'total_embeddings': 0,
                    'tenant_embeddings': 0,
                    'memory_utilization': 0.0,
                    'avg_embedding_dimension': 0
                }
            
            total_embeddings = len(self.session_embeddings)
            tenant_embeddings = total_embeddings
            
            # Filtra per tenant se specificato
            if tenant and hasattr(self, 'session_metadata'):
                tenant_embeddings = sum(1 for meta in self.session_metadata.values() 
                                      if meta.get('tenant') == tenant)
            
            # Calcola statistiche
            avg_dimension = 0
            if self.session_embeddings:
                dimensions = [len(emb) for emb in self.session_embeddings.values()]
                avg_dimension = np.mean(dimensions)
            
            max_memory = self.memory_config.get('max_embeddings', 10000)
            
            return {
                'total_embeddings': total_embeddings,
                'tenant_embeddings': tenant_embeddings,
                'memory_utilization': total_embeddings / max_memory,
                'avg_embedding_dimension': int(avg_dimension),
                'memory_config': {
                    'max_embeddings': max_memory,
                    'novelty_threshold': self.similarity_config.get('novelty_threshold', 0.3)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo statistiche novelty: {e}")
            return {'error': str(e)}

# Test del SemanticMemoryManager
if __name__ == "__main__":
    print("=== TEST SEMANTIC MEMORY MANAGER ===\n")
    
    # Crea un tenant di test per il test
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Utils.tenant import Tenant
    
    test_tenant = Tenant(
        tenant_id="015007d9-d413-11ef-86a5-96000228e7fe",
        tenant_name="humanitas",
        tenant_slug="humanitas",
        is_active=True
    )
    
    manager = SemanticMemoryManager(tenant=test_tenant)
    
    try:
        # Carica memoria
        if manager.load_semantic_memory():
            
            # Mostra statistiche
            stats = manager.get_memory_stats()
            print(f"\n📊 STATISTICHE MEMORIA:")
            print(f"  Status: {stats['status']}")
            print(f"  Sessioni totali: {stats['total_sessions']}")
            print(f"  Tag totali: {stats['total_tags']}")
            print(f"  Ultimo aggiornamento: {stats['last_updated']}")
            
            print(f"\n📋 Distribuzione tag:")
            for tag, count in stats['tags_distribution'].items():
                print(f"  {tag}: {count} campioni")
            
            # Test pulizia duplicati
            print(f"\n🧹 TEST PULIZIA DUPLICATI:")
            cleanup_report = manager.clean_duplicate_labels(dry_run=True)
            
            if cleanup_report['status'] == 'duplicates_found':
                print(f"  🔍 Trovati {cleanup_report['duplicate_groups']} gruppi di duplicati")
                print(f"  📊 Operazioni pianificate: {len(cleanup_report['operations'])}")
                
                # Chiedi conferma per applicare
                print(f"\n❓ Vuoi applicare la pulizia duplicati? (y/N)")
                # Per il test automatico, non applichiamo
                print(f"   (Test automatico - pulizia non applicata)")
            else:
                print(f"  ✅ {cleanup_report['status']}")
            
            # Test similarità
            test_texts = [
                "Non riesco ad accedere al portale Humanitas",
                "Vorrei prenotare una visita cardiologica", 
                "Come posso ritirare i referti degli esami?",
                "Ho problemi con la fatturazione"
            ]
            
            print(f"\n🧪 TEST SIMILARITÀ:")
            for test_text in test_texts:
                should_reuse, best_label, score = manager.should_reuse_label(test_text)
                
                print(f"\n📝 Testo: '{test_text}'")
                print(f"🔄 Riutilizza etichetta: {'✅' if should_reuse else '❌'}")
                if best_label:
                    print(f"🏷️  Etichetta migliore: {best_label}")
                    print(f"🎯 Score: {score:.3f}")
        
        else:
            print("❌ Impossibile caricare memoria semantica")
    
    except Exception as e:
        print(f"❌ Errore nel test: {e}")
        import traceback
        traceback.print_exc()
