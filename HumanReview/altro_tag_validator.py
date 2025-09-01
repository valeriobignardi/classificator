"""
Sistema di validazione per i tag "ALTRO" durante il training supervisionato

Nuova logica basata su embedding semantico:
1. Ottiene proposta tag dalla raw response LLM
2. Calcola embedding della proposta con motore tenant-specific
3. Confronta embedding con tutti i tag esistenti
4. Se similarità >= 85%: usa tag esistente più simile
5. Se similarità < 85%: crea nuovo tag dalla proposta LLM

Autore: Valerio Bignardi
Data: 28 Agosto 2025
Ultima modifica: 2025-08-28 - Riscrittura completa con logica embedding-based
"""

import sys
import os
import logging
import json
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Import Tenant per principio universale
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from tenant import Tenant
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass

# Aggiunge i percorsi necessari
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EmbeddingEngine'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Database'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from schema_manager import ClassificationSchemaManager
from mongo_classification_reader import MongoClassificationReader

# Import per embedding dinamico
try:
    from EmbeddingEngine.simple_embedding_manager import simple_embedding_manager
    EMBEDDING_MANAGER_AVAILABLE = True
except ImportError:
    from EmbeddingEngine.labse_embedder import LaBSEEmbedder
    EMBEDDING_MANAGER_AVAILABLE = False

@dataclass
class ValidationResult:
    """
    Risultato della validazione di un tag "ALTRO" - versione embedding-based
    """
    should_add_new_tag: bool
    final_tag: str
    confidence: float
    validation_path: str  # "similarity_match", "new_tag_created", "error"
    similarity_score: Optional[float] = None
    matched_existing_tag: Optional[str] = None
    llm_suggested_tag: Optional[str] = None
    needs_human_review: bool = False

class AltroTagValidator:
    """
    Validatore per i tag classificati come "ALTRO" - versione embedding-based
    
    Nuova implementazione basata su:
    - Embedding semantico della proposta LLM
    - Confronto con embedding di tag esistenti
    - Soglia configurabile per decidere nuovo vs esistente
    """
    
    def __init__(self, tenant: Tenant, config: Dict[str, Any] = None):
        """
        Inizializza il validatore embedding-based
        
        PRINCIPIO UNIVERSALE: Accetta oggetto Tenant completo
        
        Scopo della funzione: Inizializza validatore con embedder e configurazione
        Parametri di input: tenant (oggetto Tenant), config (configurazione opzionale)
        Parametri di output: Istanza AltroTagValidator configurata
        Valori di ritorno: None (costruttore)
        Tracciamento aggiornamenti: 2025-08-29 - Convertito a principio universale
        
        Args:
            tenant: Oggetto Tenant completo
            config: Configurazione completa (se None, carica da file)
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        self.tenant = tenant
        self.tenant_id = tenant.tenant_id  # Estrae tenant_id dall'oggetto Tenant
        self.logger = logging.getLogger(__name__)
        
        # Carica configurazione
        self.config = config if config else self._load_config()
        
        # Parametri da configurazione
        altro_config = self.config.get('altro_tag_validator', {})
        self.similarity_threshold = altro_config.get('semantic_similarity_threshold', 0.85)
        self.enable_cache = altro_config.get('enable_embedding_cache', True)
        self.max_cache_size = altro_config.get('max_embedding_cache_size', 1000)
        
        self.logger.info(f"🔧 AltroTagValidator inizializzato - soglia: {self.similarity_threshold}")
        
        # Inizializza i componenti necessari
        self.schema_manager = ClassificationSchemaManager()
        
        # Crea MongoClassificationReader usando oggetto Tenant già disponibile
        try:
            self.db_connector = MongoClassificationReader(tenant=self.tenant)
            self.logger.info(f"🗄️ MongoClassificationReader inizializzato per tenant: {self.tenant.tenant_name}")
        except Exception as e:
            self.logger.error(f"⚠️ Errore creazione MongoClassificationReader per tenant '{self.tenant.tenant_name}': {e}")
            raise ValueError(f"Impossibile creare MongoClassificationReader per tenant: {self.tenant.tenant_name}")
        
        # ✅ Embedder dinamico configurato per il tenant
        self.embedder = self._get_dynamic_embedder()
        
        # Cache per embedding dei tag esistenti (performance)
        self._embedding_cache = {} if self.enable_cache else None
        self._tags_cache = None
        self._last_cache_update = None
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Carica configurazione da config.yaml
        
        Scopo della funzione: Legge configurazione sistema da file
        Parametri di input: None
        Parametri di output: Dict con configurazione completa
        Valori di ritorno: Configurazione caricata
        Tracciamento aggiornamenti: 2025-08-28 - Creata per nuova implementazione
        
        Returns:
            Configurazione completa del sistema
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
            
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
                
        except Exception as e:
            self.logger.error(f"❌ Errore caricamento config: {e}")
            # Configurazione di fallback
            return {
                'altro_tag_validator': {
                    'semantic_similarity_threshold': 0.85,
                    'enable_embedding_cache': True,
                    'max_embedding_cache_size': 1000
                }
            }

    def _get_dynamic_embedder(self):
        """
        Ottiene embedder dinamico configurato per il tenant
        
        Scopo della funzione: Inizializza embedder tenant-specific da configurazione
        Parametri di input: None (usa self.tenant_id e self.config)
        Parametri di output: Istanza embedder configurata
        Valori di ritorno: Embedder pronto all'uso
        Tracciamento aggiornamenti: 2025-08-28 - Creata per nuova architettura
        
        Returns:
            Embedder configurato per il tenant
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        try:
            # ✅ Usa il simple_embedding_manager (istanza singleton)
            if EMBEDDING_MANAGER_AVAILABLE:
                # Ottiene embedder configurato per il tenant - usa oggetto Tenant completo
                embedder = simple_embedding_manager.get_embedder_for_tenant(self.tenant)
                
                self.logger.info(f"✅ Embedder dinamico inizializzato per tenant {self.tenant.tenant_name} ({self.tenant.tenant_id})")
                return embedder
            else:
                # Fallback al LaBSE embedder
                self.logger.warning("⚠️ Usando fallback LaBSEEmbedder")
                return LaBSEEmbedder()
                
        except Exception as e:
            self.logger.error(f"❌ Errore inizializzazione embedder: {e}")
            # Fallback estremo
            if not EMBEDDING_MANAGER_AVAILABLE:
                return LaBSEEmbedder()
            raise e

    def _get_tag_embedding(self, tag_text: str) -> np.ndarray:
        """
        Calcola embedding di un tag con caching
        
        Scopo della funzione: Ottiene embedding di un tag con gestione cache
        Parametri di input: tag_text (testo del tag)
        Parametri di output: Embedding normalizzato del tag
        Valori di ritorno: Array numpy con embedding
        Tracciamento aggiornamenti: 2025-08-28 - Creata per nuova logica
        
        Args:
            tag_text: Testo del tag da embeddare
            
        Returns:
            Embedding normalizzato del tag
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        # Verifica cache se abilitata
        if self.enable_cache and self._embedding_cache and tag_text in self._embedding_cache:
            return self._embedding_cache[tag_text]
        
        try:
            # Calcola embedding
            if hasattr(self.embedder, 'get_embedding'):
                embedding = self.embedder.get_embedding(tag_text)
            else:
                # Fallback per embedder senza get_embedding
                embedding = self.embedder.encode([tag_text])[0]
            
            # Normalizza embedding
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            # Normalizzazione L2 per similarità coseno
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Salva in cache se abilitata
            if self.enable_cache and self._embedding_cache:
                # Gestisce dimensione cache
                if len(self._embedding_cache) >= self.max_cache_size:
                    # Rimuove primo elemento (FIFO)
                    oldest_key = next(iter(self._embedding_cache))
                    del self._embedding_cache[oldest_key]
                
                self._embedding_cache[tag_text] = embedding
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"❌ Errore calcolo embedding per '{tag_text}': {e}")
            # Ritorna embedding zero come fallback
            return np.zeros(768)  # Dimensione tipica embedding

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcola similarità coseno tra due embedding
        
        Scopo della funzione: Calcola similarità semantica tra due embedding
        Parametri di input: embedding1, embedding2 (array numpy)
        Parametri di output: Score similarità [0-1]
        Valori di ritorno: Valore float della similarità coseno
        Tracciamento aggiornamenti: 2025-08-28 - Creata per logica embedding
        
        Args:
            embedding1: Primo embedding normalizzato
            embedding2: Secondo embedding normalizzato
            
        Returns:
            Similarità coseno tra 0 e 1
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        try:
            # Calcola similarità coseno (prodotto scalare per embedding normalizzati)
            similarity = np.dot(embedding1, embedding2)
            
            # Clamp tra 0 e 1 per sicurezza
            similarity = max(0.0, min(1.0, similarity))
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"❌ Errore calcolo similarità: {e}")
            return 0.0

    def _get_existing_tags(self) -> List[str]:
        """
        Ottiene lista tag esistenti con caching
        
        Scopo della funzione: Recupera tag esistenti dal database con cache
        Parametri di input: None
        Parametri di output: Lista tag esistenti
        Valori di ritorno: List di stringhe con i tag
        Tracciamento aggiornamenti: 2025-08-28 - Creata per nuova implementazione
        
        Returns:
            Lista dei tag esistenti
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        try:
            # Verifica cache
            current_time = datetime.now()
            
            if (self._tags_cache is not None and 
                self._last_cache_update is not None and 
                (current_time - self._last_cache_update).seconds < 300):  # Cache 5 minuti
                return self._tags_cache
            
            # Recupera tag dal database
            tag_objects = self.schema_manager.get_all_tags()
            
            # Estrae i nomi dei tag
            existing_tags = [tag['tag_name'] for tag in tag_objects if tag and tag.get('tag_name')]
            
            # Filtra tag validi (non vuoti, non "ALTRO")
            valid_tags = [
                tag for tag in existing_tags 
                if tag and tag.strip() and tag.upper() != "ALTRO"
            ]
            
            # Aggiorna cache
            self._tags_cache = valid_tags
            self._last_cache_update = current_time
            
            self.logger.info(f"📋 Recuperati {len(valid_tags)} tag esistenti")
            return valid_tags
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero tag esistenti: {e}")
            return []

    def _extract_tag_from_llm_response(self, llm_raw_response: str) -> Optional[str]:
        """
        Estrae tag suggerito dalla response LLM raw
        
        Scopo della funzione: Parsing della response LLM per estrarre tag proposto
        Parametri di input: llm_raw_response (response completa LLM)
        Parametri di output: Tag estratto e pulito
        Valori di ritorno: String tag o None se non trovato
        Tracciamento aggiornamenti: 2025-08-28 - Creata per parsing LLM
        
        Args:
            llm_raw_response: Response completa del LLM
            
        Returns:
            Tag suggerito dal LLM o None se non trovato
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        if not llm_raw_response or not llm_raw_response.strip():
            return None
            
        try:
            # Cerca pattern comuni per tag suggeriti
            patterns = [
                r'(?:tag|etichetta|categoria|classificazione):\s*["\']?([^"\'\n\r]+)["\']?',
                r'suggerisco.*["\']([^"\'\n\r]+)["\']',
                r'propongo.*["\']([^"\'\n\r]+)["\']',
                r'classificherei.*["\']([^"\'\n\r]+)["\']',
                r'["\']([^"\']+)["\'].*(?:appropriat|adeguat|corrett)',
                # Pattern per tag alla fine della risposta
                r'.*["\']([A-Z_][A-Z0-9_\s]{2,})["\']$'
            ]
            
            response_clean = llm_raw_response.strip()
            
            for pattern in patterns:
                matches = re.finditer(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    candidate_tag = match.group(1).strip()
                    
                    # Verifica che sia un tag valido
                    if (len(candidate_tag) > 2 and 
                        len(candidate_tag) < 100 and 
                        candidate_tag.upper() != "ALTRO"):
                        
                        # Pulisci e formatta tag
                        clean_tag = candidate_tag.upper().strip()
                        clean_tag = re.sub(r'\s+', '_', clean_tag)  # Spazi -> underscore
                        clean_tag = re.sub(r'[^\w\s]', '', clean_tag)  # Rimuovi punteggiatura
                        
                        self.logger.info(f"📝 Tag estratto da LLM: '{clean_tag}'")
                        return clean_tag
            
            # Fallback: cerca ultime parole in maiuscolo
            words = response_clean.split()
            for word in reversed(words[-5:]):  # Ultimi 5 parole
                clean_word = re.sub(r'[^\w]', '', word).upper()
                if (len(clean_word) > 2 and 
                    clean_word.isupper() and 
                    clean_word != "ALTRO"):
                    self.logger.info(f"📝 Tag fallback da LLM: '{clean_word}'")
                    return clean_word
            
            self.logger.warning(f"⚠️ Nessun tag estratto da: '{response_clean[:100]}...'")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Errore estrazione tag da LLM response: {e}")
            return None

    def validate_altro_classification(
        self, 
        conversation_id: str,
        llm_raw_response: str,
        conversation_data: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Valida classificazione "ALTRO" usando embedding semantico
        
        Scopo della funzione: Pipeline completa validazione ALTRO con embedding
        Parametri di input: conversation_id, llm_raw_response, conversation_data
        Parametri di output: ValidationResult con decisione finale
        Valori di ritorno: Risultato validazione completo
        Tracciamento aggiornamenti: 2025-08-28 - Riscrittura embedding-based
        
        Args:
            conversation_id: ID conversazione da validare
            llm_raw_response: Response raw del LLM con tag suggerito
            conversation_data: Dati conversazione (opzionale)
            
        Returns:
            ValidationResult con decisione finale e dettagli
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        self.logger.info(f"🔍 Validazione ALTRO per conversazione {conversation_id}")
        
        try:
            # Step 1: Estrai tag dalla response LLM
            suggested_tag = self._extract_tag_from_llm_response(llm_raw_response)
            
            if not suggested_tag:
                self.logger.warning(f"⚠️ Nessun tag estratto da LLM - mantiene ALTRO")
                return ValidationResult(
                    should_add_new_tag=False,
                    final_tag="ALTRO",
                    confidence=0.0,
                    validation_path="error",
                    llm_suggested_tag=None
                )
            
            # Step 2: Calcola embedding del tag suggerito
            suggested_embedding = self._get_tag_embedding(suggested_tag)
            
            # Step 3: Ottieni tag esistenti e calcola embedding
            existing_tags = self._get_existing_tags()
            
            if not existing_tags:
                self.logger.info(f"📝 Nessun tag esistente - crea nuovo: '{suggested_tag}'")
                return ValidationResult(
                    should_add_new_tag=True,
                    final_tag=suggested_tag,
                    confidence=1.0,
                    validation_path="new_tag_created",
                    llm_suggested_tag=suggested_tag
                )
            
            # Step 4: Confronta con tutti i tag esistenti
            best_similarity = 0.0
            best_matching_tag = None
            
            self.logger.info(f"🔍 Confronto con {len(existing_tags)} tag esistenti...")
            
            for existing_tag in existing_tags:
                existing_embedding = self._get_tag_embedding(existing_tag)
                similarity = self._calculate_similarity(suggested_embedding, existing_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_matching_tag = existing_tag
            
            self.logger.info(f"🎯 Migliore match: '{best_matching_tag}' (similarità: {best_similarity:.3f})")
            
            # Step 5: Decisione basata su soglia
            if best_similarity >= self.similarity_threshold:
                # Usa tag esistente più simile
                self.logger.info(f"✅ Similarità >= {self.similarity_threshold} - usa tag esistente: '{best_matching_tag}'")
                
                return ValidationResult(
                    should_add_new_tag=False,
                    final_tag=best_matching_tag,
                    confidence=best_similarity,
                    validation_path="similarity_match",
                    similarity_score=best_similarity,
                    matched_existing_tag=best_matching_tag,
                    llm_suggested_tag=suggested_tag
                )
            else:
                # ✅ CORREZIONE CRUCIALE: Crea nuovo tag (NON etichettare come ALTRO)
                self.logger.info(f"📝 Similarità < {self.similarity_threshold} - crea nuovo tag: '{suggested_tag}'")
                
                return ValidationResult(
                    should_add_new_tag=True,
                    final_tag=suggested_tag,
                    confidence=1.0 - best_similarity,  # Confidence = differenza semantica
                    validation_path="new_tag_created",
                    similarity_score=best_similarity,
                    matched_existing_tag=best_matching_tag,
                    llm_suggested_tag=suggested_tag
                )
                
        except Exception as e:
            self.logger.error(f"❌ Errore validazione altro: {e}", exc_info=True)
            
            return ValidationResult(
                should_add_new_tag=False,
                final_tag="ALTRO",
                confidence=0.0,
                validation_path="error",
                needs_human_review=True
            )

    def add_new_tag_to_database(self, new_tag: str, conversation_id: str) -> bool:
        """
        Aggiunge nuovo tag al database
        
        Scopo della funzione: Inserisce nuovo tag nel database tag
        Parametri di input: new_tag (nuovo tag), conversation_id (ID conversazione)
        Parametri di output: Successo operazione
        Valori di ritorno: True se successo, False altrimenti
        Tracciamento aggiornamenti: 2025-08-28 - Creata per gestione nuovi tag
        
        Args:
            new_tag: Tag da aggiungere
            conversation_id: ID conversazione che ha generato il tag
            
        Returns:
            True se tag aggiunto con successo, False altrimenti
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        try:
            # Verifica che tag non esista già
            tag_objects = self.schema_manager.get_all_tags()
            existing_tag_names = [tag['tag_name'] for tag in tag_objects if tag and tag.get('tag_name')]
            
            if new_tag in existing_tag_names:
                self.logger.info(f"ℹ️ Tag '{new_tag}' già esistente")
                return True
            
            # Aggiunge nuovo tag usando schema_manager
            success = self.schema_manager.add_tag_if_not_exists(
                tag_name=new_tag,
                tag_description=f"Tag creato automaticamente da conversazione {conversation_id}",
                tag_color="#9C27B0"  # Colore viola per tag auto-generati
            )
            
            if success:
                self.logger.info(f"✅ Nuovo tag aggiunto: '{new_tag}'")
                
                # Invalida cache
                self._tags_cache = None
                self._last_cache_update = None
                
                return True
            else:
                self.logger.error(f"❌ Fallimento aggiunta tag: '{new_tag}'")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Errore aggiunta nuovo tag '{new_tag}': {e}")
            return False

    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Ottiene statistiche della validazione
        
        Scopo della funzione: Fornisce metriche performance validatore
        Parametri di input: None
        Parametri di output: Dict con statistiche
        Valori di ritorno: Dizionario con metriche correnti
        Tracciamento aggiornamenti: 2025-08-28 - Creata per monitoraggio
        
        Returns:
            Dizionario con statistiche correnti del validatore
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        try:
            stats = {
                "tenant_id": self.tenant_id,
                "similarity_threshold": self.similarity_threshold,
                "embedding_cache_enabled": self.enable_cache,
                "embedding_cache_size": len(self._embedding_cache) if self._embedding_cache else 0,
                "max_cache_size": self.max_cache_size,
                "existing_tags_count": len(self._tags_cache) if self._tags_cache else 0,
                "last_cache_update": self._last_cache_update.isoformat() if self._last_cache_update else None,
                "embedder_type": type(self.embedder).__name__
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Errore recupero statistiche: {e}")
            return {"error": str(e)}

    def clear_caches(self) -> None:
        """
        Pulisce tutte le cache
        
        Scopo della funzione: Reset completo cache per aggiornamento dati
        Parametri di input: None
        Parametri di output: None
        Valori di ritorno: None
        Tracciamento aggiornamenti: 2025-08-28 - Creata per gestione cache
        
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        try:
            if self._embedding_cache:
                self._embedding_cache.clear()
            
            self._tags_cache = None
            self._last_cache_update = None
            
            self.logger.info("🧹 Cache pulite completamente")
            
        except Exception as e:
            self.logger.error(f"❌ Errore pulizia cache: {e}")
