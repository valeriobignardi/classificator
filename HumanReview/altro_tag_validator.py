"""
Sistema di validazione per i tag "ALTRO" durante il training supervisionato

Gestisce il flusso di validazione quando LLM etichetta come "ALTRO":
1. Ottiene raw response del LLM 
2. Fa classificare anche a BERTopic
3. Se concordano: procede con controllo similarità
4. Se non concordano: rimanda alla decisione umana
5. Controllo similarità semantica con tag esistenti
6. Se similarità >= 90%: usa tag esistente
7. Se < 90%: aggiunge nuovo tag

Autore: Pipeline Humanitas  
Data: 23 Agosto 2025
"""

import sys
import os
import logging
import json
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass

# Aggiunge i percorsi necessari
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EmbeddingEngine'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'TagDatabase'))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tag_manager import TagDatabaseManager
from labse_embedder import LaBSEEmbedder
from mongo_classification_reader import MongoClassificationReader

@dataclass
class ValidationResult:
    """
    Risultato della validazione di un tag "ALTRO"
    """
    should_add_new_tag: bool
    final_tag: str
    confidence: float
    validation_path: str  # "llm_bertopic_agree", "human_decision", "similarity_match"
    similarity_score: Optional[float] = None
    matched_existing_tag: Optional[str] = None
    bertopic_suggestion: Optional[str] = None
    llm_raw_response: Optional[str] = None
    needs_human_review: bool = False

class AltroTagValidator:
    """
    Validatore per i tag classificati come "ALTRO" durante il training supervisionato
    """
    
    def __init__(self, tenant_id: str, similarity_threshold: float = 0.9):
        """
        Inizializza il validatore
        
        Args:
            tenant_id: ID del tenant
            similarity_threshold: Soglia di similarità per considerare tag equivalenti (default 0.9)
        """
        self.tenant_id = tenant_id
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
        
        # Inizializza i componenti necessari
        self.tag_manager = TagDatabaseManager()
        self.embedder = LaBSEEmbedder()
        self.db_connector = MongoClassificationReader()
        
        # Cache per i tag esistenti e loro embeddings
        self._existing_tags_cache = None
        self._existing_embeddings_cache = None
        
    def validate_altro_classification(self, 
                                    conversation_text: str,
                                    llm_classifier: Any,
                                    bertopic_model: Any,
                                    force_human_decision: bool = False) -> ValidationResult:
        """
        Valida una classificazione "ALTRO" durante il training supervisionato
        
        Args:
            conversation_text: Testo della conversazione da validare
            llm_classifier: Istanza del classificatore LLM
            bertopic_model: Modello BERTopic per la classificazione alternativa
            force_human_decision: Se True, forza la decisione umana
            
        Returns:
            ValidationResult con la decisione finale
        """
        self.logger.info(f"Validando classificazione ALTRO per tenant {self.tenant_id}")
        
        try:
            # 1. Ottieni la raw response dell'LLM 
            llm_raw_response = self._get_llm_raw_response(conversation_text, llm_classifier)
            self.logger.debug(f"LLM raw response: {llm_raw_response}")
            
            # 2. Ottieni classificazione da BERTopic
            bertopic_suggestion = self._get_bertopic_classification(conversation_text, bertopic_model)
            self.logger.debug(f"BERTopic suggestion: {bertopic_suggestion}")
            
            # 3. Se forzata decisione umana, rimanda subito
            if force_human_decision:
                return ValidationResult(
                    should_add_new_tag=False,
                    final_tag="altro",
                    confidence=0.5,
                    validation_path="human_decision_forced",
                    bertopic_suggestion=bertopic_suggestion,
                    llm_raw_response=llm_raw_response,
                    needs_human_review=True
                )
            
            # 4. Controlla concordanza LLM + BERTopic
            if self._check_agreement(llm_raw_response, bertopic_suggestion):
                self.logger.info("LLM e BERTopic concordano, procedo con controllo similarità")
                
                # 5. Estrai il tag suggerito (da LLM o BERTopic)
                suggested_tag = self._extract_best_suggestion(llm_raw_response, bertopic_suggestion)
                
                # 6. Controllo similarità semantica con tag esistenti
                similarity_result = self._check_semantic_similarity(suggested_tag)
                
                if similarity_result['max_similarity'] >= self.similarity_threshold:
                    # Usa tag esistente
                    return ValidationResult(
                        should_add_new_tag=False,
                        final_tag=similarity_result['most_similar_tag'],
                        confidence=similarity_result['max_similarity'],
                        validation_path="similarity_match",
                        similarity_score=similarity_result['max_similarity'],
                        matched_existing_tag=similarity_result['most_similar_tag'],
                        bertopic_suggestion=bertopic_suggestion,
                        llm_raw_response=llm_raw_response
                    )
                else:
                    # Aggiungi nuovo tag
                    return ValidationResult(
                        should_add_new_tag=True,
                        final_tag=suggested_tag,
                        confidence=0.8,  # Alta confidenza per nuovo tag validato
                        validation_path="llm_bertopic_agree",
                        similarity_score=similarity_result['max_similarity'],
                        bertopic_suggestion=bertopic_suggestion,
                        llm_raw_response=llm_raw_response
                    )
            else:
                self.logger.info("LLM e BERTopic NON concordano, rimando alla decisione umana")
                
                # Non concordano, rimanda alla decisione umana
                return ValidationResult(
                    should_add_new_tag=False,
                    final_tag="altro",
                    confidence=0.3,  # Bassa confidenza per conflitto
                    validation_path="human_decision",
                    bertopic_suggestion=bertopic_suggestion,
                    llm_raw_response=llm_raw_response,
                    needs_human_review=True
                )
                
        except Exception as e:
            self.logger.error(f"Errore durante validazione ALTRO: {e}")
            return ValidationResult(
                should_add_new_tag=False,
                final_tag="altro",
                confidence=0.1,
                validation_path="error",
                needs_human_review=True
            )
    
    def _get_llm_raw_response(self, conversation_text: str, llm_classifier: Any) -> str:
        """
        Ottiene la raw response dell'LLM per estrazione del tag suggerito
        
        Args:
            conversation_text: Testo da classificare
            llm_classifier: Classificatore LLM
            
        Returns:
            Raw response dell'LLM come stringa
        """
        try:
            # Usa il metodo classify_with_motivation per ottenere dettagli
            result = llm_classifier.classify_with_motivation(conversation_text)
            
            # La raw response dovrebbe contenere il JSON originale
            if hasattr(result, 'raw_response'):
                return result.raw_response
            elif hasattr(result, 'motivation'):
                # Se non c'è raw_response, usa la motivazione
                return result.motivation
            else:
                return f"Predicted: {result.predicted_label}, Confidence: {result.confidence}"
                
        except Exception as e:
            self.logger.error(f"Errore nell'ottenere LLM raw response: {e}")
            return "Errore nell'ottenere raw response LLM"
    
    def _get_bertopic_classification(self, conversation_text: str, bertopic_model: Any) -> str:
        """
        Ottiene classificazione da BERTopic
        
        Args:
            conversation_text: Testo da classificare
            bertopic_model: Modello BERTopic
            
        Returns:
            Tag suggerito da BERTopic
        """
        try:
            if hasattr(bertopic_model, 'transform'):
                # Classifica il documento singolo
                topics, probs = bertopic_model.transform([conversation_text])
                
                if len(topics) > 0:
                    topic_id = topics[0]
                    
                    # Ottieni il nome del topic
                    if hasattr(bertopic_model, 'get_topic_info'):
                        topic_info = bertopic_model.get_topic_info()
                        topic_row = topic_info[topic_info['Topic'] == topic_id]
                        
                        if not topic_row.empty:
                            topic_name = topic_row.iloc[0]['Name']
                            # Pulisci il nome del topic
                            return self._clean_bertopic_topic_name(topic_name)
                
            return "sconosciuto"  # Fallback
            
        except Exception as e:
            self.logger.error(f"Errore nella classificazione BERTopic: {e}")
            return "errore_bertopic"
    
    def _clean_bertopic_topic_name(self, topic_name: str) -> str:
        """
        Pulisce il nome del topic di BERTopic per renderlo utilizzabile come tag
        
        Args:
            topic_name: Nome grezzo del topic
            
        Returns:
            Nome pulito del topic
        """
        if not topic_name:
            return "sconosciuto"
        
        # Rimuovi numeri del topic (es. "0_prenotazione_esame_medico" -> "prenotazione_esame_medico")
        cleaned = re.sub(r'^\d+_', '', str(topic_name))
        
        # Sostituisci spazi e caratteri speciali con underscore
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', cleaned)
        
        # Rimuovi underscore doppi
        cleaned = re.sub(r'_+', '_', cleaned)
        
        # Rimuovi underscore iniziali e finali
        cleaned = cleaned.strip('_').lower()
        
        return cleaned if cleaned else "sconosciuto"
    
    def _check_agreement(self, llm_raw_response: str, bertopic_suggestion: str) -> bool:
        """
        Verifica se LLM e BERTopic concordano sulla classificazione
        
        Args:
            llm_raw_response: Raw response dell'LLM
            bertopic_suggestion: Suggerimento di BERTopic
            
        Returns:
            True se concordano, False altrimenti
        """
        try:
            # Estrai tag suggerito dall'LLM response
            llm_suggested_tag = self._extract_tag_from_llm_response(llm_raw_response)
            
            if not llm_suggested_tag or llm_suggested_tag == "altro":
                # Se LLM non suggerisce nulla di specifico, considera discordia
                return False
            
            # Controlla similarità semantica tra i due suggerimenti
            similarity = self.embedder.calculate_similarity(
                llm_suggested_tag, 
                bertopic_suggestion
            )
            
            # Concordano se similarità > 0.7 
            agreement = similarity > 0.7
            
            self.logger.debug(f"Agreement check - LLM: '{llm_suggested_tag}', BERTopic: '{bertopic_suggestion}', Similarity: {similarity:.3f}, Agreement: {agreement}")
            
            return agreement
            
        except Exception as e:
            self.logger.error(f"Errore nel check agreement: {e}")
            return False
    
    def _extract_tag_from_llm_response(self, raw_response: str) -> str:
        """
        Estrae il tag suggerito dalla raw response dell'LLM
        
        Args:
            raw_response: Risposta grezza dell'LLM
            
        Returns:
            Tag estratto o stringa vuota
        """
        try:
            # Cerca pattern JSON
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                # Cerca campi comuni per il tag
                for field in ['predicted_label', 'label', 'tag', 'classification']:
                    if field in data and data[field] != 'altro':
                        return str(data[field]).lower().strip()
            
            # Se non trova JSON, cerca pattern testuali
            patterns = [
                r'tag:\s*([^,\n]+)',
                r'etichetta:\s*([^,\n]+)',
                r'classif[ic]*azione:\s*([^,\n]+)',
                r'suggerisco:\s*([^,\n]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, raw_response, re.IGNORECASE)
                if match:
                    tag = match.group(1).strip().lower()
                    if tag and tag != 'altro':
                        return tag
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Errore nell'estrazione tag da LLM response: {e}")
            return ""
    
    def _extract_best_suggestion(self, llm_raw_response: str, bertopic_suggestion: str) -> str:
        """
        Estrae il miglior suggerimento combinando LLM e BERTopic
        
        Args:
            llm_raw_response: Raw response LLM
            bertopic_suggestion: Suggerimento BERTopic
            
        Returns:
            Miglior tag suggerito
        """
        llm_tag = self._extract_tag_from_llm_response(llm_raw_response)
        
        # Priorità: LLM se disponibile e specifico, altrimenti BERTopic
        if llm_tag and llm_tag != "altro" and len(llm_tag) > 2:
            return llm_tag
        elif bertopic_suggestion and bertopic_suggestion != "sconosciuto":
            return bertopic_suggestion
        else:
            return "nuovo_tag_sconosciuto"
    
    def _check_semantic_similarity(self, suggested_tag: str) -> Dict[str, Any]:
        """
        Controlla similarità semantica con tag esistenti
        
        Args:
            suggested_tag: Tag suggerito da validare
            
        Returns:
            Dict con risultati della similarità
        """
        try:
            # Ottieni tag esistenti per il tenant
            existing_tags = self._get_existing_tags()
            
            if not existing_tags:
                self.logger.info("Nessun tag esistente trovato, nuovo tag sarà aggiunto")
                return {
                    'max_similarity': 0.0,
                    'most_similar_tag': None,
                    'all_similarities': {}
                }
            
            # Calcola similarità con tutti i tag esistenti
            similarities = {}
            max_similarity = 0.0
            most_similar_tag = None
            
            for existing_tag in existing_tags:
                try:
                    similarity = self.embedder.calculate_similarity(
                        suggested_tag, 
                        existing_tag
                    )
                    similarities[existing_tag] = similarity
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_tag = existing_tag
                        
                except Exception as e:
                    self.logger.warning(f"Errore nel calcolo similarità per tag '{existing_tag}': {e}")
                    similarities[existing_tag] = 0.0
            
            self.logger.info(f"Similarità calcolate per '{suggested_tag}': max={max_similarity:.3f} con '{most_similar_tag}'")
            
            return {
                'max_similarity': max_similarity,
                'most_similar_tag': most_similar_tag,
                'all_similarities': similarities
            }
            
        except Exception as e:
            self.logger.error(f"Errore nel controllo similarità semantica: {e}")
            return {
                'max_similarity': 0.0,
                'most_similar_tag': None,
                'all_similarities': {}
            }
    
    def _get_existing_tags(self) -> List[str]:
        """
        Ottiene la lista dei tag esistenti per il tenant (con cache)
        
        Returns:
            Lista dei tag esistenti
        """
        if self._existing_tags_cache is None:
            try:
                # Ottieni tag dal TagManager
                tags_data = self.tag_manager.get_all_tags()
                self._existing_tags_cache = [tag['tag'] for tag in tags_data if tag['tag'] != 'altro']
                
                self.logger.debug(f"Tag esistenti caricati: {len(self._existing_tags_cache)} tag")
                
            except Exception as e:
                self.logger.error(f"Errore nel caricamento tag esistenti: {e}")
                self._existing_tags_cache = []
        
        return self._existing_tags_cache
    
    def add_new_tag_immediately(self, new_tag: str, confidence: float = 0.8) -> bool:
        """
        Aggiunge immediatamente un nuovo tag al sistema per renderlo disponibile
        
        Args:
            new_tag: Nuovo tag da aggiungere
            confidence: Confidenza del tag
            
        Returns:
            True se aggiunto con successo
        """
        try:
            self.logger.info(f"Aggiungendo nuovo tag '{new_tag}' per tenant {self.tenant_id}")
            
            # Aggiungi tramite TagManager
            success = self.tag_manager.add_tag(
                tag=new_tag,
                source='altro_validation',
                confidence=confidence,
                count=1,  # Prima occorrenza
                avg_confidence=confidence
            )
            
            if success:
                # Invalida cache per forzare ricaricamento
                self._existing_tags_cache = None
                self.logger.info(f"Nuovo tag '{new_tag}' aggiunto con successo")
                return True
            else:
                self.logger.error(f"Fallimento nell'aggiunta del tag '{new_tag}'")
                return False
                
        except Exception as e:
            self.logger.error(f"Errore nell'aggiunta del nuovo tag '{new_tag}': {e}")
            return False
    
    def invalidate_cache(self):
        """
        Invalida la cache dei tag esistenti
        """
        self._existing_tags_cache = None
        self._existing_embeddings_cache = None
        self.logger.debug("Cache invalidata")
