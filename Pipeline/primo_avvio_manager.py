#!/usr/bin/env python3
"""
Sistema di Gestione Primo Avvio Training Supervisionato
Author: AI Assistant  
Date: 2025-08-23

Gestisce il primo training supervisionato per tenant senza tag esistenti,
implementando la strategia BERTopic + LLM + Validazione Umana per evitare
che tutte le conversazioni siano classificate come "altro".
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Aggiungiamo i path necessari
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from TAGS.tag import IntelligentTagSuggestionManager
from TagDatabase.tag_database_connector import TagDatabaseConnector

class PrimoAvvioManager:
    """
    Gestisce il primo avvio del training supervisionato per tenant
    senza tag esistenti, implementando strategie per evitare la 
    classificazione "altro" di massa.
    
    Strategia implementata:
    1. Controlla se il tenant ha tag esistenti
    2. Se NO: Attiva modalità "primo avvio" con discovery automatica
    3. BERTopic identifica cluster semantici
    4. LLM propone etichette per ogni cluster
    5. Validazione automatica o umana delle proposte
    6. Aggiunta tag al database PRIMA del training
    7. Training supervisionato con tag pre-popolati
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inizializza il manager del primo avvio
        
        Args:
            config_path: Percorso al file di configurazione
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'config.yaml'
            )
        
        self.config_path = config_path
        self.tag_suggestion_manager = IntelligentTagSuggestionManager(config_path)
        self.tag_db_connector = TagDatabaseConnector()
        self.logger = self._setup_logger()
        
        # Configurazioni per primo avvio
        self.primo_avvio_config = {
            'bertopic_discovery_enabled': True,
            'llm_validation_enabled': True, 
            'auto_approve_threshold': 0.85,  # Soglia per auto-approvazione
            'min_cluster_size': 3,           # Dimensione minima cluster
            'max_new_tags_per_session': 20,  # Limite nuovi tag per sessione
            'human_validation_required': False  # Se richiedere validazione umana
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup del logger specializzato"""
        logger = logging.getLogger(f"{__name__}.PrimoAvvioManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def is_primo_avvio(self, tenant_name: str) -> bool:
        """
        Determina se questo è il primo avvio per un tenant
        
        Args:
            tenant_name: Nome del tenant
            
        Returns:
            bool: True se è primo avvio (nessun tag esistente)
        """
        has_existing = self.tag_suggestion_manager.has_existing_classifications(tenant_name)
        existing_tags_count = len(self.tag_db_connector.get_all_tags())
        
        primo_avvio = not has_existing or existing_tags_count == 0
        
        self.logger.info(f"🔍 CONTROLLO PRIMO AVVIO per '{tenant_name}':")
        self.logger.info(f"   - Classificazioni esistenti: {has_existing}")
        self.logger.info(f"   - Tag nel database: {existing_tags_count}")
        self.logger.info(f"   - È primo avvio: {primo_avvio}")
        
        return primo_avvio
    
    def prepare_tags_for_primo_avvio(self, 
                                   tenant_name: str, 
                                   session_texts: List[str],
                                   bertopic_model: Any,
                                   llm_classifier: Any) -> Dict[str, Any]:
        """
        Prepara i tag per il primo avvio usando BERTopic + LLM
        
        Args:
            tenant_name: Nome del tenant
            session_texts: Lista dei testi delle sessioni
            bertopic_model: Modello BERTopic già addestrato
            llm_classifier: Classificatore LLM
            
        Returns:
            Dict con risultati della preparazione
        """
        self.logger.info(f"🚀 PREPARAZIONE TAG PRIMO AVVIO per '{tenant_name}'")
        self.logger.info(f"   - Sessioni da analizzare: {len(session_texts)}")
        
        results = {
            'success': False,
            'new_tags_created': 0,
            'tags_created': [],
            'clusters_analyzed': 0,
            'errors': []
        }
        
        try:
            # 1. Ottieni topic da BERTopic
            topics = bertopic_model.get_topics()
            topic_info = bertopic_model.get_topic_info()
            
            self.logger.info(f"📊 BERTopic ha identificato {len(topics)} topic")
            
            new_tags_proposals = []
            clusters_analyzed = 0
            
            # 2. Analizza ogni topic per proporre tag
            for topic_id, topic_words in topics.items():
                if topic_id == -1:  # Skip outliers
                    continue
                    
                clusters_analyzed += 1
                
                # Ottieni dimensione del topic
                topic_size = self._get_topic_size(bertopic_model, topic_id)
                
                if topic_size < self.primo_avvio_config['min_cluster_size']:
                    self.logger.debug(f"   ⏭️ Topic {topic_id} troppo piccolo ({topic_size} sessioni)")
                    continue
                
                # Estrai parole chiave
                keywords = [word for word, _ in topic_words[:10]]
                
                self.logger.info(f"🔍 Analizzando Topic {topic_id} ({topic_size} sessioni)")
                self.logger.info(f"   Keywords: {', '.join(keywords[:5])}...")
                
                # 3. Usa LLM per proporre tag
                tag_proposal = self._generate_tag_with_llm(
                    keywords, 
                    topic_id, 
                    topic_size,
                    llm_classifier,
                    session_texts
                )
                
                if tag_proposal and tag_proposal['confidence'] >= 0.6:
                    new_tags_proposals.append(tag_proposal)
                    self.logger.info(f"✅ Proposta tag: '{tag_proposal['name']}' "
                                   f"(confidenza: {tag_proposal['confidence']:.2f})")
                
                # Limita numero di nuovi tag per sessione
                if len(new_tags_proposals) >= self.primo_avvio_config['max_new_tags_per_session']:
                    self.logger.info(f"🛑 Raggiunto limite di {self.primo_avvio_config['max_new_tags_per_session']} nuovi tag")
                    break
            
            # 4. Valida e approva tag
            approved_tags = self._validate_and_approve_tags(
                new_tags_proposals, 
                tenant_name
            )
            
            # 5. Aggiungi tag al database
            tags_created = []
            for tag_data in approved_tags:
                success = self.tag_db_connector.add_tag_if_not_exists(
                    tag_name=tag_data['name'],
                    tag_description=tag_data['description']
                )
                
                if success:
                    tags_created.append(tag_data['name'])
                    self.logger.info(f"✅ Tag '{tag_data['name']}' aggiunto al database")
                else:
                    self.logger.warning(f"❌ Errore nell'aggiunta del tag '{tag_data['name']}'")
            
            # Risultati finali
            results.update({
                'success': True,
                'new_tags_created': len(tags_created),
                'tags_created': tags_created,
                'clusters_analyzed': clusters_analyzed,
                'proposals_generated': len(new_tags_proposals),
                'proposals_approved': len(approved_tags)
            })
            
            self.logger.info(f"🎯 RISULTATI PRIMO AVVIO:")
            self.logger.info(f"   - Cluster analizzati: {clusters_analyzed}")
            self.logger.info(f"   - Proposte generate: {len(new_tags_proposals)}")
            self.logger.info(f"   - Proposte approvate: {len(approved_tags)}")
            self.logger.info(f"   - Tag creati: {len(tags_created)}")
            self.logger.info(f"   - Tag: {', '.join(tags_created)}")
            
        except Exception as e:
            self.logger.error(f"❌ Errore nella preparazione tag primo avvio: {e}")
            results['errors'].append(str(e))
        
        return results
    
    def _get_topic_size(self, bertopic_model: Any, topic_id: int) -> int:
        """Ottieni numero di documenti nel topic"""
        try:
            topic_info = bertopic_model.get_topic_info()
            topic_row = topic_info[topic_info.Topic == topic_id]
            return topic_row.Count.iloc[0] if not topic_row.empty else 0
        except:
            return 0
    
    def _generate_tag_with_llm(self, 
                             keywords: List[str], 
                             topic_id: int, 
                             topic_size: int,
                             llm_classifier: Any,
                             session_texts: List[str]) -> Optional[Dict[str, Any]]:
        """
        Genera proposta tag usando LLM
        
        Args:
            keywords: Parole chiave del topic
            topic_id: ID del topic
            topic_size: Dimensione del topic
            llm_classifier: Classificatore LLM
            session_texts: Testi delle sessioni per contesto
            
        Returns:
            Dict con proposta tag o None se fallisce
        """
        try:
            # Costruisci prompt contestuale
            prompt = self._build_tag_generation_prompt(
                keywords, topic_id, topic_size, session_texts[:3]
            )
            
            # Chiama LLM
            response = llm_classifier._call_llm(prompt)
            
            # Parse della risposta
            import json
            tag_data = json.loads(response)
            
            # Validazione risposta
            if self._validate_tag_proposal(tag_data):
                return {
                    'name': tag_data['name'],
                    'description': tag_data['description'],
                    'confidence': tag_data['confidence'],
                    'topic_id': topic_id,
                    'topic_size': topic_size,
                    'keywords': keywords
                }
                
        except Exception as e:
            self.logger.warning(f"⚠️ Errore generazione tag per topic {topic_id}: {e}")
        
        return None
    
    def _build_tag_generation_prompt(self, 
                                   keywords: List[str], 
                                   topic_id: int, 
                                   topic_size: int,
                                   sample_texts: List[str]) -> str:
        """Costruisce prompt per generazione tag LLM"""
        
        samples_section = ""
        if sample_texts:
            samples_section = "\n\nEsempi di conversazioni:\n"
            for i, text in enumerate(sample_texts[:2], 1):
                truncated = text[:200] + "..." if len(text) > 200 else text
                samples_section += f"{i}. {truncated}\n"
        
        prompt = f"""
Analizza questo topic semantico identificato da BERTopic in conversazioni ospedaliere.

INFORMAZIONI TOPIC:
- ID Topic: {topic_id}
- Dimensione: {topic_size} conversazioni
- Parole chiave: {', '.join(keywords)}
{samples_section}
COMPITO:
Crea un tag (etichetta) significativo per questo topic che:
1. Rappresenti l'intent/bisogno principale delle conversazioni
2. Sia specifico e utile per classificazioni future  
3. Sia in formato snake_case (es: prenotazione_visite)
4. Abbia una descrizione chiara e professionale

REGOLE:
- Nome tag: massimo 3-4 parole in snake_case
- Descrizione: 1 frase chiara e professionale
- Confidenza: 0.0-1.0 basata su chiarezza del topic
- Se il topic non è chiaro o troppo generico, confidenza < 0.5

FORMATO RISPOSTA (JSON):
{{"name": "nome_tag_snake_case", "description": "Descrizione professionale del tag", "confidence": 0.8}}

Rispondi SOLO con il JSON, senza altre spiegazioni.
"""
        return prompt
    
    def _validate_tag_proposal(self, tag_data: Dict[str, Any]) -> bool:
        """Valida la proposta tag dal LLM"""
        required_keys = ['name', 'description', 'confidence']
        
        if not all(key in tag_data for key in required_keys):
            return False
        
        if not isinstance(tag_data['name'], str) or not tag_data['name'].strip():
            return False
        
        if not isinstance(tag_data['description'], str) or not tag_data['description'].strip():
            return False
        
        try:
            confidence = float(tag_data['confidence'])
            if not 0.0 <= confidence <= 1.0:
                return False
        except (ValueError, TypeError):
            return False
        
        return True
    
    def _validate_and_approve_tags(self, 
                                 proposals: List[Dict[str, Any]], 
                                 tenant_name: str) -> List[Dict[str, Any]]:
        """
        Valida e approva le proposte di tag
        
        Args:
            proposals: Lista di proposte tag
            tenant_name: Nome del tenant
            
        Returns:
            Lista di tag approvati
        """
        approved = []
        existing_tags = [tag['tag_name'] for tag in self.tag_db_connector.get_all_tags()]
        
        self.logger.info(f"🔍 Validazione di {len(proposals)} proposte tag:")
        
        for proposal in proposals:
            # Skip se tag già esiste
            if proposal['name'] in existing_tags:
                self.logger.info(f"   ⏭️ '{proposal['name']}' già esiste")
                continue
            
            # Skip se confidence troppo bassa
            if proposal['confidence'] < 0.6:
                self.logger.info(f"   ❌ '{proposal['name']}' confidenza troppo bassa ({proposal['confidence']:.2f})")
                continue
            
            # Auto-approva se confidence alta
            if proposal['confidence'] >= self.primo_avvio_config['auto_approve_threshold']:
                approved.append(proposal)
                self.logger.info(f"   ✅ '{proposal['name']}' auto-approvato (confidenza: {proposal['confidence']:.2f})")
                continue
            
            # Validazione umana se richiesta
            if self.primo_avvio_config['human_validation_required']:
                if self._human_validate_tag(proposal):
                    approved.append(proposal)
                    self.logger.info(f"   👤 '{proposal['name']}' approvato da umano")
                else:
                    self.logger.info(f"   👤 '{proposal['name']}' rifiutato da umano")
            else:
                # Approva comunque se sopra soglia minima
                approved.append(proposal) 
                self.logger.info(f"   ✅ '{proposal['name']}' approvato (confidenza: {proposal['confidence']:.2f})")
        
        return approved
    
    def _human_validate_tag(self, proposal: Dict[str, Any]) -> bool:
        """
        Validazione umana di una proposta tag (modalità interattiva)
        
        Args:
            proposal: Proposta tag da validare
            
        Returns:
            bool: True se approvato dall'umano
        """
        print(f"\n" + "="*60)
        print(f"🏷️  VALIDAZIONE TAG PROPOSTO")
        print(f"="*60)
        print(f"Nome: {proposal['name']}")
        print(f"Descrizione: {proposal['description']}")
        print(f"Confidenza: {proposal['confidence']:.2f}")
        print(f"Topic ID: {proposal.get('topic_id', 'N/A')}")
        print(f"Dimensione: {proposal.get('topic_size', 'N/A')} conversazioni")
        print(f"Keywords: {', '.join(proposal.get('keywords', []))}")
        
        while True:
            try:
                response = input(f"\nApprovi questo tag? [s/n/m(odifica)]: ").strip().lower()
                
                if response in ['s', 'si', 'y', 'yes']:
                    return True
                elif response in ['n', 'no']:
                    return False
                elif response in ['m', 'modifica', 'modify']:
                    # TODO: Implementare modifica interattiva
                    print("📝 Modifica non ancora implementata. Approvazione automatica.")
                    return True
                else:
                    print("❌ Rispondi con 's' (sì), 'n' (no) o 'm' (modifica)")
                    
            except KeyboardInterrupt:
                print("\n🛑 Validazione interrotta. Tag rifiutato.")
                return False
    
    def execute_primo_avvio_workflow(self, 
                                   tenant_name: str,
                                   session_texts: List[str],
                                   bertopic_model: Any,
                                   llm_classifier: Any) -> Dict[str, Any]:
        """
        Esegue il workflow completo per il primo avvio
        
        Args:
            tenant_name: Nome del tenant
            session_texts: Testi delle sessioni
            bertopic_model: Modello BERTopic
            llm_classifier: Classificatore LLM
            
        Returns:
            Dict con risultati del workflow
        """
        workflow_results = {
            'is_primo_avvio': False,
            'tags_preparation': {},
            'ready_for_training': False,
            'message': ''
        }
        
        # 1. Controlla se è primo avvio
        is_primo_avvio = self.is_primo_avvio(tenant_name)
        workflow_results['is_primo_avvio'] = is_primo_avvio
        
        if not is_primo_avvio:
            workflow_results['message'] = f"Tenant '{tenant_name}' non è al primo avvio. Procedure normali."
            workflow_results['ready_for_training'] = True
            return workflow_results
        
        # 2. Esegui preparazione tag per primo avvio
        self.logger.info(f"🚀 WORKFLOW PRIMO AVVIO per '{tenant_name}'")
        
        tags_preparation = self.prepare_tags_for_primo_avvio(
            tenant_name, 
            session_texts, 
            bertopic_model, 
            llm_classifier
        )
        
        workflow_results['tags_preparation'] = tags_preparation
        
        if tags_preparation['success'] and tags_preparation['new_tags_created'] > 0:
            workflow_results['ready_for_training'] = True
            workflow_results['message'] = (
                f"Primo avvio completato con successo! "
                f"Creati {tags_preparation['new_tags_created']} nuovi tag. "
                f"Sistema pronto per training supervisionato."
            )
        else:
            workflow_results['ready_for_training'] = False
            workflow_results['message'] = (
                f"Primo avvio completato ma nessun tag creato. "
                f"Il training procederà con tag di fallback."
            )
        
        return workflow_results


# Funzioni di utilità per integrazione
def integrate_primo_avvio_in_pipeline(pipeline_instance, tenant_name: str) -> bool:
    """
    Integra il primo avvio manager nella pipeline esistente
    
    Args:
        pipeline_instance: Istanza della pipeline di training
        tenant_name: Nome del tenant
        
    Returns:
        bool: True se primo avvio eseguito con successo
    """
    try:
        primo_avvio_manager = PrimoAvvioManager()
        
        # Ottieni dati necessari dalla pipeline
        if not hasattr(pipeline_instance, 'session_texts') or not hasattr(pipeline_instance, 'bertopic_provider'):
            return False
            
        session_texts = pipeline_instance.session_texts
        bertopic_model = pipeline_instance.bertopic_provider.model if pipeline_instance.bertopic_provider else None
        llm_classifier = pipeline_instance.ensemble_classifier.llm_classifier if hasattr(pipeline_instance, 'ensemble_classifier') else None
        
        if not bertopic_model or not llm_classifier:
            return False
        
        # Esegui workflow primo avvio
        results = primo_avvio_manager.execute_primo_avvio_workflow(
            tenant_name, session_texts, bertopic_model, llm_classifier
        )
        
        return results['ready_for_training']
        
    except Exception as e:
        logging.error(f"Errore integrazione primo avvio: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Sistema Primo Avvio Manager - Test")
    
    manager = PrimoAvvioManager()
    
    # Test controllo primo avvio
    test_tenants = ["Humanitas", "NuovoTenant", "TestClient"]
    
    for tenant in test_tenants:
        is_primo = manager.is_primo_avvio(tenant)
        print(f"📊 {tenant}: {'Primo Avvio' if is_primo else 'Esistente'}")
    
    print("\n✅ Test completato!")
