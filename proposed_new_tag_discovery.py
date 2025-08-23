"""
Proposta: Sistema di Scoperta Automatica Nuovi Tag
Author: AI Assistant
Date: 2025-08-21

Meccanismo per scoprire e aggiungere automaticamente nuovi tag durante 
il training supervisionato basandosi sui topic di BERTopic.
"""

class NewTagDiscoverySystem:
    """
    Sistema per scoprire automaticamente nuovi tag da topic BERTopic
    e aggiungerli alla tabella TAG.tags con descrizioni generate.
    """
    
    def __init__(self, llm_classifier, tag_db_connector):
        self.llm_classifier = llm_classifier
        self.tag_db = tag_db_connector
        
    def discover_new_tags_from_bertopic(self, bertopic_model, conversations: List[str], 
                                      existing_tags: List[str]) -> List[Dict]:
        """
        Scopre nuovi tag potenziali dai topic di BERTopic.
        
        Process:
        1. BERTopic identifica topic semantici dalle conversazioni
        2. Analizza le parole chiave di ogni topic  
        3. Usa LLM per generare nomi tag e descrizioni
        4. Filtra topic troppo simili a tag esistenti
        5. Propone nuovi tag con alta confidenza semantica
        """
        
        # 1. Ottieni topic da BERTopic
        topics = bertopic_model.get_topics()
        topic_info = bertopic_model.get_topic_info()
        
        new_tags = []
        
        for topic_id, topic_words in topics.items():
            if topic_id == -1:  # Skip outliers
                continue
                
            # 2. Estrai parole chiave del topic
            keywords = [word for word, _ in topic_words[:10]]
            
            # 3. Usa LLM per generare tag candidato
            tag_proposal = self._generate_tag_from_keywords(keywords, existing_tags)
            
            if tag_proposal and self._is_tag_novel(tag_proposal['name'], existing_tags):
                # 4. Calcola representatività del topic
                topic_size = self._get_topic_size(bertopic_model, topic_id)
                
                if topic_size >= 5:  # Soglia minima conversazioni
                    new_tags.append({
                        'tag_name': tag_proposal['name'],
                        'tag_description': tag_proposal['description'], 
                        'topic_id': topic_id,
                        'keywords': keywords,
                        'confidence': tag_proposal['confidence'],
                        'topic_size': topic_size
                    })
        
        return new_tags
    
    def _generate_tag_from_keywords(self, keywords: List[str], 
                                  existing_tags: List[str]) -> Dict:
        """Usa LLM per generare nome tag e descrizione da keywords."""
        
        prompt = f"""
        Analizza queste parole chiave di un topic semantico di conversazioni ospedaliere:
        Keywords: {', '.join(keywords)}
        
        Tag esistenti: {', '.join(existing_tags)}
        
        Genera:
        1. Nome tag (snake_case, es: info_orari_visite)
        2. Descrizione chiara (1 frase)  
        3. Confidenza (0-1) che sia un nuovo intent utile
        
        Regole:
        - Evita duplicati dei tag esistenti
        - Focalizzati su intent specifici
        - Nome tag deve essere snake_case
        - Solo se topic rappresenta intent distinto
        
        Formato JSON:
        {{"name": "nuovo_tag", "description": "Descrizione...", "confidence": 0.8}}
        """
        
        try:
            response = self.llm_classifier._call_llm(prompt)
            import json
            return json.loads(response)
        except:
            return None
    
    def _is_tag_novel(self, new_tag: str, existing_tags: List[str]) -> bool:
        """Verifica se il tag è abbastanza diverso da quelli esistenti."""
        
        # Similarità semantica con embeddings
        # (implementazione semplificata)
        for existing in existing_tags:
            similarity = self._calculate_semantic_similarity(new_tag, existing)
            if similarity > 0.8:  # Troppo simile
                return False
        
        return True
    
    def _calculate_semantic_similarity(self, tag1: str, tag2: str) -> float:
        """Calcola similarità semantica tra due tag."""
        # Implementazione con embeddings
        # Placeholder per esempio
        return 0.0
    
    def _get_topic_size(self, bertopic_model, topic_id: int) -> int:
        """Ottieni numero di documenti nel topic."""
        topic_info = bertopic_model.get_topic_info()
        topic_row = topic_info[topic_info.Topic == topic_id]
        return topic_row.Count.iloc[0] if not topic_row.empty else 0
    
    def propose_new_tags_to_database(self, new_tags: List[Dict], 
                                   min_confidence: float = 0.7) -> bool:
        """
        Propone i nuovi tag per aggiunta al database.
        
        Args:
            new_tags: Lista di nuovi tag scoperti
            min_confidence: Soglia minima confidenza
            
        Returns:
            bool: True se almeno un tag è stato aggiunto
        """
        added_count = 0
        
        for tag_data in new_tags:
            if tag_data['confidence'] >= min_confidence:
                
                # Log della proposta
                print(f"\n🆕 NUOVO TAG SCOPERTO:")
                print(f"   📛 Nome: {tag_data['tag_name']}")
                print(f"   📝 Descrizione: {tag_data['tag_description']}")
                print(f"   🎯 Topic ID: {tag_data['topic_id']}")
                print(f"   🔑 Keywords: {', '.join(tag_data['keywords'])}")
                print(f"   📊 Conversazioni: {tag_data['topic_size']}")
                print(f"   🎲 Confidenza: {tag_data['confidence']:.2f}")
                
                # Aggiunge al database
                success = self.tag_db.add_tag_if_not_exists(
                    tag_name=tag_data['tag_name'],
                    tag_description=tag_data['tag_description']
                )
                
                if success:
                    print(f"   ✅ Tag aggiunto al database TAG.tags")
                    added_count += 1
                else:
                    print(f"   ❌ Errore nell'aggiunta del tag")
        
        print(f"\n📊 RISULTATO SCOPERTA: {added_count} nuovi tag aggiunti su {len(new_tags)} proposti")
        return added_count > 0


# ESEMPIO DI INTEGRAZIONE NEL TRAINING PIPELINE
"""
def enhanced_training_with_tag_discovery(self):
    # ... codice training esistente ...
    
    # Dopo aver creato BERTopic
    if bertopic_provider is not None:
        print("\\n🔍 SCOPERTA NUOVI TAG DA BERTOPIC:")
        
        # Sistema di scoperta tag
        tag_discovery = NewTagDiscoverySystem(
            llm_classifier=self.ensemble_classifier.llm_classifier,
            tag_db_connector=self.mysql_connector
        )
        
        # Ottieni tag esistenti
        existing_tags = [tag['tag_name'] for tag in self.mysql_connector.get_all_tags()]
        
        # Scopri nuovi tag
        new_tags = tag_discovery.discover_new_tags_from_bertopic(
            bertopic_model=bertopic_provider.model,
            conversations=session_texts,
            existing_tags=existing_tags
        )
        
        # Proponi per aggiunta
        if new_tags:
            tag_discovery.propose_new_tags_to_database(new_tags, min_confidence=0.75)
        else:
            print("   📭 Nessun nuovo tag scoperto")
    
    # ... continua training normale ...
"""

print("🚀 Sistema di Scoperta Automatica Nuovi Tag - Pronto per implementazione!")
